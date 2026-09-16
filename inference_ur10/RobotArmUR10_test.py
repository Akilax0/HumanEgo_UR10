# -*- coding: utf-8 -*-
# @FileName: RobotArmUR10_template.py
"""
Script done, but have to implement the below functions so can be useful for run inference & controller


Minimal RobotArm implementation for HumanEgo inference — UR10e + Robotiq 3-Finger.

This is the COMPLETE contract. Six methods and one attribute, derived from the
actual call sites in run_inference.py and controller.py — nothing else in the
inference loop touches the arm.

    T_base_in_cam            attribute    internal use only
    get_T_ee_in_cam()        ~1.25 Hz     run_inference.py:244, controller.py:99
    get_gripper()            ~1.25 Hz     run_inference.py:245
    move_ee_in_cam()          10 Hz       controller.py:95
    set_gripper()             10 Hz       controller.py:101, 107
    go_home()                2x/episode   run_inference.py:233, controller.py:108
    close()                  once         run_inference.py:282

Anything marked IMPLEMENT is yours. Anything marked GIVEN is plumbing you should
not need to change. The CONTRACT block on each method is what the loop depends
on — violating one of those produces silent misbehaviour, not an exception.

────────────────────────────────────────────────────────────────────────────
NON-NEGOTIABLE CONVENTIONS  (interfaces.py lines 22-30)
────────────────────────────────────────────────────────────────────────────
  * Poses are 4x4 SE(3), float. Positions in METERS. Rotations proper.
  * `_in_cam` = CAMERA OPTICAL frame (OpenCV: +x right, +y down, +z forward).
    NOT the ROS camera_link frame. This is the shared world frame for an episode.
  * Gripper scalars are [0, 1] with 0.0 = OPEN, 1.0 = CLOSED.
    Robotiq 3F rPRA/gPOA is already 0=open..255=closed, so NO inversion.
    (run_inference.py's TrossenArm does `1.0 - x` — do not copy that.)
  * `T_base_in_cam` maps BASE -> CAMERA, used as T_base_in_cam @ T_ee_in_base.
    Inverting it is the single most common porting bug.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from yaml import warnings

import numpy as np
from scipy.spatial.transform import Rotation as R

# Make the repo root importable no matter how this file is invoked.
# (run_inference.py does this; without it `python inference_ur10/RobotArmUR10.py`
#  fails because sys.path[0] is the SCRIPT's dir, not your cwd.)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rtde_control
import rtde_receive

from utils.utils_io import load_cfg 

from pyModbusTCP.client import ModbusClient
import threading       
import time
import warnings

FAULT_CODES = {
    0x00: "No fault",
    0x05: "Priority fault: action delayed, activation must be completed first",
    0x06: "Priority fault: action delayed, mode change must be completed first",
    0x07: "Priority fault: the activation bit must be set prior to action",
    0x09: "Minor fault: communication chip not ready (booting)",
    0x0A: "Minor fault: changing mode fault, interference detected on scissor (< 20 s)",
    0x0B: "Minor fault: automatic release in progress",
    0x0D: "Major fault: activation fault",
    0x0E: "Major fault: changing mode fault, interference detected on scissor (> 20 s)",
    0x0F: "Major fault: automatic release completed, reset required",
}

# gIMC values - initilaization and mode change status
IMC_RESET = 0            # gripper in reset / auto-release state
IMC_ACTIVATING = 1       # activation in progress
IMC_MODE_CHANGE = 2      # mode change in progress
IMC_READY = 3            # activation and mode change completed

# gSTA values - Motion status
STA_MOVING = 0
STA_STOPPED_PARTIAL_DETECT = 1
STA_STOPPED_ALL_DETECT = 2
STA_AT_POSITION = 3

class GripperFault(RuntimeError):
    pass

class Robotiq3FGripper:
    """Minimal Robotiq 3-Finger interface for HumanEgo inference.

    This is a placeholder. The real implementation is in utils/robotiq_3f_gripper.py.
    """

    def __init__(self, ip: str, port: int = 502, unit_id: int = 9, mode: str = "pinch", socket_timeout: float = 2.0, update_interval: float = 0.05):
        self.ip = ip
        self.port = port
        self.unit_id = unit_id
        self.mode = mode
        self.socket_timeout = socket_timeout
        self.update_interval = update_interval

        # predeclare all attributes so if read before first poll land no AttributeError
        self.gSTA = self.gIMC = self.gGTO = self.gACT = None
        self.gMOD = None
        self.gDTS = self.gDTC = self.gDTB = self.gDTA = None
        self.FaultStatus = 0
        self.FingerA_PositionReqEcho = self.FingerA_Position = self.FingerA_Current = None
        self.FingerB_PositionReqEcho = self.FingerB_Position = self.FingerB_Current = None
        self.FingerC_PositionReqEcho = self.FingerC_Position = self.FingerC_Current = None
        self.Scissor_PositionReqEcho = self.Scissor_Position = self.Scissor_Current = None

        print(f"║ [Robotiq3FGripper] Connecting to {ip}:{port} unit {unit_id} in {mode} mode...")
        self._io_lock = threading.Lock()
        self._client = ModbusClient(ip, port=port, unit_id=unit_id, auto_open=True, timeout=socket_timeout)  
        if not self._client.open():
            raise ConnectionError(f"Failed to connect to Robotiq 3-Finger gripper at {ip}:{port}")

        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start() 
        # IMPLEMENT: connect to the gripper via Modbus/TCP

        # Give the update thread a moment to populate status once
        time.sleep(4 * update_interval) 

    # --------------------------I/O--------------------------
    def _read_registers(self):
        with self._io_lock:
            return self._client.read_input_registers(0,8) 

    def _write_registers(self, regs):
        with self._io_lock:
            ok = self._client.write_multiple_registers(0, regs)
        if not ok:
            warnings.warn(f"Failed to write registers to gripper: {regs}")
        return ok

    def _update_loop(self):
        while self._running:
            self._update_status()
            time.sleep(self.update_interval)

    def _update_status(self):
        readData = self._read_registers()
        if readData:
            (self.gSTA, self.gIMC, self.gGTO, self.gMOD, self.gACT,
             self.gDTS, self.gDTC, self.gDTB, self.gDTA) = self.stat(self.add_leading_zeros(bin(readData[0])))
            self.FaultStatus, self.FingerA_PositionReqEcho = self.Byte_status(self.add_leading_zeros(bin(readData[1])))
            self.FingerA_Position, self.FingerA_Current = self.Byte_status(self.add_leading_zeros(bin(readData[2])))
            self.FingerB_PositionReqEcho, self.FingerB_Position = self.Byte_status(self.add_leading_zeros(bin(readData[3])))
            self.FingerB_Current, self.FingerC_PositionReqEcho = self.Byte_status(self.add_leading_zeros(bin(readData[4])))
            self.FingerC_Position, self.FingerC_Current = self.Byte_status(self.add_leading_zeros(bin(readData[5])))
            self.Scissor_PositionReqEcho, self.Scissor_Position = self.Byte_status(self.add_leading_zeros(bin(readData[6])))
            self.Scissor_Current, _RES = self.Byte_status(self.add_leading_zeros(bin(readData[7])))
        else:
            print("Error reading data in update_status.")
    # ------------------------------------------------------------ parsing

    @staticmethod
    def add_leading_zeros(bin_num, total_length=16):
        # strips 0b in teh string 
        bin_str = str(bin_num)[2:]
        return bin_str.zfill(total_length)

    @staticmethod
    def Byte_status(variable):
        B1 = int(variable[0:8], 2)
        B2 = int(variable[8:16], 2)
        return B1, B2

    @staticmethod
    def stat(variable):
        modes = ["Basic Mode", "Pinch Mode", "Wide Mode", "Scissor Mode"]
        gSTA = int(variable[0:2], 2)
        gIMC = int(variable[2:4], 2)
        gGTO = int(variable[4], 2)
        gMOD = modes[int(variable[5:7], 2)]
        gACT = int(variable[7], 2)
        gDTS = int(variable[8:10], 2)
        gDTC = int(variable[10:12], 2)
        gDTB = int(variable[12:14], 2)
        gDTA = int(variable[14:16], 2)
        return gSTA, gIMC, gGTO, gMOD, gACT, gDTS, gDTC, gDTB, gDTA

        # ---------------------------------------------------------- fault/wait

    # Gets a readable message (Fault status read from register 1)
    def fault_string(self):
        return FAULT_CODES.get(self.FaultStatus, f"Unknown fault 0x{self.FaultStatus:02X}")

    def check_fault(self, raise_on_major=True):
        """Print/raise on any non-zero fault. Call this whenever motion stops unexpectedly."""
        if self.FaultStatus:
            msg = f"gFLT=0x{self.FaultStatus:02X}: {self.fault_string()}"
            print(msg)
            # unrecoverable faults 
            if raise_on_major and self.FaultStatus >= 0x0D:
                raise GripperFault(msg)
        return self.FaultStatus

    def wait_until_ready(self, timeout=20.0):
        """Block until activation AND any mode change are complete (gIMC == 3)."""
        start = time.time()
        while time.time() - start < timeout:
            if self.FaultStatus and self.FaultStatus >= 0x0D:
                raise GripperFault(f"gFLT=0x{self.FaultStatus:02X}: {self.fault_string()}")
            if self.gIMC == IMC_READY:
                return True
            time.sleep(0.05)
        raise TimeoutError(
            f"Gripper not ready after {timeout}s (gIMC={self.gIMC}, "
            f"gFLT=0x{self.FaultStatus:02X}: {self.fault_string()})"
        )

    def wait_until_stopped(self, timeout=10.0):
        """Block until motion finishes (gSTA != 0). Returns final gSTA."""
        start = time.time()
        # small grace period for motion to start
        time.sleep(0.2)
        while time.time() - start < timeout:
            self.check_fault()
            if self.gSTA is not None and self.gSTA != STA_MOVING:
                return self.gSTA
            time.sleep(0.05)
        raise TimeoutError(f"Gripper still moving after {timeout}s")

    # ------------------------------------------------------------ commands

    def activate(self, timeout=20.0):
        """Reset, then activate, and BLOCK until calibration is complete.

        Per the Robotiq manual: set rACT=1 alone (no rGTO, no mode, no
        position/speed/force), then poll gIMC until it reads 3 before
        sending any motion or mode-change command.
        """
        # Clear any previous state/fault first (rACT = 0)
        self._write_registers([0, 0, 0])
        time.sleep(0.1)
        # Activate only
        self._write_registers([self._action_req_variable(rACT=1), 0, 0])
        print("Gripper activating (calibration running)...")
        self.wait_until_ready(timeout=timeout)
        print("Gripper activation complete.")

    def set_mode(self, mode="Basic", timeout=15.0):
        """Change operation mode and wait for the mode change to finish."""
        modes = {"Basic": 0, "Pinch": 1, "Wide": 2, "Scissor": 3}
        rMOD = modes[mode]
        self._write_registers([
            self._action_req_variable(rACT=1, rGTO=0, rMOD=rMOD, rICF=0),
            self._position_req_variable(0),
            self._write_req_variable(0, 0),
        ])
        # gIMC drops to 2 during the change, back to 3 when done
        time.sleep(0.2)
        self.wait_until_ready(timeout=timeout)
        print(f"Mode set to {mode}.")

    def command_gripper(self, rPRA=[1, 1, 1], rSP=[250, 250, 250], rFR=[250, 250, 250],
                        rMOD="Basic", rICF=False):
        """Send a motion command. Assumes the gripper is ready (gIMC == 3).

        If rMOD differs from the current mode this also triggers a mode
        change; prefer calling set_mode() explicitly first so the scissor
        motion and the finger motion don't get conflated.
        """
        if self.gIMC != IMC_READY:
            warnings.warn(
                f"Commanding motion while gripper not ready (gIMC={self.gIMC}). "
                "Call wait_until_ready() first."
            )
        modes = {"Basic": 0, "Pinch": 1, "Wide": 2, "Scissor": 3}
        rMOD = modes[rMOD]
        if rICF:
            for var in [rPRA, rSP, rFR]:
                if isinstance(var, int):
                    raise ValueError("Input variables must be 3d vectors when using Individual Control Flag.")
            return self._write_registers([
                self._action_req_variable(rACT=1, rGTO=1, rMOD=rMOD, rICF=1),
                self._position_req_variable(rPRA[0]),
                self._write_req_variable(rSP[0], rFR[0]),
                self._write_req_variable(rPRA[1], rSP[1]),
                self._write_req_variable(rFR[1], rPRA[2]),
                self._write_req_variable(rSP[2], rFR[2]),
            ])
        else:
            for var in [rPRA, rSP, rFR]:
                if isinstance(var, list):
                    warnings.warn("only first value of 3d vector will be used when not using Individual Control Flag.")
            return self._write_registers([
                self._action_req_variable(rACT=1, rGTO=1, rMOD=rMOD, rICF=0),
                self._position_req_variable(rPRA[0]),
                self._write_req_variable(rSP[0], rFR[0]),
            ])

    def status(self):
        self._update_status()

    # ------------------------------------------------------- register build

    def _action_req_variable(self, rARD: int = 1, rATR: int = 0, rGTO: int = 0,
                             rACT: int = 0, rMOD: int = 0, rICS: int = 0, rICF: int = 0) -> int:
        for var in [rARD, rATR, rGTO, rACT]:
            if var not in [0, 1]:
                raise ValueError("Input variables must be either 0 or 1.")
        rMOD = bin(rMOD).replace("0b", "").zfill(2)
        string_variable = f"0b00{rARD}{rATR}{rGTO}{rMOD}{rACT}0000{rICS}{rICF}00"
        return int(string_variable, 2)

    def _position_req_variable(self, rPR: int = 0) -> int:
        if rPR not in range(0, 256):
            raise ValueError("Input variables must be between 0 and 255.")
        return int(f"0b00000000{format(rPR, '08b')}", 2)

    def _write_req_variable(self, X: int = 0, Y: int = 0) -> int:
        for var in [X, Y]:
            if var not in range(0, 256):
                raise ValueError("Input variables must be between 0 and 255.")
        return int(f"0b{format(X, '08b')}{format(Y, '08b')}", 2)

# -----------------------------------------------------------

    def send_command(self, position):
        position = [position[0]] * 3
        self.gripper.command_gripper(
            rPRA=position,
            rSP=self.speed,
            rFR=self.force,
            rMOD=self.mode,
            rICF=self.individual_control,
        )

    def process_gripper_command(self, command: str):
        """Call this method externally to control the gripper."""
        if command == 'c':
            print("Closing gripper")
            self.send_command([250])
            sta = self.gripper.wait_until_stopped()
            print(f"gSTA={sta} (1/2 = object detected, 3 = reached position)")
        elif command == 'o':
            print("Opening gripper")
            self.send_command([0])
            self.gripper.wait_until_stopped()
        elif command == 'e':
            print("Exiting and closing connection")
            self.gripper.close()
            return False
        else:
            print("Unknown command. Use 'c', 'o', or 'e'")
        self.gripper.check_fault(raise_on_major=False)
        return True 


    def get_joint_positions(self):
        return [
            self.gripper.FingerA_Position,
            self.gripper.FingerB_Position,
            self.gripper.FingerC_Position,
        ]

    def ind_control(self, position):
        self.gripper.command_gripper(
            rPRA=position,
            rSP=self.speed,
            rFR=self.force,
            rMOD=self.mode,
            rICF=True,
        )
        self.gripper.wait_until_stopped()
        return True
# ------------------------------------------------------------

    def get_opening(self) -> float:
        """Return the current opening, normalized [0,1], 0=open."""
        # IMPLEMENT: read from the gripper and return the normalized opening
        return 0.0  # placeholder

    def set_opening(self, value: float, blocking: bool = False) -> None:
        """Command the gripper to a specific opening [0,1], 0=open."""
        # IMPLEMENT: send command to the gripper

    def close(self) -> None:
        """Close the connection to the gripper."""
        # IMPLEMENT: disconnect from the gripper
        self._running = False
        self._thread.join()
        with self._io_lock:
            self._client.close()
        print("Connection closed.")


class RobotArmUR10:

    # =================================================================
    # Construction
    # =================================================================
    def __init__(self, cfg_path: str):
        # --- GIVEN: defaults set BEFORE connecting, so a failed init still
        # --- leaves a coherent object instead of AttributeError 200 lines later.
        self._T_base_in_cam: np.ndarray = np.eye(4)
        self._safe_z_min: Optional[float] = None
        self._home_joints: Optional[list] = None
        self._gripper = None
        self._servo_lookahead: float = 0.10      # s, valid range 0.03-0.2
        self._servo_gain: float = 300.0          # valid range 100-2000
        self._lin_speed: float = 0.10            # m/s, blocking moves
        self._lin_accel: float = 0.30            # m/s^2
        self._max_joint_jump: float = 1.0        # rad, singularity guard

        # IMPLEMENT: load your YAML. Needs at minimum:
        #   ip, tcp_offset, payload_kg, payload_cog, T_base_in_cam,
        #   home_joints, safe_z_min, gripper{ip, port, unit_id, mode, speed, force}
        self.cfg = load_cfg(cfg_path)                                  # IMPLEMENT
        print(self.cfg)  # DEBUG: verify the config loaded correctly

        if not self.init():
            raise RuntimeError(f"RobotArmUR10 init failed for {cfg_path}")

    def init(self) -> bool:
        """IMPLEMENT. Returns True on success. Do NOT swallow the failure —
        __init__ raises on False so you never get a half-built object."""
        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.cfg.ur10_ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.cfg.ur10_ip)

            # self.tcp_pose = self.rtde_r.getActualTCPPose()
            # print(f"Current TCP pose ",{self.tcp_pose})

            # # TCP offset: where the gripper's grasp point is relative to the
            # # flange. Without this every pose you command is at the flange.
            # self.rtde_c.setTcp(list(self.cfg.tcp_offset))

            # # setPayload takes (mass, cog) — BOTH required, cog in the flange
            # # frame. A 3-Finger is ~2.3 kg; a wrong CoG on a UR10e causes drift
            # # and spurious protective stops. Measure it: PolyScope ->
            # # Installation -> Payload -> Measure.
            # self.rtde_c.setPayload(float(self.cfg.payload_kg),list(self.cfg.payload_cog))

            # self._safe_z_min = float(self.cfg.safe_z_min)     # BASE-frame floor
            # self._home_joints = list(self.cfg.home_joints)
            self._T_base_in_cam = np.asarray(self.cfg.T_base_in_cam, float).reshape(4, 4)
            self.init_gripper()
            return True
        except Exception as e:
            print(f"║ [RobotArmUR10] Init failed: {e}")
            return False

    def init_gripper(self) -> None:
        """IMPLEMENT. Construct + activate the gripper.

        Do NOT default to self.cfg.ip — the 3-Finger controller box is usually
        a DIFFERENT address from the robot. Do NOT silently set _gripper = None
        on failure: get_gripper() would then report 'always open' forever, and
        that value goes straight into the ICT hand token.
        """
        gcfg = self.cfg.get("gripper", {})
        self.gripper_mode = str(gcfg.get("mode","Pinch"))
        print(gcfg)  # DEBUG: verify the gripper config loaded correctly
        if not gcfg or not gcfg.get("enabled", False):
            print("║ [RobotArmUR10] Gripper disabled in config (gripper.enabled: false).")
            return
        try:
            self._gripper = Robotiq3FGripper(self.cfg.gripper_ip, port=int(gcfg.get("port", 502)), unit_id=int(gcfg.get("unit_id", 9)), mode=self.gripper_mode)
        except Exception as e:
            print(f"║ [RobotArmUR10] Gripper init failed, continuing without it: {e}")
            self._gripper = None

        self._gripper.activate()

        self._gripper.set_mode(self.gripper_mode)
        self.gripper_individual_control=False

    # =================================================================
    # THE CONTRACT — these six + T_base_in_cam are all the loop uses
    # =================================================================

    @property
    def T_base_in_cam(self) -> np.ndarray:
        """GIVEN. (4,4) BASE -> CAMERA OPTICAL. From hand-eye calibration.

        CONTRACT
          * Direction is base->cam: get_T_ee_in_cam() == T_base_in_cam @ T_ee_in_base
          * A property, not a plain attribute, so recalibration propagates.
        """
        return self._T_base_in_cam

    # -----------------------------------------------------------------
    def get_T_ee_in_cam(self) -> np.ndarray:
        """GIVEN (once get_T_ee_in_base works). EE pose in the camera frame.

        CONTRACT
          * Returns (4,4) SE(3), meters, camera OPTICAL frame.
          * Called at replan rate (~1.25 Hz) AND by controller.py:99 on IK
            failure, so it must always return something valid.
          * NEVER return a zero/identity pose as a failure fallback — that tells
            the policy the EE is at the robot base origin.
        """
        return self.T_base_in_cam @ self.get_T_ee_in_base()

    # -----------------------------------------------------------------
    def move_ee_in_cam(self, T_ee_in_cam: np.ndarray,
                       duration: float, blocking: bool = False) -> bool:
        """IMPLEMENT. Servo the EE to a Cartesian target given in CAMERA frame.

        CONTRACT  — this is the method with the most ways to go subtly wrong
          * MUST be non-blocking when blocking=False. controller.py does its own
            time.sleep(dt); blocking here stalls the servo stream and the UR's
            watchdog decelerates mid-trajectory.
          * MUST return False when the target is unreachable / IK fails.
            controller.py:96-99 uses this for anti-windup — on False it resyncs
            its smoothing state to the measured pose. servoL returning True means
            'command accepted', NOT 'pose reachable', so you must pre-check.
          * `duration` is the control period (dt), not a speed.
          * Clamp base-frame Z to self._safe_z_min AFTER converting to base.

        Suggested body:
            T_ee_in_base = inv(self.T_base_in_cam) @ T_ee_in_cam
            p = self.T_to_p(T_ee_in_base)
            if p[2] < self._safe_z_min: p[2] = self._safe_z_min   # + count it
            if not self._pose_reachable(p): return False
            if blocking:
                self.rtde_c.servoStop()          # servoL and moveL do not mix
                return self.rtde_c.moveL(list(p), self._lin_speed, self._lin_accel, False)
            return self.rtde_c.servoL(list(p), 0.0, 0.0, duration,
                                      self._servo_lookahead, self._servo_gain)
        """
        ...                                                             # IMPLEMENT

    # -----------------------------------------------------------------
    def get_gripper(self) -> float:
        """IMPLEMENT. Current opening, normalized [0,1], 0 = OPEN.

        CONTRACT
          * This value goes DIRECTLY into the ICT hand token (policy.py:234) —
            it is a real observation, not telemetry.
          * MUST NOT block. Read a cached snapshot from a background poller;
            a Modbus round-trip with a 2 s timeout stalls the control loop.
          * NO inversion for the 3-Finger: gPOA/255 is already 0=open..1=closed.
          * On read failure, raise or return the last good value — never a
            silent 0.0, which the policy reads as a confident 'not grasping'.
        """
        ...                                                             # IMPLEMENT

    # -----------------------------------------------------------------
    def set_gripper(self, value: float, blocking: bool = False) -> None:
        """IMPLEMENT. Command the gripper. 0.0 = open, 1.0 = closed.

        CONTRACT
          * controller.py:101 calls this EVERY control step (10 Hz) with only
            0.0 or 1.0 — never intermediate values. Make it EDGE-TRIGGERED:
            cache the last commanded state and return immediately if unchanged,
            or you will saturate the Modbus link and re-trigger motion each cycle.
          * MUST NOT block when blocking=False.
          * Grasp MODE (pinch) is set once at activation and frozen. Never
            change it here — a mode change is ~1 s of physical reconfiguration
            and desyncs the policy's gripper observation mid-episode.
        """
        ...                                                             # IMPLEMENT

    # -----------------------------------------------------------------
    def go_home(self, blocking: bool = True) -> None:
        """IMPLEMENT. Move to the configured home joint configuration.

        CONTRACT
          * Called at run_inference.py:233 (setup) and controller.py:108
            (teardown, immediately after a servo session) — so servoStop() FIRST
            or moveJ can throw / behave erratically.
          * home_joints must come from config. Do not default to zeros: on a
            UR10e that is the fully-extended horizontal pose near a singularity.
        """
        ...                                                             # IMPLEMENT

    # -----------------------------------------------------------------
    def close(self) -> None:
        """IMPLEMENT. Teardown. Not the gripper — the connection.

        CONTRACT
          * Called once from run_inference.py:282, inside a finally block.
          * servoStop() then stopScript(), then disconnect the gripper.
          * Must not raise: it runs during cleanup, possibly after another error.
        """
        try:
            self.rtde_c.servoStop()
        except Exception:
            pass
        try:
            self.rtde_c.stopScript()
        except Exception:
            pass
        if self._gripper is not None:
            self._gripper.close()
        print("║ [RobotArmUR10] Connection Closed Safely.")                                                        # IMPLEMENT

    # =================================================================
    # Internals the six above depend on
    # =================================================================

    @staticmethod
    def T_to_p(T: np.ndarray) -> np.ndarray:
        """GIVEN. 4x4 -> UR's [x,y,z,rx,ry,rz] with an AXIS-ANGLE rotvec.
        Frame unchanged — this is representation only.
        NOTE: rx,ry,rz are NOT roll/pitch/yaw. Never edit them componentwise."""
        return np.concatenate([T[:3, 3], R.from_matrix(T[:3, :3]).as_rotvec()])

    @staticmethod
    def p_to_T(p) -> np.ndarray:
        """GIVEN. Exact inverse of T_to_p."""
        p = np.asarray(p, float)
        T = np.eye(4)
        T[:3, :3] = R.from_rotvec(p[3:6]).as_matrix()
        T[:3, 3] = p[:3]
        return T

    def get_T_ee_in_base(self) -> np.ndarray:
        """IMPLEMENT (one line). FK — on a UR there is no DH chain to evaluate,
        RTDE already reports the TCP pose for whatever TCP you set via setTcp.

            return self.p_to_T(self.rtde_r.getActualTCPPose())
        """
        ...                                                             # IMPLEMENT

    def _pose_reachable(self, p) -> bool:
        """IMPLEMENT. The check move_ee_in_cam needs to return an honest False.

        Three tests, cheapest first:
            1. self.rtde_c.isPoseWithinSafetyLimits(list(p))
            2. q = self.rtde_c.getInverseKinematics(list(p), self.rtde_r.getActualQ())
               -> falsy or len != 6 means no solution
            3. max|q - q_now| > self._max_joint_jump  -> a solution exists but
               requires a wrist flip; near a singularity the arm would whip.
        """
        ...                                                             # IMPLEMENT


# =====================================================================
# Bring-up CLI — not part of the contract, but do not skip step 1.
# =====================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UR10e bring-up")
    parser.add_argument("--cfg_path", type=str, required=True)
    args = parser.parse_args()

    arm = RobotArmUR10(args.cfg_path)
    try:
        gripper_speed = [250, 250, 250]  # speed for each finger
        gripper_force = [50, 50, 50]  # force for

        position = [0]
        position = [position[0]]* 3
        arm._gripper.command_gripper(rPRA=position,rSP=gripper_speed,rFR=gripper_force,rMOD=arm.gripper_mode, rICF=arm.gripper_individual_control)
        arm._gripper.wait_until_stopped()
        arm._gripper.check_fault(raise_on_major=False)

        target_pose = np.copy(arm.rtde_r.getActualTCPPose())
        target_pose += [0.05, 0, 0.0, 0, 0, 0]
        speed = 0.5 #m/s
        accel = 0.3 # m/s^2
        # Plans motion in Cartesian target
        arm.rtde_c.moveL(target_pose, speed, accel)
        print("Moved to target pose")

        position = [255]
        position = [position[0]]* 3
        arm._gripper.command_gripper(rPRA=position,rSP=gripper_speed,rFR=gripper_force,rMOD=arm.gripper_mode, rICF=arm.gripper_individual_control)
        arm._gripper.wait_until_stopped()
        arm._gripper.check_fault(raise_on_major=False)

        # 1. VERIFY THE EXTRINSIC BEFORE ANYTHING ELSE.
        #    Jog to 3 known points, run each through get_T_ee_in_cam(), project
        #    with frame.K, and check the dots land on the gripper in the RGB
        #    image. A wrong T_base_in_cam produces plausible numbers and sends
        #    the arm a metre off — no exception, no warning.
        # print("TCP (base):", arm.T_to_p(arm.get_T_ee_in_base()))
        # print("TCP (cam): ", arm.T_to_p(arm.get_T_ee_in_cam()))
        # print("gripper:   ", arm.get_gripper())
    finally:
        arm.close()
    # arm.close()