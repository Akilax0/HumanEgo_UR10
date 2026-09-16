# -*- coding: utf-8 -*-
# @FileName: RobotArmUR10.py
"""
UR10e driver over Universal Robots' RTDE interface (the `ur_rtde` python
package: `pip install ur_rtde`, imported as `rtde_control` / `rtde_receive`),
with an optional Robotiq gripper on the tool's Modbus/RS-485-to-socket
gateway (the UR controller exposes it on port 63352).

This is the UR10e + Robotiq counterpart to the paper's `RobotArmTrossen.py`
example, satisfying the same `RobotArm` surface `run_inference.py`'s adapter
wraps: FK, Cartesian servo, gripper, homing, and the `T_base_in_cam` hand-eye
extrinsic (see interfaces.py for the contract).

Frame note: unlike Trossen's driver, UR's RTDE TCP pose is already
[x, y, z, rx, ry, rz] (position + rotation vector) in the robot BASE frame for
whichever TCP offset is configured -- there is no extra frame-correction
matrix to apply. We set the TCP offset once via `setTcp()` at connect time.

Motion note: `move_ee_in_cam()` is called once per control step by
`TrajectoryController` (receding horizon, ~5-10 Hz). That maps directly onto
RTDE's `servoL()`, which is designed for exactly this "stream a new Cartesian
target every dt seconds" pattern -- so non-blocking moves use `servoL`, and
only `go_home` / blocking calls use `moveJ` / `moveL`.
"""

from __future__ import annotations

import argparse
import socket

import time
import threading 
import warnings
from pyModbusTCP.client import ModbusClient

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

import rtde_control
import rtde_receive

from utils.utils_io import load_cfg
from dataclasses import dataclass

import traceback
from workspace_minimal import fit_box,check_pose,norm_pose
import os
import sys
import json


# =====================================================================
# Robotiq gripper — Modbus-over-socket (port 63352), ASCII protocol
# =====================================================================

# Fixing here using the existing gripper control script

# -*- coding: utf-8 -*-
# @FileName: Robotiq3FGripper.py
"""
Robotiq 3-Finger Adaptive Gripper over Modbus TCP.

Replaces the port-63352 ASCII `RobotiqGripper` in RobotArmUR10.py, which spoke
the *2F* URCap socket protocol and does not apply to the 3-Finger.

Protocol / register map follows the reference `GripperController` driver
(pyModbusTCP, unit_id 9, `write_multiple_registers(0, ...)` for the robot->gripper
block and `read_input_registers(0, 8)` for gripper->robot). Bit packing was
verified bit-identical to that driver across all 65536 status words and every
action-request combination.

ROBOT -> GRIPPER (holding registers, word = hi_byte << 8 | lo_byte)
    reg0   ACTION REQUEST      | GRIPPER OPTIONS
    reg1   (reserved)          | rPRA   position request, finger A
    reg2   rSPA speed A        | rFRA   force A
    reg3   rPRB               | rSPB          } only sent when rICF = 1
    reg4   rFRB               | rPRC          }
    reg5   rSPC               | rFRC          }

GRIPPER -> ROBOT (input registers)
    reg0   GRIPPER STATUS      | OBJECT DETECTION
    reg1   gFLT fault          | gPRA request echo A
    reg2   gPOA position A     | gCUA current A
    reg3   gPRB               | gPOB
    reg4   gCUB               | gPRC
    reg5   gPOC               | gCUC
    reg6   gPRS scissor echo   | gPOS scissor position
    reg7   gCUS               | (reserved)

CONVENTIONS THAT MATTER FOR HumanEgo
    * rPRA/gPOA: 0 = fully OPEN, 255 = fully CLOSED. This already matches
      interfaces.py ("0.0 = OPEN, 1.0 = CLOSED"), so there is NO polarity flip
      here -- do not copy the `1.0 - x` from run_inference.py's TrossenArm.
    * Grasp MODE (rMOD) is set ONCE at activation and then frozen. A mode change
      is a ~1 s physical reconfiguration; doing it mid-episode would desync the
      policy's gripper observation. `set_mode()` refuses unless force=True.
    * `get_current_position()` and `get_status()` are NON-BLOCKING: they return
      the latest snapshot from the background poller, so calling them inside a
      10 Hz control loop costs nothing. Only `move(..., blocking=True)` waits.
    * `move()` is EDGE-TRIGGERED: re-sending an unchanged target is a no-op, so
      TrajectoryController calling set_gripper() every control step does not
      spam the bus or re-trigger motion.
"""


# =====================================================================
# Status snapshot
# =====================================================================

# gIMC -- initialization / mode-change status
IMC_RESET, IMC_ACTIVATING, IMC_MODE_CHANGE, IMC_COMPLETE = 0, 1, 2, 3
# gSTA -- motion status
STA_IN_MOTION, STA_SOME_STOPPED, STA_ALL_STOPPED, STA_ALL_REACHED = 0, 1, 2, 3
# gDTA/gDTB/gDTC -- per-finger object detection
DT_IN_MOTION, DT_CONTACT_OPENING, DT_CONTACT_CLOSING, DT_AT_REQUESTED = 0, 1, 2, 3

MODES: Dict[str, int] = {"basic": 0, "pinch": 1, "wide": 2, "scissor": 3}
MODE_NAMES = {v: k for k, v in MODES.items()}

POINTS_FILE = os.environ.get("UR_BOX_FILE", "workspace_box.json")
TCP_OFFSET = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

@dataclass(frozen=True)
class GripperStatus:
    """One immutable decode of the 8 input registers."""
    gACT: int; gMOD: int; gGTO: int; gIMC: int; gSTA: int
    gDTA: int; gDTB: int; gDTC: int; gDTS: int
    gFLT: int
    pos: Tuple[int, int, int]          # gPOA, gPOB, gPOC   (0 open .. 255 closed)
    cur: Tuple[int, int, int]          # gCUA, gCUB, gCUC   (motor current)
    echo: Tuple[int, int, int]         # gPRA, gPRB, gPRC   (request echo)
    scissor_pos: int
    ts: float                          # time.monotonic() of the read

    # -- readiness ---------------------------------------------------
    @property
    def activated(self) -> bool:
        return self.gACT == 1 and self.gIMC == IMC_COMPLETE

    @property
    def faulted(self) -> bool:
        return self.gFLT != 0

    @property
    def fault_is_transient(self) -> bool:
        """Priority faults (0x05-0x07) mean 'not ready yet, action deferred'.
        They clear themselves once initialization completes -- do NOT raise."""
        return self.gFLT in (0x05, 0x06, 0x07)

    @property
    def fault_is_major(self) -> bool:
        """0x0D activation fault, 0x0E scissor interference, 0x0F auto-release done.
        These need a reset; they will not clear on their own."""
        return self.gFLT in (0x0D, 0x0E, 0x0F)

    @property
    def mode_name(self) -> str:
        return MODE_NAMES.get(self.gMOD, f"?{self.gMOD}")

    # -- motion ------------------------------------------------------
    @property
    def in_motion(self) -> bool:
        return self.gSTA == STA_IN_MOTION

    @property
    def motion_done(self) -> bool:
        return self.gSTA != STA_IN_MOTION

    # -- grasp evidence (this is the useful part) ---------------------
    @property
    def object_detected(self) -> bool:
        """True if ANY finger stalled on contact rather than reaching its target.
        This is a SENSED grasp signal -- unlike 'we sent a close command'."""
        return any(d in (DT_CONTACT_OPENING, DT_CONTACT_CLOSING)
                   for d in (self.gDTA, self.gDTB, self.gDTC))

    @property
    def n_fingers_contacting(self) -> int:
        return sum(d == DT_CONTACT_CLOSING for d in (self.gDTA, self.gDTB, self.gDTC))

    @property
    def closed_on_nothing(self) -> bool:
        """Fingers finished travel with no contact -- an empty grasp."""
        return self.motion_done and not self.object_detected

    def __str__(self) -> str:
        return (f"[3F mode={self.mode_name} act={self.gACT} iMC={self.gIMC} "
                f"STA={self.gSTA} pos={self.pos} DT=({self.gDTA},{self.gDTB},{self.gDTC}) "
                f"obj={self.object_detected} flt={self.gFLT}]")


# =====================================================================
# Client
# =====================================================================

class Robotiq3FGripper:
    """Thread-safe Modbus TCP client for the Robotiq 3-Finger.

    Exposes the same surface RobotArmUR10 already calls -- `activate`,
    `move`, `move_and_wait_for_pos`, `get_current_position`, `close`,
    `_min_position`, `_max_position` -- so the arm driver needs only a
    two-line change in `init_gripper()`.
    """

    def __init__(
        self,
        hostname: str,
        port: int = 502,
        unit_id: int = 9,
        mode: str = "pinch",
        speed: int = 255,
        force: int = 150,
        poll_hz: float = 20.0,
        write_addr: int = 0,
        read_addr: int = 0,
        socket_timeout: float = 1.0,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
        self.mode = mode
        self.speed = int(speed)
        self.force = int(force)
        self._write_addr = write_addr
        self._read_addr = read_addr

        # Range kept for API compatibility with the old 2F class. The 3-Finger's
        # request register is a true 0-255 scale; auto-calibration is opt-in
        # because driving to both mechanical extremes at construction time will
        # crush or drop anything already in the fingers.
        self._min_position, self._max_position = 0, 255

        self._client = ModbusClient(host=hostname, port=port, unit_id=unit_id,
                                    auto_open=True, timeout=socket_timeout)
        if not self._client.open():
            raise ConnectionError(f"cannot open Modbus TCP to {hostname}:{port} (unit {unit_id})")

        self._io_lock = threading.Lock()      # pyModbusTCP client is NOT thread-safe
        self._status_lock = threading.Lock()
        self._status: Optional[GripperStatus] = None
        self._read_failures = 0
        self._last_sent: Optional[Tuple[int, int, int]] = None   # (pos, speed, force)
        self._activated = False

        # Prime the snapshot BEFORE starting the poller, so no caller can ever
        # observe a half-built object (the reference driver's race condition).
        self._poll_once()
        if self._status is None:
            self._client.close()
            raise ConnectionError("connected but no status registers readable")

        self._running = True
        self._poll_interval = 1.0 / max(1.0, poll_hz)
        self._thread = threading.Thread(target=self._poll_loop, daemon=True,
                                        name="robotiq3f-poll")
        self._thread.start()

    # ------------------------------------------------------------------
    # Bit packing  (verified identical to the reference driver)
    # ------------------------------------------------------------------
    @staticmethod
    def _action_request(rACT=0, rMOD=0, rGTO=0, rATR=0, rARD=0, rICF=0, rICS=0) -> int:
        """reg0 = ACTION REQUEST byte << 8 | GRIPPER OPTIONS byte."""
        b0 = (rACT & 1) | ((rMOD & 3) << 1) | ((rGTO & 1) << 3) \
             | ((rATR & 1) << 4) | ((rARD & 1) << 5)
        b1 = ((rICF & 1) << 2) | ((rICS & 1) << 3)
        return (b0 << 8) | b1

    @staticmethod
    def _word(hi: int, lo: int) -> int:
        if not (0 <= hi <= 255 and 0 <= lo <= 255):
            raise ValueError(f"byte out of range: hi={hi} lo={lo}")
        return (hi << 8) | lo

    @staticmethod
    def _decode(regs) -> GripperStatus:
        hi0, lo0 = (regs[0] >> 8) & 0xFF, regs[0] & 0xFF
        return GripperStatus(
            gACT=hi0 & 1,
            gMOD=(hi0 >> 1) & 3,
            gGTO=(hi0 >> 3) & 1,
            gIMC=(hi0 >> 4) & 3,
            gSTA=(hi0 >> 6) & 3,
            gDTA=lo0 & 3,
            gDTB=(lo0 >> 2) & 3,
            gDTC=(lo0 >> 4) & 3,
            gDTS=(lo0 >> 6) & 3,
            gFLT=(regs[1] >> 8) & 0xFF,
            echo=(regs[1] & 0xFF, (regs[3] >> 8) & 0xFF, regs[4] & 0xFF),
            pos=((regs[2] >> 8) & 0xFF, regs[3] & 0xFF, (regs[5] >> 8) & 0xFF),
            cur=(regs[2] & 0xFF, (regs[4] >> 8) & 0xFF, regs[5] & 0xFF),
            scissor_pos=regs[6] & 0xFF,
            ts=time.monotonic(),
        )

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------
    def _poll_once(self) -> bool:
        with self._io_lock:
            regs = self._client.read_input_registers(self._read_addr, 8)
        if not regs or len(regs) < 8:
            self._read_failures += 1
            return False
        st = self._decode(regs)
        with self._status_lock:
            self._status = st
            self._read_failures = 0
        return True

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self._poll_once()
            except Exception as e:                       # keep the thread alive
                self._read_failures += 1
                if self._read_failures in (1, 50):
                    print(f"║ [Robotiq3F] poll error ({self._read_failures}): {e}")
            time.sleep(self._poll_interval)

    def get_status(self) -> GripperStatus:
        """Latest snapshot. Non-blocking -- safe to call at control-loop rate."""
        with self._status_lock:
            st = self._status
        if st is None:
            raise RuntimeError("no gripper status available")
        return st

    @property
    def status_age(self) -> float:
        return time.monotonic() - self.get_status().ts

    def is_stale(self, max_age: float = 0.5) -> bool:
        return self.status_age > max_age

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def _write(self, regs) -> bool:
        with self._io_lock:
            ok = self._client.write_multiple_registers(self._write_addr, list(regs))
        return bool(ok)

    def _wait_for(self, pred, timeout: float, what: str, poll: float = 0.05,
                  tolerate_transient: bool = False) -> GripperStatus:
        t0 = time.monotonic()
        while True:
            st = self.get_status()
            if st.faulted and not (tolerate_transient and st.fault_is_transient):
                kind = "MAJOR " if st.fault_is_major else ""
                raise RuntimeError(
                    f"{kind}gripper fault gFLT=0x{st.gFLT:02X} while waiting for {what}"
                    + ("  (needs reset+reactivate)" if st.fault_is_major else ""))
            if pred(st):
                return st
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"timed out after {timeout:.1f}s waiting for {what}: {st}")
            time.sleep(poll)

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------
    def activate(self, auto_calibrate: bool = False, timeout: float = 15.0,
                 force_reactivate: bool = False) -> GripperStatus:
        """Reset, activate, and change into `self.mode`. Blocks until gIMC == 3.

        The reference driver's `activate()` fired rACT=1 and slept 1 s; that is
        not enough for a mode change, which is a real physical reconfiguration.
        Here we poll gIMC to completion instead.
        """
        st = self.get_status()
        target_mode = MODES[self.mode]

        if st.activated and st.gMOD == target_mode and not force_reactivate:
            print(f"║ [Robotiq3F] already active in {self.mode} mode; skipping activation.")
            self._activated = True
            return st

        # 1. clear rACT (reset). The gripper must see the 1->0 edge.
        if not self._write([self._action_request(rACT=0), 0, 0]):
            raise IOError("failed to write reset (rACT=0)")
        self._wait_for(lambda s: s.gIMC == IMC_RESET, timeout=5.0,
                       what="reset (gIMC=0)", tolerate_transient=True)

        # 2. ACTIVATE ONLY -- no rMOD change, no rGTO, no position request.
        #    Asking for activation and a mode change in one write raises a
        #    priority fault (gFLT=0x05, "action delayed, init must complete first").
        if not self._write([self._action_request(rACT=1, rGTO=0, rMOD=0, rICF=0), 0, 0]):
            raise IOError("failed to write activation request")
        st = self._wait_for(lambda s: s.gIMC == IMC_COMPLETE, timeout=timeout,
                            what="activation (gIMC=3)", tolerate_transient=True)
        print(f"\u2551 [Robotiq3F] activated (basic mode), pos={st.pos}")

        # 3. NOW change mode, as a separate transaction.
        if target_mode != 0:
            if not self._write([self._action_request(rACT=1, rGTO=0, rMOD=target_mode, rICF=0), 0, 0]):
                raise IOError("failed to write mode-change request")
            time.sleep(0.2)                      # let gIMC leave COMPLETE
            st = self._wait_for(lambda s: s.gIMC == IMC_COMPLETE and s.gMOD == target_mode,
                                timeout=timeout, what=f"{self.mode} mode change (gIMC=3)",
                                tolerate_transient=True)
        if st.gMOD != target_mode:
            raise RuntimeError(f"gripper reports mode {st.mode_name}, expected {self.mode}")
        if st.faulted:
            raise RuntimeError(f"gripper settled with fault gFLT=0x{st.gFLT:02X}")

        self._activated = True
        self._last_sent = None
        print(f"║ [Robotiq3F] ready: {st.mode_name} mode, pos={st.pos}, gFLT=0x{st.gFLT:02X}")

        if auto_calibrate:
            self._auto_calibrate()
        return st

    def _auto_calibrate(self) -> None:
        """Drive to both extremes once and record the reached counts.
        OPT-IN ONLY: this physically cycles the fingers. Never call it with an
        object in the gripper."""
        print("║ [Robotiq3F] auto-calibrating travel range (fingers will move)...")
        self.move_and_wait_for_pos(0, self.speed, 1)
        self._min_position = self.get_status().pos[0]
        self.move_and_wait_for_pos(255, self.speed, 1)
        self._max_position = self.get_status().pos[0]
        self.move_and_wait_for_pos(self._min_position, self.speed, 1)
        print(f"║ [Robotiq3F] travel range: {self._min_position}-{self._max_position}")

    def set_mode(self, mode: str, force: bool = False, timeout: float = 15.0) -> None:
        """Change grasp mode. Refuses once activated unless force=True, because a
        mid-episode mode change breaks the policy's gripper observation."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        if self._activated and not force:
            raise RuntimeError(
                f"refusing to change mode {self.mode!r}->{mode!r} after activation; "
                "a mode change takes ~1 s of physical reconfiguration and desyncs "
                "the policy's gripper state. Pass force=True if you really mean it.")
        self.mode = mode
        self.activate(auto_calibrate=False, timeout=timeout, force_reactivate=True)

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def move(self, position: int, speed: Optional[int] = None,
             force: Optional[int] = None, individual: Optional[Tuple[int, int, int]] = None) -> bool:
        """Command finger position, 0 = open .. 255 = closed.

        EDGE-TRIGGERED: an unchanged (pos, speed, force) triple is a no-op, so
        TrajectoryController hammering this at 10 Hz costs one dict compare.
        Pass `individual=(a, b, c)` to drive fingers separately (sets rICF=1).
        """
        speed = self.speed if speed is None else int(speed)
        force = self.force if force is None else int(force)
        rMOD = MODES[self.mode]

        if individual is not None:
            a, b, c = (int(min(255, max(0, v))) for v in individual)
            regs = [
                self._action_request(rACT=1, rGTO=1, rMOD=rMOD, rICF=1),
                self._word(0x00, a),
                self._word(speed, force),
                self._word(b, speed),
                self._word(force, c),
                self._word(speed, force),
            ]
            key = (("ind", a, b, c), speed, force)
        else:
            pos = int(min(255, max(0, position)))
            regs = [
                self._action_request(rACT=1, rGTO=1, rMOD=rMOD, rICF=0),
                self._word(0x00, pos),
                self._word(speed, force),
            ]
            key = (pos, speed, force)

        if key == self._last_sent:
            return True                       # already commanded -- no bus traffic
        ok = self._write(regs)
        if ok:
            self._last_sent = key
        return ok

    def move_and_wait_for_pos(self, position: int, speed: Optional[int] = None,
                              force: Optional[int] = None, timeout: float = 5.0) -> int:
        """Command and block until motion completes (gSTA != 0), then return gPOA.

        The reference driver's `main()` waited for gPOA to equal the request
        exactly, which essentially never happens -- encoder counts differ from
        the request, so that loop spins forever. gSTA is the correct signal.
        """
        self._last_sent = None                # force the write even if unchanged
        if not self.move(position, speed, force):
            return self.get_status().pos[0]
        # let the gripper report motion started before we look for it stopping
        t0 = time.monotonic()
        while self.get_status().motion_done and time.monotonic() - t0 < 0.3:
            time.sleep(0.02)
        try:
            st = self._wait_for(lambda s: s.motion_done, timeout=timeout, what="motion complete")
        except TimeoutError as e:
            print(f"║ [Robotiq3F] {e}")
            st = self.get_status()
        return st.pos[0]

    def stop(self) -> bool:
        """Clear rGTO -- fingers hold position."""
        self._last_sent = None
        return self._write([self._action_request(rACT=1, rGTO=0, rMOD=MODES[self.mode]), 0, 0])

    # ------------------------------------------------------------------
    # Reads used by RobotArmUR10
    # ------------------------------------------------------------------
    def get_current_position(self) -> int:
        """gPOA, 0 = open .. 255 = closed. Non-blocking (cached)."""
        return self.get_status().pos[0]

    def is_active(self) -> bool:
        return self.get_status().activated

    def is_object_detected(self) -> bool:
        return self.get_status().object_detected

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self._client.close()
        except Exception:
            pass
        print("║ [Robotiq3F] connection closed.")

    def __enter__(self) -> "Robotiq3FGripper":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# =====================================================================
# RobotArmUR10
# =====================================================================

class RobotArmUR10:
    def __init__(self, cfg_path: str):
        self.cfg_path = cfg_path
        self.cfg = load_cfg(cfg_path)
        self._gripper: Optional[Robotiq3FGripper] = None
        self._gripper_speed = 255
        self._gripper_force = 255
        self._gripper_invert = False
        # self.init()
        if not self.init():
            raise RuntimeError("RobotArmUR10 init failed.")

    def init(self) -> bool:
        try:
            print(f"║ [RobotArmUR10] Connecting to {self.cfg.ur10_ip}...")
            self.rtde_c = rtde_control.RTDEControlInterface(self.cfg.ur10_ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.cfg.ur10_ip)

            tcp_offset = list(self.cfg.get("tcp_offset", [0, 0, 0, 0, 0, 0]))
            if any(tcp_offset):
                self.rtde_c.setTcp(tcp_offset)
            payload_kg = float(self.cfg.get("payload_kg", 0.0))
            payload_cog = list(self.cfg.get("payload_cog", [0.0, 0.0, 0.0]))
            if payload_kg > 0:
                self.rtde_c.setPayload(payload_kg,payload_cog)

            self.init_T_base_in_cam()
            self.init_gripper()

            self._lin_speed = float(self.cfg.get("lin_speed", 0.25))       # m/s   (blocking moveL)
            self._lin_accel = float(self.cfg.get("lin_accel", 0.5))        # m/s^2 (blocking moveL)
            self._joint_speed = float(self.cfg.get("joint_speed", 1.0))    # rad/s (moveJ)
            self._joint_accel = float(self.cfg.get("joint_accel", 1.4))    # rad/s^2 (moveJ)
            self._servo_lookahead = float(self.cfg.get("servo_lookahead_time", 0.1))
            self._servo_gain = float(self.cfg.get("servo_gain", 300))
            self._safe_z_min = self.cfg.get("safe_z_min", None)            # base-frame Z floor ( table )

            print(f"║ [RobotArmUR10] Connected. TCP (base): {self.get_p()}")
            return True
        except Exception as e:
            print(f"║ [RobotArmUR10] Init Failed: {e}")
            return False

    def init_T_base_in_cam(self) -> None:
        if "T_base_in_cam" in self.cfg:
            self._T_base_in_cam = np.array(self.cfg.T_base_in_cam, dtype=np.float64)
        else:
            self._T_base_in_cam = np.eye(4)

    def init_gripper(self) -> None:
        gcfg = self.cfg.get("gripper", {})
        if not gcfg or not gcfg.get("enabled", False):
            print("║ [RobotArmUR10] Gripper disabled in config (gripper.enabled: false).")
            return
        try:
            self._gripper = Robotiq3FGripper(self.cfg.gripper_ip, port=int(gcfg.get("port", 502)))
            self._gripper.activate(auto_calibrate=bool(gcfg.get("auto_calibrate", True)))
            self._gripper_speed = int(gcfg.get("speed", 0))
            self._gripper_force = int(gcfg.get("force", 0))
            self._gripper_invert = bool(gcfg.get("invert", False))
            print(f"║ [RobotArmUR10] Robotiq gripper activated "
                 f"(range {self._gripper._min_position}-{self._gripper._max_position}).")
        except Exception as e:
            print(f"║ [RobotArmUR10] Gripper init failed, continuing without it: {e}")
            self._gripper = None

    def close(self) -> None:
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
        print("║ [RobotArmUR10] Connection Closed Safely.")


    # ------------------------------------------------------------------
    # Pose <-> matrix. UR's native pose IS [x, y, z, rotvec] in the base
    # frame for the configured TCP — no correction matrix needed.
    # ------------------------------------------------------------------
    def T_to_p(self, T: np.ndarray) -> np.ndarray:
        t = T[:3, 3]
        r = R.from_matrix(T[:3, :3]).as_rotvec()
        return np.concatenate([t, r])

    # single pose vector to SE(3) format
    def p_to_T(self, p: Union[List[float], np.ndarray]) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        T = np.eye(4)
        T[:3, :3] = R.from_rotvec(p[3:6]).as_matrix()
        T[:3, 3] = p[:3]
        return T

    @property
    def T_base_in_cam(self) -> np.ndarray:
        # Just the calibration matrix
        return self._T_base_in_cam

    def get_p(self) -> np.ndarray:
        """Current TCP pose [x, y, z, rx, ry, rz] (rotvec) in the robot BASE frame."""
        try:
            p = np.array(self.rtde_r.getActualTCPPose())
            self._last_valid_p = p
            return p
        except Exception:
            print("║ ⚠️ [Read Warning] Telemetry lost, using cached position.")
            return getattr(self, "_last_valid_p", np.zeros(6))

    def get_j(self) -> np.ndarray:
        # 6 element joint position vector
        return np.array(self.rtde_r.getActualQ())

    def get_T_ee_in_base(self) -> np.ndarray:
        return self.p_to_T(self.get_p())

    def get_T_ee_in_cam(self) -> np.ndarray:
        return self.T_base_in_cam @ self.get_T_ee_in_base()

    def p_in_cam_to_p_in_base(self, p_in_cam: Union[List[float], np.ndarray]) -> np.ndarray:
        T_ee_in_cam = self.p_to_T(p_in_cam)
        T_cam_in_base = np.linalg.inv(self.T_base_in_cam)
        return self.T_to_p(T_cam_in_base @ T_ee_in_cam)

    def p_in_base_to_p_in_cam(self, p_in_base: Union[List[float], np.ndarray]) -> np.ndarray:
        T_ee_in_base = self.p_to_T(p_in_base)
        return self.T_to_p(self.T_base_in_cam @ T_ee_in_base)

    def get_p_in_cam(self) -> np.ndarray:
        return self.T_to_p(self.get_T_ee_in_cam())

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def load_points(path="workspace_box.json"):
        path = "./inference_ur10/workspace_box.json"
        if not os.path.exists(path):
            sys.exit(f"{path} not found — run `teach` first.")
        with open(path) as f:
            data = json.load(f)

        m = data.get("tcp_offset_measured")
        if m is not None and np.linalg.norm(np.array(m[:3]) - np.array(TCP_OFFSET[:3])) > 0.005:
            print(f"NOTE: taught with TCP {np.round(m[:3], 4).tolist()}, "
                  f"script constant is {TCP_OFFSET[:3]}")

        corners = np.asarray(data["corners"], float)
        if corners.shape[0] < 4:
            sys.exit("need at least 4 taught corners")
        return corners
    
    def do_check(self, pose, margin):
        corners = self.load_points()
        print("corners:",corners)
        box = fit_box(corners[:, :3])
        print("box:",box)
        res = check_pose(box, pose, margin)
        print(f"pose          : {np.round(pose, 4).tolist()}")
        # print(f"min clearance : {res['min_clearance']:+.4f} m")
        print("RESULT        : " + ("INSIDE" if res else "OUTSIDE"))
        return res
        # sys.exit(0 if res["inside"] else 1)

    def move_p(self, pose: Union[List[float], np.ndarray], duration: float = 0.1,
              blocking: bool = False) -> bool:
        pose = np.array(pose, dtype=np.float64)
        if self._safe_z_min is not None and pose[2] < self._safe_z_min:
            print(f"║ [Safety] Z-Axis Clamped: {pose[2]:.4f} -> {self._safe_z_min}")
            pose[2] = self._safe_z_min
        try:
            if blocking:

                return bool(self.rtde_c.moveL(list(pose), self._lin_speed, self._lin_accel, False))
            # servoL: streamed Cartesian servo — matches TrajectoryController calling
            # this once per control step with duration == dt (receding horizon).
            print("==============================POSE LIST======================================",pose)
            pose_str = " ".join(map(str, pose))
            pose_str = pose
            print("Pose string:",pose_str)
            # print("norm_pose check:",norm_pose(pose_str))
            res = self.do_check(norm_pose(pose_str),0)
            print("OBS_CHECK", res)

            if not res:
                print("║ [Safety] Cartesian move aborted: pose outside workspace box.")
                return False
            return bool(self.rtde_c.servoL(list(pose), 0.0, 0.0, duration,
                                           self._servo_lookahead, self._servo_gain))
        except Exception as e:
            print(f"║ [Move Error] Cartesian move failed: {e}")
            print("functions: ", traceback.extract_tb(e.__traceback__))
            return False

    def move_p_in_cam(self, p_in_cam: Union[List[float], np.ndarray], duration: float = 0.1,
                      blocking: bool = False) -> bool:
        p_in_base = self.p_in_cam_to_p_in_base(p_in_cam)
        return self.move_p(p_in_base, duration, blocking)

    def move_ee_in_cam(self, T_ee_in_cam: np.ndarray, duration: float,
                       blocking: bool = False) -> bool:
        return self.move_p_in_cam(self.T_to_p(T_ee_in_cam), duration, blocking)

    def move_j(self, joints: Union[List[float], np.ndarray],
              speed: Optional[float] = None, accel: Optional[float] = None,
              blocking: bool = True) -> bool:
        try:
            return bool(self.rtde_c.moveJ(list(joints),
                                         speed if speed is not None else self._joint_speed,
                                         accel if accel is not None else self._joint_accel,
                                         not blocking))
        except Exception as e:
            print(f"║ [Move Error] Joint move failed: {e}")
            return False

    def go_home(self, blocking: bool = True) -> None:
        home = list(self.cfg.get("home_joints", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        print("║ [RobotArmUR10] Moving to home joints...")
        self.move_j(home, blocking=blocking)

    # ------------------------------------------------------------------
    # Gripper — normalized to [0, 1], 0=open .. 1=closed (matches interfaces.RobotArm)
    # ------------------------------------------------------------------
    def get_gripper_q(self) -> float:
        if self._gripper is None:
            return 0.0
        raw = self._gripper.get_current_position()
        lo, hi = self._gripper._min_position, self._gripper._max_position
        q = float(np.clip((raw - lo) / max(1, hi - lo), 0.0, 1.0))
        return 1.0 - q if self._gripper_invert else q

    def get_gripper(self) -> float:
        return self.get_gripper_q()

    def set_gripper(self, value: float, blocking: bool = False) -> None:
        print("Gripper Value:",value)
        if self._gripper is None:
            return
        value = float(np.clip(value, 0.0, 1.0))
        if self._gripper_invert:
            value = 1.0 - value
        lo, hi = self._gripper._min_position, self._gripper._max_position
        pos = int(round(lo + value * (hi - lo)))
        if blocking:
            print("Gripper Speed : ", self._gripper_speed, " Gripper Force : ", self._gripper_force)
            self._gripper.move_and_wait_for_pos(pos, self._gripper_speed, self._gripper_force)
        else:
            self._gripper.move(pos, self._gripper_speed, self._gripper_force)

    def open_gripper(self, blocking: bool = False) -> None:
        print("opening gripper...")
        self.set_gripper(0.0, blocking=blocking)

    def close_gripper(self, blocking: bool = False) -> None:
        self.set_gripper(1.0, blocking=blocking)

# ─────────────────────────────────────────────────────────────────────────────
# Replaces the __main__ block at the bottom of RobotArmUR10.py.
# Needs `import numpy as np`, `import time`, `import argparse` (already present).
# ─────────────────────────────────────────────────────────────────────────────


def pick_and_shift_test(
    arm,
    axis: int = 1,
    distance: float = 0.05,
    speed: float = 0.05,
    accel: float = 0.30,
    dwell: float = 1.0,
    assume_yes: bool = False,
) -> None:
    """Bring-up test: close the gripper (pinch), translate along ONE base axis,
    open the gripper, return to start.

    Deliberately conservative: blocking moves only (never servoL), an explicit
    IK/limits pre-check before every motion, and it always tries to return to
    the starting pose even if a step fails.

    Args:
        axis:     0 = base X, 1 = base Y, 2 = base Z. "Right" is almost always
                  base +Y, but VERIFY on your cell -- see the note printed below.
        distance: metres. Signed: negative goes the other way.
        speed:    m/s for moveL. 0.05 is a walking-pace crawl; keep it there.
    """
    axis_name = "XYZ"[axis]

    # --- 0. a servo session must be stopped before any blocking moveL/moveJ ---
    try:
        arm.rtde_c.servoStop()
    except Exception:
        pass

    p_start = np.asarray(arm.get_p(), dtype=float)
    p_target = p_start.copy()
    p_target[axis] += distance

    print("\n" + "=" * 68)
    print(f"  start  TCP (base): {np.round(p_start, 4)}")
    print(f"  target TCP (base): {np.round(p_target, 4)}")
    print(f"  motion: {distance*1000:+.0f} mm along base {axis_name}, at {speed} m/s")
    print("=" * 68)

    # --- 1. does the Z clamp change what we asked for? -----------------------
    z_floor = getattr(arm, "_safe_z_min", None)
    if z_floor is not None and p_target[2] < z_floor:
        print(f"  !! target z {p_target[2]:.4f} is below safe_z_min {z_floor:.4f}.")
        print(f"     move_p WILL clamp it and the tool will also rise "
              f"{z_floor - p_target[2]:.4f} m. Aborting instead.")
        return

    # --- 2. reachability pre-check (move_p does NOT do this for you) --------
    if not _pose_reachable(arm, p_target):
        print("  !! target failed the IK / safety-limits check. Aborting.")
        return

    # --- 3. gripper sanity --------------------------------------------------
    g = getattr(arm, "_gripper", None)
    if g is None:
        print("  !! no gripper connected -- running the motion only.")
    else:
        mode = getattr(g, "mode", "?")
        print(f"  gripper mode: {mode}")
        if mode != "pinch":
            print("  !! expected pinch mode. Set gripper.mode: pinch in the YAML.")
            return

    # --- 4. confirm ---------------------------------------------------------
    if not assume_yes:
        if input("\n  THE ROBOT WILL MOVE. Hand on the e-stop. Type 'go' to continue: ").strip() != "go":
            print("  aborted."); return

    # --- 5. run it ----------------------------------------------------------
    try:
        print("\n  [1/4] closing gripper...")
        arm.close_gripper(blocking=True)
        time.sleep(dwell)
        _report_grasp(g)

        print(f"  [2/4] moving {distance*1000:+.0f} mm along base {axis_name}...")
        if not arm.move_p(p_target, duration=speed, blocking=True):
            print("        move_p returned False -- stopping here.")
        else:
            print(f"        now at: {np.round(arm.get_p(), 4)}")
        time.sleep(dwell)

        print("  [3/4] opening gripper...")
        arm.open_gripper(blocking=True)
        time.sleep(dwell)

    finally:
        print("  [4/4] returning to start...")
        try:
            arm.rtde_c.servoStop()
        except Exception:
            pass
        if _pose_reachable(arm, p_start):
            arm.move_p(p_start, duration=speed, blocking=True)
            print(f"        back at: {np.round(arm.get_p(), 4)}")
        else:
            print("        !! start pose no longer reachable -- left in place.")
        print("  done.\n")


def _pose_reachable(arm, p) -> bool:
    """IK + joint-limit + joint-jump pre-check. This is what move_p is missing:
    servoL/moveL returning True means 'command accepted', NOT 'pose reachable'."""
    try:
        q_now = arm.rtde_r.getActualQ()
        if hasattr(arm.rtde_c, "isPoseWithinSafetyLimits"):
            if not arm.rtde_c.isPoseWithinSafetyLimits(list(map(float, p))):
                print("        pose outside safety limits"); return False
        q = arm.rtde_c.getInverseKinematics(list(map(float, p)), q_now)
        if not q or len(q) != 6:
            print("        no IK solution"); return False
        jump = float(np.max(np.abs(np.asarray(q) - np.asarray(q_now))))
        if jump > 1.0:                      # rad; a lateral 5 cm move should be tiny
            print(f"        IK solution jumps {jump:.2f} rad -- near a singularity")
            return False
        return True
    except Exception as e:
        print(f"        pre-check unavailable ({e}); proceeding without it")
        return True


def _report_grasp(g) -> None:
    """Use the 3-Finger's per-finger contact bits, not 'we sent a close command'."""
    if g is None or not hasattr(g, "get_status"):
        return
    try:
        st = g.get_status()
        if st.object_detected:
            print(f"        OBJECT DETECTED  ({st.n_fingers_contacting}/3 fingers in contact), "
                  f"gPOA={st.pos[0]}")
        else:
            print(f"        closed on nothing (gSTA={st.gSTA}, gPOA={st.pos[0]})")
    except Exception as e:
        print(f"        could not read gripper status: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UR10e bring-up: connect, print telemetry, optional home + gripper/motion test.")
    parser.add_argument("--cfg_path", type=str,
                        default="./cfg/inference/example_ur10/RobotArmUR10.yaml")
    parser.add_argument("--home", action="store_true", help="Move to home_joints on start.")
    parser.add_argument("--test_gripper", action="store_true",
                        help="Close (pinch), translate, open, return.")
    parser.add_argument("--axis", type=int, default=1, choices=[0, 1, 2],
                        help="Base axis to translate along: 0=X, 1=Y (default, usually 'right'), 2=Z.")
    parser.add_argument("--distance", type=float, default=0.05,
                        help="Metres to translate. Negative reverses direction. Default 0.05.")
    parser.add_argument("--speed", type=float, default=0.05, help="moveL speed, m/s.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    args = parser.parse_args()

    arm = RobotArmUR10(args.cfg_path)
    try:
        if args.home:
            arm.go_home(blocking=True)
        print(f"Joints:          {arm.get_j()}")
        print(f"TCP pose (base): {arm.get_p()}")
        print(f"TCP pose (cam):  {arm.get_p_in_cam()}")
        print(f"Gripper q:       {arm.get_gripper_q():.2f}")
        if args.test_gripper:
            pick_and_shift_test(arm, axis=args.axis, distance=args.distance,
                                speed=args.speed, assume_yes=args.yes)
    finally:
        arm.close()


# # ---------------------------------------------------------------------------
# # Bring-up CLI: connect, print telemetry, optionally home / exercise the gripper.
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="UR10e bring-up: connect, print telemetry, optional home + gripper test.")
#     parser.add_argument("--cfg_path", type=str,
#                         default="./cfg/inference/example_ur10/RobotArmUR10.yaml")
#     parser.add_argument("--home", action="store_true", help="Move to home_joints on start.")
#     parser.add_argument("--test_gripper", action="store_true", help="Open then close the gripper once.")
#     args = parser.parse_args()

#     arm = RobotArmUR10(args.cfg_path)
#     try:
#         if args.home:
#             arm.go_home(blocking=True)
#         print(f"Joints:          {arm.get_j()}")
#         print(f"TCP pose (base): {arm.get_p()}")
#         print(f"TCP pose (cam):  {arm.get_p_in_cam()}")
#         print(f"Gripper q:       {arm.get_gripper_q():.2f}")
#         if args.test_gripper:
#             print("Opening..."); arm.open_gripper(blocking=True)
#             time.sleep(5.0)
#             print("Closing..."); arm.close_gripper(blocking=True)
#     finally:
#         arm.close()

# # python -m inference_ur10.RobotArmUR10 --cfg_path ./cfg/inference/example_ur10/RobotArmUR10.yaml --home --test_gripper
