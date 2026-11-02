# using python ModBus TCP client. 
from pyModbusTCP.client import ModbusClient

import time
import threading
import warnings

# gFLT fault codes from the Robotiq 3-Finger Gripper instruction manual 
# read register 1 High byte -> BYTE 2 (out of all the bytes)
# priority faults - gripper refused the command because prerequisite were not met
# minor faults - recoverable
# major faults - require a reset (rACT=0) to clear

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

# dedicated class exoection for gripper specific faults rather than python bug
class GripperFault(RuntimeError):
    pass


class GripperController:
    # port 502 assigbed modBUS TCP port 
    # unit id 9 is factory slave address for Gripper
    # update interval - 10Hz status polling
    def __init__(self, server_ip, port=502, unit_id=9, update_interval=0.1):
        # Status fields (populated by the update thread)
        # predeclare all attributes so if read before first poll land no AttributeError
        self.gSTA = self.gIMC = self.gGTO = self.gACT = None
        self.gMOD = None
        self.gDTS = self.gDTC = self.gDTB = self.gDTA = None
        self.FaultStatus = 0
        self.FingerA_PositionReqEcho = self.FingerA_Position = self.FingerA_Current = None
        self.FingerB_PositionReqEcho = self.FingerB_Position = self.FingerB_Current = None
        self.FingerC_PositionReqEcho = self.FingerC_Position = self.FingerC_Current = None
        self.Scissor_PositionReqEcho = self.Scissor_Position = self.Scissor_Current = None

        # One lock guards ALL Modbus traffic: pyModbusTCP's client shares a
        # single socket, so concurrent read (update thread) + write (main
        # thread) can corrupt transactions.
        self._io_lock = threading.Lock()
        self.client = ModbusClient(host=server_ip, port=port, auto_open=True, unit_id=unit_id)
        self.client.open()

        self.update_interval = update_interval
        # Flag to keep the loop running
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

        # Give the update thread a moment to populate status once
        time.sleep(2 * update_interval)

    # ------------------------------------------------------------------ I/O

    def _read_registers(self):
        with self._io_lock:
            return self.client.read_input_registers(0, 8)

    def _write_registers(self, regs):
        with self._io_lock:
            ok = self.client.write_multiple_registers(0, regs)
        if not ok:
            warnings.warn(f"Modbus write failed for registers {regs}")
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

    def close(self):
        self._running = False
        self._thread.join()
        with self._io_lock:
            self.client.close()
        print("Connection closed.")


def main():
    gripper = GripperController("192.168.1.11")
    gripper.activate()                      # blocks until calibration done
    gripper.set_mode("Basic")               # explicit, waits for completion

    target_position = [10, 20, 11]
    speed = [250, 205, 250]
    force = [250, 250, 250]
    individual_control = False
    if not individual_control:
        target_position = [target_position[0]] * 3

    gripper.command_gripper(rPRA=target_position, rSP=speed, rFR=force, rICF=individual_control)
    sta = gripper.wait_until_stopped()
    print(f"Motion done, gSTA={sta}, positions: "
          f"A={gripper.FingerA_Position} B={gripper.FingerB_Position} C={gripper.FingerC_Position}")
    gripper.check_fault()
    gripper.close()


if __name__ == "__main__":
    main()