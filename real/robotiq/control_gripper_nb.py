from GripperController import GripperController

# Thin application layer 
class GripperControlInterface:
    def __init__(self, ip_address="10.0.0.187"):
        self.gripper = GripperController(ip_address)

        # Blocks until gIMC == 3 (calibration actually finished) — no more
        # racing a motion/mode-change command against the activation sequence.
        self.gripper.activate()

        # speed 250 ~ 108 mm/s 
        # 15 + (50/255) * 45 ~ 24 N 
        self.speed = [250, 205, 250]
        self.force = [50, 50, 50]
        self.mode = "Pinch"
        self.individual_control = False

        # Do the mode change as its own step and wait for it. The scissor
        # swing of the Pinch mode change is the "finger moving in" motion —
        # it must be allowed to complete before any go-to command.
        self.gripper.set_mode(self.mode)

        # Now command the open position
        self.target_position = [0, 0, 0]
        self.send_command(self.target_position)
        self.gripper.wait_until_stopped()

    def send_command(self, position):
        if not self.individual_control:
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


if __name__ == '__main__':
    control = GripperControlInterface()
    keep_running = True
    while keep_running:
        print("gSTA = ", control.gripper.gSTA)
        print("gIMC = ", control.gripper.gIMC)
        print("gGTO = ", control.gripper.gGTO)
        print("gACT = ", control.gripper.gACT)
        print("gMOD = ", control.gripper.gMOD)
        print("FaultStatus = ", control.gripper.FaultStatus)
        print("FingerA_Position = ", control.gripper.FingerA_Position)
        print("FingerA_PositionReqEcho = ", control.gripper.FingerA_PositionReqEcho)
        print("FingerA_Current = ", control.gripper.FingerA_Current)
        print("FingerB_Position = ", control.gripper.FingerB_Position)
        print("FingerB_PositionReqEcho = ", control.gripper.FingerB_PositionReqEcho)
        print("FingerB_Current = ", control.gripper.FingerB_Current)
        print("FingerC_Position = ", control.gripper.FingerC_Position)
        print("FingerC_PositionReqEcho = ", control.gripper.FingerC_PositionReqEcho)
        print("FingerC_Current = ", control.gripper.FingerC_Current)
        user_input = input("Press c to close, o to open, e to exit: ")
        keep_running = control.process_gripper_command(user_input)
        if keep_running:
            print(control.get_joint_positions())

    control.gripper.close()