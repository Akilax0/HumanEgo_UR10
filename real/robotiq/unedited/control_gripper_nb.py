from GripperController import GripperController

class GripperControlInterface:
    def __init__(self, ip_address="10.0.0.187"):
        self.gripper = GripperController(ip_address)
        self.gripper.activate()
        print("Gripper activated")

        self.speed = [250, 205, 250]
        self.force = [50, 50, 50]
        self.mode = "Pinch"
        self.individual_control = False

        # Default to open position
        self.target_position = [0, 0, 0]
        self.send_command(self.target_position)

    def send_command(self, position):
        if not self.individual_control:
            position = [position[0]] * 3
        self.gripper.command_gripper(
            rPRA=position,
            rSP=self.speed,
            rFR=self.force,
            rMOD=self.mode,
            rICF=self.individual_control
        )

    def process_gripper_command(self, command: str):
        """Call this method externally to control the gripper."""
        if command == 'c':
            print("Closing gripper")
            self.send_command([250])
        elif command == 'o':
            print("Opening gripper")
            self.send_command([0])
        elif command == 'e':
            print("Exiting and closing connection")
            self.gripper.close()
            return False  # signal to stop
        else:
            print("Unknown command. Use 'c', 'o', or 'e'")
        return True  # continue running
    
    def get_joint_positions(self):
        """Retrieve current joint positions."""
        return [
            self.gripper.FingerA_Position,
            self.gripper.FingerB_Position,
            self.gripper.FingerC_Position
        ]
    
    def ind_control(self, position):
        self.gripper.command_gripper(
            rPRA=position,
            rSP=self.speed,
            rFR=self.force,
            rMOD=self.mode,
            rICF=True
        )
        return True

# Example usage:
if __name__ == '__main__':
    control = GripperControlInterface()
    ans = input("Press Enter once the gripper calibration is done to continue. anything else to exit...")
    if ans != '':
        control.gripper.close()
        exit(0)
    keep_running = True
    # while keep_running:
    #     user_input = input("Press c to close, o to open, e to exit: ")
    #     keep_running = control.process_gripper_command(user_input)
    #     print(control.get_joint_positions())
    #     keep_running = control.ind_control([0, 50, 0])
    control.gripper.close()
