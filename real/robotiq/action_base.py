import rtde_control
import rtde_receive
import time
import numpy as np
from GripperController import GripperController

# Connect to the robot
ROBOT_IP = "10.0.0.219"
GRIPPER_IP = "10.0.0.187"
# Creating interfaced for communication over TCP
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

# ---- Read current state --------
# returns a list of 6 floats -> current angle of each joint in radians (base, shoulder, elbow, wrist1, wrist2, wrist3)
# joint_positions = rtde_r.getActualQ()
# tool center point pose as 6 floats: x,y,z in meters relative to robot base, rx, ry, rz orientation assh-add ~/.ssh/id_ed25519s a rotation vector 
tcp_pose = rtde_r.getActualTCPPose()
# print(f"Current joint poisitions (rad): {joint_positions}")
print(f"Current TCP pose [x,y,z,rx,ry,rz]: {tcp_pose}")

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

# exit(0)

# ----- Move in joint space ------------
# Move to a "home" position (all joints in radians)
# -1.5708 < -90 in radians 
# home = [0, -1.5708, 0, -1.5708, 0, 0]
# velocity = 0.5 #rad/s
# acceleration = 0.5 # rad/s^2
# # Plans motion in Joint space
# rtde_c.moveJ(home, velocity, acceleration)
# print("Moved to home position")


# ----- Move in Cartesian space ------
# moveL target: [x, y, z, rx, ry, rz] in meters / radians
# Coordinates from basehome = [0, -1.5708, 0, -1.5708, 0, 0]
# velocity = 0.5 #rad/s
# acceleration = 0.5 # rad/s^2
# # Plans motion in Joint space
# rtde_c.moveJ(home, velocity, acceleration)
# print("Moved to home position")


if __name__ == '__main__':
    # function blocks until target reached 
    # target_pose = [-0.143, -0.435, 0.20, -0.001, 3.12, 0.04]
    target_pose = np.copy(tcp_pose)
    target_pose += [0.05, 0, 0.0, 0, 0, 0]
    speed = 0.5 #m/s
    accel = 0.3 # m/s^2
    # Plans motion in Cartesian target
    # rtde_c.moveL(target_pose, speed, accel)
    # print("Moved to target pose")

    control = GripperControlInterface()



    while True:
        user_input = input("Press c to close, o to open, e to exit: ")
        keep_running = control.process_gripper_command(user_input)
        direction = input("Enter a direction L/R/U/D/F/B (left/right/up/down/forward/backward) or 'q' to quit: ")
        # target_pose = np.copy(tcp_pose)
        if direction.lower() == 'l':
            target_pose += [0, 0.05, 0, 0, 0, 0]  # Move left
            rtde_c.moveL(target_pose, speed, accel)
        elif direction.lower() == 'r':
            target_pose += [0, -0.05, 0, 0, 0, 0]  # Move right
            rtde_c.moveL(target_pose, speed, accel)
        elif direction.lower() == 'u':
            target_pose += [0, 0, 0.05, 0, 0, 0]  # Move up
            rtde_c.moveL(target_pose, speed, accel)
        elif direction.lower() == 'd':
            target_pose += [0, 0, -0.05, 0, 0, 0]  # Move down
            rtde_c.moveL(target_pose, speed, accel)
        elif direction.lower() == 'f':
            target_pose += [0.05, 0, 0, 0, 0, 0]  # Move forward
            rtde_c.moveL(target_pose, speed, accel)
        elif direction.lower() == 'b':
            target_pose += [-0.05, 0, 0, 0, 0, 0]  # Move backward
            rtde_c.moveL(target_pose, speed, accel)
        elif direction.lower() == 'q':
            break
        else:
            print("Invalid direction. Please enter L, R, U, D, F, B, or Q.")

    # # --- Read force / torque ----
    # # Returns a list of 6 floats [Fx, Fy, Fz, Tx, Ty, Tz] first three Newton forces along x,y,z . Torque in Nm around x,y,z
    # wrench = rtde_r.getActualTCPForce()
    # print(f"TCP force/torque: {wrench}")

    # # --- Simple waypoint sequence ---
    # # list of 4 cartesian poses. 
    # waypoints = [
    #     [-0.143, -0.435, 0.30, -0.001, 3.12, 0.04],
    #     [-0.243, -0.435, 0.30, -0.001, 3.12, 0.04],
    #     [-0.243, -0.535, 0.30, -0.001, 3.12, 0.04],
    #     [-0.143, -0.435, 0.20, -0.001, 3.12, 0.04],  # back to start
    # ]

    # # Iterates through the list. 
    # for i, wp in enumerate(waypoints):
    #     rtde_c.moveL(wp, 0.15, 0.8)
    #     print(f"Reached waypoint {i+1}")
    #     time.sleep(0.5)

    # # --- Servo (real-time) control example ---
    # # Useful for closed-loop / sensor-guided motion
    # # grab current pose as starting point
    # current = rtde_r.getActualTCPPose()
    # dt = 0.002  # 500 Hz control loop ( matches UR controllers internal rate)
    # lookahead = 0.1 # how much controller smooths incoming targets. (lower -> more responsive & jitterey)
    # gain = 300 # how aggressive robot tracks the target, -> higher stiffer.

    # for i in range(500):
    #     t_start = time.time()
    #     current[2] += 0.0001  # move 0.1 mm upward each step
    #     rtde_c.servoL(current, 0, 0, dt, lookahead, gain)
    #     elapsed = time.time() - t_start
    #     if elapsed < dt:
    #         time.sleep(dt - elapsed)

    # rtde_c.servoStop()
    # print("Servo motion complete")

    # --- Cleanup ---
    print("Cleaning up")
    control.gripper.close()
    rtde_c.stopScript()
    rtde_c.disconnect()
    rtde_r.disconnect()
    print("Disconnected")