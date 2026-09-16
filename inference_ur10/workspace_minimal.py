#!/usr/bin/env python3
"""
ur10e_teach_box.py

Collect 8 gripper-tip poses by hand-guiding, save them, and test poses
against a box you fit yourself.

Freedrive is enabled on the pendant. This script only reads.

    python ur10e_teach_box.py teach
    python ur10e_teach_box.py points
    python ur10e_teach_box.py tcp
    python ur10e_teach_box.py check -- 0.6 -0.05 0.34
    python ur10e_teach_box.py monitor

fit_box() and check_pose() are stubs. Fill them in.

Deps: pip install ur-rtde numpy scipy
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROBOT_IP = os.environ.get("UR_IP", "10.0.0.219")
POINTS_FILE = os.environ.get("UR_BOX_FILE", "workspace_box.json")

# What you believe the pendant's TCP is set to: flange -> gripper tip,
# [x, y, z, rx, ry, rz], metres and axis-angle radians. Recorded to the
# file for traceability; `tcp` cross-checks it against the robot.
# TCP_OFFSET = [0.0, 0.0, 0.2135, 0.0, 0.0, 0.0]
TCP_OFFSET = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

N_CORNERS = 8


# ===========================================================================
# YOUR CODE GOES HERE
# ===========================================================================
def fit_box(points):
    """Fit a box to the taught corners.

    Parameters
    ----------
    points : (N, 3) ndarray
        Gripper-tip positions in the robot base frame, metres.
        N is however many corners were recorded (up to 8).

    Returns
    -------
    dict
        Whatever representation you want, as long as check_pose() below
        understands it. A reasonable shape:

            {
                "center":       (3,) ndarray, base frame,
                "axes":         (3, 3) ndarray, unit row vectors,
                "half_extents": (3,) ndarray, metres,
            }

        With rows-as-axes, a base-frame point maps into the box frame as
        ``(p - center) @ axes.T`` and back out as ``local @ axes + center``.
    """
    
    
    print("fit_box().",points)
    x = []
    y = []
    z = []
    for i in range(len(points)):
        x.append(points[i][0])
        y.append(points[i][1])
        z.append(points[i][2])


    x = sorted(x)
    y = sorted(y)
    z = sorted(z)

    print(x,y,z)

    x_min = x[3]
    x_max = x[4]

    y_min = y[0]
    y_max = y[4]

    z_min = z[3]
    z_max = z[4]

    # Calculate the center of the box
    center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])

    # Calculate the half-extents of the box
    half_extents = np.array([(x_max - x_min) / 2, (y_max - y_min) / 2, (z_max - z_min) / 2])

    # Return the box representation
    return {
        "center": center,
        "half_extents": half_extents,
        "axes": np.eye(3),  # Identity matrix for axes,
        "values": [x_min, x_max, y_min, y_max, z_min, z_max]
    }

def check_pose(box, pose, margin=0.0):
    """Decide whether a pose is inside the box.

    Parameters
    ----------
    box : whatever fit_box() returned
    pose : list of 6 floats
        [x, y, z, rx, ry, rz] in the base frame. Position in metres,
        orientation as an axis-angle rotation vector in radians.
    margin : float
        Safety buffer in metres. Shrink the box by this much before testing.

    Returns
    -------
    dict
        Must contain at least:
            "inside"        : bool
            "min_clearance" : float, metres. Distance to the nearest face,
                              negative when outside.
        Add anything else you find useful.
    """
    print("Checking pose:",pose)


    x_min, x_max, y_min, y_max, z_min, z_max = box.get("values")

    if pose[0] < x_min + margin or pose[0] > x_max - margin or \
       pose[1] < y_min + margin or pose[1] > y_max - margin or \
       pose[2] < z_min + margin or pose[2] > z_max - margin:
        inside = False
    else:
        inside = True
         
    print("box:",box.get("values"))
    return inside
    # raise NotImplementedError("check_pose")


# ===========================================================================
# Robot I/O — receive only
# ===========================================================================
def connect():
    import rtde_receive
    import rtde_control
    return rtde_receive.RTDEReceiveInterface(ROBOT_IP), rtde_control.RTDEControlInterface(ROBOT_IP)


# UR10e nominal DH. Used only to sanity-check the pendant TCP setting.
_DH_A     = [0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0]
_DH_D     = [0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655]
_DH_ALPHA = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]


def _flange_pose(q):
    """Forward kinematics, base -> flange, as a 4x4 matrix."""
    T = np.eye(4)
    for i in range(6):
        ct, st = np.cos(q[i]), np.sin(q[i])
        ca, sa = np.cos(_DH_ALPHA[i]), np.sin(_DH_ALPHA[i])
        T = T @ np.array([
            [ct, -st * ca,  st * sa, _DH_A[i] * ct],
            [st,  ct * ca, -ct * sa, _DH_A[i] * st],
            [0.0,      sa,       ca, _DH_D[i]],
            [0.0,     0.0,      0.0, 1.0],
        ])
    return T


def _pose_to_matrix(pose):
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(np.asarray(pose[3:6], float)).as_matrix()
    T[:3, 3] = np.asarray(pose[:3], float)
    return T


def measure_tcp(rtde_r,rtde_c):

    """Recover the TCP offset the controller is applying, from joint angles
    and the reported TCP pose. Receive-only.

    Uses nominal DH, so treat differences under about 5 mm as calibration
    noise rather than a real mismatch.
    """
    from scipy.spatial.transform import Rotation as R
    # Returns 6 element angular vectore with current actual angular positions of robots joints in radians.
    # [Base, Shoulder, Elbow, Wrist 1, Wrist 2, Wrist 3]
    q = rtde_r.getActualQ()
    # 6 element vector with current position & orientation of the robot's tool center point. (end point)
    # [X, Y, Z, Rx, Ry, Rz] where R* are rotation vector components.
    pose = rtde_r.getActualTCPPose()
    print("Offset:",rtde_c.getTCPOffset())

    T_tcp = _pose_to_matrix(pose)
    print("Actual TCP pose:",pose)
    T_off = np.linalg.inv(_flange_pose(q)) @ T_tcp
    return list(T_off[:3, 3]) + list(R.from_matrix(T_off[:3, :3]).as_rotvec())


def do_tcp():
    rtde_r, rtde_c = connect()
    measured = measure_tcp(rtde_r, rtde_c)
    print(f"measured from the robot : {np.round(measured, 4).tolist()}")
    print(f"TCP_OFFSET in this file : {TCP_OFFSET}")
    err = np.linalg.norm(np.array(measured[:3]) - np.array(TCP_OFFSET[:3]))
    print(f"position difference     : {err * 1000:.1f} mm")
    if err > 0.005:
        print("\n*** The pendant TCP does not match TCP_OFFSET. ***")
        print("Recorded points follow the PENDANT value, not this file.")
    else:
        print("\nAgrees within nominal-kinematics noise.")


# ===========================================================================
# Teaching
# ===========================================================================
def do_teach():
    print(f"Connecting to {ROBOT_IP} ...")
    rtde_r,rtde_c = connect()

    try:
        measured = measure_tcp(rtde_r,rtde_c)
        print(f"TCP in use (measured): {np.round(measured[:3], 4).tolist()}")
        if np.linalg.norm(np.array(measured[:3]) - np.array(TCP_OFFSET[:3])) > 0.005:
            print(f"  WARNING: TCP_OFFSET says {TCP_OFFSET[:3]}")
    except Exception as e:
        print(f"  (TCP check skipped: {e})")
        measured = None

    print(f"\nEnable freedrive on the pendant, then move the GRIPPER TIP")
    print(f"to each of the {N_CORNERS} corners.")
    print("  Enter = record   r = redo last    q = abort\n")

    corners = []
    while len(corners) < N_CORNERS:
        key = input(f"corner {len(corners) + 1}/{N_CORNERS} > ").strip().lower()

        if key == "q":
            print("aborted, nothing saved")
            return
        if key == "r":
            if corners:
                dropped = corners.pop()
                print(f"  dropped {np.round(dropped[:3], 4).tolist()}")
            continue

        pose = rtde_r.getActualTCPPose()
        corners.append(list(pose))
        print(f"  recorded  xyz {np.round(pose[:3], 4).tolist()}"
              f"   rot {np.round(pose[3:], 4).tolist()}")

    save_points(corners, measured)
    summarise(np.array(corners)[:, :3])

    fit_box(np.array(corners)[:, :3]) 


def summarise(P):
    """Raw geometry of the taught points, before any fitting."""
    print("\nTaught points, base frame:")
    print("  #        x        y        z     reach")
    for i, p in enumerate(P):
        print(f"  {i + 1}  {p[0]: .4f}  {p[1]: .4f}  {p[2]: .4f}"
              f"   {np.hypot(p[0], p[1]):.3f}")

    lo, hi = P.min(axis=0), P.max(axis=0)
    d = hi - lo
    print(f"\n  extent x : {lo[0]: .4f} .. {hi[0]: .4f}   span {d[0]:.4f}")
    print(f"  extent y : {lo[1]: .4f} .. {hi[1]: .4f}   span {d[1]:.4f}")
    print(f"  extent z : {lo[2]: .4f} .. {hi[2]: .4f}   span {d[2]:.4f}")
    print(f"  axis-aligned volume: {np.prod(d) * 1e3:.1f} L")


# ===========================================================================
# Persistence
# ===========================================================================
def save_points(corners, measured_tcp, path=POINTS_FILE):
    with open(path, "w") as f:
        json.dump({
            "tcp_offset_script": TCP_OFFSET,
            "tcp_offset_measured": (list(map(float, measured_tcp))
                                    if measured_tcp is not None else None),
            "robot_ip": ROBOT_IP,
            "corners": [list(map(float, c)) for c in corners],
        }, f, indent=2)
    print(f"\nwrote {path}  ({len(corners)} corners)")


def load_points(path=POINTS_FILE):
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


# ===========================================================================
# CLI
# ===========================================================================
def norm_pose(vals):
    if len(vals) not in (3, 6):
        sys.exit("pose needs 3 numbers (position) or 6 (position + rotvec)")
    return list(vals) + ([0.0, 0.0, 0.0] if len(vals) == 3 else [])


def do_check(pose, margin):
    corners = load_points()
    print("corners:",corners)
    box = fit_box(corners[:, :3])
    print("box:",box)
    res = check_pose(box, pose, margin)
    print(f"pose          : {np.round(pose, 4).tolist()}")
    # print(f"min clearance : {res['min_clearance']:+.4f} m")
    print("RESULT        : " + ("INSIDE" if res else "OUTSIDE"))
    # sys.exit(0 if res["inside"] else 1)


def do_monitor(margin):
    corners = load_points()
    box = fit_box(corners[:, :3])
    rtde_r,rtde_c = connect()
    print("Live pose. Ctrl-C to stop.\n")
    try:
        while True:
            pose = rtde_r.getActualTCPPose()
            res = check_pose(box, pose, margin)
            state = "INSIDE " if res else "OUTSIDE"
            print(f"\r{state}  p={np.round(pose[:3], 3).tolist()}", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("teach", help="record the corners")
    sub.add_parser("points", help="print the saved points")
    sub.add_parser("tcp", help="check the pendant TCP against TCP_OFFSET")

    c = sub.add_parser("check", help="test one pose")
    c.add_argument("pose", type=float, nargs="+",
                   help="x y z [rx ry rz]. Use -- before negative values.")
    c.add_argument("--margin", type=float, default=0.0)

    m = sub.add_parser("monitor", help="live inside/outside readout")
    m.add_argument("--margin", type=float, default=0.0)

    args = ap.parse_args()

    if args.cmd == "teach":
        do_teach()
    elif args.cmd == "points":
        summarise(load_points()[:, :3])
    elif args.cmd == "tcp":
        do_tcp()
    elif args.cmd == "check":
        do_check(norm_pose(args.pose), args.margin)
    elif args.cmd == "monitor":
        do_monitor(args.margin)


if __name__ == "__main__":
    main()