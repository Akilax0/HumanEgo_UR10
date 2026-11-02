#!/usr/bin/env python3
"""How wrong is the calibration, in millimetres?

    python accuracy_mm.py --out data/right_calib --square-size 0.02275

For each view the board pose in the base frame is computed twice:
    robot path   T_bf @ T_ft
    camera path  T_bc @ T_ct
A perfect calibration makes them identical. The gap is the physical error
you will actually feel when converting a camera detection into a robot
target, which is far more useful than a reprojection number in pixels.

A consistent OFFSET across views means a frame error you could correct.
Random SCATTER means you have hit the noise floor -- usually the arm's
absolute kinematic accuracy, which no amount of board work will improve.
"""

import argparse
import os

import numpy as np

import calibration_data_capture_zed as H   # load_dataset, make_T (RealSense script removed 2026-10-29)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/handeye_calib")
    p.add_argument("--cols", type=int, default=7)
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--square-size", type=float, default=0.02275)
    args = p.parse_args()

    T_bc = np.load(os.path.join(args.out, "camera_to_base.npy"))
    T_ft = np.load(os.path.join(args.out, "target_to_flange.npy"))
    K, objp, views = H.load_dataset(args)

    pos_err, rot_err, rows = [], [], []
    for v in views:
        T_ct = H.make_T(v["R_ct"], v["t_ct"])
        via_robot = v["T_bf"] @ T_ft          # board in base, robot path
        via_camera = T_bc @ T_ct              # board in base, camera path
        dp = np.linalg.norm(via_robot[:3, 3] - via_camera[:3, 3]) * 1000.0
        dR = via_robot[:3, :3].T @ via_camera[:3, :3]
        da = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        pos_err.append(dp)
        rot_err.append(da)
        rows.append((v["name"], dp, da, via_robot[:3, 3] - via_camera[:3, 3]))

    pos_err = np.array(pos_err)
    rot_err = np.array(rot_err)
    offsets = np.array([r[3] for r in rows]) * 1000.0

    print(f"\n{'view':14s} {'pos err':>9s} {'rot err':>9s}   "
          f"{'dx':>7s} {'dy':>7s} {'dz':>7s}   (mm)")
    for name, dp, da, _ in rows:
        i = [r[0] for r in rows].index(name)
        o = offsets[i]
        print(f"{name:14s} {dp:8.2f}  {da:8.3f}   "
              f"{o[0]:7.2f} {o[1]:7.2f} {o[2]:7.2f}")

    print(f"\nposition error   mean {pos_err.mean():6.2f} mm   "
          f"median {np.median(pos_err):6.2f} mm   max {pos_err.max():6.2f} mm")
    print(f"rotation error   mean {rot_err.mean():6.3f} deg  "
          f"max {rot_err.max():6.3f} deg")

    bias = offsets.mean(axis=0)
    scatter = offsets.std(axis=0)
    print(f"\nsystematic bias  {np.round(bias, 2)} mm  "
          f"(|bias| = {np.linalg.norm(bias):.2f} mm)")
    print(f"random scatter   {np.round(scatter, 2)} mm  "
          f"(rms = {np.linalg.norm(scatter):.2f} mm)")

    if np.linalg.norm(bias) > 0.6 * pos_err.mean():
        print("\n  >> dominated by a CONSTANT OFFSET. Something in the frame chain "
              "is systematically shifted -- suspect the board mount, or a TCP that "
              "was not actually zeroed. This is correctable.")
    else:
        print("\n  >> dominated by RANDOM SCATTER, no consistent offset. This is the "
              "noise floor of the setup. On an uncalibrated UR10e that is usually "
              "the arm's absolute kinematic accuracy (millimetres), not anything "
              "about your board or camera.")

    print(f"\nFor context: a Robotiq 3-finger gripper closing on a rigid object "
          f"typically tolerates several mm of positioning error.\n"
          f"Your calibration is good to about {pos_err.mean():.1f} mm.")


if __name__ == "__main__":
    main()