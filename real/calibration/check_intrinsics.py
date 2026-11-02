#!/usr/bin/env python3
"""Is the RealSense factory K good enough? Re-derive it from your own images.

    python check_intrinsics.py --out data/right_calib

Runs cv2.calibrateCamera on the checkerboard views you already captured and
compares the result against rgb_intrinsics.npy. Note that the square size does
NOT affect the estimated intrinsics for a planar target -- it only scales the
extrinsic translations -- so a wrong print scale cannot corrupt this test.

If the refined K differs meaningfully from the factory one, that is a strong
candidate for a systematic hand-eye reprojection residual.

Pass --write to save the refined values as rgb_intrinsics.npy (the original is
backed up), then re-run the hand-eye solve with --solve-only.
"""

import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/handeye_calib")
    p.add_argument("--cols", type=int, default=7, help="board squares across")
    p.add_argument("--rows", type=int, default=5, help="board squares down")
    p.add_argument("--square-size", type=float, default=0.02275)
    p.add_argument("--fix-distortion", action="store_true",
                   help="assume zero distortion (images already undistorted)")
    p.add_argument("--write", action="store_true",
                   help="overwrite rgb_intrinsics.npy with the refined K")
    args = p.parse_args()

    pattern = (args.cols - 1, args.rows - 1)
    images = sorted(glob.glob(os.path.join(args.out, "img_*.png")))
    if not images:
        sys.exit(f"[!] no img_*.png in {args.out}")

    xs = args.square_size * np.arange(pattern[0], dtype=np.float32)
    ys = args.square_size * np.arange(pattern[1], dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    objp = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size, np.float32)], 1)

    objpoints, imgpoints, used = [], [], []
    shape = None
    for path in images:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        shape = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(
            gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
        objpoints.append(objp)
        imgpoints.append(corners)
        used.append(os.path.basename(path))

    print(f"[intrinsics] {len(used)}/{len(images)} views usable")
    if len(used) < 6:
        sys.exit("[!] need at least 6 views for a meaningful intrinsic fit")

    flags = 0
    if args.fix_distortion:
        flags |= (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                  cv2.CALIB_ZERO_TANGENT_DIST)
    rms, K_new, D_new, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, shape, None, None, flags=flags)

    # per-view reprojection, so you can see if any single view is dragging
    per_view = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K_new, D_new)
        e = np.linalg.norm(proj.reshape(-1, 2) - imgpoints[i].reshape(-1, 2), axis=1)
        per_view.append(e.mean())
    per_view = np.array(per_view)

    print(f"[intrinsics] calibrateCamera RMS: {rms:.4f} px")
    print(f"[intrinsics] per-view mean: {per_view.mean():.4f} px  "
          f"worst: {per_view.max():.4f} px ({used[int(per_view.argmax())]})")

    fpath = os.path.join(args.out, "rgb_intrinsics.npy")
    if not os.path.exists(fpath):
        sys.exit(f"[!] {fpath} missing")
    K_old = np.load(fpath)

    names = ["fx", "fy", "cx", "cy"]
    old = [K_old[0, 0], K_old[1, 1], K_old[0, 2], K_old[1, 2]]
    new = [K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]]
    print("\n            factory      refined      diff       diff %")
    worst = 0.0
    for n, o, v in zip(names, old, new):
        pct = 100.0 * (v - o) / o
        worst = max(worst, abs(pct))
        print(f"  {n:3s}  {o:11.3f}  {v:11.3f}  {v-o:+9.3f}  {pct:+8.3f}%")
    print(f"\n  distortion (refined): {np.round(D_new.ravel(), 5)}")

    if worst > 1.0:
        print(f"\n  >> factory intrinsics are off by up to {worst:.2f}%. This is "
              f"very likely your systematic hand-eye residual. Re-run with --write, "
              f"then re-solve with --solve-only.")
    elif rms > 1.0:
        print(f"\n  >> intrinsics agree, but the intrinsic fit itself is poor "
              f"({rms:.2f} px). That points at the BOARD -- wrong square size is "
              f"ruled out here, so suspect a non-flat or flexing board, or motion "
              f"blur during capture.")
    else:
        print(f"\n  >> intrinsics are fine and the board fits well ({rms:.3f} px). "
              f"The hand-eye residual is coming from the ROBOT side: flange poses, "
              f"a non-rigid board mount, or the arm moving between image and pose.")

    if args.write:
        shutil.copy(fpath, fpath + ".factory_backup")
        np.save(fpath, K_new)
        print(f"\n  wrote refined K to {fpath} "
              f"(original saved as rgb_intrinsics.npy.factory_backup)")
        if np.abs(D_new).max() > 1e-6 and not args.fix_distortion:
            print("  NOTE: nonzero distortion was fitted, which means your frames "
                  "were NOT undistorted at capture. The hand-eye solver assumes "
                  "they are. Re-capture with undistortion enabled.")


if __name__ == "__main__":
    main()