#!/usr/bin/env python3
"""Plot the calibration result without OpenGL (works fine under Wayland).

    python plot_calibration.py --out data/right_calib [--save check.png]

Draws the robot base frame, the solved camera frame, every recorded flange
pose, and the board position implied by T_ft at each pose. If the camera
frame does not land where your tripod physically is, the calibration is
wrong regardless of what the reprojection error says.
"""

import argparse
import json
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def draw_frame(ax, T, size=0.15, label=None, lw=2.5, alpha=1.0):
    o = T[:3, 3]
    for k, c in enumerate("rgb"):                 # x=red, y=green, z=blue
        d = T[:3, k] * size
        ax.plot([o[0], o[0] + d[0]], [o[1], o[1] + d[1]],
                [o[2], o[2] + d[2]], color=c, lw=lw, alpha=alpha)
    if label:
        ax.text(o[0], o[1], o[2], f"  {label}", fontsize=9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/right_calib")
    p.add_argument("--save", default=None, help="write a PNG instead of showing")
    args = p.parse_args()

    T_bc = np.load(os.path.join(args.out, "camera_to_base.npy"))
    T_ft = np.load(os.path.join(args.out, "target_to_flange.npy"))
    T_bf = np.load(os.path.join(args.out, "flange_poses.npy"))

    fig = plt.figure(figsize=(11, 5))

    ax = fig.add_subplot(121, projection="3d")
    draw_frame(ax, np.eye(4), 0.25, "base", lw=3)
    draw_frame(ax, T_bc, 0.15, "camera", lw=3)
    for T in T_bf:
        draw_frame(ax, T, 0.05, None, lw=1.0, alpha=0.45)
        draw_frame(ax, T @ T_ft, 0.03, None, lw=1.0, alpha=0.3)

    pts = np.vstack([np.zeros(3), T_bc[:3, 3], T_bf[:, :3, 3]])
    c, r = pts.mean(0), max(np.ptp(pts, axis=0).max(), 0.4) / 2 * 1.3
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("base, camera, and flange poses")

    # optical axis: where is the camera actually pointing?
    ax2 = fig.add_subplot(122, projection="3d")
    draw_frame(ax2, np.eye(4), 0.25, "base", lw=3)
    draw_frame(ax2, T_bc, 0.15, "camera", lw=3)
    o, z = T_bc[:3, 3], T_bc[:3, 2]
    ax2.plot(*zip(o, o + z * 1.5), "k--", lw=1.5, label="optical axis")
    ax2.scatter(*T_bf[:, :3, 3].T, s=14, c="tab:orange", label="flange positions")
    ax2.legend(fontsize=8)
    ax2.set_xlim(c[0] - r, c[0] + r)
    ax2.set_ylim(c[1] - r, c[1] + r)
    ax2.set_zlim(c[2] - r, c[2] + r)
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.set_zlabel("Z (m)")
    ax2.set_title("does the camera point at the workspace?")

    print(f"camera position (m):     {np.round(T_bc[:3, 3], 4)}")
    print(f"camera optical axis:     {np.round(T_bc[:3, 2], 4)}")
    print(f"board origin in flange:  {np.round(T_ft[:3, 3], 4)} m")
    print(f"board normal in flange:  {np.round(T_ft[:3, 2], 4)}")
    d = np.linalg.norm(T_bf[:, :3, 3] - T_bc[:3, 3], axis=1)
    print(f"camera-to-flange range:  {d.min():.3f} - {d.max():.3f} m")

    rep = os.path.join(args.out, "calibration_report.json")
    if os.path.exists(rep):
        r_ = json.load(open(rep))
        print(f"views {r_['n_views']}  mean {r_['mean_reproj_px']:.3f} px  "
              f"rotation spread {r_['max_pairwise_rotation_deg']:.1f} deg")

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=130)
        print(f"wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()