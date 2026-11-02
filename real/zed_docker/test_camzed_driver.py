# -*- coding: utf-8 -*-
"""
End-to-end check of the actual driver used by run_inference.py — run this
AFTER test_zed_connection.py passes.

Exercises inference_ur10/CamZED.py + cfg/inference/example_ur10/CamZED.yaml
exactly as ZedCamera (run_inference.py) does, including the crop/resize and
lift3d_for_multi_frames path that ReferencePerception relies on.

Usage (inside the container, from /workspace/HumanEgo — the Dockerfile sets
PYTHONPATH there already):
    python3 /workspace/real/zed_docker/test_camzed_driver.py
    python3 /workspace/real/zed_docker/test_camzed_driver.py --cfg cfg/inference/example_ur10/CamZED.yaml
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/HumanEgo")
sys.path.insert(0, "/workspace/HumanEgo/inference_ur10")

from CamZED import CamZED  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="/workspace/HumanEgo/cfg/inference/example_ur10/CamZED.yaml")
    parser.add_argument("--n_frames", type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.cfg):
        print(f"Config not found: {args.cfg}")
        sys.exit(1)

    print(f"Opening CamZED with {args.cfg} ...")
    cam = CamZED(args.cfg)

    print(f"k_rgb intrinsics:\n{cam.k_rgb}")

    print(f"Grabbing {args.n_frames} processed (cropped+resized) RGB-D frames...")
    for i in range(args.n_frames):
        d = cam.get_rgbd()
        if d is None:
            print(f"  frame {i}: grab failed")
            continue
        valid = d.depth_m[(d.depth_m > 0.1) & (d.depth_m < 5.0)]
        med = float(np.median(valid)) if valid.size else float("nan")
        print(f"  frame {i}: rgb={d.rgb.shape} depth={d.depth_m.shape} "
              f"valid_px={valid.size}/{d.depth_m.size} median_depth={med:.3f}m")

    # Exercise the exact call ReferencePerception.estimate_objects makes:
    # a single (u, v) pixel, lifted robustly across several depth frames.
    h, w = d.depth_m.shape[:2]
    uv = (w // 2, h // 2)
    depths = [cam.get_rgbd().depth_m for _ in range(3)]
    xyz, q = cam.lift3d_for_multi_frames(uv, depths, mask=None, k=cam.k_rgb)
    print(f"\nlift3d_for_multi_frames({uv}) -> xyz={xyz}, quality={q}")

    cam.close()
    print("\nDone. If xyz above is a sane (x, y, z) in meters, ReferencePerception "
          "should be able to build object poses through the ZED backend.")


if __name__ == "__main__":
    main()
