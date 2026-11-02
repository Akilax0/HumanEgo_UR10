# -*- coding: utf-8 -*-
"""
Minimal ZED smoke test — run this FIRST, before trying CamZED.py.

No repo imports, no config files: just "can the SDK inside this container see
the camera and pull a frame". Saves a snapshot instead of using cv2.imshow so
it also works over a plain SSH session with no X11/Wayland display forwarded.

Usage (inside the container):
    python3 test_zed_connection.py
    python3 test_zed_connection.py --serial 12345678   # pick a specific unit
"""
import argparse
import sys

import cv2
import numpy as np
import pyzed.sl as sl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="", help="ZED serial number; empty = first found")
    parser.add_argument("--out", type=str, default="./zed_test_snapshot.png")
    args = parser.parse_args()

    print("[1/4] Listing ZED devices visible to the SDK...")
    devices = sl.Camera.get_device_list()
    if not devices:
        print("    !! No ZED camera detected. Check: USB3 cable/port, `lsusb` inside the "
              "container, and that docker-compose.yml's `/dev:/dev` + `privileged: true` "
              "are in effect.")
        sys.exit(1)
    for d in devices:
        print(f"    - model={d.camera_model} serial={d.serial_number} state={d.camera_state}")

    print("[2/4] Opening camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    if args.serial.strip():
        init_params.set_from_serial_number(int(args.serial))

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"    !! zed.open() failed: {status}")
        print("    If this mentions CUDA or a driver mismatch, your host NVIDIA driver "
              "and this image's CUDA_VERSION build arg likely don't line up — see README.md.")
        sys.exit(1)

    cam_info = zed.get_camera_information()
    res = cam_info.camera_configuration.resolution
    calib = cam_info.camera_configuration.calibration_parameters.left_cam
    print(f"    OK — {cam_info.camera_model}, serial={cam_info.serial_number}, "
          f"resolution={res.width}x{res.height}, fx={calib.fx:.1f}, fy={calib.fy:.1f}")

    print("[3/4] Grabbing a frame (RGB + depth)...")
    runtime_params = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    grabbed = False
    for _ in range(30):  # a few retries — first grab(s) after open() can be empty
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            grabbed = True
            break
    if not grabbed:
        print("    !! Never got a successful grab() — camera opened but isn't streaming.")
        zed.close()
        sys.exit(1)

    rgba = image_mat.get_data()
    rgb = cv2.cvtColor(rgba, cv2.COLOR_BGRA2BGR)
    depth_m = np.nan_to_num(np.squeeze(np.asarray(depth_mat.get_data())), nan=0.0, posinf=0.0, neginf=0.0)
    valid = depth_m[(depth_m > 0.1) & (depth_m < 5.0)]

    print(f"    OK — rgb shape={rgb.shape}, depth shape={depth_m.shape}, "
          f"valid depth px={valid.size}/{depth_m.size} "
          f"({'median=%.3fm' % np.median(valid) if valid.size else 'no valid depth!'})")

    print(f"[4/4] Saving snapshot to {args.out} ...")
    cv2.imwrite(args.out, rgb)
    zed.close()
    print("Done. If depth looked sane above, CamZED.py should work too — try test_camzed_driver.py next.")


if __name__ == "__main__":
    main()
