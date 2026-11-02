# -*- coding: utf-8 -*-
"""
Live view of the ZED camera, to check it's connected and pointed where you want.

Two backends, picked automatically:
  - pyzed (inside the docker container, ZED SDK): left image + colorized depth.
  - uvc   (on the host, plain OpenCV, no SDK/GPU needed): the ZED is a standard
          UVC device that streams left|right side by side.

Keys: q/Esc quit, s save snapshot.

Usage:
    python view_zed.py                         # auto backend, live window
    python view_zed.py --backend uvc --device 4
    python view_zed.py --snapshot              # headless: grab one frame, save, exit
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

# UVC side-by-side sizes the ZED firmware supports, keyed like the SDK resolutions.
UVC_SIZES = {"HD2K": (4416, 1242), "HD1080": (3840, 1080), "HD720": (2560, 720), "VGA": (1344, 376)}


def find_zed_uvc_device():
    """Return the /dev/videoN index whose V4L2 name contains 'ZED' (capture node), or None."""
    for path in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            name = open(os.path.join(path, "name")).read().strip()
            index = int(open(os.path.join(path, "index")).read().strip())
        except OSError:
            continue
        if "ZED" in name.upper() and index == 0:
            return int(path.rsplit("video", 1)[1])
    return None


class UVCSource:
    def __init__(self, device, resolution, fps):
        if device is None:
            device = find_zed_uvc_device()
            if device is None:
                sys.exit("!! No ZED found under /sys/class/video4linux. Check the USB3 cable/port and `lsusb`.")
        print(f"[uvc] opening /dev/video{device}")
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            sys.exit(f"!! Could not open /dev/video{device} (in use by another process? permissions?)")
        w, h = UVC_SIZES[resolution]
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        got = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"[uvc] stream {got[0]}x{got[1]} (requested {w}x{h}) @ {self.cap.get(cv2.CAP_PROP_FPS):.0f} fps")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        half = frame.shape[1] // 2
        left, right = frame[:, :half], frame[:, half:]
        return np.hstack([left, right]), "left | right"

    def close(self):
        self.cap.release()


class SDKSource:
    def __init__(self, sl, serial, resolution, fps):
        self.sl = sl
        devices = sl.Camera.get_device_list()
        if not devices:
            sys.exit("!! ZED SDK sees no camera. Check USB3 and that the container has /dev + privileged.")
        for d in devices:
            print(f"[sdk] found model={d.camera_model} serial={d.serial_number} state={d.camera_state}")
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, resolution)
        init.camera_fps = fps
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER
        if serial:
            init.set_from_serial_number(int(serial))
        self.zed = sl.Camera()
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            sys.exit(f"!! zed.open() failed: {status}")
        info = self.zed.get_camera_information()
        res = info.camera_configuration.resolution
        print(f"[sdk] opened {info.camera_model} serial={info.serial_number} {res.width}x{res.height}")
        self.runtime = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.depth = sl.Mat()

    def read(self):
        sl = self.sl
        if self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return None
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
        left = cv2.cvtColor(self.image.get_data(), cv2.COLOR_BGRA2BGR)
        depth = np.nan_to_num(np.squeeze(np.asarray(self.depth.get_data())), nan=0.0, posinf=0.0, neginf=0.0)
        valid = depth[(depth > 0.1) & (depth < 5.0)]
        depth_vis = cv2.applyColorMap(
            (np.clip(depth / 3.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        depth_vis[depth <= 0] = 0
        label = f"left | depth (0-3m) median={np.median(valid):.2f}m" if valid.size else "left | depth: NO VALID DEPTH"
        return np.hstack([left, depth_vis]), label

    def close(self):
        self.zed.close()


def make_source(args):
    if args.backend in ("auto", "sdk"):
        try:
            import pyzed.sl as sl
            return SDKSource(sl, args.serial, args.resolution, args.fps)
        except ImportError:
            if args.backend == "sdk":
                sys.exit("!! pyzed not installed here — run inside the zed_docker container, or use --backend uvc.")
            print("[auto] pyzed not available, falling back to UVC (no depth)")
    return UVCSource(args.device, args.resolution, args.fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["auto", "sdk", "uvc"], default="auto")
    parser.add_argument("--device", type=int, default=None, help="UVC /dev/videoN index; default = autodetect")
    parser.add_argument("--serial", type=str, default="", help="ZED serial (sdk backend); empty = first found")
    parser.add_argument("--resolution", choices=list(UVC_SIZES), default="HD720")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--snapshot", action="store_true", help="grab one frame, save it, exit (no window)")
    parser.add_argument("--out", type=str, default="./zed_view.png")
    parser.add_argument("--max_width", type=int, default=1600, help="downscale the display window to this width")
    args = parser.parse_args()

    src = make_source(args)

    if args.snapshot:
        result = None
        for _ in range(30):  # first frames after open can be empty / still auto-exposing
            result = src.read()
        src.close()
        if result is None:
            sys.exit("!! Camera opened but no frames arrived.")
        frame, label = result
        cv2.imwrite(args.out, frame)
        print(f"OK — {frame.shape[1]}x{frame.shape[0]} ({label}), saved to {args.out}")
        return

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("!! No display found — use --snapshot (or forward X11 into the container).")

    win = "ZED view (q: quit, s: snapshot)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t_prev, fps_est, misses = time.time(), 0.0, 0
    try:
        while True:
            result = src.read()
            if result is None:
                misses += 1
                if misses > 60:
                    print("!! 60 consecutive failed grabs, giving up.")
                    break
                continue
            misses = 0
            frame, label = result
            now = time.time()
            fps_est = 0.9 * fps_est + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            disp = frame
            if disp.shape[1] > args.max_width:
                scale = args.max_width / disp.shape[1]
                disp = cv2.resize(disp, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.putText(disp, f"{label}  {fps_est:.1f} fps", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(win, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                cv2.imwrite(args.out, frame)
                print(f"saved {args.out}")
    finally:
        src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
