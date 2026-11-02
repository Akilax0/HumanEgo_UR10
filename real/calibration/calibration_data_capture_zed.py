#!/usr/bin/env python3
"""End-to-end eye-to-hand calibration for a UR arm + fixed ZED camera.

ZED counterpart of calibration_data_capture_realsense_updated.py — same
geometry, same AXZB solve, same output file layout (so check_consistency.py /
accuracy.py / plot_calib.py work unmodified against either dataset). Only the
camera backend differs: pyzed instead of pyrealsense2.

Geometry: the checkerboard is mounted rigidly on the ROBOT FLANGE (here: on
the Robotiq gripper, a fixed rigid extension of the flange), the camera is
fixed in the world. Solves  T_bf @ T_ft == T_bc @ T_ct  for the two unknowns
T_ft (target in flange) and T_bc (camera in base).

Run this INSIDE the zed_docker container (needs both pyzed and ur_rtde, plus
the ZED plugged into a USB3 port and $DISPLAY forwarded for the capture GUI):
    docker compose run --rm zed bash
    python3 real/calibration/calibration_data_capture_zed.py \\
        --out real/calibration/data/zed_handeye_calib --square-size 0.02275
        
        python3 workspace/real/calibration/calibration_data_capture_zed.py --out real/calibration/d
ata/zed_handeye_calib --square-size 0.02275

Capture keys (focus the image window):
    f  enable freedrive        e  lock the arm
    c  record this pose        u  undo last          q  finish early

--square-size / --cols / --rows default to the board used for the RealSense
run (7x5 squares @ 22.75mm) on the assumption it's the same physical board
now remounted on the gripper -- MEASURE YOURS and override if it isn't; a
wrong square size is a silent uniform scale error in the output translation.

Output T_bc = camera_to_base.npy is the camera's pose in the robot base
frame. HumanEgo's RobotArmUR10 (inference_ur10/interfaces.py) wants the
inverse, T_base_in_cam = inv(T_bc) -- see the note printed at the end.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------

class ZEDCamera:
    """Color-only ZED wrapper that hands back rectified BGR frames.

    Mirrors RealSenseCamera's surface (get_rgb_image / get_rgb_intrinsics)
    so the rest of this script -- capture loop, solve -- is unchanged.

    Unlike RealSense, sl.VIEW.LEFT is already rectified/undistorted by the
    SDK's stereo calibration pipeline (independent of depth_mode), so there's
    no separate undistort step: solvePnP gets distCoeffs=None, same as the
    RealSense wrapper after ITS software undistort.
    """

    def __init__(self, resolution="HD1080", fps=30, serial=None,
                 depth_mode="PERFORMANCE"):
        import pyzed.sl as sl
        self._sl = sl

        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = getattr(sl.RESOLUTION, resolution)
        init_params.camera_fps = fps
        # Depth is never used here; PERFORMANCE avoids the NEURAL model
        # download/optimization pass. Rectification of VIEW.LEFT does not
        # depend on this setting.
        init_params.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
        init_params.coordinate_units = sl.UNIT.METER
        if serial:
            init_params.set_from_serial_number(int(serial))

        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            sys.exit(f"[!] zed.open() failed: {status}")

        self.runtime_params = sl.RuntimeParameters()
        self._image_mat = sl.Mat()

        cam_info = self.zed.get_camera_information()
        res = cam_info.camera_configuration.resolution
        self.width, self.height = int(res.width), int(res.height)

        calib = cam_info.camera_configuration.calibration_parameters.left_cam
        self.K = np.array([[calib.fx, 0.0, calib.cx],
                            [0.0, calib.fy, calib.cy],
                            [0.0, 0.0, 1.0]], dtype=np.float64)

        # discard a few frames so auto exposure/white balance converge
        for _ in range(20):
            self.get_rgb_image()

        print(f"[cam] {cam_info.camera_model} serial={cam_info.serial_number} "
              f"{self.width}x{self.height}  fx={calib.fx:.1f} fy={calib.fy:.1f}")

    def get_rgb_image(self):
        sl = self._sl
        if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("no frame (grab() failed)")
        self.zed.retrieve_image(self._image_mat, sl.VIEW.LEFT)
        rgba = self._image_mat.get_data()
        return cv2.cvtColor(rgba, cv2.COLOR_BGRA2BGR)

    def get_rgb_intrinsics(self):
        return self.K

    def stop(self):
        try:
            self.zed.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# geometry helpers (camera-agnostic -- identical to the RealSense script)
# --------------------------------------------------------------------------

def object_points(cols, rows, square):
    """Inner-corner coordinates in the board frame, z=0, metres."""
    xs = square * np.arange(cols - 1, dtype=np.float64)
    ys = square * np.arange(rows - 1, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def find_corners(gray, cols, rows, refine=True):
    pattern = (cols - 1, rows - 1)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return False, None
    if refine:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
    return True, corners.reshape(-1, 2).astype(np.float64)


def solve_pnp(objp, imgp, K):
    for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, None, flags=flag)
        except cv2.error:
            continue
        if ok:
            return Rot.from_rotvec(rvec.ravel()).as_matrix(), tvec.reshape(3, 1)
    return None, None


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------

def capture(args):
    import rtde_control
    import rtde_receive

    os.makedirs(args.out, exist_ok=True)
    stale = glob.glob(os.path.join(args.out, "img_*.png"))
    if stale:
        if not args.force:
            sys.exit(f"[!] {len(stale)} images already in {args.out}. "
                     f"Delete them or pass --force to overwrite.")
        for p in stale:
            os.remove(p)

    rtde_r = rtde_receive.RTDEReceiveInterface(args.ip)
    rtde_c = None
    if not args.no_control:
        rtde_c = rtde_control.RTDEControlInterface(args.ip)
        rtde_c.setTcp([0.0] * 6)          # report the FLANGE, not a stale TCP
        print("[ur] TCP zeroed -> getActualTCPPose() returns the flange")
    else:
        print("[ur] --no-control: set the TCP to all zeros on the pendant yourself")

    cam = ZEDCamera(resolution=args.resolution, fps=args.fps, serial=args.serial)
    np.save(os.path.join(args.out, "rgb_intrinsics.npy"), cam.get_rgb_intrinsics())

    poses, joints, count = [], [], 0
    teaching = False
    win = "calibration capture  |  f=freedrive  e=lock  c=capture  u=undo  q=done"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    try:
        while count < args.samples:
            frame = cam.get_rgb_image()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ok, corners = find_corners(gray, args.cols, args.rows, refine=False)

            view = frame.copy()
            if ok:
                cv2.drawChessboardCorners(
                    view, (args.cols - 1, args.rows - 1),
                    corners.reshape(-1, 1, 2).astype(np.float32), True)
            colour = (0, 200, 0) if ok else (0, 0, 220)
            cv2.putText(view, f"{count}/{args.samples}   "
                              f"{'BOARD OK' if ok else 'NO BOARD'}"
                              f"{'   FREEDRIVE' if teaching else ''}",
                        (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, 3)
            cv2.imshow(win, view)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('f') and rtde_c and not teaching:
                rtde_c.teachMode(); teaching = True
            elif key == ord('e') and rtde_c and teaching:
                rtde_c.endTeachMode(); teaching = False
            elif key == ord('u') and count:
                count -= 1
                poses.pop(); joints.pop()
                os.remove(os.path.join(args.out, f"img_{count:03d}.png"))
                print(f"[capture] undo -> {count}")
            elif key == ord('q'):
                break
            elif key == ord('c'):
                if teaching:
                    rtde_c.endTeachMode(); teaching = False
                    cv2.waitKey(400)          # let the brakes settle
                # re-grab AFTER locking so image and pose agree
                frame = cam.get_rgb_image()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok, _ = find_corners(gray, args.cols, args.rows, refine=False)
                if not ok:
                    print("[capture] board lost on the locked frame, skipped")
                    continue
                pose = rtde_r.getActualTCPPose()
                cv2.imwrite(os.path.join(args.out, f"img_{count:03d}.png"), frame)
                poses.append(make_T(Rot.from_rotvec(pose[3:]).as_matrix(), pose[:3]))
                joints.append(rtde_r.getActualQ())
                count += 1
                print(f"[capture] {count}/{args.samples}")
    finally:
        if rtde_c:
            if teaching:
                rtde_c.endTeachMode()
            rtde_c.stopScript()
        cam.stop()
        cv2.destroyAllWindows()

    if count < 3:
        sys.exit(f"[!] only {count} samples, need at least 3 (10+ recommended)")
    np.save(os.path.join(args.out, "flange_poses.npy"), np.array(poses))
    np.save(os.path.join(args.out, "joint_angles.npy"), np.array(joints))
    print(f"[capture] saved {count} samples to {args.out}")
    return count


# --------------------------------------------------------------------------
# solve (identical to the RealSense script -- camera-agnostic)
# --------------------------------------------------------------------------

def load_dataset(args):
    for f in ("rgb_intrinsics.npy", "flange_poses.npy"):
        if not os.path.exists(os.path.join(args.out, f)):
            sys.exit(f"[!] {args.out}/{f} not found -- run a capture first "
                     f"(drop --solve-only)")
    K = np.load(os.path.join(args.out, "rgb_intrinsics.npy"))
    T_bf_all = np.load(os.path.join(args.out, "flange_poses.npy"))
    images = sorted(glob.glob(os.path.join(args.out, "img_*.png")))
    if len(images) != len(T_bf_all):
        sys.exit(f"[!] {len(images)} images but {len(T_bf_all)} poses. "
                 f"The directory has stale data; recapture.")

    objp = object_points(args.cols, args.rows, args.square_size)
    views = []
    for idx, (path, T_bf) in enumerate(zip(images, T_bf_all)):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ok, imgp = find_corners(gray, args.cols, args.rows)
        if not ok:
            print(f"[load] {os.path.basename(path)}: no board, dropped")
            continue
        R_ct, t_ct = solve_pnp(objp, imgp, K)
        if R_ct is None:
            print(f"[load] {os.path.basename(path)}: solvePnP failed, dropped")
            continue
        views.append(dict(idx=idx, name=os.path.basename(path), imgp=imgp,
                          T_bf=T_bf, R_ct=R_ct, t_ct=t_ct))
    print(f"[load] {len(views)}/{len(images)} views usable")
    return K, objp, views


def rotation_spread(views):
    """Largest pairwise flange-rotation angle, in degrees."""
    Rs = [v["T_bf"][:3, :3] for v in views]
    worst = 0.0
    for i in range(len(Rs)):
        for j in range(i + 1, len(Rs)):
            ang = np.linalg.norm(Rot.from_matrix(Rs[i].T @ Rs[j]).as_rotvec())
            worst = max(worst, np.degrees(ang))
    return worst


def _project_so3(M):
    """Nearest rotation matrix to M in the Frobenius sense."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R = U @ Vt
    return R


def solve_axzb_numpy(TA, TB):
    """Closed-form solve of A_i @ X == Z @ B_i for X and Z (Shah's method).

    Pure NumPy, so it works on OpenCV 5.x where the hand-eye Python
    bindings are missing.
    """
    n = len(TA)
    M = np.zeros((9 * n, 18))
    I3 = np.eye(3)
    for i, (A, B) in enumerate(zip(TA, TB)):
        M[9 * i:9 * i + 9, :9] = np.kron(I3, A[:3, :3])
        M[9 * i:9 * i + 9, 9:] = -np.kron(B[:3, :3].T, I3)

    v = np.linalg.svd(M)[2][-1]
    RX = v[:9].reshape(3, 3, order="F")
    RZ = v[9:].reshape(3, 3, order="F")
    if np.linalg.det(RX) < 0:          # flip both together: one null vector
        RX, RZ = -RX, -RZ
    RX, RZ = _project_so3(RX), _project_so3(RZ)

    C = np.zeros((3 * n, 6))
    d = np.zeros(3 * n)
    for i, (A, B) in enumerate(zip(TA, TB)):
        C[3 * i:3 * i + 3, :3] = A[:3, :3]
        C[3 * i:3 * i + 3, 3:] = -I3
        d[3 * i:3 * i + 3] = RZ @ B[:3, 3] - A[:3, 3]
    sol = np.linalg.lstsq(C, d, rcond=None)[0]
    cond = float(np.linalg.cond(C))
    if cond > 1e3:
        print(f"[warn] translation system is ill-conditioned (cond={cond:.1f}). "
              f"The flange rotations are too similar -- translations are barely "
              f"observable and the result may be meaningless.")
    return make_T(RX, sol[:3]), make_T(RZ, sol[3:])


def solve(views):
    R_bf = np.array([v["T_bf"][:3, :3] for v in views])
    t_bf = np.array([v["T_bf"][:3, 3].reshape(3, 1) for v in views])
    R_ct = np.array([v["R_ct"] for v in views])
    t_ct = np.array([v["t_ct"] for v in views])

    if hasattr(cv2, "calibrateRobotWorldHandEye"):
        R_ft, t_ft, R_bc, t_bc = cv2.calibrateRobotWorldHandEye(
            R_bf, t_bf, R_ct, t_ct)
        return make_T(R_ft, t_ft), make_T(R_bc, t_bc)

    if hasattr(cv2, "calibrateHandEye"):
        print(f"[solve] OpenCV {cv2.__version__}: calibrateHandEye fallback")
        T_fb = [np.linalg.inv(v["T_bf"]) for v in views]
        R_bc, t_bc = cv2.calibrateHandEye(
            np.array([T[:3, :3] for T in T_fb]),
            np.array([T[:3, 3].reshape(3, 1) for T in T_fb]),
            R_ct, t_ct)
        T_bc = make_T(R_bc, t_bc)
        cands = [T_fb[i] @ T_bc @ make_T(views[i]["R_ct"], views[i]["t_ct"])
                 for i in range(len(views))]
        U, _, Vt = np.linalg.svd(np.sum([T[:3, :3] for T in cands], axis=0))
        T_ft = make_T(_project_so3(U @ Vt),
                      np.mean([T[:3, 3] for T in cands], axis=0))
        return T_ft, T_bc

    print(f"[solve] OpenCV {cv2.__version__} has no hand-eye bindings: "
          f"using the built-in NumPy solver")
    TA = [v["T_bf"] for v in views]
    TB = [make_T(v["R_ct"], v["t_ct"]) for v in views]
    return solve_axzb_numpy(TA, TB)


def per_view_error(T_ft, T_bc, K, objp, views):
    T_cb = np.linalg.inv(T_bc)
    objh = np.hstack([objp, np.ones((objp.shape[0], 1))]).T      # 4xN
    errs = []
    for v in views:
        pc = T_cb @ v["T_bf"] @ T_ft @ objh                      # 4xN
        ph = K @ pc[:3, :]
        proj = (ph[:2] / ph[2:3]).T
        errs.append(float(np.mean(np.linalg.norm(proj - v["imgp"], axis=1))))
    return np.array(errs)


def run_solve(args):
    K, objp, views = load_dataset(args)
    if len(views) < 3:
        sys.exit("[!] need at least 3 usable views")

    spread = rotation_spread(views)
    print(f"[check] max pairwise flange rotation: {spread:.1f} deg")
    if spread < 30.0:
        print("[warn] wrist orientations are too similar; the solution will be "
              "poorly conditioned. Recapture with more tilt/roll variety.")

    T_ft, T_bc = solve(views)
    for name, T, lim in (("camera_to_base", T_bc, 5.0),
                         ("target_to_flange", T_ft, 1.0)):
        n = float(np.linalg.norm(T[:3, 3]))
        if n > lim:
            sys.exit(
                f"[!] DEGENERATE SOLVE: {name} translation is {n:.1f} m "
                f"(expected < {lim} m).\n"
                f"    The solution has collapsed along an unconstrained "
                f"direction. Almost always this means the flange rotations are "
                f"too similar -- rotation spread was {spread:.1f} deg, and you "
                f"want 60+ deg about several distinct axes.\n"
                f"    Recapture with the wrist rotated hard about all three "
                f"axes; do not trust any numbers from this dataset.")
    errs = per_view_error(T_ft, T_bc, K, objp, views)
    print(f"[solve] mean {errs.mean():.3f} px   max {errs.max():.3f} px "
          f"({len(views)} views)")

    if args.reject_px > 0:
        keep = [v for v, e in zip(views, errs) if e <= args.reject_px]
        dropped = [(v["name"], e) for v, e in zip(views, errs) if e > args.reject_px]
        if dropped and len(keep) >= 3:
            for name, e in dropped:
                print(f"[reject] {name}: {e:.2f} px")
            views = keep
            T_ft, T_bc = solve(views)
            errs = per_view_error(T_ft, T_bc, K, objp, views)
            print(f"[solve] after rejection: mean {errs.mean():.3f} px   "
                  f"max {errs.max():.3f} px ({len(views)} views)")
        elif dropped:
            print(f"[reject] {len(dropped)} views over threshold but too few "
                  f"would remain; keeping all")

    np.save(os.path.join(args.out, "camera_to_base.npy"), T_bc)
    np.save(os.path.join(args.out, "target_to_flange.npy"), T_ft)
    report = dict(
        square_size_m=args.square_size,
        board_squares=[args.cols, args.rows],
        n_views=len(views),
        mean_reproj_px=float(errs.mean()),
        max_reproj_px=float(errs.max()),
        max_pairwise_rotation_deg=float(spread),
        camera_to_base=T_bc.tolist(),
        target_to_flange=T_ft.tolist(),
        camera_position_in_base_m=T_bc[:3, 3].tolist(),
    )
    with open(os.path.join(args.out, "calibration_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\ncamera_to_base (T_bc):\n", np.array_str(T_bc, precision=5, suppress_small=True))
    print("\ntarget_to_flange (T_ft):\n", np.array_str(T_ft, precision=5, suppress_small=True))
    print(f"\ncamera sits at {np.round(T_bc[:3, 3], 4)} m in the base frame "
          f"-- sanity-check this against the tripod/mount.")

    if errs.mean() > 2.0:
        print("[warn] mean error > 2 px. Check square_size, board rigidity, and "
              "the 180-deg corner-ordering flip (6x4 corners is symmetric).")

    T_base_in_cam = np.linalg.inv(T_bc)
    print("\nHumanEgo's RobotArmUR10.T_base_in_cam wants inv(camera_to_base):\n",
          np.array_str(T_base_in_cam, precision=5, suppress_small=True))
    np.save(os.path.join(args.out, "T_base_in_cam.npy"), T_base_in_cam)
    print(f"(also saved to {args.out}/T_base_in_cam.npy for direct use)")

    if args.viz:
        try:
            import open3d as o3d
            base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            cam.transform(T_bc)
            o3d.visualization.draw_geometries([base, cam])
        except ImportError:
            print("[viz] open3d not installed, skipping")
    return T_ft, T_bc


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Eye-to-hand calibration: board on flange/gripper, ZED fixed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", default="real/calibration/data/zed_handeye_calib",
                   help="dataset directory")
    p.add_argument("--ip", default="10.0.0.219", help="UR controller IP")
    p.add_argument("--cols", type=int, default=7, help="board squares across")
    p.add_argument("--rows", type=int, default=5, help="board squares down")
    p.add_argument("--square-size", type=float, default=0.02275,
                   help="MEASURED square size in metres")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--resolution", default="HD1080",
                   choices=["HD2K", "HD1080", "HD1200", "HD720", "SVGA", "VGA"],
                   help="ZED capture resolution -- more pixels on target improves "
                        "corner localisation precision")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", default=None, help="ZED serial; empty = first found")
    p.add_argument("--reject-px", type=float, default=3.0,
                   help="drop views above this reprojection error, 0 disables")
    p.add_argument("--solve-only", action="store_true",
                   help="skip capture, solve from an existing directory")
    p.add_argument("--capture-only", action="store_true")
    p.add_argument("--no-control", action="store_true",
                   help="skip RTDEControl (no freedrive; zero the TCP on the pendant)")
    p.add_argument("--force", action="store_true", help="overwrite existing images")
    p.add_argument("--no-viz", dest="viz", action="store_false")
    args = p.parse_args()

    if args.square_size <= 0:
        sys.exit("[!] --square-size must be positive")
    print(f"[cfg] board {args.cols}x{args.rows} squares "
          f"({args.cols-1}x{args.rows-1} corners) @ {args.square_size*1000:.2f} mm")
    if (args.cols - 1 + args.rows - 1) % 2 == 0:
        print("[warn] inner-corner counts sum to an even number, so the board is "
              "symmetric under 180 deg. Corner ordering can flip between frames. "
              f"Consider a {args.cols+1}x{args.rows} board.")

    if not args.solve_only:
        capture(args)
    if not args.capture_only:
        run_solve(args)


if __name__ == "__main__":
    main()
