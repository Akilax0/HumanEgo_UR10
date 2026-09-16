# -*- coding: utf-8 -*-
# @FileName: run_inference.py
"""
HumanEgo real-world inference — reference loop (UR10e + RealSense/ZED).

    camera ─▶ perception ─▶ ICT + clean image ─▶ policy ─▶ EE trajectory ─▶ robot
       ▲                                                                      │
       └──────────────────────── close the loop ◀────────────────────────────┘

This is a TEMPLATE: it shows the standard structure of a HumanEgo inference
stack. It will not run as-is — you must supply working hardware drivers and a
perception module (this variant ships two camera backends, Intel RealSense
and Stereolabs ZED, picked via `camera.type` in the inference yaml — plus a
single UR10e over `ur_rtde` with a Robotiq gripper + DINO-SAM/LaMa perception).
Everything hardware/perception-specific is isolated in the adapters at the
top and clearly marked with `TODO`; the policy + control logic below is generic.

Pipeline
--------
ONE-TIME (episode start):
    1. estimate object 6DoF poses from a few RGB-D frames  -> object-centric frame
    2. home the arms, open grippers

EVERY STEP (closed loop, ~5-10 Hz):
    3. grab RGB-D
    4. read each arm's EE pose (FK) + gripper state
    5. "latch" grasped objects so their pose tracks the gripper
    6. build the clean, embodiment-agnostic image (inpaint arm + render gripper)
    7. build the ICT from hand + object poses
    8. policy.infer -> future EE trajectory (+ done probability) for both arms
    9. decode trajectory to camera-frame EE targets, execute the first few steps
   10. stop when the policy reports "done"
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List

import numpy as np
import yaml

# Make BOTH this folder and the repo root importable regardless of CWD, so the
# flat imports here (interfaces/policy/controller, CamRS/RobotArmUR10) and the
# package imports inside policy.py (training.*, utils.*) all resolve when you run
# `python inference_ur10/run_inference.py cfg/inference/example_ur10.yaml` from root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from interfaces import Camera, Frame, ObjectState, Perception, RobotArm
from policy import ICTPolicy
from controller import TrajectoryController


# =====================================================================
# 1. HARDWARE ADAPTERS  — replace the bodies with your own hardware.
#    These wrap the shipped RealSense / UR10e drivers to satisfy the
#    interfaces in interfaces.py. The mapping is the whole point: copy
#    this pattern for your camera and arm.
# =====================================================================

class RealSenseCamera(Camera):
    """Adapter over the shipped CamRS (Intel RealSense) driver."""
    def __init__(self, cam_cfg_path: str):
        from CamRS import CamRS                       # TODO: your camera SDK
        self.cam = CamRS(cam_cfg_path)

    def get_frame(self) -> Frame:
        d = self.cam.get_rgbd()                       # CamRSData(rgb, depth_m, ...)
        return Frame(rgb=d.rgb, depth_m=d.depth_m, K=self.cam.k_rgb)

    def close(self) -> None:
        self.cam.close()


class ZedCamera(Camera):
    """Adapter over the shipped CamZED (Stereolabs ZED) driver.

    Same shape as RealSenseCamera — swap `camera.type: "zed"` in the inference
    yaml (see cfg/inference/example_ur10/CamZED.yaml) to use this instead.
    """
    def __init__(self, cam_cfg_path: str):
        from CamZED import CamZED                      # requires the pyzed SDK
        self.cam = CamZED(cam_cfg_path)

    def get_frame(self) -> Frame:
        d = self.cam.get_rgbd()                       # CamZEDData(rgb, depth_m, ...)
        return Frame(rgb=d.rgb, depth_m=d.depth_m, K=self.cam.k_rgb)

    def close(self) -> None:
        self.cam.close()


_CAMERA_ADAPTERS = {
    "realsense": RealSenseCamera,
    "zed": ZedCamera,
}


def build_camera(cfg: dict) -> Camera:
    """Pick the camera adapter from cfg["camera"]["type"] (default: "realsense")."""
    cam_type = cfg.get("type", "realsense").lower()
    if cam_type not in _CAMERA_ADAPTERS:
        raise ValueError(f"Unknown camera.type '{cam_type}'. Options: {list(_CAMERA_ADAPTERS)}")
    return _CAMERA_ADAPTERS[cam_type](cfg["cfg_path"])


class UR10Arm(RobotArm):
    """Adapter over the shipped RobotArmUR10 (ur_rtde + Robotiq) driver."""
    def __init__(self, arm_cfg_path: str):
        from RobotArmUR10 import RobotArmUR10         # TODO: your robot SDK
        self.arm = RobotArmUR10(arm_cfg_path)
        self.T_base_in_cam = self.arm.T_base_in_cam   # from hand-eye calibration

    def get_T_ee_in_cam(self) -> np.ndarray:
        return self.arm.get_T_ee_in_cam()
    def move_ee_in_cam(self, T, duration, blocking=False) -> bool:
        return self.arm.move_ee_in_cam(T, duration=duration, blocking=blocking)
    def get_gripper(self) -> float:
        return self.arm.get_gripper()                 # driver already returns 0=open .. 1=closed
    def set_gripper(self, value: float, blocking: bool = False) -> None:
        self.arm.set_gripper(value, blocking=blocking)
    def go_home(self, blocking: bool = True) -> None:
        self.arm.go_home(blocking=blocking)
    def close(self) -> None:
        self.arm.close()


# =====================================================================
# 2. PERCEPTION  — the heaviest part to port. Reference impl uses the
#    shipped preprocess/ engines (DINO-SAM detect+segment, LaMa inpaint).
#    Swap in ANY detector/pose-estimator that returns object poses.
# =====================================================================

class ReferencePerception(Perception):
    """Object 6DoF poses + clean image, mirroring the training preprocessing.

    The exact same detect→segment→lift→pose / inpaint→render pipeline is used
    at training time (see preprocess/), which is why train and test images match.
    The only difference from the offline preprocess/ pipeline is *how* pixels get
    lifted to 3D: training uses CoTracker'ed keypoints across a whole recorded
    session, while here we only have a handful of live RGB-D frames grabbed at
    episode start, so we lift a dense sample of in-mask pixels directly via
    `CamRS.lift3d_for_multi_frames` (per-pixel median + MAD gating across frames).
    """
    def __init__(self, cam: Camera, cfg: dict):
        self.cam = cam                                 # RealSenseCamera or ZedCamera —
                                                        # either way `self.cam.cam` must expose
                                                        # lift3d_for_multi_frames (see below).
        self.prompts = cfg["object_prompts"]          # {"obj1": "a green cup .", ...}
        self.erase_prompt = cfg["erase_prompt"]       # e.g. "a robot arm . a gripper ."
        self.anchor_key = cfg.get("anchor_key", "obj1")
        self.max_lift_points = cfg.get("max_lift_points", 400)   # cap pixels lifted per object
        self.grasp_render_threshold = cfg.get("grasp_threshold", 0.5)
        # Heavy models — load once. (These imports/weights are the engineering cost.)
        from preprocess.DINOSAM import DINOSAM
        from preprocess.Lama import LamaEngine
        from preprocess.VisualKpts import VisualKptsEngine
        from preprocess.OrientAnything import estimate_frame_pca2
        self._estimate_frame_pca2 = estimate_frame_pca2
        self.detector = DINOSAM(cfg["dinosam_cfg_path"])
        self.inpainter = LamaEngine(cfg["lama_cfg_path"])
        self.renderer = VisualKptsEngine(cfg["visualkpts_cfg_path"])

    def estimate_objects(self, frames: List[Frame]) -> Dict[str, ObjectState]:
        objs: Dict[str, ObjectState] = {}
        depths = [f.depth_m for f in frames]
        K = frames[0].K

        # Process the anchor first: context objects' pose estimation needs its center.
        keys = list(self.prompts.keys())
        if self.anchor_key in keys:
            keys.remove(self.anchor_key)
            keys = [self.anchor_key] + keys

        anchor_center_cam = None
        for key in keys:
            prompt = self.prompts[key]
            is_anchor = key == self.anchor_key

            # (a) detect + segment the object on the first frame
            mask = self.detector.process_single(frames[0].rgb, prompt)
            if mask is None or not np.any(mask):
                raise RuntimeError(
                    f"Perception: DINO-SAM found nothing for object '{key}' "
                    f"(prompt='{prompt}'). Check the prompt / camera framing."
                )

            # (b) lift the masked pixels to 3D across frames (robust to depth noise)
            vs, us = np.where(mask > 0)                # np.where on (H,W) -> (row=v, col=u)
            stride = max(1, len(us) // self.max_lift_points)
            pts3d_cam = []
            for u, v in zip(us[::stride], vs[::stride]):
                xyz, _q = self.cam.cam.lift3d_for_multi_frames((u, v), depths, mask, K)
                if xyz is not None:
                    pts3d_cam.append(xyz)
            pts3d_cam = np.asarray(pts3d_cam, dtype=np.float64)
            if len(pts3d_cam) < 10:
                raise RuntimeError(
                    f"Perception: only {len(pts3d_cam)} valid 3D points lifted for "
                    f"object '{key}' — check depth range / mask quality."
                )

            # (c) fit a 6DoF pose to the 3D points (PCA / your favorite estimator)
            T_in_cam, _info = self._estimate_frame_pca2(
                pts3d_cam, is_anchor=is_anchor, anchor_center_cam=anchor_center_cam
            )
            if is_anchor:
                anchor_center_cam = T_in_cam[:3, 3]

            # (d) keypoints in the object's own frame (only needed for PCD features)
            R, t = T_in_cam[:3, :3], T_in_cam[:3, 3]
            kpts_local = (pts3d_cam - t[None, :]) @ R   # == inv(T_in_cam) @ homogeneous(pts3d_cam)

            objs[key] = ObjectState(
                T_in_cam=T_in_cam.astype(np.float32),
                kpts_local=kpts_local.astype(np.float32),
            )
        return objs

    def make_clean_image(self, frame, ee_poses_in_cam, grippers) -> np.ndarray:
        # (a) inpaint the real arm out
        mask = self.detector.process_single(frame.rgb, self.erase_prompt)
        clean = (
            self.inpainter.inpaint(frame.rgb, mask)
            if mask is not None and np.any(mask)
            else frame.rgb.copy()
        )
        # (b) render a virtual gripper at each EE pose onto `clean`, matching the
        #     training visualization.
        for side, T_ee in ee_poses_in_cam.items():
            if T_ee is None:
                continue
            is_grasping = grippers.get(side, 0.0) > self.grasp_render_threshold
            clean = self.renderer.process_single_gripper(clean, T_ee, is_grasping, frame.K)
        return clean


# =====================================================================
# 3. Small helpers
# =====================================================================

def latch_objects(
    static_objs: Dict[str, ObjectState],
    hands_in_cam: Dict[str, np.ndarray],
    grippers: Dict[str, float],
    latches: dict,
    anchor_key: str,
    grasp_threshold: float = 0.5,
) -> Dict[str, ObjectState]:
    """Make a grasped object's pose follow the gripper.

    When an arm closes, we lock the relative transform gripper->nearest-object;
    while it stays closed, the object's pose is driven by the gripper. This keeps
    the object token in the ICT consistent with what's physically happening even
    if vision can't see the occluded object. (Simplified single-object latch.)

    The ANCHOR object is never latched — it defines the fixed object-centric
    reference frame for the episode and must stay put.
    """
    dynamic = {k: ObjectState(v.T_in_cam.copy(), v.kpts_local) for k, v in static_objs.items()}
    latchable = {k: v for k, v in static_objs.items() if k != anchor_key}
    if not latchable:
        return dynamic
    for side, T_h in hands_in_cam.items():
        if T_h is None:
            continue
        closed = grippers.get(side, 0.0) > grasp_threshold
        if closed and latches.get(side) is None:
            # latch onto the nearest (non-anchor) object at the moment of grasp
            nearest = min(latchable, key=lambda k: np.linalg.norm(
                latchable[k].T_in_cam[:3, 3] - T_h[:3, 3]))
            latches[side] = (nearest, np.linalg.inv(T_h) @ latchable[nearest].T_in_cam)
        elif not closed:
            latches[side] = None
        if latches.get(side) is not None:
            key, T_lock = latches[side]
            dynamic[key] = ObjectState(T_h @ T_lock, static_objs[key].kpts_local)
    return dynamic


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# =====================================================================
# 4. Main loop
# =====================================================================

def run(cfg_path: str, device: str = "cuda") -> None:
    cfg = load_cfg(cfg_path)
    print(cfg)
    cfg_sides = cfg["robot"]["sides"]                  # physical arms present, e.g. ["left","right"]
    T_align = np.array(cfg["robot"].get("T_align", np.eye(4).tolist()), dtype=np.float32)
    anchor_key = cfg["perception"].get("anchor_key", "obj1")
    exec_horizon = cfg["control"].get("exec_horizon", 8)   # steps run before re-planning
    dt = 1.0 / cfg["control"].get("control_hz", 10.0)
    done_threshold = cfg["control"].get("done_threshold", 0.8)

    # ---- build the stack ----
    cam = build_camera(cfg["camera"])              # cfg["camera"]["type"]: "realsense" (default) | "zed"
    arms: Dict[str, RobotArm] = {s: UR10Arm(cfg["robot"]["cfg_paths"][s]) for s in cfg_sides}
    policy = ICTPolicy(cfg["policy"], device=device)
    controller = TrajectoryController(arms, cfg["control"])
    perception = ReferencePerception(cam, cfg["perception"])

    # The policy decides how many hands it drives (read from the checkpoint).
    sides = policy.sides                               # e.g. ["left","right"] or ["right"]
    missing = [s for s in sides if s not in arms]
    if missing:
        raise ValueError(f"checkpoint predicts {sides} but no robot arm configured for {missing}. "
                         f"Set robot.sides to include them.")

    # ---- one-time: object-centric setup + homing ----
    print("[run] estimating object poses...")
    n = cfg["perception"].get("n_init_frames", 10)
    print("n",n)
    static_objs = perception.estimate_objects([cam.get_frame() for _ in range(n)])
    anchor = static_objs.get(anchor_key)               # fixed object-centric reference for the episode
    for arm in arms.values():
        arm.set_gripper(0.0, blocking=True)            # open
        arm.go_home(blocking=True)

    latches: dict = {s: None for s in sides}
    done = {s: False for s in sides}
    print("[run] entering closed-loop control. Ctrl-C to stop.")

    try:
        while not all(done.values()):
            frame = cam.get_frame()

            # read robot state -> HAND-frame poses (EE pose bridged by T_align)
            hands_in_cam = {s: arms[s].get_T_ee_in_cam() @ T_align for s in sides}
            grippers = {s: arms[s].get_gripper() for s in sides}

            # grasped (non-anchor) objects follow the gripper
            objs = latch_objects(static_objs, hands_in_cam, grippers, latches,
                                 anchor_key, controller.grasp_threshold)

            # print("Objs: ",objs)
            # build the policy inputs: clean image, ICT, and (if region-attn) anchor UV
            clean = perception.make_clean_image(frame, hands_in_cam, grippers)
            x_ict, ict_mask = policy.build_ict(hands_in_cam, grippers, objs, anchor_key)
            x_rgb = policy.prepare_image(clean)
            anchor_uv = policy.compute_anchor_uv(anchor, frame.K,
                                                 frame.rgb.shape[1], frame.rgb.shape[0])

            # predict the future trajectory for every hand the policy drives
            traj, done_prob = policy.infer(x_rgb, x_ict, ict_mask, anchor_uv)

            # decode reference-frame predictions -> camera-frame EE targets
            ee_targets: Dict[str, List[np.ndarray]] = {}
            grasp_cmds: Dict[str, np.ndarray] = {}
            for s in sides:
                pos, o6d, grasp = traj[s]
                ee_targets[s] = [policy.decode_ee_in_cam(pos[k], o6d[k], anchor, T_align)
                                 for k in range(len(pos))]
                grasp_cmds[s] = grasp


            # print("ee_targets: ",ee_targets)
            # print("grasp_cmds: ",grasp_cmds)

            # execute the first few steps, then re-plan (receding horizon)
            controller.execute_chunk(ee_targets, grasp_cmds, dt=dt, n_steps=exec_horizon)

            if done_prob > done_threshold:
                print(f"[run] policy reports done (p={done_prob:.2f}).")
                done = {s: True for s in sides}

    except KeyboardInterrupt:
        print("\n[run] interrupted.")
    finally:
        controller.home()
        for arm in arms.values():
            arm.close()
        cam.close()


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "./cfg/inference/example_ur10.yaml"
    run(cfg_path)
