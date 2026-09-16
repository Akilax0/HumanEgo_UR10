# Inference — Real-World Deployment (UR10e)

This folder is a **reference template** for deploying a trained HumanEgo policy on
a real robot. It is intentionally clean and hardware-agnostic: it shows
the *standard structure* of a HumanEgo inference stack so you can wire in **your
own** camera, robot, and perception and reuse everything else.

This variant targets a **single UR10e arm** (over `ur_rtde`) with a **Robotiq
gripper** and an **Intel RealSense** camera — see [`../inference/`](../inference/)
for the paper's dual-arm Trossen variant.

> ⚠️ **It will not run out of the box.** It depends on physical hardware
> (camera + arm), a hand-eye calibration, and heavy perception models. Treat it
> as the blueprint to build your own deployment from — not a turn-key script.
> This example uses **Intel RealSense + a UR10e (via `ur_rtde`) + a Robotiq
> gripper**, with **DINO-SAM + LaMa** perception (same as the paper).

---

## The idea in one picture

```
 camera ─▶ perception ─▶ clean image + ICT ─▶ policy ─▶ EE trajectory ─▶ robot
    ▲                                                                      │
    └──────────────────────────── close the loop ◀─────────────────────────┘
```

A HumanEgo policy consumes two things every step and predicts a future
end-effector trajectory:

1. **A clean, embodiment-agnostic RGB image** — the real arm is inpainted out and
   a virtual gripper is rendered in its place (this closes the *visual gap* between
   human-video training and robot deployment).
2. **Interaction-Centric Tokens (ICT)** — a compact, viewpoint- and
   embodiment-invariant encoding of every hand and object as a 6DoF entity plus
   each hand's pose *relative to* that entity (this closes the *kinematic gap*).

Because the ICT and clean image are built **identically** at train and test time,
a policy trained purely on human egocentric video transfers to the robot.

---

## Files

| File | Role |
|------|------|
| [`interfaces.py`](interfaces.py) | The 3 abstractions you implement: `Camera`, `RobotArm`, `Perception`. The loop is written entirely against these. |
| [`policy.py`](policy.py) | `ICTPolicy`: load the checkpoint, `prepare_image`, `build_ict` (dual-arm), flow-matching `infer`, decode to camera-frame EE targets. |
| [`controller.py`](controller.py) | `TrajectoryController`: smooth (EMA + Slerp) + rate-limit predictions and servo the arms (receding horizon). |
| [`run_inference.py`](run_inference.py) | The main loop wiring it all together, plus example hardware adapters and a reference perception. |
| [`../cfg/inference/example_ur10.yaml`](../cfg/inference/example_ur10.yaml) | One annotated config for the whole stack. |
| `CamRS.py`, `RobotArmUR10.py` | **Example** drivers (RealSense / UR10e + Robotiq) you can adapt to your own hardware. |

---

## Read this first — frame & unit conventions

Most deployment bugs are frame bugs. The contract (see `interfaces.py`):

- **Poses are 4×4 SE(3) matrices, positions in meters, rotations proper.**
- **`_in_cam` = the camera optical frame** (OpenCV: +x right, +y down, +z forward).
  This is the single shared "world" frame for one episode.
- **`T_base_in_cam`** (per arm) — the **hand-eye extrinsic** placing the robot base
  in the camera frame. Get it from a hand-eye calibration. A wrong extrinsic is the
  #1 cause of "the robot moves to the wrong place".
- **`T_align`** — bridges *your* end-effector frame to the **"hand" frame the model
  was trained on**. Identity if you trained on robot/teleop data in the same EE
  convention; for the released Aria-MPS checkpoints use the fixed rotation provided
  in [`example_dualarm.yaml`](../cfg/inference/example_dualarm.yaml) (the `T_align`
  block). Wrong `T_align` ⇒ orientations are systematically off.

---

## The pipeline, step by step

**One-time, at episode start** (`run_inference.py: run()`):

1. **Estimate object poses** — `Perception.estimate_objects()` detects+segments each
   object, lifts mask pixels to 3D via depth, and fits a 6DoF pose. The **anchor**
   object (`obj1`) defines the object-centric reference frame the ICTs use.
2. **Home** the arms and open grippers.

**Every step, closed-loop** (~5–10 Hz):

| # | What | Where |
|---|------|-------|
| 3 | Grab an RGB-D frame | `Camera.get_frame()` |
| 4 | Read each arm's EE pose (FK) + gripper, bridge by `T_align` → **hand poses** | `RobotArm.get_T_ee_in_cam()` |
| 5 | **Latch** grasped objects so their pose tracks the gripper | `run_inference.py: latch_objects()` |
| 6 | Build the **clean image** (inpaint arm + render virtual gripper) | `Perception.make_clean_image()` |
| 7 | Build the **ICT** from hand + object poses | `ICTPolicy.build_ict()` |
| 8 | **Flow-matching inference** → future EE trajectory + done prob | `ICTPolicy.infer()` |
| 9 | **Decode** reference-frame prediction → camera-frame EE targets | `ICTPolicy.decode_ee_in_cam()` |
| 10 | Smooth + execute the first `exec_horizon` steps, then re-plan | `TrajectoryController.execute_chunk()` |
| 11 | Stop when `done_prob > done_threshold` | `run_inference.py: run()` |

### How the ICT is built (the core)

`ICTPolicy.build_ict()` mirrors `training/FlowMatchingDataloader._build_ict()` exactly.
Each entity (hand or object) becomes one token:

```
[ type_id(1) | pose_in_ref(9) | hand-in-entity(9 single / 18 dual) | flag(1) ]
```

- `pose_in_ref` — the entity's 6DoF pose in the reference frame, encoded as
  `[normalized position(3), 6D rotation(6)]`.
- `hand-in-entity` — each hand's pose **expressed in that entity's frame**. This
  relative encoding is what makes the representation *interaction*-centric and
  invariant to where the camera/robot is.
- Token order is fixed: **hand(s) first, then the anchor object, then the rest** —
  it must match training.

### How the robot moves

The policy predicts the future **hand** trajectory in the reference frame. To
command the arm, each step is mapped back (`decode_ee_in_cam`):

```
pred (hand pose in REF) --T_ref_in_cam--> hand pose in CAM --inv(T_align)--> EE pose in CAM
```

`TrajectoryController` then EMA-smooths position, Slerp-smooths rotation, clamps
the per-step motion (safety cage), and calls `RobotArm.move_ee_in_cam(...)`
non-blocking, plus opens/closes the gripper by thresholding the predicted grasp
probability. Only the first `exec_horizon` steps run before the loop re-plans on a
fresh observation — this **receding horizon** keeps the policy reactive.

---

## How to run

**Prerequisites**

1. A trained checkpoint with `config.json` **and** `dataset_stats.json` next to it
   (the trainer writes these; they carry the architecture + normalization). For
   this single-arm UR10e template it must be trained with `single_hand: true`.
2. Hardware drivers installed: `SKIP_HARDWARE=0 bash setup.sh` (RealSense +
   `ur_rtde`). The Robotiq gripper needs no extra package — `RobotArmUR10.py`
   talks to it directly over the UR controller's Modbus-over-socket gateway
   (port 63352).
3. Your perception models (the reference uses DINO-SAM + LaMa from `preprocess/`).
4. A **hand-eye calibration** → `T_base_in_cam` for the arm.

**Configure** — edit [`../cfg/inference/example_ur10.yaml`](../cfg/inference/example_ur10.yaml):
set `policy.ckpt`, the `perception.object_prompts` / `erase_prompt`, the camera
`cfg_path`, `robot.T_align`, and in
[`RobotArmUR10.yaml`](../cfg/inference/example_ur10/RobotArmUR10.yaml) the arm's
`ip`, `T_base_in_cam`, `home_joints`, and gripper block.

**Run**

```bash
python inference_ur10/run_inference.py cfg/inference/example_ur10.yaml
```

**Bring up the arm/gripper on their own first** (recommended before running the
full loop):

```bash
python -m inference_ur10.RobotArmUR10 --cfg_path cfg/inference/example_ur10/RobotArmUR10.yaml --home --test_gripper
```

---

## Configuration & tuning

All knobs live in `example_dualarm.yaml`. The ones you will actually tune:

| Knob | Section | Effect |
|------|---------|--------|
| `num_inference_steps` | `policy` | Flow ODE steps. 10–20. ↑ = smoother actions, slower. |
| `exec_horizon` | `control` | Predicted steps run before re-planning. ↓ = more reactive/closed-loop. |
| `control_hz` | `control` | Control-loop rate (sets `dt`). Match to your arm's servo rate. |
| `alpha_pos` / `alpha_rot` | `control` | EMA / Slerp smoothing. ↑ = smoother but laggier. |
| `max_pos_step` | `control` | Safety cage: max EE move per step (m). Keep small at first. |
| `grasp_threshold` | `control` | Predicted grasp prob above which the gripper closes. |
| `done_threshold` | `control` | Done prob above which the episode stops. |
| `safe_z_min` | `control` | Intended base-frame Z floor (table protection). **Note:** this template does not wire the `control` section's value through to the driver — set the *same* floor separately in `RobotArmUR10.yaml`'s own `safe_z_min` (enforced per-move in `RobotArmUR10.move_p`) — plus rely on the UR controller's own safety-plane / e-stop and set conservatively. |

**Bring-up tip:** start with a *low* `control_hz`, *small* `max_pos_step`, and a
high `safe_z_min`, hand on the e-stop. Loosen once the motion looks right.

---

## Single-arm vs dual-arm

This template defaults to **single-arm** (`robot.sides: ["right"]`, a checkpoint
trained with `single_hand: true` → `ict_dim = 20`) since a UR10e is normally
deployed alone. For **dual-arm**, add a second UR10e's config under
`robot.cfg_paths.left`, set `robot.sides: ["left", "right"]`, and use a checkpoint
trained with `single_hand: false` (`ict_dim = 29`, both hand trajectories
predicted in one forward pass). `ICTPolicy` reads `single_hand` from the
checkpoint's `config.json`, so the token layout and trajectory unpacking follow
automatically — no code changes needed either way.

---

## Writing your own camera / robot / perception

Implement the three interfaces in `interfaces.py` — that's the whole porting job:

- **`Camera`** → return `Frame(rgb, depth_m, K)`. See `RealSenseCamera` in
  `run_inference.py` wrapping `CamRS`.
- **`RobotArm`** → FK (`get_T_ee_in_cam`), Cartesian servo (`move_ee_in_cam`),
  gripper, `go_home`, and the `T_base_in_cam` extrinsic. See `UR10Arm` wrapping
  `RobotArmUR10` (which streams targets via `ur_rtde`'s `servoL`). Cartesian IK is
  handled by the UR controller itself.
- **`Perception`** → object 6DoF poses + a clean image. This is the heaviest part.
  Any source of object poses works (open-vocab detector + PCA on depth keypoints, an
  AprilTag, FoundationPose, known CAD + ICP, …). The clean image must match how your
  model was trained (same inpainting + gripper rendering as `preprocess/`).

---

## What this template leaves out

To keep the idea legible, this reference template intentionally omits several pieces
from our internal production loop (not part of this release):

- **Async control + temporal ensembling** — a worker thread servoing at a fixed rate
  decoupled from (slower) inference, averaging overlapping predictions for smoother
  motion.
- **Delta action mode**, PCD features, region attention, object-dynamics & visual-
  foresight auxiliary heads — all supported by the model; here we run the common
  absolute-action path.
- **Robustness/UX**: keyboard tele-override, live visualization, post-grasp forced
  lift, IK-failure escape, interactive extrinsic calibration, grasp latching across
  occlusion, checkpoint architecture auto-detection.

Start from this template, get a single arm reaching to an object, then layer these on
as you need them.
