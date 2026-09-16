# Changes — UR10e + RealSense port

This documents everything that was changed to turn this folder (originally a
copy of [`../inference/`](../inference/), the Trossen dual-arm reference) into
a working template for a **single UR10e arm** (via `ur_rtde`) with a **Robotiq
gripper** and an **Intel RealSense** camera.

Setup assumed: single arm (`robot.sides: ["right"]`, a checkpoint trained with
`single_hand: true`), gripper controlled via the Robotiq Modbus-over-socket
gateway (SDU's `robotiq_gripper.py` protocol) rather than plain digital I/O.

---

## New files

### `inference_ur10/RobotArmUR10.py`
New driver, replacing `RobotArmTrossen.py` for this folder. Two classes:

- **`RobotiqGripper`** — talks to the gripper over the UR controller's built-in
  Modbus-to-socket gateway (port 63352), using the same ASCII `SET`/`GET`
  protocol as SDU's reference `ur_rtde` example. Handles `activate()` (with
  optional auto-calibration of the true open/closed encoder range),
  `move()` / `move_and_wait_for_pos()`, and `get_current_position()`. Raw
  units are 0–255.
- **`RobotArmUR10`** — implements the `RobotArm` interface from
  `interfaces.py`:
  - `get_T_ee_in_cam()` / `get_T_ee_in_base()` — FK via
    `rtde_receive.getActualTCPPose()`. Unlike the Trossen driver, UR's native
    TCP pose is already `[x,y,z,rx,ry,rz]` (rotation vector) in the base frame
    for whatever TCP is configured, so **no manual frame-correction matrix is
    needed** — a `tcp_offset` is set once via `rtde_control.setTcp()` at connect
    time instead.
  - `move_ee_in_cam()` / `move_p_in_cam()` / `move_p()` — non-blocking moves use
    `rtde_control.servoL()`, which is designed for exactly the "stream a new
    Cartesian target every `dt` seconds" pattern that
    `TrajectoryController.execute_chunk()` already does (receding horizon).
    Blocking moves (e.g. homing) use `moveL` / `moveJ`.
  - `get_gripper()` / `set_gripper()` — normalized to `[0,1]` (0=open,
    1=closed), matching the `interfaces.RobotArm` contract directly (no
    inversion needed, unlike the Trossen adapter which had to flip the
    driver's native convention).
  - `go_home()`, `close()`, `T_base_in_cam` property — same shape as the
    Trossen driver.
  - A small bring-up CLI at the bottom
    (`python -m inference_ur10.RobotArmUR10 --home --test_gripper`) to sanity
    check the connection, FK, and gripper before running the full loop —
    much lighter than Trossen's Tkinter GUI, intentionally.

### `cfg/inference/example_ur10.yaml`
Single-arm counterpart to `cfg/inference/example_dualarm.yaml`. Same sections
(`policy`, `control`, `perception`, `camera`, `robot`); `robot.sides` is
`["right"]` and `robot.cfg_paths` points at the new UR10 config below.

### `cfg/inference/example_ur10/RobotArmUR10.yaml`
New hardware config read by `RobotArmUR10.py`: controller `ip`, `tcp_offset`,
`payload_kg`, `T_base_in_cam` (hand-eye extrinsic placeholder), `home_joints`,
`servoL` tuning (`servo_lookahead_time`, `servo_gain`), blocking-move speeds/
accelerations, a `safe_z_min` floor, and a `gripper:` block (`enabled`, `port`,
`speed`, `force`, `auto_calibrate`, `invert`).

### `cfg/inference/example_ur10/CamRS.yaml`
Copy of the existing RealSense camera template (unchanged content) — the
RealSense driver itself needed no hardware changes, only its own config slot
under the new `example_ur10/` folder.

---

## Removed files

### `inference_ur10/RobotArmTrossen.py`
Deleted. It depended on the `trossen_arm` package and Trossen-specific frame
corrections that don't apply to a UR10e; superseded by `RobotArmUR10.py`.

---

## Edited files

### `inference_ur10/run_inference.py`
- Replaced the `TrossenArm` adapter class with `UR10Arm`, wrapping
  `RobotArmUR10` instead of `RobotArmTrossen`. Gripper convention no longer
  needs inverting (`return self.arm.get_gripper()` instead of
  `1.0 - self.arm.get_gripper_q()`), and `move_ee_in_cam` calls straight into
  the driver's own `move_ee_in_cam` instead of manually converting through
  `T_to_p`.
- `run()` now instantiates `UR10Arm` per configured side.
- Updated the module docstring, adapter-section comment, and the `__main__`
  default config path (`./cfg/inference/example_ur10.yaml` instead of
  `example_dualarm.yaml`) to match.

### `inference_ur10/CamRS.py`
Fixed a **pre-existing bug** left over from copying `inference/` wholesale:
it imported `from inference.CamRSTypes import ...` (the *other* folder's
package path). This happened to still resolve because both folders sit under
the repo root without `__init__.py` (implicit namespace packages), so it
silently pulled in a different-but-identical file — fragile, and confusing if
the two folders ever diverge. Changed to the correct flat
`from CamRSTypes import ...`, matching how `run_inference.py` puts this
folder on `sys.path`. Also fixed the stale `python -m inference.CamRS` usage
comment at the bottom of the file to `python -m inference_ur10.CamRS`.

### `inference_ur10/interfaces.py`
Docstring-only: updated the `Camera`/`RobotArm` protocol comments that named
`RobotArmTrossen` (Trossen ViperX/WidowX) as the reference example to instead
name `RobotArmUR10` (UR10e over `ur_rtde`, Robotiq gripper).

### `inference_ur10/README.md`
Rewritten sections to describe the UR10e + Robotiq + RealSense stack instead
of dual-arm Trossen:
- Intro/hardware callout, files table, prerequisites, configure/run
  instructions (new cfg paths, added a "bring up the arm on its own first"
  step), the `safe_z_min` tuning note (now points at
  `RobotArmUR10.yaml`'s own floor instead of the Trossen driver's hardcoded
  one), the single-arm-vs-dual-arm section (default flipped to single-arm,
  since that's how UR10e is normally deployed), and the "writing your own
  robot" bullet (`UR10Arm`/`RobotArmUR10` instead of `TrossenArm`/
  `RobotArmTrossen`).

### `requirements.txt`
Added `ur_rtde>=1.5.1` next to the existing `pyrealsense2` / `trossen-arm`
hardware pins, with a comment noting it's imported as `rtde_control` /
`rtde_receive`.

### `setup.sh`
- Updated the `SKIP_HARDWARE` header comment to mention `ur_rtde`.
- Added `ur_rtde` to the `FILTER_PATTERN` that's excluded from
  `requirements.txt` install when `SKIP_HARDWARE=1` (the default).
- Added a `check_import "ur_rtde" "import rtde_control, rtde_receive"` line to
  the `[7/7] Verification` step's hardware checks.

---

## Not changed

- `controller.py`, `policy.py`, `CamRSTypes.py` — fully hardware-agnostic
  already; no edits needed.
- The `Perception` module (`ReferencePerception` in `run_inference.py`) is
  still an unimplemented stub, same as in the original template — it's
  task/detector-specific and independent of which robot/camera you use.
- No hand-eye calibration, no real `ip`/`home_joints`/gripper direction were
  filled in — those are placeholders you must set for your physical rig
  (marked `(SET)` in the new yaml files).
