# Aria Gen2 Support — Changes Log

This document records every change made to add Project Aria **Gen2** glasses support
to the HumanEgo pipeline, why each change was needed, and how it was verified. Before
this work, the pipeline only supported Gen1 (`datacollection/README.md` said so
explicitly: *"It's for Aria Gen1 glasses for now. The code for Aria Gen2 is coming
soon..."*).

All Gen2-specific behavior is gated behind an explicit `device: "gen1" | "gen2"` config
flag — nothing here changes default (Gen1) behavior unless a task config opts in.

Everything below was verified against a real Gen2 recording (`Grasp_Bellpepper_Yellow`,
a single-object grasp task) processed through the full 9-stage `Preprocess` pipeline,
first locally (Aria-stage only, on an RTX 3050) and then end-to-end on a remote
headless HPC node.

---

## 1. Summary of files changed

| File | Type | What changed |
|---|---|---|
| `cfg/preprocess/base/AriaCam.yaml` | modified | new `device` flag |
| `preprocess/AriaCam.py` | modified | conditional rotation/intrinsics; safe devignetting fallback |
| `preprocess/Preprocess.py` | modified | device-aware RGB/hand-tracking frame-range logic |
| `preprocess/AriaHands.py` | modified | timestamp-based hand-tracking alignment for Gen2 |
| `cfg/preprocess/base/AriaHands.yaml` | modified | new `hand_match_tolerance_ms` field |
| `preprocess/AriaPhasesOps.py` | modified | fixed a latent `cv2` shape bug (not Gen1/Gen2-specific) |
| `preprocess/CamTriangulator.py` | modified | NaN/degenerate-data guards; Open3D export made skippable |
| `cfg/preprocess/base/CamTriangulator.yaml` | modified | new `export_ply` flag |
| `preprocess/Lama.py` | modified | added missing CUDA 12 libs to the onnxruntime GPU path setup |
| `requirements.txt` | modified | Aria SDK version bump; `numpy`/`onnxruntime-gpu` upper-bound pins |
| `setup.sh` | modified | matching version bumps |
| `cfg/preprocess/tasks/grasp_bellpepper.yaml` | **new** | example Gen2 task config |

---

## 2. Core Gen2 compatibility changes

### 2.1 `cfg/preprocess/base/AriaCam.yaml` — the `device` switch

Added:
```yaml
device: "gen1"   # "gen1" | "gen2"
```

This is the single flag that controls every other Gen2-specific behavior below. It
defaults to `"gen1"` so existing configs/behavior are untouched; a task config opts
into Gen2 handling with `AriaCam: {device: "gen2"}`.

### 2.2 `preprocess/AriaCam.py` — no more forced 90° rotation

**Why:** Gen1's RGB sensor is physically mounted rotated 90°, so the original code
unconditionally rotated the image and adjusted the intrinsic matrix (K) to compensate
(`rotate_camera_calib_cw90deg`, `np.rot90(img, k=3)`, and a row-swapped K matrix).

**What we found:** Gen2's sensor does *not* need this. This was verified empirically,
not assumed — the raw frame straight off `VrsDataProvider.get_image_data_by_index()`
already comes back with shape `(1920, 2560, 3)`, matching the camera's configured
`(h=1920, w=2560)` exactly, i.e. already upright/landscape. Rendering the same frame
through the old Gen1 rotation logic produces a visibly sideways image; rendering it
without rotation produces the correct upright frame (confirmed by eye on a real frame
containing a robot arm, a table, and a hand — see the pipeline's own diagnostic PNGs).

**What changed:**
- `self.rotate = (device != "gen2")`, computed once in `_init_cam()`.
- `_init_cam_pinhole_calib()`: only calls `rotate_camera_calib_cw90deg` and builds the
  swapped-W/H variant when `self.rotate` is true; otherwise both pinhole calibrations
  are identical.
- `_get_intrinsics()`: only applies the row-swapped/`w - cy` adjusted K matrix when
  `self.rotate` is true; otherwise uses the standard `[[fx,0,cx],[0,fy,cy],[0,0,1]]`.
- `_rotate_img()`: returns the image unchanged when `self.rotate` is false.

**Also fixed alongside this:** `_init_devignetting_mask()` now wraps
`load_devignetting_mask()` in a `try/except RuntimeError` and sets
`self.devignetting_mask = None` on failure instead of crashing. The bundled
devignetting masks (`preprocess/aria_devignetting_masks/`) are Gen1-shaped only; Gen2's
2560×1920 resolution has no matching mask shipped. `_devignette_img()` now returns the
image unchanged when there's no mask, so this degrades gracefully (slightly uneven
brightness/vignetting in the raw frame) rather than blocking the pipeline.

### 2.3 `preprocess/Preprocess.py` + `preprocess/AriaHands.py` — timestamp-based hand-tracking alignment

**Why:** The original frame-range logic assumed hand tracking and RGB frames are 1:1,
trailing-aligned:
```python
self.start_idx = len(rgb_tss) - len(self.aria_hands_mps)
self.end_idx = self.start_idx + len(self.aria_hands_mps) - 1
```
This holds for Gen1 (hand tracking runs at ~RGB frame rate). It does **not** hold for
Gen2: on the real recording tested, RGB runs at **10 fps** (160 frames over ~16s) while
MPS hand tracking runs at its own independent rate (**640 rows** over the same ~16s,
~40 Hz) — each row carries its own absolute `tracking_timestamp`, not a frame index.
With the old logic, `start_idx = 160 - 640 = -480`, a negative index that breaks frame
extraction immediately.

**What changed:**
- `Preprocess.init_preprocess()`: reads `device` from the merged `AriaCam` config. On
  `"gen2"`, uses the full RGB range (`start_idx=0, end_idx=len(rgb_tss)-1`) instead of
  count-based trimming, since alignment now happens per-frame downstream by timestamp.
- `AriaHandsGenerator.__init__()`: new `device` parameter (default `"gen1"`, so the
  standalone `python -m preprocess.AriaHands` entry point is unaffected).
- `AriaHandsGenerator.get_aria_hands()`: on `"gen2"`, instead of `zip(aria_hands_mps,
  aria_cam.cam)`, each camera frame's timestamp is matched to the nearest hand-tracking
  row via a new `_find_nearest_hand()` helper (binary search via `np.searchsorted` over
  each row's `tracking_timestamp`), within a configurable tolerance
  (`hand_match_tolerance_ms`, default 50ms — see §2.4). If no row is close enough, the
  frame is treated as having no hand detection (matches existing behavior for
  low-confidence/missing detections).
- `_get_aria_hands_data()` now tolerates `hands_mps=None` (returns no-hand-detected for
  both hands) instead of assuming a valid MPS row is always present.

The Gen1 code path (`zip`-based, positional) is completely untouched — this only
activates on `device: "gen2"`.

### 2.4 `cfg/preprocess/base/AriaHands.yaml` — `hand_match_tolerance_ms`

New field, only consulted on Gen2:
```yaml
hand_match_tolerance_ms: 50
```
Maximum timestamp gap (ms) for an RGB frame to be matched to a hand-tracking row.
Beyond this, the frame is treated as having no hand detected.

---

## 3. A genuinely unrelated bug found and fixed along the way

### 3.1 `preprocess/AriaPhasesOps.py` — `cv2.morphologyEx` shape bug

While validating the Gen2 aria-stage pipeline, `AriaPhases.get_aria_phases()` crashed
with:
```
ValueError: The truth value of an array with more than one element is ambiguous.
```
inside `_remove_short_true_runs`. Root cause, confirmed by direct repro:
```python
cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)   # m.shape == (160,)
# → output shape (1, 160), NOT (160,)
```
`close_binary_mask()` fed this reshaped `(1, n)` array downstream, silently corrupting
every subsequent step that assumed a 1D mask (`len(m)` returned `1` instead of `n`,
etc.). **This is independent of Gen1 vs Gen2** — it would hit any recording that
reaches this code path, on whatever OpenCV version returns the row-vector shape. Fixed
by explicitly flattening the output:
```python
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel).reshape(-1)
```

---

## 4. Environment / dependency fixes

These were all discovered by actually running the pipeline end-to-end, not
speculative — each one blocked a real run.

### 4.1 `projectaria-tools` / `projectaria-client-sdk` version bump

`requirements.txt` / `setup.sh` pinned `projectaria-tools==1.7.1`, which **predates
Gen2 support entirely**. Meta added Gen2 VRS/MPS support in `projectaria-tools==2.0.0`
(same Python/C++ API as Gen1, forward compatible per Meta's docs). Bumped to:
```
projectaria-tools==2.1.4
projectaria-client-sdk==2.2.0   # (2.4.0 doesn't exist on public PyPI — max is 2.2.0)
```
Verified: every API surface `AriaCam.py`/`AriaHands.py` depend on
(`get_device_calibration`, `get_camera_calib`, `get_linear_camera_calibration`,
`rotate_camera_calib_cw90deg`, `devignetting`, `distort_by_calibration`,
`mps.hand_tracking.read_hand_tracking_results`, `HandLandmark` enum,
`wrist_and_palm_normal_device`, etc.) is intact in 2.1.4. **Not** re-verified against a
real Gen1 recording after this bump — worth a smoke test before relying on it for Gen1
data.

### 4.2 `onnxruntime-gpu` — pinned below CUDA 13

Unpinned `onnxruntime-gpu>=1.17` resolved to `1.27.0`, which requires
`libcudart.so.13` (CUDA 13 runtime) — incompatible with the pinned
`torch==2.5.1+cu121` (CUDA 12.1) stack, and crashed at **import time** (blocking
`preprocess/Lama.py`, and therefore all of `Preprocess.py`, since it's imported
unconditionally at module load). Fixed:
```
onnxruntime-gpu>=1.17,<1.21   # 1.21+ needs CUDA 13; torch==2.5.1 here is cu121
```
`1.20.1` confirmed to import cleanly and expose `CUDAExecutionProvider` correctly.

### 4.3 `numpy` — pinned to avoid ABI drift

`setup.sh` installed `numpy` **completely unpinned** (`pip install numpy`, before
anything else, for chumpy's build order), and `requirements.txt` only had a floor
(`numpy>=1.24`). This is the same failure pattern as §4.2 — a fresh install grabs
whatever's newest on PyPI, and `np.linalg.svd`/`scipy.optimize.least_squares` (compiled
C/Fortran extensions, exactly where a real crash was hit — see §5) are highly
ABI-sensitive to the numpy version they were built against. Fixed in both places:
```
numpy>=1.24,<2.2
```

### 4.4 `preprocess/Lama.py` — missing CUDA 12 libraries for onnxruntime

Even after §4.2's version fix, LaMa's `onnxruntime.InferenceSession(...,
providers=[('CUDAExecutionProvider', ...), 'CPUExecutionProvider'])` failed to
initialize CUDA on the remote server:
```
Failed to load library libonnxruntime_providers_cuda.so with error:
libnvrtc.so.12: cannot open shared object file: No such file or directory
```
`setup_cuda_paths()` (which builds `LD_LIBRARY_PATH` from the `nvidia-*-cu12` pip
packages' bundled `.so` files) was missing several of them from its search list —
`libnvrtc.so.12` (needed for CUDA JIT) genuinely wasn't on the path even though
`nvidia-cuda-nvrtc-cu12` was already installed as a `torch` dependency. Fixed by
extending the list:
```python
libs_to_find = ['nvidia.cublas', 'nvidia.cudnn', 'nvidia.cuda_runtime', 'nvidia.cufft',
                'nvidia.curand', 'nvidia.cuda_nvrtc', 'nvidia.cusparse',
                'nvidia.cusolver', 'nvidia.nvjitlink']
```
Note: `onnxruntime` already falls back to `CPUExecutionProvider` gracefully when CUDA
init fails (a warning, not a crash) — so this wasn't a hard blocker, just meant LaMa
inpainting silently ran on CPU (slower) until fixed.

---

## 5. Open3D segfault on headless servers

**Symptom:** `Segmentation fault (core dumped)`, no Python traceback, during Stage 6
(3D triangulation), immediately after `"Processing obj1 3D points..."`.

**Diagnosis process:**
1. Initially suspected degenerate/NaN triangulation data reaching `np.linalg.svd`/
   `scipy.optimize.least_squares` in `CamTriangulatorEngine.triangulate_dlt`/
   `ba_refine` — added defensive guards there regardless (see §5.1), since they're good
   hardening either way (skip non-finite pixel observations, skip singular/degenerate
   poses, validate the SVD system matrix and homogeneous divide before using them, bail
   out of `ba_refine` early on a non-finite input).
2. The segfault persisted identically. Re-ran with `PYTHONFAULTHANDLER=1` /
   `python -X faulthandler`, which pinpointed the actual crash site precisely:
   `CamTriangulator.py:400`, inside `_export_camtriangulator_vis_ply()`, at
   `sph.translate(pt)` right after `o3d.geometry.TriangleMesh.create_sphere(...)`.
3. This — combined with the target machine being a headless HPC compute node
   (`/fs/nexus-scratch/...`, no display) — matches a well-known Open3D failure mode:
   some builds initialize a GLFW/OpenGL context internally on first geometry use and
   crash natively (not a catchable Python exception) when no display/EGL is available,
   even for pure CPU geometry operations that don't look like they need rendering.

**Fix:** Since neither PLY export is ever read by training or inference (confirmed —
no `.ply`/`open3d` reference anywhere in `training/` or `inference/`; they're
diagnostic-only artifacts), both exports were made fully skippable, gated to return
*before* any `o3d.*` call executes (not just before the file write):

- **`preprocess/CamTriangulator.py`** (`_export_camtriangulator_vis_ply`): new
  `export_ply` flag (added to `cfg/preprocess/base/CamTriangulator.yaml`, defaults
  `true` — no behavior change unless a task opts out).
- **`preprocess/DatasetGen.py`**: no code change needed — it already had an
  `export_obj_and_traj_ply` flag (`cfg/preprocess/base/DatasetGen.yaml`) guarding its
  equivalent Open3D export at the top of the function; it just wasn't being set to
  `false` anywhere. Set in the task config instead.
- `preprocess/DepthLifter.py` / `preprocess/RobotDatasetGen.py` also import Open3D but
  aren't part of the default `Preprocess.run()` sequence, so they're not a concern here.

### 5.1 `CamTriangulatorEngine` defensive hardening (kept regardless of the above)

In `triangulate_dlt`:
- Skip any observation with non-finite `(u, v)` pixel coordinates.
- Skip any pose with near-zero determinant (a degenerate/singular matrix — e.g. an
  `np.eye(4)` fallback from a failed MPS pose lookup) *before* calling
  `np.linalg.inv`/feeding it into the SVD system.
- Validate the assembled system matrix `A` is fully finite before calling
  `np.linalg.svd`.
- Guard the homogeneous divide (`X[:3] / X[3]`) against `X[3] ≈ 0`.

In `ba_refine`:
- Bail out early (return the input unchanged) if the incoming `p3d` is already
  non-finite, rather than handing NaN into `scipy.optimize.least_squares`.
- Same singular-pose guard inside the residual function.
- Handle the case where every observation was skipped (empty residual vector).

In `_export_camtriangulator_vis_ply`, added an `is_sane_point()` check (finite and
`|value| < 1e4`) before ever constructing Open3D geometry from a triangulated point,
origin, or axis vector — logs a warning and skips instead.

---

## 6. New task config: `cfg/preprocess/tasks/grasp_bellpepper.yaml`

Created for the `Grasp_Bellpepper_Yellow` recordings (single-object grasp, no second
object/plate). Use this as a template for new Gen2 tasks:

```yaml
AriaCam:
  device: "gen2"
  cam_size:
    rgb: {w: 2560, h: 1920, fov: 90}   # native Gen2 RGB resolution

AriaHands:
  hand_selection: "right"
  hand_match_tolerance_ms: 50
  # Frame-count windows below are tuned for Gen1's ~30fps in the shipped defaults;
  # rescaled ~3x down here since this recording is 10fps.
  hand_min_frames: 10        # was 30
  hand_interp_max_gap: 3     # was 10
  grasp_smooth_win: 2        # was 5
  grasp_flicker_max_len: 2   # was 5
  smooth_sg_window: 7        # was 21 (must stay odd)

AriaPhases:
  finished_frames: 10              # was 30
  manip_clean_manual_offset: 3     # was 10
  stop_hold_frames: 5              # was 15
  stop_debounce_frames: 3          # was 8
  stop_min_on_frames: 25           # was 75
  stop_offset_frames: 3            # was 10

DINOSAM:
  dinosam_prompt:
    obj1: "yellow bell pepper ."
    arm:  "human arms . human hands ."

CamTriangulator:
  export_ply: false    # headless server — see §5
  pose_method:
    obj1: "pca2"
    default: "pca2"

DatasetGen:
  export_obj_and_traj_ply: false   # headless server — see §5
```

### Checklist for adding a new Gen2 task

1. Set `AriaCam.device: "gen2"` and `AriaCam.cam_size.rgb` to the native resolution of
   your recording (check with `provider.get_image_configuration(rgb_stream_id)` if
   unsure — Gen2 RGB is commonly 2560×1920 @ 10fps, but confirm per-recording since MPS
   recording profiles can differ).
2. Set `AriaHands.hand_selection` and `hand_match_tolerance_ms` (50ms is a reasonable
   default; tighten it if hand tracking runs much faster/slower than ~40Hz on your
   recording).
3. **Check the actual frame rate** of your recording (`AriaCamData`/the printed
   `fps` in `aria_cam_rgb_config.json`) — if it's meaningfully different from 30fps,
   rescale every frame-count window in `AriaHands`/`AriaPhases` proportionally (see the
   "was N" comments above for the Gen1-default baseline each one scales from).
4. Set `DINOSAM.dinosam_prompt` for your object(s) + `arm`.
5. Set `CamTriangulator.pose_method` per object.
6. **If running on a headless server** (no display/X11/EGL), set
   `CamTriangulator.export_ply: false` and `DatasetGen.export_obj_and_traj_ply: false`
   to avoid the Open3D segfault in §5. If you have a real display or can use
   `xvfb-run`, these can stay `true` for the diagnostic PLY output.
7. Run preprocessing and check `dinosam_mask_obj*.png` + the `preprocess/cfg/` snapshot
   to confirm prompts/overrides landed correctly, per the existing guidance in
   `preprocess/README.md` §5.2.

---

## 7. What was empirically verified vs. what wasn't

**Verified, on a real Gen2 recording, end-to-end:**
- Stream label `camera-rgb` unchanged from Gen1; SLAM cameras are now
  `slam-front-left`/`slam-front-right`/`slam-side-left`/`slam-side-right` (4 cameras,
  not 2) — irrelevant here since `Preprocess.py` only ever consumes the `rgb` stream.
- No 90° rotation needed (rendered and visually confirmed both ways).
- Timestamp-based hand-tracking alignment (145/160 frames with a right-hand detection,
  2 grasp events, on a recording with RGB at 10fps and hand tracking at ~40Hz).
- Phase segmentation producing a coherent navigation → manipulation → finished
  breakdown.
- Full 9-stage pipeline (aria extraction → phase indexing → DINOSAM → keypoint
  selection → CoTracker → triangulation → LaMa inpainting → visualization → dataset
  generation) completing successfully on a remote headless server after the fixes in
  §4–§5.

**Not verified / worth checking before relying on it:**
- Gen1 recordings haven't been re-tested against the bumped `projectaria-tools`/
  `numpy`/`onnxruntime-gpu` versions in this session.
- The exact `hand_match_tolerance_ms` default (50ms) and the fps-rescaled window
  values in `grasp_bellpepper.yaml` were chosen by direct calculation from the known
  Gen1 defaults' implied real-world durations, not independently tuned/validated beyond
  "the pipeline runs and produces a sane-looking phase segmentation."
- Devignetting is silently skipped for Gen2 (no matching mask shipped) — frames will
  have slightly uneven brightness at the edges; not corrected.
