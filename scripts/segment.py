#!/usr/bin/env python3
"""Split Aria MPS output into time-segmented copies."""
import gzip, json, shutil
from pathlib import Path
import pandas as pd
import subprocess

MPS_ROOT = Path("/home/ax0/Documents/aria_data/Bellpepper_grasp_retake/mps/mps_Grasp_Bellpepper_Red_7_20260908_170304_vrs")  # where the original MPS output is
OUT_ROOT  = Path("/home/ax0/Documents/aria_data/Bellpepper_grasp_retake/segmented/")  # where to write the segments
MARKS_FILE = Path("/home/ax0/Documents/aria_data/Bellpepper_grasp_retake/marks_files/Grasp_BellPepper_Red_7_marks.txt") 
VRS = MPS_ROOT / "sample.vrs"

# Define segments in device time, microseconds.
# Get these from the VRS (first RGB frame ts of each logical "recording")
START_IDX = 90
prefix = "mps_Grasp_BellPepper_Red"
ts = [int(l.split("\t")[0]) for l in open(MARKS_FILE) if l.strip()]
assert len(ts) % 2 == 0, f"odd number of marks ({len(ts)}) — missing a boundary?"
SEGMENTS = [
    (f"{prefix}_{START_IDX + k:03d}_vrs", ts[2*k], ts[2*k + 1])
    for k in range(len(ts) // 2)
]
# SEGMENTS = [
#     ("mps_Grasp_BellPepper_Red_046_vrs", 8665837647, 8679736754),
#     ("mps_Grasp_BellPepper_Red_047_vrs", 8685236122, 8697335218),
#     ("mps_Grasp_BellPepper_Red_048_vrs", 8704334693, 8718333644),
#     ("mps_Grasp_BellPepper_Red_049_vrs", 8724033362, 8735232672),
#     ("mps_Grasp_BellPepper_Red_050_vrs", 8739832030,8751831127),
#     ("mps_Grasp_BellPepper_Red_051_vrs", 8756430781, 8768529869),
#     ("mps_Grasp_BellPepper_Red_052_vrs", 8776829242, 8788228381),
#     ("mps_Grasp_BellPepper_Red_053_vrs", 8792228079, 8803527224),
#     ("mps_Grasp_BellPepper_Red_054_vrs", 8807526921, 8816926209),
#     ("mps_Grasp_BellPepper_Red_055_vrs", 8826025519, 8838224593),
#     ("mps_Grasp_BellPepper_Red_056_vrs", 8843324206, 8856023386),
#     ("mps_Grasp_BellPepper_Red_057_vrs", 8862022785, 8873621902),
#     ("mps_Grasp_BellPepper_Red_058_vrs", 8878721514, 8890820592),
#     ("mps_Grasp_BellPepper_Red_059_vrs", 8899420038, 8908619735),
#     ("mps_Grasp_BellPepper_Red_060_vrs", 8915918679, 8926417877),
#     ("mps_Grasp_BellPepper_Red_061_vrs", 8935017221, 8944916464),
#     ("mps_Grasp_BellPepper_Red_062_vrs", 8951715945, 8963015081),
# ]

TIME_FILES = [
    ("hand_tracking/hand_tracking_results.csv",     "tracking_timestamp_us"),
    ("slam/closed_loop_trajectory.csv",             "tracking_timestamp_us"),
    ("slam/open_loop_trajectory.csv",               "tracking_timestamp_us"),
    ("slam/semidense_observations.csv.gz",          "frame_tracking_timestamp_us"),
]

# copied verbatim into every segment (not time-indexed)
COPY_FILES = [
    "hand_tracking/summary.json",
    "slam/summary.json",
    "else/vrs_health_check.json",
]




# ---------------------------------------------------- load sources once
print("loading sources...")
tables = {}
for rel, col in TIME_FILES:
    src = MPS_ROOT / rel
    if not src.exists():
        print(f"  MISSING {rel}")
        continue
    df = pd.read_csv(src, compression="infer")
    if col not in df.columns:
        raise KeyError(f"{rel}: no column '{col}'; found {list(df.columns)[:8]}")
    tables[rel] = (df, col)
    print(f"  {rel:42s} {len(df):>9d} rows  "
          f"[{df[col].min()} .. {df[col].max()}]")

points = None
pts_src = MPS_ROOT / "slam/semidense_points.csv.gz"
if pts_src.exists():
    points = pd.read_csv(pts_src)
    print(f"  {'slam/semidense_points.csv.gz':42s} {len(points):>9d} rows")

calib_lines = []
calib_src = MPS_ROOT / "slam/online_calibration.jsonl"
if calib_src.exists():
    with open(calib_src) as f:
        for line in f:
            if line.strip():
                ts = json.loads(line).get("tracking_timestamp_us")
                if ts is not None:
                    calib_lines.append((ts, line))
    print(f"  {'slam/online_calibration.jsonl':42s} {len(calib_lines):>9d} rows")

# ---------------------------------------------------------------- split
for name, t0, t1 in SEGMENTS:
    seg = OUT_ROOT / name
    print(f"\n== {name}  [{t0}, {t1})  {(t1 - t0) / 1e6:.1f}s")

    obs_seg = None
    for rel, col in TIME_FILES:
        if rel not in tables:
            continue
        df, col = tables[rel]
        out = df[(df[col] >= t0) & (df[col] < t1)]
        dst = seg / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dst, index=False,
                   compression="gzip" if dst.name.endswith(".gz") else None)
        flag = "  <-- EMPTY" if len(out) == 0 else ""
        print(f"   {rel:42s} {len(out):>7d}{flag}")
        if rel.endswith("semidense_observations.csv.gz"):
            obs_seg = out

    # points: keep only landmarks observed inside this window
    if points is not None:
        if obs_seg is not None:
            pts_seg = points[points["uid"].isin(set(obs_seg["uid"]))]
        else:
            pts_seg = points
        dst = seg / "slam/semidense_points.csv.gz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        pts_seg.to_csv(dst, index=False, compression="gzip")
        print(f"   {'slam/semidense_points.csv.gz':42s} {len(pts_seg):>7d}")

    # online calibration
    if calib_lines:
        dst = seg / "slam/online_calibration.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        kept = 0
        with open(dst, "w") as out:
            for ts, line in calib_lines:
                if t0 <= ts < t1:
                    out.write(line)
                    kept += 1
        print(f"   {'slam/online_calibration.jsonl':42s} {kept:>7d}")

    # verbatim metadata
    for rel in COPY_FILES:
        src = MPS_ROOT / rel
        if src.exists():
            dst = seg / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)


    # --- per-segment VRS clip ---
    if VRS.exists():
        dst = seg / "sample.vrs"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "vrs", "copy", str(VRS),
            "--after", f"{t0 / 1e6:.6f}",
            "--before", f"{t1 / 1e6:.6f}",
            "--to", str(dst),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"   vrs copy FAILED: {r.stderr.strip().splitlines()[-1:]}")
        else:
            mb = dst.stat().st_size / 1e6
            print(f"   {'sample.vrs':42s} {mb:>7.1f} MB")

print(f"\ndone -> {OUT_ROOT}/")