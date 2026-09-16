#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Report each raw recording's duration and the cumulative total, BEFORE preprocessing.

Duration is derived from the first/last timestamp in each recording's
hand_tracking/hand_tracking_results.csv (tracking_timestamp_us column) — this is
already produced by MPS for every recording, so it's a fast, dependency-free way to
check segment length without decoding the VRS file. Cross-checked against a recording
that has already been preprocessed (Grasp_Bellpepper_Yellow_000: ~16.0s here, 160
frames in preprocess/all_data at the Gen2 RGB rate of 10fps — 16.0 * 10 = 160, matches).

  dir           directory containing the raw mps_*_vrs recordings (required)

Examples
--------
    python scripts/check_recording_durations.py ../aria_data/Bellpepper/grasp/Grasp_Bellpepper_Yellow
"""
import argparse
import csv
import os


def get_duration_seconds(csv_path: str):
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        first_ts = None
        last_ts = None
        n_rows = 0
        for row in reader:
            if not row:
                continue
            ts = int(row[0])
            if first_ts is None:
                first_ts = ts
            last_ts = ts
            n_rows += 1
    if first_ts is None:
        return None, 0
    return (last_ts - first_ts) / 1e6, n_rows


def format_hms(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="directory containing the raw mps_*_vrs recordings")
    args = parser.parse_args()

    recordings = sorted(
        name for name in os.listdir(args.dir)
        if os.path.isdir(os.path.join(args.dir, name))
        and os.path.isfile(os.path.join(args.dir, name, "hand_tracking", "hand_tracking_results.csv"))
    )
    if not recordings:
        print(f"No recordings with hand_tracking/hand_tracking_results.csv found under {args.dir}")
        return

    total_seconds = 0.0
    print(f"{'Recording':<40} {'Duration':>10}")
    print("-" * 51)
    for name in recordings:
        csv_path = os.path.join(args.dir, name, "hand_tracking", "hand_tracking_results.csv")
        duration, n_rows = get_duration_seconds(csv_path)
        if duration is None:
            print(f"{name:<40} {'no data rows':>10}")
            continue
        total_seconds += duration
        print(f"{name:<40} {format_hms(duration):>10}")

    print("-" * 51)
    print(f"{'Cumulative':<40} {format_hms(total_seconds):>10}  ({total_seconds:.1f}s across {len(recordings)} recordings)")


if __name__ == "__main__":
    main()
