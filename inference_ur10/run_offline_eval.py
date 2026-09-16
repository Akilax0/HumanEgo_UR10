# -*- coding: utf-8 -*-
# @FileName: run_offline_eval.py
"""
Offline policy inference on a recorded session — no robot, no camera, no perception
stack. Loads a trained checkpoint and runs the policy against a preprocessed Aria
recording's stored frames/poses (teacher-forced), rendering a GT-vs-Pred video.

Usage:
    python inference/run_offline_eval.py \
        --run_dir runs/serve_bread/HumanEgo \
        --mps_path data/serve_bread/aria/mps_serve_bread_000_vrs \
        --out_dir runs/serve_bread/HumanEgo/eval_render/manual


        ● Done. run_offline_eval.py now also writes predictions.json alongside the video, under
  .../teacher_forced_vis/predictions.json.

  What's in it:
  - Top-level: pred_horizon (50), num_inference_steps (20), action_mode, hand_sides, and the frequency summary —
    num_model_calls, mean_dt_between_calls_s, inferred_call_hz.
  - frames[]: one entry per input frame the model was actually run on (26 here), each with frame_idx, timestamp_ns
    (from the recording), gt_done, pred_done_prob, and hands.<side>.
  - Each hand entry has current_T_hand_to_world (4×4 pose at that input) and pred_trajectory — all 50 predicted future
    steps, each a full T_hand_to_world 4×4 pose plus grasp_pred.

  On "how frequently the model is run": it's invoked once per teacher-forced input frame, i.e. at the recording's
  native camera rate — for this session that's confirmed empirically from consecutive frame timestamps: 10.00 Hz
  (mean_dt_between_calls_s ≈ 0.0999925). Each call then predicts 50 steps ahead at that same 10 Hz spacing, so it
  forecasts ~5 seconds into the future per call. This is distinct from a live control loop's rate (that's
  control.control_hz / exec_horizon in cfg/inference/*.yaml) — this number describes the offline teacher-forced replay
  cadence, not deployed control frequency.

"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.FlowMatchingModel import FlowMatchingModel
from training.FlowMatchingEvaluator import run_teacher_forced_vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. runs/serve_bread/HumanEgo (has latest.pt, config.json, dataset_stats.json)")
    ap.add_argument("--mps_path", required=True, help="e.g. data/serve_bread/aria/mps_serve_bread_000_vrs")
    ap.add_argument("--out_dir", default=None, help="defaults to <run_dir>/eval_render/manual")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_frames", type=int, default=None)
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(args.run_dir, "dataset_stats.json")) as f:
        stats = json.load(f)

    model = FlowMatchingModel(
        single_hand=cfg["single_hand"],
        pred_horizon=cfg["pred_horizon"],
        max_ict=cfg["max_ict"],
        img_size=cfg["image_size"],
        patch_size=cfg["patch_size"],
        vision_embed_dim=cfg["vision_embed_dim"],
        num_decoder_layers=cfg["num_decoder_layers"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        dropout=cfg["dropout"],
        horizon_weighting=cfg["model_horizon_weighting"],
        horizon_beta=cfg["model_horizon_beta"],
        use_pcd_features=cfg["use_pcd_features"],
        use_aux_obj_dynamics=cfg["use_aux_obj_dynamics"],
        use_aux_visual_foresight=cfg["use_aux_visual_foresight"],
        use_aux_temporal_contrastive=cfg["use_aux_temporal_contrastive"],
        use_region_attn=cfg["use_region_attn"],
        use_pre_norm=cfg["use_pre_norm"],
        use_ctx_norm=cfg["use_ctx_norm"],
        use_done_in_flow=cfg["use_done_in_flow"],
    ).to(args.device)

    ckpt_path = os.path.join(args.run_dir, "latest.pt")
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    out_dir = args.out_dir or os.path.join(args.run_dir, "eval_render", "manual")
    session_name = os.path.basename(args.mps_path.rstrip("/"))
    out_dir = os.path.join(out_dir, session_name)

    run_teacher_forced_vis(
        model=model,
        ckpt_path=ckpt_path,
        mps_path=args.mps_path,
        out_dir=out_dir,
        single_hand_side=cfg["single_hand_side"],
        pred_horizon=cfg["pred_horizon"],
        image_size=tuple(cfg["image_size"]),
        device=args.device,
        centric_mode=cfg["centric_mode"],
        frame_mode=cfg["frame_mode"],
        action_mode=cfg["action_mode"],
        max_ict=cfg["max_ict"],
        stats=stats,
        num_inference_steps=cfg["num_inference_steps"],
        max_frames=args.max_frames,
        make_video=cfg["make_video"],
        use_done_in_flow=cfg["use_done_in_flow"],
    )

    print(f"[done] video written under {out_dir}/teacher_forced_vis/")


if __name__ == "__main__":
    main()
