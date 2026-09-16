#!/bin/bash
#SBATCH --job-name=humanego_bellpepper
#SBATCH --output=/fs/nexus-scratch/akilax0/conformal_VLA/HumanEgo/logs/bellpepper_%j.out
#SBATCH --error=/fs/nexus-scratch/akilax0/conformal_VLA/HumanEgo/logs/bellpepper_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=12:00:00           # preprocessing (10 recordings) + training in one job;
                                   # bump this (or split into two sbatch scripts, see
                                   # bottom of file) if 400 epochs doesn't fit in the window
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger

# Exit immediately if a command exits with a non-zero status
set -e

# --- 0. PATHS (edit these for your layout) ---
REPO_DIR="/nfshomes/akilax0/research/conformal_VLA/HumanEgo"
# Where the raw Gen2 recordings (mps_Grasp_Bellpepper_Yellow_000_vrs .. _009_vrs) live.
# Adjust if the path on the cluster differs from the repo-relative one used locally.
RAW_DATA_DIR="/nfshomes/akilax0/research/Bellpepper/grasp/grasp_bellpepper_red"
TASK="grasp_bellpepper"

cd "$REPO_DIR"
mkdir -p logs

# --- 1. ENVIRONMENT SETUP ---
echo "===== Initializing Conda Environment ====="
eval "$(conda shell.bash hook)"
conda activate humanego-gen2

# # --- 2. LINK RAW RECORDINGS INTO data/<task>/aria/ ---
# # Preprocessing writes its output in-place under <mps_path>/preprocess/..., and training
# # auto-discovers sessions from data/<task>/aria/<session_dir>/preprocess/all_data/...
# # Symlinking (instead of copying) avoids duplicating the raw VRS/MPS data on disk.
# echo "===== Linking raw recordings into data/${TASK}/aria/ ====="
# mkdir -p "data/${TASK}/aria"
# for rec in "$RAW_DATA_DIR"/mps_*_vrs; do
#     name="$(basename "$rec")"
#     ln -sfn "$rec" "data/${TASK}/aria/${name}"
# done

# # --- 3. PREPROCESSING ---
# # --mps_path pointed at the parent dir runs batch mode: it scans every subfolder with a
# # sample.vrs and runs the full 9-stage pipeline on each, continuing past per-recording
# # failures. --no-video/--no-gif skip the (slow, GPU-idle) visualization exports, which
# # aren't needed for training.

# # echo "===== Running preprocessing on all ${TASK} recordings ====="
# # python -m preprocess.Preprocess \
# #     --mps_path "./data/${TASK}/aria" \
# #     --task "$TASK" \
# #     --no-video --no-gif

# # --- 3. PREPROCESSING ---
# echo "===== Running preprocessing on recordings starting from 098 ====="
# for rec in "$RAW_DATA_DIR"/mps_*_vrs; do
#     name="$(basename "$rec")"
    
#     # Lexicographically compare strings to skip earlier recordings
#     # Note: Double-check that your casing (Bellpepper vs BellPepper) matches the actual folder names
#     #if [[ "$name" < "mps_Grasp_Bellpepper_Red_098_vrs" ]]; then
#     #    continue
#     #fi
    
#     echo "Preprocessing: $name"
#     # Run preprocessing on this specific directory rather than the parent batch
#     python -m preprocess.Preprocess \
#         --mps_path "./data/${TASK}/aria/${name}" \
#         --task "$TASK" \
#         --no-video --no-gif
# done


# --- 4. TRAINING ---
echo "===== Launching training ====="
python -m training.FlowMatchingTrainer --task "$TASK" --use_cfg --job HumanEgo

echo "Preprocessing + training finished successfully."

# ------------------------------------------------------------------------------------
# To split this into two jobs instead (e.g. so a slow preprocessing run doesn't eat into
# the training time budget), submit sections 0-3 and section 4 as separate .sbatch files,
# and chain them with:
#   sbatch --dependency=afterok:<preprocess_job_id> run_grasp_bellpepper_train.sbatch
# ------------------------------------------------------------------------------------
