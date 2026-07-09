#!/bin/bash

#SBATCH --job-name=xgaze-vat
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -c 10
#SBATCH --mem 64G
#SBATCH -t 48:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ── Settings ──────────────────────────────────────────────────────────────── #
TASKS="train+test"                              # train+test (finetune then test) | test (zero-shot test only)
WEIGHTS="/home/jinwoongjung/XGaze/checkpoints/best_gf.ckpt"       # ckpt to warm-start/eval from (eg. from train_gf.sh);

python main.py --config-name=config_vat \
    experiment.task="$TASKS" \
    model.weights="$WEIGHTS"
