#!/bin/bash

#SBATCH --job-name=xgaze-gf
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -c 10
#SBATCH --mem 64G
#SBATCH -t 48:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ── Settings ──────────────────────────────────────────────────────────────── #
TASKS="train+test"   # train+test (train from scratch then test) | test (zero-shot test only)

python main.py --config-name=config_gf \
    experiment.task="$TASKS" \
