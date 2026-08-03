#!/bin/bash

#SBATCH --job-name=xgaze-cp
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx6000:1
#SBATCH -c 10
#SBATCH --mem 64G
#SBATCH -t 48:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ── Settings ──────────────────────────────────────────────────────────────── #
TASKS="train+test"                               # train+test (finetune then test) | test (zero-shot test only)
WEIGHTS="/home/jinwoongjung/XGaze/checkpoints/GF_layer-3_dim-768.ckpt" # ckpt to warm-start/eval from train_gf.sh
EXPERIMENT_NAME="CP_ft"

TOKEN_DIM="768"                                  # Shared DINO/gaze/decoder token dimension
DECODER_DEPTH="3"                                # Number of cross-attention decoder blocks

LR="5e-6"
LR_INOUT="1e-3"
INOUT_ONLY="False"                         # train only the in/out decoder; keep gaze localization frozen

CHECKPOINT_MONITOR="metric/val/dist"  # metric/val/inout_ap | metric/val/dist
CHECKPOINT_MODE="min"                      # max for AP | min for distance

DINO_SIZE="L"                                    # B (ViT-B/16) | L (ViT-L/16)

case "$DINO_SIZE" in
    B)
        DINO_MODEL="facebook/dinov3-vitb16-pretrain-lvd1689m"
        DINO_DIR="/home/jinwoongjung/XGaze/weights/dinov3-vitb16"
        ;;
    L)
        DINO_MODEL="facebook/dinov3-vitl16-pretrain-lvd1689m"
        DINO_DIR="/home/jinwoongjung/XGaze/weights/dinov3-vitl16"
        ;;
    *)
        echo "Invalid DINO_SIZE='$DINO_SIZE'. Use B or L." >&2
        exit 1
        ;;
esac

python main.py --config-name=config_childplay \
    experiment.task="$TASKS" \
    experiment.name="$EXPERIMENT_NAME" \
    model.XGaze.token_dim="$TOKEN_DIM" \
    model.XGaze.decoder_depth="$DECODER_DEPTH" \
    model.XGaze.image_encoder.dinov3.model_name="$DINO_MODEL" \
    model.XGaze.image_encoder.dinov3.local_dir="$DINO_DIR" \
    optimizer.lr="$LR" \
    train.checkpointing.monitor="$CHECKPOINT_MONITOR" \
    train.checkpointing.mode="$CHECKPOINT_MODE" \
    optimizer.lr_inout="$LR_INOUT" \
    train.freeze.inout_only="$INOUT_ONLY" \
    model.weights="$WEIGHTS" \
    "hydra.run.dir=\${hydra:runtime.cwd}/experiments/\${now:%Y-%m-%d}/${EXPERIMENT_NAME}"
