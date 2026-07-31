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
TASKS="train+test"                            # train+test (train from scratch then test) | test (zero-shot test only)
EXPERIMENT_NAME="GF_layer-3_dim-256"

TOKEN_DIM="256"                               # Shared DINO/gaze/decoder token dimension
DECODER_DEPTH="3"                             # Number of cross-attention decoder blocks

LR="2e-4"

DINO_SIZE="L"                                 # B (ViT-B/16) | L (ViT-L/16)

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

python main.py --config-name=config_gazefollow \
    experiment.task="$TASKS" \
    experiment.name="$EXPERIMENT_NAME" \
    model.XGaze.token_dim="$TOKEN_DIM" \
    model.XGaze.decoder_depth="$DECODER_DEPTH" \
    model.XGaze.image_encoder.dinov3.model_name="$DINO_MODEL" \
    model.XGaze.image_encoder.dinov3.local_dir="$DINO_DIR" \
    optimizer.lr="$LR" \
    "hydra.run.dir=\${hydra:runtime.cwd}/experiments/\${now:%Y-%m-%d}/${EXPERIMENT_NAME}"
