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
EXPERIMENT_NAME="GF_layer-3_dim-768_attnbias"

TOKEN_DIM="768"                               # Shared DINO/gaze/decoder token dimension
DECODER_DEPTH="3"                             # Number of cross-attention decoder blocks
USE_LORA="False"                              # LoRA on the frozen DINOv3 encoder (see model.lora in the config)

LR="2e-4"


# ── Out-of-cone (OOC) penalty ─────────────────────────────────────────────── #
# Penalises the share of predicted probability mass falling outside the ground-truth gaze cone.
# WEIGHT_OOC=0 reproduces the baseline exactly (the term is then skipped entirely).
WEIGHT_OOC="0"                                # 0 disables. 3 -> ~11% of the total loss at convergence
LOG_OOC="False"                               # True + WEIGHT_OOC=0 to measure the raw term without training on it

OOC_THETA_DEG="60.0"                          # Cone aperture. Narrower approaches rigid cone supervision
OOC_ALPHA="30.0"                              # Cone boundary sharpness (relative to head-target distance)
OOC_NORMALIZE_ALPHA="True"                    # Scale alpha by 1/||g-h||; False reproduces the paper's fixed alpha
OOC_TAU="1.0"                                 # Spatial-softmax temperature (measured support 482/4096 cells at 1.0)
OOC_MIN_GAZE_DIST="0.05"                      # Skip samples whose target is this close to the head centre


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
    model.lora.use="$USE_LORA" \
    model.XGaze.image_encoder.dinov3.model_name="$DINO_MODEL" \
    model.XGaze.image_encoder.dinov3.local_dir="$DINO_DIR" \
    optimizer.lr="$LR" \
    loss.weight_ooc="$WEIGHT_OOC" \
    loss.log_ooc="$LOG_OOC" \
    loss.ooc.theta_deg="$OOC_THETA_DEG" \
    loss.ooc.alpha="$OOC_ALPHA" \
    loss.ooc.normalize_alpha="$OOC_NORMALIZE_ALPHA" \
    loss.ooc.tau="$OOC_TAU" \
    loss.ooc.min_gaze_dist="$OOC_MIN_GAZE_DIST" \
    "hydra.run.dir=\${hydra:runtime.cwd}/experiments/\${now:%Y-%m-%d}/${EXPERIMENT_NAME}"
