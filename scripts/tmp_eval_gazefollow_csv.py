"""Run GazeFollow test and save per-image predictions and metrics to CSV."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.functional.classification.auroc import binary_auroc

from XGaze.datasets.gazefollow import GazeFollowDataset, IMG_MEAN, IMG_STD
from XGaze.modeling.xgaze import XGazeModule
from XGaze.transforms import Compose, Normalize, Resize, ToTensor
from XGaze.utils.common import dark_coordinate_decoding, generate_binary_gaze_heatmap


def move_batch(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_cfg(repo, checkpoint, batch_size, num_workers):
    with initialize_config_dir(version_base="1.1", config_dir=str(Path(repo) / "XGaze" / "conf")):
        cfg = compose(config_name="config_gazefollow")
    cfg.project.root = str(repo)
    cfg.experiment.task = "test"
    cfg.experiment.dataset = "gazefollow"
    cfg.model.XGaze.token_dim = 768
    cfg.model.XGaze.decoder_depth = 3
    cfg.model.XGaze.image_encoder.dinov3.model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    cfg.model.XGaze.image_encoder.dinov3.local_dir = str(Path(repo) / "weights" / "dinov3-vitl16")
    cfg.model.weights = None
    cfg.test.checkpoint = str(checkpoint)
    cfg.test.batch_size = batch_size
    cfg.data.num_workers = num_workers
    cfg.train.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.test.device = cfg.train.device
    cfg.wandb.log = False
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--data-root", default="/home/jinwoongjung/MTGS/data/gazefollow_extended")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    cfg = make_cfg(repo, args.checkpoint, args.batch_size, args.num_workers)
    device = torch.device(cfg.train.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    transform = Compose([
        Resize(img_size=(cfg.data.image_size, cfg.data.image_size), head_size=(224, 224)),
        ToTensor(),
        Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
    ])
    dataset = GazeFollowDataset(
        root=args.data_root,
        root_project=str(repo),
        root_heads=cfg.data.gf.root_heads,
        split="test",
        transform=transform,
        heatmap_sigma=cfg.data.heatmap_sigma,
        heatmap_size=cfg.data.heatmap_size,
        num_people=1,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    print(f"Loading checkpoint: {args.checkpoint}")
    model = XGazeModule(cfg=cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    current = model.state_dict()
    filtered = {k: v for k, v in checkpoint["state_dict"].items() if k in current and current[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    del checkpoint, filtered, current
    model.eval().to(device)

    # The annotation dataframe is retained so the CSV contains the original (pre-expansion)
    # bbox and all individual test gaze annotations, while the batch contains transformed inputs.
    annotations = dataset.annotations
    rows = []
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            batch_device = move_batch(batch, device)
            heatmap_logits, _, _, _ = model._forward_step(batch_device)
            heatmap_prob = heatmap_logits.sigmoid()
            shape = heatmap_prob.shape
            pred = dark_coordinate_decoding(
                heatmap_prob.reshape(-1, *shape[-2:]),
                kernel_size=cfg.data.heatmap_sigma * 3,
                normalize=True,
            ).reshape(*shape[:-2], 2).detach().cpu()
            probs = heatmap_prob.detach().cpu()
            bsz = pred.shape[0]

            for j in range(bsz):
                path = batch["path"][j]
                ann = annotations[annotations.path == path]
                gt = ann[["gaze_x", "gaze_y"]].to_numpy(dtype="float32")
                gt_t = torch.from_numpy(gt)
                pred_t = pred[j, -1] if pred[j].ndim == 2 else pred[j]
                dists = torch.linalg.vector_norm(gt_t - pred_t, dim=1)
                gt_avg = gt_t.mean(0)
                hm_gt = generate_binary_gaze_heatmap(gt_t, size=probs[j, -1].shape if probs[j].ndim == 3 else probs[j].shape)
                hm = probs[j, -1] if probs[j].ndim == 3 else probs[j]
                auc = binary_auroc(hm.flatten(), hm_gt.flatten()).item()
                img_w, img_h = [int(x) for x in batch["img_size"][j].tolist()]
                raw_bbox = ann[["head_xmin", "head_ymin", "head_xmax", "head_ymax"]].iloc[0].to_numpy(dtype="float32")
                input_bbox = batch["head_bboxes"][j, -1].tolist()
                pred_xy = pred_t.tolist()
                rows.append({
                    "image_path": path,
                    "image_file": os.path.join(args.data_root, path),
                    "image_width": img_w,
                    "image_height": img_h,
                    "bbox_xmin": float(raw_bbox[0]), "bbox_ymin": float(raw_bbox[1]),
                    "bbox_xmax": float(raw_bbox[2]), "bbox_ymax": float(raw_bbox[3]),
                    "input_bbox_xmin_norm": float(input_bbox[0]), "input_bbox_ymin_norm": float(input_bbox[1]),
                    "input_bbox_xmax_norm": float(input_bbox[2]), "input_bbox_ymax_norm": float(input_bbox[3]),
                    "annotation_ids": json.dumps(ann.id.astype(int).tolist()),
                    "gt_gaze_points_norm": json.dumps(gt.tolist()),
                    "gt_avg_x_norm": float(gt_avg[0]), "gt_avg_y_norm": float(gt_avg[1]),
                    "pred_gaze_x_norm": float(pred_xy[0]), "pred_gaze_y_norm": float(pred_xy[1]),
                    "pred_gaze_x_px": float(pred_xy[0] * img_w), "pred_gaze_y_px": float(pred_xy[1] * img_h),
                    "min_l2": float(dists.min()),
                    "avg_l2": float(dists.mean()),
                    "dist_to_avg": float(torch.linalg.vector_norm(gt_avg - pred_t)),
                    "auc": float(auc),
                })
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {len(rows)}/{len(dataset)} images")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    result = pd.DataFrame(rows)
    print(f"Saved {len(result)} rows to {output}")
    print(result[["auc", "avg_l2", "min_l2", "dist_to_avg"]].mean().to_string())


if __name__ == "__main__":
    main()
