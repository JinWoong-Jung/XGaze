"""Pick `loss.ooc.tau` and `loss.weight_ooc` from a trained checkpoint, without training anything.

Both hyper-parameters only depend on how peaked the heatmap logits are once the model has
converged, so a forward-only pass over the validation split answers them:

  * tau  - the BCE term drives the logits towards a near one-hot spatial softmax, which would make
           the OOC term degenerate. Pick the smallest tau whose effective support (exp of the
           softmax entropy, out of h*w cells) is comfortably above 1.
  * weight_ooc - compare the raw OOC term against the already-weighted heatmap term. The script
           prints the resulting share for a range of candidate weights.

Example:
    python scripts/probe_ooc_calibration.py \
        --ckpt experiments/2026-07-31/GF_layer-1_dim-256/checkpoints/best.ckpt \
        --overrides model.XGaze.token_dim=256 model.XGaze.decoder_depth=1 \
        --device cuda:7
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra import compose, initialize_config_dir  # noqa: E402

from XGaze.datasets.gazefollow import GazeFollowDataModule  # noqa: E402
from XGaze.losses import build_gaze_cone_mask, compute_heatmap_loss  # noqa: E402
from XGaze.modeling.xgaze import XGazeModule  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config-name", default="config_gazefollow")
    p.add_argument("--overrides", nargs="*", default=[], help="hydra overrides matching the checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--taus", type=float, nargs="*", default=[1.0, 2.0, 5.0, 10.0, 20.0])
    p.add_argument("--theta-deg", type=float, default=60.0)
    p.add_argument("--alpha", type=float, default=30.0)
    p.add_argument("--min-gaze-dist", type=float, default=0.05)
    p.add_argument("--no-normalize-alpha", action="store_true",
                   help="use the paper's fixed alpha instead of scaling it by the head-to-target distance")
    p.add_argument("--weights", type=float, nargs="*", default=[1.0, 3.0, 5.0, 10.0])
    return p.parse_args()


def main():
    args = parse_args()

    with initialize_config_dir(config_dir=str(REPO / "XGaze" / "conf"), version_base="1.1"):
        cfg = compose(config_name=args.config_name, overrides=list(args.overrides))

    # This probe hard-codes GazeFollowDataModule and the single-target (b, h, w) prediction layout,
    # so it would silently mis-measure the multi-person datasets. Fail loudly instead.
    if cfg.experiment.dataset != "gazefollow":
        raise SystemExit(
            f"probe_ooc_calibration.py only supports the gazefollow dataset, but "
            f"`{args.config_name}` sets experiment.dataset={cfg.experiment.dataset}. "
            f"VideoAttentionTarget/ChildPlay produce (b, n, h, w) predictions and per-person "
            f"targets, which this script does not handle."
        )
    normalize_alpha = not args.no_normalize_alpha

    print(f"loading {args.ckpt}")
    module = XGazeModule(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    module.on_load_checkpoint(ckpt)  # re-injects the frozen DINO backbone if it was stripped on save
    module.load_state_dict(ckpt["state_dict"], strict=True)
    module.to(args.device).eval()

    dm = GazeFollowDataModule(
        root=cfg.data.gf.root,
        root_project=cfg.project.root,
        root_heads=cfg.data.gf.root_heads,
        batch_size=args.batch_size,
        image_size=cfg.data.image_size,
        heatmap_size=cfg.data.heatmap_size,
        heatmap_sigma=cfg.data.heatmap_sigma,
        num_people=cfg.data.num_people,
        return_head_mask=cfg.data.return_head_mask,
        num_workers=4,
    )
    dm.setup("fit")

    # Everything is accumulated per sample so it can be reported per distance bin: the short-distance
    # behaviour is exactly where the cone and the fixed-width supervision blob disagree.
    logits_all, hm_losses = [], []
    per = {k: [] for k in ("dist", "uniform", "gt_cell", "gt_dist")}
    for t in args.taus:
        per[f"ooc@{t}"], per[f"top1@{t}"], per[f"sup@{t}"] = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(dm.val_dataloader()):
            if i >= args.batches:
                break
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            hm_pred, _, _, _ = module._forward_step(batch)  # (b, 64, 64) for gazefollow
            head_center, gaze_pt = module._ooc_targets(batch)
            io = batch["inout"]

            logits_all.append(hm_pred.flatten().float().cpu())
            hm_losses.append(compute_heatmap_loss(hm_pred, batch["gaze_heatmap"], io.clamp(min=0)).item())

            cone, dist = build_gaze_cone_mask(head_center, gaze_pt, hm_pred.shape[-2:],
                                              theta_deg=args.theta_deg, alpha=args.alpha,
                                              normalize_alpha=normalize_alpha)
            outside = 1.0 - cone
            elig = (io == 1) & (dist > args.min_gaze_dist)
            if not elig.any():
                continue

            def masked(x):
                return x[elig].float().cpu()

            per["dist"].append(masked(dist))
            # Reference distributions, both restricted to the eligible samples:
            #   uniform  - what an untrained, flat prediction scores (the initial value of the term).
            #   gt_cell  - all mass on the target cell. This is the limit BCE converges to, since the
            #              GT heatmap is normalized to max 1 and logit(1) is +inf.
            #   gt_dist  - mass shaped like the GT Gaussian itself. Its width is fixed (sigma=3 cells)
            #              while the cone half-width shrinks as r*tan(theta/2), so below r~0.16 the
            #              cone clips it. Raising tau moves the model from `gt_cell` towards this.
            hcell, wcell = cone.shape[-2:]
            uniform = outside.mean(dim=(-2, -1))
            gj = (gaze_pt[..., 0] * wcell).long().clamp(0, wcell - 1)
            gi = (gaze_pt[..., 1] * hcell).long().clamp(0, hcell - 1)
            gt_cell = outside[torch.arange(cone.shape[0], device=cone.device), gi, gj]
            gt_hm = batch["gaze_heatmap"]
            gt_prob = gt_hm / gt_hm.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
            gt_dist = (gt_prob * outside).sum(dim=(-2, -1))
            per["uniform"].append(masked(uniform))
            per["gt_cell"].append(masked(gt_cell))
            per["gt_dist"].append(masked(gt_dist))

            for tau in args.taus:
                p = torch.softmax((hm_pred / tau).flatten(start_dim=-2), dim=-1).view_as(hm_pred)
                per[f"ooc@{tau}"].append(masked((p * outside).sum(dim=(-2, -1))))
                per[f"top1@{tau}"].append(masked(p.flatten(start_dim=-2).max(dim=-1).values))
                ent = -(p * (p + 1e-12).log()).sum(dim=(-2, -1))
                per[f"sup@{tau}"].append(masked(ent.exp()))

    per = {k: torch.cat(v) for k, v in per.items()}
    r = per["dist"]
    cells = 64 * 64
    mean = lambda xs: sum(xs) / len(xs)
    logits = torch.cat(logits_all)

    print(f"\neligible samples: {len(r)} (in-frame and farther than min_gaze_dist={args.min_gaze_dist})")

    print("\n=== heatmap logits ===")
    q = torch.quantile(logits, torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0]))
    print(f"min={q[0]:.3f}  p1={q[1]:.3f}  median={q[2]:.3f}  p99={q[3]:.3f}  max={q[4]:.3f}  std={logits.std():.3f}")
    print(f"range (max-min) = {(q[4] - q[0]):.3f}   <- a large range makes the tau=1 softmax near one-hot")

    print("\n=== head-to-target distance ===")
    print(f"p10={r.quantile(0.10):.3f}  p50={r.quantile(0.50):.3f}  p90={r.quantile(0.90):.3f}")

    print(f"\n=== reference distributions (normalize_alpha={normalize_alpha}) ===")
    print(f"{'':>14} {'uniform':>10} {'GT-cell':>10} {'GT-dist':>10}")
    print(f"{'overall':>14} {per['uniform'].mean():10.4f} {per['gt_cell'].mean():10.4f} {per['gt_dist'].mean():10.4f}")
    bins = [(args.min_gaze_dist, 0.15), (0.15, 0.25), (0.25, 0.40), (0.40, 1.5)]
    for lo, hi in bins:
        m = (r >= lo) & (r < hi)
        if not m.any():
            continue
        print(f"{f'r in [{lo:.2f},{hi:.2f})':>14} {per['uniform'][m].mean():10.4f} "
              f"{per['gt_cell'][m].mean():10.4f} {per['gt_dist'][m].mean():10.4f}   n={int(m.sum())}")
    print("GT-cell is the peaked limit BCE converges to; GT-dist is the diffuse limit. A high GT-dist")
    print("in the near bin means the cone clips the supervision blob there - watch it as tau rises.")

    print("\n=== softmax temperature ===")
    print(f"{'tau':>6} {'top-1 prob':>12} {'eff. support':>14} {'L_OOC':>10}   per-bin L_OOC / eff. support")
    for tau in args.taus:
        o, s = per[f"ooc@{tau}"], per[f"sup@{tau}"]
        detail = "  ".join(
            f"[{lo:.2f},{hi:.2f}):{o[(r >= lo) & (r < hi)].mean():.3f}/{s[(r >= lo) & (r < hi)].mean():.0f}"
            for lo, hi in bins if ((r >= lo) & (r < hi)).any()
        )
        print(f"{tau:6.1f} {per[f'top1@{tau}'].mean():12.4f} {s.mean():14.1f} {o.mean():10.4f}   {detail}")
    print(f"(effective support is out of {cells} cells; 1 = one-hot, {cells} = uniform)")
    print("pick the smallest tau whose effective support is >~50 cells, then check that the near bin's")
    print("L_OOC has not drifted towards its GT-dist value - that would mean OOC is fighting the BCE target.")

    print("\n=== weight_ooc ===")
    hm_term = cfg.loss.weight_heatmap * mean(hm_losses)
    print(f"converged  {cfg.loss.weight_heatmap} * L_heat = {hm_term:.4f}")
    print(f"uniform-heatmap (init) L_OOC = {per['uniform'].mean():.4f}")
    print(f"\n{'tau':>6} {'L_OOC':>10} {'weight':>8} {'w*L_OOC':>10} {'share of total':>16}")
    for tau in args.taus:
        l = per[f"ooc@{tau}"].mean().item()
        for w in args.weights:
            print(f"{tau:6.1f} {l:10.4f} {w:8.1f} {w * l:10.4f} {w * l / (hm_term + w * l) * 100:15.2f}%")
    print("\naim for roughly 5-10% at convergence; the share is larger early in training, when L_OOC")
    print("is closer to the uniform-heatmap value printed above.")


if __name__ == "__main__":
    main()
