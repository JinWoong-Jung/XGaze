"""TEMPORARY: decompose GazeFollow test error into along-ray and perpendicular components.

The tail split showed that 71% of the worst 25% of images have the gaze direction roughly right
(<=30deg) but still land far from the target. That is consistent with a depth/where-along-the-ray
failure, but "consistent with" is not evidence: this measures the two components directly, and in
particular whether the model systematically stops short of the target or overshoots it.

    python _tmp_ray.py --ckpt checkpoints/GF_layer-3_dim-768.ckpt --device cuda:1
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydra import compose, initialize_config_dir  # noqa: E402

from XGaze.datasets.gazefollow import GazeFollowDataModule  # noqa: E402
from XGaze.modeling.xgaze import XGazeModule  # noqa: E402
from XGaze.utils.common import dark_coordinate_decoding  # noqa: E402

REPO = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/GF_layer-3_dim-768.ckpt")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def med(xs):
    s = sorted(x for x in xs if x == x)
    return s[len(s) // 2] if s else float("nan")


def main():
    args = parse_args()
    with initialize_config_dir(config_dir=str(REPO / "XGaze" / "conf"), version_base="1.1"):
        cfg = compose(config_name="config_gazefollow", overrides=[
            "model.XGaze.token_dim=768", "model.XGaze.decoder_depth=3"])
    module = XGazeModule(cfg)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    module.on_load_checkpoint(ck)
    module.load_state_dict(ck["state_dict"], strict=False)
    module.to(args.device).eval()

    dm = GazeFollowDataModule(
        root=cfg.data.gf.root, root_project=cfg.project.root, root_heads=cfg.data.gf.root_heads,
        batch_size=args.batch_size, image_size=cfg.data.image_size, heatmap_size=cfg.data.heatmap_size,
        heatmap_sigma=cfg.data.heatmap_sigma, num_people=cfg.data.num_people,
        return_head_mask=cfg.data.return_head_mask, num_workers=4)
    dm.setup("test")

    rows = []
    with torch.no_grad():
        for batch in dm.test_dataloader():
            bt = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            hm, _, _, _ = module._forward_step(bt)
            prob = hm.sigmoid()
            pred = dark_coordinate_decoding(prob.reshape(-1, *prob.shape[-2:]),
                                            kernel_size=cfg.data.heatmap_sigma * 3,
                                            normalize=True).reshape(*prob.shape[:-2], 2)
            head = bt["head_centers"][:, -1, :]
            for j in range(len(batch["path"])):
                gt = bt["gaze_pt"][j]
                gt = gt[gt[:, 0] != -1]
                g, h, p = gt.mean(0), head[j], pred[j]
                r = (g - h).norm()
                if r < 1e-6:
                    continue
                d = (g - h) / r                      # unit vector along the ground-truth ray
                e = p - g                            # error vector, from target to prediction
                along = (e * d).sum()                # signed: negative = stopped short of the target
                perp = (e - along * d).norm()
                ang = torch.acos((((p - h) / (p - h).norm().clamp(min=1e-8)) * d).sum().clamp(-1, 1)).rad2deg()
                rows.append({
                    "err": e.norm().item(), "along": along.item(), "perp": perp.item(),
                    "r": r.item(), "ang": ang.item(),
                    "pred_r": (p - h).norm().item(),
                    "spread": gt.std(0).norm().item() if len(gt) > 1 else 0.0,
                })

    n = len(rows)
    rows.sort(key=lambda x: x["err"])
    easy, tail = rows[:n // 2], rows[int(n * 0.75):]
    dirfail = [x for x in tail if x["ang"] > 30]
    ray = [x for x in tail if x["ang"] <= 30]

    print(f"\n=== error decomposition, {n} images ===")
    print(f"{'group':>34} {'n':>5} {'|err|':>8} {'along':>9} {'|along|':>9} {'perp':>8} {'r(gt)':>8} {'r(pred)':>9}")
    for nm, g in (("all", rows), ("easy half", easy), ("worst 25%", tail),
                  ("  of which: direction failed", dirfail), ("  of which: DIRECTION OK", ray)):
        print(f"{nm:>34} {len(g):5d} {mean([x['err'] for x in g]):8.4f} {mean([x['along'] for x in g]):+9.4f} "
              f"{mean([abs(x['along']) for x in g]):9.4f} {mean([x['perp'] for x in g]):8.4f} "
              f"{mean([x['r'] for x in g]):8.4f} {mean([x['pred_r'] for x in g]):9.4f}")
    print("\n  'along' is signed: negative means the prediction stopped short of the target,")
    print("  positive means it went past. 'perp' is the sideways miss. r is head-to-target distance.")

    print(f"\n=== is the error mostly along the ray or sideways? ===")
    for nm, g in (("easy half", easy), ("worst 25%", tail), ("  direction OK subset", ray)):
        a, p = mean([abs(x["along"]) for x in g]), mean([x["perp"] for x in g])
        print(f"  {nm:>22}: |along|={a:.4f}  perp={p:.4f}   along/perp = {a / p:.2f}")

    print(f"\n=== does the model stop short or overshoot? (direction-OK subset) ===")
    short = sum(1 for x in ray if x["along"] < 0)
    print(f"  stopped short: {short}/{len(ray)} ({short / len(ray) * 100:.1f}%)   "
          f"mean signed along = {mean([x['along'] for x in ray]):+.4f}")
    print(f"  predicted distance from head: {mean([x['pred_r'] for x in ray]):.4f} "
          f"vs true {mean([x['r'] for x in ray]):.4f}  "
          f"(ratio {mean([x['pred_r'] for x in ray]) / mean([x['r'] for x in ray]):.3f})")

    print(f"\n=== by head-to-target distance (all images) ===")
    print(f"  {'r bin':>14} {'n':>6} {'|err|':>8} {'along(signed)':>14} {'perp':>8} {'r(pred)/r(gt)':>14}")
    for lo, hi in ((0, 0.15), (0.15, 0.25), (0.25, 0.40), (0.40, 0.60), (0.60, 2.0)):
        s = [x for x in rows if lo <= x["r"] < hi]
        if s:
            print(f"  [{lo:.2f},{hi:.2f}) {len(s):6d} {mean([x['err'] for x in s]):8.4f} "
                  f"{mean([x['along'] for x in s]):+14.4f} {mean([x['perp'] for x in s]):8.4f} "
                  f"{mean([x['pred_r'] for x in s]) / mean([x['r'] for x in s]):14.3f}")

    print(f"\n=== how far along the ray does the model tend to stop? ===")
    frac = [x["pred_r"] / x["r"] for x in rows if x["r"] > 0.05]
    print(f"  pred_r / r  over {len(frac)} images:  median={med(frac):.3f}  mean={mean(frac):.3f}")
    for q in (10, 25, 50, 75, 90):
        s = sorted(frac)
        print(f"    p{q:<3d} {s[int(q / 100 * (len(s) - 1))]:.3f}")


if __name__ == "__main__":
    main()
