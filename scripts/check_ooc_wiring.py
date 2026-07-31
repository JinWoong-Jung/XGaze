"""Wiring checks for the out-of-cone (OOC) penalty inside `XGazeModule`.

Complements `check_ooc_loss.py`, which validates the loss function in isolation. This one drives
`compute_loss` / `_ooc_targets` / `_ooc_active` with the real Hydra configs for all three datasets,
using a lightweight stand-in for the module so that no model (and no GPU) is needed. It verifies
that the term is inert by default, that `log_ooc` measures without weighting, that `weight_ooc`
enters the total exactly once, and that the gradient reaches the heatmap logits.

    python scripts/check_ooc_wiring.py
"""

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra import compose, initialize_config_dir  # noqa: E402

from XGaze.modeling.xgaze import XGazeModule  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONF = str(REPO / "XGaze" / "conf")
SIZE = 64
failures = []


def check(name, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}{f'  ({detail})' if detail else ''}")
    if not ok:
        failures.append(name)


def make_stub(cfg, dataset):
    """A minimal object exposing what `compute_loss` actually touches: `cfg`, `dataset` and the two
    OOC helpers. Avoids instantiating the real model (which would download/load DINOv3)."""
    stub = types.SimpleNamespace(cfg=cfg, dataset=dataset)
    stub._ooc_active = XGazeModule._ooc_active.__get__(stub, types.SimpleNamespace)
    stub._ooc_targets = XGazeModule._ooc_targets.__get__(stub, types.SimpleNamespace)
    stub.compute_loss = XGazeModule.compute_loss.__get__(stub, types.SimpleNamespace)
    return stub


for config_name, dataset in [
    ("config_gazefollow", "gazefollow"),
    ("config_vat", "video_attention_target"),
    ("config_childplay", "childplay"),
]:
    with initialize_config_dir(config_dir=CONF, version_base="1.1"):
        cfg = compose(config_name=config_name)

    print(f"\n=== {config_name} (dataset={dataset}) ===")
    print(f"    weight_ooc={cfg.loss.weight_ooc}  log_ooc={cfg.loss.log_ooc}  ooc={dict(cfg.loss.ooc)}")
    check("config declares experiment.dataset consistently", cfg.experiment.dataset == dataset,
          f"cfg={cfg.experiment.dataset}")

    stub = make_stub(cfg, dataset)
    check("OOC is inert with shipped defaults", stub._ooc_active() is False)

    # Shapes follow `_forward_step`: gazefollow keeps only the last person, the others keep all.
    b, n = 4, (1 if dataset == "gazefollow" else 3)
    lead = (b,) if dataset == "gazefollow" else (b, n)
    batch = {
        "head_centers": torch.rand(b, n, 2) * 0.6 + 0.2,
        "gaze_pt": (torch.rand(b, 2) if dataset == "gazefollow" else torch.rand(b, n, 2)) * 0.6 + 0.2,
    }
    head_center, gaze_pt = stub._ooc_targets(batch)
    check("_ooc_targets aligns head_center with gaze_pt", head_center.shape == gaze_pt.shape,
          f"{tuple(head_center.shape)} vs {tuple(gaze_pt.shape)}")
    check("_ooc_targets matches the prediction leading dims", head_center.shape[:-1] == lead,
          f"{tuple(head_center.shape[:-1])} vs {lead}")

    hm_pred = torch.randn(*lead, SIZE, SIZE, requires_grad=True)
    hm_gt = torch.rand(*lead, SIZE, SIZE)
    gv_pred = torch.nn.functional.normalize(torch.randn(*lead, 2), dim=-1)
    gv_gt = torch.nn.functional.normalize(torch.randn(*lead, 2), dim=-1)
    io = torch.ones(*lead)
    io.view(-1)[0] = 0  # one out-of-frame entry, which every term must ignore
    io_pred = None if dataset == "gazefollow" else torch.randn(*lead)

    args = (hm_gt, gv_gt, io, hm_pred, gv_pred, io_pred, head_center, gaze_pt)

    _, off = stub.compute_loss(*args)
    check("default config skips the term entirely", off["ooc_loss"] == 0.0 and off["ooc_count"] == 0,
          f"ooc_loss={off['ooc_loss']}, count={off['ooc_count']}")

    cfg.loss.log_ooc = True
    _, logged = stub.compute_loss(*args)
    check("log_ooc computes a non-zero term", logged["ooc_loss"] > 0, f"L_OOC={logged['ooc_loss']:.6f}")
    check("log_ooc leaves total_loss untouched",
          abs(logged["total_loss"] - off["total_loss"]) < 1e-6,
          f"{logged['total_loss']:.6f} vs {off['total_loss']:.6f}")
    check("eligible count excludes the out-of-frame entry",
          logged["ooc_count"] <= int(io.sum().item()),
          f"count={logged['ooc_count']}, in-frame={int(io.sum().item())}")

    cfg.loss.log_ooc = False
    cfg.loss.weight_ooc = 3.0
    total_on, on = stub.compute_loss(*args)
    delta = on["total_loss"] - off["total_loss"]
    check("weight_ooc enters the total exactly once", abs(delta - 3 * logged["ooc_loss"]) < 1e-4,
          f"delta={delta:.6f}, expected={3 * logged['ooc_loss']:.6f}")

    hm_pred.grad = None
    total_on.backward()
    check("gradient reaches the heatmap logits", bool((hm_pred.grad != 0).any()),
          f"|grad|={hm_pred.grad.norm():.6f}")
    check("no gradient on out-of-frame entries",
          hm_pred.grad.reshape(-1, SIZE, SIZE)[0].abs().max() == 0.0)

    _, missing = stub.compute_loss(hm_gt, gv_gt, io, hm_pred.detach(), gv_pred, io_pred, None, None)
    check("missing targets degrade gracefully", missing["ooc_loss"] == 0.0)

    cfg.loss.weight_ooc = 0

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED: {failures}"))
sys.exit(1 if failures else 0)
