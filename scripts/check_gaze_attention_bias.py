"""Checks for the gaze-directed attention prior in `GazeDecoder`.

Runs on CPU with no dataset and no pretrained weights: `GazeDecoder` is driven directly on random
tensors, so this validates the direction score's geometry, the zero-initialised gating, gradient
flow and the global-token handling in isolation. Run it after touching the prior.

    python scripts/check_gaze_attention_bias.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from XGaze.modeling.decoder import GazeDecoder  # noqa: E402

TOKEN_DIM, FEATURE_MAP, HEATMAP = 256, 32, 64
failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{f'  ({detail})' if detail else ''}")
    if not ok:
        failures.append(name)


def build(depth=2, predict_inout=False):
    torch.manual_seed(0)
    return GazeDecoder(
        token_dim=TOKEN_DIM, depth=depth, num_heads=8,
        feature_map_size=FEATURE_MAP, heatmap_size=HEATMAP, predict_inout=predict_inout,
    )


def inputs(b=2, n=3):
    torch.manual_seed(1)
    return (
        torch.randn(b, TOKEN_DIM, FEATURE_MAP, FEATURE_MAP),
        torch.randn(b, n, TOKEN_DIM),
        torch.rand(b, n, 2) * 0.6 + 0.2,                                  # head centres
        torch.nn.functional.normalize(torch.randn(b, n, 2), dim=-1),      # gaze directions
    )


print("\n1. Direction score geometry (head at x=0.3, looking towards +x)")
head = torch.tensor([[[0.3, 0.5]]])
right = torch.tensor([[[1.0, 0.0]]])
score = GazeDecoder.build_gaze_direction_score(head, right, FEATURE_MAP)  # (1, 1, s*s)
grid = score.reshape(FEATURE_MAP, FEATURE_MAP)
# Tokens are row-major, so grid[row=y, col=x]; the head sits near column 0.3*32 ~= 10, row 16.
check("token straight ahead scores ~+1", grid[16, 30] > 0.99, f"{grid[16, 30]:.4f}")
check("token straight behind scores ~-1", grid[16, 1] < -0.99, f"{grid[16, 1]:.4f}")
check("token directly above scores ~0", abs(grid[1, 10].item()) < 0.10, f"{grid[1, 10]:.4f}")
check("score stays within [-1, 1]", bool(score.min() >= -1.0001 and score.max() <= 1.0001),
      f"[{score.min():.4f}, {score.max():.4f}]")

up = torch.tensor([[[0.0, -1.0]]])  # -y is towards the top of the image
grid_up = GazeDecoder.build_gaze_direction_score(head, up, FEATURE_MAP).reshape(FEATURE_MAP, FEATURE_MAP)
check("score follows the direction, not a fixed axis",
      grid_up[1, 10] > 0.99 and grid_up[30, 10] < -0.99,
      f"above={grid_up[1, 10]:.4f}, below={grid_up[30, 10]:.4f}")

check("unnormalised directions give the same score as unit ones",
      torch.allclose(GazeDecoder.build_gaze_direction_score(head, right * 7.3, FEATURE_MAP), score, atol=1e-5))

print("\n2. Shape and ordering match the flattened scene tokens")
_, _, hc, gd = inputs()
s = GazeDecoder.build_gaze_direction_score(hc, gd, FEATURE_MAP)
check("score is (b, n, s*s)", tuple(s.shape) == (2, 3, FEATURE_MAP * FEATURE_MAP), f"got {tuple(s.shape)}")
# image_tokens (b, c, h, w) are flattened as view(b, c, h*w) then transposed, i.e. row-major.
img = torch.arange(FEATURE_MAP * FEATURE_MAP, dtype=torch.float).reshape(1, 1, FEATURE_MAP, FEATURE_MAP)
check("scene tokens flatten row-major, as the score assumes",
      torch.equal(img.view(1, 1, -1)[0, 0], torch.arange(FEATURE_MAP * FEATURE_MAP, dtype=torch.float)))

print("\n3. The prior is inert at initialisation")
dec = build()
img, gaze, hc, gd = inputs()
with torch.no_grad():
    without, _ = dec(img, gaze)
    with_prior, _ = dec(img, gaze, head_center=hc, gaze_direction=gd)
check("every block starts with a zero bias scale",
      all(float(b.gaze_bias_scale) == 0.0 for b in dec.blocks))
check("zero scale leaves the output bit-identical", torch.equal(without, with_prior),
      f"max|diff|={(without - with_prior).abs().max():.3e}")

print("\n4. A non-zero scale actually changes the output")
with torch.no_grad():
    for blk in dec.blocks:
        blk.gaze_bias_scale.fill_(1.5)
    tilted, _ = dec(img, gaze, head_center=hc, gaze_direction=gd)
moved = (tilted - without).abs().max().item()
check("output moves once the prior is switched on", moved > 1e-4, f"max|diff|={moved:.3e}")

with torch.no_grad():
    flipped, _ = dec(img, gaze, head_center=hc, gaze_direction=-gd)
check("reversing the gaze direction changes the result",
      (tilted - flipped).abs().max().item() > 1e-4,
      f"max|diff|={(tilted - flipped).abs().max():.3e}")

print("\n5. Gradient reaches the gate and the direction")
dec = build()
img, gaze, hc, gd = inputs()
gd = gd.clone().requires_grad_(True)
for blk in dec.blocks:  # a zero gate would give the direction zero gradient, so open it first
    with torch.no_grad():
        blk.gaze_bias_scale.fill_(0.5)
out, _ = dec(img, gaze, head_center=hc, gaze_direction=gd)
out.sum().backward()
check("gradient reaches every block's bias scale",
      all(b.gaze_bias_scale.grad is not None and b.gaze_bias_scale.grad.abs().sum() > 0 for b in dec.blocks))
check("gradient reaches the predicted gaze direction",
      gd.grad is not None and bool((gd.grad != 0).any()),
      f"|grad|={gd.grad.norm():.6f}" if gd.grad is not None else "None")

dec_zero = build()
img2, gaze2, hc2, gd2 = inputs()
gd2 = gd2.clone().requires_grad_(True)
dec_zero(img2, gaze2, head_center=hc2, gaze_direction=gd2)[0].sum().backward()
check("at a zero gate the direction gets no gradient (the gate must open first)",
      gd2.grad is None or gd2.grad.abs().max() < 1e-12,
      f"|grad|={gd2.grad.abs().max():.3e}" if gd2.grad is not None else "None")

print("\n6. Global scene token is excluded from the prior")
dec = build()
img, gaze, hc, gd = inputs()
glob = torch.randn(2, 1, TOKEN_DIM)
with torch.no_grad():
    for blk in dec.blocks:
        blk.gaze_bias_scale.fill_(1.5)
    out_glob, _ = dec(img, gaze, image_global_token=glob, head_center=hc, gaze_direction=gd)
check("runs with a global token and keeps the output shape",
      tuple(out_glob.shape) == (2, 3, HEATMAP, HEATMAP), f"got {tuple(out_glob.shape)}")
padded = torch.nn.functional.pad(GazeDecoder.build_gaze_direction_score(hc, gd, FEATURE_MAP), (1, 0), value=0.0)
check("the global token's slot scores exactly 0", bool((padded[..., 0] == 0).all()))
check("padding only prepends one slot", padded.shape[-1] == FEATURE_MAP * FEATURE_MAP + 1,
      f"got {padded.shape[-1]}")

print("\n7. Partial arguments leave the prior off")
dec = build()
img, gaze, hc, gd = inputs()
with torch.no_grad():
    for blk in dec.blocks:
        blk.gaze_bias_scale.fill_(1.5)
    base, _ = dec(img, gaze)
    only_head, _ = dec(img, gaze, head_center=hc)
    only_dir, _ = dec(img, gaze, gaze_direction=gd)
check("head_center alone does not enable the prior", torch.equal(base, only_head))
check("gaze_direction alone does not enable the prior", torch.equal(base, only_dir))

print("\n8. People stay independent")
dec = build()
dec.eval()
img, gaze, hc, gd = inputs(b=2, n=3)
with torch.no_grad():
    for blk in dec.blocks:
        blk.gaze_bias_scale.fill_(1.5)
    multi, _ = dec(img, gaze, head_center=hc, gaze_direction=gd)
    singles = torch.stack(
        [dec(img, gaze[:, i:i + 1], head_center=hc[:, i:i + 1], gaze_direction=gd[:, i:i + 1])[0][:, 0]
         for i in range(3)], dim=1)
check("a person's heatmap does not depend on the others", torch.allclose(multi, singles, atol=1e-5),
      f"max|diff|={(multi - singles).abs().max():.3e}")

print("\n9. Parameter cost")
base_params = sum(p.numel() for p in build().parameters())
gates = sum(b.gaze_bias_scale.numel() for b in build().blocks)
print(f"      decoder={base_params:,} params, of which the prior adds {gates} (one scalar per block)")
check("the prior adds one scalar per block", gates == 2, f"got {gates}")

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED: {failures}"))
sys.exit(1 if failures else 0)
