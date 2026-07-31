"""Correctness checks for the out-of-cone (OOC) penalty.

Runs on CPU with no dataset and no model: every input is synthetic, so this validates the geometry,
the coordinate convention, the gradient, and the masking in isolation. Run it after touching
`build_gaze_cone_mask` / `compute_ooc_loss`.

    python scripts/check_ooc_loss.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from XGaze.losses import build_gaze_cone_mask, compute_ooc_loss  # noqa: E402
from XGaze.utils.common import generate_gaze_heatmap  # noqa: E402

SIZE = 64
THETA, ALPHA = 60.0, 30.0
failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{f'  ({detail})' if detail else ''}")
    if not ok:
        failures.append(name)


def as_logits(prob):
    """Turn a probability map into logits whose spatial softmax reproduces it (at tau=1)."""
    return torch.log(prob / prob.sum(dim=(-2, -1), keepdim=True) + 1e-12)


def gaussian_prob(gaze_pt):
    return generate_gaze_heatmap(gaze_pt, sigma=3, size=SIZE)


print("\n1. Coordinate convention (head at x=0.3, gazing towards +x)")
h = torch.tensor([[0.3, 0.5]])
g = torch.tensor([[0.8, 0.5]])
cone, dist = build_gaze_cone_mask(h, g, (SIZE, SIZE), theta_deg=THETA, alpha=ALPHA)
cone = cone[0]
# heatmap is indexed [y, x]; +x is to the right of the head, so high mask values must be at high columns.
check("on-axis ahead of head is inside the cone", cone[32, 50] > 0.9, f"C[32,50]={cone[32, 50]:.4f}")
check("behind the head is outside the cone", cone[32, 5] < 0.05, f"C[32,5]={cone[32, 5]:.4f}")
check("far off-axis is outside the cone", cone[2, 50] < 0.05, f"C[2,50]={cone[2, 50]:.4f}")
check("head-to-target distance is correct", abs(dist.item() - 0.5) < 1e-6, f"dist={dist.item():.6f}")

# The mask must follow the direction, not a fixed axis: rotate the target and re-check.
cone_up = build_gaze_cone_mask(h, torch.tensor([[0.3, 0.1]]), (SIZE, SIZE), theta_deg=THETA, alpha=ALPHA)[0][0]
check("gazing towards -y puts the cone above the head", cone_up[10, 19] > 0.9 and cone_up[50, 19] < 0.05,
      f"C[10,19]={cone_up[10, 19]:.4f}, C[50,19]={cone_up[50, 19]:.4f}")

print("\n2. The three reference distributions")
# Head at the image centre so that the mirrored target still lands inside the frame.
io = torch.ones(1)
hc = torch.tensor([[0.5, 0.5]])
R = 0.35
gc = torch.tensor([[0.5 + R, 0.5]])
cone_c = build_gaze_cone_mask(hc, gc, (SIZE, SIZE), theta_deg=THETA, alpha=ALPHA)[0][0]

gt = gaussian_prob(gc[0]).unsqueeze(0)
l_gt = compute_ooc_loss(as_logits(gt), hc, gc, io, theta_deg=THETA, alpha=ALPHA)
check("ground-truth heatmap gives ~0 loss", l_gt.item() < 0.02, f"L_OOC={l_gt.item():.5f}")

uniform_logits = torch.zeros(1, SIZE, SIZE)
l_uniform = compute_ooc_loss(uniform_logits, hc, gc, io, theta_deg=THETA, alpha=ALPHA)
cone_area = cone_c.mean().item()
check("uniform heatmap gives 1 - cone area", abs(l_uniform.item() - (1 - cone_area)) < 1e-5,
      f"L_OOC={l_uniform.item():.5f}, cone area={cone_area:.5f}")

mirrored = gaussian_prob((2 * hc - gc)[0]).unsqueeze(0)  # reflect the target through the head centre
l_mirror = compute_ooc_loss(as_logits(mirrored), hc, gc, io, theta_deg=THETA, alpha=ALPHA)
check("diametrically opposite heatmap gives ~1 loss", l_mirror.item() > 0.98, f"L_OOC={l_mirror.item():.5f}")

print("\n3. Loss is bounded and monotone in angular offset")
offsets, losses = [0, 15, 30, 45, 60, 90, 180], []
for deg in offsets:
    rad = math.radians(deg)
    off = torch.tensor([[0.5 + R * math.cos(rad), 0.5 + R * math.sin(rad)]])
    losses.append(compute_ooc_loss(as_logits(gaussian_prob(off[0]).unsqueeze(0)), hc, gc, io,
                                   theta_deg=THETA, alpha=ALPHA).item())
print("      " + "  ".join(f"{d}deg={v:.3f}" for d, v in zip(offsets, losses)))
check("loss increases monotonically with angular offset", all(a <= b + 1e-6 for a, b in zip(losses, losses[1:])))
check("loss stays within [0, 1]", all(0.0 <= v <= 1.0 for v in losses))
check("predictions inside the cone are barely penalised", losses[1] < 0.1, f"15deg -> {losses[1]:.4f}")

print("\n4. Gradient identity  dL/dz_k = (1/tau) * P_k * ((1 - C_k) - L)")
tau = 2.0
z = torch.randn(3, SIZE, SIZE, requires_grad=True)
hh = torch.tensor([[0.3, 0.5], [0.6, 0.2], [0.5, 0.5]])
gg = torch.tensor([[0.8, 0.5], [0.2, 0.9], [0.9, 0.1]])
loss = compute_ooc_loss(z, hh, gg, torch.ones(3), theta_deg=THETA, alpha=ALPHA, tau=tau)
loss.backward()
with torch.no_grad():
    c, _ = build_gaze_cone_mask(hh, gg, (SIZE, SIZE), theta_deg=THETA, alpha=ALPHA)
    p = torch.softmax((z / tau).flatten(start_dim=-2), dim=-1).view_as(z)
    per_sample = (p * (1 - c)).sum(dim=(-2, -1))
    expected = (p * ((1 - c) - per_sample[:, None, None])) / tau / 3  # /3 from the mean over the batch
check("autograd matches the analytic gradient", torch.allclose(z.grad, expected, atol=1e-6),
      f"max|diff|={(z.grad - expected).abs().max():.3e}")
check("gradient sums to zero per sample (pure redistribution)", z.grad.sum(dim=(-2, -1)).abs().max() < 1e-7,
      f"max|sum|={z.grad.sum(dim=(-2, -1)).abs().max():.3e}")

print("\n5. Masking")
z2 = torch.randn(4, SIZE, SIZE)
h2 = torch.tensor([[0.3, 0.5]] * 4)
g2 = torch.tensor([[0.8, 0.5]] * 4)
check("all-outside batch returns exactly 0",
      compute_ooc_loss(z2, h2, g2, torch.zeros(4), theta_deg=THETA, alpha=ALPHA).item() == 0.0)
check("-1 (unknown/padded) is excluded like 0",
      compute_ooc_loss(z2, h2, g2, -torch.ones(4), theta_deg=THETA, alpha=ALPHA).item() == 0.0)

mixed = compute_ooc_loss(z2, h2, g2, torch.tensor([1.0, 0.0, 1.0, -1.0]), theta_deg=THETA, alpha=ALPHA)
only_valid = compute_ooc_loss(z2[[0, 2]], h2[[0, 2]], g2[[0, 2]], torch.ones(2), theta_deg=THETA, alpha=ALPHA)
check("masked batch equals the mean over valid entries only", torch.allclose(mixed, only_valid, atol=1e-6),
      f"{mixed.item():.6f} vs {only_valid.item():.6f}")

degenerate = compute_ooc_loss(z2, h2, h2 + 0.01, torch.ones(4), theta_deg=THETA, alpha=ALPHA, min_gaze_dist=0.05)
check("targets closer than min_gaze_dist are skipped", degenerate.item() == 0.0, f"L_OOC={degenerate.item():.6f}")

print("\n6. Multi-person shape (b, n, h, w), as used by VAT/ChildPlay")
z3 = torch.randn(2, 3, SIZE, SIZE)
h3 = torch.rand(2, 3, 2) * 0.6 + 0.2
g3 = torch.rand(2, 3, 2) * 0.6 + 0.2
io3 = torch.tensor([[1.0, 0.0, -1.0], [1.0, 1.0, 1.0]])
multi = compute_ooc_loss(z3, h3, g3, io3, theta_deg=THETA, alpha=ALPHA)
check("multi-person batch produces a finite scalar", multi.ndim == 0 and torch.isfinite(multi),
      f"L_OOC={multi.item():.5f}")

flat = compute_ooc_loss(z3.reshape(6, SIZE, SIZE), h3.reshape(6, 2), g3.reshape(6, 2), io3.reshape(6),
                        theta_deg=THETA, alpha=ALPHA)
check("(b, n) and flattened batches agree", torch.allclose(multi, flat, atol=1e-6),
      f"{multi.item():.6f} vs {flat.item():.6f}")

print("\n7. Distance invariance of the mask at the ground truth (regression: the outward-pull bias)")
# With the paper's fixed alpha, C at the target decays for nearby targets (0.58 at r=0.05), so a
# correct prediction is still penalised and cells further along the ray score better than the target.
# `normalize_alpha` must remove that. Verified in continuous coordinates to avoid cell quantisation.
print(f"{'r':>7} {'C(GT) normalized':>18} {'C(GT) paper form':>18}")
c_norm, c_paper = [], []
for r in (0.05, 0.10, 0.20, 0.35, 0.60):
    hr, gr = torch.tensor([[0.2, 0.5]]), torch.tensor([[0.2 + r, 0.5]])
    t_gt = torch.tensor(r)
    for flag, acc in ((True, c_norm), (False, c_paper)):
        sharp = ALPHA / max(r, 1e-3) if flag else ALPHA
        acc.append((torch.sigmoid(sharp * t_gt * math.tan(math.radians(THETA) / 2))
                    * torch.sigmoid(sharp * t_gt)).item())
    print(f"{r:7.2f} {c_norm[-1]:18.4f} {c_paper[-1]:18.4f}")
check("normalized alpha keeps C(GT) ~1 at every distance", min(c_norm) > 0.99,
      f"min={min(c_norm):.4f}")
check("normalized alpha makes C(GT) distance-invariant", max(c_norm) - min(c_norm) < 0.01,
      f"spread={max(c_norm) - min(c_norm):.5f}")
check("the paper form is the biased one this guards against", min(c_paper) < 0.7,
      f"min={min(c_paper):.4f} at r=0.05")

# The target must also be the best point along its own gaze ray, not merely a good one.
hr, gr = torch.tensor([[0.2, 0.5]]), torch.tensor([[0.35, 0.5]])  # r = 0.15
for flag, label in ((True, "normalized"), (False, "paper form")):
    cone_r, _ = build_gaze_cone_mask(hr, gr, (SIZE, SIZE), theta_deg=THETA, alpha=ALPHA, normalize_alpha=flag)
    row = cone_r[0, 32]  # the gaze ray runs along this row
    gt_col = int(0.35 * SIZE)
    gap = row.max().item() - row[gt_col].item()
    print(f"      {label:11s}: C at GT cell={row[gt_col]:.4f}, max along ray={row.max():.4f}, gap={gap:.4f}")
    if flag:
        check("target is (near) optimal along its own gaze ray", gap < 0.01, f"gap={gap:.5f}")

print("\n8. Conflict with the heatmap loss, at both ends of the peakedness range")
# "C(target cell) is ~1" does NOT imply "a ground-truth-shaped prediction scores ~0": the GT Gaussian
# has a fixed width (sigma=3 cells) while the cone half-width shrinks as r*tan(theta/2), so below
# r~0.16 the cone clips the supervision blob. The peaked end (what BCE converges to) is unaffected.
# tau moves the model between the two ends, so both bounds are pinned here.
print(f"{'r':>7} {'diffuse (P~Gauss)':>19} {'peaked (target cell)':>21} {'cone half-width':>17}")
diffuse, peaked = {}, {}
for r in (0.10, 0.126, 0.20, 0.35):
    hr, gr = torch.tensor([[0.2, 0.5]]), torch.tensor([[0.2 + r, 0.5]])
    prob = gaussian_prob(gr[0]).unsqueeze(0)
    diffuse[r] = compute_ooc_loss(as_logits(prob), hr, gr, torch.ones(1), theta_deg=THETA, alpha=ALPHA).item()
    cone_r, _ = build_gaze_cone_mask(hr, gr, (SIZE, SIZE), theta_deg=THETA, alpha=ALPHA)
    peaked[r] = 1 - cone_r[0, int(0.5 * SIZE), int((0.2 + r) * SIZE)].item()
    print(f"{r:7.3f} {diffuse[r]:19.4f} {peaked[r]:21.6f} {r * math.tan(math.radians(THETA) / 2):17.4f}")
print(f"      (GT Gaussian sigma = 3 cells = {3 / SIZE:.4f}; the cone is narrower than 2 sigma below r~0.16)")

check("peaked end is ~0 at every distance", max(peaked.values()) < 1e-3,
      f"max={max(peaked.values()):.2e}")
check("diffuse end is negligible at typical distances (r>=0.20)",
      diffuse[0.20] < 0.05 and diffuse[0.35] < 0.01,
      f"r=0.20 -> {diffuse[0.20]:.4f}, r=0.35 -> {diffuse[0.35]:.4f}")
check("diffuse end decreases monotonically with distance",
      all(diffuse[a] >= diffuse[b] for a, b in zip([0.10, 0.126, 0.20], [0.126, 0.20, 0.35])))
# Pin the known short-distance conflict so a change in theta/alpha/sigma cannot silently worsen it.
check("short-distance diffuse conflict stays within the measured envelope",
      0.20 < diffuse[0.10] < 0.35 and 0.12 < diffuse[0.126] < 0.25,
      f"r=0.10 -> {diffuse[0.10]:.4f}, r=0.126 -> {diffuse[0.126]:.4f}")

print("\n9. Cone area vs aperture (the uniform-heatmap baseline used to pick weight_ooc)")
for theta in (45.0, 60.0, 90.0):
    c, _ = build_gaze_cone_mask(h, g, (SIZE, SIZE), theta_deg=theta, alpha=ALPHA)
    print(f"      theta={theta:5.1f}deg -> cone area={c.mean().item():.4f}, uniform L_OOC={1 - c.mean().item():.4f}")

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED: {failures}"))
sys.exit(1 if failures else 0)
