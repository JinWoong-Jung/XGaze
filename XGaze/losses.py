#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_dist_loss(gp_pred, gp_gt, io_gt):
    dist_loss = (gp_pred - gp_gt).pow(2).sum(dim=1)
    dist_loss = torch.mul(dist_loss, io_gt)
    dist_loss = torch.sum(dist_loss) / torch.sum(io_gt)
    return dist_loss


def compute_heatmap_loss(hm_pred, hm_gt, io_gt, loss_fn="bce"):
    # hm_pred/hm_gt: (..., h, w), io_gt: (...) matching the leading dims (works for both the
    # single-target (b, h, w) case and the multi-person (b, n, h, w) case).
    if loss_fn == "mse":
        heatmap_loss = F.mse_loss(hm_pred, hm_gt, reduction="none").mean([-2, -1])
    elif loss_fn == "bce":
        heatmap_loss = F.binary_cross_entropy_with_logits(hm_pred, hm_gt, reduction="none").mean([-2, -1])
    else:
        raise Exception("loss_fn should be either 'mse' or 'bce'.")
    heatmap_loss = torch.mul(heatmap_loss, io_gt)
    heatmap_loss = torch.sum(heatmap_loss) / torch.sum(io_gt)
    return heatmap_loss


def compute_angular_loss(gv_pred, gv_gt, io_gt):
    # gv_pred/gv_gt: (..., gaze_vec_dim), io_gt: (...) matching the leading dims.
    angular_loss = (1 - (gv_pred * gv_gt).sum(dim=-1)) / 2
    angular_loss = torch.mul(angular_loss, io_gt)
    angular_loss = torch.sum(angular_loss) / torch.sum(io_gt)
    return angular_loss


def build_gaze_cone_mask(
    head_center, gaze_pt, heatmap_size, theta_deg=60.0, alpha=30.0, normalize_alpha=True, eps=1e-6
):
    """
    Build the soft gaze-cone mask `C(p)` of Eq. (6) in "Enhancing Gaze Reasoning" (Wang et al.).

    The mask is derived purely from ground truth, so it carries no gradient: it is a constant
    weight map telling `compute_ooc_loss` which heatmap cells are gaze-consistent.

    Cell (i, j) maps to normalized coordinate (j / w, i / h) to match `generate_gaze_heatmap`,
    which places the ground-truth Gaussian at `gaze_pt * size` in `heatmap[y, x]` order.

    `normalize_alpha` divides the sharpness by the head-to-target distance `r`. With the paper's
    fixed `alpha`, the mask at the ground-truth location is `sigmoid(alpha*r*tan(theta/2)) *
    sigmoid(alpha*r)`, which decays badly for nearby targets - 0.58 at r=0.05 and 0.81 at r=0.10
    with alpha=30 - so a perfect prediction still gets penalised, and cells further along the gaze
    ray score *better* than the target itself. Measured on 6000 GazeFollow train rows, that leaves a
    residual penalty above 0.1 at the target cell for 9.9% of the samples that pass
    `min_gaze_dist=0.05` (11.6% of all rows before that filter), biasing predictions outward along
    the ray; raising `min_gaze_dist` far enough to remove it would discard ~14% of the training
    data, so it is not a usable remedy. Scaling `alpha` by `1/r` makes the mask at the
    target constant (~1.0) regardless of distance, leaving the cone boundary geometry untouched and
    only making the boundary sharpness relative to each sample's scale. Set it to False to
    reproduce the paper's formulation exactly.

    Args:
        head_center: normalized head centers `h`, shape (..., 2) in (x, y) order.
        gaze_pt: normalized gaze targets `g`, shape (..., 2) in (x, y) order.
        heatmap_size: (h, w) of the heatmap the mask is built for.
        theta_deg: cone aperture in degrees. Wider means a more permissive penalty.
        alpha: sigmoid sharpness of the cone boundary. Relative to the head-to-target distance when
            `normalize_alpha` is set, otherwise in absolute normalized-coordinate units.
        normalize_alpha: divide `alpha` by the head-to-target distance (see above).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: the cone mask of shape (..., h, w) with values in
        [0, 1], and the head-to-target distances of shape (...).
    """
    h, w = heatmap_size
    device, dtype = head_center.device, head_center.dtype

    xs = torch.arange(w, device=device, dtype=dtype) / w
    ys = torch.arange(h, device=device, dtype=dtype) / h
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")  # both (h, w)
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (h, w, 2), last dim is (x, y)

    delta = gaze_pt - head_center  # (..., 2)
    dist = delta.norm(p=2, dim=-1)  # (...)
    direction = delta / (dist.unsqueeze(-1) + eps)  # (..., 2)

    rel = grid - head_center[..., None, None, :]  # (..., h, w, 2)
    t = (rel * direction[..., None, None, :]).sum(dim=-1)  # (..., h, w), signed projection on the ray
    perp = (rel - t.unsqueeze(-1) * direction[..., None, None, :]).norm(p=2, dim=-1)  # (..., h, w)

    if normalize_alpha:
        # `dist` is clamped only to keep degenerate samples finite; they are dropped by
        # `min_gaze_dist` in `compute_ooc_loss` anyway.
        sharpness = (alpha / dist.clamp(min=1e-3))[..., None, None]
    else:
        sharpness = alpha

    half_angle = math.radians(theta_deg) / 2.0
    # First factor draws the cone boundary; second suppresses everything behind the head.
    cone = torch.sigmoid(sharpness * (t.clamp(min=0.0) * math.tan(half_angle) - perp)) * torch.sigmoid(sharpness * t)
    return cone, dist


def compute_ooc_loss(
    hm_logits,
    head_center,
    gaze_pt,
    io_gt,
    theta_deg=60.0,
    alpha=30.0,
    tau=1.0,
    min_gaze_dist=0.05,
    normalize_alpha=True,
    return_count=False,
):
    """
    Out-of-cone (OOC) penalty: the share of predicted probability mass that falls outside the
    ground-truth gaze cone (Eq. (7) of "Enhancing Gaze Reasoning").

    The heatmap logits are turned into a distribution with a spatial softmax, so the paper's
    normalizing denominator is exactly 1 and the loss reduces to `1 - E_{p~P}[C(p)]`, bounded in
    [0, 1]. Because `C` is constant, the gradient is
        dL/dz_k = (1/tau) * P_k * ((1 - C_k) - L),
    i.e. mass is pushed away from cells that are more out-of-cone than the current average, scaled
    by how much probability the model already puts there. It sums to zero over cells, so the
    penalty only redistributes mass spatially and leaves the absolute logit level to the BCE term.

    How much this competes with the heatmap loss depends on how peaked the prediction is, and the
    two ends of that range behave very differently at short gaze distances:

      * peaked (what BCE converges to - the ground-truth heatmap is normalized to max 1, so the
        optimal logit at the target is +inf and the spatial softmax tends to one-hot at the target
        cell): the loss tends to `1 - C(target cell)`, which `normalize_alpha` keeps at ~0 for every
        distance. No conflict.
      * diffuse (a prediction shaped like the ground-truth Gaussian itself): the Gaussian has a
        fixed width (sigma=3 cells, 0.047 in normalized units) while the cone half-width shrinks
        with distance as `r*tan(theta/2)`. They cross around r=0.16, below which the cone clips the
        supervision blob: with theta=60 and normalized alpha the loss is 0.285 at r=0.10 and 0.036
        at r=0.20 (0.417 and 0.135 with the paper's fixed alpha).

    Raising `tau` moves the prediction towards the diffuse end, so it trades OOC signal against this
    short-distance conflict. `scripts/probe_ooc_calibration.py` reports both per distance bin.

    Args:
        hm_logits: predicted heatmap logits, shape (..., h, w).
        head_center: normalized head centers, shape (..., 2) in (x, y) order.
        gaze_pt: normalized gaze targets, shape (..., 2) in (x, y) order.
        io_gt: in-frame mask matching the leading dims (...); only entries equal to 1 are supervised.
        theta_deg: cone aperture in degrees.
        alpha: sigmoid sharpness of the cone boundary.
        tau: spatial-softmax temperature. Values above 1 flatten `P`, which matters because the BCE
            term drives the logits towards a near one-hot softmax.
        min_gaze_dist: samples whose target is closer than this to the head center are skipped, as
            the gaze direction is numerically unstable there.
        normalize_alpha: see `build_gaze_cone_mask`.
        return_count: also return how many entries were eligible, which is the correct `batch_size`
            for logging this term (it is stricter than the in-frame count because of `min_gaze_dist`).

    Returns:
        torch.Tensor: scalar loss, or 0 when no sample in the batch is eligible. When `return_count`
        is set, a `(loss, num_eligible)` tuple instead.
    """
    cone, dist = build_gaze_cone_mask(
        head_center,
        gaze_pt,
        hm_logits.shape[-2:],
        theta_deg=theta_deg,
        alpha=alpha,
        normalize_alpha=normalize_alpha,
    )

    prob = torch.softmax((hm_logits / tau).flatten(start_dim=-2), dim=-1).view_as(hm_logits)
    ooc = (prob * (1.0 - cone)).sum(dim=(-2, -1))  # (...)

    mask = ((io_gt == 1) & (dist > min_gaze_dist)).to(ooc.dtype)
    count = int(mask.sum().item())
    if count == 0:
        loss = torch.zeros((), device=hm_logits.device, dtype=ooc.dtype)
    else:
        loss = torch.sum(ooc * mask) / torch.sum(mask)
    return (loss, count) if return_count else loss


def compute_inout_loss(io_pred, io_gt):
    """
    BCE loss for in/out-of-frame classification. `io_gt` may contain -1 for unknown/padding
    entries (eg. VAT/ChildPlay gaze classes ignored per the dataset README, or padded people
    slots) which are excluded from the loss.

    Args:
        io_pred: predicted in/out logits, any shape (...).
        io_gt: ground-truth in/out labels in {-1, 0, 1}, same shape as `io_pred`.
    """
    mask = (io_gt != -1).float()
    if torch.sum(mask) == 0:
        return torch.tensor(0.0, device=io_pred.device)
    inout_loss = F.binary_cross_entropy_with_logits(io_pred, io_gt.clamp(min=0).float(), reduction="none")
    inout_loss = torch.sum(inout_loss * mask) / torch.sum(mask)
    return inout_loss
