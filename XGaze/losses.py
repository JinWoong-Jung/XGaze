#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

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
