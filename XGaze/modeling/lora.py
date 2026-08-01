#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

# ****************************************************** #
#                          LORA                          #
# ****************************************************** #
# Low-Rank Adaptation (Hu et al., ICLR 2022): keep a pretrained projection frozen and learn a
# low-rank residual beside it. Used here to adapt the frozen DINOv3 image encoder without paying
# for full fine-tuning, and without the checkpoint bloat that comes with it - the adapter weights
# are a few million parameters against the backbone's 300M.

import math
import re

import torch
from torch import Tensor, nn

# Every trainable tensor this module introduces has this in its name. `XGazeModule` relies on it to
# tell adapter weights apart from the frozen backbone when stripping checkpoints, so do not rename
# the submodules below without updating that check.
LORA_MARKER = "lora_"

# Matches the layer index in a module path such as `model.layer.7.attention.q_proj`, so adaptation
# can be restricted to the last few blocks.
_LAYER_INDEX = re.compile(r"(?:^|\.)(?:layer|layers|blocks|block|h)\.(\d+)\.")


class LoRALinear(nn.Module):
    """
    A frozen `nn.Linear` with a trainable low-rank residual: `y = W x + (alpha / r) * B A x`.

    `B` is zero-initialised, so the wrapped layer starts out numerically identical to the original
    and the adaptation has to be learned rather than injected.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got r={r}.")

        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

    def extra_repr(self) -> str:
        return f"r={self.r}, scaling={self.scaling:.3f}"


def _matches(name: str, target_modules) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return any(t == leaf or name.endswith(f".{t}") for t in target_modules)


def apply_lora(
    root: nn.Module,
    target_modules,
    r: int,
    alpha: float,
    dropout: float = 0.0,
    last_n_layers: int = -1,
) -> tuple[int, int]:
    """
    Wrap every matching `nn.Linear` under `root` in a `LoRALinear`, in place.

    Arguments:
      root: the module to adapt (here, the frozen image-encoder backbone).
      target_modules: leaf names to adapt, eg. `["q_proj", "v_proj"]`.
      r: rank of the residual.
      alpha: scaling numerator; the residual is scaled by `alpha / r`.
      dropout: dropout applied to the adapter's input.
      last_n_layers: restrict adaptation to the last N transformer blocks, or -1 for all of them.
        Blocks are identified from the module path, so this silently applies to everything if the
        paths carry no recognisable layer index.

    Returns:
      tuple[int, int]: how many layers were wrapped, and how many trainable parameters were added.
    """
    targets = [
        name for name, module in root.named_modules()
        if isinstance(module, nn.Linear) and _matches(name, target_modules)
    ]
    if not targets:
        raise ValueError(
            f"No nn.Linear matched target_modules={list(target_modules)}. "
            f"Available leaf names include: "
            f"{sorted({n.rsplit('.', 1)[-1] for n, m in root.named_modules() if isinstance(m, nn.Linear)})}"
        )

    if last_n_layers is not None and last_n_layers > 0:
        indexed = [(name, _LAYER_INDEX.search(name)) for name in targets]
        depths = [int(m.group(1)) for _, m in indexed if m is not None]
        if depths:
            cutoff = max(depths) + 1 - last_n_layers
            targets = [name for name, m in indexed if m is not None and int(m.group(1)) >= cutoff]

    added = 0
    for name in targets:
        parent = root.get_submodule(name.rsplit(".", 1)[0]) if "." in name else root
        leaf = name.rsplit(".", 1)[-1]
        base = getattr(parent, leaf)
        wrapped = LoRALinear(base, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, leaf, wrapped)
        added += sum(p.numel() for p in (wrapped.lora_A.weight, wrapped.lora_B.weight))

    return len(targets), added


def is_lora_param(name: str) -> bool:
    """Whether a state-dict key belongs to a LoRA adapter rather than the frozen base weights."""
    return LORA_MARKER in name


def lora_parameters(module: nn.Module):
    """Iterate the adapter parameters under `module`, skipping the frozen base weights."""
    for name, param in module.named_parameters():
        if is_lora_param(name):
            yield name, param
