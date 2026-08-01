"""Checks for the LoRA adaptation of the frozen image encoder.

Runs on CPU with no dataset and no pretrained weights: `apply_lora` is exercised on a small stand-in
transformer, and the checkpoint-hook interaction is tested on synthetic state dicts. Two behaviours
matter most and are easy to break silently:
  - the adapters must survive `on_save_checkpoint`, which strips everything under the backbone
    prefix to keep checkpoints small;
  - `on_load_checkpoint` re-injects the pretrained backbone and must not overwrite the checkpoint's
    trained adapters with freshly initialised ones.

    python scripts/check_lora.py
"""

import sys
import types
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra import compose, initialize_config_dir  # noqa: E402

from XGaze.modeling.lora import LoRALinear, apply_lora, is_lora_param  # noqa: E402
from XGaze.modeling.xgaze import XGazeModule  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONF = str(REPO / "XGaze" / "conf")
failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{f'  ({detail})' if detail else ''}")
    if not ok:
        failures.append(name)


class Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class Backbone(nn.Module):
    """Mirrors the DINOv3 path layout (`model.layer.<i>.attention.<proj>`)."""

    def __init__(self, d=32, depth=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layer = nn.ModuleList([Attn(d) for _ in range(depth)])

    def forward(self, x):
        for blk in self.model.layer:
            x = blk(x)
        return x


print("\n1. LoRALinear starts as a no-op and is shaped correctly")
base = nn.Linear(16, 24)
lora = LoRALinear(base, r=4, alpha=8)
x = torch.randn(3, 16)
check("wrapped layer initially matches the original", torch.allclose(lora(x), base(x), atol=1e-6),
      f"max|diff|={(lora(x) - base(x)).abs().max():.2e}")
check("lora_B is zero-initialised", bool((lora.lora_B.weight == 0).all()))
check("lora_A is not zero-initialised", bool((lora.lora_A.weight != 0).any()))
check("scaling is alpha / r", abs(lora.scaling - 2.0) < 1e-9, f"{lora.scaling}")
with torch.no_grad():
    lora.lora_B.weight.normal_()
check("output moves once lora_B is non-zero", not torch.allclose(lora(x), base(x), atol=1e-6))
try:
    LoRALinear(nn.Linear(4, 4), r=0, alpha=1)
    check("rank r=0 is rejected", False, "no exception")
except ValueError:
    check("rank r=0 is rejected", True)


print("\n2. Base weights stay frozen, adapters stay trainable")
bb = Backbone()
for p in bb.parameters():  # emulate freeze() running before apply_lora
    p.requires_grad = False
n_wrapped, n_added = apply_lora(bb, ["q_proj", "v_proj"], r=4, alpha=8)
check("wrapped one module per target per layer", n_wrapped == 8, f"{n_wrapped} (expect 4 layers x 2)")
trainable = [n for n, p in bb.named_parameters() if p.requires_grad]
check("only adapters are trainable", trainable and all(is_lora_param(n) for n in trainable),
      f"{len(trainable)} trainable, all lora={all(is_lora_param(n) for n in trainable)}")
check("reported parameter count matches", n_added == sum(p.numel() for n, p in bb.named_parameters() if is_lora_param(n)),
      f"{n_added}")
check("untargeted projections are untouched", isinstance(bb.model.layer[0].k_proj, nn.Linear)
      and not isinstance(bb.model.layer[0].k_proj, LoRALinear))
check("targeted projections are wrapped", isinstance(bb.model.layer[0].q_proj, LoRALinear))
check("the adapted backbone still runs", tuple(bb(torch.randn(2, 5, 32)).shape) == (2, 5, 32))

print("\n3. last_n_layers restricts the adaptation depth")
bb2 = Backbone()
n2, _ = apply_lora(bb2, ["q_proj"], r=4, alpha=8, last_n_layers=2)
check("only the last N blocks are wrapped", n2 == 2, f"{n2}")
check("early blocks untouched", not isinstance(bb2.model.layer[0].q_proj, LoRALinear))
check("late blocks wrapped", isinstance(bb2.model.layer[3].q_proj, LoRALinear))

print("\n4. A bad target name fails loudly instead of silently doing nothing")
try:
    apply_lora(Backbone(), ["not_a_real_proj"], r=4, alpha=8)
    check("unmatched target_modules raises ValueError", False, "no exception")
except ValueError as e:
    check("unmatched target_modules raises ValueError", "No nn.Linear matched" in str(e))

print("\n5. Checkpoint hooks preserve adapters (the silent-loss hazard)")
with initialize_config_dir(config_dir=CONF, version_base="1.1"):
    cfg = compose(config_name="config_gazefollow")
prefix = "model.image_encoder.backbone."
stub = types.SimpleNamespace(cfg=cfg)
strip = XGazeModule._strip_dino_backbone.__get__(stub, types.SimpleNamespace)

state = OrderedDict([
    (f"{prefix}model.layer.0.attention.q_proj.base.weight", torch.zeros(2)),
    (f"{prefix}model.layer.0.attention.q_proj.lora_A.weight", torch.ones(2)),
    (f"{prefix}model.layer.0.attention.q_proj.lora_B.weight", torch.ones(2) * 2),
    (f"{prefix}model.layer.0.attention.k_proj.weight", torch.zeros(2)),
    ("model.gaze_decoder.query_tokens", torch.ones(2)),
])
kept = strip(state, prefix)
check("frozen backbone weights are stripped",
      not any(k.endswith("q_proj.base.weight") or k.endswith("k_proj.weight") for k in kept))
check("adapter weights survive the strip",
      sum(1 for k in kept if is_lora_param(k)) == 2, f"{sorted(k.split('.')[-2] for k in kept if is_lora_param(k))}")
check("modules outside the backbone are untouched", "model.gaze_decoder.query_tokens" in kept)

print("\n6. Reloading keeps the checkpoint's adapters, not the fresh ones")
# on_load_checkpoint re-injects the live backbone state; if it included adapters it would clobber
# the trained ones sitting in the checkpoint.
live = OrderedDict([
    ("model.layer.0.attention.q_proj.base.weight", torch.zeros(2)),
    ("model.layer.0.attention.q_proj.lora_A.weight", torch.full((2,), 99.0)),  # freshly initialised
])
reinjected = OrderedDict(
    (f"{prefix}{n}", v) for n, v in live.items() if not is_lora_param(n)
)
ckpt_state = OrderedDict([(f"{prefix}model.layer.0.attention.q_proj.lora_A.weight", torch.ones(2))])
ckpt_state.update(reinjected)
check("re-injection excludes adapters",
      not any(is_lora_param(k) for k in reinjected), f"{list(reinjected)}")
check("the checkpoint's trained adapter is preserved",
      bool((ckpt_state[f"{prefix}model.layer.0.attention.q_proj.lora_A.weight"] == 1).all()),
      "would be 99 if clobbered")
check("frozen weights are still restored",
      f"{prefix}model.layer.0.attention.q_proj.base.weight" in ckpt_state)

print("\n7. Config wiring")
for name in ("config_gazefollow", "config_vat", "config_childplay"):
    with initialize_config_dir(config_dir=CONF, version_base="1.1"):
        c = compose(config_name=name)
    lc = c.model.get("lora", None)
    check(f"{name}: lora block present and off by default",
          lc is not None and lc.use is False, f"{dict(lc) if lc else None}")
with initialize_config_dir(config_dir=CONF, version_base="1.1"):
    c = compose(config_name="config_gazefollow", overrides=["model.lora.use=True"])
check("model.lora.use overrides to a real bool", c.model.lora.use is True,
      f"{c.model.lora.use!r} ({type(c.model.lora.use).__name__})")
script = (REPO / "train_gf.sh").read_text()
check("train_gf.sh exposes USE_LORA", "USE_LORA=" in script)
check("train_gf.sh forwards model.lora.use", 'model.lora.use="$USE_LORA"' in script)

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED: {failures}"))
sys.exit(1 if failures else 0)
