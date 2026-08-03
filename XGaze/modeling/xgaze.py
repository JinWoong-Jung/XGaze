#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

# ==================================================================================================================
#                                                      IMPORTS                                                     #
# ==================================================================================================================
import math
from termcolor import colored
from collections import OrderedDict

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.classification import BinaryAveragePrecision

import pytorch_lightning as pl

from XGaze.modeling.encoder import GazeEncoder, HuggingFaceDinoImageEncoder, SpatialInputTokenizer, ViTEncoder
from XGaze.modeling.decoder import GazeDecoder
from XGaze.modeling.lora import apply_lora, is_lora_param
from XGaze.losses import compute_heatmap_loss, compute_angular_loss, compute_inout_loss, compute_ooc_loss
from XGaze.metrics import Distance, GFTestAUC, GFTestDistance, TestAUC
from XGaze.utils.common import spatial_argmax2d, dark_coordinate_decoding

TERM_COLOR = "cyan"

# Datasets that annotate every person in a frame (multi-person heatmap/gaze_vec/inout supervision),
# as opposed to `gazefollow` which annotates a single target person per sample.
MULTI_PERSON_DATASETS = ("video_attention_target", "childplay")

# ==================================================================================================================
#                                                   XGAZE MODULE                                                 #
# ==================================================================================================================
class XGazeModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dataset = cfg.experiment.dataset
        if self.dataset not in ("gazefollow", "video_attention_target", "childplay"):
            raise ValueError(
                f"Dataset {self.dataset} not supported. Only `gazefollow`, `video_attention_target` and `childplay` are available."
            )

        xgaze_cfg = cfg.model.XGaze
        image_encoder_cfg = xgaze_cfg.get("image_encoder", {})
        image_encoder_type = xgaze_cfg.get("image_encoder_type", image_encoder_cfg.get("type", "mae"))
        image_encoder_type = {"mae": "multimae", "dinov3": "dinov3_hf"}.get(image_encoder_type, image_encoder_type)
        multimae_cfg = image_encoder_cfg.get("mae", image_encoder_cfg.get("multimae", {}))
        dinov3_cfg = image_encoder_cfg.get("dinov3", image_encoder_cfg.get("dinov3_hf", {}))

        self.model = XGaze(
            image_size=xgaze_cfg.image_size,
            patch_size=xgaze_cfg.patch_size,
            token_dim=xgaze_cfg.token_dim,
            gaze_vec_dim=xgaze_cfg.gaze_vec_dim,
            encoder_num_heads=xgaze_cfg.get("encoder_num_heads", multimae_cfg.get("encoder_num_heads", 12)),
            encoder_depth=xgaze_cfg.get("encoder_depth", multimae_cfg.get("encoder_depth", 12)),
            encoder_num_global_tokens=xgaze_cfg.get("encoder_num_global_tokens", multimae_cfg.get("encoder_num_global_tokens", 1)),
            decoder_depth=xgaze_cfg.decoder_depth,
            decoder_num_heads=xgaze_cfg.decoder_num_heads,
            heatmap_size=cfg.data.heatmap_size,
            image_encoder_type=image_encoder_type,
            hf_image_encoder_name=xgaze_cfg.get("hf_image_encoder_name", dinov3_cfg.get("model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m")),
            hf_image_encoder_local_dir=xgaze_cfg.get("hf_image_encoder_local_dir", dinov3_cfg.get("local_dir", None)),
            hf_image_encoder_trust_remote_code=xgaze_cfg.get("hf_image_encoder_trust_remote_code", dinov3_cfg.get("trust_remote_code", True)),
            use_image_global_token=dinov3_cfg.get("use_cls_token", True),
            predict_inout=self.dataset != "gazefollow",
            inout_dropout=xgaze_cfg.get("inout_dropout", 0.1),
        )

        self.feature_map_size = cfg.model.XGaze.image_size // cfg.model.XGaze.patch_size

        if self.dataset == "gazefollow":
            self.num_train_samples = cfg.data.gf.num_train_samples
        elif self.dataset == "video_attention_target":
            self.num_train_samples = cfg.data.vat.get("num_train_samples", 0)
        else:
            self.num_train_samples = cfg.data.childplay.get("num_train_samples", 0)

        self.num_steps_in_epoch = math.ceil(self.num_train_samples / cfg.train.batch_size) if self.num_train_samples > 0 else 0

        # Define Metrics
        self.metrics = nn.ModuleDict({
            "val_dist": Distance(),
            "test_dist": GFTestDistance(),
            "test_auc": GFTestAUC(),
            "test_dist_vat": Distance(),
            "test_auc_vat": TestAUC(),
            "test_dist_childplay": Distance(),
            "test_auc_childplay": TestAUC(),
            "val_inout_ap_vat": BinaryAveragePrecision(),
            "val_inout_ap_childplay": BinaryAveragePrecision(),
            "test_inout_ap_vat": BinaryAveragePrecision(),
            "test_inout_ap_childplay": BinaryAveragePrecision(),
        })
        
        # Initialize Weights
        self._init_weights()
        

    def _init_weights(self):
        if self.cfg.model.weights is not None:
            model_ckpt = torch.load(self.cfg.model.weights, map_location="cpu", weights_only=False)
            current_state = self.state_dict()
            ckpt_state = OrderedDict(
                (name, value)
                for name, value in model_ckpt["state_dict"].items()
                if name in current_state and current_state[name].shape == value.shape
            )
            self.load_state_dict(ckpt_state, strict=False)
            print(colored(f"Loaded the model pre-trained weights from {self.cfg.model.weights}.", TERM_COLOR))
            del model_ckpt
        else:
            # Load ViT weights for Image Encoder (from MultiMAE)
            if self.model.image_encoder_type == "multimae":
                image_encoder_weights = self.cfg.model.get("pretraining", {}).get("image_encoder", None)
                if image_encoder_weights is None:
                    image_encoder_cfg = self.cfg.model.XGaze.get("image_encoder", {})
                    image_encoder_weights = image_encoder_cfg.get("mae", image_encoder_cfg.get("multimae", {})).get("weights", None)
                if image_encoder_weights is None:
                    raise ValueError("MultiMAE image encoder selected, but no pretrained weight path was configured.")
                vit_ckpt = torch.load(image_encoder_weights, map_location="cpu")
                
                vit_tokenizer_weights = OrderedDict([
                    (name.replace("input_adapters.rgb.", ""), value) 
                    for name, value in vit_ckpt["model"].items() 
                    if "input_adapters.rgb" in name
                ])
                vit_tokenizer_weights["pos_emb"] = F.interpolate(
                    vit_tokenizer_weights["pos_emb"], 
                    size=(self.feature_map_size, self.feature_map_size), 
                    mode="bilinear"
                )
                vit_encoder_weights = OrderedDict([
                    (name.replace("encoder.", ""), value) 
                    for name, value in vit_ckpt["model"].items() 
                    if "encoder" in name
                ])
                vit_encoder_weights = OrderedDict([
                    (name, value)
                    for name, value in vit_encoder_weights.items()
                    if name in self.model.encoder.encoder.state_dict()
                    and self.model.encoder.encoder.state_dict()[name].shape == value.shape
                ])
                
                self.model.image_tokenizer.load_state_dict(vit_tokenizer_weights, strict=True)
                self.model.encoder.encoder.load_state_dict(vit_encoder_weights, strict=False)
                print(colored(f"Loaded Image Encoder weights from {image_encoder_weights}.", TERM_COLOR))
                del vit_ckpt, vit_tokenizer_weights, vit_encoder_weights
            else:
                print(colored(f"Using Hugging Face Image Encoder: {self.model.hf_image_encoder_name}.", TERM_COLOR))

            # Load Gaze360 Weights for Gaze Encoder Backbone
            gaze_backbone_ckpt = torch.load(self.cfg.model.pretraining.gaze_backbone, map_location="cpu")
            gaze_backbone_weights = OrderedDict([
                (name.replace("base_head.", ""), value) 
                for name, value in gaze_backbone_ckpt["model_state_dict"].items() 
                if "base_head" in name
            ])
            self.model.gaze_encoder.backbone.load_state_dict(gaze_backbone_weights, strict=True)
            print(colored(f"Loaded Gaze Backbone weights from {self.cfg.model.pretraining.gaze_backbone}.", TERM_COLOR))

            # Delete checkpoints
            del gaze_backbone_ckpt, gaze_backbone_weights
        
        # Freeze weights
        self.freeze()

        # LoRA goes on last: `freeze` would otherwise switch the adapters off along with the
        # backbone they sit inside.
        self._apply_lora()

        # In/out-only adaptation: keep the gaze localization pathway exactly at its
        # checkpoint values and train only the task-specific classifier.
        if self.cfg.train.freeze.get("inout_only", False):
            self.freeze_inout_only()


    def _apply_lora(self):
        lora_cfg = self.cfg.model.get("lora", {})
        if not lora_cfg.get("use", False):
            return
        if self.model.image_encoder_type != "dinov3_hf" or self.model.image_encoder is None:
            raise ValueError("LoRA is only supported for the Hugging Face DINOv3 image encoder.")
        if not self.cfg.train.freeze.image_encoder:
            print(colored(
                "LoRA is enabled but train.freeze.image_encoder is False, so the backbone is being "
                "fully fine-tuned and the adapters are redundant.", "yellow"
            ))

        target_modules = list(lora_cfg.get("target_modules", ["q_proj", "v_proj"]))
        num_wrapped, num_params = apply_lora(
            self.model.image_encoder.backbone,
            target_modules=target_modules,
            r=lora_cfg.get("r", 8),
            alpha=lora_cfg.get("alpha", 16),
            dropout=lora_cfg.get("dropout", 0.0),
            last_n_layers=lora_cfg.get("last_n_layers", -1),
        )
        scope = lora_cfg.get("last_n_layers", -1)
        scope = "all layers" if scope is None or scope <= 0 else f"the last {scope} layers"
        print(colored(
            f"Applied LoRA (r={lora_cfg.get('r', 8)}, alpha={lora_cfg.get('alpha', 16)}) to "
            f"{num_wrapped} projections across {scope} of the Image Encoder: "
            f"{num_params:,} trainable parameters added ({', '.join(target_modules)}).",
            TERM_COLOR,
        ))

        # Putting trainable weights inside the backbone forces autograd to keep every layer's
        # activations, which it previously discarded because nothing there needed a gradient. At
        # batch 256 that took the encoder from ~36 GB to ~279 GB and ran the GPU out of memory.
        # Recomputing the activations in the backward pass brings it back to ~68 GB.
        # `use_reentrant=False` is required: the backbone's input does not require grad, and the
        # reentrant implementation would silently produce no gradients for the adapters.
        if lora_cfg.get("gradient_checkpointing", True):
            self.model.image_encoder.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print(colored(
                "Enabled gradient checkpointing on the Image Encoder (trades ~30% step time for "
                "the activation memory LoRA would otherwise need).", TERM_COLOR,
            ))

    
    def _set_batchnorm_eval(self, module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

            
    def _set_dropout_eval(self, module):
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.eval()

            
    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def freeze_inout_only(self):
        """Freeze the complete model except the multi-person in/out classifier."""
        inout_decoder = self.model.gaze_decoder.inout_decoder
        if inout_decoder is None:
            raise ValueError("inout_only=True requires a dataset with an in/out decoder.")

        for param in self.parameters():
            param.requires_grad = False
        for param in inout_decoder.parameters():
            param.requires_grad = True
        print(colored("In/out-only adaptation: all parameters frozen except `gaze_decoder.inout_decoder`.", TERM_COLOR))

            
    def freeze(self):
        if self.cfg.train.freeze.gaze_encoder:
            print(colored(f"Freezing the Gaze Encoder layers.", TERM_COLOR))
            self.freeze_module(self.model.gaze_encoder)
        if self.cfg.train.freeze.image_tokenizer:
            print(colored(f"Freezing the Image Tokenizer layers.", TERM_COLOR))
            if self.model.image_tokenizer is not None:
                self.freeze_module(self.model.image_tokenizer)
        if self.cfg.train.freeze.image_encoder:
            print(colored(f"Freezing the Image Encoder layers.", TERM_COLOR))
            if self.model.encoder is not None:
                self.freeze_module(self.model.encoder)
            if self.model.image_encoder is not None:
                if self.model.image_encoder_type == "dinov3_hf":
                    self.freeze_module(self.model.image_encoder.backbone)
                    for param in self.model.image_encoder.proj.parameters():
                        param.requires_grad = True
                    print(colored("Keeping the DINO feature projection layer trainable.", TERM_COLOR))
                else:
                    self.freeze_module(self.model.image_encoder)
        if self.cfg.train.freeze.gaze_decoder:
            print(colored(f"Freezing the Gaze Decoder layers.", TERM_COLOR))
            self.freeze_module(self.model.gaze_decoder)


    def _dino_backbone_prefix(self):
        # The DINOv3 backbone is frozen and always reloaded from its pretrained source in
        # `_init_weights`, so it doesn't need to be persisted in checkpoints (it accounts for
        # ~95% of checkpoint size). Only applies when the HF DINOv3 encoder is in use.
        if self.model.image_encoder_type == "dinov3_hf" and self.model.image_encoder is not None:
            return "model.image_encoder.backbone."
        return None

    def _strip_dino_backbone(self, state_dict, prefix):
        # LoRA adapters live inside the backbone but are trained, so they must survive the strip -
        # dropping them would silently discard everything the adaptation learned.
        return OrderedDict(
            (name, value) for name, value in state_dict.items()
            if not name.startswith(prefix) or is_lora_param(name)
        )

    def on_save_checkpoint(self, checkpoint):
        prefix = self._dino_backbone_prefix()
        if prefix is None:
            return
        checkpoint["state_dict"] = self._strip_dino_backbone(checkpoint["state_dict"], prefix)
        # The SWA callback keeps its own full copy of the model weights; strip it there too.
        swa_state = checkpoint.get("callbacks", {}).get("StochasticWeightAveraging")
        if swa_state is not None and swa_state.get("average_model_state") is not None:
            swa_state["average_model_state"] = self._strip_dino_backbone(swa_state["average_model_state"], prefix)

    def on_load_checkpoint(self, checkpoint):
        prefix = self._dino_backbone_prefix()
        if prefix is None:
            return
        # Re-inject the (already loaded, pretrained) backbone weights before Lightning restores
        # the state dict, so both the model and the SWA callback see a complete state dict.
        # Only the frozen weights are re-injected: the checkpoint's own LoRA adapters must win over
        # the freshly initialised ones currently on the model.
        backbone_state = OrderedDict(
            (f"{prefix}{name}", value)
            for name, value in self.model.image_encoder.backbone.state_dict().items()
            if not is_lora_param(name)
        )
        checkpoint["state_dict"].update(backbone_state)
        swa_state = checkpoint.get("callbacks", {}).get("StochasticWeightAveraging")
        if swa_state is not None and swa_state.get("average_model_state") is not None:
            swa_state["average_model_state"].update(backbone_state)


    def forward(self, batch):
        return self.model(batch)
        
    
    def compute_loss(
        self,
        gaze_heatmap_gt,
        gaze_vec_gt,
        inout_gt,
        gaze_heatmap_pred,
        gaze_vec_pred,
        inout_pred=None,
        head_center_gt=None,
        gaze_pt_gt=None,
    ):
        """
        `inout_gt` is in {0, 1} for `gazefollow` (single target person) and in {-1, 0, 1} for the
        multi-person datasets (`video_attention_target`/`childplay`), where -1 marks unknown gaze
        class or padded person slots (see the dataset READMEs) and is excluded from every loss term.

        `head_center_gt`/`gaze_pt_gt` are only needed by the out-of-cone penalty; when they are
        omitted (or the penalty is both disabled and unlogged) the OOC term is skipped entirely.
        """

        device = gaze_heatmap_pred.device

        heatmap_loss = torch.tensor(0.0, device=device)
        angular_loss = torch.tensor(0.0, device=device)
        inout_loss = torch.tensor(0.0, device=device)
        ooc_loss = torch.tensor(0.0, device=device)

        inside_gt = inout_gt.clamp(min=0)
        if torch.sum(inside_gt) > 0:  # to avoid case where all samples of the batch are outside (i.e. division by 0)
            heatmap_loss = compute_heatmap_loss(gaze_heatmap_pred, gaze_heatmap_gt, inside_gt)
            angular_loss = compute_angular_loss(gaze_vec_pred, gaze_vec_gt, inside_gt)

        weight_inout = self.cfg.loss.get("weight_inout", 0.0)
        if inout_pred is not None and weight_inout != 0:
            inout_loss = compute_inout_loss(inout_pred, inout_gt)

        # Out-of-cone penalty. `log_ooc` lets a run measure the raw term with `weight_ooc: 0`, which
        # is how its weight is calibrated against the (much larger) heatmap term.
        weight_ooc = self.cfg.loss.get("weight_ooc", 0.0)
        ooc_count = 0
        if self._ooc_active() and head_center_gt is not None:
            ooc_cfg = self.cfg.loss.get("ooc", {})
            ooc_loss, ooc_count = compute_ooc_loss(
                gaze_heatmap_pred,
                head_center_gt,
                gaze_pt_gt,
                inout_gt,
                theta_deg=ooc_cfg.get("theta_deg", 60.0),
                alpha=ooc_cfg.get("alpha", 30.0),
                tau=ooc_cfg.get("tau", 1.0),
                min_gaze_dist=ooc_cfg.get("min_gaze_dist", 0.05),
                normalize_alpha=ooc_cfg.get("normalize_alpha", True),
                return_count=True,
            )

        total_loss = (
            self.cfg.loss.weight_heatmap * heatmap_loss +
            self.cfg.loss.weight_angular * angular_loss +
            weight_inout * inout_loss +
            weight_ooc * ooc_loss
        )

        logs = {
            "heatmap_loss": heatmap_loss.item(),
            "angular_loss": angular_loss.item(),
            "inout_loss": inout_loss.item(),
            "ooc_loss": ooc_loss.item(),
            "ooc_count": ooc_count,  # eligible entries, the correct batch_size for logging the term
            "total_loss": total_loss.item(),
        }
        return total_loss, logs

    def _ooc_active(self):
        """Whether the out-of-cone term should be computed: either it is weighted into the total
        loss, or the run is measuring it with `weight_ooc: 0` to calibrate that weight."""
        return self.cfg.loss.get("weight_ooc", 0.0) != 0 or self.cfg.loss.get("log_ooc", False)

    def _ooc_targets(self, batch):
        """
        Return `(head_center, gaze_pt)` aligned with the leading dims of the predictions produced by
        `_forward_step`: (b, 2) for `gazefollow`, where only the last person slot is annotated, and
        (b, n, 2) for the multi-person datasets, where every slot carries its own ground truth.

        Returns `(None, None)` when the batch lacks the tensors (eg. the GazeFollow test split, whose
        `gaze_pt` holds multiple annotators per image).
        """
        if "head_centers" not in batch or "gaze_pt" not in batch:
            return None, None

        head_center, gaze_pt = batch["head_centers"], batch["gaze_pt"]
        if self.dataset == "gazefollow":
            head_center = head_center[:, -1, :]  # (b, n, 2) >> (b, 2)
        if head_center.shape != gaze_pt.shape:
            return None, None
        return head_center, gaze_pt

    
    def configure_optimizers(self):
        lr = self.cfg.optimizer.get("lr_non_inout", self.cfg.optimizer.lr)
        lr_inout = self.cfg.optimizer.get("lr_inout", None)
        inout_decoder = self.model.gaze_decoder.inout_decoder

        if lr_inout is not None and inout_decoder is not None:
            inout_params = []
            other_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith("model.gaze_decoder.inout_decoder."):
                    inout_params.append(param)
                else:
                    other_params.append(param)

            param_groups = []
            if other_params:
                param_groups.append({"params": other_params, "lr": lr})
            if inout_params:
                param_groups.append({"params": inout_params, "lr": lr_inout})
            optimizer = optim.AdamW(param_groups, weight_decay=self.cfg.optimizer.weight_decay)
        else:
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )
        
        # Scheduler: Cosine Annealing with Warmup or None
        if self.cfg.scheduler.type == "cosine_warmup":
            warmup_steps = self.cfg.scheduler.warmup_epochs * self.num_steps_in_epoch
            max_steps = self.cfg.train.epochs * self.num_steps_in_epoch
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
            scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        return optimizer
            
                
    def on_fit_start(self):
        # Define metrics
        if self.cfg.wandb.log:
            wandb.define_metric('metric/test/dist_to_avg', summary='min')
            wandb.define_metric('metric/test/avg_dist', summary='min')
            wandb.define_metric('metric/test/min_dist', summary='min')
            wandb.define_metric('metric/test/auc', summary='max')
            wandb.define_metric('metric/val/inout_ap', summary='max')
            wandb.define_metric('metric/test/inout_ap', summary='max')
            
            wandb.define_metric('loss/train_epoch', summary='min')
            wandb.define_metric('loss/val', summary='min')
            wandb.define_metric('metric/val/dist', summary='min')

            
    def on_train_epoch_start(self):
        if self.cfg.train.freeze.get("inout_only", False):
            # Prevent frozen BatchNorm/dropout state from changing during adaptation.
            self.model.eval()
            self.model.gaze_decoder.inout_decoder.train()
            return

        # Set BN layers to eval mode for frozen modules
        if self.cfg.train.freeze.gaze_encoder:
            self.model.gaze_encoder.apply(self._set_batchnorm_eval)
            self.model.gaze_encoder.apply(self._set_dropout_eval)
        if self.cfg.train.freeze.image_tokenizer:
            if self.model.image_tokenizer is not None:
                self.model.image_tokenizer.apply(self._set_batchnorm_eval)
                self.model.image_tokenizer.apply(self._set_dropout_eval)
        if self.cfg.train.freeze.image_encoder:
            if self.model.encoder is not None:
                self.model.encoder.apply(self._set_batchnorm_eval)
                self.model.encoder.apply(self._set_dropout_eval)
            if self.model.image_encoder is not None:
                if self.model.image_encoder_type == "dinov3_hf":
                    self.model.image_encoder.backbone.apply(self._set_batchnorm_eval)
                    self.model.image_encoder.backbone.apply(self._set_dropout_eval)
                else:
                    self.model.image_encoder.apply(self._set_batchnorm_eval)
                    self.model.image_encoder.apply(self._set_dropout_eval)
        if self.cfg.train.freeze.gaze_decoder:
            self.model.gaze_decoder.apply(self._set_batchnorm_eval)
            self.model.gaze_decoder.apply(self._set_dropout_eval)
            
            
    def _forward_step(self, batch):
        """
        Runs the model and returns (heatmap_pred, gaze_vec_pred, inout_pred, inout_gt) at the
        granularity appropriate for `self.dataset`: for `gazefollow`, only the single annotated
        target person (last slot) is kept and no in/out decoder is instantiated; for the multi-person
        datasets every person slot is kept since ground truth is annotated per person.
        """
        gaze_heatmap_pred, gaze_vec_pred, inout_pred = self(batch)

        if self.dataset == "gazefollow":
            gaze_heatmap_pred = gaze_heatmap_pred[:, -1, ...]  # (b, n, 64, 64) >> (b, 64, 64)
            gaze_vec_pred = gaze_vec_pred[:, -1, ...]  # (b, n, 2) >> (b, 2)
            inout_pred_for_loss = None
        else:
            inout_pred_for_loss = inout_pred

        return gaze_heatmap_pred, gaze_vec_pred, inout_pred, inout_pred_for_loss

    def training_step(self, batch, batch_idx):
        n = len(batch["image"])

        # Forward pass
        gaze_heatmap_pred, gaze_vec_pred, _, inout_pred_for_loss = self._forward_step(batch)
        ni = int(batch["inout"].clamp(min=0).sum().item())

        # Compute loss
        head_center_gt, gaze_pt_gt = self._ooc_targets(batch)
        loss, logs = self.compute_loss(
            batch["gaze_heatmap"],
            batch["gaze_vec"],
            batch["inout"],
            gaze_heatmap_pred,
            gaze_vec_pred,
            inout_pred_for_loss,
            head_center_gt,
            gaze_pt_gt,
        )

        # Logging losses
        self.log("loss/train/heatmap", logs["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/angular", logs["angular_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        if inout_pred_for_loss is not None and self.cfg.loss.get("weight_inout", 0.0) != 0:
            self.log("loss/train/inout", logs["inout_loss"], batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        # Weighted by the eligible count, which `min_gaze_dist` makes stricter than the in-frame count.
        if self._ooc_active() and logs["ooc_count"] > 0:
            self.log("loss/train/ooc", logs["ooc_loss"], batch_size=logs["ooc_count"], prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train", logs["total_loss"], batch_size=n, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        n = len(batch["image"])

        # Forward pass
        gaze_heatmap_pred, gaze_vec_pred, _, inout_pred_for_loss = self._forward_step(batch)
        ni = int(batch["inout"].clamp(min=0).sum().item())

        gaze_heatmap_prob = gaze_heatmap_pred.sigmoid()
        # spatial_argmax2d expects a (b, h, w) heatmap, so flatten the person dim in for multi-person batches
        hm_shape = gaze_heatmap_prob.shape
        gaze_pt_pred = spatial_argmax2d(gaze_heatmap_prob.reshape(-1, *hm_shape[-2:]), normalize=True).reshape(*hm_shape[:-2], 2)

        # Compute loss
        head_center_gt, gaze_pt_gt = self._ooc_targets(batch)
        loss, logs = self.compute_loss(
            batch["gaze_heatmap"],
            batch["gaze_vec"],
            batch["inout"],
            gaze_heatmap_pred,
            gaze_vec_pred,
            inout_pred_for_loss,
            head_center_gt,
            gaze_pt_gt,
        )

        # Update metrics
        self.metrics["val_dist"].update(gaze_pt_pred, batch["gaze_pt"], batch["inout"])
        val_inout_ap = None

        # AP is evaluated on every validation epoch for the datasets that provide
        # per-person in/out labels. GazeFollow has no in/out decoder.
        if self.dataset in MULTI_PERSON_DATASETS and inout_pred_for_loss is not None:
            suffix = "vat" if self.dataset == "video_attention_target" else "childplay"
            inout_gt = batch["inout"]
            known_mask = inout_gt != -1
            if known_mask.any():
                self.metrics[f"val_inout_ap_{suffix}"].update(
                    inout_pred_for_loss.sigmoid()[known_mask],
                    inout_gt[known_mask].long(),
                )
                val_inout_ap = self.metrics[f"val_inout_ap_{suffix}"]

        # Logging losses
        self.log("loss/val/heatmap", logs["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val/angular", logs["angular_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True)
        if inout_pred_for_loss is not None and self.cfg.loss.get("weight_inout", 0.0) != 0:
            self.log("loss/val/inout", logs["inout_loss"], batch_size=n, prog_bar=False, on_step=False, on_epoch=True)
        if self._ooc_active() and logs["ooc_count"] > 0:
            self.log("loss/val/ooc", logs["ooc_loss"], batch_size=logs["ooc_count"], prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val", logs["total_loss"], batch_size=n, prog_bar=True, on_step=False, on_epoch=True)

        # Logging metrics
        self.log("metric/val/dist", self.metrics["val_dist"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
        if val_inout_ap is not None:
            self.log(
                "metric/val/inout_ap",
                val_inout_ap,
                batch_size=int(known_mask.sum().item()),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

    def test_step(self, batch, batch_idx):

        n = len(batch["image"])
        ni = int(batch["inout"].clamp(min=0).sum().item())

        # Forward pass
        gaze_heatmap_pred, gaze_vec_pred, inout_pred, _ = self._forward_step(batch)
        gaze_heatmap_prob = gaze_heatmap_pred.sigmoid()
        # dark_coordinate_decoding expects a (b, h, w) heatmap, so flatten the person dim for multi-person batches
        hm_shape = gaze_heatmap_prob.shape
        gaze_pt_pred = dark_coordinate_decoding(
            gaze_heatmap_prob.reshape(-1, *hm_shape[-2:]), kernel_size=self.cfg.data.heatmap_sigma * 3, normalize=True
        ).reshape(*hm_shape[:-2], 2)

        if self.dataset == "gazefollow":
            test_dist_to_avg, test_avg_dist, test_min_dist = self.metrics["test_dist"](gaze_pt_pred, batch["gaze_pt"])
            self.metrics["test_auc"].update(gaze_heatmap_prob, batch["gaze_pt"])

            self.log("metric/test/auc", self.metrics["test_auc"], batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/dist_to_avg", test_dist_to_avg, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/avg_dist", test_avg_dist, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/min_dist", test_min_dist, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
        elif self.dataset in ("video_attention_target", "childplay"):
            suffix = "vat" if self.dataset == "video_attention_target" else "childplay"
            inout_gt = batch["inout"]
            known_mask = inout_gt != -1

            self.metrics[f"test_dist_{suffix}"].update(gaze_pt_pred, batch["gaze_pt"], inout_gt)
            self.metrics[f"test_auc_{suffix}"].update(gaze_heatmap_prob, batch["gaze_pt"], inout_gt)
            self.metrics[f"test_inout_ap_{suffix}"].update(inout_pred.sigmoid()[known_mask], inout_gt[known_mask].long())

            self.log("metric/test/auc", self.metrics[f"test_auc_{suffix}"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/dist", self.metrics[f"test_dist_{suffix}"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
            self.log(
                "metric/test/inout_ap",
                self.metrics[f"test_inout_ap_{suffix}"],
                batch_size=int(known_mask.sum().item()),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )



# ==================================================================================================================== #
#                                                  XGAZE ARCHITECTURE                                                #
# ==================================================================================================================== #
class XGaze(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        token_dim: int = 768,
        gaze_vec_dim: int = 2,
        encoder_num_heads: int = 12,
        encoder_depth: int = 12,
        encoder_num_global_tokens: int = 1,
        decoder_depth: int = 2,
        decoder_num_heads: int = 8,
        heatmap_size: int = 64,
        image_encoder_type: str = "multimae",
        hf_image_encoder_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        hf_image_encoder_local_dir: str | None = None,
        hf_image_encoder_trust_remote_code: bool = True,
        use_image_global_token: bool = True,
        predict_inout: bool = True,
        inout_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.image_embedding_size = image_size // patch_size
        self.heatmap_size = heatmap_size
        self.encoder_num_global_tokens = encoder_num_global_tokens
        image_encoder_type = {"mae": "multimae", "dinov3": "dinov3_hf"}.get(image_encoder_type, image_encoder_type)
        self.image_encoder_type = image_encoder_type
        self.hf_image_encoder_name = hf_image_encoder_name
        self.use_image_global_token = use_image_global_token

        self.gaze_encoder = GazeEncoder(
            token_dim=token_dim, 
            feature_dim=512, 
            gaze_vec_dim=gaze_vec_dim
        )

        if image_encoder_type == "multimae":
            self.image_tokenizer = SpatialInputTokenizer(
                num_channels=3, 
                stride_level=1, 
                patch_size=patch_size, 
                token_dim=token_dim, 
                use_sincos_pos_emb=True, 
                is_learnable_pos_emb=False, 
                image_size=image_size
            )

            self.encoder = ViTEncoder(
                token_dim=token_dim, 
                depth=encoder_depth, 
                num_heads=encoder_num_heads, 
                num_global_tokens=encoder_num_global_tokens, 
                mlp_ratio=4.0, 
                use_qkv_bias=True, 
                drop_rate=0.0, 
                attn_drop_rate=0.0, 
                drop_path_rate=0.0
            )
            self.image_encoder = None
        elif image_encoder_type == "dinov3_hf":
            self.image_tokenizer = None
            self.encoder = None
            self.encoder_num_global_tokens = 0
            self.image_encoder = HuggingFaceDinoImageEncoder(
                model_name=hf_image_encoder_name,
                local_dir=hf_image_encoder_local_dir,
                image_size=image_size,
                patch_size=patch_size,
                token_dim=token_dim,
                trust_remote_code=hf_image_encoder_trust_remote_code,
            )
        else:
            raise ValueError(f"Unsupported image_encoder_type={image_encoder_type}.")

        self.gaze_decoder = GazeDecoder(
            token_dim=token_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            feature_map_size=self.image_embedding_size,
            heatmap_size=heatmap_size,
            predict_inout=predict_inout,
            inout_dropout=inout_dropout,
        )


    def forward(self, sample):
        # Expected sample = {"image": image, "heads": heads, "head_bboxes": head_bboxes}
        
        # Encode Gaze Tokens ===================================================
        gaze_tokens, gaze_vec = self.gaze_encoder(sample["heads"], sample["head_bboxes"])  # (b, n, d), (b, n, 2)
        
        if self.image_encoder_type == "multimae":
            # Tokenize Inputs ===================================================
            image_tokens = self.image_tokenizer(sample["image"])  # (b, t, d) / t = num_tokens, d = token_dim
            b, t, d = image_tokens.shape
            s = int(math.sqrt(t))
            
            # Encode Image =====================================================        
            image_tokens = self.encoder(image_tokens, return_all_layers=False)  # (b, t+gt, d) / gt = num global tokens
            image_tokens = image_tokens[:, :-self.encoder_num_global_tokens, :] # (b, t, d)
            image_tokens = image_tokens.permute(0, 2, 1).view(b, d, s, s) # (b, t, d) >> (b, d, t) >> (b, d, s, s)
            image_global_token = None
        else:
            image_tokens, image_global_token = self.image_encoder(sample["image"])  # (b, d, h, w), (b, 1, d)
            if not self.use_image_global_token:
                image_global_token = None
        
        # Decode Gaze Target =====================================================
        gaze_heatmap, inout_logits = self.gaze_decoder(
            image_tokens,
            gaze_tokens,
            image_global_token=image_global_token,
        )  # (b, n, hm_h, hm_w), (b, n)

        return gaze_heatmap, gaze_vec, inout_logits
