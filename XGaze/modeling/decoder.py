#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import math
from typing import Type

import torch
from torch import Tensor, nn


# ****************************************************** #
#                      GAZE DECODER                      #
# ****************************************************** #
# The implementation of the Gaze Decoder is adapted from the Segment Anything repository (https://github.com/facebookresearch/segment-anything).
class GazeDecoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        depth: int,
        num_heads: int,
        feature_map_size: int,
        heatmap_size: int,
        predict_inout: bool = True,
        inout_dropout: float = 0.1,
    ) -> None:
        """
        Predicts gaze heatmaps given an image and gaze tokens, using a transformer architecture.

        Arguments:
          token_dim (int): the token dimension of the transformer
          depth (int): the number of layers in the transformer
          num_heads (int): the number of attention heads in the transformer
        """
        super().__init__()
        
        self.token_dim = token_dim
        self.depth = depth
        self.num_heads = num_heads
        self.feature_map_size = feature_map_size
        self.heatmap_size = heatmap_size
        self.predict_inout = predict_inout

        self.query_tokens = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.blocks = nn.ModuleList(
            [
                QueryGazeAttentionBlock(
                    token_dim=token_dim,
                    num_heads=num_heads,
                    mlp_dim=2048,
                    attention_downsample_rate=2,
                )
                for _ in range(depth)
            ]
        )

        self.upscaler, heatmap_emb_dim = self._build_upscaler(
            token_dim=token_dim,
            feature_map_size=feature_map_size,
            heatmap_size=heatmap_size,
        )

        # MLP for heatmap embeddings
        self.heatmap_mlp = MLP(token_dim, token_dim, heatmap_emb_dim, 3)

        if self.predict_inout:
            # Bottlenecked in/out head: token_dim -> token_dim // 2 -> 1.
            # Keep logits here; BCEWithLogitsLoss is applied by the Lightning module.
            inout_hidden_dim = max(token_dim // 2, 1)
            self.inout_decoder = nn.Sequential(
                nn.Linear(token_dim, inout_hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=inout_dropout),
                nn.Linear(inout_hidden_dim, 1),
            )
        else:
            self.inout_decoder = None

    @staticmethod
    def _build_upscaler(token_dim: int, feature_map_size: int, heatmap_size: int) -> tuple[nn.Sequential, int]:
        if heatmap_size % feature_map_size != 0:
            raise ValueError(
                f"heatmap_size={heatmap_size} must be divisible by feature_map_size={feature_map_size}."
            )

        upsample_factor = heatmap_size // feature_map_size
        if upsample_factor == 1:
            return nn.Sequential(
                nn.Conv2d(token_dim, token_dim // 8, kernel_size=3, padding=1),
                LayerNorm2d(token_dim // 8),
                nn.GELU(),
            ), token_dim // 8
        if upsample_factor == 2:
            return nn.Sequential(
                Interpolate(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(token_dim, token_dim // 8, kernel_size=3, padding=1),
                LayerNorm2d(token_dim // 8),
                nn.GELU(),
            ), token_dim // 8
        if upsample_factor == 4:
            return nn.Sequential(
                Interpolate(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(token_dim, token_dim // 4, kernel_size=3, padding=1),
                LayerNorm2d(token_dim // 4),
                nn.GELU(),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(token_dim // 4, token_dim // 8, kernel_size=3, padding=1),
                nn.GELU(),
            ), token_dim // 8

        raise ValueError(
            f"Unsupported upsample factor {upsample_factor}. "
            "Use heatmap_size equal to feature_map_size, 2x, or 4x."
        )
        
        
    def forward(
        self,
        image_tokens: torch.Tensor,
        gaze_tokens: torch.Tensor,
        image_global_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Predict per-person gaze heatmaps and optional in/out-of-frame logits given image and gaze tokens.
        One learnable query is instantiated per person (batched over `n`) without adding gaze-token
        content at initialization. Each query then only ever sees (a) that same person's own gaze
        token (`gaze_tokens[:, i]`) and (b) the shared scene tokens (`image_context`, same for every
        person) - never another person's gaze token - so the n people are processed fully
        independently of one another, just batched for efficiency.

        Arguments:
          image_tokens (torch.Tensor): the patch tokens encoding the scene image, shape (b, c, h, w)
          gaze_tokens (torch.Tensor): the tokens encoding people's head/gaze information, shape (b, n, c)
          image_global_token (torch.Tensor | None): optional CLS/global scene token, shape (b, 1, c)

        Returns:
          tuple[torch.Tensor, torch.Tensor | None]: batched predicted gaze heatmaps (b, n, hm_h, hm_w)
          and optional in/out-of-frame logits (b, n)
        """

        b, ic, ih, iw = image_tokens.shape  # (b, c, h, w)
        n = gaze_tokens.shape[1]

        image_context = image_tokens.view(b, ic, ih * iw).permute(0, 2, 1)  # (b, h*w, c)
        if image_global_token is not None:
            if image_global_token.ndim == 2:
                image_global_token = image_global_token.unsqueeze(1)
            image_context = torch.cat([image_global_token, image_context], dim=1)  # (b, 1+h*w, c)
        queries = self.query_tokens.expand(b, n, -1)  # (b, n, c)
        for block in self.blocks:
            queries = block(queries=queries, image_context=image_context, gaze_context=gaze_tokens)

        upscaled_image_tokens = self.upscaler(image_tokens)  # (b, c', hm_h, hm_w)
        b, c, h, w = upscaled_image_tokens.shape

        gaze_heatmap_emb = self.heatmap_mlp(queries)  # (b, n, c')
        gaze_heatmap = torch.einsum("bnc,bchw->bnhw", gaze_heatmap_emb, upscaled_image_tokens)  # (b, n, hm_h, hm_w)

        inout_logits = self.inout_decoder(queries).squeeze(-1) if self.inout_decoder is not None else None  # (b, n)

        return gaze_heatmap, inout_logits


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class QueryGazeAttentionBlock(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.GELU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.cross_attn_query_to_image = Attention(token_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_image = nn.LayerNorm(token_dim)

        self.cross_attn_query_to_gaze = Attention(token_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_gaze = nn.LayerNorm(token_dim)

        self.mlp = MLPBlock(token_dim, mlp_dim, activation)
        self.norm_mlp = nn.LayerNorm(token_dim)

    def forward(
        self,
        queries: Tensor,
        image_context: Tensor,
        gaze_context: Tensor,
    ) -> Tensor:
        b, n, c = queries.shape

        # Attend to this person's OWN gaze token first: merge (b, n) into the attention batch so
        # query i only ever sees gaze_context i, never another person's gaze token.
        q_own = queries.reshape(b * n, 1, c)
        gaze_own = gaze_context.reshape(b * n, 1, c)
        queries = queries + self.cross_attn_query_to_gaze(q=q_own, k=gaze_own, v=gaze_own).reshape(b, n, c)
        queries = self.norm_gaze(queries)

        # Each person's query independently attends to the shared scene tokens (no cross-person mixing here either).
        queries = queries + self.cross_attn_query_to_image(q=queries, k=image_context, v=image_context)
        queries = self.norm_image(queries)

        queries = queries + self.mlp(queries)
        queries = self.norm_mlp(queries)
        return queries


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.internal_dim = token_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide token_dim."

        self.q_proj = nn.Linear(token_dim, self.internal_dim)
        self.k_proj = nn.Linear(token_dim, self.internal_dim)
        self.v_proj = nn.Linear(token_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, token_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head) # inject fov here
        attn = torch.softmax(attn, dim=-1)
        
        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
    
    
class MLPBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
    
class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interpolate = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x

    def __repr__(self):
        return f"Interpolate(scale_factor={self.scale_factor}, mode={self.mode}, align_corners={self.align_corners})"
