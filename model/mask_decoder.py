# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type, Optional, Any

from .common import LayerNorm2d
from .transformer import TwoWayTransformer, Attention

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
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
    
class features_feat(nn.Module):
    def __init__(
        self,
        vit_dim,
        transformer_dim=256,
        cnn_dim=256,
    ) -> None:
        super().__init__()
        self.embedding_encoder = nn.Sequential(
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                    )
        self.compress_vit_feat = nn.Sequential(
                                        nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim),
                                        nn.GELU(), 
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        self.compress_cnn_feat = nn.Sequential(
                                        nn.ConvTranspose2d(cnn_dim, cnn_dim // 2, kernel_size=2, stride=2),
                                        LayerNorm2d(cnn_dim // 2),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(cnn_dim // 2, cnn_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(cnn_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(cnn_dim // 4, 32, kernel_size=2, stride=2),
                                    )

    def forward(
        self,
        cnn_feature: torch.Tensor,
        image_embeddings: torch.Tensor,
        interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:   
        vit_features = interm_embeddings[2].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT   
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)
        cnn_features_feat = self.compress_cnn_feat(cnn_feature)
        cnn_features_feat = F.interpolate(cnn_features_feat, size=hq_features.shape[2:], mode='bilinear')
        return hq_features + cnn_features_feat
    
class MaskDecoder(nn.Module):
    def __init__(
        self,
        model_type,
        transformer_dim=256,
        cnn_dim=256,
        transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
        num_multimask_outputs=3,
        activation=nn.GELU,
        iou_head_depth= 3,
        iou_head_hidden_dim= 256,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        embed_dim = 256
        # self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.image_embedding_size = (1024 // 16, 1024 // 16)

        vit_dim_dict = {"vit_b":768,"vit_l":1024,"vit_h":1280,'efficient_sam_vitt':192}
        vit_dim = vit_dim_dict[model_type]
        # self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))
        self.dense_prompt_tokens = nn.Parameter(torch.randn(1, 256, 64, 64))  # learnable token
        self.feature_fusion = features_feat(vit_dim, cnn_dim=cnn_dim)
        self.cnn_modify = nn.Sequential(nn.Conv2d(cnn_dim, transformer_dim, 3, 1, 1),
                                        LayerNorm2d(transformer_dim),
                                        nn.GELU())


    def forward(
        self,
        cnn_feature: torch.Tensor,
        image_embeddings: torch.Tensor,
        multimask_output: bool,
        interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:   
        fusion_features = self.feature_fusion(cnn_feature, image_embeddings, interm_embeddings)
        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        for i_batch in range(batch_len):
            mask, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                fusion_feature=fusion_features[i_batch].unsqueeze(0),
                cnn_feature=cnn_feature[i_batch].unsqueeze(0),
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks,0)
        iou_preds = torch.cat(iou_preds,0)      

        # Select the correct mask or masks for output
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_preds = iou_preds[:, mask_slice]

        # Prepare output
        return masks, iou_preds

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        fusion_feature: torch.Tensor,
        cnn_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        tokens = output_tokens.unsqueeze(0)

        # Modify CNN feature
        cnn_feature = self.cnn_modify(cnn_feature)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) # torch.Size([1, 256, 64, 64])
        src = src + self.dense_prompt_tokens
        # image_pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)        # torch.Size([1, 256, 64, 64])
        image_pe = F.interpolate(cnn_feature, size=src.shape[2:], mode='bilinear')        # torch.Size([1, 256, 64, 64])
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)     # torch.Size([1, 256, 64, 64])
        b, c, h, w = src.shape

        # !!!
        cnn_k  = F.interpolate(cnn_feature, size=(h, w), mode='bilinear')

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens, cnn_k) # queries, keys # torch.Size([1, 15, 256]) torch.Size([1, 4096, 256])
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding = self.embedding_maskfeature(upscaled_embedding_sam) + fusion_feature # !!!!!
        
        hyper_in_list: List[torch.Tensor] = [
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            for i in range(self.num_mask_tokens)
        ]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred