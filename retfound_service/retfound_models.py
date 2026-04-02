"""Minimal RETFound model builders for optional inference mode.

These helpers mirror the public RETFound model definitions closely enough
to load a fine-tuned classification checkpoint for inference.
"""

from functools import partial


def build_retfound_model(model_name, num_classes, drop_path_rate, model_arch=None):
    import timm
    import torch.nn as nn
    import timm.models.vision_transformer

    class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
        def __init__(self, global_pool=False, **kwargs):
            super().__init__(**kwargs)
            self.global_pool = global_pool
            if self.global_pool:
                norm_layer = kwargs["norm_layer"]
                embed_dim = kwargs["embed_dim"]
                self.fc_norm = norm_layer(embed_dim)
                del self.norm

        def forward_features(self, x):
            batch_size = x.shape[0]
            x = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for block in self.blocks:
                x = block(x)

            if self.global_pool:
                x = x[:, 1:, :].mean(dim=1, keepdim=True)
                return self.fc_norm(x)

            x = self.norm(x)
            return x[:, 0]

    import torch

    if model_name == "RETFound_mae":
        return VisionTransformer(
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            img_size=224,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            global_pool=True,
        )

    if model_name == "RETFound_dinov2":
        return timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=True,
            img_size=224,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

    raise ValueError(f"Unsupported RETFound model '{model_name}' with arch '{model_arch}'.")
