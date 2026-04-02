"""Minimal RETFound position-embedding interpolation helper.

Derived from the RETFound upstream utility module so a task-specific
checkpoint can be loaded without depending on the full training repo.
"""

import torch


def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" not in checkpoint_model:
        return

    pos_embed_checkpoint = checkpoint_model["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches**0.5)

    if orig_size == new_size:
        return

    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens,
        size=(new_size, new_size),
        mode="bicubic",
        align_corners=False,
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    checkpoint_model["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)
