''

from __future__ import annotations

import torch
from torch import Tensor


def compose_sliding_ref_video_latents(
    *,
    local_latents: Tensor,
    sink_latents: Tensor,
    reference_latents: Tensor,
) -> tuple[Tensor, Tensor]:
    ''

    for name, value in (
        ("local_latents", local_latents),
        ("sink_latents", sink_latents),
        ("reference_latents", reference_latents),
    ):
        if not torch.is_tensor(value) or value.ndim != 5:
            shape = getattr(value, "shape", None)
            raise ValueError(f"{name} must be a 5D [B,C,F,H,W] tensor, got {type(value)} {shape}")
    if sink_latents.shape[2] != 1 or reference_latents.shape[2] != 1:
        raise ValueError(
            "Sliding-ref sink/reference must each encode to one latent frame, got "
            f"sink={tuple(sink_latents.shape)} ref={tuple(reference_latents.shape)}."
        )
    if local_latents.shape[2] < 2:
        raise ValueError(
            f"Sliding-ref local VAE path must produce an anchor plus tail, got {tuple(local_latents.shape)}."
        )
    if sink_latents.shape[:2] != local_latents.shape[:2] or sink_latents.shape[-2:] != local_latents.shape[-2:]:
        raise ValueError(
            "Sliding-ref global sink and local window latent geometry differ: "
            f"sink={tuple(sink_latents.shape)} local={tuple(local_latents.shape)}."
        )
    if reference_latents.shape[:2] != local_latents.shape[:2]:
        raise ValueError(
            "Sliding-ref reference batch/channel geometry differs from the local window: "
            f"ref={tuple(reference_latents.shape)} local={tuple(local_latents.shape)}."
        )
    return torch.cat([sink_latents, local_latents[:, :, 1:]], dim=2), reference_latents


__all__ = ["compose_sliding_ref_video_latents"]
