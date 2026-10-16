''

from __future__ import annotations

import torch
from torch import Tensor


CAUSAL_RCM_UPDATE_MODE = "consistency_renoise"


def consistency_renoise_step(
    *,
    denoised: Tensor,
    next_sigma: Tensor,
    generator: torch.Generator,
    output_dtype: torch.dtype,
) -> Tensor:
    ''

    if next_sigma.numel() != 1:
        raise ValueError(
            f"Causal-rCM next_sigma must be scalar, got {tuple(next_sigma.shape)}."
        )
    sigma = float(next_sigma.item())
    if not 0.0 <= sigma < 1.0:
        raise ValueError(f"Causal-rCM next_sigma must lie in [0, 1), got {sigma}.")
    if sigma == 0.0:
        return denoised.to(dtype=output_dtype)

    noise = torch.randn(
        denoised.shape,
        generator=generator,
        device=denoised.device,
        dtype=output_dtype,
    )
    return ((1.0 - sigma) * denoised.float() + sigma * noise.float()).to(
        dtype=output_dtype
    )


__all__ = ["CAUSAL_RCM_UPDATE_MODE", "consistency_renoise_step"]
