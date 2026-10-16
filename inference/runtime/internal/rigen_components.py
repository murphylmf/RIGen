from __future__ import annotations

import gc
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import torch

from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_pipelines.utils.blocks import DiffusionStage
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_core.types import LatentState


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def collect_decoded_video(chunks: Any) -> torch.Tensor:
    chunk_list = [chunk.detach().cpu() for chunk in chunks]
    if not chunk_list:
        raise RuntimeError("Video decoder produced no chunks")
    video_fhwc = torch.cat(chunk_list, dim=0)
    return video_fhwc.permute(3, 0, 1, 2).float().div(255.0).clamp(0.0, 1.0)


class SimpleDenoiserNoCamTokens:
    def __init__(self, v_context: torch.Tensor | None, a_context: torch.Tensor | None) -> None:
        self.v_context = v_context
        self.a_context = a_context

    def __call__(
        self,
        transformer: Any,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(video_state, self.v_context, sigma) if video_state is not None else None
        pos_audio = modality_from_latent_state(audio_state, self.a_context, sigma) if audio_state is not None else None
        return transformer(video=pos_video, audio=pos_audio, perturbations=None)


class CachedOriginalDiffusionStage(DiffusionStage):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cached_transformer: Any | None = None

    def _transformer_ctx(self, streaming_prefetch_count: int | None, **kwargs: object) -> Any:
        if streaming_prefetch_count is not None:
            raise ValueError("Cached original refiner does not support layer streaming")
        if self._cached_transformer is None:
            self._cached_transformer = super()._build_transformer(**kwargs)
            print(
                "[rigen-refiner] loaded the original Stage-2 transformer "
                "with the distilled LoRA",
                flush=True,
            )
        return nullcontext(self._cached_transformer)

    def release(self) -> None:
        if self._cached_transformer is not None:
            self._cached_transformer.to("cpu")
            self._cached_transformer = None


class FixedVideoNoiser:
    def __init__(self, noise: torch.Tensor) -> None:
        if noise.ndim != 5:
            raise ValueError(f"Expected [B,C,T,H,W] fixed noise, got {tuple(noise.shape)}")
        self.noise = noise.detach().cpu().contiguous()
        self.patchifier = VideoLatentPatchifier(patch_size=1)

    def __call__(self, latent_state: LatentState, noise_scale: float = 1.0) -> LatentState:
        target_noise = self.patchifier.patchify(
            self.noise.to(device=latent_state.latent.device, dtype=latent_state.latent.dtype)
        )
        if target_noise.shape[0] != latent_state.latent.shape[0] or target_noise.shape[2] != latent_state.latent.shape[2]:
            raise ValueError(
                "Fixed-noise batch/channel mismatch: "
                f"noise={tuple(target_noise.shape)} state={tuple(latent_state.latent.shape)}"
            )
        if target_noise.shape[1] > latent_state.latent.shape[1]:
            raise ValueError(
                "Fixed noise has more tokens than conditioned latent state: "
                f"noise={tuple(target_noise.shape)} state={tuple(latent_state.latent.shape)}"
            )
        noise = torch.zeros_like(latent_state.latent)
        noise[:, : target_noise.shape[1]] = target_noise
        if target_noise.shape[1] < latent_state.latent.shape[1]:
            extra_mask = latent_state.denoise_mask[:, target_noise.shape[1] :]
            if torch.count_nonzero(extra_mask).item() != 0:
                raise RuntimeError("Conditioning tokens unexpectedly require fixed noise")
        scaled_mask = latent_state.denoise_mask * noise_scale
        latent = noise * scaled_mask + latent_state.latent * (1 - scaled_mask)
        return replace(latent_state, latent=latent.to(latent_state.latent.dtype))
