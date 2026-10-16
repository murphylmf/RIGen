from dataclasses import dataclass, field
from typing import Protocol

import torch

from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.conditioning import ConditioningItem
from ltx_core.model.transformer import X0Model
from ltx_core.types import LatentState
from ltx_pipelines.utils.constants import VIDEO_LATENT_CHANNELS, VIDEO_SCALE_FACTORS


class PipelineComponents:
    ''

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.dtype = dtype
        self.device = device

        self.video_scale_factors = VIDEO_SCALE_FACTORS
        self.video_latent_channels = VIDEO_LATENT_CHANNELS

        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)


class Denoiser(Protocol):
    ''

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]: ...


@dataclass(frozen=True)
class ModalitySpec:
    ''

    context: torch.Tensor
    conditionings: list[ConditioningItem] = field(default_factory=list)
    noise_scale: float = 1.0
    frozen: bool = False
    initial_latent: torch.Tensor | None = None
