''

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from ltx_core.components.patchifiers import (
    AudioPatchifier,
    VideoLatentPatchifier,
    get_pixel_coords,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape
from ltx_trainer.timestep_samplers import TimestepSampler

                                                                 
DEFAULT_FPS = 24

                             
VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class TrainingStrategyConfigBase(BaseModel):
    ''

    model_config = ConfigDict(extra="forbid")

    name: Literal[
        "text_to_video",
        "video_to_video",
        "phase_aware",
        "bidir_prompt_schedule",
        "ar_teacher",
        "ar_sliding_ref",
    ] = Field(
        description="Unique name identifying the training strategy type"
    )
    conditioning_dropout_p: float = Field(
        default=0.1,
        description="Probability of dropping text conditioning to preserve classifier-free guidance behavior",
        ge=0.0,
        le=1.0,
    )
    cfg_training_enabled: bool = Field(
        default=False,
        description="Enable CFG-compatible joint unconditional training samples.",
    )
    joint_uncond_prob: float = Field(
        default=0.0,
        description="Probability of replacing a sample with empty text conditioning.",
        ge=0.0,
        le=1.0,
    )
    text_dropout_prob: float | None = Field(
        default=None,
        description="Optional text conditioning dropout probability. Defaults to joint_uncond_prob when unset.",
        ge=0.0,
        le=1.0,
    )


@dataclass
class ModelInputs:
    ''

    video: Modality
    audio: Modality | None
    action_joystick_tokens: Tensor | None = None
    action_joystick_condition_mask: Tensor | None = None
    action_button_state: Tensor | None = None
    action_button_values: Tensor | None = None
    action_button_condition_mask: Tensor | None = None
    action_button_meaning_embeds: Tensor | None = None
    action_button_temporal_alignment: str = "global"
    action_button_prefix_tokens: int = 0
    action_button_frame_rate: float = DEFAULT_FPS
    action_button_query_latent_indices: Tensor | None = None
    first_frame_latent: Tensor | None = None

                                             
    video_targets: Tensor | None = None
    audio_targets: Tensor | None = None

                                
    video_loss_mask: Tensor | None = None                                                    
    audio_loss_mask: Tensor | None = None
    video_loss_weights: Tensor | None = None
    audio_loss_weights: Tensor | None = None

                                                             
    ref_seq_len: int | None = None                                             
    video_clean_latent: Tensor | None = None
    video_noisy_latent: Tensor | None = None
    audio_clean_latent: Tensor | None = None
    audio_noisy_latent: Tensor | None = None
    ar_video_block_token_ranges: tuple[tuple[int, int], ...] | None = None
    ar_audio_block_token_ranges: tuple[tuple[int, int], ...] | None = None


@dataclass
class ModelOutputs:
    video_pred: Tensor | None
    audio_pred: Tensor | None


@dataclass
class LossOutputs:
    total: Tensor
    metrics: dict[str, Tensor]


class TrainingStrategy(ABC):
    ''

    def __init__(self, config: TrainingStrategyConfigBase):
        ''
        self.config = config
        self._video_patchifier = VideoLatentPatchifier(patch_size=1)
        self._audio_patchifier = AudioPatchifier(patch_size=1)

    @property
    def requires_audio(self) -> bool:
        ''
        return False

    @abstractmethod
    def get_data_sources(self) -> list[str] | dict[str, str]:
        ''

    @abstractmethod
    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        ''

    @abstractmethod
    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> LossOutputs:
        ''

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        ''
        return {}

    def collect_diagnostics(
        self,
        batch: dict[str, Any],
        outputs: ModelOutputs,
        losses: LossOutputs,
    ) -> dict[str, Tensor] | None:
        ''
        return None

    def _get_video_positions(
        self,
        num_frames: int,
        height: int,
        width: int,
        batch_size: int,
        fps: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        ''
        latent_coords = self._video_patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                frames=num_frames,
                height=height,
                width=width,
                batch=batch_size,
                channels=128,                         
            ),
            device=device,
        )

                                                               
        pixel_coords = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=VIDEO_SCALE_FACTORS,
            causal_fix=True,
        ).to(dtype)

                                                                  
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps

        return pixel_coords

    def _get_audio_positions(
        self,
        num_time_steps: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        ''
        mel_bins = 16

        latent_coords = self._audio_patchifier.get_patch_grid_bounds(
            output_shape=AudioLatentShape(
                frames=num_time_steps,
                mel_bins=mel_bins,
                batch=batch_size,
                channels=8,                         
            ),
            device=device,
        )

        return latent_coords.to(dtype)

    @staticmethod
    def _create_per_token_timesteps(conditioning_mask: Tensor, sampled_sigma: Tensor) -> Tensor:
        ''
                                                              
        expanded_sigma = sampled_sigma.view(-1, 1).expand_as(conditioning_mask)

                                                                        
        return torch.where(conditioning_mask, torch.zeros_like(expanded_sigma), expanded_sigma)

    @staticmethod
    def _create_first_frame_conditioning_mask(
        batch_size: int,
        sequence_length: int,
        height: int,
        width: int,
        device: torch.device,
        first_frame_conditioning_p: float = 0.0,
    ) -> Tensor:
        ''
        conditioning_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)

        if first_frame_conditioning_p > 0 and random.random() < first_frame_conditioning_p:
            first_frame_end_idx = height * width
            if first_frame_end_idx < sequence_length:
                conditioning_mask[:, :first_frame_end_idx] = True

        return conditioning_mask

    def _apply_conditioning_dropout(
        self,
        video_prompt_embeds: Tensor,
        audio_prompt_embeds: Tensor | None,
        prompt_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        ''
        dropout_p = self.config.conditioning_dropout_p
        if dropout_p <= 0:
            return video_prompt_embeds, audio_prompt_embeds, prompt_attention_mask

        batch_size = video_prompt_embeds.shape[0]
        drop_mask = torch.rand(batch_size, device=video_prompt_embeds.device) < dropout_p
        if not torch.any(drop_mask):
            return video_prompt_embeds, audio_prompt_embeds, prompt_attention_mask

        video_prompt_embeds = video_prompt_embeds.clone()
        video_prompt_embeds[drop_mask] = 0

        if audio_prompt_embeds is not None:
            audio_prompt_embeds = audio_prompt_embeds.clone()
            audio_prompt_embeds[drop_mask] = 0

        prompt_attention_mask = prompt_attention_mask.clone()
        prompt_attention_mask[drop_mask] = 0
        return video_prompt_embeds, audio_prompt_embeds, prompt_attention_mask
