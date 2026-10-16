''

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import torch
from ltx_trainer.training_strategies.ar_teacher_masks import (
    AVBlock,
    compute_ti2av_blocks,
)
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.phase_aware import (
    PhaseAwareConfig,
    PhaseAwareStrategy,
)
from pydantic import Field, model_validator
from torch import Tensor


class BidirPromptScheduleConfig(PhaseAwareConfig):
    ''

    name: Literal["bidir_prompt_schedule"] = "bidir_prompt_schedule"
    forward_mode: Literal["audio_video"] = "audio_video"
    i2v_self_frame: Literal[True] = True
    action_button_temporal_alignment: Literal["causal_prefix_rope"] = (
        "causal_prefix_rope"
    )
    chunk_size: int = Field(default=2, ge=1)
    video_temporal_scale: int = Field(default=8, ge=1)
    audio_latent_fps: float = Field(default=25.0, gt=0.0)

    @model_validator(mode="after")
    def validate_block_prompt_contract(self) -> "BidirPromptScheduleConfig":
        if self.chunk_size != 2:
            raise ValueError(
                "The current five-block Xonotic Stage-2 contract requires chunk_size=2."
            )
        if self.synthetic_audio_state:
            raise ValueError(
                "Block-local Stage-2 training requires real aligned audio."
            )
        return self


def build_bidirectional_prompt_context_mask(
    blocks: list[AVBlock],
    *,
    kind: Literal["video", "audio"],
    tokens_per_latent: int,
    prompt_attention_mask: Tensor,
) -> Tensor:
    ''

    if prompt_attention_mask.ndim != 3:
        raise ValueError(
            "prompt_attention_mask must have shape [B, generated_blocks, prompt_tokens], "
            f"got {tuple(prompt_attention_mask.shape)}."
        )
    if tokens_per_latent <= 0:
        raise ValueError(
            f"tokens_per_latent must be positive, got {tokens_per_latent}."
        )
    if not blocks:
        raise ValueError("At least one AV block is required.")

    batch_size, prompt_count, prompt_length = prompt_attention_mask.shape
    expected_prompt_count = len(blocks) - 1
    if prompt_count != expected_prompt_count:
        raise ValueError(
            f"Expected {expected_prompt_count} per-chunk prompts, got {prompt_count}."
        )

    total_latents = blocks[-1].range_for(kind)[1]
    total_queries = total_latents * tokens_per_latent
    total_context = prompt_count * prompt_length
    routed = torch.zeros(
        batch_size,
        total_queries,
        total_context,
        dtype=torch.bool,
        device=prompt_attention_mask.device,
    )
    valid = prompt_attention_mask.to(dtype=torch.bool)
    for block in blocks:
        latent_start, latent_end = block.range_for(kind)
        query_start = latent_start * tokens_per_latent
        query_end = latent_end * tokens_per_latent
        prompt_index = max(block.index - 1, 0)
        context_start = prompt_index * prompt_length
        context_end = context_start + prompt_length
        routed[:, query_start:query_end, context_start:context_end] = valid[
            :, prompt_index, None, :
        ]
    return routed


class BidirPromptScheduleStrategy(PhaseAwareStrategy):
    ''

    config: BidirPromptScheduleConfig

    def __init__(self, config: BidirPromptScheduleConfig):
        super().__init__(config)

    @staticmethod
    def _batch_fps(batch: dict[str, Any]) -> float:
        fps_value = batch.get("fps")
        if torch.is_tensor(fps_value):
            flattened = fps_value.float().flatten()
            if flattened.numel() == 0:
                raise ValueError("The block-prompt batch contains an empty fps tensor.")
            if not torch.allclose(flattened, flattened[:1].expand_as(flattened)):
                raise ValueError(
                    f"Mixed FPS values are unsupported: {flattened.tolist()}"
                )
            return float(flattened[0].item())
        if isinstance(fps_value, (list, tuple)):
            if not fps_value:
                raise ValueError("The block-prompt batch contains an empty fps list.")
            values = [float(value) for value in fps_value]
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"Mixed FPS values are unsupported: {values}")
            return values[0]
        return float(fps_value) if fps_value is not None else 24.0

    @staticmethod
    def _validate_schedule(
        batch: dict[str, Any],
        *,
        batch_size: int,
        prompt_count: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        required = (
            "video_prompt_schedule_embeds",
            "audio_prompt_schedule_embeds",
            "prompt_schedule_attention_mask",
        )
        missing = [name for name in required if not torch.is_tensor(batch.get(name))]
        if missing:
            raise ValueError(
                "bidir_prompt_schedule requires an explicit manifest prompt_schedule; "
                f"missing encoded tensors: {missing}."
            )

        video_schedule = batch["video_prompt_schedule_embeds"]
        audio_schedule = batch["audio_prompt_schedule_embeds"]
        schedule_mask = batch["prompt_schedule_attention_mask"]
        expected_prefix = (batch_size, prompt_count)
        if video_schedule.ndim != 4 or video_schedule.shape[:2] != expected_prefix:
            raise ValueError(
                "video_prompt_schedule_embeds must have shape "
                f"[B, {prompt_count}, L, D], got {tuple(video_schedule.shape)}."
            )
        if audio_schedule.ndim != 4 or audio_schedule.shape[:2] != expected_prefix:
            raise ValueError(
                "audio_prompt_schedule_embeds must have shape "
                f"[B, {prompt_count}, L, D], got {tuple(audio_schedule.shape)}."
            )
        if video_schedule.shape[2] != audio_schedule.shape[2]:
            raise ValueError(
                "Video/audio prompt schedules must use the same token length, "
                f"got {video_schedule.shape[2]} and {audio_schedule.shape[2]}."
            )
        if schedule_mask.shape != video_schedule.shape[:3]:
            raise ValueError(
                "prompt_schedule_attention_mask must match the video schedule prefix: "
                f"{tuple(schedule_mask.shape)} vs {tuple(video_schedule.shape[:3])}."
            )
        return video_schedule, audio_schedule, schedule_mask

    @staticmethod
    def _align_audio_boundary_surplus(
        blocks: list[AVBlock], actual_audio_latents: int
    ) -> list[AVBlock]:
        expected_audio_latents = blocks[-1].audio_end
        surplus = actual_audio_latents - expected_audio_latents
        if surplus < 0 or surplus > 2:
            raise ValueError(
                "Audio/video latent geometry is outside the allowed causal-VAE boundary surplus: "
                f"actual={actual_audio_latents}, expected={expected_audio_latents}, "
                f"surplus={surplus}, allowed=0..2."
            )
        if surplus == 0:
            return blocks
        adjusted = list(blocks)
        adjusted[-1] = replace(adjusted[-1], audio_end=actual_audio_latents)
        return adjusted

    def prepare_training_inputs(
        self, batch: dict[str, Any], timestep_sampler
    ) -> ModelInputs:
        video_latents = batch.get("video_latents")
        if not torch.is_tensor(video_latents) or video_latents.ndim != 5:
            raise ValueError("video_latents must be a 5D tensor [B, C, T, H, W].")
        batch_size, _channels, video_frames, height, width = video_latents.shape
        fps = self._batch_fps(batch)
        blocks = compute_ti2av_blocks(
            video_frames,
            None,
            chunk_size=self.config.chunk_size,
            video_fps=fps,
            video_temporal_scale=self.config.video_temporal_scale,
            audio_latent_fps=self.config.audio_latent_fps,
        )
        prompt_count = len(blocks) - 1
        video_schedule, audio_schedule, schedule_mask = self._validate_schedule(
            batch,
            batch_size=batch_size,
            prompt_count=prompt_count,
        )

                                                                            
                                                                         
                                                                     
        local_batch = dict(batch)
        local_batch["video_prompt_embeds"] = video_schedule.flatten(1, 2)
        local_batch["audio_prompt_embeds"] = audio_schedule.flatten(1, 2)
        local_batch["prompt_attention_mask"] = schedule_mask.flatten(1, 2)
        inputs = super().prepare_training_inputs(local_batch, timestep_sampler)

        if inputs.audio is None:
            raise RuntimeError("bidir_prompt_schedule requires the audio modality.")
        routed_schedule_mask = inputs.video.context_mask
        if routed_schedule_mask is None:
            raise RuntimeError(
                "PhaseAware preparation unexpectedly removed the prompt attention mask."
            )
        prompt_length = schedule_mask.shape[2]
        routed_schedule_mask = routed_schedule_mask.reshape(
            batch_size, prompt_count, prompt_length
        )

        blocks = self._align_audio_boundary_surplus(
            blocks, inputs.audio.latent.shape[1]
        )
        video_context_mask = build_bidirectional_prompt_context_mask(
            blocks,
            kind="video",
            tokens_per_latent=height * width,
            prompt_attention_mask=routed_schedule_mask,
        )
        audio_context_mask = build_bidirectional_prompt_context_mask(
            blocks,
            kind="audio",
            tokens_per_latent=1,
            prompt_attention_mask=routed_schedule_mask,
        )
        inputs.video = replace(inputs.video, context_mask=video_context_mask)
        inputs.audio = replace(inputs.audio, context_mask=audio_context_mask)
        self._last_inputs = inputs
        return inputs

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "training_strategy": "bidir_prompt_schedule",
            "prompt_policy": "block_local",
            "prompt_schedule_length": 5,
            "chunk_size": self.config.chunk_size,
            "video_attention": "bidirectional",
            "audio_attention": "bidirectional",
            "av_cross_attention": "bidirectional",
            "action_button_temporal_alignment": self.config.action_button_temporal_alignment,
        }


__all__ = [
    "BidirPromptScheduleConfig",
    "BidirPromptScheduleStrategy",
    "build_bidirectional_prompt_context_mask",
]
