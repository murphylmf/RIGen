''

from __future__ import annotations

from typing import Any, Literal

import torch
from ltx_core.model.transformer.modality import Modality
from ltx_core.utils import to_denoised
from pydantic import Field, model_validator
from torch import Tensor

from ltx_trainer.training_strategies.ar_sliding_ref_masks import (
    build_sliding_ref_cross_mask,
    build_sliding_ref_prompt_context_mask,
    build_sliding_ref_prompt_context_timesteps,
    build_sliding_ref_self_mask,
)
from ltx_trainer.training_strategies.ar_teacher import ARTeacherConfig, ARTeacherStrategy
from ltx_trainer.training_strategies.ar_teacher_masks import (
    compute_ti2av_blocks,
    expand_block_values_to_tokens,
)
from ltx_trainer.training_strategies.base_strategy import LossOutputs, ModelInputs


class ARSlidingRefConfig(ARTeacherConfig):
    ''

    name: Literal["ar_sliding_ref"] = "ar_sliding_ref"
    execution_mode: Literal["full_mask"] = "full_mask"
    prompt_execution_mode: Literal["masked_schedule"] = "masked_schedule"
    mask_backend: Literal["dense"] = "dense"
    history_chunks: Literal[4] = 4
    reference_frames: Literal[1] = 1
    reference_token_order: Literal["suffix"] = "suffix"
    local_rope_rebase: Literal[True] = True
    require_reference_contract: bool = Field(
        default=True,
        description="Require the long manifest's active_ref_schedule/reference_bank contract.",
    )

    @model_validator(mode="after")
    def validate_sliding_ref_contract(self) -> "ARSlidingRefConfig":
        if self.chunk_size != 2:
            raise ValueError("Sliding-ref training currently requires chunk_size=2.")
        if self.history_chunks != 4:
            raise ValueError("Sliding-ref training currently requires exactly four history chunks.")
        if self.reference_frames != 1:
            raise ValueError("Sliding-ref training currently requires one fixed-view reference frame.")
        return self


class ARSlidingRefStrategy(ARTeacherStrategy):
    ''

    config: ARSlidingRefConfig

    def __init__(self, config: ARSlidingRefConfig):
        super().__init__(config)
        self._sliding_mask_cache: dict[tuple[Any, ...], dict[str, Tensor]] = {}

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "training_strategy": "ar_sliding_ref",
            "ar_teacher_task": "ti2av_sliding_ref",
            "ar_teacher_chunk_size": self.config.chunk_size,
            "ar_teacher_first_video_block_latents": 1,
            "ar_teacher_context_policy": "global_sink_plus_four_chunk_sliding_window",
            "ar_teacher_execution_mode": "full_mask",
            "ar_teacher_prompt_execution_mode": "masked_schedule",
            "ar_teacher_prompt_policy": "local_per_chunk_causal",
            "ar_teacher_loss_target": "target_chunk_x0_only",
            "sliding_history_chunks": self.config.history_chunks,
            "reference_frames": self.config.reference_frames,
            "reference_view_policy": "manifest_default_single_view",
            "reference_token_order": self.config.reference_token_order,
            "reference_attention_policy": "target_reads_ref_ref_self_only",
            "rope_policy": "local_window_rebase_sink_and_ref_time_zero",
            "action_button_temporal_alignment": self.config.action_button_temporal_alignment,
        }

    def _preserve_prompt_routing_for_unconditional(self) -> bool:
                                                                              
                                                                             
                                                                               
        return True

    def _get_sliding_masks(
        self,
        *,
        blocks: list[Any],
        spatial_tokens: int,
        reference_tokens: int,
        device: torch.device,
    ) -> dict[str, Tensor]:
        key = (
            tuple((block.video_start, block.video_end, block.audio_start, block.audio_end) for block in blocks),
            spatial_tokens,
            reference_tokens,
            device.type,
            device.index,
        )
        cached = self._sliding_mask_cache.get(key)
        if cached is not None:
            return cached
        masks = {
            "video_self": build_sliding_ref_self_mask(
                blocks,
                kind="video",
                tokens_per_latent=spatial_tokens,
                reference_tokens=reference_tokens,
                device=device,
            ),
            "audio_self": build_sliding_ref_self_mask(
                blocks,
                kind="audio",
                tokens_per_latent=1,
                device=device,
            ),
            "a2v": build_sliding_ref_cross_mask(
                blocks,
                query_kind="video",
                key_kind="audio",
                query_tokens_per_latent=spatial_tokens,
                key_tokens_per_latent=1,
                query_reference_tokens=reference_tokens,
                device=device,
            ),
            "v2a": build_sliding_ref_cross_mask(
                blocks,
                query_kind="audio",
                key_kind="video",
                query_tokens_per_latent=1,
                key_tokens_per_latent=spatial_tokens,
                key_reference_tokens=reference_tokens,
                device=device,
            ),
        }
        self._sliding_mask_cache[key] = masks
        return masks

    @staticmethod
    def _reference_present(batch: dict[str, Any], *, batch_size: int, device: torch.device) -> Tensor:
        value = batch.get("reference_present")
        if value is None:
            raise ValueError("Sliding-ref training requires reference_present metadata.")
        if torch.is_tensor(value):
            present = value.to(device=device, dtype=torch.bool).flatten()
        elif isinstance(value, list):
            present = torch.tensor(value, device=device, dtype=torch.bool).flatten()
        else:
            present = torch.tensor([bool(value)], device=device, dtype=torch.bool)
        if present.shape != (batch_size,):
            raise ValueError(f"reference_present must have shape [{batch_size}], got {tuple(present.shape)}.")
        return present

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: Any,
    ) -> ModelInputs:
        video_latents = batch["video_latents"]
        reference_latents = batch.get("reference_video_latents")
        if not torch.is_tensor(reference_latents):
            raise ValueError("Sliding-ref training requires online-encoded reference_video_latents.")

        batch_size, channels, video_frames, height, width = video_latents.shape
        ref_batch, ref_channels, ref_frames, ref_height, ref_width = reference_latents.shape
                                                                            
                                                                           
                                                                              
                                                                     
        inference_history_chunks = batch.get("_sliding_ref_inference_history_chunks")
        active_history_chunks = (
            self.config.history_chunks
            if inference_history_chunks is None
            else int(inference_history_chunks)
        )
        if not 0 <= active_history_chunks <= self.config.history_chunks:
            raise ValueError(
                "_sliding_ref_inference_history_chunks must lie in "
                f"[0, {self.config.history_chunks}], got {active_history_chunks}."
            )
        expected_video_frames = 1 + (active_history_chunks + 1) * self.config.chunk_size
        if video_frames != expected_video_frames:
            raise ValueError(
                "Sliding-ref video geometry mismatch: "
                f"expected {expected_video_frames} latents, got {video_frames}."
            )
        if (ref_batch, ref_channels, ref_frames) != (batch_size, channels, self.config.reference_frames):
            raise ValueError(
                "Sliding-ref reference geometry mismatch: "
                f"world={tuple(video_latents.shape)} ref={tuple(reference_latents.shape)}."
            )

        fps = self._batch_fps(batch)
        clean_world = self._video_patchifier.patchify(video_latents)
        clean_reference = self._video_patchifier.patchify(
            reference_latents.to(device=clean_world.device, dtype=clean_world.dtype)
        )
        clean_video = torch.cat([clean_world, clean_reference], dim=1)
        device = clean_video.device
        dtype = clean_video.dtype
        spatial_tokens = height * width
        reference_tokens = ref_frames * ref_height * ref_width
        world_tokens = clean_world.shape[1]
        reference_start = world_tokens
        reference_present = self._reference_present(batch, batch_size=batch_size, device=device)

        blocks = compute_ti2av_blocks(
            video_frames,
            None,
            chunk_size=self.config.chunk_size,
            video_fps=fps,
            video_temporal_scale=self.config.video_temporal_scale,
            audio_latent_fps=self.config.audio_latent_fps,
        )
        if len(blocks) != active_history_chunks + 2:
            raise ValueError(
                f"Expected sink + {active_history_chunks} history + target blocks, got {len(blocks)}."
            )
        target_video_start = blocks[-1].video_start * spatial_tokens
        target_video_end = blocks[-1].video_end * spatial_tokens

        target_audio_frames = blocks[-1].audio_end
        clean_audio, audio_loss_eligible = self._prepare_clean_audio(
            batch,
            batch_size=batch_size,
            target_audio_frames=target_audio_frames,
            device=device,
            dtype=dtype,
        )
        audio_frames = clean_audio.shape[1]

        sampled_sigma = timestep_sampler.sample(
            batch_size=batch_size,
            seq_length=target_video_end - target_video_start,
            device=device,
        ).to(device=device, dtype=torch.float32).reshape(batch_size)
        block_sigmas = torch.zeros(batch_size, len(blocks), device=device, dtype=torch.float32)
        block_sigmas[:, -1] = sampled_sigma
        world_video_sigmas = expand_block_values_to_tokens(
            block_sigmas,
            blocks,
            kind="video",
            tokens_per_latent=spatial_tokens,
        )
        video_timesteps = torch.cat(
            [world_video_sigmas, torch.zeros(batch_size, reference_tokens, device=device)],
            dim=1,
        )
        audio_timesteps = expand_block_values_to_tokens(
            block_sigmas,
            blocks,
            kind="audio",
            tokens_per_latent=1,
        )

        video_sigma = video_timesteps.unsqueeze(-1).to(dtype=dtype)
        audio_sigma = audio_timesteps.unsqueeze(-1).to(dtype=dtype)
        video_input = (1 - video_sigma) * clean_video + video_sigma * torch.randn_like(clean_video)
        audio_input = (1 - audio_sigma) * clean_audio + audio_sigma * torch.randn_like(clean_audio)

        world_positions = self._get_video_positions(
            num_frames=video_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )
        reference_positions = self._get_video_positions(
            num_frames=ref_frames,
            height=ref_height,
            width=ref_width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        ).clone()
                                                                         
                                                                             
        reference_positions[:, 1] *= float(height) / float(ref_height)
        reference_positions[:, 2] *= float(width) / float(ref_width)
        video_positions = torch.cat([world_positions, reference_positions], dim=2)
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_frames,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        video_schedule, audio_schedule, schedule_mask = self._prepare_prompt_schedule(
            batch,
            generated_blocks=len(blocks) - 1,
            batch_size=batch_size,
            device=device,
        )
        video_context = video_schedule.flatten(1, 2)
        audio_context = audio_schedule.flatten(1, 2)
        context_timesteps = build_sliding_ref_prompt_context_timesteps(block_sigmas, schedule_mask)
        video_context_mask = build_sliding_ref_prompt_context_mask(
            blocks,
            kind="video",
            tokens_per_latent=spatial_tokens,
            prompt_attention_mask=schedule_mask,
            reference_tokens=reference_tokens,
        )
        audio_context_mask = build_sliding_ref_prompt_context_mask(
            blocks,
            kind="audio",
            tokens_per_latent=1,
            prompt_attention_mask=schedule_mask,
        )

        structural_masks = self._get_sliding_masks(
            blocks=blocks,
            spatial_tokens=spatial_tokens,
            reference_tokens=reference_tokens,
            device=device,
        )
        video_self = structural_masks["video_self"].unsqueeze(0).expand(batch_size, -1, -1).clone()
                                                                           
                                                       
        if torch.any(~reference_present):
            video_self[
                ~reference_present,
                target_video_start:target_video_end,
                reference_start:,
            ] = False
        audio_self = structural_masks["audio_self"].unsqueeze(0).expand(batch_size, -1, -1)
        a2v_mask = structural_masks["a2v"].unsqueeze(0).expand(batch_size, -1, -1)
        v2a_mask = structural_masks["v2a"].unsqueeze(0).expand(batch_size, -1, -1)

        representative_sigma = sampled_sigma
        video_modality = Modality(
            enabled=True,
            latent=video_input,
            sigma=representative_sigma,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_context,
            context_mask=video_context_mask,
            attention_mask=video_self,
            cross_attention_mask=a2v_mask,
            cross_attention_timesteps=video_timesteps,
            context_timesteps=context_timesteps,
        )
        audio_modality = Modality(
            enabled=True,
            latent=audio_input,
            sigma=representative_sigma,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_context,
            context_mask=audio_context_mask,
            attention_mask=audio_self,
            cross_attention_mask=v2a_mask,
            cross_attention_timesteps=audio_timesteps,
            context_timesteps=context_timesteps,
        )

        action_joystick_tokens, action_button_state, action_button_values = self._prepare_action_tokens(
            batch,
            latent_frames=video_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        action_joystick_tokens, action_button_state, action_button_values = self._apply_action_dropout(
            action_joystick_tokens,
            action_button_state,
            action_button_values,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        joystick_present = self._sample_condition_present_mask(
            action_joystick_tokens,
            uncond_prob=self.config.action_joystick_uncond_prob,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        button_present = self._sample_condition_present_mask(
            action_button_state,
            uncond_prob=self.config.action_button_uncond_prob,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        action_button_meaning_embeds = self._prepare_button_meaning_embeds(
            batch,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        action_button_state, action_button_meaning_embeds = self._apply_action_binding_dropout(
            action_button_state,
            action_button_meaning_embeds,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        action_token_mask = torch.ones(batch_size, clean_video.shape[1], 1, device=device, dtype=dtype)
        action_token_mask[:, :spatial_tokens] = 0               
        action_token_mask[:, reference_start:] = 0                      
        if action_joystick_tokens is not None:
            action_joystick_tokens = torch.cat(
                [
                    action_joystick_tokens,
                    torch.zeros(batch_size, reference_tokens, 32, device=device, dtype=dtype),
                ],
                dim=1,
            )
        action_joystick_condition_mask = (
            action_token_mask * joystick_present if joystick_present is not None else None
        )
        action_button_condition_mask = action_token_mask * button_present if button_present is not None else None

        target_video_mask = torch.zeros(batch_size, clean_video.shape[1], dtype=torch.bool, device=device)
        target_video_mask[:, target_video_start:target_video_end] = True
        target_audio_mask = torch.zeros(batch_size, audio_frames, dtype=torch.bool, device=device)
        target_audio_start = blocks[-1].audio_start
        target_audio_end = blocks[-1].audio_end
        target_audio_mask[:, target_audio_start:target_audio_end] = audio_loss_eligible[:, None]

        video_loss_weights = torch.ones_like(video_timesteps)
        audio_loss_weights = torch.ones_like(audio_timesteps)
        if self.config.use_gaussian_loss_weight:
            video_loss_weights = torch.exp(-2.0 * (video_timesteps - 0.5).square())
            audio_loss_weights = torch.exp(-2.0 * (audio_timesteps - 0.5).square())

        world_query_indices = torch.arange(video_frames, device=device, dtype=torch.long).repeat_interleave(
            spatial_tokens
        )
        action_query_indices = torch.cat(
            [world_query_indices, torch.zeros(reference_tokens, device=device, dtype=torch.long)]
        )
        inputs = ModelInputs(
            video=video_modality,
            audio=audio_modality,
            action_joystick_tokens=action_joystick_tokens,
            action_joystick_condition_mask=action_joystick_condition_mask,
            action_button_state=action_button_state,
            action_button_values=action_button_values,
            action_button_condition_mask=action_button_condition_mask,
            action_button_meaning_embeds=action_button_meaning_embeds,
            action_button_temporal_alignment=self.config.action_button_temporal_alignment,
            action_button_prefix_tokens=0,
            action_button_frame_rate=fps,
            action_button_query_latent_indices=action_query_indices,
            first_frame_latent=video_latents[:, :, :1].clone(),
            video_targets=clean_video,
            audio_targets=clean_audio,
            video_loss_mask=target_video_mask,
            audio_loss_mask=target_audio_mask,
            video_loss_weights=video_loss_weights,
            audio_loss_weights=audio_loss_weights,
            ref_seq_len=reference_tokens,
            video_clean_latent=clean_video,
            video_noisy_latent=video_input,
            audio_clean_latent=clean_audio,
            audio_noisy_latent=audio_input,
            ar_video_block_token_ranges=tuple(
                (block.video_start * spatial_tokens, block.video_end * spatial_tokens) for block in blocks
            ),
            ar_audio_block_token_ranges=tuple((block.audio_start, block.audio_end) for block in blocks),
        )
        self._last_inputs = inputs
        return inputs

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> LossOutputs:
        if inputs.video_targets is None or inputs.video_loss_mask is None or inputs.video_loss_weights is None:
            raise ValueError("Sliding-ref video targets/masks/weights are missing.")
        predicted_video_x0 = to_denoised(inputs.video.latent, video_pred, inputs.video.timesteps)
        video_loss = self._weighted_token_mse(
            predicted_video_x0,
            inputs.video_targets,
            inputs.video_loss_mask,
            inputs.video_loss_weights,
        )

        audio_loss_scalar = torch.zeros((), device=video_pred.device, dtype=torch.float32)
        total = video_loss
        if audio_pred is not None:
            if (
                inputs.audio is None
                or inputs.audio_targets is None
                or inputs.audio_loss_mask is None
                or inputs.audio_loss_weights is None
            ):
                raise ValueError("Sliding-ref audio prediction was returned without targets/masks/weights.")
            predicted_audio_x0 = to_denoised(inputs.audio.latent, audio_pred, inputs.audio.timesteps)
            audio_loss = self._weighted_token_mse(
                predicted_audio_x0,
                inputs.audio_targets,
                inputs.audio_loss_mask,
                inputs.audio_loss_weights,
            )
            has_audio = inputs.audio_loss_mask.any(dim=1).float()
            audio_scale = has_audio.numel() / has_audio.sum().clamp(min=1.0)
            total = total + self.config.loss_alpha_audio * audio_loss * has_audio * audio_scale
            if torch.any(has_audio):
                audio_loss_scalar = audio_loss[has_audio.bool()].mean()

        return LossOutputs(
            total=total,
            metrics={
                "loss": total.mean(),
                "loss_video_x0": video_loss.mean(),
                "loss_audio_x0": audio_loss_scalar,
                "sliding/target_sigma": inputs.video.sigma.float().mean(),
            },
        )

    def collect_diagnostics(self, *args: Any, **kwargs: Any) -> None:
        return None


__all__ = ["ARSlidingRefConfig", "ARSlidingRefStrategy"]
