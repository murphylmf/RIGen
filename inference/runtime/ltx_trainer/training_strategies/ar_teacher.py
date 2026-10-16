''

from __future__ import annotations

from typing import Any, Literal

import torch
from ltx_core.model.transformer.modality import Modality
from ltx_core.tools import AudioLatentTools
from ltx_core.utils import to_denoised
from pydantic import Field, model_validator
from torch import Tensor

from ltx_trainer.training_strategies.ar_teacher_masks import (
    AVBlock,
    MaskBackend,
    build_action_query_latent_indices,
    build_prompt_context_mask,
    build_prompt_context_timesteps,
    build_teacher_forcing_cross_mask,
    build_teacher_forcing_self_mask,
    compute_ti2av_blocks,
    expand_block_values_to_tokens,
)
from ltx_trainer.training_strategies.base_strategy import LossOutputs, ModelInputs
from ltx_trainer.training_strategies.phase_aware import PhaseAwareConfig, PhaseAwareStrategy


class ARTeacherConfig(PhaseAwareConfig):
    ''

    name: Literal["ar_teacher"] = "ar_teacher"
    forward_mode: Literal["audio_video"] = "audio_video"
    i2v_self_frame: Literal[True] = True
    action_button_temporal_alignment: Literal["causal_prefix_rope"] = "causal_prefix_rope"
    chunk_size: int = Field(default=2, ge=1)
    mask_backend: MaskBackend = "auto"
    execution_mode: Literal["full_mask", "sequential_kv"] = Field(
        default="full_mask",
        description=(
            "Transformer execution geometry. full_mask preserves the established doubled-sequence path; "
            "sequential_kv is an opt-in compact cache-aware training experiment."
        ),
    )
    prompt_execution_mode: Literal["masked_schedule", "compact_current"] = Field(
        default="masked_schedule",
        description=(
            "Text-context execution geometry. masked_schedule retains all per-chunk prompt copies and routes "
            "queries with a mask. compact_current physically selects only the current chunk prompt and removes "
            "the prompt mask before every sequential forward."
        ),
    )
    loss_alpha_audio: float = Field(default=0.16, ge=0.0)
    use_gaussian_loss_weight: bool = True
    video_temporal_scale: int = Field(default=8, ge=1)
    audio_latent_fps: float = Field(default=25.0, gt=0.0)

    @model_validator(mode="after")
    def validate_ar_teacher_contract(self) -> "ARTeacherConfig":
        if self.chunk_size != 2:
            raise ValueError("The current TI2AV AR-teacher contract requires chunk_size=2.")
        if self.synthetic_audio_state:
            raise ValueError("AR teacher training requires real aligned audio; synthetic_audio_state must be false.")
        if self.execution_mode == "sequential_kv" and self.prompt_execution_mode != "compact_current":
            raise ValueError(
                "sequential_kv requires prompt_execution_mode='compact_current'; retaining a masked global "
                "prompt schedule would not match online chunk-by-chunk inference."
            )
        if self.execution_mode != "sequential_kv" and self.prompt_execution_mode != "masked_schedule":
            raise ValueError("compact_current prompt execution is only valid with execution_mode='sequential_kv'.")
        if self.prompt_execution_mode == "compact_current":
            if self.cfg_training_enabled or self.joint_uncond_prob != 0.0:
                raise ValueError(
                    "compact_current requires cfg_training_enabled=false and joint_uncond_prob=0: the current "
                    "zero-mask unconditional representation would create an empty prompt and cannot use mask-free FA."
                )
            if self.conditioning_dropout_p != 0.0 or self.text_dropout_prob not in (None, 0.0):
                raise ValueError(
                    "compact_current requires all text-conditioning dropout to be disabled so every physical "
                    "current prompt is non-empty and mask-free."
                )
        return self


class ARTeacherStrategy(PhaseAwareStrategy):
    ''

    config: ARTeacherConfig

    def __init__(self, config: ARTeacherConfig):
        super().__init__(config)
        self._structural_mask_cache: dict[tuple[Any, ...], dict[str, Tensor | object]] = {}

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "training_strategy": "ar_teacher",
            "ar_teacher_task": "ti2av",
            "ar_teacher_chunk_size": self.config.chunk_size,
            "ar_teacher_first_video_block_latents": 1,
            "ar_teacher_context_policy": "full_prefix",
            "ar_teacher_execution_mode": self.config.execution_mode,
            "ar_teacher_prompt_execution_mode": self.config.prompt_execution_mode,
            "ar_teacher_prompt_policy": "per_chunk",
            "ar_teacher_loss_target": "x0",
            "action_button_temporal_alignment": self.config.action_button_temporal_alignment,
        }

    @staticmethod
    def _batch_fps(batch: dict[str, Any]) -> float:
        fps_value = batch.get("fps")
        if torch.is_tensor(fps_value):
            flattened = fps_value.float().flatten()
            if flattened.numel() == 0:
                raise ValueError("The AR teacher batch contains an empty fps tensor.")
            if not torch.allclose(flattened, flattened[:1].expand_as(flattened)):
                raise ValueError(f"Mixed FPS values in one AR teacher batch are unsupported: {flattened.tolist()}")
            return float(flattened[0].item())
        if isinstance(fps_value, list):
            values = [float(value) for value in fps_value]
            if not values or any(value != values[0] for value in values[1:]):
                raise ValueError(f"Mixed or empty FPS values in one AR teacher batch: {values}")
            return values[0]
        if fps_value is None:
            raise ValueError("AR teacher training requires explicit FPS metadata.")
        return float(fps_value)

    def _sample_block_sigmas(
        self,
        *,
        batch_size: int,
        blocks: list[AVBlock],
        sequence_length: int,
        device: torch.device,
        timestep_sampler: Any,
    ) -> Tensor:
                                                                              
                                                                     
        sigmas = torch.zeros(batch_size, len(blocks), device=device, dtype=torch.float32)
        for block in blocks[1:]:
            sampled = timestep_sampler.sample(
                batch_size=batch_size,
                seq_length=sequence_length,
                device=device,
            )
            sigmas[:, block.index] = sampled.to(device=device, dtype=torch.float32).reshape(batch_size)
        return sigmas

    def _prepare_clean_audio(
        self,
        batch: dict[str, Any],
        *,
        batch_size: int,
        target_audio_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        has_audio = batch.get("has_audio")
        if has_audio is None:
            has_audio = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            has_audio = has_audio.to(device=device, dtype=torch.bool)
        if not torch.all(has_audio):
            raise ValueError("TI2AV AR teacher training requires aligned audio for every sample.")

        audio_latents = batch.get("audio_latents")
        if isinstance(audio_latents, Tensor):
            clean_audio = audio_latents.to(device=device, dtype=dtype)
        else:
            if not self.config.synthetic_audio_state:
                raise ValueError("TI2AV AR teacher training requires audio_latents.")
            shape = self._infer_audio_shape(batch)._replace(batch=batch_size)
            clean_audio = (
                AudioLatentTools(
                    patchifier=self._audio_patchifier,
                    target_shape=shape,
                )
                .create_initial_state(device=device, dtype=dtype)
                .latent
            )
            clean_audio = self._audio_patchifier.unpatchify(clean_audio, output_shape=shape)

        audio_loss_eligible = batch.get("audio_loss_eligible")
        if audio_loss_eligible is None:
            audio_loss_eligible = has_audio
        else:
            audio_loss_eligible = audio_loss_eligible.to(device=device, dtype=torch.bool) & has_audio

        clean_audio_patch = self._audio_patchifier.patchify(clean_audio)
        actual_audio_frames = int(clean_audio_patch.shape[1])
        if actual_audio_frames < target_audio_frames:
            raise ValueError(
                "AR teacher audio latents are shorter than the required TI2AV geometry: "
                f"actual={actual_audio_frames}, target={target_audio_frames}. "
                "Refusing to silently zero-pad the sequence."
            )
        excess_audio_frames = actual_audio_frames - target_audio_frames
        if excess_audio_frames > 2:
            raise ValueError(
                "AR teacher audio latents exceed the required TI2AV geometry by more than the "
                f"allowed causal-VAE boundary surplus: actual={actual_audio_frames}, "
                f"target={target_audio_frames}, surplus={excess_audio_frames}, allowed<=2."
            )
        if excess_audio_frames:
            clean_audio_patch = clean_audio_patch[:, :target_audio_frames].clone()
        else:
            clean_audio_patch = clean_audio_patch.clone()
                                                                            
                                                                         
        clean_audio_patch[:, :1] = 0
        return clean_audio_patch, audio_loss_eligible

    def _prepare_prompt_schedule(
        self,
        batch: dict[str, Any],
        *,
        generated_blocks: int,
        batch_size: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        schedule_mask = batch.get("prompt_schedule_attention_mask")
        video_schedule = batch.get("video_prompt_schedule_embeds")
        audio_schedule = batch.get("audio_prompt_schedule_embeds")
        if video_schedule is None:
            video_prompt = batch["video_prompt_embeds"]
            audio_prompt = batch.get("audio_prompt_embeds", video_prompt)
            base_mask = batch["prompt_attention_mask"]
            video_schedule = video_prompt[:, None].expand(-1, generated_blocks, -1, -1)
            audio_schedule = audio_prompt[:, None].expand(-1, generated_blocks, -1, -1)
            schedule_mask = base_mask[:, None].expand(-1, generated_blocks, -1)
        elif schedule_mask is None:
            raise ValueError("Prompt schedule embeddings require prompt_schedule_attention_mask.")

        if audio_schedule is None:
            audio_schedule = video_schedule
                                                                              
                                                                              
                                                                                
        video_schedule = video_schedule.to(device=device)
        audio_schedule = audio_schedule.to(device=device)
        schedule_mask = schedule_mask.to(device=device, dtype=torch.bool)
        expected_prefix = (batch_size, generated_blocks)
        if video_schedule.ndim != 4 or video_schedule.shape[:2] != expected_prefix:
            raise ValueError(
                "video_prompt_schedule_embeds must have shape "
                f"[B, {generated_blocks}, L, D], got {tuple(video_schedule.shape)}."
            )
        if audio_schedule.ndim != 4 or audio_schedule.shape[:2] != expected_prefix:
            raise ValueError(
                "audio_prompt_schedule_embeds must have shape "
                f"[B, {generated_blocks}, L, D], got {tuple(audio_schedule.shape)}."
            )
        if audio_schedule.shape[2] != video_schedule.shape[2]:
            raise ValueError(
                "Video/audio prompt schedules must use the same token length, "
                f"got {video_schedule.shape[2]} and {audio_schedule.shape[2]}."
            )
        if schedule_mask.shape != video_schedule.shape[:3]:
            raise ValueError(
                "prompt_schedule_attention_mask must match the video schedule prefix: "
                f"{tuple(schedule_mask.shape)} vs {tuple(video_schedule.shape[:3])}."
            )

        if self.config.cfg_training_enabled:
            drop_mask = torch.rand(batch_size, device=device) < self.config.joint_uncond_prob
        else:
            drop_mask = torch.rand(batch_size, device=device) < self.config.conditioning_dropout_p
        if torch.any(drop_mask):
            video_schedule = video_schedule.clone()
            audio_schedule = audio_schedule.clone()
            schedule_mask = schedule_mask.clone()
            video_schedule[drop_mask] = 0
            audio_schedule[drop_mask] = 0
            if not self._preserve_prompt_routing_for_unconditional():
                schedule_mask[drop_mask] = False
        return video_schedule, audio_schedule, schedule_mask

    def _preserve_prompt_routing_for_unconditional(self) -> bool:
        ''

        return False

    def _get_structural_masks(
        self,
        *,
        blocks: list[AVBlock],
        spatial_tokens: int,
        device: torch.device,
    ) -> dict[str, Tensor | object]:
        audio_tokens_per_latent = 1
        key = (
            tuple((block.video_start, block.video_end, block.audio_start, block.audio_end) for block in blocks),
            spatial_tokens,
            device.type,
            device.index,
            self.config.mask_backend,
        )
        cached = self._structural_mask_cache.get(key)
        if cached is not None:
            return cached
        masks: dict[str, Tensor | object] = {
            "video_self": build_teacher_forcing_self_mask(
                blocks,
                kind="video",
                tokens_per_latent=spatial_tokens,
                device=device,
                backend=self.config.mask_backend,
            ),
            "audio_self": build_teacher_forcing_self_mask(
                blocks,
                kind="audio",
                tokens_per_latent=audio_tokens_per_latent,
                device=device,
                backend=self.config.mask_backend,
            ),
            "a2v": build_teacher_forcing_cross_mask(
                blocks,
                query_kind="video",
                key_kind="audio",
                query_tokens_per_latent=spatial_tokens,
                key_tokens_per_latent=audio_tokens_per_latent,
                device=device,
            ),
            "v2a": build_teacher_forcing_cross_mask(
                blocks,
                query_kind="audio",
                key_kind="video",
                query_tokens_per_latent=audio_tokens_per_latent,
                key_tokens_per_latent=spatial_tokens,
                device=device,
            ),
        }
        self._structural_mask_cache[key] = masks
        return masks

    @staticmethod
    def _as_modality_self_masks(
        mask: Tensor | object,
        *,
        batch_size: int,
    ) -> tuple[Tensor | None, object | None]:
        if torch.is_tensor(mask):
            return mask.unsqueeze(0).expand(batch_size, -1, -1), None
        return None, mask

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: Any,
    ) -> ModelInputs:
        video_latents = batch["video_latents"]
        batch_size, _channels, video_frames, height, width = video_latents.shape
        fps = self._batch_fps(batch)
        clean_video = self._video_patchifier.patchify(video_latents)
        device = clean_video.device
        dtype = clean_video.dtype
        spatial_tokens = height * width

        blocks = compute_ti2av_blocks(
            video_frames,
            None,
            chunk_size=self.config.chunk_size,
            video_fps=fps,
            video_temporal_scale=self.config.video_temporal_scale,
            audio_latent_fps=self.config.audio_latent_fps,
        )
        target_audio_frames = blocks[-1].audio_end
        clean_audio, audio_loss_eligible = self._prepare_clean_audio(
            batch,
            batch_size=batch_size,
            target_audio_frames=target_audio_frames,
            device=device,
            dtype=dtype,
        )
        audio_frames = clean_audio.shape[1]
        block_sigmas = self._sample_block_sigmas(
            batch_size=batch_size,
            blocks=blocks,
            sequence_length=clean_video.shape[1],
            device=device,
            timestep_sampler=timestep_sampler,
        )

        video_sigma_tokens = expand_block_values_to_tokens(
            block_sigmas,
            blocks,
            kind="video",
            tokens_per_latent=spatial_tokens,
        )
        audio_sigma_tokens = expand_block_values_to_tokens(
            block_sigmas,
            blocks,
            kind="audio",
            tokens_per_latent=1,
        )
        noisy_video = (
            1 - video_sigma_tokens.unsqueeze(-1).to(dtype=dtype)
        ) * clean_video + video_sigma_tokens.unsqueeze(-1).to(dtype=dtype) * torch.randn_like(clean_video)
        noisy_audio = (
            1 - audio_sigma_tokens.unsqueeze(-1).to(dtype=dtype)
        ) * clean_audio + audio_sigma_tokens.unsqueeze(-1).to(dtype=dtype) * torch.randn_like(clean_audio)
        video_input = torch.cat([clean_video, noisy_video], dim=1)
        audio_input = torch.cat([clean_audio, noisy_audio], dim=1)
        video_timesteps = torch.cat([torch.zeros_like(video_sigma_tokens), video_sigma_tokens], dim=1)
        audio_timesteps = torch.cat([torch.zeros_like(audio_sigma_tokens), audio_sigma_tokens], dim=1)
        video_targets = torch.cat([clean_video, clean_video], dim=1)
        audio_targets = torch.cat([clean_audio, clean_audio], dim=1)

        video_positions = self._get_video_positions(
            num_frames=video_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )
        video_positions = torch.cat([video_positions, video_positions], dim=2)
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_frames,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        audio_positions = torch.cat([audio_positions, audio_positions], dim=2)

        video_schedule, audio_schedule, schedule_mask = self._prepare_prompt_schedule(
            batch,
            generated_blocks=len(blocks) - 1,
            batch_size=batch_size,
            device=device,
        )
        video_prompt_half = video_schedule.flatten(1, 2)
        audio_prompt_half = audio_schedule.flatten(1, 2)
        video_context = torch.cat([video_prompt_half, video_prompt_half], dim=1)
        audio_context = torch.cat([audio_prompt_half, audio_prompt_half], dim=1)
        context_timesteps = build_prompt_context_timesteps(block_sigmas, schedule_mask)
        video_context_mask = build_prompt_context_mask(
            blocks,
            kind="video",
            tokens_per_latent=spatial_tokens,
            prompt_attention_mask=schedule_mask,
        )
        audio_context_mask = build_prompt_context_mask(
            blocks,
            kind="audio",
            tokens_per_latent=1,
            prompt_attention_mask=schedule_mask,
        )

        video_dense_self: Tensor | None = None
        video_block_self: object | None = None
        audio_dense_self: Tensor | None = None
        audio_block_self: object | None = None
        a2v_mask: Tensor | None = None
        v2a_mask: Tensor | None = None
        if self.config.execution_mode == "full_mask":
            masks = self._get_structural_masks(
                blocks=blocks,
                spatial_tokens=spatial_tokens,
                device=device,
            )
            video_dense_self, video_block_self = self._as_modality_self_masks(
                masks["video_self"],
                batch_size=batch_size,
            )
            audio_dense_self, audio_block_self = self._as_modality_self_masks(
                masks["audio_self"],
                batch_size=batch_size,
            )
            a2v_mask = masks["a2v"].unsqueeze(0).expand(batch_size, -1, -1)
            v2a_mask = masks["v2a"].unsqueeze(0).expand(batch_size, -1, -1)
        representative_sigma = block_sigmas[:, 1:].mean(dim=1)
        video_modality = Modality(
            enabled=True,
            latent=video_input,
            sigma=representative_sigma,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_context,
            context_mask=video_context_mask,
            attention_mask=video_dense_self,
            self_attention_block_mask=video_block_self,
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
            attention_mask=audio_dense_self,
            self_attention_block_mask=audio_block_self,
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
        action_joystick_condition_mask = self._sample_condition_present_mask(
            action_joystick_tokens,
            uncond_prob=self.config.action_joystick_uncond_prob,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        action_button_condition_mask = self._sample_condition_present_mask(
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
        if action_joystick_tokens is not None:
            action_joystick_tokens = torch.cat([action_joystick_tokens, action_joystick_tokens], dim=1)

        generated_block_mask = torch.ones_like(block_sigmas, dtype=torch.bool)
        generated_block_mask[:, 0] = False
        generated_video_mask = expand_block_values_to_tokens(
            generated_block_mask,
            blocks,
            kind="video",
            tokens_per_latent=spatial_tokens,
        )
        generated_audio_mask = expand_block_values_to_tokens(
            generated_block_mask,
            blocks,
            kind="audio",
            tokens_per_latent=1,
        )
        generated_audio_mask &= audio_loss_eligible[:, None]
        video_loss_mask = torch.cat([torch.zeros_like(generated_video_mask), generated_video_mask], dim=1)
        audio_loss_mask = torch.cat([torch.zeros_like(generated_audio_mask), generated_audio_mask], dim=1)
        video_loss_weights = torch.ones_like(video_timesteps)
        audio_loss_weights = torch.ones_like(audio_timesteps)
        if self.config.use_gaussian_loss_weight:
            video_loss_weights = torch.exp(-2.0 * (video_timesteps - 0.5).square())
            audio_loss_weights = torch.exp(-2.0 * (audio_timesteps - 0.5).square())

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
            action_button_query_latent_indices=build_action_query_latent_indices(
                total_video_latents=video_frames,
                spatial_tokens=spatial_tokens,
                device=device,
            ),
            first_frame_latent=video_latents[:, :, :1].clone(),
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
            video_loss_weights=video_loss_weights,
            audio_loss_weights=audio_loss_weights,
            video_clean_latent=video_targets,
            video_noisy_latent=video_input,
            audio_clean_latent=audio_targets,
            audio_noisy_latent=audio_input,
            ar_video_block_token_ranges=tuple(
                (block.video_start * spatial_tokens, block.video_end * spatial_tokens) for block in blocks
            ),
            ar_audio_block_token_ranges=tuple((block.audio_start, block.audio_end) for block in blocks),
        )
        self._last_inputs = inputs
        return inputs

    @staticmethod
    def _weighted_token_mse(
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
        weights: Tensor,
    ) -> Tensor:
        per_token = (prediction.float() - target.float()).square().mean(dim=-1)
        effective_weight = mask.float() * weights.float()
        return (per_token * effective_weight).sum(dim=1) / effective_weight.sum(dim=1).clamp(min=1e-8)

    def _per_block_metrics(
        self,
        *,
        predicted_video_x0: Tensor,
        predicted_audio_x0: Tensor | None,
        inputs: ModelInputs,
    ) -> dict[str, Tensor]:
        video_ranges = inputs.ar_video_block_token_ranges
        audio_ranges = inputs.ar_audio_block_token_ranges
        if video_ranges is None or audio_ranges is None or len(video_ranges) != len(audio_ranges):
            raise ValueError("AR teacher block token ranges are missing or inconsistent.")
        if inputs.video_targets is None or inputs.video_loss_mask is None or inputs.video_loss_weights is None:
            raise ValueError("AR teacher video targets/masks/weights are missing.")

        metrics: dict[str, Tensor] = {}
        video_half = inputs.video_targets.shape[1] // 2
        audio_half = inputs.audio_targets.shape[1] // 2 if inputs.audio_targets is not None else 0
        for block_index, ((video_start, video_end), (audio_start, audio_end)) in enumerate(
            zip(video_ranges[1:], audio_ranges[1:], strict=True),
            start=1,
        ):
            video_slice = slice(video_half + video_start, video_half + video_end)
            metrics[f"ar/block{block_index}/sigma"] = inputs.video.timesteps[:, video_slice].float().mean()
            metrics[f"ar/block{block_index}/loss_video"] = self._weighted_token_mse(
                predicted_video_x0[:, video_slice],
                inputs.video_targets[:, video_slice],
                inputs.video_loss_mask[:, video_slice],
                inputs.video_loss_weights[:, video_slice],
            ).mean()

            audio_loss_scalar = predicted_video_x0.new_zeros((), dtype=torch.float32)
            if predicted_audio_x0 is not None:
                if (
                    inputs.audio_targets is None
                    or inputs.audio_loss_mask is None
                    or inputs.audio_loss_weights is None
                ):
                    raise ValueError("AR teacher audio block metrics require audio targets/masks/weights.")
                audio_slice = slice(audio_half + audio_start, audio_half + audio_end)
                block_audio_mask = inputs.audio_loss_mask[:, audio_slice]
                block_audio_loss = self._weighted_token_mse(
                    predicted_audio_x0[:, audio_slice],
                    inputs.audio_targets[:, audio_slice],
                    block_audio_mask,
                    inputs.audio_loss_weights[:, audio_slice],
                )
                audio_eligible = block_audio_mask.any(dim=1).float()
                audio_loss_scalar = (block_audio_loss * audio_eligible).sum() / audio_eligible.sum().clamp(min=1.0)
            metrics[f"ar/block{block_index}/loss_audio"] = audio_loss_scalar
        return metrics

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> LossOutputs:
        if inputs.video_targets is None or inputs.video_loss_mask is None or inputs.video_loss_weights is None:
            raise ValueError("AR teacher video targets/masks/weights are missing.")
        predicted_video_x0 = to_denoised(inputs.video.latent, video_pred, inputs.video.timesteps)
        video_loss = self._weighted_token_mse(
            predicted_video_x0,
            inputs.video_targets,
            inputs.video_loss_mask,
            inputs.video_loss_weights,
        )

        audio_loss_scalar = torch.zeros((), device=video_pred.device, dtype=torch.float32)
        predicted_audio_x0 = None
        total = video_loss
        if audio_pred is not None:
            if (
                inputs.audio is None
                or inputs.audio_targets is None
                or inputs.audio_loss_mask is None
                or inputs.audio_loss_weights is None
            ):
                raise ValueError("AR teacher audio prediction was returned without audio targets/masks/weights.")
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

        metrics = {
            "loss": total.mean(),
            "loss_video_x0": video_loss.mean(),
            "loss_audio_x0": audio_loss_scalar,
        }
        metrics.update(
            self._per_block_metrics(
                predicted_video_x0=predicted_video_x0,
                predicted_audio_x0=predicted_audio_x0,
                inputs=inputs,
            )
        )
        return LossOutputs(
            total=total,
            metrics=metrics,
        )

    def collect_diagnostics(self, *args: Any, **kwargs: Any) -> None:
                                                                           
                                                                             
        return None


__all__ = ["ARTeacherConfig", "ARTeacherStrategy"]
