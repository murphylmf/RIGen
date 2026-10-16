''

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import torch
from torch import Tensor, nn

from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer.modality import Modality
from ltx_core.utils import to_denoised

from ltx_trainer.training_strategies.base_strategy import ModelInputs


@dataclass(frozen=True)
class GeneratedX0:
    ''

    video: Tensor
    audio: Tensor
    video_mask: Tensor
    audio_mask: Tensor


@dataclass(frozen=True)
class DMDLossOutput:
    total: Tensor
    video: Tensor
    audio: Tensor
    metrics: dict[str, Tensor]


class SharedDiscreteSigmaSampler:
    ''

    def __init__(self, sigmas: tuple[float, ...] | list[float]):
        values = tuple(float(value) for value in sigmas)
        if len(values) < 2:
            raise ValueError(
                "Stage-2 denoising sigmas require at least one step and a zero endpoint."
            )
        if values[-1] != 0.0:
            raise ValueError(
                f"Stage-2 denoising sigmas must end at 0, got {values[-1]}."
            )
        if any(
            left <= right for left, right in zip(values[:-1], values[1:], strict=True)
        ):
            raise ValueError(
                f"Stage-2 denoising sigmas must be strictly descending, got {values}."
            )
        if values[0] > 1.0 or values[-2] <= 0.0:
            raise ValueError(
                f"Stage-2 denoising sigmas must lie in (0, 1] before the endpoint, got {values}."
            )
        self.sigmas = values
        self._current: Tensor | None = None

    def reset(self) -> None:
        self._current = None

    def sample(
        self,
        batch_size: int,
        seq_length: int | None = None,                
        device: torch.device | None = None,
    ) -> Tensor:
        if self._current is None:
            choices = torch.tensor(self.sigmas[:-1], device=device, dtype=torch.float32)
            indices = torch.randint(0, len(choices), (batch_size,), device=device)
            self._current = choices.index_select(0, indices)
        if self._current.shape != (batch_size,):
            raise RuntimeError(
                "SharedDiscreteSigmaSampler was reused without reset for a different batch size: "
                f"cached={tuple(self._current.shape)}, requested={(batch_size,)}."
            )
        return self._current

    def sample_for(self, batch: Tensor) -> Tensor:
        return self.sample(batch.shape[0], batch.shape[1], batch.device)


class DMDScoreSigmaSampler:
    ''

    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        minimum_step_ratio: float = 0.02,
        maximum_step_ratio: float = 0.98,
    ) -> None:
        if num_train_timesteps < 2:
            raise ValueError(
                "DMD score sampling requires at least two train timesteps."
            )
        if not 0.0 < minimum_step_ratio < maximum_step_ratio < 1.0:
            raise ValueError(
                "DMD score step ratios must satisfy 0 < min < max < 1, got "
                f"{minimum_step_ratio}, {maximum_step_ratio}."
            )
        self.num_train_timesteps = int(num_train_timesteps)
        self.minimum_step = int(minimum_step_ratio * self.num_train_timesteps)
        self.maximum_step = int(maximum_step_ratio * self.num_train_timesteps)
                                                                        
                                                                           
        self._lookup_cpu = (
            LTX2Scheduler().execute(steps=self.num_train_timesteps).flip(0)
        )
        if self._lookup_cpu.shape != (self.num_train_timesteps + 1,):
            raise RuntimeError("Unexpected LTX-2 DMD sigma lookup shape.")

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device,
        synchronize: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        timesteps = torch.randint(
            self.minimum_step,
            self.maximum_step + 1,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        if synchronize is not None:
            timesteps = synchronize(timesteps)
        lookup = self._lookup_cpu.to(device=device)
        return lookup.index_select(0, timesteps)


def _validate_doubled_inputs(inputs: ModelInputs) -> tuple[int, int]:
    if inputs.audio is None:
        raise ValueError("Stage-2 TI2AV DMD requires the audio modality.")
    video_tokens = int(inputs.video.latent.shape[1])
    audio_tokens = int(inputs.audio.latent.shape[1])
    if video_tokens % 2 or audio_tokens % 2:
        raise ValueError(
            "Stage-2 generator inputs must retain the AR [clean|noisy] doubled sequence: "
            f"video={video_tokens}, audio={audio_tokens}."
        )
    return video_tokens // 2, audio_tokens // 2


def extract_generated_x0(
    inputs: ModelInputs,
    video_velocity: Tensor,
    audio_velocity: Tensor | None,
) -> GeneratedX0:
    ''

    video_half, audio_half = _validate_doubled_inputs(inputs)
    if audio_velocity is None or inputs.audio is None:
        raise ValueError("Stage-2 generator returned no audio prediction.")
    if inputs.video_targets is None or inputs.audio_targets is None:
        raise ValueError("Stage-2 generator inputs are missing clean targets.")
    if inputs.video_loss_mask is None or inputs.audio_loss_mask is None:
        raise ValueError("Stage-2 generator inputs are missing generated-block masks.")

    full_video_x0 = to_denoised(
        inputs.video.latent, video_velocity, inputs.video.timesteps
    )
    full_audio_x0 = to_denoised(
        inputs.audio.latent, audio_velocity, inputs.audio.timesteps
    )
    predicted_video = full_video_x0[:, video_half:]
    predicted_audio = full_audio_x0[:, audio_half:]
    clean_video = inputs.video_targets[:, :video_half]
    clean_audio = inputs.audio_targets[:, :audio_half]
    video_mask = inputs.video_loss_mask[:, video_half:].to(dtype=torch.bool)
    audio_mask = inputs.audio_loss_mask[:, audio_half:].to(dtype=torch.bool)

    video = torch.where(video_mask.unsqueeze(-1), predicted_video, clean_video)
    audio = torch.where(audio_mask.unsqueeze(-1), predicted_audio, clean_audio)
    return GeneratedX0(
        video=video, audio=audio, video_mask=video_mask, audio_mask=audio_mask
    )


def _score_prompt_view(
    modality: Modality, *, token_half: int, prefix_tokens: int
) -> tuple[Tensor, Tensor]:
    context_tokens = int(modality.context.shape[1])
    if context_tokens % 2:
        raise ValueError(
            f"AR prompt context must be doubled, got {context_tokens} tokens."
        )
    context_half = context_tokens // 2
    context = modality.context[:, context_half:]
    if modality.context_mask is None:
        raise ValueError(
            "Stage-2 per-block prompt routing requires the AR context mask."
        )
    expected = (modality.latent.shape[0], 2 * token_half, 2 * context_half)
    if modality.context_mask.shape != expected:
        raise ValueError(
            "Unexpected doubled prompt-routing mask shape: "
            f"got={tuple(modality.context_mask.shape)}, expected={expected}."
        )
                                                                            
                                                                            
                                                                         
    mask = modality.context_mask[:, token_half:, context_half:].clone()
    if prefix_tokens:
        mask[:, :prefix_tokens] = modality.context_mask[
            :, :prefix_tokens, :context_half
        ]
    if not torch.all(mask.any(dim=-1)):
        bad = int((~mask.any(dim=-1)).sum().item())
        raise ValueError(
            f"Score prompt routing contains {bad} query rows with no valid context token."
        )
    return context, mask


def _undouble_action_tensor(
    value: Tensor | None, *, token_half: int, name: str
) -> Tensor | None:
    if value is None:
        return None
    if value.ndim >= 2 and value.shape[1] == 2 * token_half:
        return value[:, token_half:]
    if value.ndim == 1 and value.numel() == 2 * token_half:
        return value[token_half:]
    raise ValueError(
        f"{name} must follow the doubled video token geometry before Stage-2 score conversion; "
        f"got shape {tuple(value.shape)}, video_half={token_half}."
    )


def build_score_inputs(
    generator_inputs: ModelInputs,
    clean_x0: GeneratedX0,
    sigma: Tensor,
    *,
    video_noise: Tensor | None = None,
    audio_noise: Tensor | None = None,
) -> tuple[ModelInputs, Tensor, Tensor]:
    ''

    video_half, audio_half = _validate_doubled_inputs(generator_inputs)
    if generator_inputs.audio is None:
        raise AssertionError("validated audio modality disappeared")
    batch_size = clean_x0.video.shape[0]
    if sigma.shape != (batch_size,):
        raise ValueError(
            f"Score sigma must have shape {(batch_size,)}, got {tuple(sigma.shape)}."
        )
    if clean_x0.video.shape[:2] != (batch_size, video_half):
        raise ValueError(
            "Generated video x0 does not match the undoubled token geometry."
        )
    if clean_x0.audio.shape[:2] != (batch_size, audio_half):
        raise ValueError(
            "Generated audio x0 does not match the undoubled token geometry."
        )

    video_noise = (
        torch.randn_like(clean_x0.video) if video_noise is None else video_noise
    )
    audio_noise = (
        torch.randn_like(clean_x0.audio) if audio_noise is None else audio_noise
    )
    if (
        video_noise.shape != clean_x0.video.shape
        or audio_noise.shape != clean_x0.audio.shape
    ):
        raise ValueError(
            "Explicit Stage-2 noise tensors must match the clean x0 shapes."
        )

    video_timesteps = sigma[:, None].expand(-1, video_half) * clean_x0.video_mask
    audio_timesteps = sigma[:, None].expand(-1, audio_half)
    video_sigma_bc = video_timesteps.unsqueeze(-1).to(dtype=clean_x0.video.dtype)
    audio_sigma_bc = audio_timesteps.unsqueeze(-1).to(dtype=clean_x0.audio.dtype)
    noisy_video = (1 - video_sigma_bc) * clean_x0.video + video_sigma_bc * video_noise
    noisy_audio = (1 - audio_sigma_bc) * clean_x0.audio + audio_sigma_bc * audio_noise

    video_ranges = generator_inputs.ar_video_block_token_ranges
    audio_ranges = generator_inputs.ar_audio_block_token_ranges
    if not video_ranges or not audio_ranges:
        raise ValueError("Stage-2 score conversion requires AR block token ranges.")
    video_prefix_tokens = int(video_ranges[0][1] - video_ranges[0][0])
    audio_prefix_tokens = int(audio_ranges[0][1] - audio_ranges[0][0])
    video_context, video_context_mask = _score_prompt_view(
        generator_inputs.video,
        token_half=video_half,
        prefix_tokens=video_prefix_tokens,
    )
    audio_context, audio_context_mask = _score_prompt_view(
        generator_inputs.audio,
        token_half=audio_half,
        prefix_tokens=audio_prefix_tokens,
    )
    video_modality = Modality(
        enabled=True,
        latent=noisy_video,
        sigma=sigma,
        timesteps=video_timesteps,
        positions=generator_inputs.video.positions[:, :, video_half:],
        context=video_context,
        context_mask=video_context_mask,
        attention_mask=None,
        self_attention_block_mask=None,
        cross_attention_mask=None,
        cross_attention_timesteps=None,
        context_timesteps=None,
    )
    audio_modality = Modality(
        enabled=True,
        latent=noisy_audio,
        sigma=sigma,
        timesteps=audio_timesteps,
        positions=generator_inputs.audio.positions[:, :, audio_half:],
        context=audio_context,
        context_mask=audio_context_mask,
        attention_mask=None,
        self_attention_block_mask=None,
        cross_attention_mask=None,
        cross_attention_timesteps=None,
        context_timesteps=None,
    )

    score_inputs = ModelInputs(
        video=video_modality,
        audio=audio_modality,
        action_joystick_tokens=_undouble_action_tensor(
            generator_inputs.action_joystick_tokens,
            token_half=video_half,
            name="action_joystick_tokens",
        ),
        action_joystick_condition_mask=generator_inputs.action_joystick_condition_mask,
        action_button_state=generator_inputs.action_button_state,
        action_button_values=generator_inputs.action_button_values,
        action_button_condition_mask=generator_inputs.action_button_condition_mask,
        action_button_meaning_embeds=generator_inputs.action_button_meaning_embeds,
        action_button_temporal_alignment=generator_inputs.action_button_temporal_alignment,
        action_button_prefix_tokens=generator_inputs.action_button_prefix_tokens,
        action_button_frame_rate=generator_inputs.action_button_frame_rate,
        action_button_query_latent_indices=_undouble_action_tensor(
            generator_inputs.action_button_query_latent_indices,
            token_half=video_half,
            name="action_button_query_latent_indices",
        ),
        first_frame_latent=generator_inputs.first_frame_latent,
        video_targets=clean_x0.video,
        audio_targets=clean_x0.audio,
        video_loss_mask=clean_x0.video_mask,
        audio_loss_mask=clean_x0.audio_mask,
        ar_video_block_token_ranges=video_ranges,
        ar_audio_block_token_ranges=audio_ranges,
    )
    return score_inputs, video_noise, audio_noise


def _strip_sliding_ref_tokens(
    value: Tensor | None,
    *,
    total_tokens: int,
    world_tokens: int,
    name: str,
) -> Tensor | None:
    if value is None:
        return None
    if value.ndim >= 2 and value.shape[1] == total_tokens:
        return value[:, :world_tokens]
    if value.ndim == 1 and value.numel() == total_tokens:
        return value[:world_tokens]
    raise ValueError(
        f"{name} must follow the sliding-ref video token geometry; got "
        f"{tuple(value.shape)}, expected token length {total_tokens}."
    )


def build_sliding_ref_score_inputs(
    generator_inputs: ModelInputs,
    clean_x0: GeneratedX0,
    sigma: Tensor,
    *,
    video_noise: Tensor | None = None,
    audio_noise: Tensor | None = None,
) -> tuple[ModelInputs, Tensor, Tensor]:
    ''

    if generator_inputs.audio is None:
        raise ValueError("Sliding-ref Stage-2 score conversion requires audio.")
    reference_tokens = int(generator_inputs.ref_seq_len or 0)
    total_video_tokens = int(generator_inputs.video.latent.shape[1])
    if reference_tokens <= 0 or reference_tokens >= total_video_tokens:
        raise ValueError(
            "Sliding-ref score conversion requires a non-empty suffix reference."
        )
    world_tokens = total_video_tokens - reference_tokens
    audio_tokens = int(generator_inputs.audio.latent.shape[1])
    batch_size = int(generator_inputs.video.latent.shape[0])
    if sigma.shape != (batch_size,):
        raise ValueError(
            f"Score sigma must have shape {(batch_size,)}, got {tuple(sigma.shape)}."
        )
    if clean_x0.video.shape[:2] != (batch_size, world_tokens):
        raise ValueError("Generated video does not match the ref-free world geometry.")
    if clean_x0.audio.shape[:2] != (batch_size, audio_tokens):
        raise ValueError("Generated audio does not match the local score geometry.")
    if clean_x0.video_mask.shape != (batch_size, world_tokens):
        raise ValueError("Generated video mask does not match the world geometry.")
    if clean_x0.audio_mask.shape != (batch_size, audio_tokens):
        raise ValueError("Generated audio mask does not match the score geometry.")

    video_noise = (
        torch.randn_like(clean_x0.video) if video_noise is None else video_noise
    )
    audio_noise = (
        torch.randn_like(clean_x0.audio) if audio_noise is None else audio_noise
    )
    if video_noise.shape != clean_x0.video.shape:
        raise ValueError("Explicit sliding-ref video noise has the wrong shape.")
    if audio_noise.shape != clean_x0.audio.shape:
        raise ValueError("Explicit sliding-ref audio noise has the wrong shape.")

    video_timesteps = sigma[:, None].expand(-1, world_tokens) * clean_x0.video_mask
                                                                           
                                                      
    audio_timesteps = sigma[:, None].expand(-1, audio_tokens)
    video_sigma = video_timesteps.unsqueeze(-1).to(dtype=clean_x0.video.dtype)
    audio_sigma = audio_timesteps.unsqueeze(-1).to(dtype=clean_x0.audio.dtype)
    noisy_video = (1.0 - video_sigma) * clean_x0.video + video_sigma * video_noise
    noisy_audio = (1.0 - audio_sigma) * clean_x0.audio + audio_sigma * audio_noise

    video = Modality(
        enabled=True,
        latent=noisy_video,
        sigma=sigma,
        timesteps=video_timesteps,
        positions=generator_inputs.video.positions[:, :, :world_tokens],
        context=generator_inputs.video.context,
        context_mask=(
            generator_inputs.video.context_mask[:, :world_tokens]
            if generator_inputs.video.context_mask is not None
            else None
        ),
        attention_mask=None,
        self_attention_block_mask=None,
        cross_attention_mask=None,
        cross_attention_timesteps=None,
        context_timesteps=generator_inputs.video.context_timesteps,
    )
    audio = Modality(
        enabled=True,
        latent=noisy_audio,
        sigma=sigma,
        timesteps=audio_timesteps,
        positions=generator_inputs.audio.positions,
        context=generator_inputs.audio.context,
        context_mask=generator_inputs.audio.context_mask,
        attention_mask=None,
        self_attention_block_mask=None,
        cross_attention_mask=None,
        cross_attention_timesteps=None,
        context_timesteps=generator_inputs.audio.context_timesteps,
    )
    score_inputs = ModelInputs(
        video=video,
        audio=audio,
        action_joystick_tokens=_strip_sliding_ref_tokens(
            generator_inputs.action_joystick_tokens,
            total_tokens=total_video_tokens,
            world_tokens=world_tokens,
            name="action_joystick_tokens",
        ),
        action_joystick_condition_mask=_strip_sliding_ref_tokens(
            generator_inputs.action_joystick_condition_mask,
            total_tokens=total_video_tokens,
            world_tokens=world_tokens,
            name="action_joystick_condition_mask",
        ),
        action_button_state=generator_inputs.action_button_state,
        action_button_values=generator_inputs.action_button_values,
        action_button_condition_mask=_strip_sliding_ref_tokens(
            generator_inputs.action_button_condition_mask,
            total_tokens=total_video_tokens,
            world_tokens=world_tokens,
            name="action_button_condition_mask",
        ),
        action_button_meaning_embeds=generator_inputs.action_button_meaning_embeds,
        action_button_temporal_alignment=generator_inputs.action_button_temporal_alignment,
        action_button_prefix_tokens=generator_inputs.action_button_prefix_tokens,
        action_button_frame_rate=generator_inputs.action_button_frame_rate,
        action_button_query_latent_indices=_strip_sliding_ref_tokens(
            generator_inputs.action_button_query_latent_indices,
            total_tokens=total_video_tokens,
            world_tokens=world_tokens,
            name="action_button_query_latent_indices",
        ),
        first_frame_latent=generator_inputs.first_frame_latent,
        video_targets=clean_x0.video,
        audio_targets=clean_x0.audio,
        video_loss_mask=clean_x0.video_mask,
        audio_loss_mask=clean_x0.audio_mask,
        ar_video_block_token_ranges=generator_inputs.ar_video_block_token_ranges,
        ar_audio_block_token_ranges=generator_inputs.ar_audio_block_token_ranges,
    )
    return score_inputs, video_noise, audio_noise


def build_prompt_routing_mask(
    block_ranges: tuple[tuple[int, int], ...],
    prompt_attention_mask: Tensor,
) -> Tensor:
    ''

    if prompt_attention_mask.ndim != 3:
        raise ValueError(
            "Prompt schedule attention mask must be [B, generated_blocks, L]."
        )
    batch, prompt_count, prompt_length = prompt_attention_mask.shape
    if len(block_ranges) != prompt_count + 1:
        raise ValueError(
            f"Prompt count {prompt_count} is incompatible with {len(block_ranges)} AR blocks."
        )
    total_tokens = int(block_ranges[-1][1])
    result = torch.zeros(
        batch,
        total_tokens,
        prompt_count * prompt_length,
        device=prompt_attention_mask.device,
        dtype=torch.bool,
    )
    valid = prompt_attention_mask.to(dtype=torch.bool)
    for block_index, (start, end) in enumerate(block_ranges):
        prompt_index = max(block_index - 1, 0)
        context_start = prompt_index * prompt_length
        result[:, start:end, context_start : context_start + prompt_length] = valid[
            :, prompt_index, None, :
        ]
    if not torch.all(result.any(dim=-1)):
        raise ValueError("Prompt routing produced an all-masked score query.")
    return result


def replace_score_prompt_schedule(
    inputs: ModelInputs,
    *,
    video_schedule: Tensor,
    audio_schedule: Tensor,
    attention_mask: Tensor,
) -> ModelInputs:
    ''

    if inputs.audio is None:
        raise ValueError("Stage-2 prompt replacement requires audio.")
    if video_schedule.ndim != 4 or audio_schedule.ndim != 4:
        raise ValueError("Prompt schedules must be [B, generated_blocks, L, D].")
    if (
        video_schedule.shape[:3] != attention_mask.shape
        or audio_schedule.shape[:3] != attention_mask.shape
    ):
        raise ValueError("Video/audio schedule prefixes must match the attention mask.")
    video_context = video_schedule.flatten(1, 2)
    audio_context = audio_schedule.flatten(1, 2)
    sigma = inputs.video.sigma
    video = replace(
        inputs.video,
        context=video_context,
        context_mask=build_prompt_routing_mask(
            inputs.ar_video_block_token_ranges or (), attention_mask
        ),
        context_timesteps=sigma[:, None].expand(-1, video_context.shape[1]),
    )
    audio = replace(
        inputs.audio,
        context=audio_context,
        context_mask=build_prompt_routing_mask(
            inputs.ar_audio_block_token_ranges or (), attention_mask
        ),
        context_timesteps=sigma[:, None].expand(-1, audio_context.shape[1]),
    )
    return replace(inputs, video=video, audio=audio)


def replace_score_full_prompt(
    inputs: ModelInputs,
    *,
    video_context: Tensor,
    audio_context: Tensor,
    attention_mask: Tensor,
) -> ModelInputs:
    ''

    if inputs.audio is None:
        raise ValueError("Stage-2 full-prompt replacement requires audio.")
    if video_context.ndim != 3 or audio_context.ndim != 3:
        raise ValueError("Full score prompts must have shape [B, L, D].")
    if attention_mask.ndim != 2:
        raise ValueError("Full score prompt attention_mask must have shape [B, L].")
    if (
        video_context.shape[:2] != attention_mask.shape
        or audio_context.shape[:2] != attention_mask.shape
    ):
        raise ValueError(
            "Video/audio full-prompt prefixes must match the attention mask."
        )
    if video_context.shape[0] != inputs.video.latent.shape[0]:
        raise ValueError("Full score prompt batch size does not match score latents.")
    if not torch.all(attention_mask.to(dtype=torch.bool).any(dim=-1)):
        raise ValueError(
            "Every full score prompt must contain at least one valid token."
        )

    video = replace(
        inputs.video,
        context=video_context,
        context_mask=attention_mask.to(dtype=torch.bool),
        context_timesteps=None,
        cross_attention_timesteps=None,
    )
    audio = replace(
        inputs.audio,
        context=audio_context,
        context_mask=attention_mask.to(dtype=torch.bool),
        context_timesteps=None,
        cross_attention_timesteps=None,
    )
    return replace(inputs, video=video, audio=audio)


def forward_action_transformer(
    model: nn.Module,
    inputs: ModelInputs,
    *,
    action_button_meaning_fusion: str,
) -> tuple[Tensor, Tensor]:
    ''

    video_pred, audio_pred = model(
        video=inputs.video,
        audio=inputs.audio,
        perturbations=None,
        action_joystick_tokens=inputs.action_joystick_tokens,
        action_joystick_condition_mask=inputs.action_joystick_condition_mask,
        action_button_state=inputs.action_button_state,
        action_button_values=inputs.action_button_values,
        action_button_condition_mask=inputs.action_button_condition_mask,
        action_button_meaning_embeds=inputs.action_button_meaning_embeds,
        action_button_meaning_fusion=action_button_meaning_fusion,
        action_button_temporal_alignment=inputs.action_button_temporal_alignment,
        action_button_prefix_tokens=inputs.action_button_prefix_tokens,
        action_button_frame_rate=inputs.action_button_frame_rate,
        action_button_query_latent_indices=inputs.action_button_query_latent_indices,
    )
    if video_pred is None or audio_pred is None or inputs.audio is None:
        raise RuntimeError(
            "Stage-2 score model must return both video and audio predictions."
        )
    return video_pred, audio_pred


def predict_score_x0(
    model: nn.Module,
    inputs: ModelInputs,
    *,
    action_button_meaning_fusion: str,
) -> tuple[Tensor, Tensor]:
    video_velocity, audio_velocity = forward_action_transformer(
        model,
        inputs,
        action_button_meaning_fusion=action_button_meaning_fusion,
    )
    if inputs.audio is None:
        raise AssertionError("validated audio modality disappeared")
    return (
        to_denoised(inputs.video.latent, video_velocity, inputs.video.timesteps),
        to_denoised(inputs.audio.latent, audio_velocity, inputs.audio.timesteps),
    )


def _masked_per_sample_mean(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape[:2] != mask.shape:
        raise ValueError(
            f"Masked reduction shape mismatch: values={tuple(values.shape)}, mask={tuple(mask.shape)}."
        )
    reduce_dims = tuple(range(2, values.ndim))
    per_token = values.mean(dim=reduce_dims) if reduce_dims else values
    active = mask.to(dtype=per_token.dtype)
    return (per_token * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def compute_dmd_gradient(
    *,
    real_score: nn.Module,
    fake_score: nn.Module,
    conditional_inputs: ModelInputs,
    unconditional_inputs: ModelInputs,
    clean_x0: GeneratedX0,
    action_button_meaning_fusion: str,
    video_guidance_scale: float,
    audio_guidance_scale: float,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    fake_video, fake_audio = predict_score_x0(
        fake_score,
        conditional_inputs,
        action_button_meaning_fusion=action_button_meaning_fusion,
    )
    real_cond_video, real_cond_audio = predict_score_x0(
        real_score,
        conditional_inputs,
        action_button_meaning_fusion=action_button_meaning_fusion,
    )
    real_uncond_video, real_uncond_audio = predict_score_x0(
        real_score,
        unconditional_inputs,
        action_button_meaning_fusion=action_button_meaning_fusion,
    )
    real_video = real_cond_video + (float(video_guidance_scale) - 1.0) * (
        real_cond_video - real_uncond_video
    )
    real_audio = real_cond_audio + (float(audio_guidance_scale) - 1.0) * (
        real_cond_audio - real_uncond_audio
    )
    grad_video = fake_video.float() - real_video.float()
    grad_audio = fake_audio.float() - real_audio.float()

    video_error = (clean_x0.video.float() - real_video.float()).abs()
    audio_error = (clean_x0.audio.float() - real_audio.float()).abs()
    video_norm = _masked_per_sample_mean(video_error, clean_x0.video_mask).view(
        -1, 1, 1
    )
    audio_norm = _masked_per_sample_mean(audio_error, clean_x0.audio_mask).view(
        -1, 1, 1
    )
    grad_video = torch.nan_to_num(grad_video / (video_norm + 1e-8))
    grad_audio = torch.nan_to_num(grad_audio / (audio_norm + 1e-8))
    metrics = {
        "dmd_gradient_video": _masked_per_sample_mean(
            grad_video.abs(), clean_x0.video_mask
        ).mean(),
        "dmd_gradient_audio": _masked_per_sample_mean(
            grad_audio.abs(), clean_x0.audio_mask
        ).mean(),
        "real_score_video": _masked_per_sample_mean(
            real_video.abs(), clean_x0.video_mask
        ).mean(),
        "fake_score_video": _masked_per_sample_mean(
            fake_video.abs(), clean_x0.video_mask
        ).mean(),
    }
    return grad_video, grad_audio, metrics


def distribution_matching_loss(
    clean_x0: GeneratedX0,
    grad_video: Tensor,
    grad_audio: Tensor,
    *,
    audio_weight: float,
) -> DMDLossOutput:
    video_delta = (
        clean_x0.video.float() - (clean_x0.video.float() - grad_video.float()).detach()
    )
    audio_delta = (
        clean_x0.audio.float() - (clean_x0.audio.float() - grad_audio.float()).detach()
    )
    video = 0.5 * _masked_per_sample_mean(video_delta.square(), clean_x0.video_mask)
    audio = 0.5 * _masked_per_sample_mean(audio_delta.square(), clean_x0.audio_mask)
    total = video + float(audio_weight) * audio
    return DMDLossOutput(
        total=total,
        video=video,
        audio=audio,
        metrics={"dmd_video_loss": video.mean(), "dmd_audio_loss": audio.mean()},
    )


def critic_flow_loss(
    *,
    fake_score: nn.Module,
    score_inputs: ModelInputs,
    clean_x0: GeneratedX0,
    video_noise: Tensor,
    audio_noise: Tensor,
    action_button_meaning_fusion: str,
    audio_weight: float,
) -> DMDLossOutput:
    pred_video_x0, pred_audio_x0 = predict_score_x0(
        fake_score,
        score_inputs,
        action_button_meaning_fusion=action_button_meaning_fusion,
    )
    if score_inputs.audio is None:
        raise AssertionError("validated audio modality disappeared")
                                                                           
                                                                           
                                                                          
    video_sigma = (
        torch.where(
            clean_x0.video_mask,
            score_inputs.video.timesteps,
            torch.ones_like(score_inputs.video.timesteps),
        )
        .unsqueeze(-1)
        .float()
    )
    audio_sigma = (
        torch.where(
            clean_x0.audio_mask,
            score_inputs.audio.timesteps,
            torch.ones_like(score_inputs.audio.timesteps),
        )
        .unsqueeze(-1)
        .float()
    )
    pred_video_flow = (
        score_inputs.video.latent.float() - pred_video_x0.float()
    ) / video_sigma
    pred_audio_flow = (
        score_inputs.audio.latent.float() - pred_audio_x0.float()
    ) / audio_sigma
    target_video_flow = video_noise.float() - clean_x0.video.float()
    target_audio_flow = audio_noise.float() - clean_x0.audio.float()
    video = _masked_per_sample_mean(
        (pred_video_flow - target_video_flow).square(),
        clean_x0.video_mask,
    )
    audio = _masked_per_sample_mean(
        (pred_audio_flow - target_audio_flow).square(),
        clean_x0.audio_mask,
    )
    total = video + float(audio_weight) * audio
    return DMDLossOutput(
        total=total,
        video=video,
        audio=audio,
        metrics={"critic_video_loss": video.mean(), "critic_audio_loss": audio.mean()},
    )


__all__ = [
    "DMDScoreSigmaSampler",
    "DMDLossOutput",
    "GeneratedX0",
    "SharedDiscreteSigmaSampler",
    "build_prompt_routing_mask",
    "build_score_inputs",
    "build_sliding_ref_score_inputs",
    "compute_dmd_gradient",
    "critic_flow_loss",
    "distribution_matching_loss",
    "extract_generated_x0",
    "forward_action_transformer",
    "predict_score_x0",
    "replace_score_full_prompt",
    "replace_score_prompt_schedule",
]
