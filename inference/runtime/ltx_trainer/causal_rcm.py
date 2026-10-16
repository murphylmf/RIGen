''

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

from ltx_core.model.transformer.modality import Modality
from ltx_core.utils import to_denoised

from ltx_trainer.dmd import GeneratedX0, predict_score_x0
from ltx_trainer.training_strategies.base_strategy import ModelInputs


                                                                           
                                                       
FOUR_STEP_SIGMAS: tuple[float, ...] = (1.0, 15.0 / 16.0, 5.0 / 6.0, 5.0 / 8.0, 0.0)


@dataclass(frozen=True)
class TFDCMTimePair:
    ''

    start: Tensor
    end: Tensor
    grid_index: Tensor


@dataclass(frozen=True)
class ConsistencyLossOutput:
    ''

    total: Tensor
    video: Tensor
    audio: Tensor


@dataclass(frozen=True)
class ConditionalDMDTarget:
    ''

    video: Tensor
    audio: Tensor
    metrics: dict[str, Tensor]


def extract_masked_generated_x0(
    inputs: ModelInputs,
    video_velocity: Tensor,
    audio_velocity: Tensor | None,
) -> GeneratedX0:
    ''

    if inputs.audio is None or audio_velocity is None:
        raise ValueError("TF-DCM masked extraction requires the audio modality.")
    if inputs.video_targets is None or inputs.audio_targets is None:
        raise ValueError("TF-DCM masked inputs are missing clean targets.")
    if inputs.video_loss_mask is None or inputs.audio_loss_mask is None:
        raise ValueError("TF-DCM masked inputs are missing target loss masks.")
    if video_velocity.shape != inputs.video.latent.shape:
        raise ValueError(
            "Video velocity does not match the masked sequence geometry: "
            f"velocity={tuple(video_velocity.shape)}, latent={tuple(inputs.video.latent.shape)}."
        )
    if audio_velocity.shape != inputs.audio.latent.shape:
        raise ValueError(
            "Audio velocity does not match the masked sequence geometry: "
            f"velocity={tuple(audio_velocity.shape)}, latent={tuple(inputs.audio.latent.shape)}."
        )
    if inputs.video_targets.shape != inputs.video.latent.shape:
        raise ValueError("Video targets do not match the masked sequence geometry.")
    if inputs.audio_targets.shape != inputs.audio.latent.shape:
        raise ValueError("Audio targets do not match the masked sequence geometry.")

    video_mask = inputs.video_loss_mask.to(dtype=torch.bool)
    audio_mask = inputs.audio_loss_mask.to(dtype=torch.bool)
    if video_mask.shape != inputs.video.latent.shape[:2]:
        raise ValueError("Video loss mask does not match the masked sequence geometry.")
    if audio_mask.shape != inputs.audio.latent.shape[:2]:
        raise ValueError("Audio loss mask does not match the masked sequence geometry.")

    predicted_video = to_denoised(
        inputs.video.latent, video_velocity, inputs.video.timesteps
    )
    predicted_audio = to_denoised(
        inputs.audio.latent, audio_velocity, inputs.audio.timesteps
    )
    video = torch.where(video_mask.unsqueeze(-1), predicted_video, inputs.video_targets)
    audio = torch.where(audio_mask.unsqueeze(-1), predicted_audio, inputs.audio_targets)
    return GeneratedX0(
        video=video,
        audio=audio,
        video_mask=video_mask,
        audio_mask=audio_mask,
    )


def shift_rectified_flow_time(unshifted: Tensor, *, log_shift: float) -> Tensor:
    ''

    if log_shift < 0:
        raise ValueError(f"log_shift must be non-negative, got {log_shift}.")
    factor = torch.exp(
        torch.tensor(float(log_shift), device=unshifted.device, dtype=unshifted.dtype)
    )
    return factor * unshifted / (1.0 + (factor - 1.0) * unshifted)


class TFDCMTimePairSampler:
    ''

    def __init__(
        self,
        *,
        total_steps: int = 48,
        skipping_interval_steps: int = 1,
        log_timestep_shift: float = 2.05,
    ) -> None:
        if total_steps < 2:
            raise ValueError("TF-DCM requires total_steps >= 2.")
        if not 1 <= skipping_interval_steps < total_steps:
            raise ValueError(
                "skipping_interval_steps must lie in [1, total_steps), got "
                f"{skipping_interval_steps} for total_steps={total_steps}."
            )
        if log_timestep_shift < 0:
            raise ValueError("log_timestep_shift must be non-negative.")
        self.total_steps = int(total_steps)
        self.skipping_interval_steps = int(skipping_interval_steps)
        self.log_timestep_shift = float(log_timestep_shift)

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device,
        synchronize: Callable[[Tensor], Tensor] | None = None,
    ) -> TFDCMTimePair:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
                                                                    
                                                                              
                                                                           
                                                                 
        max_index = self.total_steps - self.skipping_interval_steps
        grid_index = torch.randint(
            0, max_index, (batch_size,), device=device, dtype=torch.long
        )
        within_cell = torch.rand(batch_size, device=device, dtype=torch.float32)
        raw = grid_index.to(dtype=torch.float32) + within_cell
        if synchronize is not None:
            raw = synchronize(raw)
                                                                  
            grid_index = raw.floor().to(dtype=torch.long)
        start_unshifted = 1.0 - raw / float(self.total_steps)
        end_unshifted = start_unshifted - self.skipping_interval_steps / float(
            self.total_steps
        )
        start = shift_rectified_flow_time(
            start_unshifted, log_shift=self.log_timestep_shift
        )
        end = shift_rectified_flow_time(
            end_unshifted.clamp_min(0.0), log_shift=self.log_timestep_shift
        )
        if torch.any(start <= end):
            raise RuntimeError("TF-DCM sampled a non-descending time pair.")
        return TFDCMTimePair(start=start, end=end, grid_index=grid_index)

    def integration_times(self, pair: TFDCMTimePair) -> tuple[Tensor, ...]:
        ''

        if pair.start.shape != pair.end.shape or pair.start.ndim != 1:
            raise ValueError("TF-DCM time pair tensors must have matching [B] shapes.")
                                                                               
                                                                              
                                                      
        factor = torch.exp(
            torch.tensor(
                self.log_timestep_shift,
                device=pair.start.device,
                dtype=pair.start.dtype,
            )
        )
        unshifted_start = pair.start / (factor - (factor - 1.0) * pair.start)
        result: list[Tensor] = []
        for offset in range(self.skipping_interval_steps + 1):
            unshifted = unshifted_start - offset / float(self.total_steps)
            result.append(
                shift_rectified_flow_time(
                    unshifted.clamp_min(0.0), log_shift=self.log_timestep_shift
                )
            )
        result[0] = pair.start
        result[-1] = pair.end
        return tuple(result)


class FixedBatchSigmaSampler:
    ''

    def __init__(self, sigma: Tensor) -> None:
        if sigma.ndim != 1:
            raise ValueError(
                f"FixedBatchSigmaSampler expects [B], got {tuple(sigma.shape)}."
            )
        self.sigma = sigma

    def sample(
        self,
        batch_size: int,
        seq_length: int | None = None,                
        device: torch.device | None = None,
    ) -> Tensor:
        if batch_size != self.sigma.shape[0]:
            raise ValueError(
                f"Requested batch size {batch_size} does not match fixed sigma batch {self.sigma.shape[0]}."
            )
        if device is not None and device != self.sigma.device:
            return self.sigma.to(device=device)
        return self.sigma

    def sample_for(self, batch: Tensor) -> Tensor:
        return self.sample(batch.shape[0], batch.shape[1], batch.device)


class UniformShiftSigmaSampler:
    ''

    def __init__(self, *, shift: float = 5.0) -> None:
        if shift <= 0:
            raise ValueError(
                f"UniformShift requires a positive multiplicative shift, got {shift}."
            )
        self.shift = float(shift)

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device,
        synchronize: Callable[[Tensor], Tensor] | None = None,
    ) -> Tensor:
        unshifted = torch.rand(batch_size, device=device, dtype=torch.float32)
        if synchronize is not None:
            unshifted = synchronize(unshifted)
        return self.shift * unshifted / (1.0 + (self.shift - 1.0) * unshifted)


def cyclic_simulation_steps(update_index: int, *, maximum_steps: int = 4) -> int:
    ''

    if update_index < 0:
        raise ValueError(f"update_index must be non-negative, got {update_index}.")
    if maximum_steps <= 0:
        raise ValueError(f"maximum_steps must be positive, got {maximum_steps}.")
    return update_index % maximum_steps + 1


@lru_cache(maxsize=8)
def _power_ema_exponent(rate: float) -> float:
    if not 0.0 < rate < 1.0:
        raise ValueError(f"Power EMA rate must lie in (0, 1), got {rate}.")
    roots = np.roots([1.0, 7.0, 16.0 - rate**-2, 12.0 - rate**-2])
    real_roots = roots.real[np.isclose(roots.imag, 0.0)]
    if real_roots.size == 0:
        raise RuntimeError(f"Power EMA rate {rate} produced no real exponent root.")
    return float(real_roots.max())


def power_ema_beta(
    update_index: int, *, rate: float = 0.1, iteration_shift: int = 0
) -> float:
    ''

    iteration = int(update_index) + int(iteration_shift)
    if iteration < 1:
        return 0.0
    exponent = _power_ema_exponent(float(rate))
    return float((1.0 - 1.0 / (iteration + 1.0)) ** (exponent + 1.0))


@torch.no_grad()
def update_ema_parameters(
    *,
    student: nn.Module,
    ema: nn.Module,
    beta: float,
) -> None:
    ''

    if not 0.0 <= beta < 1.0:
        raise ValueError(f"EMA beta must lie in [0, 1), got {beta}.")
    student_parameters = list(student.named_parameters())
    ema_parameters = list(ema.named_parameters())
    if len(student_parameters) != len(ema_parameters):
        raise RuntimeError(
            f"Student/EMA parameter counts differ: {len(student_parameters)} vs {len(ema_parameters)}."
        )
    for (student_name, student_parameter), (ema_name, ema_parameter) in zip(
        student_parameters,
        ema_parameters,
        strict=True,
    ):
        if student_name != ema_name or student_parameter.shape != ema_parameter.shape:
            raise RuntimeError(
                "Student/EMA parameter layouts differ: "
                f"{student_name}{tuple(student_parameter.shape)} vs {ema_name}{tuple(ema_parameter.shape)}."
            )
        source = student_parameter.detach().to(
            device=ema_parameter.device, dtype=ema_parameter.dtype
        )
        ema_parameter.mul_(beta).add_(source, alpha=1.0 - beta)


def _replace_positive(values: Tensor | None, *, replacement: Tensor) -> Tensor | None:
    if values is None:
        return None
    if values.shape[0] != replacement.shape[0]:
        raise ValueError("Replacement sigma batch does not match timestep batch.")
    expanded = replacement.reshape(replacement.shape[0], *([1] * (values.ndim - 1)))
    return torch.where(
        values > 0, expanded.to(dtype=values.dtype), torch.zeros_like(values)
    )


def _advance_modality(
    modality: Modality,
    *,
    velocity: Tensor,
    next_sigma: Tensor,
) -> Modality:
    if velocity.shape != modality.latent.shape:
        raise ValueError(
            f"Velocity shape {tuple(velocity.shape)} does not match latent shape {tuple(modality.latent.shape)}."
        )
    if next_sigma.shape != modality.sigma.shape:
        raise ValueError(
            f"next_sigma shape {tuple(next_sigma.shape)} does not match modality sigma {tuple(modality.sigma.shape)}."
        )
    active = modality.timesteps > 0
    next_timesteps = _replace_positive(modality.timesteps, replacement=next_sigma)
    if next_timesteps is None:
        raise AssertionError("Modality timesteps cannot be None.")
    dt = (
        (modality.timesteps - next_timesteps)
        .unsqueeze(-1)
        .to(dtype=modality.latent.dtype)
    )
    next_latent = modality.latent - dt * velocity.to(dtype=modality.latent.dtype)
                                                                           
                                                                             
                                       
    next_latent = torch.where(active.unsqueeze(-1), next_latent, modality.latent)
    return replace(
        modality,
        latent=next_latent,
        sigma=next_sigma.to(dtype=modality.sigma.dtype),
        timesteps=next_timesteps,
        cross_attention_timesteps=_replace_positive(
            modality.cross_attention_timesteps,
            replacement=next_sigma,
        ),
        context_timesteps=_replace_positive(
            modality.context_timesteps,
            replacement=next_sigma,
        ),
    )


def advance_causal_teacher_inputs(
    inputs: ModelInputs,
    *,
    video_velocity: Tensor,
    audio_velocity: Tensor,
    next_sigma: Tensor,
) -> ModelInputs:
    ''

    if inputs.audio is None:
        raise ValueError("TF-DCM requires the audio modality.")
    video = _advance_modality(
        inputs.video, velocity=video_velocity, next_sigma=next_sigma
    )
    audio = _advance_modality(
        inputs.audio, velocity=audio_velocity, next_sigma=next_sigma
    )
    return replace(
        inputs,
        video=video,
        audio=audio,
        video_noisy_latent=video.latent,
        audio_noisy_latent=audio.latent,
    )


def inject_self_forcing_state(
    inputs: ModelInputs,
    *,
    block_index: int,
    clean_video: Tensor,
    clean_audio: Tensor,
    current_video: Tensor,
    current_audio: Tensor,
) -> ModelInputs:
    ''

    if inputs.audio is None:
        raise ValueError("Self-forcing rollout requires the audio modality.")
    video_ranges = inputs.ar_video_block_token_ranges
    audio_ranges = inputs.ar_audio_block_token_ranges
    if video_ranges is None or audio_ranges is None:
        raise ValueError("Self-forcing rollout requires AR block token ranges.")
    if not 1 <= block_index < len(video_ranges) or len(video_ranges) != len(
        audio_ranges
    ):
        raise ValueError(f"Invalid self-forcing block index {block_index}.")
    video_half = inputs.video.latent.shape[1] // 2
    audio_half = inputs.audio.latent.shape[1] // 2
    if clean_video.shape != inputs.video.latent[:, :video_half].shape:
        raise ValueError(
            "Generated clean video does not match the undoubled AR geometry."
        )
    if clean_audio.shape != inputs.audio.latent[:, :audio_half].shape:
        raise ValueError(
            "Generated clean audio does not match the undoubled AR geometry."
        )
    video_start, video_end = video_ranges[block_index]
    audio_start, audio_end = audio_ranges[block_index]
    if current_video.shape != clean_video[:, video_start:video_end].shape:
        raise ValueError("Current video state does not match its AR block range.")
    if current_audio.shape != clean_audio[:, audio_start:audio_end].shape:
        raise ValueError("Current audio state does not match its AR block range.")

    video_latent = inputs.video.latent.clone()
    audio_latent = inputs.audio.latent.clone()
    video_latent[:, :video_half] = clean_video
    audio_latent[:, :audio_half] = clean_audio
    video_latent[:, video_half + video_start : video_half + video_end] = current_video
    audio_latent[:, audio_half + audio_start : audio_half + audio_end] = current_audio
    video_targets = torch.cat([clean_video, clean_video], dim=1)
    audio_targets = torch.cat([clean_audio, clean_audio], dim=1)
    return replace(
        inputs,
        video=replace(inputs.video, latent=video_latent),
        audio=replace(inputs.audio, latent=audio_latent),
        video_targets=video_targets,
        audio_targets=audio_targets,
        video_clean_latent=video_targets,
        video_noisy_latent=video_latent,
        audio_clean_latent=audio_targets,
        audio_noisy_latent=audio_latent,
    )


def current_block_x0(
    inputs: ModelInputs,
    *,
    block_index: int,
    video_velocity: Tensor,
    audio_velocity: Tensor,
) -> tuple[Tensor, Tensor]:
    ''

    if inputs.audio is None:
        raise ValueError("Current-block extraction requires audio.")
    video_ranges = inputs.ar_video_block_token_ranges
    audio_ranges = inputs.ar_audio_block_token_ranges
    if video_ranges is None or audio_ranges is None:
        raise ValueError("Current-block extraction requires AR block token ranges.")
    video_half = inputs.video.latent.shape[1] // 2
    audio_half = inputs.audio.latent.shape[1] // 2
    video_start, video_end = video_ranges[block_index]
    audio_start, audio_end = audio_ranges[block_index]
    video_t = inputs.video.timesteps[
        :, video_half + video_start : video_half + video_end
    ].unsqueeze(-1)
    audio_t = inputs.audio.timesteps[
        :, audio_half + audio_start : audio_half + audio_end
    ].unsqueeze(-1)
    video_latent = inputs.video.latent[
        :, video_half + video_start : video_half + video_end
    ]
    audio_latent = inputs.audio.latent[
        :, audio_half + audio_start : audio_half + audio_end
    ]
    video_v = video_velocity[:, video_half + video_start : video_half + video_end]
    audio_v = audio_velocity[:, audio_half + audio_start : audio_half + audio_end]
    return video_latent - video_t * video_v, audio_latent - audio_t * audio_v


def inject_sliding_ref_target_state(
    inputs: ModelInputs,
    *,
    current_video: Tensor,
    current_audio: Tensor,
) -> ModelInputs:
    ''

    if inputs.audio is None:
        raise ValueError("Sliding-ref self-forcing requires the audio modality.")
    video_ranges = inputs.ar_video_block_token_ranges
    audio_ranges = inputs.ar_audio_block_token_ranges
    if not video_ranges or not audio_ranges or len(video_ranges) != len(audio_ranges):
        raise ValueError("Sliding-ref self-forcing requires aligned block ranges.")
    video_start, video_end = video_ranges[-1]
    audio_start, audio_end = audio_ranges[-1]
    if current_video.shape != inputs.video.latent[:, video_start:video_end].shape:
        raise ValueError("Current video state does not match the sliding target range.")
    if current_audio.shape != inputs.audio.latent[:, audio_start:audio_end].shape:
        raise ValueError("Current audio state does not match the sliding target range.")
    video_latent = inputs.video.latent.clone()
    audio_latent = inputs.audio.latent.clone()
    video_latent[:, video_start:video_end] = current_video
    audio_latent[:, audio_start:audio_end] = current_audio
    return replace(
        inputs,
        video=replace(inputs.video, latent=video_latent),
        audio=replace(inputs.audio, latent=audio_latent),
        video_noisy_latent=video_latent,
        audio_noisy_latent=audio_latent,
    )


def sliding_ref_target_x0(
    inputs: ModelInputs,
    *,
    video_velocity: Tensor,
    audio_velocity: Tensor,
) -> tuple[Tensor, Tensor]:
    ''

    if inputs.audio is None:
        raise ValueError("Sliding-ref target extraction requires audio.")
    video_ranges = inputs.ar_video_block_token_ranges
    audio_ranges = inputs.ar_audio_block_token_ranges
    if not video_ranges or not audio_ranges:
        raise ValueError("Sliding-ref target extraction requires block ranges.")
    video_start, video_end = video_ranges[-1]
    audio_start, audio_end = audio_ranges[-1]
    video_t = inputs.video.timesteps[:, video_start:video_end].unsqueeze(-1)
    audio_t = inputs.audio.timesteps[:, audio_start:audio_end].unsqueeze(-1)
    video_latent = inputs.video.latent[:, video_start:video_end]
    audio_latent = inputs.audio.latent[:, audio_start:audio_end]
    video_v = video_velocity[:, video_start:video_end]
    audio_v = audio_velocity[:, audio_start:audio_end]
    return video_latent - video_t * video_v, audio_latent - audio_t * audio_v


def _masked_feature_mean(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape[:2] != mask.shape:
        raise ValueError(
            f"Masked feature reduction mismatch: values={tuple(values.shape)}, mask={tuple(mask.shape)}."
        )
    per_token = values.mean(dim=tuple(range(2, values.ndim)))
    active = mask.to(dtype=per_token.dtype)
    return (per_token * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def conditional_dmd_target(
    *,
    real_score: nn.Module,
    fake_score: nn.Module,
    score_inputs: ModelInputs,
    generated: GeneratedX0,
    action_button_meaning_fusion: str,
) -> ConditionalDMDTarget:
    ''

    fake_video, fake_audio = predict_score_x0(
        fake_score,
        score_inputs,
        action_button_meaning_fusion=action_button_meaning_fusion,
    )
    real_video, real_audio = predict_score_x0(
        real_score,
        score_inputs,
        action_button_meaning_fusion=action_button_meaning_fusion,
    )
    video_norm = _masked_feature_mean(
        (generated.video.float() - real_video.float()).abs(),
        generated.video_mask,
    ).view(-1, 1, 1)
    audio_norm = _masked_feature_mean(
        (generated.audio.float() - real_audio.float()).abs(),
        generated.audio_mask,
    ).view(-1, 1, 1)
    grad_video = torch.nan_to_num(
        (fake_video.float() - real_video.float()) / (video_norm + 1e-8)
    )
    grad_audio = torch.nan_to_num(
        (fake_audio.float() - real_audio.float()) / (audio_norm + 1e-8)
    )
    target_video = (generated.video.float() - grad_video).detach()
    target_audio = (generated.audio.float() - grad_audio).detach()
    metrics = {
        "dmd_gradient_video": _masked_feature_mean(
            grad_video.abs(), generated.video_mask
        ).mean(),
        "dmd_gradient_audio": _masked_feature_mean(
            grad_audio.abs(), generated.audio_mask
        ).mean(),
        "dmd_normalizer_video": video_norm.mean(),
        "dmd_normalizer_audio": audio_norm.mean(),
        "real_fake_delta_video": _masked_feature_mean(
            (real_video.float() - fake_video.float()).abs(),
            generated.video_mask,
        ).mean(),
        "real_fake_delta_audio": _masked_feature_mean(
            (real_audio.float() - fake_audio.float()).abs(),
            generated.audio_mask,
        ).mean(),
    }
    return ConditionalDMDTarget(video=target_video, audio=target_audio, metrics=metrics)


def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Consistency tensors must match, got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    if prediction.shape[:2] != mask.shape:
        raise ValueError(
            f"Consistency mask {tuple(mask.shape)} does not match tensor prefix {tuple(prediction.shape[:2])}."
        )
    per_token = (prediction.float() - target.float()).square().mean(dim=-1)
    active = mask.to(dtype=per_token.dtype)
    return (per_token * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)


def consistency_loss(
    prediction: GeneratedX0,
    target: GeneratedX0,
    *,
    audio_weight: float,
) -> ConsistencyLossOutput:
    ''

    if audio_weight < 0:
        raise ValueError(f"audio_weight must be non-negative, got {audio_weight}.")
    if not torch.equal(prediction.video_mask, target.video_mask):
        raise ValueError("Video consistency masks differ between adjacent DCM times.")
    if not torch.equal(prediction.audio_mask, target.audio_mask):
        raise ValueError("Audio consistency masks differ between adjacent DCM times.")
    video = _masked_mse(prediction.video, target.video.detach(), prediction.video_mask)
    audio = _masked_mse(prediction.audio, target.audio.detach(), prediction.audio_mask)
    return ConsistencyLossOutput(
        total=video + float(audio_weight) * audio, video=video, audio=audio
    )


__all__ = [
    "FOUR_STEP_SIGMAS",
    "ConsistencyLossOutput",
    "ConditionalDMDTarget",
    "FixedBatchSigmaSampler",
    "TFDCMTimePair",
    "TFDCMTimePairSampler",
    "UniformShiftSigmaSampler",
    "advance_causal_teacher_inputs",
    "conditional_dmd_target",
    "consistency_loss",
    "current_block_x0",
    "cyclic_simulation_steps",
    "extract_masked_generated_x0",
    "inject_self_forcing_state",
    "inject_sliding_ref_target_state",
    "power_ema_beta",
    "shift_rectified_flow_time",
    "sliding_ref_target_x0",
    "update_ema_parameters",
]
