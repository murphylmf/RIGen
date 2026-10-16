from __future__ import annotations

from typing import Any, Literal

import torch
from ltx_core.model.transformer.modality import Modality
from ltx_core.tools import AudioLatentTools
from ltx_core.types import AudioLatentShape, VideoLatentShape
from ltx_core.utils import to_denoised
from pydantic import Field
from torch import Tensor

from ltx_trainer.action_binding import (
    CANONICAL_BUTTONS,
    canonicalize_button_tensor,
    crossfps_fixed_button_state_dim,
    crossfps_fixed_button_names,
    is_crossfps_fixed_button_mode,
)
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    LossOutputs,
    ModelInputs,
    ModelOutputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class PhaseAwareConfig(TrainingStrategyConfigBase):
    name: Literal["phase_aware"] = "phase_aware"
    forward_mode: Literal["audio_video", "video_only", "audio_only"] = "audio_video"
    loss_alpha_audio: float = Field(default=1.0, ge=0.0)
    i2v_self_frame: bool = False
    synthetic_audio_state: bool = False
    action_button_temporal_alignment: Literal["global", "vae_window", "causal_prefix_rope"] = Field(
        default="global",
        description=(
            "global preserves the original all-raw-frame button cross-attention; "
            "vae_window aligns each video latent time to the causal 8-frame VAE window; "
            "causal_prefix_rope lets each latent attend once to all current/past raw actions "
            "with a prefix mask and shared Q/K temporal RoPE."
        ),
    )
    action_dropout_prob: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Probability of replacing action conditioning with an explicit idle/no-action condition. "
            "This trains the action-dropped branch needed for action guidance at inference."
        ),
    )
    action_button_uncond_prob: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Probability of disabling the discrete-button conditioning branch for a sample. "
            "Unlike action_dropout_prob, this is a true null condition and does not masquerade as idle."
        ),
    )
    action_joystick_uncond_prob: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Probability of disabling the continuous-joystick conditioning branch for a sample. "
            "A disabled branch is distinct from a valid centered (all-zero) joystick."
        ),
    )

    def model_post_init(self, __context: Any) -> None:
        if self.cfg_training_enabled:
            if self.conditioning_dropout_p != 0.0:
                raise ValueError("joint CFG training requires training_strategy.conditioning_dropout_p=0.0")
            if self.joint_uncond_prob <= 0.0:
                raise ValueError("joint CFG training requires training_strategy.joint_uncond_prob > 0")


class PhaseAwareStrategy(TrainingStrategy):
    config: PhaseAwareConfig

    def __init__(self, config: PhaseAwareConfig):
        super().__init__(config)
        self._last_inputs: ModelInputs | None = None
        self._action_binding_config: Any | None = None

    def set_action_binding_config(self, config: Any) -> None:
        self._action_binding_config = config

    def _action_binding_mode(self) -> str:
        action_binding = self._action_binding_config
        if action_binding is None or not getattr(action_binding, "enabled", False):
            return "text_binding"
        return str(getattr(action_binding, "button_conditioning_mode", "text_binding"))

    @property
    def requires_audio(self) -> bool:
        return self.config.forward_mode in {"audio_video", "audio_only"}

    def get_data_sources(self) -> list[str] | dict[str, str]:
        return []

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler,
    ) -> ModelInputs:
        video_latents = batch["video_latents"]
        batch_size, _channels, num_frames, height, width = video_latents.shape
        fps_value = batch.get("fps")
        if torch.is_tensor(fps_value):
            fps = float(fps_value[0].item())
        else:
            fps = float(fps_value[0]) if isinstance(fps_value, list) else DEFAULT_FPS

        video_latents_patch = self._video_patchifier.patchify(video_latents)
        device = video_latents_patch.device
        dtype = video_latents_patch.dtype
        sigmas = timestep_sampler.sample_for(video_latents_patch)
        joint_uncond_mask = None
        if self.config.cfg_training_enabled:
            joint_uncond_mask = torch.rand(batch_size, device=device) < self.config.joint_uncond_prob
            video_prompt_embeds = batch["video_prompt_embeds"]
            audio_prompt_embeds = batch.get("audio_prompt_embeds")
            prompt_attention_mask = batch["prompt_attention_mask"]
            if torch.any(joint_uncond_mask):
                video_prompt_embeds = video_prompt_embeds.clone()
                video_prompt_embeds[joint_uncond_mask] = 0
                if audio_prompt_embeds is not None:
                    audio_prompt_embeds = audio_prompt_embeds.clone()
                    audio_prompt_embeds[joint_uncond_mask] = 0
                prompt_attention_mask = prompt_attention_mask.clone()
                prompt_attention_mask[joint_uncond_mask] = 0
        else:
            video_prompt_embeds, audio_prompt_embeds, prompt_attention_mask = self._apply_conditioning_dropout(
                video_prompt_embeds=batch["video_prompt_embeds"],
                audio_prompt_embeds=batch.get("audio_prompt_embeds"),
                prompt_attention_mask=batch["prompt_attention_mask"],
            )

        video_conditioning_mask = torch.zeros(
            batch_size, video_latents_patch.shape[1], dtype=torch.bool, device=device
        )
        if self.config.i2v_self_frame:
            video_conditioning_mask[:, : height * width] = True

        video_noise = torch.randn_like(video_latents_patch)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * video_latents_patch + sigmas_expanded * video_noise
        noisy_video = torch.where(video_conditioning_mask.unsqueeze(-1), video_latents_patch, noisy_video)
        video_targets = video_noise - video_latents_patch
        video_timesteps = self._create_per_token_timesteps(video_conditioning_mask, sigmas.squeeze())
        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )
        video_modality = Modality(
            enabled=self.config.forward_mode != "audio_only",
            latent=noisy_video,
            sigma=sigmas,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        video_loss_mask = ~video_conditioning_mask

        action_joystick_tokens, action_button_state, action_button_values = self._prepare_action_tokens(
            batch,
            latent_frames=num_frames,
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

        audio_modality = None
        audio_targets = None
        audio_loss_mask = None
        audio_clean_patch = None
        audio_noisy_patch = None
        if self.config.forward_mode in {"audio_video", "audio_only"}:
            audio_modality, audio_targets, audio_loss_mask, audio_clean_patch, audio_noisy_patch = self._prepare_audio(
                batch=batch,
                sigmas=sigmas,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
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
            first_frame_latent=video_latents[:, :, :1].clone() if self.config.i2v_self_frame else None,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
            video_clean_latent=video_latents_patch,
            video_noisy_latent=noisy_video,
            audio_clean_latent=audio_clean_patch,
            audio_noisy_latent=audio_noisy_patch,
        )
        self._last_inputs = inputs
        return inputs

    @staticmethod
    def _sample_condition_present_mask(
        condition: Tensor | None,
        *,
        uncond_prob: float,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        ''
        if condition is None:
            return None
        if uncond_prob <= 0.0:
            return torch.ones(batch_size, 1, 1, device=device, dtype=dtype)
        present = torch.rand(batch_size, device=device) >= float(uncond_prob)
        return present.to(dtype=dtype).view(batch_size, 1, 1)

    def _text_dropout_prob(self) -> float:
        if self.config.text_dropout_prob is not None:
            return float(self.config.text_dropout_prob)
        if self.config.joint_uncond_prob > 0:
            return float(self.config.joint_uncond_prob)
        return float(self.config.conditioning_dropout_p)

    def _apply_action_dropout(
        self,
        joystick_tokens: Tensor | None,
        button_state: Tensor | None,
        button_values: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        dropout_p = float(self.config.action_dropout_prob)
        if (
            dropout_p <= 0.0
            or joystick_tokens is None
            or button_state is None
            or button_values is None
        ):
            return joystick_tokens, button_state, button_values

        drop_mask = torch.rand(batch_size, device=device) < dropout_p
        if not torch.any(drop_mask):
            return joystick_tokens, button_state, button_values

        joystick_tokens = joystick_tokens.clone()
        button_state = button_state.clone()
        button_values = button_values.clone()

        joystick_tokens[drop_mask] = 0
        button_dim = button_values.shape[-1]
        button_state[drop_mask, :, :button_dim] = 0
        if button_state.shape[-1] == button_dim + 1:
            button_state[drop_mask, :, button_dim] = 1
        elif button_state.shape[-1] == button_dim + 2:
            button_state[drop_mask, :, button_dim] = 0
            button_state[drop_mask, :, button_dim + 1] = 1
        else:
            raise ValueError(
                f"Unsupported action button state layout: buttons={button_dim} state={button_state.shape[-1]}"
            )
        button_values[drop_mask] = 0
        return joystick_tokens, button_state.to(dtype=dtype), button_values.to(dtype=dtype)

    def _prepare_button_meaning_embeds(
        self,
        batch: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if is_crossfps_fixed_button_mode(self._action_binding_mode()):
            return None
        embeds = batch.get("action_button_meaning_embeds")
        if not torch.is_tensor(embeds):
            return None
        embeds = embeds.to(device=device, dtype=dtype)
        if embeds.ndim != 3 or embeds.shape[0] != batch_size or embeds.shape[1] != len(CANONICAL_BUTTONS):
            raise ValueError(
                "action_button_meaning_embeds must have shape "
                f"[B, {len(CANONICAL_BUTTONS)}, D], got {tuple(embeds.shape)}"
            )
        return embeds.contiguous()

    def _apply_action_binding_dropout(
        self,
        button_state: Tensor | None,
        button_meaning_embeds: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor | None, Tensor | None]:
        action_binding = self._action_binding_config
        if action_binding is None or not getattr(action_binding, "enabled", False):
            return button_state, button_meaning_embeds
        if is_crossfps_fixed_button_mode(self._action_binding_mode()):
            return button_state, None
        dropout_cfg = getattr(action_binding, "dropout", None)
        if dropout_cfg is None:
            return button_state, button_meaning_embeds

        button_id_dropout_p = float(getattr(dropout_cfg, "button_id_dropout_prob", 0.0))
        if button_state is not None and button_id_dropout_p > 0.0:
            drop_mask = torch.rand(batch_size, device=device) < button_id_dropout_p
            if torch.any(drop_mask):
                button_state = button_state.clone()
                button_dim = max(int(button_state.shape[-1]) - 1, 0)
                button_state[drop_mask, :, :button_dim] = 0
                button_state = button_state.to(dtype=dtype)

        meaning_dropout_p = float(getattr(dropout_cfg, "button_meaning_dropout_prob", 0.0))
        if button_meaning_embeds is not None and meaning_dropout_p > 0.0:
            drop_mask = torch.rand(
                button_meaning_embeds.shape[0],
                button_meaning_embeds.shape[1],
                1,
                device=device,
            ) < meaning_dropout_p
            if torch.any(drop_mask):
                button_meaning_embeds = button_meaning_embeds.clone()
                button_meaning_embeds = torch.where(
                    drop_mask,
                    torch.zeros((), dtype=dtype, device=device),
                    button_meaning_embeds,
                )
        return button_state, button_meaning_embeds

    def _prepare_action_tokens(
        self,
        batch: dict[str, Any],
        *,
        latent_frames: int,
        height: int,
        width: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        joystick = batch.get("action_joystick")
        buttons = batch.get("action_buttons")
        if not isinstance(joystick, torch.Tensor) or not isinstance(buttons, torch.Tensor):
            return None, None, None

        joystick = joystick.to(device=device, dtype=dtype)
        button_mode = self._action_binding_mode()
        if button_mode == "semantic_vocab" or is_crossfps_fixed_button_mode(button_mode):
            buttons = buttons.to(device=device, dtype=dtype)
        else:
            buttons = canonicalize_button_tensor(buttons).to(device=device, dtype=dtype)
        if joystick.ndim != 3 or joystick.shape[-1] != 4:
            raise ValueError(f"action_joystick must have shape [B, raw_frames, 4], got {tuple(joystick.shape)}")
        if buttons.ndim != 3:
            raise ValueError(f"action_buttons must have shape [B, raw_frames, button_dim], got {tuple(buttons.shape)}")
        if is_crossfps_fixed_button_mode(button_mode):
            expected_buttons = crossfps_fixed_button_names(button_mode)
            if buttons.shape[-1] != len(expected_buttons):
                raise ValueError(
                    f"{button_mode} action_buttons must have shape "
                    f"[B, raw_frames, {len(expected_buttons)}], got {tuple(buttons.shape)}"
                )
        if (
            button_mode != "semantic_vocab"
            and not is_crossfps_fixed_button_mode(button_mode)
            and buttons.shape[-1] != len(CANONICAL_BUTTONS)
        ):
            raise ValueError(
                f"action_buttons must have shape [B, raw_frames, {len(CANONICAL_BUTTONS)}], got {tuple(buttons.shape)}"
            )
        if button_mode == "semantic_vocab" and buttons.shape[-1] <= 0:
            raise ValueError(f"semantic_vocab action_buttons must have a positive button dim, got {tuple(buttons.shape)}")
        if joystick.shape[:2] != buttons.shape[:2]:
            raise ValueError(
                f"action_joystick/action_buttons shape mismatch: {tuple(joystick.shape)} vs {tuple(buttons.shape)}"
            )
        if joystick.shape[0] != batch_size:
            raise ValueError(f"Action batch size {joystick.shape[0]} does not match video batch size {batch_size}")

        raw_frames = int(joystick.shape[1])
        expected_raw_frames = (latent_frames - 1) * 8 + 1
        if raw_frames != expected_raw_frames:
            raise ValueError(
                f"Action raw frame count {raw_frames} does not match latent_frames={latent_frames}; "
                f"expected {(latent_frames - 1) * 8 + 1}"
            )

        latent_joystick = torch.zeros(batch_size, latent_frames, 32, device=device, dtype=dtype)
        for latent_index in range(1, latent_frames):
            start = 8 * (latent_index - 1) + 1
            end = 8 * latent_index + 1
            latent_joystick[:, latent_index] = joystick[:, start:end, :].reshape(batch_size, 32)
        joystick_tokens = (
            latent_joystick[:, :, None, None, :]
            .expand(batch_size, latent_frames, height, width, 32)
            .reshape(batch_size, latent_frames * height * width, 32)
            .contiguous()
        )

        if button_mode == "prompt_timeline":
            return joystick_tokens, None, None

        active = buttons.abs().sum(dim=-1, keepdim=True) > 0
        idle = (~active).to(dtype=dtype)
        expected_state_dim = (
            crossfps_fixed_button_state_dim(button_mode)
            if is_crossfps_fixed_button_mode(button_mode)
            else buttons.shape[-1] + 1
        )
        if expected_state_dim == buttons.shape[-1] + 1:
            button_state = torch.cat([buttons, idle], dim=-1).contiguous()
        elif expected_state_dim == buttons.shape[-1] + 2:
            button_state = torch.cat([buttons, active.to(dtype=dtype), idle], dim=-1).contiguous()
        else:
            raise ValueError(
                f"Unsupported action button state layout: buttons={buttons.shape[-1]} state={expected_state_dim}"
            )
        button_values = torch.where(active, buttons, torch.zeros_like(buttons)).contiguous()
        return joystick_tokens, button_state, button_values

    def _prepare_audio(
        self,
        batch: dict[str, Any],
        sigmas: Tensor,
        audio_prompt_embeds: Tensor | None,
        prompt_attention_mask: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Modality, Tensor, Tensor, Tensor, Tensor]:
        has_audio = batch.get("has_audio")
        if has_audio is None:
            has_audio = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            has_audio = has_audio.to(device=device, dtype=torch.bool)
        audio_loss_eligible = batch.get("audio_loss_eligible")
        if audio_loss_eligible is None:
            audio_loss_eligible = has_audio
        else:
            audio_loss_eligible = audio_loss_eligible.to(device=device, dtype=torch.bool) & has_audio

        audio_latents = batch.get("audio_latents")
        if isinstance(audio_latents, torch.Tensor):
            clean_audio = audio_latents.to(device=device, dtype=dtype)
        else:
            if not self.config.synthetic_audio_state:
                raise ValueError("Missing audio latents for PhaseAwareStrategy without synthetic_audio_state enabled.")
            shape = self._infer_audio_shape(batch)
            clean_audio = AudioLatentTools(
                patchifier=self._audio_patchifier,
                target_shape=shape._replace(batch=batch_size),
            ).create_initial_state(device=device, dtype=dtype).latent
            clean_audio = self._audio_patchifier.unpatchify(clean_audio, output_shape=shape._replace(batch=batch_size))

        audio_patch = self._audio_patchifier.patchify(clean_audio)
        audio_noise = torch.randn_like(audio_patch)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_audio = (1 - sigmas_expanded) * audio_patch + sigmas_expanded * audio_noise
        audio_targets = audio_noise - audio_patch
        audio_loss_mask = audio_loss_eligible[:, None].expand(-1, audio_patch.shape[1])

        audio_timesteps = sigmas.view(-1, 1).expand(-1, audio_patch.shape[1])
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_patch.shape[1],
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        audio_modality = Modality(
            enabled=self.config.forward_mode != "video_only",
            latent=noisy_audio,
            sigma=sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
        )
        return audio_modality, audio_targets, audio_loss_mask, audio_patch, noisy_audio

    def _infer_audio_shape(self, batch: dict[str, Any]) -> AudioLatentShape:
        audio_latents = batch.get("audio_latents")
        if isinstance(audio_latents, torch.Tensor):
            return AudioLatentShape.from_torch_shape(audio_latents.shape)
        video_latents = batch["video_latents"]
        fps = batch.get("fps")
        fps_value = float(fps[0].item()) if torch.is_tensor(fps) else DEFAULT_FPS
        duration = ((video_latents.shape[2] - 1) * 8 + 1) / fps_value
        return AudioLatentShape.from_duration(batch=1, duration=duration)

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> LossOutputs:
        video_pred = video_pred.float()
        video_targets = inputs.video_targets.float()
        video_loss = (video_pred - video_targets).pow(2)
        video_loss_mask = inputs.video_loss_mask.unsqueeze(-1).float()
        video_loss = (video_loss * video_loss_mask).mean(dim=[-2, -1]) / video_loss_mask.mean(dim=[-2, -1]).clamp(
            min=1e-8
        )

        audio_loss_scalar = torch.zeros((), device=video_pred.device, dtype=torch.float32)
        total = video_loss
        if audio_pred is not None and inputs.audio_targets is not None and inputs.audio_loss_mask is not None:
            audio_loss = (audio_pred.float() - inputs.audio_targets.float()).pow(2).mean(dim=[-2, -1])
            has_audio = inputs.audio_loss_mask.any(dim=1).float()
            has_audio_count = has_audio.sum().clamp(min=1.0)
            audio_scale = has_audio.numel() / has_audio_count
            total = total + self.config.loss_alpha_audio * audio_loss * has_audio * audio_scale
            if has_audio.any():
                audio_loss_scalar = audio_loss[has_audio.bool()].mean()

        return LossOutputs(
            total=total,
            metrics={
                "loss": total.mean(),
                "loss_video": video_loss.mean(),
                "loss_audio": audio_loss_scalar,
            },
        )

    def collect_diagnostics(
        self,
        batch: dict[str, Any],
        outputs: ModelOutputs,
        losses: LossOutputs,
    ) -> dict[str, Tensor] | None:
        if self._last_inputs is None or outputs.video_pred is None:
            return None

        inputs = self._last_inputs
        video_shape = VideoLatentShape.from_torch_shape(batch["video_latents"].shape)
        pred_video = to_denoised(inputs.video_noisy_latent, outputs.video_pred, inputs.video.timesteps)
        clean_video = self._video_patchifier.unpatchify(
            inputs.video_clean_latent,
            output_shape=video_shape,
        )[0].detach().cpu().float()
        noisy_video = self._video_patchifier.unpatchify(
            inputs.video_noisy_latent,
            output_shape=video_shape,
        )[0].detach().cpu().float()
        pred_video_unpatch = self._video_patchifier.unpatchify(
            pred_video,
            output_shape=video_shape,
        )[0].detach().cpu().float()
        per_frame_loss = (pred_video_unpatch - clean_video).pow(2).mean(dim=(0, 2, 3))

        diag = {
            "clean_sample": clean_video.permute(1, 0, 2, 3).contiguous(),
            "noisy_sample": noisy_video.permute(1, 0, 2, 3).contiguous(),
            "pred_sample": pred_video_unpatch.permute(1, 0, 2, 3).contiguous(),
            "target_sample": clean_video.permute(1, 0, 2, 3).contiguous(),
            "per_frame_loss": per_frame_loss,
            "timestep": inputs.video.sigma[0].detach().cpu().float(),
            "fps": batch["fps"][0].detach().cpu().float() if torch.is_tensor(batch["fps"]) else torch.tensor(DEFAULT_FPS),
            "loss_mask_video": inputs.video_loss_mask[0].detach().cpu().float(),
            "loss_mask_audio": (
                inputs.audio_loss_mask[0].detach().cpu().float()
                if inputs.audio_loss_mask is not None
                else torch.zeros(1, dtype=torch.float32)
            ),
        }

        if outputs.audio_pred is not None and inputs.audio_clean_latent is not None and inputs.audio_noisy_latent is not None:
            audio_shape = self._infer_audio_shape(batch)._replace(batch=batch["video_latents"].shape[0])
            pred_audio = to_denoised(inputs.audio_noisy_latent, outputs.audio_pred, inputs.audio.timesteps)
            diag["audio_latent_clean"] = self._audio_patchifier.unpatchify(
                inputs.audio_clean_latent,
                output_shape=audio_shape,
            )[0].detach().cpu().float()
            diag["audio_latent_pred"] = self._audio_patchifier.unpatchify(
                pred_audio,
                output_shape=audio_shape,
            )[0].detach().cpu().float()
        return diag
