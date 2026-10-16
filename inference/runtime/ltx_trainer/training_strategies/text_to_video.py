''

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    LossOutputs,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class TextToVideoConfig(TrainingStrategyConfigBase):
    ''

    name: Literal["text_to_video"] = "text_to_video"

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    with_audio: bool = Field(
        default=False,
        description="Whether to include audio in training (joint audio-video generation)",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for audio latents when with_audio is True",
    )


class TextToVideoStrategy(TrainingStrategy):
    ''

    config: TextToVideoConfig

    def __init__(self, config: TextToVideoConfig):
        ''
        super().__init__(config)

    @property
    def requires_audio(self) -> bool:
        ''
        return self.config.with_audio

    def get_data_sources(self) -> list[str] | dict[str, str]:
        ''
        sources = {
            "latents": "latents",
            "conditions": "conditions",
        }

        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"

        return sources

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        ''
                                                                                                  
        latents = batch["latents"]
        video_latents = latents["latents"]

                                                                   
        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

                                                              
        video_latents = self._video_patchifier.patchify(video_latents)

                                                
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

                                                                                    
        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]
        video_prompt_embeds, audio_prompt_embeds, prompt_attention_mask = self._apply_conditioning_dropout(
            video_prompt_embeds=video_prompt_embeds,
            audio_prompt_embeds=audio_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
        )

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

                                                             
        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

                                 
        sigmas = timestep_sampler.sample_for(video_latents)
        video_noise = torch.randn_like(video_latents)

                                                                  
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

                                                    
        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

                                                     
        video_targets = video_noise - video_latents

                                    
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
            enabled=True,
            sigma=sigmas,
            latent=noisy_video,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

                                                                                               
        video_loss_mask = ~video_conditioning_mask

                                 
        audio_modality = None
        audio_targets = None
        audio_loss_mask = None

        if self.config.with_audio:
            audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
                batch=batch,
                sigmas=sigmas,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
        )

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        sigmas: Tensor,
        audio_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Modality, Tensor, Tensor]:
        ''
                                                                                         
        audio_data = batch["audio_latents"]
        audio_latents = audio_data["latents"]

                                                             
        audio_latents = self._audio_patchifier.patchify(audio_latents)

        audio_seq_len = audio_latents.shape[1]

                            
        audio_noise = torch.randn_like(audio_latents)

                                                    
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_audio = (1 - sigmas_expanded) * audio_latents + sigmas_expanded * audio_noise

                               
        audio_targets = audio_noise - audio_latents

                                                                                  
        audio_timesteps = sigmas.view(-1, 1).expand(-1, audio_seq_len)

                                  
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

                               
        audio_modality = Modality(
            enabled=True,
            latent=noisy_audio,
            sigma=sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

                                                                          
        audio_loss_mask = torch.ones(batch_size, audio_seq_len, dtype=torch.bool, device=device)

        return audio_modality, audio_targets, audio_loss_mask

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> LossOutputs:
        ''
                                                                 
        video_loss = (video_pred.float() - inputs.video_targets.float()).pow(2)
        video_loss_mask = inputs.video_loss_mask.unsqueeze(-1).float()
        masked = video_loss.mul(video_loss_mask)
        video_loss = masked.mean(dim=[-2, -1]) / video_loss_mask.mean(dim=[-2, -1]).clamp(min=1e-8)

                                             
        if not self.config.with_audio or audio_pred is None or inputs.audio_targets is None:
            return LossOutputs(
                total=video_loss,
                metrics={
                    "loss": video_loss.mean(),
                    "loss_video": video_loss.mean(),
                    "loss_audio": torch.zeros((), device=video_loss.device, dtype=torch.float32),
                },
            )

                                                                 
        audio_loss = (audio_pred.float() - inputs.audio_targets.float()).pow(2).mean(dim=[-2, -1])

                            
        total = video_loss + audio_loss
        return LossOutputs(
            total=total,
            metrics={
                "loss": total.mean(),
                "loss_video": video_loss.mean(),
                "loss_audio": audio_loss.mean(),
            },
        )
