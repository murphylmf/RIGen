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


class VideoToVideoConfig(TrainingStrategyConfigBase):
    ''

    name: Literal["video_to_video"] = "video_to_video"

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    reference_latents_dir: str = Field(
        default="reference_latents",
        description="Directory name for latents of reference videos",
    )


class VideoToVideoStrategy(TrainingStrategy):
    ''

    config: VideoToVideoConfig
    reference_downscale_factor: int | None

    def __init__(self, config: VideoToVideoConfig):
        ''
        super().__init__(config)
        self.reference_downscale_factor = None                                     

    def get_data_sources(self) -> dict[str, str]:
        ''
        return {
            "latents": "latents",
            "conditions": "conditions",
            self.config.reference_latents_dir: "ref_latents",
        }

    def prepare_training_inputs(                 
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        ''
                                                                                                  
        latents = batch["latents"]
        target_latents = latents["latents"]
        ref_latents = batch["ref_latents"]["latents"]

                        
        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        ref_latents_info = batch["ref_latents"]
        ref_frames = ref_latents_info["num_frames"][0].item()
        ref_height = ref_latents_info["height"][0].item()
        ref_width = ref_latents_info["width"][0].item()

                                                                
                                                                              
        reference_downscale_factor = self._infer_reference_downscale_factor(
            target_height=height,
            target_width=width,
            ref_height=ref_height,
            ref_width=ref_width,
        )

                                                                          
        if self.reference_downscale_factor is None:
            self.reference_downscale_factor = reference_downscale_factor
        elif self.reference_downscale_factor != reference_downscale_factor:
            raise ValueError(
                f"Inconsistent reference downscale factor across batches. "
                f"First batch had factor={self.reference_downscale_factor}, "
                f"but current batch has factor={reference_downscale_factor}. "
                f"All training samples must use the same reference/target resolution ratio."
            )

                                                              
        target_latents = self._video_patchifier.patchify(target_latents)
        ref_latents = self._video_patchifier.patchify(ref_latents)

                    
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

                                                                                    
                                                   
        conditions = batch["conditions"]
        prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]
        prompt_embeds, _, prompt_attention_mask = self._apply_conditioning_dropout(
            video_prompt_embeds=prompt_embeds,
            audio_prompt_embeds=None,
            prompt_attention_mask=prompt_attention_mask,
        )

        batch_size = target_latents.shape[0]
        ref_seq_len = ref_latents.shape[1]
        target_seq_len = target_latents.shape[1]
        device = target_latents.device
        dtype = target_latents.dtype

                                  
                                                               
        ref_conditioning_mask = torch.ones(batch_size, ref_seq_len, dtype=torch.bool, device=device)

                                                           
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

                                    
        conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)

                                            
        sigmas = timestep_sampler.sample_for(target_latents)
        noise = torch.randn_like(target_latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)

                               
        noisy_target = (1 - sigmas_expanded) * target_latents + sigmas_expanded * noise

                                                                   
        target_conditioning_mask_expanded = target_conditioning_mask.unsqueeze(-1)
        noisy_target = torch.where(target_conditioning_mask_expanded, target_latents, noisy_target)

                                      
        targets = noise - target_latents

                                                          
        combined_latents = torch.cat([ref_latents, noisy_target], dim=1)

                                    
        timesteps = self._create_per_token_timesteps(conditioning_mask, sigmas.squeeze())

                                                                                  
        ref_positions = self._get_video_positions(
            num_frames=ref_frames,
            height=ref_height,
            width=ref_width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

                                                                    
                                                                                   
                                                                                        
        if reference_downscale_factor != 1:
            ref_positions = ref_positions.clone()
            ref_positions[:, 1, ...] *= reference_downscale_factor               
            ref_positions[:, 2, ...] *= reference_downscale_factor              
                                                   

        target_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

                                                        
        positions = torch.cat([ref_positions, target_positions], dim=2)

                               
        video_modality = Modality(
            enabled=True,
            latent=combined_latents,
            sigma=sigmas,
            timesteps=timesteps,
            positions=positions,
            context=prompt_embeds,
            context_mask=prompt_attention_mask,
        )

                                                                        
                                               
                                                    
        ref_loss_mask = torch.zeros(batch_size, ref_seq_len, dtype=torch.bool, device=device)
        target_loss_mask = ~target_conditioning_mask
        video_loss_mask = torch.cat([ref_loss_mask, target_loss_mask], dim=1)

        return ModelInputs(
            video=video_modality,
            audio=None,
            video_targets=targets,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
            ref_seq_len=ref_seq_len,
        )

    def compute_loss(
        self,
        video_pred: Tensor,
        _audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> LossOutputs:
        ''
                                              
        ref_seq_len = inputs.ref_seq_len
        target_pred = video_pred[:, ref_seq_len:, :]

                                         
        target_loss_mask = inputs.video_loss_mask[:, ref_seq_len:]

                                       
        loss = (target_pred.float() - inputs.video_targets.float()).pow(2)
        loss_mask = target_loss_mask.unsqueeze(-1).float()
        masked = loss.mul(loss_mask)
        total = masked.mean(dim=[-2, -1]) / loss_mask.mean(dim=[-2, -1]).clamp(min=1e-8)
        return LossOutputs(
            total=total,
            metrics={
                "loss": total.mean(),
                "loss_video": total.mean(),
                "loss_audio": torch.zeros((), device=total.device, dtype=torch.float32),
            },
        )

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        ''
        metadata: dict[str, Any] = {}
                                                                             
                                                                        
        if self.reference_downscale_factor is not None:
            metadata["reference_downscale_factor"] = self.reference_downscale_factor
        return metadata

    @staticmethod
    def _infer_reference_downscale_factor(
        target_height: int,
        target_width: int,
        ref_height: int,
        ref_width: int,
    ) -> int:
        ''
                                                
        if target_height == ref_height and target_width == ref_width:
            return 1

                                                    
        if target_height % ref_height != 0 or target_width % ref_width != 0:
            raise ValueError(
                f"Target dimensions ({target_height}x{target_width}) must be exact multiples "
                f"of reference dimensions ({ref_height}x{ref_width})"
            )

        scale_h = target_height // ref_height
        scale_w = target_width // ref_width

        if scale_h != scale_w:
            raise ValueError(
                f"Reference scale must be uniform. Got height scale {scale_h} and width scale {scale_w}. "
                f"Target: {target_height}x{target_width}, Reference: {ref_height}x{ref_width}"
            )

        if scale_h < 1:
            raise ValueError(
                f"Reference dimensions ({ref_height}x{ref_width}) cannot be larger than "
                f"target dimensions ({target_height}x{target_width})"
            )

        return scale_h
