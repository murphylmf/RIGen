''

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import torch
from einops import rearrange
from torch import Tensor

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import (
    AudioPatchifier,
    VideoLatentPatchifier,
    get_pixel_coords,
)
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import X0Model
from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, LatentState, SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape
from ltx_trainer.action_binding import CANONICAL_BUTTONS, canonicalize_button_tensor
from ltx_trainer.progress import SamplingContext

if TYPE_CHECKING:
    from ltx_core.model.audio_vae import AudioDecoder, Vocoder
    from ltx_core.model.transformer import LTXModel
    from ltx_core.model.video_vae import VideoDecoder, VideoEncoder
    from ltx_core.text_encoders.gemma import GemmaTextEncoder
    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor

VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()
_POSITIVE_ONLY_GUIDER = MultiModalGuider(params=MultiModalGuiderParams(cfg_scale=1.0, stg_scale=0.0, modality_scale=1.0))


def _repeat_state(state: LatentState, n: int) -> LatentState:
    def _repeat(tensor: Tensor) -> Tensor:
        repeats = [1] * tensor.dim()
        repeats[0] = n
        return tensor.repeat(repeats)

    return LatentState(
        latent=_repeat(state.latent),
        denoise_mask=_repeat(state.denoise_mask),
        positions=_repeat(state.positions),
        clean_latent=_repeat(state.clean_latent),
        attention_mask=_repeat(state.attention_mask) if state.attention_mask is not None else None,
    )


def _repeat_optional_action_tensor(tensor: Tensor | None, n: int) -> Tensor | None:
    if tensor is None:
        return None
    repeats = [1] * tensor.dim()
    repeats[0] = n
    return tensor.repeat(repeats)


ActionConditioning = tuple[
    Tensor | None,                   
    Tensor | None,                
    Tensor | None,                 
    Tensor | None,                             
    Tensor | None,                                   
    Tensor | None,                                 
]


def _button_null_conditioning(action_conditioning: ActionConditioning) -> ActionConditioning:
    ''
    joystick, button_state, button_values, button_meaning_embeds, joystick_mask, button_mask = action_conditioning
    if button_mask is None:
        return action_conditioning
    return (
        joystick,
        button_state,
        button_values,
        button_meaning_embeds,
        joystick_mask,
        torch.zeros_like(button_mask),
    )


def _joystick_null_conditioning(action_conditioning: ActionConditioning) -> ActionConditioning:
    ''
    joystick, button_state, button_values, button_meaning_embeds, joystick_mask, button_mask = action_conditioning
    if joystick_mask is None:
        return action_conditioning
    return (
        joystick,
        button_state,
        button_values,
        button_meaning_embeds,
        torch.zeros_like(joystick_mask),
        button_mask,
    )


def _timesteps_from_mask(denoise_mask: Tensor, sigma: float | Tensor) -> Tensor:
    if isinstance(sigma, torch.Tensor) and sigma.dim() == 1:
        sigma = sigma.view(-1, *([1] * (denoise_mask.dim() - 1)))
    return denoise_mask * sigma


def _modality_from_latent_state(
    state: LatentState,
    context: Tensor,
    sigma: Tensor,
    *,
    enabled: bool = True,
) -> Modality:
    return Modality(
        enabled=enabled,
        latent=state.latent,
        sigma=sigma,
        timesteps=_timesteps_from_mask(state.denoise_mask, sigma),
        positions=state.positions,
        context=context,
        context_mask=None,
        attention_mask=state.attention_mask,
    )


@dataclass
class CachedPromptEmbeddings:
    ''

    video_context_positive: Tensor                            
    audio_context_positive: Tensor                            
    video_context_negative: Tensor | None = None
    audio_context_negative: Tensor | None = None


@dataclass
class TiledDecodingConfig:
    ''

    enabled: bool = True                                                      
    tile_size_pixels: int = 192                                                                 
    tile_overlap_pixels: int = 64                                                            
    tile_size_frames: int = 48                                                                 
    tile_overlap_frames: int = 24                                                            


@dataclass
class GenerationConfig:
    ''

    prompt: str                              
    negative_prompt: str = ""                                               
    height: int = 544                                 
    width: int = 960                                
    num_frames: int = 97                                
    frame_rate: float = 25.0                                            
    num_inference_steps: int = 30                             
    seed: int = 42                                   
    condition_image: Tensor | None = None                                                 
    reference_video: Tensor | None = None                                       
    reference_downscale_factor: int = 1                                                                            
    generate_audio: bool = True                                                
    use_audio_modality: bool | None = None                                                                  
    include_reference_in_output: bool = False                                                                     
    cached_embeddings: CachedPromptEmbeddings | None = None                                                       
    ground_truth_video: Tensor | None = None                                                         
    action_joystick: Tensor | None = None                                                           
    action_buttons: Tensor | None = None                                                                           
    action_guidance_scale: float = 0.0                                                        
    joystick_guidance_scale: float = 0.0                                                          
    action_button_temporal_alignment: Literal["global", "vae_window", "causal_prefix_rope"] = "global"
    button_meaning_texts: dict[str, str] | None = None                                                           
    action_button_meaning_fusion: str = "add"                                                                
    source_prompt: str | None = None                                                 
    video_guider_params: MultiModalGuiderParams = field(default_factory=MultiModalGuiderParams)
    audio_guider_params: MultiModalGuiderParams = field(default_factory=MultiModalGuiderParams)
    scheduler_timestep_shift: float | None = 2.05                                                                
    capture_intermediates: bool = True                                                                       
    intermediate_capture_count: int = 4                                                                           
                                                                                                                       
    tiled_decoding: TiledDecodingConfig | Literal[False] | None = None

    def __post_init__(self) -> None:
        ''
        if self.use_audio_modality is None:
            object.__setattr__(self, "use_audio_modality", self.generate_audio)
        if self.tiled_decoding is None:
                                                    
            object.__setattr__(self, "tiled_decoding", TiledDecodingConfig())
        elif self.tiled_decoding is False:
                                                                 
            object.__setattr__(self, "tiled_decoding", TiledDecodingConfig(enabled=False))


class ValidationSampler:
    ''

    def __init__(
        self,
        transformer: "LTXModel",
        vae_decoder: "VideoDecoder",
        vae_encoder: "VideoEncoder | None",
        text_encoder: "GemmaTextEncoder | None" = None,
        audio_decoder: "AudioDecoder | None" = None,
        vocoder: "Vocoder | None" = None,
        sampling_context: SamplingContext | None = None,
        embeddings_processor: "EmbeddingsProcessor | None" = None,
    ):
        ''
        self._transformer = transformer
        self._vae_decoder = vae_decoder
        self._vae_encoder = vae_encoder
        self._text_encoder = text_encoder
        self._embeddings_processor = embeddings_processor
        self._audio_decoder = audio_decoder
        self._vocoder = vocoder
        self._sampling_context = sampling_context

                     
        self._video_patchifier = VideoLatentPatchifier(patch_size=1)
        self._audio_patchifier = AudioPatchifier(patch_size=1)

                                                                                                                        
    @torch.no_grad()
    def generate(
        self,
        config: GenerationConfig,
        device: torch.device | str = "cuda",
    ) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
        ''
        device = torch.device(device) if isinstance(device, str) else device
        self._validate_config(config)

                                                
        if config.reference_video is not None:
            return self._generate_with_reference(config, device)
        return self._generate_standard(config, device)

    def _generate_standard(self, config: GenerationConfig, device: torch.device) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
        ''
                                                                 
        v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg = self._get_prompt_embeddings(config, device)

                         
        generator = torch.Generator(device=device).manual_seed(config.seed)

                             
        video_tools = self._create_video_latent_tools(config)
        audio_tools = self._create_audio_latent_tools(config) if config.use_audio_modality else None

                               
        video_clean_state = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)
        audio_clean_state = (
            audio_tools.create_initial_state(device=device, dtype=torch.bfloat16) if audio_tools else None
        )

                                              
        if config.condition_image is not None:
            video_clean_state = self._apply_image_conditioning(
                video_clean_state, config.condition_image, config, device
            )

                   
        noiser = GaussianNoiser(generator=generator)
        video_state = noiser(latent_state=video_clean_state, noise_scale=1.0)
        audio_state = noiser(latent_state=audio_clean_state, noise_scale=1.0) if audio_clean_state else None

                            
        video_state, audio_state, artifacts = self._run_denoising(
            config=config,
            video_state=video_state,
            audio_state=audio_state,
            video_clean_state=video_clean_state,
            audio_clean_state=audio_clean_state,
            v_ctx_pos=v_ctx_pos,
            a_ctx_pos=a_ctx_pos,
            v_ctx_neg=v_ctx_neg,
            a_ctx_neg=a_ctx_neg,
            action_conditioning=self._prepare_action_conditioning(
                config,
                video_state,
                device,
                context_dim=int(v_ctx_pos.shape[-1]),
            ),
            device=device,
        )

                        
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        video_output = self._decode_video(video_state, device, config.tiled_decoding)

        audio_output = None
        if config.generate_audio and audio_state is not None and audio_tools is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            audio_output = self._decode_audio(audio_state, device)

        artifacts.update(
            {
                "prompt": config.prompt,
                "source_prompt": config.source_prompt or config.prompt,
                "negative_prompt": config.negative_prompt,
                "condition_image": config.condition_image.detach().cpu().float() if config.condition_image is not None else None,
                "ground_truth_video": config.ground_truth_video.detach().cpu().float()
                if config.ground_truth_video is not None
                else None,
            }
        )
        return video_output, audio_output, artifacts

    def _generate_with_reference(self, config: GenerationConfig, device: torch.device) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
        ''
                                                                 
        v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg = self._get_prompt_embeddings(config, device)

                         
        generator = torch.Generator(device=device).manual_seed(config.seed)

                                               
        ref_video_preprocessed = self._preprocess_reference_video(config)
        ref_latent, ref_positions = self._encode_video(ref_video_preprocessed, config.frame_rate, device)
        ref_seq_len = ref_latent.shape[1]

                                                                    
                                                                                        
        if config.reference_downscale_factor != 1:
            ref_positions = ref_positions.clone()
            ref_positions[:, 1, ...] *= config.reference_downscale_factor               
            ref_positions[:, 2, ...] *= config.reference_downscale_factor              
                                                   

                                   
        video_tools = self._create_video_latent_tools(config)
        target_clean_state = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)

                                                                    
        if config.condition_image is not None:
            target_clean_state = self._apply_image_conditioning(
                target_clean_state, config.condition_image, config, device
            )

                                                    
                                                                    
        ref_denoise_mask = torch.zeros(1, ref_seq_len, 1, device=device, dtype=torch.float32)
        combined_clean_state = LatentState(
            latent=torch.cat([ref_latent, target_clean_state.latent], dim=1),
            denoise_mask=torch.cat([ref_denoise_mask, target_clean_state.denoise_mask], dim=1),
            positions=torch.cat([ref_positions, target_clean_state.positions], dim=2),
            clean_latent=torch.cat([ref_latent, target_clean_state.clean_latent], dim=1),
        )

                                                                 
        noiser = GaussianNoiser(generator=generator)
        combined_state = noiser(latent_state=combined_clean_state, noise_scale=1.0)

                                      
        audio_tools = self._create_audio_latent_tools(config) if config.use_audio_modality else None
        audio_clean_state = (
            audio_tools.create_initial_state(device=device, dtype=torch.bfloat16) if audio_tools else None
        )
        audio_state = noiser(latent_state=audio_clean_state, noise_scale=1.0) if audio_clean_state else None

                            
        combined_state, audio_state, artifacts = self._run_denoising(
            config=config,
            video_state=combined_state,
            audio_state=audio_state,
            video_clean_state=combined_clean_state,
            audio_clean_state=audio_clean_state,
            v_ctx_pos=v_ctx_pos,
            a_ctx_pos=a_ctx_pos,
            v_ctx_neg=v_ctx_neg,
            a_ctx_neg=a_ctx_neg,
            action_conditioning=self._prepare_action_conditioning(
                config,
                combined_state,
                device,
                context_dim=int(v_ctx_pos.shape[-1]),
            ),
            device=device,
        )

                                           
        target_latent = combined_state.latent[:, ref_seq_len:]
        video_output = self._decode_video_latent(target_latent, config, device)

                                                                      
        if config.include_reference_in_output:
                                                                                  
                                                          
            ref_video_pixels = ref_video_preprocessed[0].cpu()
                                              
            ref_video_pixels = ((ref_video_pixels + 1.0) / 2.0).clamp(0.0, 1.0)
            video_output = self._concatenate_videos_side_by_side(ref_video_pixels, video_output)

                      
        audio_output = None
        if config.generate_audio and audio_state is not None and audio_tools is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            audio_output = self._decode_audio(audio_state, device)

        artifacts.update(
            {
                "prompt": config.prompt,
                "source_prompt": config.source_prompt or config.prompt,
                "negative_prompt": config.negative_prompt,
                "condition_image": config.condition_image.detach().cpu().float() if config.condition_image is not None else None,
                "ground_truth_video": config.ground_truth_video.detach().cpu().float()
                if config.ground_truth_video is not None
                else None,
                "reference_video": config.reference_video.detach().cpu().float()
                if config.reference_video is not None
                else None,
            }
        )
        return video_output, audio_output, artifacts

    def _create_video_latent_tools(self, config: GenerationConfig) -> VideoLatentTools:
        ''
        pixel_shape = VideoPixelShape(
            batch=1,
            frames=config.num_frames,
            height=config.height,
            width=config.width,
            fps=config.frame_rate,
        )
        return VideoLatentTools(
            patchifier=self._video_patchifier,
            target_shape=VideoLatentShape.from_pixel_shape(shape=pixel_shape),
            fps=config.frame_rate,
            scale_factors=VIDEO_SCALE_FACTORS,
            causal_fix=True,
        )

    def _create_audio_latent_tools(self, config: GenerationConfig) -> AudioLatentTools:
        ''
        return AudioLatentTools(
            patchifier=self._audio_patchifier,
            target_shape=AudioLatentShape.from_duration(batch=1, duration=config.num_frames / config.frame_rate),
        )

    @staticmethod
    def _normalize_action_tensor(tensor: Tensor, *, expected_dim: int, name: str) -> Tensor:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[-1] != expected_dim:
            raise ValueError(f"{name} must have shape [raw_frames, {expected_dim}] or [1, raw_frames, {expected_dim}], got {tuple(tensor.shape)}")
        return tensor

    def _normalize_button_tensor(self, tensor: Tensor) -> Tensor:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError(
                "action_buttons must have shape [raw_frames, button_dim] or [1, raw_frames, button_dim], "
                f"got {tuple(tensor.shape)}"
            )
        expected_dim = int(getattr(self._transformer, "action_button_dim", len(CANONICAL_BUTTONS)))
        if expected_dim == 8 and tensor.shape[-1] in (6, 8):
            return canonicalize_button_tensor(tensor)
        if tensor.shape[-1] != expected_dim:
            raise ValueError(
                f"action_buttons width {tensor.shape[-1]} does not match checkpoint button dim {expected_dim}"
            )
        return tensor

    def _prepare_action_conditioning(
        self,
        config: GenerationConfig,
        video_state: LatentState,
        device: torch.device,
        context_dim: int,
    ) -> ActionConditioning:
        if config.action_joystick is None and config.action_buttons is None:
            return None, None, None, None, None, None
        if config.action_joystick is None:
            raise ValueError("Action-conditioned generation requires action_joystick when action_buttons are provided")

        dtype = video_state.latent.dtype
        joystick = self._normalize_action_tensor(config.action_joystick, expected_dim=4, name="action_joystick")
        buttons = self._normalize_button_tensor(config.action_buttons) if config.action_buttons is not None else None
        joystick = joystick.to(device=device, dtype=dtype)
        if buttons is not None:
            buttons = buttons.to(device=device, dtype=dtype)
        if buttons is not None and joystick.shape[:2] != buttons.shape[:2]:
            raise ValueError(f"action_joystick/action_buttons shape mismatch: {tuple(joystick.shape)} vs {tuple(buttons.shape)}")

        latent_frames = config.num_frames // VIDEO_SCALE_FACTORS.time + 1
        latent_height = config.height // VIDEO_SCALE_FACTORS.height
        latent_width = config.width // VIDEO_SCALE_FACTORS.width
        expected_raw_frames = (latent_frames - 1) * VIDEO_SCALE_FACTORS.time + 1
        raw_frames = int(joystick.shape[1])
        if raw_frames < expected_raw_frames:
            raise ValueError(
                f"Action raw frame count {raw_frames} is shorter than expected {expected_raw_frames} "
                f"for num_frames={config.num_frames}"
            )
        if raw_frames > expected_raw_frames:
            joystick = joystick[:, :expected_raw_frames]
            if buttons is not None:
                buttons = buttons[:, :expected_raw_frames]

        latent_joystick = torch.zeros(1, latent_frames, 4 * VIDEO_SCALE_FACTORS.time, device=device, dtype=dtype)
        for latent_index in range(1, latent_frames):
            start = VIDEO_SCALE_FACTORS.time * (latent_index - 1) + 1
            end = VIDEO_SCALE_FACTORS.time * latent_index + 1
            latent_joystick[:, latent_index] = joystick[:, start:end, :].reshape(1, 4 * VIDEO_SCALE_FACTORS.time)
        joystick_tokens = (
            latent_joystick[:, :, None, None, :]
            .expand(1, latent_frames, latent_height, latent_width, 4 * VIDEO_SCALE_FACTORS.time)
            .reshape(1, latent_frames * latent_height * latent_width, 4 * VIDEO_SCALE_FACTORS.time)
            .contiguous()
        )

        target_token_count = latent_frames * latent_height * latent_width
        prefix_tokens = int(video_state.latent.shape[1]) - target_token_count
        if prefix_tokens < 0:
            raise ValueError(
                f"Action token target length {target_token_count} exceeds video latent tokens {video_state.latent.shape[1]}"
            )
        if prefix_tokens > 0:
            joystick_tokens = torch.cat(
                [torch.zeros(1, prefix_tokens, joystick_tokens.shape[-1], device=device, dtype=dtype), joystick_tokens],
                dim=1,
            )

        joystick_mask = torch.ones(1, 1, 1, device=device, dtype=dtype)
        if buttons is None:
            return joystick_tokens, None, None, None, joystick_mask, None

        active = buttons.abs().sum(dim=-1, keepdim=True) > 0
        idle = (~active).to(dtype=dtype)
        expected_state_dim = int(
            getattr(self._transformer, "action_button_state_dim", buttons.shape[-1] + 1)
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
        button_meaning_embeds = self._prepare_button_meaning_embeddings(
            config,
            device=device,
            context_dim=context_dim,
            dtype=dtype,
        )
        button_mask = torch.ones(1, 1, 1, device=device, dtype=dtype)
        return joystick_tokens, button_state, button_values, button_meaning_embeds, joystick_mask, button_mask

    def _prepare_button_meaning_embeddings(
        self,
        config: GenerationConfig,
        *,
        device: torch.device,
        context_dim: int,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if not config.button_meaning_texts:
            return None
        if self._text_encoder is None:
            raise ValueError("button_meaning_texts requires the text encoder to be loaded.")
        if self._embeddings_processor is None or self._embeddings_processor.feature_extractor is None:
            raise ValueError("button_meaning_texts requires the embeddings processor feature extractor.")

        phrases = [
            str(config.button_meaning_texts.get(button, "") or "")
            for button in CANONICAL_BUTTONS
        ]
        unique_phrases = sorted({phrase for phrase in phrases if phrase})
        phrase_cache: dict[str, Tensor] = {}

        self._text_encoder.to(device)
        self._embeddings_processor.to(device)
        with torch.no_grad():
            for phrase in unique_phrases:
                hidden_states, attention_mask = self._text_encoder.encode(phrase)
                processed = self._embeddings_processor.process_hidden_states(hidden_states, attention_mask)
                encoded = processed.video_encoding.squeeze(0)
                encoded_mask = processed.attention_mask.squeeze(0).to(dtype=torch.bool)
                if encoded_mask.any():
                    pooled = encoded[encoded_mask].float().mean(dim=0)
                else:
                    pooled = encoded.float().mean(dim=0)
                phrase_cache[phrase] = pooled.to(device=device, dtype=dtype)
        self._text_encoder.model.to("cpu")

        embeds = []
        for phrase in phrases:
            if phrase:
                embeds.append(phrase_cache[phrase].to(device=device, dtype=dtype))
            else:
                embeds.append(torch.zeros(context_dim, device=device, dtype=dtype))
        return torch.stack(embeds, dim=0).unsqueeze(0).contiguous()

    def _apply_image_conditioning(
        self, video_state: LatentState, image: Tensor, config: GenerationConfig, device: torch.device
    ) -> LatentState:
        ''
                          
        encoded_image = self._encode_conditioning_image(image, config.height, config.width, device)

                                                   
        patchified_image = self._video_patchifier.patchify(encoded_image)                                    
        num_image_tokens = patchified_image.shape[1]

                                                     
        new_latent = video_state.latent.clone()
        new_latent[:, :num_image_tokens] = patchified_image.to(new_latent.dtype)

                                                                   
        new_clean_latent = video_state.clean_latent.clone()
        new_clean_latent[:, :num_image_tokens] = patchified_image.to(new_clean_latent.dtype)

                                                                           
        new_denoise_mask = video_state.denoise_mask.clone()
        new_denoise_mask[:, :num_image_tokens] = 0.0

        return LatentState(
            latent=new_latent,
            denoise_mask=new_denoise_mask,
            positions=video_state.positions,
            clean_latent=new_clean_latent,
        )

    @staticmethod
    def _preprocess_reference_video(config: GenerationConfig) -> Tensor:
        ''
        ref_video = config.reference_video                          
        scale_factor = config.reference_downscale_factor

                                                                           
        target_height = config.height // scale_factor
        target_width = config.width // scale_factor

                                    
        if target_height % 32 != 0 or target_width % 32 != 0:
            raise ValueError(
                f"Scaled reference dimensions ({target_height}x{target_width}) must be divisible by 32. "
                f"Original: {config.height}x{config.width}, scale_factor: {scale_factor}"
            )

        current_height, current_width = ref_video.shape[2:]

                                                                   
        if current_height != target_height or current_width != target_width:
            aspect_ratio = current_width / current_height
            target_aspect_ratio = target_width / target_height

            if aspect_ratio > target_aspect_ratio:
                resize_height, resize_width = target_height, int(target_height * aspect_ratio)
            else:
                resize_height, resize_width = int(target_width / aspect_ratio), target_width

            ref_video = torch.nn.functional.interpolate(
                ref_video, size=(resize_height, resize_width), mode="bilinear", align_corners=False
            )

                         
            h_start = (resize_height - target_height) // 2
            w_start = (resize_width - target_width) // 2
            ref_video = ref_video[:, :, h_start : h_start + target_height, w_start : w_start + target_width]

                                                                            
        ref_video = rearrange(ref_video, "f c h w -> 1 c f h w")
        valid_frames = (ref_video.shape[2] - 1) // 8 * 8 + 1
        ref_video = ref_video[:, :, :valid_frames]

                                  
        return ref_video * 2.0 - 1.0

    def _encode_video(self, video: Tensor, fps: float, device: torch.device) -> tuple[Tensor, Tensor]:
        ''
        video = video.to(device=device, dtype=torch.float32)

                         
        self._vae_encoder.to(device)
        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
            latents = self._vae_encoder(video)
        self._vae_encoder.to("cpu")

        latents = latents.to(torch.bfloat16)
        patchified = self._video_patchifier.patchify(latents)

                           
        latent_shape = VideoLatentShape(
            batch=1,
            channels=latents.shape[1],
            frames=latents.shape[2],
            height=latents.shape[3],
            width=latents.shape[4],
        )
        latent_coords = self._video_patchifier.get_patch_grid_bounds(output_shape=latent_shape, device=device)
        positions = get_pixel_coords(latent_coords, scale_factors=VIDEO_SCALE_FACTORS, causal_fix=True)
        positions = positions.to(torch.bfloat16)
        positions[:, 0, ...] = positions[:, 0, ...] / fps

        return patchified, positions

    def _run_denoising(
        self,
        config: GenerationConfig,
        video_state: LatentState,
        audio_state: LatentState | None,
        video_clean_state: LatentState,
        audio_clean_state: LatentState | None,
        v_ctx_pos: Tensor,
        a_ctx_pos: Tensor,
        v_ctx_neg: Tensor | None,
        a_ctx_neg: Tensor | None,
        action_conditioning: ActionConditioning,
        device: torch.device,
    ) -> tuple[LatentState, LatentState | None, dict[str, Any]]:
        ''
        scheduler = LTX2Scheduler()
        if config.scheduler_timestep_shift is None:
            sigmas = scheduler.execute(steps=config.num_inference_steps).to(device).float()
        else:
            timestep_shift = float(config.scheduler_timestep_shift)
            sigmas = scheduler.execute(
                steps=config.num_inference_steps,
                max_shift=timestep_shift,
                base_shift=timestep_shift,
            ).to(device).float()
        stepper = EulerDiffusionStep()
        capture_steps = self._resolve_capture_steps(config, total_steps=len(sigmas) - 1)
        captured_states: list[tuple[int, float, Tensor]] = []
        video_guider = MultiModalGuider(params=config.video_guider_params, negative_context=v_ctx_neg)
        audio_guider = (
            MultiModalGuider(params=config.audio_guider_params, negative_context=a_ctx_neg)
            if audio_state is not None
            else _POSITIVE_ONLY_GUIDER
        )

                                                                                           
        self._transformer.to(device)
        x0_model = X0Model(self._transformer)
        last_denoised_video: Tensor | None = None
        last_denoised_audio: Tensor | None = None

        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
            for step_idx, sigma in enumerate(sigmas[:-1]):
                denoised_video, denoised_audio = self._guided_denoise_step(
                    config=config,
                    x0_model=x0_model,
                    video_state=video_state,
                    audio_state=audio_state,
                    sigma=sigma,
                    step_index=step_idx,
                    video_guider=video_guider,
                    audio_guider=audio_guider,
                    v_ctx_pos=v_ctx_pos,
                    a_ctx_pos=a_ctx_pos,
                    last_denoised_video=last_denoised_video,
                    last_denoised_audio=last_denoised_audio,
                    action_conditioning=action_conditioning,
                )
                last_denoised_video = denoised_video
                last_denoised_audio = denoised_audio

                                                                         
                denoised_video = denoised_video * video_state.denoise_mask + video_clean_state.latent.float() * (
                    1 - video_state.denoise_mask
                )
                if audio_state is not None and audio_clean_state is not None and denoised_audio is not None:
                    denoised_audio = denoised_audio * audio_state.denoise_mask + audio_clean_state.latent.float() * (
                        1 - audio_state.denoise_mask
                    )

                            
                video_state = replace(
                    video_state,
                    latent=stepper.step(
                        sample=video_state.latent,
                        denoised_sample=denoised_video,
                        sigmas=sigmas,
                        step_index=step_idx,
                    ),
                )
                if audio_state is not None and denoised_audio is not None:
                    audio_state = replace(
                        audio_state,
                        latent=stepper.step(
                            sample=audio_state.latent,
                            denoised_sample=denoised_audio,
                            sigmas=sigmas,
                            step_index=step_idx,
                        ),
                    )

                if step_idx in capture_steps:
                    captured_states.append((step_idx, float(sigma.item()), denoised_video.detach().cpu().float()))

                                 
                if self._sampling_context is not None:
                    self._sampling_context.advance_step()

        artifacts: dict[str, Any] = {
            "capture_steps": [item[0] for item in captured_states],
            "capture_sigmas": [item[1] for item in captured_states],
            "capture_videos": [
                self._decode_video_latent(latent.to(device=device, dtype=torch.bfloat16), config, device)
                for _step_idx, _sigma, latent in captured_states
            ],
        }
        return video_state, audio_state, artifacts

    @staticmethod
    def _resolve_capture_steps(config: GenerationConfig, total_steps: int) -> set[int]:
        if not config.capture_intermediates or total_steps <= 0:
            return set()
        count = max(1, min(config.intermediate_capture_count, total_steps))
        if count == 1:
            return {total_steps - 1}
        positions = torch.linspace(0, total_steps - 1, steps=count).round().to(torch.int64).tolist()
        return {int(position) for position in positions}

    def _guided_denoise_step(
        self,
        *,
        config: GenerationConfig,
        x0_model: X0Model,
        video_state: LatentState,
        audio_state: LatentState | None,
        sigma: Tensor,
        step_index: int,
        video_guider: MultiModalGuider,
        audio_guider: MultiModalGuider,
        v_ctx_pos: Tensor,
        a_ctx_pos: Tensor,
        last_denoised_video: Tensor | None,
        last_denoised_audio: Tensor | None,
        action_conditioning: ActionConditioning,
    ) -> tuple[Tensor, Tensor | None]:
        v_skip = video_guider.should_skip_step(step_index)
        a_skip = audio_state is None or audio_guider.should_skip_step(step_index)

        if v_skip and a_skip:
            if last_denoised_video is None:
                raise ValueError("Video guidance skipped before an initial denoised prediction was produced")
            return last_denoised_video, last_denoised_audio

        passes: list[tuple[str, Tensor, Tensor | None, PerturbationConfig]] = [
            ("cond", v_ctx_pos, a_ctx_pos if audio_state is not None else None, PerturbationConfig.empty())
        ]
        use_action_guidance = (
            config.action_guidance_scale > 0.0
            and action_conditioning[1] is not None
            and action_conditioning[2] is not None
            and action_conditioning[5] is not None
        )
        use_joystick_guidance = (
            config.joystick_guidance_scale > 0.0
            and action_conditioning[0] is not None
            and action_conditioning[4] is not None
        )
        action_conditioning_per_pass: list[ActionConditioning] = [action_conditioning]

        if video_guider.do_unconditional_generation() or (audio_state is not None and audio_guider.do_unconditional_generation()):
            if video_guider.do_unconditional_generation() and video_guider.negative_context is None:
                raise ValueError("Video CFG requires negative prompt embeddings")
            if audio_state is not None and audio_guider.do_unconditional_generation() and audio_guider.negative_context is None:
                raise ValueError("Audio CFG requires negative prompt embeddings")
            passes.append(
                (
                    "uncond",
                    video_guider.negative_context if video_guider.negative_context is not None else v_ctx_pos,
                    (
                        audio_guider.negative_context
                        if audio_state is not None and audio_guider.negative_context is not None
                        else (a_ctx_pos if audio_state is not None else None)
                    ),
                    PerturbationConfig.empty(),
                )
            )
            action_conditioning_per_pass.append(action_conditioning)

        stg_perturbations: list[Perturbation] = []
        if video_guider.do_perturbed_generation():
            stg_perturbations.append(
                Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=video_guider.params.stg_blocks)
            )
        if audio_state is not None and audio_guider.do_perturbed_generation():
            stg_perturbations.append(
                Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=audio_guider.params.stg_blocks)
            )
        if stg_perturbations:
            passes.append(
                (
                    "ptb",
                    v_ctx_pos,
                    a_ctx_pos if audio_state is not None else None,
                    PerturbationConfig(stg_perturbations),
                )
            )
            action_conditioning_per_pass.append(action_conditioning)

        if video_guider.do_isolated_modality_generation() or (
            audio_state is not None and audio_guider.do_isolated_modality_generation()
        ):
            passes.append(
                (
                    "mod",
                    v_ctx_pos,
                    a_ctx_pos if audio_state is not None else None,
                    PerturbationConfig(
                        [
                            Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                            Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
                        ]
                    ),
                )
            )
            action_conditioning_per_pass.append(action_conditioning)

        if use_action_guidance:
            passes.append(
                (
                    "null_button",
                    v_ctx_pos,
                    a_ctx_pos if audio_state is not None else None,
                    PerturbationConfig.empty(),
                )
            )
            action_conditioning_per_pass.append(_button_null_conditioning(action_conditioning))

        if use_joystick_guidance:
            passes.append(
                (
                    "null_joystick",
                    v_ctx_pos,
                    a_ctx_pos if audio_state is not None else None,
                    PerturbationConfig.empty(),
                )
            )
            action_conditioning_per_pass.append(_joystick_null_conditioning(action_conditioning))

        num_passes = len(passes)
        perturbations = BatchedPerturbationConfig(perturbations=[perturbation for _, _, _, perturbation in passes])

        batched_video = _modality_from_latent_state(
            _repeat_state(video_state, num_passes),
            context=torch.cat([context for _, context, _, _ in passes], dim=0),
            sigma=sigma.expand(video_state.latent.shape[0] * num_passes),
            enabled=not v_skip,
        )

        batched_audio = None
        if audio_state is not None:
            batched_audio = _modality_from_latent_state(
                _repeat_state(audio_state, num_passes),
                context=torch.cat(
                    [context for _, _, context, _ in passes if context is not None],
                    dim=0,
                ),
                sigma=sigma.expand(audio_state.latent.shape[0] * num_passes),
                enabled=not a_skip,
            )

        all_video, all_audio = x0_model(
            video=batched_video,
            audio=batched_audio,
            perturbations=perturbations,
            action_joystick_tokens=torch.cat(
                [item[0] for item in action_conditioning_per_pass if item[0] is not None],
                dim=0,
            ) if action_conditioning[0] is not None else None,
            action_button_state=torch.cat(
                [item[1] for item in action_conditioning_per_pass if item[1] is not None],
                dim=0,
            ) if action_conditioning[1] is not None else None,
            action_button_values=torch.cat(
                [item[2] for item in action_conditioning_per_pass if item[2] is not None],
                dim=0,
            ) if action_conditioning[2] is not None else None,
            action_button_meaning_embeds=torch.cat(
                [item[3] for item in action_conditioning_per_pass if item[3] is not None],
                dim=0,
            ) if action_conditioning[3] is not None else None,
            action_joystick_condition_mask=torch.cat(
                [item[4] for item in action_conditioning_per_pass if item[4] is not None],
                dim=0,
            ) if action_conditioning[4] is not None else None,
            action_button_condition_mask=torch.cat(
                [item[5] for item in action_conditioning_per_pass if item[5] is not None],
                dim=0,
            ) if action_conditioning[5] is not None else None,
            action_button_meaning_fusion=config.action_button_meaning_fusion,
            action_button_temporal_alignment=config.action_button_temporal_alignment,
            action_button_prefix_tokens=(
                int(video_state.latent.shape[1])
                - (
                    (config.num_frames // VIDEO_SCALE_FACTORS.time + 1)
                    * (config.height // VIDEO_SCALE_FACTORS.height)
                    * (config.width // VIDEO_SCALE_FACTORS.width)
                )
            ),
            action_button_frame_rate=float(config.frame_rate),
        )

        video_splits = list(all_video.chunk(num_passes))
        audio_splits = list(all_audio.chunk(num_passes)) if all_audio is not None else [None] * num_passes
        outputs = dict(
            zip(
                [name for name, _, _, _ in passes],
                zip(video_splits, audio_splits, strict=True),
                strict=True,
            )
        )

        cond_video, cond_audio = outputs["cond"]
        uncond_video, uncond_audio = outputs.get("uncond", (0.0, 0.0))
        ptb_video, ptb_audio = outputs.get("ptb", (0.0, 0.0))
        mod_video, mod_audio = outputs.get("mod", (0.0, 0.0))
        null_button_video, _null_button_audio = outputs.get("null_button", (0.0, 0.0))
        null_joystick_video, _null_joystick_audio = outputs.get("null_joystick", (0.0, 0.0))

        denoised_video = last_denoised_video if v_skip else video_guider.calculate(cond_video, uncond_video, ptb_video, mod_video)
        if not v_skip and use_action_guidance:
            denoised_video = denoised_video + config.action_guidance_scale * (cond_video - null_button_video)
        if not v_skip and use_joystick_guidance:
            denoised_video = denoised_video + config.joystick_guidance_scale * (cond_video - null_joystick_video)
        denoised_audio = (
            last_denoised_audio
            if a_skip
            else audio_guider.calculate(cond_audio, uncond_audio, ptb_audio, mod_audio)
        )
        return denoised_video, denoised_audio

    def _decode_video_latent(self, latent: Tensor, config: GenerationConfig, device: torch.device) -> Tensor:
        ''
                    
        latent_frames = config.num_frames // VIDEO_SCALE_FACTORS.time + 1
        latent_height = config.height // VIDEO_SCALE_FACTORS.height
        latent_width = config.width // VIDEO_SCALE_FACTORS.width

        unpatchified = self._video_patchifier.unpatchify(
            latent,
            output_shape=VideoLatentShape(
                height=latent_height,
                width=latent_width,
                frames=latent_frames,
                batch=1,
                channels=128,
            ),
        )

                                                           
        self._vae_decoder.to(device)
        unpatchified = unpatchified.to(dtype=torch.bfloat16)
        tiled_config = config.tiled_decoding

        if tiled_config is not None and tiled_config.enabled:
                                                 
            tiling_config = TilingConfig(
                spatial_config=SpatialTilingConfig(
                    tile_size_in_pixels=tiled_config.tile_size_pixels,
                    tile_overlap_in_pixels=tiled_config.tile_overlap_pixels,
                ),
                temporal_config=TemporalTilingConfig(
                    tile_size_in_frames=tiled_config.tile_size_frames,
                    tile_overlap_in_frames=tiled_config.tile_overlap_frames,
                ),
            )
            chunks = []
            for video_chunk in self._vae_decoder.tiled_decode(
                unpatchified,
                tiling_config=tiling_config,
            ):
                chunks.append(video_chunk)
            decoded_video = torch.cat(chunks, dim=2)
        else:
                                    
            decoded_video = self._vae_decoder(unpatchified)

        decoded_video = ((decoded_video + 1.0) / 2.0).clamp(0.0, 1.0)
        self._vae_decoder.to("cpu")

        return decoded_video[0].float().cpu()

    def _validate_config(self, config: GenerationConfig) -> None:
        ''
        if config.height % 32 != 0 or config.width % 32 != 0:
            raise ValueError(f"height and width must be divisible by 32, got {config.height}x{config.width}")
        if config.num_frames % 8 != 1:
            raise ValueError(f"num_frames must satisfy num_frames % 8 == 1, got {config.num_frames}")
        if config.generate_audio and not config.use_audio_modality:
            raise ValueError("generate_audio requires use_audio_modality to be enabled")
        if config.generate_audio and (self._audio_decoder is None or self._vocoder is None):
            raise ValueError("Audio generation requires audio_decoder and vocoder")
        if config.condition_image is not None and self._vae_encoder is None:
            raise ValueError("Image conditioning requires vae_encoder")
        if config.reference_video is not None and self._vae_encoder is None:
            raise ValueError("Reference video conditioning requires vae_encoder")
        if config.action_joystick is None and config.action_buttons is not None:
            raise ValueError("action_buttons require action_joystick")
        if config.action_guidance_scale < 0.0:
            raise ValueError(f"action_guidance_scale must be >= 0, got {config.action_guidance_scale}")
        if config.action_guidance_scale > 0.0 and (
            config.action_joystick is None or config.action_buttons is None
        ):
            raise ValueError("action_guidance_scale requires action_joystick and action_buttons")
        if config.joystick_guidance_scale < 0.0:
            raise ValueError(f"joystick_guidance_scale must be >= 0, got {config.joystick_guidance_scale}")
        if config.joystick_guidance_scale > 0.0 and config.action_joystick is None:
            raise ValueError("joystick_guidance_scale requires action_joystick")
        if config.action_button_temporal_alignment not in {"global", "vae_window", "causal_prefix_rope"}:
            raise ValueError(
                "action_button_temporal_alignment must be 'global', 'vae_window', or 'causal_prefix_rope', "
                f"got {config.action_button_temporal_alignment!r}"
            )

                                          
        if config.cached_embeddings is None and self._text_encoder is None:
            raise ValueError("Either text_encoder or config.cached_embeddings must be provided")
        if config.cached_embeddings is None and self._embeddings_processor is None:
            raise ValueError("embeddings_processor is required when encoding prompts on-the-fly")

    def _get_prompt_embeddings(
        self, config: GenerationConfig, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        ''
        if config.cached_embeddings is not None:
                                                     
            cached = config.cached_embeddings
            v_ctx_pos = cached.video_context_positive.to(device)
            a_ctx_pos = cached.audio_context_positive.to(device)
            v_ctx_neg = cached.video_context_negative.to(device) if cached.video_context_negative is not None else None
            a_ctx_neg = cached.audio_context_negative.to(device) if cached.audio_context_negative is not None else None
            return v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg

                                          
        return self._encode_prompts(config, device)

    def _encode_prompts(
        self, config: GenerationConfig, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        ''
        self._text_encoder.to(device)
        self._embeddings_processor.to(device)

        pos_hs, pos_mask = self._text_encoder.encode(config.prompt)
        pos_out = self._embeddings_processor.process_hidden_states(pos_hs, pos_mask)
        v_ctx_pos, a_ctx_pos = pos_out.video_encoding, pos_out.audio_encoding

        v_ctx_neg, a_ctx_neg = None, None
        needs_negative = (
            config.video_guider_params.cfg_scale != 1.0
            or (config.use_audio_modality and config.audio_guider_params.cfg_scale != 1.0)
        )
        if needs_negative:
            neg_hs, neg_mask = self._text_encoder.encode(config.negative_prompt)
            neg_out = self._embeddings_processor.process_hidden_states(neg_hs, neg_mask)
            v_ctx_neg, a_ctx_neg = neg_out.video_encoding, neg_out.audio_encoding

                                          
        self._text_encoder.model.to("cpu")

        return v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg

    def _decode_video(
        self, video_state: LatentState, device: torch.device, tiled_config: TiledDecodingConfig | None = None
    ) -> Tensor:
        ''
        self._vae_decoder.to(device)
                                                            
        latent = video_state.latent.to(dtype=torch.bfloat16)

        if tiled_config is not None and tiled_config.enabled:
                                                 
            tiling_config = TilingConfig(
                spatial_config=SpatialTilingConfig(
                    tile_size_in_pixels=tiled_config.tile_size_pixels,
                    tile_overlap_in_pixels=tiled_config.tile_overlap_pixels,
                ),
                temporal_config=TemporalTilingConfig(
                    tile_size_in_frames=tiled_config.tile_size_frames,
                    tile_overlap_in_frames=tiled_config.tile_overlap_frames,
                ),
            )
            chunks = []
            for video_chunk in self._vae_decoder.tiled_decode(
                latent,
                tiling_config=tiling_config,
            ):
                chunks.append(video_chunk)
            decoded_video = torch.cat(chunks, dim=2)
        else:
                                    
            decoded_video = self._vae_decoder(latent)

        decoded_video = ((decoded_video + 1.0) / 2.0).clamp(0.0, 1.0)
        self._vae_decoder.to("cpu")
        return decoded_video[0].float().cpu()

    def _decode_audio(self, audio_state: LatentState, device: torch.device) -> Tensor:
        ''
        self._audio_decoder.to(device)
        first_param = next(self._audio_decoder.parameters(), None)
        decoder_dtype = first_param.dtype if first_param is not None else audio_state.latent.dtype
        latent = audio_state.latent.to(dtype=decoder_dtype, device=device)
        decoded_audio = self._audio_decoder(latent)
        self._audio_decoder.to("cpu")

        self._vocoder.to(device)
        audio_waveform = self._vocoder(decoded_audio)
        self._vocoder.to("cpu")

        return audio_waveform.squeeze(0).float().cpu()

    @staticmethod
    def _concatenate_videos_side_by_side(left_video: Tensor, right_video: Tensor) -> Tensor:
        ''
        left_height, left_width = left_video.shape[2], left_video.shape[3]
        right_height = right_video.shape[2]

                                                                   
        if left_height != right_height:
                                                                 
            scale = right_height / left_height
            new_width = int(left_width * scale)
                                                                    
                                                                      
            c, f, h, w = left_video.shape
            left_video = left_video.reshape(c * f, 1, h, w)
            left_video = torch.nn.functional.interpolate(
                left_video, size=(right_height, new_width), mode="bilinear", align_corners=False
            )
            left_video = left_video.reshape(c, f, right_height, new_width)

        left_frames = left_video.shape[1]
        right_frames = right_video.shape[1]

                                                   
        if left_frames < right_frames:
            padding = left_video[:, -1:, :, :].expand(-1, right_frames - left_frames, -1, -1)
            left_video = torch.cat([left_video, padding], dim=1)
        elif right_frames < left_frames:
            padding = right_video[:, -1:, :, :].expand(-1, left_frames - right_frames, -1, -1)
            right_video = torch.cat([right_video, padding], dim=1)

                                           
        return torch.cat([left_video, right_video], dim=3)

    def _encode_conditioning_image(
        self,
        image: Tensor,
        target_height: int,
        target_width: int,
        device: torch.device,
    ) -> Tensor:
        ''
                                                      
        current_height, current_width = image.shape[1:]

                                                                          
        if current_height != target_height or current_width != target_width:
            aspect_ratio = current_width / current_height
            target_aspect_ratio = target_width / target_height

            if aspect_ratio > target_aspect_ratio:
                                                                                 
                resize_height = target_height
                resize_width = int(target_height * aspect_ratio)
            else:
                                                                                  
                resize_height = int(target_width / aspect_ratio)
                resize_width = target_width

            image = rearrange(image, "c h w -> 1 c h w")
            image = torch.nn.functional.interpolate(
                image, size=(resize_height, resize_width), mode="bilinear", align_corners=False
            )

                                              
            h_start = (resize_height - target_height) // 2
            w_start = (resize_width - target_width) // 2
            image = image[:, :, h_start : h_start + target_height, w_start : w_start + target_width]
        else:
            image = rearrange(image, "c h w -> 1 c h w")

                                                    
        image = rearrange(image, "b c h w -> b c 1 h w")
        image = (image * 2.0 - 1.0).to(device=device, dtype=torch.float32)

                
        self._vae_encoder.to(device)
        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
            encoded = self._vae_encoder(image)
        self._vae_encoder.to("cpu")

        return encoded
