import gc
import logging

import torch

from ltx_core.components.noisers import Noiser
from ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
)
from ltx_core.model.transformer import Modality
from ltx_core.model.video_vae import TilingConfig, VideoEncoder
from ltx_core.text_encoders.gemma import GemmaTextEncoder
from ltx_core.tools import LatentTools
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import (
    decode_audio_from_file,
    decode_image,
    decode_video_from_file,
    get_videostream_fps,
    load_image_and_preprocess,
    resize_aspect_ratio_preserving,
    video_preprocess,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def cleanup_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _conform_latent_length(latent: torch.Tensor, expected_frames_count: int) -> torch.Tensor:
    actual_frames = latent.shape[2]
    if actual_frames > expected_frames_count:
        latent = latent[:, :, :expected_frames_count]
    elif actual_frames < expected_frames_count:
        shape_as_list = list(latent.shape)
        shape_as_list[2] = expected_frames_count - actual_frames
        pad = torch.zeros(
            shape_as_list,
            device=latent.device,
            dtype=latent.dtype,
        )
        latent = torch.cat([latent, pad], dim=2)
    return latent


def video_latent_from_file(
    video_encoder: VideoEncoder,
    file_path: str,
    output_shape: VideoPixelShape,
    device: torch.device,
    dtype: torch.dtype,
    start_time: float = 0.0,
    max_duration: float | None = None,
    tiling_config: TilingConfig | None = None,
) -> torch.Tensor | None:
    ''
    fps = get_videostream_fps(file_path)
    if fps != output_shape.fps:
        raise ValueError(f"Input video FPS {fps} does not match output FPS {output_shape.fps}, not supported")
    max_duration = max_duration or output_shape.frames / fps
    frame_gen = decode_video_from_file(path=file_path, device=device, start_time=start_time, max_duration=max_duration)
    frames = video_preprocess(frame_gen, output_shape.height, output_shape.width, dtype, device)
    latents = video_encoder.tiled_encode(frames, tiling_config or TilingConfig.default())
    required_latent_frames = VideoLatentShape.from_pixel_shape(output_shape).frames
    return _conform_latent_length(latents, required_latent_frames)


def audio_latent_from_file(
    audio_encoder: torch.nn.Module,
    file_path: str,
    output_shape: VideoPixelShape,
    device: torch.device,
    dtype: torch.dtype,
    start_time: float = 0.0,
    max_duration: float | None = None,
) -> torch.Tensor | None:
    ''
    max_duration = max_duration or output_shape.frames / output_shape.fps
    audio_in = decode_audio_from_file(file_path, device, start_time, max_duration)
    if audio_in is None:
        return None
    from ltx_core.model.audio_vae import encode_audio

    latents = encode_audio(audio_in, audio_encoder, None).to(device, dtype)
    required_latent_frames = AudioLatentShape.from_video_pixel_shape(output_shape).frames
    return _conform_latent_length(latents, required_latent_frames)


def combined_image_conditionings(
    images: list[ImageConditioningInput],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
) -> list[ConditioningItem]:
    ''
    conditionings = []
    for img in images:
        image = load_image_and_preprocess(
            image_path=img.path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            crf=img.crf,
        )
        encoded_image = video_encoder(image)
        if img.frame_idx == 0:
            conditioning = VideoConditionByLatentIndex(
                latent=encoded_image,
                strength=img.strength,
                latent_idx=0,
            )
        else:
            conditioning = VideoConditionByKeyframeIndex(
                keyframes=encoded_image,
                strength=img.strength,
                frame_idx=img.frame_idx,
            )
        conditionings.append(conditioning)
    return conditionings


def image_conditionings_by_replacing_latent(
    images: list[ImageConditioningInput],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
) -> list[ConditioningItem]:
    conditionings = []
    for img in images:
        image = load_image_and_preprocess(
            image_path=img.path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            crf=img.crf,
        )
        encoded_image = video_encoder(image)
        conditionings.append(
            VideoConditionByLatentIndex(
                latent=encoded_image,
                strength=img.strength,
                latent_idx=img.frame_idx,
            )
        )

    return conditionings


def image_conditionings_by_adding_guiding_latent(
    images: list[ImageConditioningInput],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
) -> list[ConditioningItem]:
    conditionings = []
    for img in images:
        image = load_image_and_preprocess(
            image_path=img.path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            crf=img.crf,
        )
        encoded_image = video_encoder(image)
        conditionings.append(
            VideoConditionByKeyframeIndex(keyframes=encoded_image, frame_idx=img.frame_idx, strength=img.strength)
        )
    return conditionings


def create_noised_state(
    tools: LatentTools,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> LatentState:
    ''
    state = tools.create_initial_state(device, dtype, initial_latent)
    state = state_with_conditionings(state, conditionings, tools)
    state = noiser(state, noise_scale)

    return state


def state_with_conditionings(
    latent_state: LatentState, conditioning_items: list[ConditioningItem], latent_tools: LatentTools
) -> LatentState:
    ''
    for conditioning in conditioning_items:
        latent_state = conditioning.apply_to(latent_state=latent_state, latent_tools=latent_tools)

    return latent_state


def post_process_latent(denoised: torch.Tensor, denoise_mask: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    ''
    return (denoised * denoise_mask + clean.float() * (1 - denoise_mask)).to(denoised.dtype)


def modality_from_latent_state(
    state: LatentState,
    context: torch.Tensor,
    sigma: torch.Tensor,
    enabled: bool = True,
) -> Modality:
    ''
    return Modality(
        enabled=enabled,
        latent=state.latent,
        sigma=sigma,
        timesteps=timesteps_from_mask(state.denoise_mask, sigma),
        positions=state.positions,
        context=context,
        context_mask=None,
        attention_mask=state.attention_mask,
    )


def timesteps_from_mask(denoise_mask: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    ''
    if isinstance(sigma, torch.Tensor) and sigma.dim() == 1:
        sigma = sigma.view(-1, *([1] * (denoise_mask.dim() - 1)))
    return denoise_mask * sigma


_UNICODE_REPLACEMENTS = str.maketrans("\u2018\u2019\u201c\u201d\u2014\u2013\u00a0\u2032\u2212", "''\"\"-- '-")


def clean_response(text: str) -> str:
    ''
    text = text.translate(_UNICODE_REPLACEMENTS)

                                          
    for i, char in enumerate(text):
        if char.isalpha():
            return text[i:]
    return text


def generate_enhanced_prompt(
    text_encoder: GemmaTextEncoder,
    prompt: str,
    image_path: str | None = None,
    image_long_side: int = 896,
    seed: int = 42,
) -> str:
    ''
    image = None
    if image_path:
        image = decode_image(image_path=image_path)
        image = torch.tensor(image)
        image = resize_aspect_ratio_preserving(image, image_long_side).to(torch.uint8)
        prompt = text_encoder.enhance_i2v(prompt, image, seed=seed)
    else:
        prompt = text_encoder.enhance_t2v(prompt, seed=seed)
    logging.info(f"Enhanced prompt: {prompt}")
    return clean_response(prompt)


def assert_resolution(height: int, width: int, is_two_stage: bool) -> None:
    ''
    divisor = 64 if is_two_stage else 32
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"Resolution ({height}x{width}) is not divisible by {divisor}. "
            f"For {'two-stage' if is_two_stage else 'one-stage'} pipelines, "
            f"height and width must be multiples of {divisor}."
        )
