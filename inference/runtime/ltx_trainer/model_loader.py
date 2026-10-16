                     

''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ltx_trainer import logger

                                     
Device = str | torch.device

                                               
if TYPE_CHECKING:
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.model.audio_vae import AudioDecoder, AudioEncoder, Vocoder
    from ltx_core.model.transformer import LTXModel
    from ltx_core.model.video_vae import VideoDecoder, VideoEncoder
    from ltx_core.text_encoders.gemma import GemmaTextEncoder
    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor


def _to_torch_device(device: Device) -> torch.device:
    ''
    return torch.device(device) if isinstance(device, str) else device


                                                                               
                              
                                                                               


def load_transformer(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "LTXModel":
    ''
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXModelConfigurator,
    )

    transformer = SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    ).build(device=_to_torch_device(device), dtype=dtype)
    return transformer


def load_video_vae_encoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "VideoEncoder":
    ''
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.video_vae import VAE_ENCODER_COMFY_KEYS_FILTER, VideoEncoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_video_vae_decoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "VideoDecoder":
    ''
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.video_vae import VAE_DECODER_COMFY_KEYS_FILTER, VideoDecoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VideoDecoderConfigurator,
        model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_audio_vae_encoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "AudioEncoder":
    ''
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.audio_vae import AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER, AudioEncoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AudioEncoderConfigurator,
        model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_audio_vae_decoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "AudioDecoder":
    ''
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.audio_vae import AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, AudioDecoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=AudioDecoderConfigurator,
        model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_vocoder(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "Vocoder":
    ''
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.audio_vae import VOCODER_COMFY_KEYS_FILTER, VocoderConfigurator

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VocoderConfigurator,
        model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
    ).build(device=_to_torch_device(device), dtype=dtype)


def load_text_encoder(
    gemma_model_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
) -> "GemmaTextEncoder":
    ''
    if not Path(gemma_model_path).is_dir():
        raise ValueError(f"Gemma model path is not a directory: {gemma_model_path}")

                                         
    if load_in_8bit:
        from ltx_trainer.gemma_8bit import load_8bit_gemma

        return load_8bit_gemma(gemma_model_path, dtype, device)

                           
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma import (
        GEMMA_LLM_KEY_OPS,
        GEMMA_MODEL_OPS,
        GemmaTextEncoderConfigurator,
        module_ops_from_gemma_root,
    )
    from ltx_core.utils import find_matching_file

    torch_device = _to_torch_device(device)

    gemma_model_folder = find_matching_file(str(gemma_model_path), "model*.safetensors").parent
    gemma_weight_paths = [str(p) for p in gemma_model_folder.rglob("*.safetensors")]

    text_encoder = SingleGPUModelBuilder(
        model_path=tuple(gemma_weight_paths),
        model_class_configurator=GemmaTextEncoderConfigurator,
        model_sd_ops=GEMMA_LLM_KEY_OPS,
        module_ops=(GEMMA_MODEL_OPS, *module_ops_from_gemma_root(str(gemma_model_path))),
    ).build(device=torch_device, dtype=dtype)

    return text_encoder


def load_embeddings_processor(
    checkpoint_path: str | Path,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> "EmbeddingsProcessor":
    ''
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma import (
        EMBEDDINGS_PROCESSOR_KEY_OPS,
        EmbeddingsProcessorConfigurator,
    )

    torch_device = _to_torch_device(device)

    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=EmbeddingsProcessorConfigurator,
        model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
    ).build(device=torch_device, dtype=dtype)


                                                                               
                           
                                                                               


@dataclass
class LtxModelComponents:
    ''

    transformer: "LTXModel"
    video_vae_encoder: "VideoEncoder | None" = None
    video_vae_decoder: "VideoDecoder | None" = None
    audio_vae_encoder: "AudioEncoder | None" = None
    audio_vae_decoder: "AudioDecoder | None" = None
    vocoder: "Vocoder | None" = None
    text_encoder: "GemmaTextEncoder | None" = None
    scheduler: "LTX2Scheduler | None" = None


def load_model(
    checkpoint_path: str | Path,
    text_encoder_path: str | Path | None = None,
    device: Device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    with_video_vae_encoder: bool = False,
    with_video_vae_decoder: bool = True,
    with_audio_vae_encoder: bool = False,
    with_audio_vae_decoder: bool = True,
    with_vocoder: bool = True,
    with_text_encoder: bool = True,
) -> LtxModelComponents:
    ''
    from ltx_core.components.schedulers import LTX2Scheduler

    checkpoint_path = Path(checkpoint_path)

                                
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading LTX-2 model from {checkpoint_path}")

    torch_device = _to_torch_device(device)

                      
    logger.debug("Loading transformer...")
    transformer = load_transformer(
        checkpoint_path,
        torch_device,
        dtype,
    )

                            
    video_vae_encoder = None
    if with_video_vae_encoder:
        logger.debug("Loading video VAE encoder...")
        video_vae_encoder = load_video_vae_encoder(checkpoint_path, torch_device, dtype)

                            
    video_vae_decoder = None
    if with_video_vae_decoder:
        logger.debug("Loading video VAE decoder...")
        video_vae_decoder = load_video_vae_decoder(checkpoint_path, torch_device, dtype)

    audio_vae_encoder = None
    if with_audio_vae_encoder:
        logger.debug("Loading audio VAE encoder...")
        audio_vae_encoder = load_audio_vae_encoder(checkpoint_path, torch_device, dtype)

                            
    audio_vae_decoder = None
    if with_audio_vae_decoder:
        logger.debug("Loading audio VAE decoder...")
        audio_vae_decoder = load_audio_vae_decoder(checkpoint_path, torch_device, dtype)

                  
    vocoder = None
    if with_vocoder:
        logger.debug("Loading vocoder...")
        vocoder = load_vocoder(checkpoint_path, torch_device, dtype)

                       
    text_encoder = None
    if with_text_encoder:
        if text_encoder_path is None:
            raise ValueError("text_encoder_path must be provided when with_text_encoder=True")
        logger.debug("Loading Gemma text encoder...")
        text_encoder = load_text_encoder(text_encoder_path, torch_device, dtype)

                                                     
    scheduler = LTX2Scheduler()

    return LtxModelComponents(
        transformer=transformer,
        video_vae_encoder=video_vae_encoder,
        video_vae_decoder=video_vae_decoder,
        audio_vae_encoder=audio_vae_encoder,
        audio_vae_decoder=audio_vae_decoder,
        vocoder=vocoder,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
