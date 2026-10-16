''
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image


class ObservationRefiner:
    def __init__(self, *, context_cache: Path, prompt: str, output: Path) -> None:
        import torch
        from rigen_refine import (
            BASE_MODEL, DISTILLED_LORA, SPATIAL_UPSAMPLER, ImageConditioner,
            VideoUpsampler, VideoDecoder,
            LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP,
        )
        from rigen_components import CachedOriginalDiffusionStage

        self.device, self.dtype = torch.device("cuda:0"), torch.bfloat16
        self.output = output
        output.mkdir(parents=True, exist_ok=False)
        contexts = torch.load(context_cache, map_location="cpu", weights_only=True)
        self.context = contexts[prompt].to(self.device)
        if self.context.ndim == 2:
            self.context = self.context.unsqueeze(0)
        self.config = dict(schema="single_rgb_T1_soft_anchor075_v1", sigmas=[.725, .421875, 0.],
                           partial_anchor_strength=.75, denoise_mask=.25, frames=1,
                           raw_size=[832, 480], refiner_size=[1664, 960],
                           context_cache=str(context_cache), prompt=prompt,
                           models=[str(BASE_MODEL), str(DISTILLED_LORA), str(SPATIAL_UPSAMPLER)],
                           feeds_back_to_ar_history=False, planner_uses_raw=True)
        (output / "config.json").write_text(json.dumps(self.config, indent=2))
                                                                                 
        with torch.random.fork_rng(devices=[0]):
            self.encoder = ImageConditioner(str(BASE_MODEL), self.dtype, self.device)._build_encoder()
            owner = VideoUpsampler(str(BASE_MODEL), str(SPATIAL_UPSAMPLER), self.dtype, self.device)
            self.upsampler = owner._upsampler_builder.build(device=self.device, dtype=self.dtype).eval()
            decoder_owner = VideoDecoder(str(BASE_MODEL), self.dtype, self.device)
            self.decoder = decoder_owner._decoder_builder.build(device=self.device, dtype=self.dtype).eval()
            loras = (LoraPathStrengthAndSDOps(str(DISTILLED_LORA), 1., LTXV_LORA_COMFY_RENAMING_MAP),)
            self.model = CachedOriginalDiffusionStage(str(BASE_MODEL), self.dtype, self.device, loras=loras)
        self.cache: dict[str, tuple[Image.Image, dict[str, Any]]] = {}

    def __call__(self, image: Image.Image, label: str) -> Image.Image:
        import torch
        from ltx_pipelines.utils.blocks import upsample_video
        from rigen_refine import (
            VideoConditionByLatentIndex, ModalitySpec,
        )
        from rigen_components import FixedVideoNoiser, SimpleDenoiserNoCamTokens, collect_decoded_video

        if image.size != (832, 480) or image.mode != "RGB":
            raise ValueError(f"Unexpected observation geometry {image.size}/{image.mode}")
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        image.save(self.output / f"{label}_raw.png")
        if digest in self.cache:
            result, metrics = self.cache[digest]
            record = dict(metrics, label=label, cache_hit=True)
        else:
            started = time.perf_counter()
            seed = int(digest[:12], 16)

            class CheckedNoiser(FixedVideoNoiser):
                def __call__(self, state, noise_scale: float = 1.):
                    if not torch.all(state.denoise_mask == .25):
                        raise RuntimeError("T=1 must have mutable mask=.25, not a locked first frame")
                    noised = super().__call__(state, noise_scale)
                    self.delta = (noised.latent.float()-state.latent.float()).abs().mean().item()
                    return noised

            with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
                pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
                pixels = pixels.to(device=self.device, dtype=self.dtype)[None, :, None] / 127.5 - 1.
                low = self.encoder(pixels)
                if tuple(low.shape) != (1, 128, 1, 15, 26):
                    raise RuntimeError(f"Expected independent T=1, got {low.shape}")
                up = upsample_video(latent=low, video_encoder=self.encoder, upsampler=self.upsampler)
                generator = torch.Generator(device=self.device).manual_seed(seed)
                noise = torch.randn(up.shape, device=self.device, dtype=self.dtype, generator=generator)
                noiser = CheckedNoiser(noise)
                value, _ = self.model(
                    denoiser=SimpleDenoiserNoCamTokens(v_context=self.context, a_context=None),
                    sigmas=torch.tensor(self.config["sigmas"], device=self.device, dtype=torch.float32),
                    noiser=noiser, width=1664, height=960, frames=1, fps=20.,
                    video=ModalitySpec(context=self.context, initial_latent=up, noise_scale=.725,
                        conditionings=[VideoConditionByLatentIndex(latent=up, strength=.75, latent_idx=0)]),
                    audio=None,
                )
                if tuple(value.latent.shape) != (1, 128, 1, 30, 52) or not torch.isfinite(value.latent).all():
                    raise RuntimeError("Invalid T=1 refined latent")
                delta = (value.latent.float()-up.float()).abs().mean().item()
                if delta <= 1e-6 or noiser.delta <= 1e-6:
                    raise RuntimeError("Refiner/noiser is a no-op")
                chunks = self.decoder.decode_video(value.latent, None, generator)
                video = collect_decoded_video(chunks)
                if tuple(video.shape) != (3, 1, 960, 1664):
                    raise RuntimeError(f"Unexpected decoded shape {video.shape}")
                full = Image.fromarray((video[:, 0].permute(1, 2, 0).numpy()*255).round().clip(0,255).astype(np.uint8))
                full.save(self.output / f"{label}_x2.png")
                result = full.resize((832, 480), Image.Resampling.LANCZOS)
                metrics = dict(input_rgb_sha256=digest, seed=seed, label=label,
                    latent_mean_abs_change=delta, noised_mean_abs_change=noiser.delta,
                    actual_denoise_mask=.25, seconds=time.perf_counter()-started)
            self.cache[digest] = (result, metrics)
            record = dict(metrics, cache_hit=False)
        result.save(self.output / f"{label}_refined.png")
        with (self.output / "frames.jsonl").open("a") as stream:
            stream.write(json.dumps(record)+"\n")
        return result.copy()

    def close(self) -> None:
        self.model._cached_transformer = None
        self.encoder = self.upsampler = self.decoder = self.model = self.context = None
        self.cache.clear()
