                      
''

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import inspect
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import av
import torch
import torch.distributed as dist
import torch.nn.functional as torch_f
import yaml
from safetensors import safe_open
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
for relative_path in (
    "packages/ltx-trainer/src",
    "packages/ltx-core/src",
    "packages/ltx-pipelines/src",
):
    absolute_path = REPO_ROOT / relative_path
    if str(absolute_path) not in sys.path:
        sys.path.insert(0, str(absolute_path))

import rigen_foundation_model as legacy
from ltx_core.components.patchifiers import VideoLatentPatchifier              
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps              
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder              
from ltx_core.model.transformer.attention import (              
    Attention,
    AttentionFunction,
    FlashAttention4,
    FlashAttention4BooleanMask,
    configure_attention_backend,
)
from ltx_core.model.transformer.modality import Modality              
from ltx_core.model.transformer.model_configurator import (              
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
)
from ltx_trainer.model_loader import (              
    load_embeddings_processor,
    load_text_encoder,
    load_video_vae_encoder,
)
from ltx_trainer.training_strategies.ar_teacher_masks import (              
    compute_ti2av_blocks,
)


LEGACY_OBJECTIVE_RECIPE = "legacy_two_point_gt_denoise_v1"
SANA_OBJECTIVE_RECIPE = "sana_paired_truncated_flow_v1"
SANA_LORA_TARGETS = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
SANA_STAGE2_SIGMA_SCHEDULES = (
    (0.909375, 0.725, 0.421875),
    (0.725, 0.421875),
)
PIXEL_DOWN_UP_SOURCE = "pixel_down_up"
AR10K_TEACHER_FORCED_SOURCE = "ar10k_teacher_forced_x0_cache"
SFDMD_FREE_ROLLOUT_SOURCE = "sfdmd_free_rollout_x0_cache"
CACHED_SOURCE_MODES = frozenset(
    (AR10K_TEACHER_FORCED_SOURCE, SFDMD_FREE_ROLLOUT_SOURCE)
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", default="auto", help="auto, none, or an integer step")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable all periodic and terminal checkpoint writes.",
    )
    return parser.parse_args()


def _nested(config: dict[str, Any], *keys: str) -> Any:
    return legacy._nested(config, *keys)


def _resolved_path(value: str | Path) -> Path:
    return legacy._resolved_path(value)


def _geometry(config: dict[str, Any]) -> tuple[int, int, int, float]:
    data = _nested(config, "data")
    return (
        int(_nested(data, "width")),
        int(_nested(data, "height")),
        int(_nested(data, "num_frames")),
        float(_nested(data, "fps")),
    )


def _objective_recipe(config: dict[str, Any]) -> str:
    return str(_nested(config, "objective").get("recipe", LEGACY_OBJECTIVE_RECIPE))


def _fuse_distilled_during_training(config: dict[str, Any]) -> bool:
    return bool(_nested(config, "model").get("fuse_distilled_lora_during_training", True))


def _source_mode(config: dict[str, Any]) -> str:
    return str(_nested(config, "data", "source_degradation", "mode"))


def _inference_sigmas(config: dict[str, Any]) -> tuple[float, ...]:
    training = _nested(config, "training")
    values = training.get("stage2_sigmas", training.get("refiner_sigmas"))
    if values is None:
        raise ValueError("training must define stage2_sigmas or refiner_sigmas")
    return tuple(float(value) for value in values)


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    paths = _nested(config, "paths")
    resolved = {
        "base_model": _resolved_path(_nested(paths, "base_model")),
        "distilled_lora": _resolved_path(_nested(paths, "distilled_lora")),
        "text_encoder": _resolved_path(_nested(paths, "text_encoder")),
        "data_root": _resolved_path(_nested(paths, "data_root")),
        "manifest": _resolved_path(_nested(paths, "manifest")),
    }
    missing = [f"{name}={path}" for name, path in resolved.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + "; ".join(missing))
    if "spatial_upsampler" in paths:
        raise ValueError("Same-resolution refiner must not configure a spatial upsampler")

    width, height, num_frames, fps = _geometry(config)
    if (width, height, num_frames, fps) != (832, 480, 81, 20.0):
        raise ValueError(
            "The 1v1 same-resolution pilot is locked to 832x480, 81 frames, 20 fps; "
            f"got {width}x{height}, {num_frames} frames, {fps} fps"
        )
    if width % 32 or height % 32 or (num_frames - 1) % 8:
        raise ValueError("Invalid LTX video geometry")

    objective = _nested(config, "objective")
    recipe = _objective_recipe(config)
    if recipe not in {LEGACY_OBJECTIVE_RECIPE, SANA_OBJECTIVE_RECIPE}:
        raise ValueError(f"Unsupported objective.recipe={recipe!r}")
    expected_name = (
        "velocity_flow_matching"
        if recipe == LEGACY_OBJECTIVE_RECIPE
        else "paired_truncated_flow_matching"
    )
    if str(_nested(objective, "name")) != expected_name:
        raise ValueError(f"objective.name must be {expected_name} for recipe={recipe}")
    if bool(objective.get("first_frame_conditioning")) is not True:
        raise ValueError("The same-resolution refiner must preserve the clean first frame")

    training = _nested(config, "training")
    inference_sigmas = _inference_sigmas(config)
    if recipe == LEGACY_OBJECTIVE_RECIPE:
        if inference_sigmas != (0.725, 0.421875):
            raise ValueError(
                "The legacy recipe must match refiner0.7 nonzero sigmas [0.725, 0.421875]"
            )
    else:
        if inference_sigmas not in SANA_STAGE2_SIGMA_SCHEDULES:
            raise ValueError(
                "SANA inference sigmas must use a supported schedule: "
                f"{[list(values) for values in SANA_STAGE2_SIGMA_SCHEDULES]}, "
                f"got {inference_sigmas}"
            )
        if _fuse_distilled_during_training(config):
            raise ValueError(
                "SANA refiner adaptation must train on the non-distilled base; "
                "set model.fuse_distilled_lora_during_training=false"
            )
        sigma_sampling = _nested(training, "sigma_sampling")
        if str(_nested(sigma_sampling, "mode")) != "truncated_logit_normal":
            raise ValueError("training.sigma_sampling.mode must be truncated_logit_normal")
        sigma_start = float(_nested(sigma_sampling, "sigma_start"))
        min_sigma = float(_nested(sigma_sampling, "min_sigma"))
        logit_std = float(_nested(sigma_sampling, "logit_std"))
        if abs(sigma_start - inference_sigmas[0]) > 1e-9:
            raise ValueError(
                "SANA sigma_start must match the first inference sigma: "
                f"{sigma_start} != {inference_sigmas[0]}"
            )
        if not 0.0 < min_sigma < sigma_start < 1.0:
            raise ValueError(f"Expected 0 < min_sigma < sigma_start < 1, got {min_sigma}, {sigma_start}")
        if not torch.isfinite(torch.tensor(float(_nested(sigma_sampling, "logit_mean")))):
            raise ValueError("training.sigma_sampling.logit_mean must be finite")
        if not torch.isfinite(torch.tensor(logit_std)) or logit_std <= 0.0:
            raise ValueError("training.sigma_sampling.logit_std must be positive")
        degradation = _nested(config, "data", "source_degradation")
        source_mode = str(_nested(degradation, "mode"))
        if source_mode == PIXEL_DOWN_UP_SOURCE:
            spatial_scale = float(_nested(degradation, "spatial_scale"))
            if not 0.0 < spatial_scale < 1.0:
                raise ValueError(
                    f"source degradation spatial_scale must be in (0,1), got {spatial_scale}"
                )
        elif source_mode in CACHED_SOURCE_MODES:
            cache_manifest = _resolved_path(_nested(degradation, "cache_manifest"))
            cache_info = _resolved_path(_nested(degradation, "cache_info"))
            if not cache_manifest.is_file() or not cache_info.is_file():
                raise FileNotFoundError(
                    "AR10k teacher-forced cache is incomplete: "
                    f"manifest={cache_manifest}, info={cache_info}"
                )
            payload = json.loads(cache_info.read_text(encoding="utf-8"))
            if payload.get("status") != "complete":
                raise ValueError(f"AR10k cache is not marked complete: {cache_info}")
            expected_checkpoint = _resolved_path(_nested(degradation, "ar_checkpoint"))
            cached_checkpoint = _resolved_path(str(payload.get("ar_checkpoint") or ""))
            if cached_checkpoint != expected_checkpoint:
                raise ValueError(
                    "AR10k cache checkpoint mismatch: "
                    f"cached={cached_checkpoint}, configured={expected_checkpoint}"
                )
            if int(payload.get("sample_count", -1)) != 24_288:
                raise ValueError(
                    f"AR10k cache expected 24,288 samples, got {payload.get('sample_count')}"
                )
            if source_mode == SFDMD_FREE_ROLLOUT_SOURCE:
                if payload.get("guidance_mode") != "conditional_only":
                    raise ValueError("SF-DMD free-rollout cache must be conditional-only")
                if payload.get("denoising_update_mode") != "consistency_renoise":
                    raise ValueError("SF-DMD free-rollout cache must use consistency re-noising")
                if tuple(float(value) for value in payload.get("denoising_sigmas", ())) != (
                    1.0,
                    0.9375,
                    0.8333333333333334,
                    0.625,
                    0.0,
                ):
                    raise ValueError("SF-DMD free-rollout cache has the wrong four-step schedule")
                generation_path = str(degradation.get("generation_path", "full_prefix"))
                expected_history = (
                    "generated_prefix_incremental_kv_cache"
                    if generation_path == "kv_cache"
                    else "generated_prefix_full_prefix_rerun"
                )
                if payload.get("history") != expected_history:
                    raise ValueError("SF-DMD free-rollout cache does not contain generated history")
                if generation_path == "kv_cache":
                    if payload.get("kv_cache_enabled") is not True:
                        raise ValueError("SF-DMD free-rollout cache is not KV-enabled")
                    if payload.get("kv_cache_attention_backend") != "native":
                        raise ValueError("SF-DMD free-rollout cache did not use native FA4 KV")
                    if int(payload.get("kv_cache_full_prefix_interval", -1)) != 0:
                        raise ValueError("SF-DMD free-rollout cache contains full-prefix anchors")
            resolved["cache_manifest"] = cache_manifest
            resolved["cache_info"] = cache_info
            resolved["ar_checkpoint"] = expected_checkpoint
        else:
            raise ValueError(
                "Unsupported data.source_degradation.mode: "
                f"{source_mode!r}; expected {PIXEL_DOWN_UP_SOURCE!r} or "
                f"one of {sorted(CACHED_SOURCE_MODES)!r}"
            )
        if bool(objective.get("reference_blockwise_mask")) is not True:
            raise ValueError("SANA recipe requires objective.reference_blockwise_mask=true")
    if int(_nested(training, "steps")) <= 0:
        raise ValueError("training.steps must be positive")
    if int(_nested(training, "gradient_accumulation_steps")) <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive")

    model = _nested(config, "model")
    if str(_nested(model, "attention_backend")) != "flash_attention_4":
        raise ValueError("The refiner trainer is fail-closed on flash_attention_4")
    lora = _nested(model, "lora")
    rank = int(_nested(lora, "rank"))
    alpha = int(_nested(lora, "alpha"))
    distilled_metadata = legacy._checkpoint_metadata(resolved["distilled_lora"])
    official_rank = int(distilled_metadata.get("lora_rank") or 0)
    official_alpha = int(distilled_metadata.get("lora_alpha") or 0)
    if (rank, alpha) != (official_rank, official_alpha) or (rank, alpha) != (384, 384):
        raise ValueError(
            "The domain LoRA must match the official LTX-2.3 distilled LoRA rank/alpha: "
            f"configured={rank}/{alpha}, official={official_rank}/{official_alpha}"
        )
    targets = list(_nested(lora, "target_modules"))
    if not targets or any("audio" in target for target in targets):
        raise ValueError("The domain LoRA must have explicit video-only target modules")
    if recipe == SANA_OBJECTIVE_RECIPE and tuple(targets) != SANA_LORA_TARGETS:
        raise ValueError(
            "SANA rank-384 LoRA targets must exactly cover video attention Q/K/V/O and FFN; "
            f"expected={list(SANA_LORA_TARGETS)}, got={targets}"
        )

    world_size = int(_nested(config, "distributed", "expected_world_size"))
    batch_size_per_gpu = int(_nested(config, "data", "batch_size_per_gpu"))
    supported_batch_topologies = {(4, 1), (8, 2), (16, 2)}
    if (world_size, batch_size_per_gpu) not in supported_batch_topologies:
        raise ValueError(
            "Unsupported rank-384 refiner batch topology: "
            f"world_size={world_size}, batch_size_per_gpu={batch_size_per_gpu}; "
            "expected pilot (4, 1), formal (8, 2), or scaled formal (16, 2)"
        )
    return {
        **resolved,
        "official_lora_rank": official_rank,
        "official_lora_alpha": official_alpha,
        "objective_recipe": recipe,
    }


class XonoticDuelRefinerDataset(legacy.CrossFPSRefinerDataset):
    ''

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        schedules: dict[str, tuple[str, ...]] = {}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row.get("sample_id") or row.get("clip_id") or "")
                schedule = row.get("prompt_schedule")
                if not sample_id or not isinstance(schedule, list) or len(schedule) != 5:
                    raise ValueError(
                        f"AR prompt schedule must contain five strings at manifest line {line_number}"
                    )
                if not all(isinstance(prompt, str) and prompt.strip() for prompt in schedule):
                    raise ValueError(f"Empty AR prompt at manifest line {line_number}")
                schedules[sample_id] = tuple(prompt.strip() for prompt in schedule)
        if len(schedules) != len(self.samples):
            raise ValueError(
                f"Prompt schedule/sample mismatch: schedules={len(schedules):,}, samples={len(self.samples):,}"
            )
        self.prompt_schedules = schedules

    @staticmethod
    def _clean_prompt(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    def _decode_video(self, sample: legacy.RefinerSample, indices: list[int]) -> Tensor:
        if not sample.video_path.is_file():
            raise FileNotFoundError(f"Missing 1v1 video for {sample.sample_id}: {sample.video_path}")
        target_map: dict[int, list[int]] = {}
        for output_index, source_index in enumerate(indices):
            target_map.setdefault(source_index, []).append(output_index)
        first_needed = min(target_map)
        last_needed = max(target_map)
        decoded: list[Tensor | None] = [None] * len(indices)

        with av.open(str(sample.video_path)) as container:
            stream = container.streams.video[0]
            if first_needed > 0 and sample.video_source_fps > 0 and stream.time_base is not None:
                seek_sec = max(first_needed / sample.video_source_fps - 1.0, 0.0)
                container.seek(
                    int(seek_sec / float(stream.time_base)),
                    stream=stream,
                    any_frame=False,
                    backward=True,
                )
            fallback_index = 0
            for frame in container.decode(video=0):
                if frame.time is not None and sample.video_source_fps > 0:
                    source_index = int(round(float(frame.time) * sample.video_source_fps))
                else:
                    source_index = fallback_index
                    fallback_index += 1
                if source_index > last_needed:
                    break
                if source_index not in target_map:
                    continue
                frame_tensor = (
                    torch.from_numpy(frame.to_ndarray(format="rgb24"))
                    .permute(2, 0, 1)
                    .float()
                    .div_(255.0)
                )
                for output_index in target_map[source_index]:
                    decoded[output_index] = frame_tensor

        first_valid = next((frame for frame in decoded if frame is not None), None)
        if first_valid is None:
            raise RuntimeError(
                f"Could not decode requested frames {first_needed}:{last_needed} from {sample.video_path}"
            )
        for index, frame in enumerate(decoded):
            if frame is None:
                decoded[index] = (
                    decoded[index - 1]
                    if index > 0 and decoded[index - 1] is not None
                    else first_valid
                )
        stacked = torch.stack([frame for frame in decoded if frame is not None], dim=0)
        if stacked.shape[0] != self.num_frames:
            raise RuntimeError(
                f"Decoded {stacked.shape[0]} frames for {sample.sample_id}; expected {self.num_frames}"
            )
        return self._resize_and_center_crop(stacked)

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = super().__getitem__(index)
        schedule = self.prompt_schedules[result["sample_id"]]
        for prompt_index, prompt in enumerate(schedule):
            result[f"prompt_{prompt_index}"] = prompt
        return result


class CachedXonoticDuelRefinerDataset(Dataset[dict[str, Any]]):
    ''

    def __init__(self, *, manifest_path: Path, cache_manifest: Path) -> None:
        schedules: dict[str, tuple[str, ...]] = {}
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row.get("sample_id") or row.get("clip_id") or "")
                schedule = row.get("prompt_schedule")
                if not sample_id or not isinstance(schedule, list) or len(schedule) != 5:
                    raise ValueError(
                        f"AR prompt schedule must contain five strings at manifest line {line_number}"
                    )
                if not all(isinstance(prompt, str) and prompt.strip() for prompt in schedule):
                    raise ValueError(f"Empty AR prompt at manifest line {line_number}")
                if sample_id in schedules:
                    raise ValueError(f"Duplicate sample_id={sample_id!r} in {manifest_path}")
                schedules[sample_id] = tuple(prompt.strip() for prompt in schedule)

        rows: list[dict[str, Any]] = []
        with cache_manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row.get("sample_id") or "")
                if sample_id not in schedules:
                    raise ValueError(
                        f"Unknown cached sample_id={sample_id!r} at {cache_manifest}:{line_number}"
                    )
                shard_path = Path(str(row.get("shard_path") or ""))
                if not shard_path.is_absolute():
                    shard_path = (cache_manifest.parent / shard_path).resolve()
                if not shard_path.is_file():
                    raise FileNotFoundError(f"Missing AR10k latent shard: {shard_path}")
                row["shard_path"] = str(shard_path)
                rows.append(row)
        rows.sort(key=lambda row: int(row["global_index"]))
        cached_ids = [str(row["sample_id"]) for row in rows]
        if len(cached_ids) != len(set(cached_ids)):
            raise ValueError(f"Duplicate sample IDs in {cache_manifest}")
        if set(cached_ids) != set(schedules):
            missing = sorted(set(schedules) - set(cached_ids))[:5]
            extra = sorted(set(cached_ids) - set(schedules))[:5]
            raise ValueError(
                "AR10k cache/source manifest coverage mismatch: "
                f"cache={len(cached_ids)}, source={len(schedules)}, missing={missing}, extra={extra}"
            )
        if [int(row["global_index"]) for row in rows] != list(range(len(rows))):
            raise ValueError("AR10k cache global_index values must be contiguous from zero")
        self.rows = rows
        self.prompt_schedules = schedules

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with safe_open(row["shard_path"], framework="pt", device="cpu") as shard:
            clean_latent = shard.get_tensor(str(row["clean_key"]))
            source_latent = shard.get_tensor(str(row["source_key"]))
        expected_shape = (128, 11, 15, 26)
        if tuple(clean_latent.shape) != expected_shape or tuple(source_latent.shape) != expected_shape:
            raise ValueError(
                f"Cached latent shape mismatch for {row['sample_id']}: "
                f"clean={tuple(clean_latent.shape)}, source={tuple(source_latent.shape)}"
            )
        result: dict[str, Any] = {
            "clean_latent": clean_latent,
            "source_latent": source_latent,
            "sample_id": str(row["sample_id"]),
            "proxy_sigma": float(row.get("proxy_sigma", row.get("source_sigma_start", 1.0))),
        }
        for prompt_index, prompt in enumerate(self.prompt_schedules[result["sample_id"]]):
            result[f"prompt_{prompt_index}"] = prompt
        return result


def _build_dataset(
    config: dict[str, Any], validated: dict[str, Any]
) -> XonoticDuelRefinerDataset | CachedXonoticDuelRefinerDataset:
    if _source_mode(config) in CACHED_SOURCE_MODES:
        return CachedXonoticDuelRefinerDataset(
            manifest_path=validated["manifest"],
            cache_manifest=validated["cache_manifest"],
        )
    width, height, num_frames, fps = _geometry(config)
    return XonoticDuelRefinerDataset(
        manifest_path=validated["manifest"],
        data_root=validated["data_root"],
        target_width=width,
        target_height=height,
        num_frames=num_frames,
        fps=fps,
        random_temporal_crop=bool(_nested(config, "data", "random_temporal_crop")),
    )


def _sample_truncated_logit_normal(
    *,
    batch_size: int,
    sigma_start: float,
    min_sigma: float,
    logit_mean: float,
    logit_std: float,
    device: torch.device,
) -> Tensor:
    ''

    low_logit = math.log(min_sigma / (1.0 - min_sigma))
    high_logit = math.log(sigma_start / (1.0 - sigma_start))
    sqrt_two = math.sqrt(2.0)

    def normal_cdf(value: float) -> float:
        standardized = (value - logit_mean) / (logit_std * sqrt_two)
        return 0.5 * (1.0 + math.erf(standardized))

    cdf_low = normal_cdf(low_logit)
    cdf_high = normal_cdf(high_logit)
    if not 0.0 <= cdf_low < cdf_high <= 1.0:
        raise ValueError(
            "Invalid truncated logit-normal CDF interval: "
            f"cdf_low={cdf_low}, cdf_high={cdf_high}"
        )
    quantile = torch.rand(batch_size, device=device, dtype=torch.float32)
    quantile = quantile.mul(cdf_high - cdf_low).add(cdf_low)
    epsilon = torch.finfo(quantile.dtype).eps
    quantile = quantile.clamp(min=epsilon, max=1.0 - epsilon)
    standard_normal = sqrt_two * torch.erfinv(2.0 * quantile - 1.0)
    return torch.sigmoid(logit_mean + logit_std * standard_normal).clamp(
        min=min_sigma,
        max=sigma_start,
    )


def _objective_self_test(config: dict[str, Any], *, sigma_sampler: Any = None) -> None:
    generator = torch.Generator(device="cpu").manual_seed(17)
    clean = torch.randn(2, 13, 7, generator=generator)
    anchor = torch.randn(2, 13, 7, generator=generator)
    noise = torch.randn(2, 13, 7, generator=generator)
    recipe = _objective_recipe(config)
    if recipe == LEGACY_OBJECTIVE_RECIPE:
        sigma = torch.tensor([0.725, 0.421875]).view(-1, 1, 1)
        noisy = (1.0 - sigma) * clean + sigma * noise
        target = noise - clean
    elif recipe == SANA_OBJECTIVE_RECIPE:
        sampling = _nested(config, "training", "sigma_sampling")
        sigma_start = float(_nested(sampling, "sigma_start"))
        min_sigma = float(_nested(sampling, "min_sigma"))
        sampled = sigma_sampler(4096, torch.device("cpu")) if sigma_sampler is not None else _sample_truncated_logit_normal(
            batch_size=4096,
            sigma_start=sigma_start,
            min_sigma=min_sigma,
            logit_mean=float(_nested(sampling, "logit_mean")),
            logit_std=float(_nested(sampling, "logit_std")),
            device=torch.device("cpu"),
        )
        if sampled.min().item() < min_sigma or sampled.max().item() > sigma_start:
            raise AssertionError("Continuous sigma sampler escaped configured bounds")
        sigma = torch.tensor([sigma_start, 0.37]).view(-1, 1, 1)
        source = (1.0 - sigma_start) * anchor + sigma_start * noise
        alpha = sigma / sigma_start
        noisy = (1.0 - alpha) * clean + alpha * source
        target = (source - clean) / sigma_start
        if not torch.allclose(noisy[:1], source[:1], atol=1e-6, rtol=1e-6):
            raise AssertionError("Paired bridge does not reach its source at sigma_start")
    else:
        raise AssertionError(f"Unhandled objective recipe: {recipe}")
    reconstructed = noisy - sigma * target
    if not torch.allclose(reconstructed, clean, atol=1e-6, rtol=1e-6):
        raise AssertionError(f"Objective algebra self-test failed for recipe={recipe}")


def _degrade_video_for_source(video: Tensor, config: dict[str, Any]) -> Tensor:
    ''

    degradation = _nested(config, "data", "source_degradation")
    if str(_nested(degradation, "mode")) != PIXEL_DOWN_UP_SOURCE:
        raise ValueError("Only pixel_down_up source degradation is implemented")
    scale = float(_nested(degradation, "spatial_scale"))
    batch, channels, frames, height, width = video.shape
    low_height = max(1, int(round(height * scale)))
    low_width = max(1, int(round(width * scale)))
    frame_batch = video.permute(0, 2, 1, 3, 4).reshape(
        batch * frames,
        channels,
        height,
        width,
    )
    low = torch_f.interpolate(frame_batch.float(), size=(low_height, low_width), mode="area")
    restored = torch_f.interpolate(
        low,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return (
        restored.clamp_(-1.0, 1.0)
        .to(dtype=video.dtype)
        .reshape(batch, frames, channels, height, width)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


def _build_ar_prompt_context_mask(
    *,
    prompt_attention_mask: Tensor,
    latent_frames: int,
    spatial_tokens: int,
) -> Tensor:
    ''

    if prompt_attention_mask.ndim != 3 or prompt_attention_mask.shape[1] != 5:
        raise ValueError(
            "prompt_attention_mask must be [B, 5, L], got "
            f"{tuple(prompt_attention_mask.shape)}"
        )
    blocks = compute_ti2av_blocks(
        latent_frames,
        None,
        chunk_size=2,
        video_fps=20.0,
        video_temporal_scale=8,
    )
    if len(blocks) != 6:
        raise ValueError(f"Expected image prefix plus five generated blocks, got {len(blocks)}")
    batch, prompt_count, prompt_length = prompt_attention_mask.shape
    query_tokens = latent_frames * spatial_tokens
    context_mask = torch.zeros(
        batch,
        query_tokens,
        prompt_count * prompt_length,
        dtype=torch.bool,
        device=prompt_attention_mask.device,
    )
    valid = prompt_attention_mask.to(dtype=torch.bool)
    for block in blocks:
        prompt_index = max(block.index - 1, 0)
        query_start = block.video_start * spatial_tokens
        query_end = block.video_end * spatial_tokens
        context_start = prompt_index * prompt_length
        context_end = context_start + prompt_length
        context_mask[:, query_start:query_end, context_start:context_end] = valid[
            :, prompt_index, None, :
        ]
    return context_mask


def _run_preflight(
    config: dict[str, Any],
) -> tuple[
    dict[str, Any],
    XonoticDuelRefinerDataset | CachedXonoticDuelRefinerDataset,
]:
    validated = _validate_config(config)
    target_counts = legacy._validate_lora_targets(config, validated["base_model"])
    dataset = _build_dataset(config, validated)
    if len(dataset) != 24_288:
        raise ValueError(f"Expected 24,288 1v1 samples, found {len(dataset):,}")
    probe = dataset[len(dataset) // 2]
    width, height, num_frames, _fps = _geometry(config)
    if _source_mode(config) in CACHED_SOURCE_MODES:
        for key in ("clean_latent", "source_latent"):
            if tuple(probe[key].shape) != (128, 11, 15, 26):
                raise ValueError(f"Cached probe {key} has wrong shape: {tuple(probe[key].shape)}")
            if probe[key].dtype != torch.bfloat16 or not torch.isfinite(probe[key]).all():
                raise ValueError(f"Cached probe {key} must be finite BF16")
    else:
        if tuple(probe["video_pixels"].shape) != (num_frames, 3, height, width):
            raise ValueError(f"Decoded probe has wrong shape: {tuple(probe['video_pixels'].shape)}")
        if not torch.isfinite(probe["video_pixels"]).all():
            raise ValueError("Decoded probe contains non-finite pixels")
    for prompt_index in range(5):
        prompt = probe[f"prompt_{prompt_index}"]
        if "[BLOCK START STATE]" not in prompt or "[COMBAT OUTCOME]" not in prompt:
            raise ValueError(f"AR prompt {prompt_index} is missing required event sections")
    _objective_self_test(config)
    print("[duel-refiner-preflight] PASS", flush=True)
    print(f"[duel-refiner-preflight] manifest={validated['manifest']}", flush=True)
    print(
        f"[duel-refiner-preflight] samples={len(dataset):,} probe={probe['sample_id']} "
        f"source_mode={_source_mode(config)}",
        flush=True,
    )
    print(
        f"[duel-refiner-preflight] official_lora_rank={validated['official_lora_rank']} "
        f"new_lora_rank={_nested(config, 'model', 'lora', 'rank')} targets={target_counts}",
        flush=True,
    )
    print(
        "[duel-refiner-preflight] "
        f"prompt=ar_5x0p8s_masked recipe={validated['objective_recipe']} "
        f"distilled_lora_fused={_fuse_distilled_during_training(config)} "
        f"blockwise_reference={bool(_nested(config, 'objective').get('reference_blockwise_mask', False))} "
        "upsampler=disabled audio=passthrough action=stage1_only",
        flush=True,
    )
    return validated, dataset


class SameResolutionRefinerModule(nn.Module):
    def __init__(
        self,
        *,
        transformer: nn.Module,
        objective_recipe: str,
        sigma_values: tuple[float, ...],
        sigma_sampling: dict[str, Any] | None,
        first_frame_conditioning: bool,
        reference_blockwise_mask: bool,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.register_buffer(
            "sigma_values",
            torch.tensor(sigma_values, dtype=torch.float32),
            persistent=False,
        )
        self.objective_recipe = objective_recipe
        self.sigma_sampling = dict(sigma_sampling or {})
        self.first_frame_conditioning = first_frame_conditioning
        self.reference_blockwise_mask = reference_blockwise_mask
        self.patchifier = VideoLatentPatchifier(patch_size=1)
        self._reference_mask_cache: dict[tuple[torch.device, int, int], FlashAttention4BooleanMask] = {}

    def _reference_mask(
        self,
        *,
        device: torch.device,
        sequence_length: int,
        first_frame_tokens: int,
    ) -> FlashAttention4BooleanMask | None:
        if not self.reference_blockwise_mask:
            return None
        cache_key = (device, sequence_length, first_frame_tokens)
        cached = self._reference_mask_cache.get(cache_key)
        if cached is not None:
            return cached
                                                                               
                                                       
        keep = torch.ones(
            sequence_length,
            sequence_length,
            device=device,
            dtype=torch.bool,
        )
        keep[:first_frame_tokens, first_frame_tokens:] = False
        mask = FlashAttention4BooleanMask(keep.contiguous())
        self._reference_mask_cache[cache_key] = mask
        return mask

    def forward(
        self,
        *,
        clean_latent: Tensor,
        source_latent: Tensor | None,
        prompt_context: Tensor,
        prompt_mask: Tensor,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        clean_tokens = self.patchifier.patchify(clean_latent)
        batch_size, sequence_length, _channels = clean_tokens.shape
        if self.objective_recipe == LEGACY_OBJECTIVE_RECIPE:
            sigma_indices = torch.randint(
                low=0,
                high=self.sigma_values.numel(),
                size=(batch_size,),
                device=clean_latent.device,
            )
            sigma = self.sigma_values[sigma_indices].to(device=clean_latent.device)
            sigma_expanded = sigma.view(-1, 1, 1).to(dtype=clean_tokens.dtype)
            noise = torch.randn_like(clean_tokens)
            noisy_tokens = (1.0 - sigma_expanded) * clean_tokens + sigma_expanded * noise
            velocity_target = noise - clean_tokens
        elif self.objective_recipe == SANA_OBJECTIVE_RECIPE:
            if source_latent is None:
                raise ValueError("SANA paired objective requires source_latent")
            if source_latent.shape != clean_latent.shape:
                raise ValueError(
                    f"source/clean latent mismatch: {tuple(source_latent.shape)} vs "
                    f"{tuple(clean_latent.shape)}"
                )
            anchor_tokens = self.patchifier.patchify(source_latent)
            sigma_start = float(self.sigma_sampling["sigma_start"])
            sigma = _sample_truncated_logit_normal(
                batch_size=batch_size,
                sigma_start=sigma_start,
                min_sigma=float(self.sigma_sampling["min_sigma"]),
                logit_mean=float(self.sigma_sampling["logit_mean"]),
                logit_std=float(self.sigma_sampling["logit_std"]),
                device=clean_latent.device,
            )
            sigma_expanded = sigma.view(-1, 1, 1).to(dtype=clean_tokens.dtype)
            source_noise = torch.randn_like(clean_tokens)
            source_tokens = (1.0 - sigma_start) * anchor_tokens + sigma_start * source_noise
            alpha = sigma_expanded / sigma_start
            noisy_tokens = (1.0 - alpha) * clean_tokens + alpha * source_tokens
            velocity_target = (source_tokens - clean_tokens) / sigma_start
        else:
            raise RuntimeError(f"Unhandled objective recipe: {self.objective_recipe}")

        timesteps = sigma.view(-1, 1).expand(-1, sequence_length).clone()
        loss_mask = torch.ones(
            batch_size,
            sequence_length,
            device=clean_latent.device,
            dtype=torch.bool,
        )
        first_frame_tokens = clean_latent.shape[-2] * clean_latent.shape[-1]
        if self.first_frame_conditioning:
            noisy_tokens[:, :first_frame_tokens] = clean_tokens[:, :first_frame_tokens]
            timesteps[:, :first_frame_tokens] = 0
            loss_mask[:, :first_frame_tokens] = False

        video = Modality(
            enabled=True,
            latent=noisy_tokens,
            sigma=sigma,
            timesteps=timesteps,
            positions=positions,
            context=prompt_context,
            context_mask=prompt_mask,
            self_attention_block_mask=self._reference_mask(
                device=clean_latent.device,
                sequence_length=sequence_length,
                first_frame_tokens=first_frame_tokens,
            ),
        )
        velocity_prediction, _audio_prediction = self.transformer(
            video=video,
            audio=None,
            perturbations=None,
        )
        if velocity_prediction is None:
            raise RuntimeError("LTX transformer returned no video prediction")
        squared_error = (velocity_prediction.float() - velocity_target.float()).pow(2)
        expanded_mask = loss_mask.unsqueeze(-1).expand_as(squared_error)
        return squared_error.masked_select(expanded_mask).mean(), sigma.detach()


def _load_modules(
    *,
    config: dict[str, Any],
    validated: dict[str, Any],
    device: torch.device,
) -> tuple[SameResolutionRefinerModule, nn.Module | None, legacy.OnlinePromptEncoder]:
    from peft import LoraConfig, get_peft_model

    rank = int(os.environ.get("RANK", "0"))
    vae_encoder: nn.Module | None = None
    if _source_mode(config) == PIXEL_DOWN_UP_SOURCE:
        print(f"[duel-refiner-load rank={rank}] loading frozen video VAE encoder", flush=True)
        vae_encoder = load_video_vae_encoder(
            validated["base_model"],
            device=device,
            dtype=torch.bfloat16,
        ).eval()
        vae_encoder.requires_grad_(False)
    else:
        print(
            f"[duel-refiner-load rank={rank}] using cached AR10k clean/source latents; "
            "video VAE encoder not loaded",
            flush=True,
        )

    print(f"[duel-refiner-load rank={rank}] loading frozen text stack", flush=True)
    text_encoder = load_text_encoder(
        validated["text_encoder"],
        device=device,
        dtype=torch.bfloat16,
        load_in_8bit=bool(_nested(config, "model", "load_text_encoder_in_8bit")),
    )
    if getattr(text_encoder, "model", None) is not None:
        text_encoder.model.eval()
        text_encoder.model.requires_grad_(False)
    embeddings_processor = load_embeddings_processor(
        validated["base_model"],
        device=device,
        dtype=torch.bfloat16,
    ).eval()
    embeddings_processor.requires_grad_(False)

    fuse_distilled = _fuse_distilled_during_training(config)
    training_chain = "dev+official-distilled-lora" if fuse_distilled else "non-distilled-dev"
    print(f"[duel-refiner-load rank={rank}] loading LTX-2.3 {training_chain}", flush=True)
    loras = (
        (
            LoraPathStrengthAndSDOps(
                str(validated["distilled_lora"]),
                1.0,
                LTXV_LORA_COMFY_RENAMING_MAP,
            ),
        )
        if fuse_distilled
        else ()
    )
    transformer = SingleGPUModelBuilder(
        model_path=str(validated["base_model"]),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        loras=loras,
    ).build(device=device, dtype=torch.bfloat16)
    configured = configure_attention_backend(transformer, AttentionFunction.FLASH_ATTENTION_4)
    if configured == 0:
        raise RuntimeError("Transformer contains no configurable attention modules")
    wrong_backends = [
        name
        for name, module in transformer.named_modules()
        if isinstance(module, Attention) and not isinstance(module.attention_function, FlashAttention4)
    ]
    if wrong_backends:
        raise RuntimeError(f"FA4 audit failed; no fallback is allowed: {wrong_backends[:8]}")
    transformer.requires_grad_(False)
    transformer.set_gradient_checkpointing(bool(_nested(config, "model", "gradient_checkpointing")))

    lora = _nested(config, "model", "lora")
    transformer = get_peft_model(
        transformer,
        LoraConfig(
            r=int(_nested(lora, "rank")),
            lora_alpha=int(_nested(lora, "alpha")),
            target_modules=list(_nested(lora, "target_modules")),
            lora_dropout=float(_nested(lora, "dropout")),
            init_lora_weights=True,
        ),
    )
    module = SameResolutionRefinerModule(
        transformer=transformer,
        objective_recipe=validated["objective_recipe"],
        sigma_values=_inference_sigmas(config),
        sigma_sampling=_nested(config, "training").get("sigma_sampling"),
        first_frame_conditioning=bool(_nested(config, "objective", "first_frame_conditioning")),
        reference_blockwise_mask=bool(
            _nested(config, "objective").get("reference_blockwise_mask", False)
        ),
    ).to(device)
    prompt_encoder = legacy.OnlinePromptEncoder(
        text_encoder=text_encoder,
        embeddings_processor=embeddings_processor,
        device=device,
        cache_size=int(_nested(config, "data", "prompt_cache_size")),
    )
    print(
        f"[duel-refiner-load rank={rank}] FA4 configured on {configured} modules; fallback=disabled",
        flush=True,
    )
    return module, vae_encoder, prompt_encoder


@dataclass(frozen=True)
class ResumeBundle:
    step: int
    lora_path: Path
    training_state_path: Path | None


def _bundle_for_step(output_dir: Path, step: int) -> ResumeBundle | None:
    checkpoint_dir = output_dir / "checkpoints"
    lora_path = checkpoint_dir / f"refiner_lora_step_{step:06d}.safetensors"
    state_path = checkpoint_dir / f"training_state_step_{step:06d}.pt"
    if not lora_path.is_file():
        return None
    return ResumeBundle(
        step=step,
        lora_path=lora_path,
        training_state_path=state_path if state_path.is_file() else None,
    )


def _resolve_resume(output_dir: Path, resume: str) -> ResumeBundle | None:
    normalized = resume.strip().lower()
    checkpoint_dir = output_dir / "checkpoints"
    candidates: list[int] = []
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.glob("refiner_lora_step_*.safetensors"):
            try:
                candidates.append(int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
    if normalized == "none":
        if candidates:
            raise FileExistsError(f"Existing refiner checkpoints under {checkpoint_dir}")
        return None
    if normalized == "auto":
        return _bundle_for_step(output_dir, max(candidates)) if candidates else None
    try:
        step = int(normalized)
    except ValueError as exc:
        raise ValueError(f"--resume must be auto, none, or an integer; got {resume!r}") from exc
    bundle = _bundle_for_step(output_dir, step)
    if bundle is None:
        raise FileNotFoundError(f"No refiner LoRA at step {step} under {checkpoint_dir}")
    return bundle


def _load_resume_weights(module: SameResolutionRefinerModule, bundle: ResumeBundle) -> None:
                                                                          
                                                                          
                                                                               
                                                                             
                                                                      
    from peft.tuners.lora.layer import LoraLayer
    from transformers.integrations import tensor_parallel as transformers_tp

    if not hasattr(transformers_tp, "EmbeddingParallel"):
        tp_layers: list[str] = []
        for name, child in module.transformer.get_base_model().named_modules():
            if not isinstance(child, LoraLayer):
                continue
            base_layer = child.get_base_layer()
            if (
                getattr(base_layer, "_hf_tp_plan", None) is not None
                or getattr(base_layer, "_hf_device_mesh", None) is not None
            ):
                tp_layers.append(name)
        if tp_layers:
            raise RuntimeError(
                "PEFT/transformers tensor-parallel compatibility is unavailable for "
                f"LoRA resume; TP layers include {tp_layers[:8]}"
            )

        class _UnavailableEmbeddingParallel:
            pass

        transformers_tp.EmbeddingParallel = _UnavailableEmbeddingParallel

    from peft import set_peft_model_state_dict

    state = {
        key.replace("diffusion_model.", "", 1): value
        for key, value in load_file(bundle.lora_path).items()
    }
    result = set_peft_model_state_dict(module.transformer.get_base_model(), state)
    if getattr(result, "unexpected_keys", None):
        raise RuntimeError(f"Unexpected resumed LoRA keys: {result.unexpected_keys[:8]}")


def _fingerprint(config: dict[str, Any], validated: dict[str, Any]) -> str:
    payload = {
        "base_model": str(validated["base_model"]),
        "distilled_lora": str(validated["distilled_lora"]),
        "manifest": str(validated["manifest"]),
        "data": _nested(config, "data"),
        "objective": _nested(config, "objective"),
        "model": {
            "fuse_distilled_lora_during_training": _fuse_distilled_during_training(config),
            "attention_backend": _nested(config, "model", "attention_backend"),
        },
        "lora": _nested(config, "model", "lora"),
        "inference_sigmas": _inference_sigmas(config),
        "sigma_sampling": _nested(config, "training").get("sigma_sampling"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_training_state(
    *,
    bundle: ResumeBundle,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_fingerprint: str,
    device: torch.device,
) -> None:
    if bundle.training_state_path is None:
        for _ in range(bundle.step):
            scheduler.step()
        return
    state = torch.load(bundle.training_state_path, map_location=device, weights_only=False)
    if state.get("fingerprint") != expected_fingerprint or int(state.get("step", -1)) != bundle.step:
        raise ValueError("Resume state does not match this refiner recipe")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if "torch_rng_state" in state:
        torch.random.set_rng_state(state["torch_rng_state"].cpu())
    if "cuda_rng_state" in state:
        torch.cuda.set_rng_state(state["cuda_rng_state"].cpu(), device=device)


def _save_checkpoint(
    *,
    distributed: legacy.DistributedContext,
    module: SameResolutionRefinerModule,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    validated: dict[str, Any],
    output_dir: Path,
    step: int,
    fingerprint: str,
) -> None:
    dist.barrier()
    if distributed.is_main:
        from peft import get_peft_model_state_dict

        checkpoint_dir = output_dir / "checkpoints"
        lora_path = checkpoint_dir / f"refiner_lora_step_{step:06d}.safetensors"
        state_path = checkpoint_dir / f"training_state_step_{step:06d}.pt"
        bundle_path = checkpoint_dir / f"refiner_bundle_step_{step:06d}.json"
        for path in (lora_path, bundle_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
        raw_state = get_peft_model_state_dict(module.transformer)
        saved_state = {
            f"diffusion_model.{key.replace('base_model.model.', '', 1)}": value.detach()
            .to(device="cpu", dtype=torch.bfloat16)
            .contiguous()
            for key, value in raw_state.items()
        }
        recipe = validated["objective_recipe"]
        metadata = {
            "training_kind": "xonotic_duel_same_resolution_refiner",
            "training_step": str(step),
            "base_model": str(validated["base_model"]),
            "distilled_lora": str(validated["distilled_lora"]),
            "manifest": str(validated["manifest"]),
            "objective": str(_nested(config, "objective", "name")),
            "objective_recipe": recipe,
            "training_backbone": (
                "dev_plus_official_distilled_lora"
                if _fuse_distilled_during_training(config)
                else "non_distilled_dev"
            ),
            "reference_blockwise_mask": str(
                bool(_nested(config, "objective").get("reference_blockwise_mask", False))
            ).lower(),
            "lora_rank": str(_nested(config, "model", "lora", "rank")),
            "lora_alpha": str(_nested(config, "model", "lora", "alpha")),
            "model_version": "2.3.0",
            "spatial_upsampler": "disabled",
            "fingerprint": fingerprint,
        }
        legacy._atomic_save_safetensors(saved_state, lora_path, metadata)
        del raw_state, saved_state
        if bool(_nested(config, "checkpointing", "save_optimizer_state")):
            legacy._atomic_torch_save(
                {
                    "step": step,
                    "fingerprint": fingerprint,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "torch_rng_state": torch.random.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(distributed.device),
                },
                state_path,
            )
        bundle = {
            "step": step,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "base_model": str(validated["base_model"]),
            "official_distilled_lora": str(validated["distilled_lora"]),
            "domain_refiner_lora": str(lora_path),
            "inference_lora_order": [str(validated["distilled_lora"]), str(lora_path)],
            "resolution": [832, 480, 81],
            "sigma_schedule": [*_inference_sigmas(config), 0.0],
            "spatial_upsampler": None,
            "audio": "stage1_passthrough",
            "objective": str(_nested(config, "objective", "name")),
            "objective_recipe": recipe,
            "training_fused_official_distilled_lora": _fuse_distilled_during_training(config),
            "reference_blockwise_mask": bool(
                _nested(config, "objective").get("reference_blockwise_mask", False)
            ),
            "source_degradation": _nested(config, "data").get("source_degradation"),
            "fingerprint": fingerprint,
        }
        temporary = bundle_path.with_name(f".{bundle_path.name}.tmp-{os.getpid()}")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        temporary.replace(bundle_path)

        keep_states = int(_nested(config, "checkpointing", "keep_optimizer_states"))
        state_paths = sorted(
            checkpoint_dir.glob("training_state_step_*.pt"),
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
        )
        for old_path in state_paths[: max(0, len(state_paths) - keep_states)]:
            old_path.unlink()
        print(
            f"[duel-refiner-checkpoint] step={step} lora={lora_path} "
            f"state={state_path if state_path.exists() else 'disabled'}",
            flush=True,
        )
    dist.barrier()


def _write_snapshot(
    *,
    config: dict[str, Any],
    validated: dict[str, Any],
    output_dir: Path,
    distributed: legacy.DistributedContext,
    fingerprint: str,
) -> None:
    if not distributed.is_main:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "training_config.yaml"
    if not config_path.exists():
        clean = {key: value for key, value in config.items() if not key.startswith("_")}
        config_path.write_text(yaml.safe_dump(clean, sort_keys=False), encoding="utf-8")
    source_path = output_dir / Path(__file__).name
    if not source_path.exists():
        shutil.copy2(Path(__file__).resolve(), source_path)
    (output_dir / "run_info.json").write_text(
        json.dumps(
            {
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "world_size": distributed.world_size,
                "base_model": str(validated["base_model"]),
                "distilled_lora": str(validated["distilled_lora"]),
                "manifest": str(validated["manifest"]),
                "objective": str(_nested(config, "objective", "name")),
                "objective_recipe": validated["objective_recipe"],
                "training_fused_official_distilled_lora": _fuse_distilled_during_training(config),
                "inference_sigma_schedule": [*_inference_sigmas(config), 0.0],
                "reference_blockwise_mask": bool(
                    _nested(config, "objective").get("reference_blockwise_mask", False)
                ),
                "spatial_upsampler": None,
                "fingerprint": fingerprint,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _next_batch(
    iterator: Iterator[dict[str, Any]],
    loader: DataLoader,
    sampler: DistributedSampler,
    epoch: int,
) -> tuple[dict[str, Any], Iterator[dict[str, Any]], int]:
    return legacy._next_batch(iterator, loader, sampler, epoch)


def _train(
    *,
    config: dict[str, Any],
    validated: dict[str, Any],
    dataset: XonoticDuelRefinerDataset | CachedXonoticDuelRefinerDataset,
    output_dir: Path,
    resume_mode: str,
    save_checkpoints: bool,
) -> None:
    distributed = legacy._init_distributed()
    expected_world_size = int(_nested(config, "distributed", "expected_world_size"))
    if distributed.world_size != expected_world_size:
        raise RuntimeError(f"Expected world_size={expected_world_size}, got {distributed.world_size}")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed = int(_nested(config, "training", "seed"))
    legacy._seed_model_initialization(seed)

    fingerprint = _fingerprint(config, validated)
    resume = _resolve_resume(output_dir, resume_mode)
    _write_snapshot(
        config=config,
        validated=validated,
        output_dir=output_dir,
        distributed=distributed,
        fingerprint=fingerprint,
    )
    dist.barrier()
    legacy._rank_print(
        distributed,
        f"same-res 832x480 dataset={len(dataset):,} output={output_dir}",
    )

    module, vae_encoder, prompt_encoder = _load_modules(
        config=config,
        validated=validated,
        device=distributed.device,
    )
    if resume is not None:
        _load_resume_weights(module, resume)
    legacy._verify_trainable_initialization(module, distributed)

    trainable_parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    unexpected = [
        name
        for name, parameter in module.transformer.named_parameters()
        if parameter.requires_grad and "lora_" not in name.lower()
    ]
    if not trainable_parameters or unexpected:
        raise RuntimeError(
            f"Invalid trainable set: count={len(trainable_parameters)} unexpected={unexpected[:8]}"
        )
    optimizer = legacy._build_optimizer(
        [
            {
                "params": trainable_parameters,
                "lr": float(_nested(config, "training", "learning_rate")),
                "name": "domain_refiner_lora",
            }
        ],
        config,
    )
    scheduler = legacy._build_scheduler(optimizer, config)
    if resume is not None:
        _load_training_state(
            bundle=resume,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_fingerprint=fingerprint,
            device=distributed.device,
        )
    start_step = resume.step if resume is not None else 0

    ddp_kwargs: dict[str, Any] = {
        "device_ids": [distributed.local_rank],
        "output_device": distributed.local_rank,
        "broadcast_buffers": False,
        "find_unused_parameters": False,
        "gradient_as_bucket_view": True,
        "static_graph": True,
        "bucket_cap_mb": int(_nested(config, "distributed", "ddp_bucket_cap_mb")),
    }
    if "init_sync" in inspect.signature(DistributedDataParallel).parameters:
        ddp_kwargs["init_sync"] = False
    ddp_model = DistributedDataParallel(module, **ddp_kwargs)
    ddp_model.train()

    legacy._seed_data(seed, distributed.rank)
    batch_size = int(_nested(config, "data", "batch_size_per_gpu"))
    sampler = DistributedSampler(
        dataset,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    accumulation_steps = int(_nested(config, "training", "gradient_accumulation_steps"))
    initial_epoch = start_step * accumulation_steps * batch_size * distributed.world_size // len(dataset)
    sampler.set_epoch(initial_epoch)
    num_workers = int(_nested(config, "data", "num_workers"))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
        worker_init_fn=legacy._worker_init_fn,
        generator=torch.Generator().manual_seed(seed + distributed.rank),
        prefetch_factor=int(_nested(config, "data", "prefetch_factor")) if num_workers > 0 else None,
    )
    iterator = iter(loader)
    epoch = initial_epoch
    global_batch = batch_size * distributed.world_size * accumulation_steps
    total_parameters, trainable_count = legacy._count_parameters(module)
    legacy._rank_print(
        distributed,
        f"parameters total={total_parameters:,} trainable_lora={trainable_count:,} "
        f"rank={_nested(config, 'model', 'lora', 'rank')} global_batch={global_batch}",
    )

    wandb_run = legacy._initialize_wandb(
        config=config,
        output_dir=output_dir,
        distributed=distributed,
    )
    metrics_path = output_dir / "metrics.jsonl"
    total_steps = int(_nested(config, "training", "steps"))
    log_interval = int(_nested(config, "training", "log_interval"))
    checkpoint_interval = int(_nested(config, "checkpointing", "interval"))
    max_grad_norm = float(_nested(config, "training", "max_grad_norm"))
    _width, _height, _num_frames, fps = _geometry(config)
    optimizer.zero_grad(set_to_none=True)
    interval_start = time.monotonic()
    run_start = interval_start
    last_logged_step = start_step
    if distributed.is_main and not save_checkpoints:
        print("[duel-refiner-train] checkpoint saving disabled", flush=True)

    for step in range(start_step + 1, total_steps + 1):
        accumulated_loss = torch.zeros((), device=distributed.device)
        accumulated_sigma = torch.zeros((), device=distributed.device)
        for accumulation_index in range(accumulation_steps):
            batch, iterator, epoch = _next_batch(iterator, loader, sampler, epoch)
            source_mode = _source_mode(config)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if source_mode in CACHED_SOURCE_MODES:
                    clean_latent = batch["clean_latent"].to(
                        device=distributed.device,
                        dtype=torch.bfloat16,
                        non_blocking=True,
                    )
                    source_latent = batch["source_latent"].to(
                        device=distributed.device,
                        dtype=torch.bfloat16,
                        non_blocking=True,
                    )
                else:
                    if vae_encoder is None:
                        raise RuntimeError("Pixel degradation requires a loaded video VAE encoder")
                    pixels = batch["video_pixels"].to(
                        device=distributed.device,
                        dtype=torch.bfloat16,
                        non_blocking=True,
                    )
                    video = pixels.permute(0, 2, 1, 3, 4).contiguous().mul_(2).sub_(1)
                    clean_latent = vae_encoder(video).detach()
                    source_latent = None
                    if validated["objective_recipe"] == SANA_OBJECTIVE_RECIPE:
                        source_video = _degrade_video_for_source(video, config)
                        source_latent = vae_encoder(source_video).detach()
                        del source_video
                flattened_prompts = [
                    batch[f"prompt_{prompt_index}"][batch_index]
                    for batch_index in range(clean_latent.shape[0])
                    for prompt_index in range(5)
                ]
                encoded_prompts, encoded_masks = prompt_encoder.encode(flattened_prompts)
                prompt_length = encoded_prompts.shape[1]
                prompt_context = encoded_prompts.view(
                    clean_latent.shape[0], 5 * prompt_length, encoded_prompts.shape[-1]
                )
                schedule_mask = encoded_masks.view(clean_latent.shape[0], 5, prompt_length)
                prompt_mask = _build_ar_prompt_context_mask(
                    prompt_attention_mask=schedule_mask,
                    latent_frames=clean_latent.shape[2],
                    spatial_tokens=clean_latent.shape[3] * clean_latent.shape[4],
                )
            positions = legacy._build_video_positions(
                batch_size=clean_latent.shape[0],
                latent_frames=clean_latent.shape[2],
                latent_height=clean_latent.shape[3],
                latent_width=clean_latent.shape[4],
                fps=fps,
                device=distributed.device,
                dtype=clean_latent.dtype,
            )
            sync_context = (
                ddp_model.no_sync()
                if accumulation_index < accumulation_steps - 1
                else contextlib.nullcontext()
            )
            with sync_context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, sigma = ddp_model(
                    clean_latent=clean_latent,
                    source_latent=source_latent,
                    prompt_context=prompt_context,
                    prompt_mask=prompt_mask,
                    positions=positions,
                )
                scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            accumulated_loss += loss.detach() / accumulation_steps
            accumulated_sigma += sigma.float().mean() / accumulation_steps

        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        should_log = step == start_step + 1 or step % log_interval == 0 or step == total_steps
        if should_log:
            reduced = legacy._reduce_metrics(
                torch.stack([accumulated_loss, accumulated_sigma, gradient_norm.detach().float()]),
                distributed,
            )
            now = time.monotonic()
            steps_per_second = (step - last_logged_step) / max(now - interval_start, 1e-6)
            objective_metric_name = (
                "loss_paired_truncated_flow_matching"
                if validated["objective_recipe"] == SANA_OBJECTIVE_RECIPE
                else "loss_velocity_flow_matching"
            )
            record = {
                "step": step,
                objective_metric_name: float(reduced[0]),
                "sigma_mean": float(reduced[1]),
                "grad_norm": float(reduced[2]),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "steps_per_second": steps_per_second,
                "samples_per_second": steps_per_second * global_batch,
                "eta_hours": (total_steps - step) / max(steps_per_second, 1e-9) / 3600.0,
                "epoch_approx": step * global_batch / len(dataset),
                "gpu_allocated_gb": torch.cuda.memory_allocated(distributed.device) / 1024**3,
                "gpu_reserved_gb": torch.cuda.memory_reserved(distributed.device) / 1024**3,
                "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            if distributed.is_main:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(
                    f"[duel-refiner-train] step={step}/{total_steps} "
                    f"loss={record[objective_metric_name]:.6f} "
                    f"sigma={record['sigma_mean']:.4f} grad={record['grad_norm']:.3f} "
                    f"samples/s={record['samples_per_second']:.3f} "
                    f"eta={record['eta_hours']:.2f}h "
                    f"vram={record['gpu_allocated_gb']:.1f}G/{record['gpu_reserved_gb']:.1f}G",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(record, step=step)
            interval_start = now
            last_logged_step = step

        if save_checkpoints and (step % checkpoint_interval == 0 or step == total_steps):
            _save_checkpoint(
                distributed=distributed,
                module=module,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                validated=validated,
                output_dir=output_dir,
                step=step,
                fingerprint=fingerprint,
            )

    if distributed.is_main:
        print(
            f"[duel-refiner-train] complete steps={total_steps} "
            f"wall_time={(time.monotonic() - run_start) / 3600.0:.2f}h",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.finish()
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    args = _parse_args()
    config = legacy._load_config(args.config)
    validated, dataset = _run_preflight(config)
    if args.preflight_only:
        return
    output_value = args.output_dir or Path(str(_nested(config, "output_dir")))
    _train(
        config=config,
        validated=validated,
        dataset=dataset,
        output_dir=_resolved_path(output_value),
        resume_mode=args.resume,
        save_checkpoints=not args.no_save,
    )


if __name__ == "__main__":
    main()
