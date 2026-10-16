                      
                                                     
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
import random
import re
import shutil
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import av
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as torch_f
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
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

from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords              
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps              
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder              
from ltx_core.model.transformer.modality import Modality              
from ltx_core.model.transformer.model_configurator import (              
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
)
from ltx_core.model.upsampler import LatentUpsamplerConfigurator              
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape              
from ltx_trainer.model_loader import (              
    load_embeddings_processor,
    load_text_encoder,
    load_video_vae_encoder,
)

OFFICIAL_STAGE2_SIGMAS = (0.909375, 0.725, 0.421875)
VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()
CHECKPOINT_PATTERN = re.compile(r"refiner_lora_step_(\d+)\.safetensors$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto, none, or an integer checkpoint step. Default: auto.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate paths, checkpoint metadata, LoRA targets, and one decoded sample without loading CUDA models.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Config does not exist: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config root must be a mapping: {resolved}")
    payload["_config_path"] = str(resolved)
    return payload


def _nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            dotted = ".".join(keys)
            raise KeyError(f"Missing required config field: {dotted}")
        value = value[key]
    return value


def _resolved_path(value: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _checkpoint_metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt") as handle:
        return dict(handle.metadata() or {})


def _model_config_from_checkpoint(path: Path) -> dict[str, Any]:
    metadata = _checkpoint_metadata(path)
    raw = metadata.get("config")
    if not raw:
        raise ValueError(f"Checkpoint has no config metadata: {path}")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise TypeError(f"Checkpoint config metadata is not an object: {path}")
    return parsed


def _target_dimensions(config: dict[str, Any]) -> tuple[int, int, int, int, int, float]:
    data_config = _nested(config, "data")
    high_width = int(_nested(data_config, "high_width"))
    high_height = int(_nested(data_config, "high_height"))
    low_width = int(_nested(data_config, "low_width"))
    low_height = int(_nested(data_config, "low_height"))
    num_frames = int(_nested(data_config, "num_frames"))
    fps = float(_nested(data_config, "fps"))
    return high_width, high_height, low_width, low_height, num_frames, fps


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    paths_config = _nested(config, "paths")
    required_paths = {
        "base_model": _resolved_path(_nested(paths_config, "base_model")),
        "distilled_lora": _resolved_path(_nested(paths_config, "distilled_lora")),
        "spatial_upsampler": _resolved_path(_nested(paths_config, "spatial_upsampler")),
        "text_encoder": _resolved_path(_nested(paths_config, "text_encoder")),
        "data_root": _resolved_path(_nested(paths_config, "data_root")),
        "manifest": _resolved_path(_nested(paths_config, "manifest")),
    }
    missing = [f"{name}={path}" for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + "; ".join(missing))

    high_width, high_height, low_width, low_height, num_frames, fps = _target_dimensions(config)
    if high_width != 2 * low_width or high_height != 2 * low_height:
        raise ValueError(
            "The official spatial upsampler is x2, so high dimensions must be exactly "
            f"2x low dimensions; got low={low_width}x{low_height}, high={high_width}x{high_height}"
        )
    if high_width % 64 or high_height % 64:
        raise ValueError(
            f"Two-stage output dimensions must be divisible by 64, got {high_width}x{high_height}"
        )
    if low_width % 32 or low_height % 32:
        raise ValueError(f"Low-resolution dimensions must be divisible by 32, got {low_width}x{low_height}")
    if (num_frames - 1) % 8:
        raise ValueError(f"num_frames must satisfy frames=1+8*k, got {num_frames}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    objective = _nested(config, "objective")
    if bool(objective.get("roi_loss", False)):
        raise ValueError("ROI loss is explicitly disabled for this refiner run.")
    forbidden_roi_fields = [
        key
        for key in objective
        if "roi" in str(key).lower() and key != "roi_loss"
    ]
    if forbidden_roi_fields:
        raise ValueError(f"ROI-related objective fields are not supported: {forbidden_roi_fields}")
    if str(objective.get("upsampler_loss", "")) != "full_frame_latent_mse":
        raise ValueError("objective.upsampler_loss must be full_frame_latent_mse")
    if str(objective.get("stage2_loss", "")) != "anchored_x0_full_frame_mse":
        raise ValueError("objective.stage2_loss must be anchored_x0_full_frame_mse")
    if not bool(objective.get("detach_anchor_for_stage2", True)):
        raise ValueError(
            "detach_anchor_for_stage2=false is intentionally unsupported in v1; "
            "the direct upsampler target and stage-2 target must remain independently identifiable."
        )

    training = _nested(config, "training")
    sigma_values = tuple(float(value) for value in _nested(training, "stage2_sigmas"))
    if sigma_values != OFFICIAL_STAGE2_SIGMAS:
        raise ValueError(
            "training.stage2_sigmas must match the inference schedule exactly: "
            f"{list(OFFICIAL_STAGE2_SIGMAS)}, got {list(sigma_values)}"
        )
    if int(_nested(training, "steps")) <= 0:
        raise ValueError("training.steps must be positive")
    if int(_nested(training, "gradient_accumulation_steps")) <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive")
    if int(_nested(config, "data", "batch_size_per_gpu")) <= 0:
        raise ValueError("data.batch_size_per_gpu must be positive")

    lora = _nested(config, "model", "lora")
    rank = int(_nested(lora, "rank"))
    alpha = int(_nested(lora, "alpha"))
    if rank <= 0 or alpha != rank:
        raise ValueError(
            "The pipeline's LoRA fuser applies B@A directly, so model.lora.alpha must equal rank; "
            f"got rank={rank}, alpha={alpha}"
        )
    targets = list(_nested(lora, "target_modules"))
    if not targets:
        raise ValueError("model.lora.target_modules must not be empty")
    if any("audio" in target for target in targets):
        raise ValueError(f"Refiner LoRA must be video-only; got target_modules={targets}")

    upsampler_config = _model_config_from_checkpoint(required_paths["spatial_upsampler"])
    if float(upsampler_config.get("spatial_scale", 0.0)) != 2.0:
        raise ValueError(f"Expected x2 spatial upsampler checkpoint, got config={upsampler_config}")
    if not bool(upsampler_config.get("spatial_upsample", False)):
        raise ValueError(f"Checkpoint is not a spatial upsampler: {required_paths['spatial_upsampler']}")
    if bool(upsampler_config.get("temporal_upsample", False)):
        raise ValueError("Temporal upsampling is not part of this refiner run.")

    distilled_metadata = _checkpoint_metadata(required_paths["distilled_lora"])
    if distilled_metadata.get("model_version", "").split(".")[:2] != ["2", "3"]:
        raise ValueError(
            "The configured distilled LoRA is not marked as LTX-2.3: "
            f"{required_paths['distilled_lora']}"
        )

    return {**required_paths, "upsampler_config": upsampler_config}


@dataclass(frozen=True)
class RefinerSample:
    sample_id: str
    video_path: Path
    prompt: str
    source_fps: float
    source_frames: int
    video_source_fps: float
    video_offset_sec: float


class CrossFPSRefinerDataset(Dataset[dict[str, Any]]):
    ''

    def __init__(
        self,
        *,
        manifest_path: Path,
        data_root: Path,
        target_width: int,
        target_height: int,
        num_frames: int,
        fps: float,
        random_temporal_crop: bool,
    ) -> None:
        self.manifest_path = manifest_path
        self.data_root = data_root
        self.target_width = target_width
        self.target_height = target_height
        self.num_frames = num_frames
        self.fps = fps
        self.random_temporal_crop = random_temporal_crop
        self.samples = self._read_manifest()
        if not self.samples:
            raise ValueError(f"No usable samples in {self.manifest_path}")

    def _resolve_video_path(self, value: Any) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            manifest_relative = (self.manifest_path.parent / path).resolve()
            if manifest_relative.exists():
                return manifest_relative
            return (self.data_root / path).resolve()
        if path.exists():
            return path

        train_indices = [index for index, part in enumerate(path.parts) if part == "train"]
        candidates: list[Path] = []
        if train_indices:
            train_suffix = Path(*path.parts[train_indices[-1] :])
            candidates.append(self.data_root / train_suffix)
        if len(path.parts) >= 2:
            candidates.append(self.data_root / "train" / path.parent.name / path.name)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return path

    @staticmethod
    def _clean_prompt(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return re.sub(r"\s+", " ", value.replace("\n", " ")).strip()

    def _read_manifest(self) -> list[RefinerSample]:
        samples: list[RefinerSample] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"Manifest line {line_number} is not an object")
                prompt = self._clean_prompt(
                    row.get("prompt_text")
                    or row.get("prompt")
                    or row.get("training_prompt_static_clean")
                    or row.get("training_prompt_static")
                    or row.get("caption")
                )
                video_value = row.get("video_path")
                if not prompt or not video_value:
                    continue
                source_fps = float(
                    row.get("source_fps")
                    or row.get("standard_action_hz")
                    or row.get("fps")
                    or self.fps
                )
                if abs(source_fps - self.fps) > 1e-3:
                    raise ValueError(
                        f"Manifest line {line_number} has source_fps={source_fps}, expected {self.fps}; "
                        "integer frame alignment is required."
                    )
                source_frames = int(
                    row.get("source_frames")
                    or row.get("segment_action_rows")
                    or row.get("chunk_size")
                    or 0
                )
                source_duration = row.get("source_duration") or row.get("duration_sec")
                if source_frames <= 0 and source_duration is not None:
                    source_frames = round(float(source_duration) * source_fps)
                if source_frames < self.num_frames:
                    continue
                video_source_fps = float(row.get("video_source_fps") or row.get("video_fps") or source_fps)
                sample_id = str(
                    row.get("sample_id")
                    or row.get("clip_id")
                    or Path(str(video_value)).parent.name
                    or line_number
                )
                samples.append(
                    RefinerSample(
                        sample_id=sample_id,
                        video_path=self._resolve_video_path(video_value),
                        prompt=prompt,
                        source_fps=source_fps,
                        source_frames=source_frames,
                        video_source_fps=video_source_fps,
                        video_offset_sec=float(row.get("video_offset_sec") or 0.0),
                    )
                )
        return samples

    def _pick_indices(self, sample: RefinerSample) -> tuple[int, list[int]]:
        max_start = sample.source_frames - self.num_frames
        crop_start = (
            random.randint(0, max_start)
            if self.random_temporal_crop and max_start > 0
            else 0
        )
        action_indices = torch.arange(crop_start, crop_start + self.num_frames, dtype=torch.float64)
        if (
            abs(sample.video_source_fps - sample.source_fps) > 1e-3
            or abs(sample.video_offset_sec) > 1e-9
        ):
            video_indices = (
                (action_indices / sample.source_fps + sample.video_offset_sec)
                * sample.video_source_fps
            ).round()
        else:
            video_indices = action_indices
        return crop_start, [int(value) for value in video_indices.tolist()]

    def _resize_and_center_crop(self, frames: Tensor) -> Tensor:
        _, _, source_height, source_width = frames.shape
        scale = max(self.target_height / source_height, self.target_width / source_width)
        new_height = math.ceil(source_height * scale)
        new_width = math.ceil(source_width * scale)
        resized = torch_f.interpolate(
            frames,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
        top = (new_height - self.target_height) // 2
        left = (new_width - self.target_width) // 2
        return resized[:, :, top : top + self.target_height, left : left + self.target_width].contiguous()

    def _decode_video(self, sample: RefinerSample, indices: list[int]) -> Tensor:
        if not sample.video_path.is_file():
            raise FileNotFoundError(
                f"CrossFPS video is missing for sample {sample.sample_id}: {sample.video_path}"
            )
        target_map: dict[int, list[int]] = {}
        for output_index, source_index in enumerate(indices):
            target_map.setdefault(source_index, []).append(output_index)
        first_needed = min(target_map)
        last_needed = max(target_map)
        decoded: list[Tensor | None] = [None] * len(indices)

        with av.open(str(sample.video_path)) as container:
            fallback_index = 0
            for frame in container.decode(video=0):
                if frame.time is not None and sample.video_source_fps > 0:
                    source_index = round(float(frame.time) * sample.video_source_fps)
                else:
                    source_index = fallback_index
                fallback_index += 1
                if source_index < first_needed:
                    continue
                if source_index > last_needed:
                    break
                if source_index not in target_map:
                    continue
                array = frame.to_ndarray(format="rgb24")
                frame_tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
                for output_index in target_map[source_index]:
                    decoded[output_index] = frame_tensor

        first_valid = next((frame for frame in decoded if frame is not None), None)
        if first_valid is None:
            raise RuntimeError(
                f"Could not decode requested frames {first_needed}:{last_needed} from {sample.video_path}"
            )
        for index, frame in enumerate(decoded):
            if frame is None:
                decoded[index] = decoded[index - 1] if index > 0 and decoded[index - 1] is not None else first_valid
        stacked = torch.stack([frame for frame in decoded if frame is not None], dim=0)
        if stacked.shape[0] != self.num_frames:
            raise RuntimeError(
                f"Decoded {stacked.shape[0]} frames for {sample.sample_id}, expected {self.num_frames}"
            )
        return self._resize_and_center_crop(stacked)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        crop_start, indices = self._pick_indices(sample)
        return {
            "video_pixels": self._decode_video(sample, indices),
            "prompt_text": sample.prompt,
            "sample_id": sample.sample_id,
            "crop_start": crop_start,
        }


def _build_dataset(config: dict[str, Any], validated: dict[str, Any]) -> CrossFPSRefinerDataset:
    high_width, high_height, _low_width, _low_height, num_frames, fps = _target_dimensions(config)
    return CrossFPSRefinerDataset(
        manifest_path=validated["manifest"],
        data_root=validated["data_root"],
        target_width=high_width,
        target_height=high_height,
        num_frames=num_frames,
        fps=fps,
        random_temporal_crop=bool(_nested(config, "data", "random_temporal_crop")),
    )


def _module_matches_target(name: str, target: str) -> bool:
    return name == target or name.endswith(f".{target}")


def _validate_lora_targets(config: dict[str, Any], base_model: Path) -> dict[str, int]:
    model_config = _model_config_from_checkpoint(base_model)
    with torch.device("meta"):
        model = LTXModelConfigurator.from_config(model_config)
    module_names = [name for name, _module in model.named_modules()]
    counts: dict[str, int] = {}
    for target in _nested(config, "model", "lora", "target_modules"):
        matches = [name for name in module_names if _module_matches_target(name, target)]
        if not matches:
            raise ValueError(f"LoRA target matches zero modules: {target}")
        audio_matches = [name for name in matches if "audio" in name or "video_to_audio" in name]
        if audio_matches:
            raise ValueError(
                f"Video-only LoRA target {target!r} unexpectedly matches audio modules: {audio_matches[:4]}"
            )
        counts[str(target)] = len(matches)
    del model
    return counts


def _objective_self_test() -> None:
    generator = torch.Generator(device="cpu").manual_seed(7)
    clean = torch.randn(2, 5, 3, generator=generator)
    anchor = torch.randn(2, 5, 3, generator=generator)
    noise = torch.randn(2, 5, 3, generator=generator)
    sigma = torch.tensor([OFFICIAL_STAGE2_SIGMAS[0], OFFICIAL_STAGE2_SIGMAS[-1]]).view(-1, 1, 1)
    noisy = (1.0 - sigma) * anchor + sigma * noise
    exact_velocity = (noisy - clean) / sigma
    reconstructed = noisy - sigma * exact_velocity
    if not torch.allclose(reconstructed, clean, atol=1e-6, rtol=1e-6):
        raise AssertionError("Anchored x0 objective self-test failed")


def _run_preflight(config: dict[str, Any]) -> tuple[dict[str, Any], CrossFPSRefinerDataset]:
    validated = _validate_config(config)
    target_counts = _validate_lora_targets(config, validated["base_model"])
    dataset = _build_dataset(config, validated)
    probe_count = min(int(_nested(config, "data", "preflight_samples")), len(dataset))
    if probe_count <= 0:
        raise ValueError("data.preflight_samples must be positive")
    probe_indices = np.linspace(0, len(dataset) - 1, num=probe_count, dtype=np.int64).tolist()
    missing_probe_paths = [
        str(dataset.samples[index].video_path)
        for index in probe_indices
        if not dataset.samples[index].video_path.is_file()
    ]
    if missing_probe_paths:
        raise FileNotFoundError(
            "Preflight sample video path(s) are missing: " + "; ".join(missing_probe_paths[:8])
        )
    decoded = dataset[probe_indices[0]]
    high_width, high_height, _low_width, _low_height, num_frames, _fps = _target_dimensions(config)
    expected_shape = (num_frames, 3, high_height, high_width)
    if tuple(decoded["video_pixels"].shape) != expected_shape:
        raise ValueError(
            f"Decoded probe shape mismatch: got {tuple(decoded['video_pixels'].shape)}, expected {expected_shape}"
        )
    if not torch.isfinite(decoded["video_pixels"]).all():
        raise ValueError("Decoded probe contains non-finite pixels")
    _objective_self_test()

    print("[refiner-preflight] PASS", flush=True)
    print(f"[refiner-preflight] manifest={validated['manifest']}", flush=True)
    print(f"[refiner-preflight] usable_samples={len(dataset):,}", flush=True)
    print(
        f"[refiner-preflight] decoded={decoded['sample_id']} shape={tuple(decoded['video_pixels'].shape)} "
        f"range=({decoded['video_pixels'].min().item():.4f},{decoded['video_pixels'].max().item():.4f})",
        flush=True,
    )
    print(f"[refiner-preflight] lora_target_counts={target_counts}", flush=True)
    print("[refiner-preflight] roi_loss=false full_frame_only=true", flush=True)
    return validated, dataset


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _init_distributed() -> DistributedContext:
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "This training entrypoint must be launched with torchrun; missing " + ", ".join(missing)
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for refiner training")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def _rank_print(distributed: DistributedContext, message: str, *, all_ranks: bool = False) -> None:
    if distributed.is_main or all_ranks:
        print(f"[refiner-train rank={distributed.rank}] {message}", flush=True)


def _seed_model_initialization(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_data(seed: int, rank: int) -> None:
    rank_seed = seed + 1009 * rank
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed((worker_seed + worker_id) % (2**32))


class OnlinePromptEncoder:
    def __init__(
        self,
        *,
        text_encoder: Any,
        embeddings_processor: nn.Module,
        device: torch.device,
        cache_size: int,
    ) -> None:
        self.text_encoder = text_encoder
        self.embeddings_processor = embeddings_processor
        self.device = device
        self.cache_size = max(0, cache_size)
        self.cache: OrderedDict[str, tuple[Tensor, Tensor]] = OrderedDict()

    def _encode_one(self, prompt: str) -> tuple[Tensor, Tensor]:
        cached = self.cache.get(prompt)
        if cached is not None:
            self.cache.move_to_end(prompt)
            return cached[0].to(self.device), cached[1].to(self.device)

        with torch.no_grad():
            hidden_states, attention_mask = self.text_encoder.encode(prompt)
            processed = self.embeddings_processor.process_hidden_states(hidden_states, attention_mask)
            video_encoding = processed.video_encoding.squeeze(0).to(dtype=torch.bfloat16)
            binary_mask = processed.attention_mask.squeeze(0).to(dtype=torch.bool)

        if self.cache_size > 0:
            self.cache[prompt] = (
                video_encoding.detach().cpu(),
                binary_mask.detach().cpu(),
            )
            self.cache.move_to_end(prompt)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return video_encoding, binary_mask

    def encode(self, prompts: list[str]) -> tuple[Tensor, Tensor]:
        encodings: list[Tensor] = []
        masks: list[Tensor] = []
        for prompt in prompts:
            encoding, mask = self._encode_one(prompt)
            encodings.append(encoding)
            masks.append(mask)
        return (
            pad_sequence(encodings, batch_first=True).to(device=self.device, dtype=torch.bfloat16),
            pad_sequence(masks, batch_first=True, padding_value=False).to(device=self.device),
        )


def _build_video_positions(
    *,
    batch_size: int,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    fps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    patchifier = VideoLatentPatchifier(patch_size=1)
    latent_coords = patchifier.get_patch_grid_bounds(
        output_shape=VideoLatentShape(
            batch=batch_size,
            channels=128,
            frames=latent_frames,
            height=latent_height,
            width=latent_width,
        ),
        device=device,
    )
    positions = get_pixel_coords(
        latent_coords=latent_coords,
        scale_factors=VIDEO_SCALE_FACTORS,
        causal_fix=True,
    ).to(dtype)
    positions[:, 0, ...] = positions[:, 0, ...] / fps
    return positions


class RefinerTrainingModule(nn.Module):
    def __init__(
        self,
        *,
        transformer: nn.Module,
        upsampler: nn.Module,
        latent_mean: Tensor,
        latent_std: Tensor,
        sigma_values: tuple[float, ...],
        first_frame_conditioning: bool,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.upsampler = upsampler
        self.register_buffer(
            "latent_mean",
            latent_mean.detach().float().view(1, -1, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latent_std",
            latent_std.detach().float().view(1, -1, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "sigma_values",
            torch.tensor(sigma_values, dtype=torch.float32),
            persistent=False,
        )
        self.first_frame_conditioning = first_frame_conditioning
        self.patchifier = VideoLatentPatchifier(patch_size=1)

    def _normalize(self, latent: Tensor) -> Tensor:
        return (latent - self.latent_mean.to(latent)) / self.latent_std.to(latent)

    def forward(
        self,
        *,
        low_latent_unnormalized: Tensor,
        high_latent: Tensor,
        prompt_context: Tensor,
        prompt_mask: Tensor,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        upsampled = self._normalize(self.upsampler(low_latent_unnormalized))
        if tuple(upsampled.shape) != tuple(high_latent.shape):
            raise RuntimeError(
                f"Upsampler output shape {tuple(upsampled.shape)} does not match target {tuple(high_latent.shape)}"
            )

        upsampler_loss = torch_f.mse_loss(upsampled.float(), high_latent.float())
        clean_tokens = self.patchifier.patchify(high_latent)
                                                                               
                                                                            
        anchor_tokens = self.patchifier.patchify(upsampled.detach())
        batch_size, sequence_length, _channels = clean_tokens.shape

        sigma_indices = torch.randint(
            low=0,
            high=self.sigma_values.numel(),
            size=(batch_size,),
            device=high_latent.device,
        )
        sigma = self.sigma_values[sigma_indices].to(device=high_latent.device)
        sigma_expanded = sigma.view(-1, 1, 1).to(dtype=clean_tokens.dtype)
        noise = torch.randn_like(clean_tokens)
        noisy_tokens = (1.0 - sigma_expanded) * anchor_tokens + sigma_expanded * noise

        loss_mask = torch.ones(
            batch_size,
            sequence_length,
            device=high_latent.device,
            dtype=torch.bool,
        )
        timesteps = sigma.view(-1, 1).expand(-1, sequence_length).clone()
        if self.first_frame_conditioning:
            first_frame_tokens = high_latent.shape[-2] * high_latent.shape[-1]
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
        )
        velocity_prediction, _audio_prediction = self.transformer(
            video=video,
            audio=None,
            perturbations=None,
        )
        if velocity_prediction is None:
            raise RuntimeError("Stage-2 transformer returned no video prediction")

        predicted_x0 = noisy_tokens - timesteps.unsqueeze(-1).to(noisy_tokens) * velocity_prediction
        squared_error = (predicted_x0.float() - clean_tokens.float()).pow(2)
        expanded_mask = loss_mask.unsqueeze(-1).expand_as(squared_error)
        stage2_loss = squared_error.masked_select(expanded_mask).mean()
        return upsampler_loss, stage2_loss, sigma.detach()


def _load_trainable_module(
    *,
    config: dict[str, Any],
    validated: dict[str, Any],
    device: torch.device,
) -> tuple[RefinerTrainingModule, nn.Module, OnlinePromptEncoder]:
    from peft import LoraConfig, get_peft_model

    _rank = int(os.environ.get("RANK", "0"))
    print(f"[refiner-load rank={_rank}] loading frozen video VAE encoder", flush=True)
    vae_encoder = load_video_vae_encoder(
        validated["base_model"],
        device=device,
        dtype=torch.bfloat16,
    ).eval()
    vae_encoder.requires_grad_(False)

    print(f"[refiner-load rank={_rank}] loading 8-bit Gemma and embeddings processor", flush=True)
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

    print(f"[refiner-load rank={_rank}] loading base transformer and fusing official distilled LoRA", flush=True)
    transformer = SingleGPUModelBuilder(
        model_path=str(validated["base_model"]),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        loras=(
            LoraPathStrengthAndSDOps(
                str(validated["distilled_lora"]),
                1.0,
                LTXV_LORA_COMFY_RENAMING_MAP,
            ),
        ),
    ).build(device=device, dtype=torch.bfloat16)
    transformer.requires_grad_(False)
    transformer.set_gradient_checkpointing(bool(_nested(config, "model", "gradient_checkpointing")))

    lora_config_values = _nested(config, "model", "lora")
    lora_config = LoraConfig(
        r=int(_nested(lora_config_values, "rank")),
        lora_alpha=int(_nested(lora_config_values, "alpha")),
        target_modules=list(_nested(lora_config_values, "target_modules")),
        lora_dropout=float(_nested(lora_config_values, "dropout")),
        init_lora_weights=True,
    )
    transformer = get_peft_model(transformer, lora_config)

    print(f"[refiner-load rank={_rank}] loading trainable spatial upsampler in fp32", flush=True)
    upsampler = SingleGPUModelBuilder(
        model_path=str(validated["spatial_upsampler"]),
        model_class_configurator=LatentUpsamplerConfigurator,
    ).build(device=device, dtype=torch.float32)
    upsampler.requires_grad_(True)

    latent_mean = vae_encoder.per_channel_statistics.get_buffer("mean-of-means")
    latent_std = vae_encoder.per_channel_statistics.get_buffer("std-of-means")
    module = RefinerTrainingModule(
        transformer=transformer,
        upsampler=upsampler,
        latent_mean=latent_mean,
        latent_std=latent_std,
        sigma_values=tuple(float(value) for value in _nested(config, "training", "stage2_sigmas")),
        first_frame_conditioning=bool(_nested(config, "training", "first_frame_conditioning")),
    ).to(device)

    prompt_encoder = OnlinePromptEncoder(
        text_encoder=text_encoder,
        embeddings_processor=embeddings_processor,
        device=device,
        cache_size=int(_nested(config, "data", "prompt_cache_size")),
    )
    return module, vae_encoder, prompt_encoder


def _parameter_groups(
    module: RefinerTrainingModule,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[Tensor]]:
    upsampler_parameters = [parameter for parameter in module.upsampler.parameters() if parameter.requires_grad]
    lora_parameters = [
        parameter
        for name, parameter in module.transformer.named_parameters()
        if parameter.requires_grad and "lora_" in name.lower()
    ]
    unexpected_transformer_parameters = [
        name
        for name, parameter in module.transformer.named_parameters()
        if parameter.requires_grad and "lora_" not in name.lower()
    ]
    if unexpected_transformer_parameters:
        raise RuntimeError(
            "Non-LoRA transformer parameters are unexpectedly trainable: "
            f"{unexpected_transformer_parameters[:12]}"
        )
    if not upsampler_parameters or not lora_parameters:
        raise RuntimeError(
            f"Missing trainable parameters: upsampler={len(upsampler_parameters)}, lora={len(lora_parameters)}"
        )
    groups = [
        {
            "params": upsampler_parameters,
            "lr": float(_nested(config, "training", "upsampler_learning_rate")),
            "name": "upsampler",
        },
        {
            "params": lora_parameters,
            "lr": float(_nested(config, "training", "stage2_lora_learning_rate")),
            "name": "stage2_lora",
        },
    ]
    return groups, [*upsampler_parameters, *lora_parameters]


def _build_optimizer(parameter_groups: list[dict[str, Any]], config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = str(_nested(config, "training", "optimizer")).lower()
    kwargs = {
        "betas": tuple(float(value) for value in _nested(config, "training", "betas")),
        "eps": float(_nested(config, "training", "epsilon")),
        "weight_decay": float(_nested(config, "training", "weight_decay")),
    }
    if optimizer_name == "adamw8bit":
        from bitsandbytes.optim import AdamW8bit

        return AdamW8bit(parameter_groups, **kwargs)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameter_groups, **kwargs)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = int(_nested(config, "training", "steps"))
    warmup_steps = int(_nested(config, "training", "warmup_steps"))
    min_ratio = float(_nested(config, "training", "min_lr_ratio"))

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _parse_checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class ResumeBundle:
    step: int
    lora_path: Path
    upsampler_path: Path
    training_state_path: Path | None
    manifest_path: Path | None


def _bundle_for_step(output_dir: Path, step: int) -> ResumeBundle | None:
    checkpoint_dir = output_dir / "checkpoints"
    lora_path = checkpoint_dir / f"refiner_lora_step_{step:06d}.safetensors"
    upsampler_path = checkpoint_dir / f"spatial_upsampler_step_{step:06d}.safetensors"
    training_state_path = checkpoint_dir / f"training_state_step_{step:06d}.pt"
    manifest_path = checkpoint_dir / f"refiner_bundle_step_{step:06d}.json"
    if not lora_path.is_file() or not upsampler_path.is_file():
        return None
    return ResumeBundle(
        step=step,
        lora_path=lora_path,
        upsampler_path=upsampler_path,
        training_state_path=training_state_path if training_state_path.is_file() else None,
        manifest_path=manifest_path if manifest_path.is_file() else None,
    )


def _resolve_resume(output_dir: Path, resume: str) -> ResumeBundle | None:
    normalized = resume.strip().lower()
    checkpoint_dir = output_dir / "checkpoints"
    candidates: list[int] = []
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.glob("refiner_lora_step_*.safetensors"):
            step = _parse_checkpoint_step(path)
            if step is not None and _bundle_for_step(output_dir, step) is not None:
                candidates.append(step)

    if normalized == "none":
        if candidates:
            raise FileExistsError(
                f"Complete refiner checkpoints already exist under {checkpoint_dir}; "
                "use --resume auto or choose a new output directory."
            )
        return None
    if normalized == "auto":
        return _bundle_for_step(output_dir, max(candidates)) if candidates else None
    try:
        requested_step = int(normalized)
    except ValueError as exc:
        raise ValueError(f"--resume must be auto, none, or an integer step; got {resume!r}") from exc
    bundle = _bundle_for_step(output_dir, requested_step)
    if bundle is None:
        raise FileNotFoundError(
            f"No complete refiner checkpoint bundle for step {requested_step} under {checkpoint_dir}"
        )
    return bundle


def _load_resume_weights(module: RefinerTrainingModule, bundle: ResumeBundle) -> None:
    from peft import set_peft_model_state_dict

    lora_state = load_file(bundle.lora_path)
    lora_state = {
        key.replace("diffusion_model.", "", 1): value
        for key, value in lora_state.items()
    }
    set_peft_model_state_dict(module.transformer.get_base_model(), lora_state)
    upsampler_state = load_file(bundle.upsampler_path)
    incompatible = module.upsampler.load_state_dict(upsampler_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Upsampler resume mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def _state_fingerprint(config: dict[str, Any], validated: dict[str, Any]) -> str:
    important = {
        "base_model": str(validated["base_model"]),
        "distilled_lora": str(validated["distilled_lora"]),
        "spatial_upsampler": str(validated["spatial_upsampler"]),
        "manifest": str(validated["manifest"]),
        "data": _nested(config, "data"),
        "objective": _nested(config, "objective"),
        "lora": _nested(config, "model", "lora"),
        "sigmas": _nested(config, "training", "stage2_sigmas"),
    }
    encoded = json.dumps(important, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_training_state(
    *,
    bundle: ResumeBundle,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_fingerprint: str,
    device: torch.device,
) -> None:
    if bundle.training_state_path is None:
        print(
            f"[refiner-resume] step={bundle.step} has weights but no optimizer state; "
            "continuing with a fresh optimizer/scheduler.",
            flush=True,
        )
        for _ in range(bundle.step):
            scheduler.step()
        return
    state = torch.load(bundle.training_state_path, map_location=device, weights_only=False)
    if str(state.get("fingerprint")) != expected_fingerprint:
        raise ValueError(
            f"Resume fingerprint mismatch in {bundle.training_state_path}; "
            "refusing to mix a different refiner recipe."
        )
    if int(state.get("step", -1)) != bundle.step:
        raise ValueError(
            f"Training-state step mismatch: bundle={bundle.step}, state={state.get('step')}"
        )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if "torch_rng_state" in state:
        torch.random.set_rng_state(state["torch_rng_state"].cpu())
    if "cuda_rng_state" in state:
        torch.cuda.set_rng_state(state["cuda_rng_state"].to(device="cpu"), device=device)


def _atomic_save_safetensors(
    state_dict: dict[str, Tensor],
    destination: Path,
    metadata: dict[str, str],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    save_file(state_dict, str(temporary), metadata=metadata)
    temporary.replace(destination)


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    torch.save(payload, temporary)
    temporary.replace(destination)


def _save_checkpoint(
    *,
    distributed: DistributedContext,
    module: RefinerTrainingModule,
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
        checkpoint_dir = output_dir / "checkpoints"
        lora_path = checkpoint_dir / f"refiner_lora_step_{step:06d}.safetensors"
        upsampler_path = checkpoint_dir / f"spatial_upsampler_step_{step:06d}.safetensors"
        training_state_path = checkpoint_dir / f"training_state_step_{step:06d}.pt"
        manifest_path = checkpoint_dir / f"refiner_bundle_step_{step:06d}.json"
        for path in (lora_path, upsampler_path, manifest_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint artifact: {path}")

        from peft import get_peft_model_state_dict

        raw_lora_state = get_peft_model_state_dict(module.transformer)
        lora_state = {
            f"diffusion_model.{key.replace('base_model.model.', '', 1)}": value.detach()
            .to(device="cpu", dtype=torch.bfloat16)
            .contiguous()
            for key, value in raw_lora_state.items()
        }
        lora_config = _nested(config, "model", "lora")
        common_metadata = {
            "training_kind": "crossfps_stage2_refiner",
            "training_step": str(step),
            "base_model": str(validated["base_model"]),
            "distilled_lora": str(validated["distilled_lora"]),
            "manifest": str(validated["manifest"]),
            "objective": "anchored_x0_full_frame_mse",
            "roi_loss": "false",
            "fingerprint": fingerprint,
        }
        _atomic_save_safetensors(
            lora_state,
            lora_path,
            metadata={
                **common_metadata,
                "lora_rank": str(_nested(lora_config, "rank")),
                "lora_alpha": str(_nested(lora_config, "alpha")),
                "model_version": "2.3.0",
            },
        )
        del lora_state, raw_lora_state

        upsampler_state = {
            key: value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            for key, value in module.upsampler.state_dict().items()
        }
        _atomic_save_safetensors(
            upsampler_state,
            upsampler_path,
            metadata={
                **common_metadata,
                "config": json.dumps(validated["upsampler_config"], separators=(",", ":")),
            },
        )
        del upsampler_state

        if bool(_nested(config, "checkpointing", "save_optimizer_state")):
            _atomic_torch_save(
                {
                    "step": step,
                    "fingerprint": fingerprint,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "torch_rng_state": torch.random.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(distributed.device),
                },
                training_state_path,
            )

        bundle_manifest = {
            "step": step,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "base_model": str(validated["base_model"]),
            "official_distilled_lora": str(validated["distilled_lora"]),
            "refiner_lora": str(lora_path),
            "spatial_upsampler": str(upsampler_path),
            "stage2_sigmas": [*OFFICIAL_STAGE2_SIGMAS, 0.0],
            "inference_lora_order": [
                str(validated["distilled_lora"]),
                str(lora_path),
            ],
            "roi_loss": False,
            "fingerprint": fingerprint,
        }
        temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
        temporary_manifest.write_text(
            json.dumps(bundle_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)

        keep_states = int(_nested(config, "checkpointing", "keep_optimizer_states"))
        if keep_states >= 0:
            state_paths = sorted(
                checkpoint_dir.glob("training_state_step_*.pt"),
                key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
            )
            for old_path in state_paths[: max(0, len(state_paths) - keep_states)]:
                old_path.unlink()

        print(
            f"[refiner-checkpoint] step={step} lora={lora_path} upsampler={upsampler_path} "
            f"training_state={training_state_path if training_state_path.exists() else 'disabled'}",
            flush=True,
        )
    dist.barrier()


def _verify_trainable_initialization(module: RefinerTrainingModule, distributed: DistributedContext) -> None:
    values: list[Tensor] = []
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    for parameter in (trainable[:2] + trainable[-2:]):
        values.append(parameter.detach().reshape(-1)[:32].float().sum())
    checksum = torch.stack(values).sum().to(device=distributed.device, dtype=torch.float64)
    minimum = checksum.clone()
    maximum = checksum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if not torch.allclose(minimum, maximum, atol=1e-8, rtol=1e-8):
        raise RuntimeError(
            f"Trainable initialization differs across ranks: min={minimum.item()}, max={maximum.item()}"
        )


def _downsample_video(video: Tensor, *, height: int, width: int) -> Tensor:
    batch, channels, frames, source_height, source_width = video.shape
    flattened = video.permute(0, 2, 1, 3, 4).reshape(
        batch * frames,
        channels,
        source_height,
        source_width,
    )
    resized = torch_f.interpolate(
        flattened,
        size=(height, width),
        mode="area",
    )
    return resized.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()


def _next_batch(
    iterator: Iterator[dict[str, Any]],
    loader: DataLoader[dict[str, Any]],
    sampler: DistributedSampler,
    epoch: int,
) -> tuple[dict[str, Any], Iterator[dict[str, Any]], int]:
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


def _reduce_metrics(values: Tensor, distributed: DistributedContext) -> Tensor:
    reduced = values.detach().float().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / distributed.world_size


def _count_parameters(module: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total, trainable


def _initialize_wandb(
    *,
    config: dict[str, Any],
    output_dir: Path,
    distributed: DistributedContext,
) -> Any | None:
    wandb_config = config.get("wandb") or {}
    if not distributed.is_main or not bool(wandb_config.get("enabled", False)):
        return None
    try:
        import wandb

        return wandb.init(
            project=str(wandb_config.get("project") or "ltx2"),
            entity=wandb_config.get("entity"),
            name=str(config.get("run_name") or output_dir.name),
            tags=list(wandb_config.get("tags") or []),
            config={key: value for key, value in config.items() if not key.startswith("_")},
            dir=str(output_dir),
            mode=os.environ.get("WANDB_MODE", "offline"),
            resume="allow",
        )
    except Exception as exc:
        print(f"[refiner-wandb] disabled after initialization error: {exc}", flush=True)
        return None


def _write_run_snapshot(
    *,
    output_dir: Path,
    config: dict[str, Any],
    validated: dict[str, Any],
    distributed: DistributedContext,
    fingerprint: str,
) -> None:
    if not distributed.is_main:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "training_config.yaml"
    clean_config = {key: value for key, value in config.items() if not key.startswith("_")}
    if not snapshot_path.exists():
        snapshot_path.write_text(
            yaml.safe_dump(clean_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    source_snapshot = output_dir / "rigen_foundation_model.py"
    if not source_snapshot.exists():
        shutil.copy2(Path(__file__).resolve(), source_snapshot)
    run_info = {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "world_size": distributed.world_size,
        "hostname": os.uname().nodename,
        "base_model": str(validated["base_model"]),
        "distilled_lora": str(validated["distilled_lora"]),
        "spatial_upsampler": str(validated["spatial_upsampler"]),
        "manifest": str(validated["manifest"]),
        "fingerprint": fingerprint,
        "roi_loss": False,
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _train(
    *,
    config: dict[str, Any],
    validated: dict[str, Any],
    dataset: CrossFPSRefinerDataset,
    output_dir: Path,
    resume_mode: str,
) -> None:
    distributed = _init_distributed()
    expected_world_size = int(_nested(config, "distributed", "expected_world_size"))
    if distributed.world_size != expected_world_size:
        raise RuntimeError(
            f"This config expects world_size={expected_world_size}, got {distributed.world_size}"
        )

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed = int(_nested(config, "training", "seed"))
    _seed_model_initialization(seed)

    fingerprint = _state_fingerprint(config, validated)
    resume_bundle = _resolve_resume(output_dir, resume_mode)
    _write_run_snapshot(
        output_dir=output_dir,
        config=config,
        validated=validated,
        distributed=distributed,
        fingerprint=fingerprint,
    )
    dist.barrier()

    _rank_print(
        distributed,
        f"topology={distributed.world_size} GPUs device={distributed.device} "
        f"dataset={len(dataset):,} output={output_dir}",
    )
    if resume_bundle is not None:
        _rank_print(distributed, f"resuming weights from step={resume_bundle.step}")

    module, vae_encoder, prompt_encoder = _load_trainable_module(
        config=config,
        validated=validated,
        device=distributed.device,
    )
    if resume_bundle is not None:
        _load_resume_weights(module, resume_bundle)
    _verify_trainable_initialization(module, distributed)

    parameter_groups, trainable_parameters = _parameter_groups(module, config)
    optimizer = _build_optimizer(parameter_groups, config)
    scheduler = _build_scheduler(optimizer, config)
    if resume_bundle is not None:
        _load_training_state(
            bundle=resume_bundle,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_fingerprint=fingerprint,
            device=distributed.device,
        )
    start_step = resume_bundle.step if resume_bundle is not None else 0

    ddp_signature = inspect.signature(DistributedDataParallel)
    ddp_kwargs: dict[str, Any] = {
        "device_ids": [distributed.local_rank],
        "output_device": distributed.local_rank,
        "broadcast_buffers": False,
        "find_unused_parameters": False,
        "gradient_as_bucket_view": True,
        "static_graph": True,
        "bucket_cap_mb": int(_nested(config, "distributed", "ddp_bucket_cap_mb")),
    }
                                                                            
                                                                              
                                                        
    if "init_sync" in ddp_signature.parameters:
        ddp_kwargs["init_sync"] = False
    ddp_model = DistributedDataParallel(module, **ddp_kwargs)
    ddp_model.train()

    _seed_data(seed, distributed.rank)
    batch_size = int(_nested(config, "data", "batch_size_per_gpu"))
    sampler = DistributedSampler(
        dataset,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    initial_epoch = (
        start_step
        * int(_nested(config, "training", "gradient_accumulation_steps"))
        * batch_size
        * distributed.world_size
    ) // max(1, len(dataset))
    sampler.set_epoch(initial_epoch)
    loader_generator = torch.Generator().manual_seed(seed + distributed.rank)
    num_workers = int(_nested(config, "data", "num_workers"))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        generator=loader_generator,
        prefetch_factor=int(_nested(config, "data", "prefetch_factor")) if num_workers > 0 else None,
    )
    iterator = iter(loader)
    epoch = initial_epoch

    total_parameters, trainable_count = _count_parameters(module)
    lora_count = sum(
        parameter.numel()
        for name, parameter in module.transformer.named_parameters()
        if parameter.requires_grad and "lora_" in name.lower()
    )
    upsampler_count = sum(parameter.numel() for parameter in module.upsampler.parameters())
    global_batch = (
        batch_size
        * distributed.world_size
        * int(_nested(config, "training", "gradient_accumulation_steps"))
    )
    _rank_print(
        distributed,
        f"parameters total={total_parameters:,} trainable={trainable_count:,} "
        f"upsampler={upsampler_count:,} stage2_lora={lora_count:,}",
    )
    _rank_print(
        distributed,
        f"batch per_gpu={batch_size} global={global_batch} start_step={start_step} "
        f"target_steps={_nested(config, 'training', 'steps')}",
    )

    metrics_path = output_dir / "metrics.jsonl"
    wandb_run = _initialize_wandb(
        config=config,
        output_dir=output_dir,
        distributed=distributed,
    )
    training = _nested(config, "training")
    objective = _nested(config, "objective")
    total_steps = int(_nested(training, "steps"))
    accumulation_steps = int(_nested(training, "gradient_accumulation_steps"))
    log_interval = int(_nested(training, "log_interval"))
    checkpoint_interval = int(_nested(config, "checkpointing", "interval"))
    max_grad_norm = float(_nested(training, "max_grad_norm"))
    upsampler_weight = float(_nested(objective, "upsampler_loss_weight"))
    stage2_weight = float(_nested(objective, "stage2_loss_weight"))
    _high_width, _high_height, low_width, low_height, _num_frames, fps = _target_dimensions(config)

    optimizer.zero_grad(set_to_none=True)
    interval_start = time.monotonic()
    run_start = interval_start
    last_logged_step = start_step
    for step in range(start_step + 1, total_steps + 1):
        accumulated_up = torch.zeros((), device=distributed.device)
        accumulated_stage2 = torch.zeros((), device=distributed.device)
        accumulated_sigma = torch.zeros((), device=distributed.device)

        for accumulation_index in range(accumulation_steps):
            batch, iterator, epoch = _next_batch(iterator, loader, sampler, epoch)
            pixels = batch["video_pixels"].to(
                device=distributed.device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )
            if pixels.ndim != 5 or pixels.shape[2] != 3:
                raise ValueError(f"Expected video_pixels [B,F,3,H,W], got {tuple(pixels.shape)}")
            high_video = pixels.permute(0, 2, 1, 3, 4).contiguous().mul_(2).sub_(1)
            low_video = _downsample_video(high_video, height=low_height, width=low_width)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                high_latent = vae_encoder(high_video).detach()
                low_latent = vae_encoder(low_video).detach()
                low_latent_unnormalized = (
                    low_latent
                    * vae_encoder.per_channel_statistics.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(low_latent)
                    + vae_encoder.per_channel_statistics.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(low_latent)
                )
                prompt_context, prompt_mask = prompt_encoder.encode(list(batch["prompt_text"]))

            positions = _build_video_positions(
                batch_size=high_latent.shape[0],
                latent_frames=high_latent.shape[2],
                latent_height=high_latent.shape[3],
                latent_width=high_latent.shape[4],
                fps=fps,
                device=distributed.device,
                dtype=high_latent.dtype,
            )
            synchronize_context = (
                ddp_model.no_sync()
                if accumulation_index < accumulation_steps - 1
                else contextlib.nullcontext()
            )
            with synchronize_context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                upsampler_loss, stage2_loss, sampled_sigma = ddp_model(
                    low_latent_unnormalized=low_latent_unnormalized,
                    high_latent=high_latent,
                    prompt_context=prompt_context,
                    prompt_mask=prompt_mask,
                    positions=positions,
                )
                total_loss = (
                    upsampler_weight * upsampler_loss
                    + stage2_weight * stage2_loss
                ) / accumulation_steps
            total_loss.backward()
            accumulated_up += upsampler_loss.detach() / accumulation_steps
            accumulated_stage2 += stage2_loss.detach() / accumulation_steps
            accumulated_sigma += sampled_sigma.float().mean() / accumulation_steps

        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        should_log = step == start_step + 1 or step % log_interval == 0 or step == total_steps
        if should_log:
            reduced = _reduce_metrics(
                torch.stack(
                    [
                        accumulated_up,
                        accumulated_stage2,
                        accumulated_sigma,
                        gradient_norm.detach().float(),
                    ]
                ),
                distributed,
            )
            now = time.monotonic()
            elapsed = max(now - interval_start, 1e-6)
            steps_since_log = step - last_logged_step
            steps_per_second = steps_since_log / elapsed
            samples_per_second = steps_per_second * global_batch
            eta_seconds = (total_steps - step) / max(steps_per_second, 1e-9)
            learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
            record = {
                "step": step,
                "loss": float(upsampler_weight * reduced[0] + stage2_weight * reduced[1]),
                "loss_upsampler_full_frame": float(reduced[0]),
                "loss_stage2_x0_full_frame": float(reduced[1]),
                "sigma_mean": float(reduced[2]),
                "grad_norm": float(reduced[3]),
                "lr_upsampler": learning_rates[0],
                "lr_stage2_lora": learning_rates[1],
                "steps_per_second": steps_per_second,
                "samples_per_second": samples_per_second,
                "eta_hours": eta_seconds / 3600.0,
                "epoch_approx": step * global_batch / len(dataset),
                "gpu_allocated_gb": torch.cuda.memory_allocated(distributed.device) / 1024**3,
                "gpu_reserved_gb": torch.cuda.memory_reserved(distributed.device) / 1024**3,
                "roi_loss": False,
                "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            if distributed.is_main:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"[refiner-train] step={step}/{total_steps} "
                    f"loss={record['loss']:.6f} up={record['loss_upsampler_full_frame']:.6f} "
                    f"stage2={record['loss_stage2_x0_full_frame']:.6f} "
                    f"sigma={record['sigma_mean']:.4f} grad={record['grad_norm']:.3f} "
                    f"samples/s={record['samples_per_second']:.3f} "
                    f"epoch={record['epoch_approx']:.3f} eta={record['eta_hours']:.2f}h "
                    f"vram={record['gpu_allocated_gb']:.1f}G/{record['gpu_reserved_gb']:.1f}G",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(record, step=step)
            interval_start = now
            last_logged_step = step

        if step % checkpoint_interval == 0 or step == total_steps:
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
        duration_hours = (time.monotonic() - run_start) / 3600.0
        print(f"[refiner-train] complete steps={total_steps} wall_time={duration_hours:.2f}h", flush=True)
        if wandb_run is not None:
            wandb_run.finish()
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    validated, dataset = _run_preflight(config)
    if args.preflight_only:
        return

    output_value = args.output_dir or Path(str(_nested(config, "output_dir")))
    output_dir = _resolved_path(output_value)
    _train(
        config=config,
        validated=validated,
        dataset=dataset,
        output_dir=output_dir,
        resume_mode=args.resume,
    )


if __name__ == "__main__":
    main()
