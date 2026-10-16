from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Mapping

import av
import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image
from torch.utils.data import Dataset

from ltx_trainer import logger
from ltx_trainer.action_binding import (
    BUTTON_COLUMN_ALIASES,
    CANONICAL_BUTTONS,
    CROSS_FPS_FOUR_ACTION_BUTTONS,
    CROSS_FPS_UNIFIED_BUTTONS,
    apply_button_remap,
    build_button_action_timeline_text,
    build_control_binding_text,
    crossfps_fixed_button_names,
    crossfps_fixed_raw_button_columns,
    format_timeline_seconds,
    is_crossfps_fixed_button_mode,
    load_control_schemas,
    load_semantic_action_vocab,
    normalize_game_id,
    resolve_button_meanings,
)
from ltx_trainer.video_utils import read_video

CROSSFPS_BUTTON_COLUMNS = CANONICAL_BUTTONS
CROSSFPS_ACTION_CHANNELS = (
    "j_left_x",
    "j_left_y",
    "j_right_x",
    "j_right_y",
    *CROSSFPS_BUTTON_COLUMNS,
)
CROSSFPS_UNIFIED_ACTION_CHANNELS = (
    "j_left_x",
    "j_left_y",
    "j_right_x",
    "j_right_y",
    *CROSS_FPS_UNIFIED_BUTTONS,
)
CROSSFPS_FOUR_ACTION_CHANNELS = (
    "j_left_x",
    "j_left_y",
    "j_right_x",
    "j_right_y",
    *CROSS_FPS_FOUR_ACTION_BUTTONS,
)

CROSSFPS_BUTTON_RESAMPLING_POINT = "point"
CROSSFPS_BUTTON_RESAMPLING_CAUSAL_RUN_MAX = "causal_run_max"
_CROSSFPS_BUTTON_RESAMPLING_MODES = {
    CROSSFPS_BUTTON_RESAMPLING_POINT,
    CROSSFPS_BUTTON_RESAMPLING_CAUSAL_RUN_MAX,
}

try:
    import torchaudio.functional as torchaudio_f
except ModuleNotFoundError:                                                             
    torchaudio_f = None

_RAW_YOUTUBE_LAYOUT = "raw_youtube_local"
_INT_FORMAT_MAX: dict[str, float] = {
    "u8": 128.0,
    "u8p": 128.0,
    "s16": 32768.0,
    "s16p": 32768.0,
    "s32": 2147483648.0,
    "s32p": 2147483648.0,
}
_PROMPT_TAG_PATTERN = re.compile(r"<[^>]+>|\[[^\]]+\]")
_PROMPT_ID_FIELDS = ("sample_id", "id", "clip_id", "meta_path", "metadata_path")
_EXPLICIT_TRACKED_IDENTITY_PATTERN = re.compile(
    r"\bthe tracked (?:enemy|opponent) (?:has this appearance:|is\b)",
    flags=re.IGNORECASE,
)


def clean_crossfps_prompt_text(text: Any) -> str | None:
    ''

    if not isinstance(text, str):
        return None
    cleaned = _PROMPT_TAG_PATTERN.sub(" ", text).replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def has_explicit_tracked_identity(prompt: Any) -> bool:
    ''

    if not isinstance(prompt, str):
        return False
    return (
        "reference image" not in prompt.lower()
        and _EXPLICIT_TRACKED_IDENTITY_PATTERN.search(prompt) is not None
    )


def _cleaned_prompt_lookup_keys(entry: dict[str, Any], index: int) -> list[str]:
    keys: list[str] = []
    for field in _PROMPT_ID_FIELDS:
        value = entry.get(field)
        if value is not None:
            keys.append(str(value))
    video_path = entry.get("video_path")
    if video_path is not None:
        keys.append(str(video_path))
    keys.append(str(index))
    return keys


def _apply_cleaned_prompt(
    entry: dict[str, Any],
    index: int,
    cleaned_prompts: dict[str, str] | None,
    fallback: Literal["original", "error"],
) -> dict[str, Any]:
    if cleaned_prompts is None:
        return entry

    for key in _cleaned_prompt_lookup_keys(entry, index):
        cleaned_prompt = cleaned_prompts.get(key)
        if cleaned_prompt:
            entry = entry.copy()
            entry["prompt_text"] = cleaned_prompt
            return entry

    if fallback == "error":
        raise KeyError(f"No cleaned prompt found for sample keys: {_cleaned_prompt_lookup_keys(entry, index)}")
    return entry


def _apply_prompt_mix(prompt: str, index: int, prompt_mix: Any | None) -> str:
    if prompt_mix is None or not getattr(prompt_mix, "enabled", False):
        return prompt

    rng = random.Random(int(getattr(prompt_mix, "seed", 42)) + int(index))
    draw = rng.random()
    caption_prob = float(getattr(prompt_mix, "caption_prob", 0.5))
    neutral_prob = float(getattr(prompt_mix, "neutral_prob", 0.3))
    if draw < caption_prob:
        return prompt
    if draw < caption_prob + neutral_prob:
        return str(getattr(prompt_mix, "neutral_prompt", "a video"))
    return ""


def _has_cleaned_prompt(entry: dict[str, Any], index: int, cleaned_prompts: dict[str, str]) -> bool:
    return any(cleaned_prompts.get(key) for key in _cleaned_prompt_lookup_keys(entry, index))


def _filter_samples_with_cleaned_prompts(
    samples: list[dict[str, Any] | Path],
    cleaned_prompts: dict[str, str] | None,
) -> list[dict[str, Any] | Path]:
    if cleaned_prompts is None:
        return samples

    filtered: list[dict[str, Any] | Path] = []
    for index, sample in enumerate(samples):
        if isinstance(sample, Path):
            filtered.append(sample)
            continue
        if _has_cleaned_prompt(sample, index, cleaned_prompts):
            filtered.append(sample)
    return filtered


def _load_id_list(path: str | Path) -> set[str]:
    id_path = Path(path).expanduser().resolve()
    if not id_path.exists():
        raise FileNotFoundError(f"ID list path does not exist: {id_path}")
    ids = {line.strip() for line in id_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not ids:
        raise ValueError(f"ID list path is empty: {id_path}")
    return ids


def _load_manifest_ids(path: str | Path) -> set[str]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Audio-loss manifest path does not exist: {manifest_path}")
    ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            clip_id = entry.get("clip_id") or entry.get("sample_id")
            if not clip_id:
                raise ValueError(
                    f"Audio-loss manifest entry #{line_index} has neither clip_id nor sample_id: {manifest_path}"
                )
            ids.add(str(clip_id))
    if not ids:
        raise ValueError(f"Audio-loss manifest path is empty: {manifest_path}")
    return ids


def _parse_fps(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, str) and "/" in value:
        num, den = value.split("/", 1)
        den_value = float(den)
        return float(num) / den_value if den_value else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class AudioOnlyDataset(Dataset):
    ''

    def __init__(
        self,
        data_root: str | Path,
        *,
        layout: str = "auto",
        metadata_root: str | Path | None = None,
        media_root: str | Path | None = None,
        manifest_path: str | Path | None = None,
        target_width: int | None = None,
        target_height: int | None = None,
        target_num_frames: int | None = None,
        target_fps: float | None = None,
        target_audio_sample_rate: int | None = None,
        prompt_fields: list[str] | None = None,
        random_temporal_crop: bool = True,
        min_source_duration: float | None = None,
        cleaned_prompts: dict[str, str] | None = None,
        cleaned_prompt_fallback: Literal["original", "error"] = "original",
        source_name: str | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.data_root}")

        self._manifest_path = Path(manifest_path).expanduser().resolve() if manifest_path is not None else None
        self._layout = self._resolve_layout(layout, metadata_root)
        self._metadata_root = Path(metadata_root).expanduser().resolve() if metadata_root is not None else None
        self._media_root = Path(media_root).expanduser().resolve() if media_root is not None else None
        self._target_width = target_width
        self._target_height = target_height
        self._target_num_frames = target_num_frames
        self._target_fps = target_fps
        self._target_audio_sample_rate = target_audio_sample_rate
        self._prompt_fields = prompt_fields or [
            "summary_with_audio_seed2_timestamp",
            "summary_with_audio_timestamp",
            "summary_with_audio_seed2",
            "summary_with_audio",
            "caption_o",
        ]
        self._random_temporal_crop = random_temporal_crop
        self._min_source_duration = min_source_duration
        self._cleaned_prompts = cleaned_prompts
        self._cleaned_prompt_fallback = cleaned_prompt_fallback
        self._source_name = source_name

        if self._layout == _RAW_YOUTUBE_LAYOUT and self._manifest_path is None:
            self._metadata_root, self._media_root = self._resolve_raw_roots(
                metadata_root=self._metadata_root,
                media_root=self._media_root,
            )
            missing = [
                name
                for name, value in (
                    ("target_width", self._target_width),
                    ("target_height", self._target_height),
                    ("target_num_frames", self._target_num_frames),
                    ("target_fps", self._target_fps),
                    ("target_audio_sample_rate", self._target_audio_sample_rate),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "Raw youtube layout requires fixed clip settings, missing: " + ", ".join(sorted(missing))
                )

        self._samples = _filter_samples_with_cleaned_prompts(self._discover_samples(), self._cleaned_prompts)
        if not self._samples:
            raise ValueError(f"No AudioOnly samples found under {self.data_root} with accepted cleaned prompts")

    def _resolve_layout(self, requested_layout: str, metadata_root: str | Path | None) -> str:
        if requested_layout != "auto":
            return requested_layout

        if self._manifest_path is not None:
            return _RAW_YOUTUBE_LAYOUT

        manifest_jsonl = self.data_root / "manifest.jsonl"
        manifest_json = self.data_root / "manifest.json"
        if manifest_jsonl.exists() or manifest_json.exists() or next(self.data_root.glob("*.pt"), None) is not None:
            return "precomputed"

        metadata_path = Path(metadata_root).expanduser().resolve() if metadata_root is not None else self.data_root / "meta"
        if metadata_path.exists():
            return _RAW_YOUTUBE_LAYOUT

        raise ValueError(
            f"Could not auto-detect dataset layout under {self.data_root}. "
            "Provide `layout='precomputed'` or `layout='raw_youtube_local'` explicitly."
        )

    def _resolve_raw_roots(
        self,
        *,
        metadata_root: Path | None,
        media_root: Path | None,
    ) -> tuple[Path, Path]:
        resolved_metadata = metadata_root
        if resolved_metadata is None:
            candidate = self.data_root / "meta"
            if not candidate.exists():
                raise FileNotFoundError(
                    "Raw youtube dataset requires `metadata_root` or a `meta/` directory under the dataset root."
                )
            resolved_metadata = candidate
        if not resolved_metadata.exists():
            raise FileNotFoundError(f"Metadata root does not exist: {resolved_metadata}")

        resolved_media = media_root or (self.data_root / "youtube")
        if not resolved_media.exists():
            raise FileNotFoundError(f"Media root does not exist: {resolved_media}")
        return resolved_metadata, resolved_media

    def _discover_samples(self) -> list[dict[str, Any] | Path]:
        if self._layout == _RAW_YOUTUBE_LAYOUT:
            return self._discover_raw_samples()

        manifest_jsonl = self.data_root / "manifest.jsonl"
        manifest_json = self.data_root / "manifest.json"
        if manifest_jsonl.exists():
            return [json.loads(line) for line in manifest_jsonl.read_text().splitlines() if line.strip()]
        if manifest_json.exists():
            payload = json.loads(manifest_json.read_text())
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict) and "samples" in payload:
                return list(payload["samples"])
            raise ValueError(f"Unsupported manifest structure in {manifest_json}")
        return sorted(self.data_root.glob("*.pt"))

    def _discover_raw_samples(self) -> list[dict[str, Any]]:
        if self._manifest_path is not None:
            return self._discover_from_manifest()

        assert self._metadata_root is not None
        assert self._media_root is not None

        samples: list[dict[str, Any]] = []
        for metadata_path in sorted(self._metadata_root.rglob("*.json")):
            payload = json.loads(metadata_path.read_text())
            video_path = self._resolve_video_path(payload, metadata_path)
            audio_path = self._resolve_audio_path(payload, metadata_path)
            prompt_text = self._extract_prompt_text(payload)
            if video_path is None or audio_path is None or prompt_text is None:
                continue
            source_fps = float(payload.get("fps") or self._target_fps or 24.0)
            source_frames = int(payload.get("f_num") or 0)
            raw_duration = payload.get("duration")
            if raw_duration is not None:
                source_duration = float(raw_duration)
            elif source_frames > 0:
                source_duration = source_frames / max(source_fps, 1e-6)
            else:
                source_duration = 0.0
            if self._min_source_duration is not None and source_duration < self._min_source_duration:
                continue
            samples.append(
                {
                    "_layout": _RAW_YOUTUBE_LAYOUT,
                    "video_path": str(video_path),
                    "audio_path": str(audio_path),
                    "prompt_text": prompt_text,
                    "meta_path": str(metadata_path),
                    "source_fps": source_fps,
                    "source_frames": source_frames,
                    "source_duration": source_duration,
                }
            )
        return samples

    def _discover_from_manifest(self) -> list[dict[str, Any]]:
        ''
        assert self._manifest_path is not None
        if not self._manifest_path.exists():
            raise FileNotFoundError(f"Manifest path does not exist: {self._manifest_path}")

        samples: list[dict[str, Any]] = []
        default_fps = float(self._target_fps or 24.0)
        with self._manifest_path.open() as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                entry = json.loads(stripped)
                if "video_path" not in entry or "audio_path" not in entry or "prompt_text" not in entry:
                    continue
                source_fps = float(
                    entry.get("source_fps")
                    or entry.get("validated_video_fps")
                    or default_fps
                )
                source_frames = int(
                    entry.get("source_frames")
                    or entry.get("decoded_video_frames")
                    or 0
                )
                raw_duration = entry.get("source_duration")
                if raw_duration is not None and float(raw_duration) > 0:
                    source_duration = float(raw_duration)
                elif source_fps > 0 and source_frames > 0:
                    source_duration = source_frames / source_fps
                else:
                    source_duration = 0.0
                if self._min_source_duration is not None and source_duration < self._min_source_duration:
                    continue
                samples.append(
                    {
                        "_layout": _RAW_YOUTUBE_LAYOUT,
                        "video_path": str(entry["video_path"]),
                        "audio_path": str(entry["audio_path"]),
                        "prompt_text": str(entry["prompt_text"]),
                        "meta_path": str(entry["meta_path"]) if entry.get("meta_path") is not None else None,
                        "source_fps": source_fps,
                        "source_frames": source_frames,
                        "source_duration": source_duration,
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def _resolve_video_path(self, payload: dict[str, Any], metadata_path: Path) -> Path | None:
        assert self._media_root is not None
        clip_url = payload.get("clip_url")
        candidates = []
        if clip_url:
            candidates.append(self._media_root / Path(clip_url).name)
        candidates.append(self._media_root / f"{metadata_path.stem}.mp4")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _resolve_audio_path(self, payload: dict[str, Any], metadata_path: Path) -> Path | None:
        assert self._media_root is not None
        audio_url = payload.get("audio_url")
        candidates = []
        if audio_url:
            candidates.append(self._media_root / Path(audio_url).name)
        candidates.append(self._media_root / f"{metadata_path.stem}.wav")
        if metadata_path.stem.endswith(".bc.oc"):
            candidates.append(self._media_root / f"{metadata_path.stem.removesuffix('.bc.oc')}.wav")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _extract_prompt_text(self, payload: dict[str, Any]) -> str | None:
        parts = payload.get("parts_a2vovi_v4") or []
        for field in self._prompt_fields:
            if field == "caption_o":
                cleaned = self._clean_prompt_text(payload.get("caption_o"))
                if cleaned:
                    return cleaned
                continue

            values = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                value = part.get(field)
                if isinstance(value, str) and value.strip():
                    values.append(value)
            cleaned = self._clean_prompt_text(" ".join(values))
            if cleaned:
                return cleaned
        return None

    @staticmethod
    def _clean_prompt_text(text: Any) -> str | None:
        return clean_crossfps_prompt_text(text)

    def _load_tensor_or_path(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        path = Path(value)
        if not path.is_absolute():
            path = self.data_root / path
        if path.suffix == ".pt":
            data = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(data, torch.Tensor):
                raise TypeError(f"Expected tensor payload at {path}, got {type(data)}")
            return data
        raise ValueError(f"Unsupported tensor payload path: {path}")

    def _resize_and_center_crop_frame(self, frame: torch.Tensor) -> torch.Tensor:
        assert self._target_height is not None
        assert self._target_width is not None

        channels, src_h, src_w = frame.shape
        scale = max(self._target_height / src_h, self._target_width / src_w)
        new_h = math.ceil(src_h * scale)
        new_w = math.ceil(src_w * scale)
        resized = torch_f.interpolate(
            frame.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        top = (new_h - self._target_height) // 2
        left = (new_w - self._target_width) // 2
        return resized[:, top : top + self._target_height, left : left + self._target_width]

    def _target_duration(self) -> float:
        assert self._target_num_frames is not None and self._target_fps is not None
        return self._target_num_frames / float(self._target_fps)

    def _pick_temporal_window(self, source_duration: float) -> tuple[float, float]:
        target_duration = self._target_duration()
        if source_duration <= 0.0:
            return 0.0, target_duration
        if source_duration <= target_duration:
            return 0.0, source_duration
        if self._random_temporal_crop:
            start = random.uniform(0.0, source_duration - target_duration)
        else:
            start = 0.0
        return start, start + target_duration

    def _build_target_frame_indices(
        self,
        source_fps: float,
        source_frames: int,
        window_start_sec: float,
        window_end_sec: float,
    ) -> torch.Tensor:
        assert self._target_num_frames is not None and self._target_fps is not None
        window_duration = max(window_end_sec - window_start_sec, 1.0 / max(source_fps, 1e-6))
        target_duration = self._target_duration()

        if window_duration >= target_duration - 1e-6:
            positions = torch.arange(self._target_num_frames, dtype=torch.float64) / float(self._target_fps)
            positions = positions + float(window_start_sec)
            indices = (positions * source_fps).round().to(torch.int64)
        else:
            window_start_frame = max(int(round(window_start_sec * source_fps)), 0)
            window_end_frame = max(int(round(window_end_sec * source_fps)) - 1, window_start_frame)
            indices = (
                torch.linspace(window_start_frame, window_end_frame, steps=self._target_num_frames)
                .round()
                .to(torch.int64)
            )

        if source_frames > 0:
            indices = indices.clamp(0, source_frames - 1)
        else:
            indices = indices.clamp(min=0)
        return indices

    def _load_raw_video(
        self,
        entry: dict[str, Any],
        window_start_sec: float,
        window_end_sec: float,
    ) -> tuple[torch.Tensor, float]:
        assert self._target_num_frames is not None
        assert self._target_fps is not None

        video_path = Path(entry["video_path"])
        metadata_source_fps = float(entry.get("source_fps") or self._target_fps)
        metadata_source_frames = max(int(entry.get("source_frames") or 0), 0)

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream_source_fps = float(stream.average_rate) if stream.average_rate else metadata_source_fps
            stream_source_frames = int(stream.frames or metadata_source_frames)
            source_fps = stream_source_fps if stream_source_fps > 0 else metadata_source_fps
            source_frames = stream_source_frames if stream_source_frames > 0 else metadata_source_frames

            target_indices = self._build_target_frame_indices(
                source_fps=source_fps,
                source_frames=source_frames,
                window_start_sec=window_start_sec,
                window_end_sec=window_end_sec,
            )
            target_map: dict[int, list[int]] = defaultdict(list)
            for output_index, source_index in enumerate(target_indices.tolist()):
                target_map[int(source_index)].append(output_index)

            last_needed_index = max(target_map.keys()) if target_map else -1
            frames: list[torch.Tensor | None] = [None] * self._target_num_frames

            for source_index, frame in enumerate(container.decode(video=0)):
                if source_index > last_needed_index:
                    break
                if source_index not in target_map:
                    continue
                frame_tensor = (
                    torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1).float().div(255.0)
                )
                frame_tensor = self._resize_and_center_crop_frame(frame_tensor)
                for output_index in target_map[source_index]:
                    frames[output_index] = frame_tensor

        first_valid = next((frame for frame in frames if frame is not None), None)
        if first_valid is None:
            raise ValueError(f"Failed to decode any frames from {video_path}")

        for index, frame in enumerate(frames):
            if frame is None:
                frames[index] = frames[index - 1] if index > 0 and frames[index - 1] is not None else first_valid

        return torch.stack([frame for frame in frames if frame is not None]), float(self._target_fps)

    @staticmethod
    def _audio_frame_to_float(frame: av.AudioFrame) -> np.ndarray:
        fmt = frame.format.name
        arr = frame.to_ndarray().astype(np.float32)
        if fmt in _INT_FORMAT_MAX:
            arr = arr / _INT_FORMAT_MAX[fmt]
        if not frame.format.is_planar:
            channels = len(frame.layout.channels)
            arr = arr.reshape(-1, channels).T
        return arr

    @staticmethod
    def _canonicalize_audio_channels(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim != 2:
            raise ValueError(
                "Expected audio waveform to have shape [samples] or [channels, samples], "
                f"got {tuple(waveform.shape)}"
            )

        num_channels = int(waveform.shape[0])
        if num_channels == 1:
            waveform = waveform.repeat(2, 1)
        elif num_channels != 2:
            raise ValueError(f"Expected mono or stereo audio, got {num_channels} channels")

        return waveform.contiguous()

    def _load_audio_file(self, audio_path: Path) -> tuple[torch.Tensor, int]:
        with av.open(str(audio_path)) as container:
            try:
                audio_stream = next(stream for stream in container.streams if stream.type == "audio")
            except StopIteration as exc:                                            
                raise ValueError(f"No audio stream found in {audio_path}") from exc

            chunks = [self._audio_frame_to_float(frame) for frame in container.decode(audio=0)]
            sample_rate = int(audio_stream.rate)

        if not chunks:
            raise ValueError(f"Failed to decode any audio samples from {audio_path}")

        waveform = torch.from_numpy(np.concatenate(chunks, axis=-1)).float()
        return self._canonicalize_audio_channels(waveform), sample_rate

    def _load_audio_file_window(self, audio_path: Path, start_sec: float, end_sec: float) -> tuple[torch.Tensor, int]:
        with av.open(str(audio_path)) as container:
            try:
                audio_stream = next(stream for stream in container.streams if stream.type == "audio")
            except StopIteration as exc:                                            
                raise ValueError(f"No audio stream found in {audio_path}") from exc

            sample_rate = int(audio_stream.rate)
            wanted_start = max(0, int(round(float(start_sec) * sample_rate)))
            wanted_end = max(wanted_start + 1, int(round(float(end_sec) * sample_rate)))
            seek_sec = max(float(start_sec) - 0.5, 0.0)
            if audio_stream.time_base is not None:
                container.seek(
                    int(seek_sec / float(audio_stream.time_base)),
                    stream=audio_stream,
                    any_frame=False,
                    backward=True,
                )

            chunks: list[tuple[int, np.ndarray]] = []
            fallback_start = int(round(seek_sec * sample_rate))
            for frame in container.decode(audio=0):
                chunk = self._audio_frame_to_float(frame)
                chunk_samples = int(chunk.shape[-1])
                if chunk_samples <= 0:
                    continue
                if frame.time is not None:
                    chunk_start = max(0, int(round(float(frame.time) * sample_rate)))
                else:
                    chunk_start = fallback_start
                chunk_end = chunk_start + chunk_samples
                fallback_start = chunk_end
                if chunk_end <= wanted_start:
                    continue
                if chunk_start >= wanted_end:
                    break
                chunks.append((chunk_start, chunk))

        if not chunks:
            raise ValueError(f"Failed to decode audio window [{start_sec:.3f}, {end_sec:.3f}] from {audio_path}")

        channels = int(chunks[0][1].shape[0])
        window = np.zeros((channels, wanted_end - wanted_start), dtype=np.float32)
        covered_samples = np.zeros(wanted_end - wanted_start, dtype=np.bool_)
        for chunk_start, chunk in chunks:
            chunk_end = chunk_start + int(chunk.shape[-1])
            overlap_start = max(wanted_start, chunk_start)
            overlap_end = min(wanted_end, chunk_end)
            if overlap_end <= overlap_start:
                continue
            src_start = overlap_start - chunk_start
            src_end = src_start + (overlap_end - overlap_start)
            dst_start = overlap_start - wanted_start
            dst_end = dst_start + (overlap_end - overlap_start)
            window[:, dst_start:dst_end] = chunk[:, src_start:src_end]
            covered_samples[dst_start:dst_end] = True

        covered_count = int(np.count_nonzero(covered_samples))
        requested_count = int(covered_samples.size)
        if covered_count != requested_count:
            raise ValueError(
                f"Decoded audio window [{start_sec:.3f}, {end_sec:.3f}] from {audio_path} covers "
                f"{covered_count} of {requested_count} requested samples; refusing silent zero-padding."
            )

        waveform = torch.from_numpy(window).float()
        return self._canonicalize_audio_channels(waveform), sample_rate

    def _normalize_raw_audio(
        self,
        waveform: torch.Tensor,
        source_rate: int,
        window_start_sec: float,
        window_end_sec: float,
    ) -> torch.Tensor:
        assert self._target_audio_sample_rate is not None
        assert self._target_num_frames is not None
        assert self._target_fps is not None

        waveform = self._canonicalize_audio_channels(waveform.float())
        target_rate = int(self._target_audio_sample_rate)
        target_samples = max(1, round(target_rate * self._target_num_frames / self._target_fps))

        total_samples = int(waveform.shape[-1])
        if total_samples == 0:
            return torch.zeros((2, target_samples), dtype=torch.float32)

        start_sample = max(0, min(total_samples - 1, int(round(window_start_sec * source_rate))))
        end_sample = max(start_sample + 1, min(total_samples, int(round(window_end_sec * source_rate))))
        clip = waveform[:, start_sample:end_sample].contiguous()
        if clip.shape[-1] == 0:
            clip = waveform[:, :1].contiguous()

        if source_rate != target_rate:
            if torchaudio_f is not None:
                clip = torchaudio_f.resample(clip, source_rate, target_rate)
            else:
                resampled_samples = max(1, round(clip.shape[-1] * target_rate / max(source_rate, 1)))
                clip = torch_f.interpolate(
                    clip.unsqueeze(0),
                    size=resampled_samples,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0)

        current = int(clip.shape[-1])
        if current < target_samples:
            clip = torch_f.pad(clip, (0, target_samples - current))
        elif current > target_samples:
            clip = clip[:, :target_samples]
        return clip.contiguous()

    def _load_video(
        self,
        entry: dict[str, Any],
        window_start_sec: float = 0.0,
        window_end_sec: float | None = None,
    ) -> tuple[torch.Tensor, float]:
        if entry.get("_layout") == _RAW_YOUTUBE_LAYOUT:
            if window_end_sec is None:
                window_end_sec = window_start_sec + self._target_duration()
            return self._load_raw_video(entry, window_start_sec, window_end_sec)

        if "video_pixels" in entry:
            video = self._load_tensor_or_path(entry["video_pixels"]).float()
            if video.ndim != 4:
                raise ValueError(f"Expected `video_pixels` to have 4 dims, got {tuple(video.shape)}")
            if video.shape[1] == 3:
                return video, float(entry["fps"])
            if video.shape[0] == 3:
                return video.permute(1, 0, 2, 3).contiguous(), float(entry["fps"])
            raise ValueError(
                "Expected `video_pixels` in [F, C, H, W] or [C, F, H, W] format with 3 channels, "
                f"got {tuple(video.shape)}"
            )
        if "video_path" not in entry:
            raise KeyError("AudioOnlyDataset sample must provide `video_pixels` or `video_path`.")
        video_path = Path(entry["video_path"])
        if not video_path.is_absolute():
            video_path = self.data_root / video_path
        return read_video(video_path)

    def _load_audio(
        self,
        entry: dict[str, Any],
        window_start_sec: float = 0.0,
        window_end_sec: float | None = None,
    ) -> tuple[torch.Tensor, int | None]:
        if "audio_waveform" in entry:
            waveform = self._canonicalize_audio_channels(self._load_tensor_or_path(entry["audio_waveform"]).float())
            sample_rate = entry.get("audio_sample_rate")
            return waveform, int(sample_rate) if sample_rate is not None else None
        if "audio_path" not in entry:
            raise KeyError("AudioOnlyDataset sample must provide `audio_waveform` or `audio_path`.")

        audio_path = Path(entry["audio_path"])
        if not audio_path.is_absolute():
            audio_path = self.data_root / audio_path
        waveform, sample_rate = self._load_audio_file(audio_path)
        if entry.get("_layout") == _RAW_YOUTUBE_LAYOUT:
            if window_end_sec is None:
                window_end_sec = window_start_sec + self._target_duration()
            normalized = self._normalize_raw_audio(waveform, sample_rate, window_start_sec, window_end_sec)
            return normalized, int(self._target_audio_sample_rate)
        return self._canonicalize_audio_channels(waveform), sample_rate

    def _load_embedding(self, entry: dict[str, Any], key: str) -> torch.Tensor:
        value = entry.get(key)
        if value is None:
            raise KeyError(f"AudioOnlyDataset sample is missing `{key}`")
        return self._load_tensor_or_path(value).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index]
        if isinstance(sample, Path):
            entry = torch.load(sample, map_location="cpu", weights_only=False)
            if not isinstance(entry, dict):
                raise TypeError(f"Expected dict sample in {sample}, got {type(entry)}")
        else:
            entry = sample
        entry = _apply_cleaned_prompt(entry, index, self._cleaned_prompts, self._cleaned_prompt_fallback)

        if entry.get("_layout") == _RAW_YOUTUBE_LAYOUT:
            source_duration = float(entry.get("source_duration") or 0.0)
            window_start_sec, window_end_sec = self._pick_temporal_window(source_duration)
        else:
            window_start_sec, window_end_sec = 0.0, None

        video_pixels, fps = self._load_video(entry, window_start_sec, window_end_sec)
        audio_waveform, audio_sample_rate = self._load_audio(entry, window_start_sec, window_end_sec)

        sample = {
            "video_pixels": video_pixels,
            "audio_waveform": audio_waveform,
            "audio_sample_rate": audio_sample_rate,
            "has_audio": True,
            "fps": float(entry.get("fps", fps)),
            "source": entry.get("source", self._source_name),
        }

        if "video_prompt_embeds" in entry:
            sample["video_prompt_embeds"] = self._load_embedding(entry, "video_prompt_embeds")
            sample["audio_prompt_embeds"] = self._load_embedding(entry, "audio_prompt_embeds")
            sample["prompt_attention_mask"] = self._load_embedding(entry, "prompt_attention_mask").to(dtype=torch.bool)
        elif "prompt_embeds" in entry:
            sample["prompt_embeds"] = self._load_embedding(entry, "prompt_embeds")
            sample["prompt_attention_mask"] = self._load_embedding(entry, "prompt_attention_mask").to(dtype=torch.bool)
        elif "prompt_text" in entry:
            sample["prompt_text"] = str(entry["prompt_text"])
        else:
            raise KeyError(
                "AudioOnlyDataset sample must provide either precomputed prompt embeddings or `prompt_text`."
            )

        return sample


class CrossFPSAudioActionDataset(AudioOnlyDataset):
    ''

    def __init__(
        self,
        data_root: str | Path,
        *,
        manifest_path: str | Path | None = None,
        target_width: int | None = None,
        target_height: int | None = None,
        target_num_frames: int | None = None,
        target_fps: float | None = None,
        target_audio_sample_rate: int | None = None,
        source_name: str | None = None,
        cleaned_prompts: dict[str, str] | None = None,
        cleaned_prompt_fallback: Literal["original", "error"] = "original",
        prompt_mix: Any | None = None,
        audio_loss_eligible_ids_path: str | Path | None = None,
        audio_loss_eligible_manifest_path: str | Path | None = None,
        random_temporal_crop: bool = True,
        min_source_duration: float | None = None,
        require_audio: bool = True,
        action_binding_config: Any | None = None,
        action_binding_training: bool = True,
        joystick_normalization: Any | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.data_root}")
        self._manifest_path = Path(manifest_path).expanduser().resolve() if manifest_path is not None else None
        if self._manifest_path is not None and not self._manifest_path.exists():
            raise FileNotFoundError(f"Manifest path does not exist: {self._manifest_path}")
        self._target_width = target_width
        self._target_height = target_height
        self._target_num_frames = target_num_frames
        self._target_fps = target_fps
        self._target_audio_sample_rate = target_audio_sample_rate
        self._source_name = source_name
        self._cleaned_prompts = cleaned_prompts
        self._cleaned_prompt_fallback = cleaned_prompt_fallback
        self._prompt_mix = prompt_mix
        self._random_temporal_crop = random_temporal_crop
        self._min_source_duration = min_source_duration
        self._require_audio = require_audio
        self._action_binding = action_binding_config
        self._action_binding_training = action_binding_training
        self._joystick_normalization = joystick_normalization
        self._action_binding_mode = (
            str(getattr(action_binding_config, "button_conditioning_mode", "text_binding"))
            if action_binding_config is not None and getattr(action_binding_config, "enabled", False)
            else "text_binding"
        )
        self._control_schemas = (
            load_control_schemas(getattr(action_binding_config, "schema_path", None))
            if (
                action_binding_config is not None
                and getattr(action_binding_config, "enabled", False)
                and not is_crossfps_fixed_button_mode(self._action_binding_mode)
            )
            else {}
        )
        self._semantic_action_vocab = self._load_semantic_action_vocab(action_binding_config)
        self._semantic_vocab_index_by_game_button = {
            (item["game_id"], item["button"]): index for index, item in enumerate(self._semantic_action_vocab)
        }
        self._semantic_vocab_game_ids = {item["game_id"] for item in self._semantic_action_vocab}
        timeline_cfg = getattr(action_binding_config, "timeline_prompt", None) if action_binding_config is not None else None
        self._semantic_vocab_active_threshold = float(getattr(timeline_cfg, "active_threshold", 0.1))
        self._warned_unknown_semantic_games: set[tuple[str, str]] = set()
        self._warned_dropped_semantic_buttons: set[tuple[str, str]] = set()
        if audio_loss_eligible_ids_path is not None and audio_loss_eligible_manifest_path is not None:
            raise ValueError(
                "Set only one of audio_loss_eligible_ids_path and audio_loss_eligible_manifest_path"
            )
        if audio_loss_eligible_manifest_path is not None:
            self._audio_loss_eligible_ids = _load_manifest_ids(audio_loss_eligible_manifest_path)
        elif audio_loss_eligible_ids_path is not None:
            self._audio_loss_eligible_ids = _load_id_list(audio_loss_eligible_ids_path)
        else:
            self._audio_loss_eligible_ids = None
        self._dropped_short_samples = 0
        self._dropped_short_audio_samples = 0
        self._audio_duration_cache: dict[Path, float | None] = {}

        missing = [
            name
            for name, value in (
                ("target_width", self._target_width),
                ("target_height", self._target_height),
                ("target_num_frames", self._target_num_frames),
                ("target_fps", self._target_fps),
                ("target_audio_sample_rate", self._target_audio_sample_rate),
            )
            if value is None
        ]
        if missing:
            raise ValueError("CrossFPSAudioActionDataset requires fixed clip settings, missing: " + ", ".join(missing))

        self._samples = _filter_samples_with_cleaned_prompts(
            self._discover_crossfps_samples(),
            self._cleaned_prompts,
        )
        if self._dropped_short_samples:
            logger.warning(
                "Dropped %d CrossFPS samples shorter than fixed_clip.num_frames=%d",
                self._dropped_short_samples,
                int(self._target_num_frames or 0),
            )
        if self._dropped_short_audio_samples:
            logger.warning(
                "Dropped %d CrossFPS samples whose audio_path does not cover the aligned video window",
                self._dropped_short_audio_samples,
            )
        if not self._samples:
            location = str(self._manifest_path) if self._manifest_path is not None else str(self.data_root)
            raise ValueError(f"No CrossFPS samples found from {location} with aligned audio and actions")

    def _load_semantic_action_vocab(self, action_binding_config: Any | None) -> list[dict[str, str]]:
        if (
            action_binding_config is None
            or not getattr(action_binding_config, "enabled", False)
            or getattr(action_binding_config, "button_conditioning_mode", "text_binding") != "semantic_vocab"
        ):
            return []
        semantic_vocab_cfg = getattr(action_binding_config, "semantic_vocab", None)
        vocab_path = getattr(semantic_vocab_cfg, "path", None)
        if not vocab_path:
            raise ValueError("action_binding.semantic_vocab.path is required for semantic_vocab mode")
        return load_semantic_action_vocab(vocab_path)

    _resize_and_center_crop_frame = AudioOnlyDataset._resize_and_center_crop_frame
    _load_audio_file = AudioOnlyDataset._load_audio_file
    _normalize_raw_audio = AudioOnlyDataset._normalize_raw_audio

    def _clip_root(self) -> Path:
        train_root = self.data_root / "train"
        return train_root if train_root.exists() else self.data_root

    def _discover_crossfps_samples(self) -> list[dict[str, Any]]:
        if self._manifest_path is not None:
            return self._discover_crossfps_samples_from_manifest()

        samples: list[dict[str, Any]] = []
        for metadata_path in sorted(self._clip_root().glob("*/metadata.json")):
            clip_dir = metadata_path.parent
            video_path = clip_dir / "video.mp4"
            audio_path = clip_dir / "audio_aligned.wav"
            action_path = clip_dir / "actions_raw.parquet"
            if not (video_path.exists() and action_path.exists()) or (self._require_audio and not audio_path.exists()):
                continue

            payload = json.loads(metadata_path.read_text())
            prompt_text = self._clean_prompt_text(payload.get("prompt"))
            if prompt_text is None:
                continue
            source_fps = float(payload.get("fps") or self._target_fps or 20.0)
            source_frames = int(payload.get("chunk_size") or 0)
            if source_frames <= 0:
                source_frames = int(round(float(payload.get("original_video", {}).get("duration", 0.0)) * source_fps))
            source_duration = source_frames / max(source_fps, 1e-6)
            if self._min_source_duration is not None and source_duration < self._min_source_duration:
                continue
            if source_frames < int(self._target_num_frames or 0):
                self._dropped_short_samples += 1
                continue

            audio_loss_eligible = True
            if self._audio_loss_eligible_ids is not None:
                audio_loss_eligible = clip_dir.name in self._audio_loss_eligible_ids

            samples.append(
                {
                    "video_path": str(video_path),
                    "audio_path": str(audio_path) if audio_path.exists() else None,
                    "action_path": str(action_path),
                    "meta_path": str(metadata_path),
                    "prompt_text": prompt_text,
                    "sample_id": clip_dir.name,
                    "clip_id": clip_dir.name,
                    "game": payload.get("game") or payload.get("source_game"),
                    "split": payload.get("split"),
                    "source_fps": source_fps,
                    "source_frames": source_frames,
                    "source_duration": source_duration,
                    "audio_loss_eligible": audio_loss_eligible,
                }
            )
        return samples

    def _discover_crossfps_samples_from_manifest(self) -> list[dict[str, Any]]:
        assert self._manifest_path is not None
        if self._manifest_path.suffix == ".jsonl":
            with self._manifest_path.open("r", encoding="utf-8") as handle:
                entries = [json.loads(line) for line in handle if line.strip()]
        elif self._manifest_path.suffix == ".json":
            payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                entries = payload
            elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
                entries = payload["samples"]
            else:
                raise ValueError(f"Unsupported CrossFPS manifest structure in {self._manifest_path}")
        else:
            raise ValueError(f"Unsupported CrossFPS manifest format: {self._manifest_path}")

        samples: list[dict[str, Any]] = []
        manifest_dir = self._manifest_path.parent
        for line_index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"CrossFPS manifest entry #{line_index} in {self._manifest_path} is not an object")

            prompt_text = self._clean_prompt_text(
                entry.get("prompt_text")
                or entry.get("prompt")
                or entry.get("training_prompt_static_clean")
                or entry.get("training_prompt_static")
                or entry.get("caption")
            )
            if prompt_text is None:
                continue

            paths: dict[str, str] = {}
            video_value = entry.get("video_path")
            if not video_value:
                raise ValueError(f"CrossFPS manifest entry #{line_index} missing required `video_path`")
            paths["video_path"] = str(self._resolve_manifest_path(video_value, manifest_dir))

            audio_value = entry.get("audio_path")
            if audio_value:
                paths["audio_path"] = str(self._resolve_manifest_path(audio_value, manifest_dir))
            elif self._require_audio:
                raise ValueError(f"CrossFPS manifest entry #{line_index} missing required `audio_path`")

            action_value = entry.get("action_path") or entry.get("actions_raw_path")
            if not action_value:
                raise ValueError(f"CrossFPS manifest entry #{line_index} missing required `action_path`/`actions_raw_path`")
            paths["action_path"] = str(self._resolve_manifest_path(action_value, manifest_dir))

            meta_value = entry.get("meta_path") or entry.get("metadata_path")
            if meta_value:
                paths["meta_path"] = str(self._resolve_manifest_path(meta_value, manifest_dir))

            source_fps = float(
                entry.get("source_fps")
                or entry.get("standard_action_hz")
                or entry.get("fps")
                or self._target_fps
                or 20.0
            )
            source_frames = int(
                entry.get("source_frames")
                or entry.get("segment_action_rows")
                or entry.get("chunk_size")
                or 0
            )
            source_duration = entry.get("source_duration") or entry.get("duration_sec")
            if source_frames <= 0 and source_duration is not None:
                source_frames = int(round(float(source_duration) * source_fps))
            if source_frames <= 0:
                raise ValueError(
                    f"CrossFPS manifest entry #{line_index} must provide positive `source_frames` or `source_duration`"
                )
            duration = float(source_duration) if source_duration is not None else source_frames / max(source_fps, 1e-6)
            if self._min_source_duration is not None and duration < self._min_source_duration:
                continue
            if source_frames < int(self._target_num_frames or 0):
                self._dropped_short_samples += 1
                continue
            audio_offset_sec = float(
                entry.get("audio_offset_sec")
                if entry.get("audio_offset_sec") is not None
                else entry.get("video_offset_sec") or 0.0
            )
            audio_duration_sec = None
            if self._require_audio:
                                                                                   
                                                                                     
                                                                              
                                                                               
                                                                        
                audio_duration_sec = float(entry.get("audio_duration_sec") or duration)
                max_crop_start = max(0, source_frames - int(self._target_num_frames or 0))
                required_audio_end = audio_offset_sec + float(max_crop_start + int(self._target_num_frames or 0)) / max(
                    source_fps, 1e-6
                )
                if audio_duration_sec is None or required_audio_end > audio_duration_sec + (1.0 / max(source_fps, 1e-6)):
                    self._dropped_short_audio_samples += 1
                    continue
            sample_id = str(entry.get("sample_id") or entry.get("clip_id") or Path(paths["video_path"]).parent.name)
            clip_id = str(entry.get("clip_id") or entry.get("sample_id") or Path(paths["video_path"]).parent.name)
            button_resampling = str(
                entry.get("button_resampling") or CROSSFPS_BUTTON_RESAMPLING_POINT
            ).strip()
            if button_resampling not in _CROSSFPS_BUTTON_RESAMPLING_MODES:
                raise ValueError(
                    f"CrossFPS manifest entry #{line_index} has unsupported button_resampling="
                    f"{button_resampling!r}; expected one of {sorted(_CROSSFPS_BUTTON_RESAMPLING_MODES)}"
                )
            audio_loss_eligible = True
            if self._audio_loss_eligible_ids is not None:
                audio_loss_eligible = clip_id in self._audio_loss_eligible_ids

            sample = {
                **paths,
                "prompt_text": prompt_text,
                "sample_id": sample_id,
                "clip_id": clip_id,
                "game": normalize_game_id(
                    entry.get("game") or entry.get("source_game") or entry.get("selected_game")
                ),
                "game_id": normalize_game_id(
                    entry.get("game_id")
                    or entry.get("selected_game")
                    or entry.get("game")
                    or entry.get("source_game")
                ),
                "split": entry.get("split"),
                "source": entry.get("source", self._source_name),
                "source_fps": source_fps,
                "source_frames": source_frames,
                "source_duration": duration,
                "video_source_fps": _parse_fps(
                    entry.get("video_source_fps") or entry.get("video_fps"),
                    source_fps,
                ),
                "video_offset_sec": float(entry.get("video_offset_sec") or 0.0),
                "audio_offset_sec": audio_offset_sec,
                "audio_duration_sec": audio_duration_sec,
                "action_row_start": int(entry.get("action_row_start") or entry.get("segment_action_row_start") or 0),
                "action_row_end": int(entry.get("action_row_end") or entry.get("segment_action_row_end") or 0),
                "raw_action_chunk_size": int(entry.get("raw_action_chunk_size") or 0),
                "standard_chunk_action_rows": int(entry.get("standard_chunk_action_rows") or 0),
                "button_resampling": button_resampling,
                "button_meanings": entry.get("button_meanings"),
                "control_profile_id": entry.get("control_profile_id"),
                "audio_loss_eligible": audio_loss_eligible,
            }
            prompt_schedule_value = entry.get("prompt_schedule")
            if prompt_schedule_value is not None:
                if not isinstance(prompt_schedule_value, list) or not prompt_schedule_value:
                    raise ValueError(
                        f"CrossFPS manifest entry #{line_index} has a non-list or empty prompt_schedule."
                    )
                prompt_schedule = [self._clean_prompt_text(value) for value in prompt_schedule_value]
                if any(value is None for value in prompt_schedule):
                    raise ValueError(
                        f"CrossFPS manifest entry #{line_index} contains an empty prompt_schedule item."
                    )
                sample["prompt_schedule"] = prompt_schedule
            score_prompt_schedule_value = entry.get("score_prompt_schedule")
            if score_prompt_schedule_value is not None:
                if (
                    not isinstance(score_prompt_schedule_value, list)
                    or not score_prompt_schedule_value
                ):
                    raise ValueError(
                        f"CrossFPS manifest entry #{line_index} has a non-list or empty "
                        "score_prompt_schedule."
                    )
                score_prompt_schedule = [
                    self._clean_prompt_text(value)
                    for value in score_prompt_schedule_value
                ]
                if any(value is None for value in score_prompt_schedule):
                    raise ValueError(
                        f"CrossFPS manifest entry #{line_index} contains an empty "
                        "score_prompt_schedule item."
                    )
                sample["score_prompt_schedule"] = score_prompt_schedule
                                                                          
                                                                       
                                                                        
            for field in (
                "action_channels",
                "action_contract_version",
                "active_enemy_id_schedule",
                "active_ref_schedule",
                "reference_bank",
                "reference_bank_index_schedule",
                "reference_catalog_path",
                "reference_catalog_sha256",
                "reference_conditioning_contract",
                "reference_conditioning_ready",
                "reference_conditioning_schema_version",
                "reference_manifest_path",
                "reference_manifest_sha256",
                "reference_release_path",
                "reference_valid_mask",
                "ref_asset_release_id",
                "prompt_schedule_chunking",
                "prompt_schedule_schema_version",
                "prompt_schedule_target_frame_ranges_inclusive",
                "prompt_schedule_video_latents_per_chunk",
                "score_prompt_contract",
                "score_prompt_schema_version",
                "visibility_phase_schedule",
            ):
                if field in entry:
                    sample[field] = entry[field]
            samples.append(sample)
        return samples

    def _resolve_manifest_path(self, value: Any, manifest_dir: Path) -> Path:
        path = Path(str(value)).expanduser()
        if path.is_absolute():
                                                                                  
                                                                                   
                                        
            try:
                path.relative_to(self.data_root)
                return path
            except ValueError:
                pass
            if path.exists():
                return path.resolve()

            candidates: list[Path] = []
            train_indices = [index for index, part in enumerate(path.parts) if part == "train"]
            if train_indices:
                train_suffix = Path(*path.parts[train_indices[-1] :])
                candidates.append(self.data_root / train_suffix)
                if self.data_root.name == "train":
                    candidates.append(self.data_root / Path(*train_suffix.parts[1:]))
                                                                                  
                                                                              
            if len(path.parts) >= 2:
                candidates.append(self.data_root / "train" / path.parent.name / path.name)

            resolved_data_root = self.data_root.resolve()
            for candidate in candidates:
                resolved_candidate = candidate.resolve()
                try:
                    resolved_candidate.relative_to(resolved_data_root)
                except ValueError:
                    continue
                if resolved_candidate.exists():
                    return resolved_candidate
            attempted = ", ".join(str(candidate) for candidate in dict.fromkeys(candidates)) or "<none>"
            raise FileNotFoundError(
                f"CrossFPS manifest path does not exist: {path}; rebased candidates tried: {attempted}"
            )
        manifest_relative = (manifest_dir / path).resolve()
        if manifest_relative.exists():
            return manifest_relative
        data_root_relative = (self.data_root / path).resolve()
        if data_root_relative.exists():
            return data_root_relative
        raise FileNotFoundError(
            f"CrossFPS manifest path does not exist: {path}; tried {manifest_relative} and {data_root_relative}"
        )

    def _probe_audio_duration_sec(self, audio_path: Path) -> float | None:
        if audio_path in self._audio_duration_cache:
            return self._audio_duration_cache[audio_path]
        try:
            with av.open(str(audio_path)) as container:
                audio_streams = [stream for stream in container.streams if stream.type == "audio"]
                if not audio_streams:
                    duration = None
                else:
                    stream = audio_streams[0]
                    if stream.duration is not None:
                        duration = float(stream.duration * stream.time_base)
                    elif container.duration is not None:
                        duration = float(container.duration / av.time_base)
                    else:
                        duration = None
        except Exception as exc:
            logger.warning("Failed to probe CrossFPS audio duration for %s: %s", audio_path, exc)
            duration = None
        self._audio_duration_cache[audio_path] = duration
        return duration

    @staticmethod
    def _clean_prompt_text(text: Any) -> str | None:
        return AudioOnlyDataset._clean_prompt_text(text)

    def _pick_frame_window(self, entry: dict[str, Any]) -> tuple[int, torch.Tensor]:
        assert self._target_num_frames is not None
        assert self._target_fps is not None
        source_fps = float(entry["source_fps"])
        if abs(source_fps - float(self._target_fps)) > 1e-3:
            raise ValueError(
                f"CrossFPS source fps must match fixed_clip.fps for integer action alignment: "
                f"source={source_fps}, target={self._target_fps}"
            )

        source_frames = int(entry["source_frames"])
        if source_frames < self._target_num_frames:
            raise ValueError(
                f"CrossFPS sample {entry.get('sample_id')} has {source_frames} frames, "
                f"but fixed_clip.num_frames={self._target_num_frames}"
            )
        max_start = source_frames - self._target_num_frames
        crop_start = random.randint(0, max_start) if self._random_temporal_crop and max_start > 0 else 0
        frame_indices = torch.arange(crop_start, crop_start + self._target_num_frames, dtype=torch.int64)
        return crop_start, frame_indices

    def _decode_crossfps_video(self, entry: dict[str, Any], frame_indices: torch.Tensor) -> torch.Tensor:
        video_path = Path(entry["video_path"])
        source_fps = float(entry.get("source_fps") or self._target_fps or 20.0)
        video_source_fps = float(entry.get("video_source_fps") or source_fps)
        video_offset_sec = float(entry.get("video_offset_sec") or 0.0)
        if abs(video_source_fps - source_fps) > 1e-3 or abs(video_offset_sec) > 1e-6:
            video_indices = ((frame_indices.to(torch.float64) / source_fps + video_offset_sec) * video_source_fps).round()
            frame_indices = video_indices.to(torch.int64)

        target_map: dict[int, list[int]] = defaultdict(list)
        for out_index, source_index in enumerate(frame_indices.tolist()):
            target_map[int(source_index)].append(out_index)

        frames: list[torch.Tensor | None] = [None] * len(frame_indices)
        last_needed = max(target_map)
        first_needed = min(target_map)
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            if first_needed > 0 and video_source_fps > 0 and stream.time_base is not None:
                seek_sec = max((float(first_needed) / video_source_fps) - 1.0, 0.0)
                container.seek(
                    int(seek_sec / float(stream.time_base)),
                    stream=stream,
                    any_frame=False,
                    backward=True,
                )

            fallback_index = 0
            for frame in container.decode(video=0):
                if frame.time is not None and video_source_fps > 0:
                    source_index = int(round(float(frame.time) * video_source_fps))
                else:
                    source_index = fallback_index
                    fallback_index += 1
                if source_index > last_needed:
                    break
                if source_index not in target_map:
                    continue
                frame_tensor = (
                    torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1).float().div(255.0)
                )
                frame_tensor = self._resize_and_center_crop_frame(frame_tensor)
                for out_index in target_map[source_index]:
                    frames[out_index] = frame_tensor

        first_valid = next((frame for frame in frames if frame is not None), None)
        if first_valid is None:
            raise ValueError(f"Failed to decode selected frames from {video_path}")
        for idx in range(len(frames)):
            if frames[idx] is None:
                frames[idx] = frames[idx - 1] if idx > 0 and frames[idx - 1] is not None else first_valid
        return torch.stack([frame for frame in frames if frame is not None])

    def _load_crossfps_audio(self, entry: dict[str, Any], crop_start: int) -> tuple[torch.Tensor, int]:
        assert self._target_num_frames is not None
        assert self._target_fps is not None
        assert self._target_audio_sample_rate is not None
        if not entry.get("audio_path"):
            raise KeyError("CrossFPS audio is required but this sample does not provide `audio_path`.")
        offset_sec = float(
            entry.get("audio_offset_sec")
            if entry.get("audio_offset_sec") is not None
            else entry.get("video_offset_sec") or 0.0
        )
        start_sec = offset_sec + float(crop_start) / float(self._target_fps)
        end_sec = start_sec + float(self._target_num_frames) / float(self._target_fps)
        waveform, sample_rate = self._load_audio_file_window(Path(entry["audio_path"]), start_sec, end_sec)
        return self._normalize_raw_audio(waveform, sample_rate, 0.0, end_sec - start_sec), int(
            self._target_audio_sample_rate
        )

    @staticmethod
    def _load_crossfps_rows(action_path: Path) -> list[dict[str, Any]]:
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:                                             
            raise ModuleNotFoundError(
                "CrossFPS action loading requires `pyarrow`. Install the updated ltx-trainer dependencies."
            ) from exc

        table = pq.read_table(action_path)
        return table.to_pylist()

    @staticmethod
    def _joystick_pair(row: dict[str, Any], key: str) -> tuple[float, float]:
        value = row.get(key)
        if value is None:
            return 0.0, 0.0
        pair = list(value)
        if len(pair) < 2:
            return 0.0, 0.0
        return float(pair[0]), float(pair[1])

    @staticmethod
    def _crossfps_button_values(
        row: dict[str, Any],
        fixed_raw_columns: tuple[str, ...] | None,
    ) -> list[float]:
        if fixed_raw_columns is not None:
            return [float(row.get(column, 0) or 0) for column in fixed_raw_columns]

        values: list[float] = []
        for button in CROSSFPS_BUTTON_COLUMNS:
            aliases = BUTTON_COLUMN_ALIASES.get(button, ())
            value = 0.0
            for column in aliases:
                if column in row:
                    value = float(row.get(column, 0) or 0)
                    break
            values.append(value)
        return values

    @staticmethod
    def _first_standard_frame_at_or_after_raw_row(
        raw_row_index: int,
        *,
        action_row_start: int,
        action_row_scale: float,
    ) -> int:
        ''

        frame_index = int(math.floor(float(raw_row_index) / action_row_scale - action_row_start))

        def mapped(index: int) -> int:
            return int(round((action_row_start + index) * action_row_scale))

        while mapped(frame_index) < raw_row_index:
            frame_index += 1
        while mapped(frame_index - 1) >= raw_row_index:
            frame_index -= 1
        return frame_index

    def _resample_crossfps_button_runs(
        self,
        rows: list[dict[str, Any]],
        *,
        frame_indices: list[int],
        action_row_start: int,
        action_row_scale: float,
        fixed_raw_columns: tuple[str, ...] | None,
    ) -> list[list[float]]:
        ''

        raw_values = [self._crossfps_button_values(row, fixed_raw_columns) for row in rows]
        button_dim = len(raw_values[0])
        output = [[0.0] * button_dim for _ in frame_indices]
        for channel in range(button_dim):
            raw_index = 0
            while raw_index < len(raw_values):
                if raw_values[raw_index][channel] <= 0.0:
                    raw_index += 1
                    continue
                run_start = raw_index
                run_peak = raw_values[raw_index][channel]
                raw_index += 1
                while raw_index < len(raw_values) and raw_values[raw_index][channel] > 0.0:
                    run_peak = max(run_peak, raw_values[raw_index][channel])
                    raw_index += 1
                run_end = raw_index

                target_start = self._first_standard_frame_at_or_after_raw_row(
                    run_start,
                    action_row_start=action_row_start,
                    action_row_scale=action_row_scale,
                )
                target_end = self._first_standard_frame_at_or_after_raw_row(
                    run_end,
                    action_row_start=action_row_start,
                    action_row_scale=action_row_scale,
                )
                                                                                  
                target_end = max(target_end, target_start + 1)
                for output_index, frame_index in enumerate(frame_indices):
                    if target_start <= frame_index < target_end:
                        output[output_index][channel] = max(output[output_index][channel], run_peak)
        return output

    def _load_crossfps_actions(
        self,
        entry: dict[str, Any],
        frame_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = self._load_crossfps_rows(Path(entry["action_path"]))
        if not rows:
            raise ValueError(f"CrossFPS actions for {entry.get('sample_id')} have no rows")
        action_row_start = int(entry.get("action_row_start") or 0)
        raw_action_chunk_size = int(entry.get("raw_action_chunk_size") or 0)
        standard_chunk_action_rows = int(entry.get("standard_chunk_action_rows") or 0)
        action_row_scale = 1.0
        if raw_action_chunk_size > 0 and standard_chunk_action_rows > 0:
            action_row_scale = float(raw_action_chunk_size) / float(standard_chunk_action_rows)
            nearest_integer_scale = round(action_row_scale)
            if nearest_integer_scale > 0 and abs(action_row_scale - nearest_integer_scale) < 0.02:
                action_row_scale = float(nearest_integer_scale)
        mapped_indices = [
            int(round((action_row_start + int(frame_index)) * action_row_scale))
            for frame_index in frame_indices.tolist()
        ]
        button_resampling = str(
            entry.get("button_resampling") or CROSSFPS_BUTTON_RESAMPLING_POINT
        ).strip()
        if button_resampling not in _CROSSFPS_BUTTON_RESAMPLING_MODES:
            raise ValueError(
                f"CrossFPS actions for {entry.get('sample_id')} have unsupported button_resampling="
                f"{button_resampling!r}; expected one of {sorted(_CROSSFPS_BUTTON_RESAMPLING_MODES)}"
            )
        needed_max = max(mapped_indices) if mapped_indices else action_row_start
        allowed_tail_pad = max(2, int(round(action_row_scale)) if action_row_scale > 1.0 else 0)
        if len(rows) <= needed_max and needed_max - (len(rows) - 1) > allowed_tail_pad:
            raise ValueError(
                f"CrossFPS actions for {entry.get('sample_id')} have {len(rows)} rows, "
                f"but frame index {needed_max} is required"
            )

        fixed_raw_columns: tuple[str, ...] | None = None
        if is_crossfps_fixed_button_mode(self._action_binding_mode):
            fixed_raw_columns = crossfps_fixed_raw_button_columns(
                self._raw_semantic_game_id(entry),
                self._action_binding_mode,
            )

        joystick_values: list[list[float]] = []
        point_button_values: list[list[float]] = []
        frame_index_values = [int(frame_index) for frame_index in frame_indices.tolist()]
        for row_index in mapped_indices:
            row_index = min(max(row_index, 0), len(rows) - 1)
            row = rows[row_index]
            left_x, left_y = self._joystick_pair(row, "j_left")
            right_x, right_y = self._joystick_pair(row, "j_right")
            joystick_values.append([left_x, left_y, right_x, right_y])
            point_button_values.append(self._crossfps_button_values(row, fixed_raw_columns))

        if button_resampling == CROSSFPS_BUTTON_RESAMPLING_CAUSAL_RUN_MAX and action_row_scale > 1.0:
            button_values = self._resample_crossfps_button_runs(
                rows,
                frame_indices=frame_index_values,
                action_row_start=action_row_start,
                action_row_scale=action_row_scale,
                fixed_raw_columns=fixed_raw_columns,
            )
        else:
            button_values = point_button_values

        joystick = torch.tensor(joystick_values, dtype=torch.float32)
        joystick = self._normalize_crossfps_joystick(joystick, entry=entry)
        buttons = torch.tensor(button_values, dtype=torch.float32)
        actions = torch.cat([joystick, buttons], dim=-1)
        return actions, joystick, buttons

    def _normalize_crossfps_joystick(
        self,
        joystick: torch.Tensor,
        *,
        entry: dict[str, Any],
    ) -> torch.Tensor:
        config = getattr(self, "_joystick_normalization", None)
        mode = str(getattr(config, "mode", "none")) if config is not None else "none"
        if mode == "none":
            return joystick
        if mode != "fixed_scale_to_unit":
            raise ValueError(f"Unsupported CrossFPS joystick normalization mode: {mode!r}")

        scales = torch.as_tensor(
            getattr(config, "scales", (1.0, 1.0, 1.0, 1.0)),
            dtype=joystick.dtype,
            device=joystick.device,
        )
        if scales.shape != (4,) or not torch.isfinite(scales).all() or not torch.all(scales > 0):
            raise ValueError(f"CrossFPS joystick normalization requires four finite positive scales, got {scales}")
        normalized = joystick / scales
        invalid = ~torch.isfinite(normalized)
        outside = normalized.abs() > 1.000001
        policy = str(getattr(config, "out_of_range_policy", "error"))
        if invalid.any() or (outside.any() and policy == "error"):
            channel_max = normalized.abs().amax(dim=0).tolist() if normalized.numel() else [0.0] * 4
            raise ValueError(
                "CrossFPS normalized joystick is outside [-1, 1] for "
                f"sample {entry.get('sample_id')!r}: abs_max={channel_max}, scales={scales.tolist()}"
            )
        if policy not in {"error", "clip"}:
            raise ValueError(f"Unsupported CrossFPS joystick out_of_range_policy: {policy!r}")
        return normalized.clamp_(-1.0, 1.0)

    def _semantic_game_id(self, entry: dict[str, Any]) -> str | None:
        return normalize_game_id(self._raw_semantic_game_id(entry))

    @staticmethod
    def _raw_semantic_game_id(entry: dict[str, Any]) -> str:
        for key in ("game_id", "game", "selected_game", "source_game", "control_profile_id"):
            value = entry.get(key)
            if value:
                return str(value)
        return ""

    def _map_buttons_to_semantic_vocab(self, entry: dict[str, Any], buttons: torch.Tensor) -> torch.Tensor:
        if buttons.ndim != 2 or buttons.shape[-1] != len(CANONICAL_BUTTONS):
            raise ValueError(
                f"semantic_vocab mode expects canonical buttons [frames, {len(CANONICAL_BUTTONS)}], "
                f"got {tuple(buttons.shape)}"
            )
        if not self._semantic_action_vocab:
            raise ValueError("semantic_vocab mode is active but no semantic action vocab was loaded")

        raw_game_id = self._raw_semantic_game_id(entry)
        game_id = normalize_game_id(raw_game_id)
        semantic_buttons = torch.zeros(
            buttons.shape[0],
            len(self._semantic_action_vocab),
            dtype=buttons.dtype,
            device=buttons.device,
        )
        if not game_id or game_id not in self._semantic_vocab_game_ids:
            warn_key = (raw_game_id, str(game_id or ""))
            if warn_key not in self._warned_unknown_semantic_games:
                self._warned_unknown_semantic_games.add(warn_key)
                logger.warning(
                    "Semantic action vocab has no channels for game_id raw=%r normalized=%r; "
                    "button actions for matching samples will map to all-zero semantic buttons.",
                    raw_game_id,
                    game_id,
                )
            return semantic_buttons

        for button_index, button in enumerate(CANONICAL_BUTTONS):
            vocab_index = self._semantic_vocab_index_by_game_button.get((game_id, button))
            if vocab_index is None:
                if bool((buttons[:, button_index].abs() > self._semantic_vocab_active_threshold).any().item()):
                    warn_key = (game_id, button)
                    if warn_key not in self._warned_dropped_semantic_buttons:
                        self._warned_dropped_semantic_buttons.add(warn_key)
                        logger.warning(
                            "Semantic action vocab does not cover active button game_id=%r button=%s; "
                            "this button channel will be dropped for semantic_vocab mode.",
                            game_id,
                            button,
                        )
                continue
            semantic_buttons[:, vocab_index] = torch.maximum(
                semantic_buttons[:, vocab_index],
                buttons[:, button_index],
            )
        return semantic_buttons

    def _build_timeline_prompt_text(
        self,
        buttons: torch.Tensor,
        button_meaning_texts: Mapping[str, str],
        *,
        fps: float,
        joystick: torch.Tensor | None = None,
    ) -> str:
        timeline_cfg = getattr(self._action_binding, "timeline_prompt", None)
        active_threshold = float(getattr(timeline_cfg, "active_threshold", 0.1))
        quantize_sec = float(getattr(timeline_cfg, "quantize_sec", 0.1))
        prefix = str(getattr(timeline_cfg, "prefix", "Actions:"))
        timeline_text = build_button_action_timeline_text(
            buttons,
            button_meaning_texts,
            fps=fps,
            active_threshold=active_threshold,
            quantize_sec=quantize_sec,
            max_events=int(getattr(timeline_cfg, "max_events", 24)),
            min_event_frames=int(getattr(timeline_cfg, "min_event_frames", 1)),
            point_event_max_frames=int(getattr(timeline_cfg, "point_event_max_frames", 2)),
            prefix=prefix,
        )
        if timeline_text or not bool(getattr(timeline_cfg, "idle_when_no_events", False)):
            return timeline_text

        def format_empty_button_event(text: str) -> str:
            start_sec = float(getattr(timeline_cfg, "idle_start_sec", 0.0))
            end_sec_value = getattr(timeline_cfg, "idle_end_sec", None)
            if end_sec_value is None:
                end_sec = float(buttons.shape[0]) / max(float(fps), 1e-6)
            else:
                end_sec = float(end_sec_value)
            end_sec = max(end_sec, start_sec)

            start_text = format_timeline_seconds(start_sec, quantize_sec)
            end_text = format_timeline_seconds(end_sec, quantize_sec)
            event_text = f"[{start_text}-{end_text}] {text.rstrip('.')}."
            stripped_prefix = prefix.strip()
            return f"{stripped_prefix} {event_text}" if stripped_prefix else event_text

        joystick_active = False
        if joystick is not None:
            joystick_threshold = float(getattr(timeline_cfg, "idle_joystick_threshold", 0.05))
            joystick_active = bool((joystick.detach().abs() > joystick_threshold).any().item())

        if joystick_active:
            if bool(getattr(timeline_cfg, "no_button_when_joystick_active", False)):
                no_button_text = str(getattr(timeline_cfg, "no_button_joystick_text", "") or "").strip()
                if no_button_text:
                    return format_empty_button_event(no_button_text)
            if bool(getattr(timeline_cfg, "idle_requires_quiet_joystick", True)):
                return ""

        idle_text = str(getattr(timeline_cfg, "idle_text", "") or "").strip()
        if not idle_text:
            return ""

        return format_empty_button_event(idle_text)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = _apply_cleaned_prompt(
            self._samples[index],
            index,
            self._cleaned_prompts,
            self._cleaned_prompt_fallback,
        )
        crop_start, frame_indices = self._pick_frame_window(entry)
        video_pixels = self._decode_crossfps_video(entry, frame_indices)
        audio_waveform = None
        audio_sample_rate = None
        if self._require_audio:
            audio_waveform, audio_sample_rate = self._load_crossfps_audio(entry, crop_start)
        actions, joystick, buttons = self._load_crossfps_actions(entry, frame_indices)
        prompt_text = _apply_prompt_mix(str(entry["prompt_text"]), index, self._prompt_mix)
        button_meanings: dict[str, Any] | None = None
        button_meaning_texts: dict[str, str] | None = None
        control_binding_text = ""

        if (
            self._action_binding is not None
            and getattr(self._action_binding, "enabled", False)
            and not is_crossfps_fixed_button_mode(self._action_binding_mode)
        ):
            null_text = getattr(
                getattr(self._action_binding, "button_meaning_embedding", None),
                "null_meaning_text",
                "",
            )
            button_meanings = resolve_button_meanings(entry, self._control_schemas, null_meaning_text=str(null_text))
            rng = random
            remap_cfg = getattr(self._action_binding, "remap", None)
            if (
                self._action_binding_mode == "text_binding"
                and self._action_binding_training
                and remap_cfg is not None
                and getattr(remap_cfg, "enabled", False)
                and rng.random() < float(getattr(remap_cfg, "prob", 0.0))
            ):
                buttons, button_meanings, _ = apply_button_remap(
                    buttons,
                    button_meanings,
                    groups=list(getattr(remap_cfg, "groups", [])),
                    rng=rng,
                )
                actions = torch.cat([joystick, buttons], dim=-1)

            use_scene_prompt = bool(getattr(self._action_binding, "use_scene_prompt", False))
            use_control_prompt = bool(getattr(self._action_binding, "use_control_prompt", True))
            if self._action_binding_mode == "semantic_vocab":
                buttons = self._map_buttons_to_semantic_vocab(entry, buttons)
                actions = torch.cat([joystick, buttons], dim=-1)
                if not use_scene_prompt:
                    prompt_text = ""
            elif self._action_binding_mode == "prompt_timeline":
                timeline_text = self._build_timeline_prompt_text(
                    buttons,
                    button_meanings,
                    fps=float(entry["source_fps"]),
                    joystick=joystick,
                )
                timeline_cfg = getattr(self._action_binding, "timeline_prompt", None)
                if (
                    self._action_binding_training
                    and timeline_text
                    and rng.random() < float(getattr(timeline_cfg, "dropout_prob", 0.0))
                ):
                    timeline_text = ""
                if use_scene_prompt:
                    prompt_parts = [part for part in (prompt_text, timeline_text) if part]
                    prompt_text = "\n".join(prompt_parts)
                else:
                    prompt_text = timeline_text
            else:
                control_binding_text, button_meaning_texts = build_control_binding_text(
                    button_meanings,
                    rng,
                    use_alias=self._action_binding_training,
                    alias_sampling_prob=float(getattr(self._action_binding, "alias_sampling_prob", 0.3)),
                    deterministic=not self._action_binding_training,
                )

                dropout_cfg = getattr(self._action_binding, "dropout", None)
                if (
                    self._action_binding_training
                    and dropout_cfg is not None
                    and control_binding_text
                    and rng.random() < float(getattr(dropout_cfg, "control_prompt_dropout_prob", 0.0))
                ):
                    control_binding_text = ""

                if use_scene_prompt and use_control_prompt:
                    prompt_parts = [part for part in (prompt_text, control_binding_text) if part]
                    prompt_text = "\n".join(prompt_parts)
                elif use_scene_prompt:
                    pass
                elif use_control_prompt:
                    prompt_text = control_binding_text
                else:
                    prompt_text = ""

        sample = {
            "video_pixels": video_pixels,
            "actions": actions,
            "action_joystick": joystick,
            "action_buttons": buttons,
            "action_frame_indices": frame_indices,
            "action_fps": float(entry["source_fps"]),
            "video_path": str(entry["video_path"]),
            "audio_path": str(entry.get("audio_path") or ""),
            "action_path": str(entry["action_path"]),
            "meta_path": str(entry.get("meta_path", "")),
            "crop_start": int(crop_start),
            "source_frames": int(entry["source_frames"]),
            "video_offset_sec": float(entry.get("video_offset_sec") or 0.0),
            "audio_offset_sec": float(entry.get("audio_offset_sec") or 0.0),
            "has_action": True,
            "has_audio": self._require_audio,
            "audio_loss_eligible": bool(entry.get("audio_loss_eligible", True)) and self._require_audio,
            "fps": float(entry["source_fps"]),
            "source": entry.get("source", self._source_name),
            "sample_id": str(entry.get("sample_id", index)),
            "clip_id": str(entry.get("clip_id", entry.get("sample_id", index))),
            "scene_id": str(entry.get("game_id") or entry.get("game") or ""),
            "source_video_id": str(entry.get("clip_id", entry.get("sample_id", index))),
            "prompt_text": prompt_text,
            "crossfps_action_channels": [
                "j_left_x",
                "j_left_y",
                "j_right_x",
                "j_right_y",
                *(
                    [item["id"] for item in self._semantic_action_vocab]
                    if self._action_binding_mode == "semantic_vocab"
                    else (
                        list(crossfps_fixed_button_names(self._action_binding_mode))
                        if is_crossfps_fixed_button_mode(self._action_binding_mode)
                        else list(CROSSFPS_BUTTON_COLUMNS)
                    )
                ),
            ],
            "crossfps_split": str(entry.get("split", "")),
        }
        if audio_waveform is not None:
            sample["audio_waveform"] = audio_waveform
            sample["audio_sample_rate"] = audio_sample_rate
        if button_meanings is not None:
            sample["button_meanings"] = button_meanings
        if button_meaning_texts is not None:
            sample["button_meaning_texts"] = button_meaning_texts
        if control_binding_text:
            sample["control_binding_text"] = control_binding_text
        if entry.get("prompt_schedule") is not None:
            sample["prompt_schedule"] = list(entry["prompt_schedule"])
        if entry.get("score_prompt_schedule") is not None:
            sample["score_prompt_schedule"] = list(entry["score_prompt_schedule"])
        return sample


class CrossFPSLongSlidingRefDataset(CrossFPSAudioActionDataset):
    ''

    _EXPECTED_ACTION_CONTRACTS = frozenset(
        {
            "c25k-crossfps-legacy6-xonotic-v1",
            "c25k-crossfps-legacy6-xonotic-duel-v11-v1",
        }
    )
    _REFERENCE_PROMPT_MARKER = "reference image"

    def __init__(
        self,
        *args: Any,
        history_chunks: int = 4,
        rollout_target_chunks: int = 1,
        chunk_size: int = 2,
        video_temporal_scale: int = 8,
        require_reference_contract: bool = True,
        require_score_prompt_schedule: bool = False,
        **kwargs: Any,
    ) -> None:
        self._sliding_history_chunks = int(history_chunks)
        self._sliding_rollout_target_chunks = int(rollout_target_chunks)
        self._sliding_chunk_size = int(chunk_size)
        self._sliding_video_temporal_scale = int(video_temporal_scale)
        self._sliding_require_reference_contract = bool(require_reference_contract)
        self._sliding_require_score_prompt_schedule = bool(
            require_score_prompt_schedule
        )
        if (
            self._sliding_history_chunks <= 0
            or self._sliding_rollout_target_chunks <= 0
            or self._sliding_chunk_size <= 0
        ):
            raise ValueError(
                "Sliding history_chunks, rollout_target_chunks, and chunk_size must be positive."
            )
        if self._sliding_video_temporal_scale <= 0:
            raise ValueError("Sliding video_temporal_scale must be positive.")
        self._sliding_pixel_frames_per_chunk = self._sliding_chunk_size * self._sliding_video_temporal_scale
        super().__init__(*args, **kwargs)

        expected_frames = 1 + (
            self._sliding_history_chunks + self._sliding_rollout_target_chunks
        ) * self._sliding_pixel_frames_per_chunk
        if int(self._target_num_frames or 0) != expected_frames:
            raise ValueError(
                "Long sliding-ref loader requires one local VAE anchor plus history and target frames: "
                f"expected target_num_frames={expected_frames}, got {self._target_num_frames}."
            )
        for sample in self._samples:
            self._validate_long_manifest_sample(sample)

    def _validate_long_manifest_sample(self, sample: dict[str, Any]) -> None:
        sample_id = str(sample.get("sample_id") or "<unknown>")
        schedule = sample.get("prompt_schedule")
        score_schedule = sample.get("score_prompt_schedule")
        minimum_schedule = (
            self._sliding_history_chunks
            + getattr(self, "_sliding_rollout_target_chunks", 1)
        )
        if not isinstance(schedule, list) or len(schedule) < minimum_schedule:
            raise ValueError(
                f"Long sample {sample_id!r} needs at least {minimum_schedule} prompt blocks."
            )
        if (
            getattr(self, "_sliding_require_score_prompt_schedule", False)
            and score_schedule is None
        ):
            raise ValueError(
                f"Long sample {sample_id!r} is missing the required "
                "score_prompt_schedule."
            )
        if score_schedule is not None:
            if not isinstance(score_schedule, list) or len(score_schedule) != len(schedule):
                raise ValueError(
                    f"Long sample {sample_id!r} has an unaligned score_prompt_schedule."
                )
            if any("reference image" in str(prompt).lower() for prompt in score_schedule):
                raise ValueError(
                    f"Long sample {sample_id!r} score_prompt_schedule still depends on a "
                    "reference image that the bidirectional score model does not receive."
                )
            if any(not has_explicit_tracked_identity(prompt) for prompt in score_schedule):
                raise ValueError(
                    f"Long sample {sample_id!r} has a score prompt without an "
                    "explicit tracked-enemy/opponent identity."
                )

        source_fps = float(sample.get("source_fps") or 0.0)
        target_fps = float(self._target_fps or 0.0)
        if not math.isclose(source_fps, target_fps, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"Long sample {sample_id!r} source_fps={source_fps} does not match target_fps={target_fps}."
            )
        if int(sample.get("prompt_schedule_video_latents_per_chunk") or -1) != self._sliding_chunk_size:
            raise ValueError(
                f"Long sample {sample_id!r} prompt_schedule_video_latents_per_chunk must equal "
                f"chunk_size={self._sliding_chunk_size}."
            )
        chunk_seconds = self._sliding_pixel_frames_per_chunk / target_fps
        chunk_seconds_label = f"{chunk_seconds:g}".replace(".", "p")
        expected_chunking = f"{self._sliding_chunk_size}_latents_{chunk_seconds_label}s"
        if sample.get("prompt_schedule_chunking") != expected_chunking:
            raise ValueError(
                f"Long sample {sample_id!r} prompt_schedule_chunking must be {expected_chunking!r}, "
                f"got {sample.get('prompt_schedule_chunking')!r}."
            )

        frame_ranges = sample.get("prompt_schedule_target_frame_ranges_inclusive")
        if not isinstance(frame_ranges, list) or len(frame_ranges) != len(schedule):
            raise ValueError(f"Long sample {sample_id!r} has no aligned target frame ranges.")
        for block_index, frame_range in enumerate(frame_ranges):
            expected_range = [
                block_index * self._sliding_pixel_frames_per_chunk + 1,
                (block_index + 1) * self._sliding_pixel_frames_per_chunk,
            ]
            if frame_range != expected_range:
                raise ValueError(
                    f"Long sample {sample_id!r} block {block_index} frame range must be "
                    f"{expected_range}, got {frame_range!r}."
                )
        if int(sample.get("source_frames") or 0) <= int(frame_ranges[-1][1]):
            raise ValueError(
                f"Long sample {sample_id!r} source_frames does not cover its final prompt target range."
            )

        if self._action_binding_mode == "crossfps_legacy6":
            expected_channels = (
                "j_left_x",
                "j_left_y",
                "j_right_x",
                "j_right_y",
                *crossfps_fixed_raw_button_columns("xonotic", "crossfps_legacy6"),
            )
            if (
                sample.get("action_contract_version")
                not in self._EXPECTED_ACTION_CONTRACTS
            ):
                raise ValueError(
                    f"Long sample {sample_id!r} action_contract_version must be one of "
                    f"{sorted(self._EXPECTED_ACTION_CONTRACTS)!r}."
                )
            if tuple(sample.get("action_channels") or ()) != expected_channels:
                raise ValueError(
                    f"Long sample {sample_id!r} action_channels do not match crossfps_legacy6: "
                    f"expected={expected_channels}, got={sample.get('action_channels')!r}."
                )

        if not self._sliding_require_reference_contract:
            return
        active_enemy_schedule = sample.get("active_enemy_id_schedule")
        active_ref_schedule = sample.get("active_ref_schedule")
        reference_valid_mask = sample.get("reference_valid_mask")
        reference_index_schedule = sample.get("reference_bank_index_schedule")
        visibility_schedule = sample.get("visibility_phase_schedule")
        reference_bank = sample.get("reference_bank")
        aligned_schedules = {
            "active_enemy_id_schedule": active_enemy_schedule,
            "active_ref_schedule": active_ref_schedule,
            "reference_valid_mask": reference_valid_mask,
            "reference_bank_index_schedule": reference_index_schedule,
            "visibility_phase_schedule": visibility_schedule,
        }
        for name, values in aligned_schedules.items():
            if not isinstance(values, list) or len(values) != len(schedule):
                raise ValueError(f"Long sample {sample_id!r} has no aligned {name}.")
        if not isinstance(reference_bank, list) or not reference_bank:
            raise ValueError(f"Long sample {sample_id!r} has no reference_bank.")
        if sample.get("reference_conditioning_ready") is not True:
            raise ValueError(f"Long sample {sample_id!r} is not reference_conditioning_ready.")

        for block_index, prompt in enumerate(schedule):
            ref_state = active_ref_schedule[block_index]
            if not isinstance(ref_state, dict):
                raise ValueError(
                    f"Long sample {sample_id!r} block {block_index} active_ref_schedule entry is not an object."
                )
            ref_valid = bool(reference_valid_mask[block_index])
            schedule_valid = bool(ref_state.get("valid", False))
            ref_index = int(reference_index_schedule[block_index])
            schedule_ref_index = int(ref_state.get("ref_bank_index", -1))
            if ref_valid != schedule_valid or ref_index != schedule_ref_index or ref_valid != (ref_index >= 0):
                raise ValueError(
                    f"Long sample {sample_id!r} block {block_index} has inconsistent reference validity/index."
                )
            active_enemy_id = active_enemy_schedule[block_index]
            if active_enemy_id is not None:
                if not ref_valid or str(ref_state.get("actor_id") or "") != str(active_enemy_id):
                    raise ValueError(
                        f"Long sample {sample_id!r} block {block_index} has active enemy "
                        "without its matching conditioning reference."
                    )
            if ref_valid:
                if ref_index >= len(reference_bank):
                    raise ValueError(
                        f"Long sample {sample_id!r} block {block_index} reference index {ref_index} is out of range."
                    )
                bank_entry = reference_bank[ref_index]
                if str(ref_state.get("actor_id") or "") != str(bank_entry.get("actor_id") or ""):
                    raise ValueError(
                        f"Long sample {sample_id!r} block {block_index} reference actor does not match its bank entry."
                    )
                if not bank_entry.get("default_image_path") or not bank_entry.get("default_view"):
                    raise ValueError(
                        f"Long sample {sample_id!r} reference bank entry {ref_index} has no fixed default view."
                    )
            elif ref_index != -1 or ref_state.get("actor_id") is not None:
                raise ValueError(
                    f"Long sample {sample_id!r} block {block_index} invalid ref must use index -1 and null actor."
                )

            phase = str(visibility_schedule[block_index])
            if phase == "no_enemy" and (active_enemy_id is not None or ref_valid):
                raise ValueError(
                    f"Long sample {sample_id!r} block {block_index} no_enemy phase must have no active/ref actor."
                )
            if self._REFERENCE_PROMPT_MARKER in str(prompt).lower() and not ref_valid:
                raise ValueError(
                    f"Long sample {sample_id!r} block {block_index} prompt refers to a reference image "
                    "but active_ref_schedule marks the reference invalid."
                )

    def _pick_frame_window(self, entry: dict[str, Any]) -> tuple[int, torch.Tensor]:
        schedule = entry.get("prompt_schedule")
        minimum_schedule = (
            self._sliding_history_chunks
            + getattr(self, "_sliding_rollout_target_chunks", 1)
        )
        if not isinstance(schedule, list) or len(schedule) < minimum_schedule:
            raise ValueError(f"Invalid long prompt schedule for {entry.get('sample_id')!r}.")
        min_target_block = self._sliding_history_chunks
        rollout_target_chunks = getattr(self, "_sliding_rollout_target_chunks", 1)
        max_first_target_block = len(schedule) - rollout_target_chunks
        first_target_block = (
            random.randint(min_target_block, max_first_target_block)
            if self._random_temporal_crop and max_first_target_block > min_target_block
            else min_target_block
        )
        window_start_block = first_target_block - self._sliding_history_chunks
        crop_start = window_start_block * self._sliding_pixel_frames_per_chunk
        assert self._target_num_frames is not None
        frame_indices = torch.arange(crop_start, crop_start + self._target_num_frames, dtype=torch.int64)
        if int(frame_indices[-1]) >= int(entry["source_frames"]):
            raise ValueError(
                f"Long sample {entry.get('sample_id')!r} rollout beginning at target block "
                f"{first_target_block} exceeds source frames."
            )
        return crop_start, frame_indices

    @staticmethod
    def _decode_reference_image(path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Reference image does not exist: {path}")
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        pixels = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        if pixels.shape[-2] % 32 or pixels.shape[-1] % 32:
            raise ValueError(
                f"Reference image dimensions must be divisible by 32, got {tuple(pixels.shape[-2:])}: {path}"
            )
        return pixels

    @staticmethod
    def _null_reference_from_bank(reference_bank: list[dict[str, Any]]) -> torch.Tensor:
        first = reference_bank[0]
        images = first.get("images")
        image_meta = images[0] if isinstance(images, list) and images else {}
        height = int(image_meta.get("height") or 512)
        width = int(image_meta.get("width") or 512)
        if height % 32 or width % 32:
            raise ValueError(f"Null reference dimensions must be divisible by 32, got {(height, width)}.")
        return torch.zeros(3, height, width, dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        entry = self._samples[index]
        crop_start = int(sample["crop_start"])
        window_start_block = crop_start // self._sliding_pixel_frames_per_chunk
        first_target_block = window_start_block + self._sliding_history_chunks
        target_blocks = list(
            range(
                first_target_block,
                first_target_block
                + getattr(self, "_sliding_rollout_target_chunks", 1),
            )
        )
        final_target_block = target_blocks[-1]

        full_schedule = entry.get("prompt_schedule")
        assert isinstance(full_schedule, list)
        local_schedule = list(
            full_schedule[window_start_block : final_target_block + 1]
        )
        expected_prompt_count = (
            self._sliding_history_chunks
            + getattr(self, "_sliding_rollout_target_chunks", 1)
        )
        if len(local_schedule) != expected_prompt_count:
            raise ValueError(
                f"Failed to slice {expected_prompt_count} local prompts for "
                f"{entry.get('sample_id')!r} rollout ending at block {final_target_block}."
            )
        sample["prompt_schedule"] = local_schedule
        sample["prompt_text"] = local_schedule[-1]
        full_score_schedule = entry.get("score_prompt_schedule")
        if full_score_schedule is not None:
            if not isinstance(full_score_schedule, list):
                raise TypeError(
                    f"Invalid score_prompt_schedule for {entry.get('sample_id')!r}."
                )
            local_score_schedule = list(
                full_score_schedule[window_start_block : final_target_block + 1]
            )
            if len(local_score_schedule) != expected_prompt_count:
                raise ValueError(
                    f"Failed to slice {expected_prompt_count} local score prompts for "
                    f"{entry.get('sample_id')!r}."
                )
            sample["score_prompt_schedule"] = local_score_schedule
            sample["score_prompt_text"] = local_score_schedule[-1]

        if crop_start == 0:
            sink_pixels = sample["video_pixels"][:1].clone()
        else:
            sink_pixels = self._decode_crossfps_video(entry, torch.tensor([0], dtype=torch.int64))
        sample["sink_video_pixels"] = sink_pixels

        active_schedule = entry.get("active_ref_schedule")
        reference_bank = entry.get("reference_bank")
        if not isinstance(active_schedule, list) or not isinstance(reference_bank, list) or not reference_bank:
            if self._sliding_require_reference_contract:
                raise ValueError(f"Missing reference contract for {entry.get('sample_id')!r}.")
            active_ref: dict[str, Any] = {"valid": False, "ref_bank_index": -1}
            reference_bank = [{}]
        else:
            active_ref = {"valid": False, "ref_bank_index": -1}

        rollout_reference_pixels: list[torch.Tensor] = []
        rollout_reference_present: list[bool] = []
        rollout_reference_image_paths: list[str] = []
        rollout_reference_views: list[str] = []
        rollout_reference_actor_ids: list[str] = []
        rollout_reference_appearance_ids: list[str] = []
        rollout_reference_bank_indices: list[int] = []
        for target_block in target_blocks:
            if isinstance(active_schedule, list):
                active_ref = active_schedule[target_block]
            ref_bank_index = int(active_ref.get("ref_bank_index", -1))
            reference_present = (
                bool(active_ref.get("valid", False)) and ref_bank_index >= 0
            )
            reference_image_path = ""
            reference_view = "none"
            reference_actor_id = ""
            reference_appearance_id = ""
            if reference_present:
                if ref_bank_index >= len(reference_bank):
                    raise ValueError(
                        f"Reference bank index {ref_bank_index} is out of range for "
                        f"{entry.get('sample_id')!r}."
                    )
                bank_entry = reference_bank[ref_bank_index]
                reference_image_path = str(
                    bank_entry.get("default_image_path") or ""
                )
                if not reference_image_path:
                    raise ValueError(
                        f"Reference bank entry {ref_bank_index} has no default_image_path."
                    )
                reference_view = str(bank_entry.get("default_view") or "front")
                reference_actor_id = str(bank_entry.get("actor_id") or "")
                reference_appearance_id = str(
                    bank_entry.get("appearance_id") or ""
                )
                if active_ref.get("actor_id") not in (None, reference_actor_id):
                    raise ValueError(
                        f"Active reference actor mismatch for {entry.get('sample_id')!r}: "
                        f"schedule={active_ref.get('actor_id')!r} "
                        f"bank={reference_actor_id!r}."
                    )
                reference_pixels = self._decode_reference_image(
                    Path(reference_image_path)
                )
            else:
                reference_pixels = self._null_reference_from_bank(reference_bank)
            rollout_reference_pixels.append(reference_pixels.unsqueeze(0))
            rollout_reference_present.append(reference_present)
            rollout_reference_image_paths.append(reference_image_path)
            rollout_reference_views.append(reference_view)
            rollout_reference_actor_ids.append(reference_actor_id)
            rollout_reference_appearance_ids.append(reference_appearance_id)
            rollout_reference_bank_indices.append(ref_bank_index)

        reference_pixels = rollout_reference_pixels[-1].squeeze(0)
        reference_present = rollout_reference_present[-1]
        reference_image_path = rollout_reference_image_paths[-1]
        reference_view = rollout_reference_views[-1]
        reference_actor_id = rollout_reference_actor_ids[-1]
        reference_appearance_id = rollout_reference_appearance_ids[-1]
        ref_bank_index = rollout_reference_bank_indices[-1]

        joystick = sample["action_joystick"]
        buttons = sample["action_buttons"]
        if not torch.isfinite(joystick).all() or bool((joystick.abs() > 1.000001).any()):
            raise ValueError(
                f"Long sample {entry.get('sample_id')!r} joystick is not finite or exceeds the [-1,1] contract."
            )
        if not torch.isfinite(buttons).all() or bool(((buttons < 0) | (buttons > 1)).any()):
            raise ValueError(f"Long sample {entry.get('sample_id')!r} buttons are outside [0,1].")

        sample.update(
            {
                "reference_video_pixels": reference_pixels.unsqueeze(0),
                "reference_present": reference_present,
                "reference_image_path": reference_image_path,
                "reference_view": reference_view,
                "reference_actor_id": reference_actor_id,
                "reference_appearance_id": reference_appearance_id,
                "reference_bank_index": ref_bank_index,
                "rollout_reference_video_pixels": torch.stack(
                    rollout_reference_pixels, dim=0
                ),
                "rollout_reference_present": torch.tensor(
                    rollout_reference_present, dtype=torch.bool
                ),
                "rollout_reference_image_paths": rollout_reference_image_paths,
                "rollout_reference_views": rollout_reference_views,
                "rollout_reference_actor_ids": rollout_reference_actor_ids,
                "rollout_reference_appearance_ids": rollout_reference_appearance_ids,
                "rollout_reference_bank_indices": rollout_reference_bank_indices,
                "window_start_block": window_start_block,
                "target_block_index": final_target_block,
                "target_block_indices": target_blocks,
                "absolute_prompt_block_indices": list(
                    range(window_start_block, final_target_block + 1)
                ),
                "sliding_window_contract": (
                    "sink1_history4_target1_ref1_v1"
                    if getattr(self, "_sliding_rollout_target_chunks", 1) == 1
                    else "sink1_history4_multitarget_ref_per_target_v1"
                ),
            }
        )
        return sample


class CompositeDataset(Dataset):
    def __len__(self) -> int:
        raise NotImplementedError("CompositeDataset is reserved for P2 and not implemented in P0.")

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError("CompositeDataset is reserved for P2 and not implemented in P0.")


def _stack_optional_tensors(values: list[Any]) -> torch.Tensor | list[Any] | None:
    if all(value is None for value in values):
        return None
    tensor_values = [value for value in values if isinstance(value, torch.Tensor)]
    if not tensor_values or len(tensor_values) != sum(value is not None for value in values):
        return values

    fill = torch.zeros_like(tensor_values[0])
    tensors = [value if isinstance(value, torch.Tensor) else fill.clone() for value in values]
    shapes = {tuple(value.shape) for value in tensors}
    return torch.stack(tensors) if len(shapes) == 1 else tensors


def collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    keys = sorted({key for sample in samples for key in sample})
    for key in keys:
        values = [sample.get(key) for sample in samples]
        if key in {"has_audio", "has_action", "audio_loss_eligible", "reference_present"}:
            batch[key] = torch.tensor(values, dtype=torch.bool)
        elif key in {"fps", "action_fps"}:
            batch[key] = torch.tensor(values, dtype=torch.float32)
        elif key == "audio_sample_rate":
            if all(value is None for value in values):
                batch[key] = None
            else:
                present_rates = [int(value) for value in values if value is not None]
                unique_rates = set(present_rates)
                if len(unique_rates) != 1:
                    raise ValueError(
                        f"Mixed `audio_sample_rate` values within one batch are not supported: {sorted(unique_rates)}"
                    )
                rate = present_rates[0]
                batch[key] = torch.tensor([rate if value is None else int(value) for value in values], dtype=torch.int64)
        elif key == "audio_waveform":
            batch[key] = _stack_optional_tensors(values)
        else:
            batch[key] = _stack_optional_tensors(values)
            if batch[key] is None:
                continue
            if not isinstance(batch[key], torch.Tensor):
                batch[key] = values
    return batch
