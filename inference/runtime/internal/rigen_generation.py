                      
''

from __future__ import annotations

import argparse
import concurrent.futures as futures
import ctypes
import datetime as dt
import gc
import json
import math
import os
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
for _rel_src in (
    "packages/ltx-trainer/src",
    "packages/ltx-core/src",
    "packages/ltx-pipelines/src",
):
    _src_path = REPO_ROOT / _rel_src
    if str(_src_path) not in sys.path:
        sys.path.insert(0, str(_src_path))

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from safetensors.torch import load_file

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_trainer.action_normalization import (
    JoystickUnitScaleProfile,
    load_joystick_unit_scale_profile,
    normalized_unit_joystick_to_model_input,
)
from ltx_trainer.action_binding import (
    BUTTON_COLUMN_ALIASES,
    CANONICAL_BUTTONS,
    CROSS_FPS_UNIFIED_BUTTONS,
    LEGACY_CROSSFPS_BUTTONS,
    build_button_action_timeline_text,
    canonicalize_button_tensor,
    crossfps_unified_raw_button_columns,
    load_control_schemas,
    load_semantic_action_vocab,
    normalize_button_name,
    normalize_game_id,
    resolve_button_meanings,
)
from ltx_trainer.config import ValidationConfig
from ltx_trainer.model_loader import (
    load_audio_vae_decoder,
    load_embeddings_processor,
    load_model,
    load_text_encoder,
    load_video_vae_decoder,
    load_video_vae_encoder,
    load_vocoder,
)
from ltx_trainer.validation_sampler import GenerationConfig, TiledDecodingConfig, ValidationSampler
from ltx_trainer.video_utils import save_video


BUTTON_COLUMNS = (
    "right_trigger",
    "left_trigger",
    "right_thumb",
    "south",
    "west",
    "north",
)
LEGACY_BUTTON_LABELS = ("RT", "LT", "R3", "A", "X", "Y")
DEFAULT_NUM_FRAMES = 81
DEFAULT_FPS = 20
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 480
DEFAULT_NEGATIVE_PROMPT = "low quality, distorted, blurry, text, watermark"
DEFAULT_RANDOM_REAL_CASES = 16
DEFAULT_JOYSTICK_REVERSE_CASES = 4
MANIFEST_DATA_PATH_FIELDS = (
    "video_path",
    "audio_path",
    "action_path",
    "meta_path",
    "qa_path",
    "frames_path",
)
LONG_TAIL_EVAL_PROMPT_TERMS = (
    "third-person",
    "third person",
    "spectating",
    "spectator",
    "killcam",
    "pre-game lobby",
    "pregame lobby",
    "falling through the air",
    "parachute",
    "parachuting",
    "aerial view",
    "overhead",
    "top-down",
    "top down",
    "map view",
    "tactical map",
    "helicopter",
    "plane",
    "aircraft",
    "propeller",
    "banshee",
    "vehicle",
    "drone",
    "reconnaissance drone",
    "piloting",
    "driving",
    "riding",
    "inside a helicopter",
    "operating a small",
    "operating a red hover",
    "operating a red",
    "glowing red object",
    "mechanical structure",
    "mechanical enemy",
    "unarmed",
    "face-cam",
    "face cam",
    "facecam",
    "webcam",
    "streamer",
)


def adapt_eval_joystick_input(
    joystick: torch.Tensor,
    *,
    input_space: str,
    profile: JoystickUnitScaleProfile | None,
) -> torch.Tensor:
    ''
    if input_space == "model_raw":
        if profile is not None:
            raise ValueError("model_raw joystick input must not use a normalization profile")
        return joystick
    if input_space == "normalized_unit":
        if profile is None:
            raise ValueError("normalized_unit joystick input requires a normalization profile")
        return normalized_unit_joystick_to_model_input(joystick, profile)
    raise ValueError(f"Unsupported joystick input space: {input_space!r}")


def action_tensor_stats(
    joystick: torch.Tensor,
    buttons: torch.Tensor,
) -> dict[str, float | int | list[float]]:
    return {
        "joystick_abs_mean": float(joystick.abs().mean().item()),
        "joystick_abs_max": float(joystick.abs().max().item()),
        "button_active_frames": int((buttons > 0.05).any(dim=1).sum().item()),
        "button_max": float(buttons.max().item()),
        "joystick_min": [float(value) for value in joystick.amin(dim=0).tolist()],
        "joystick_max": [float(value) for value in joystick.amax(dim=0).tolist()],
        "joystick_abs_mean_by_channel": [
            float(value) for value in joystick.abs().mean(dim=0).tolist()
        ],
        "button_sum": [float(value) for value in buttons.sum(dim=0).tolist()],
    }
PRIMARY_FPS_EVAL_PROMPT_TERMS = (
    "first-person",
    "first person",
    "holding",
    "wielding",
    "equipped with",
    "aiming",
    "aiming down",
    "targeting",
    "rifle",
    "gun",
    "weapon",
    "firearm",
    "pistol",
    "submachine gun",
    "scope",
    "sight",
    "crosshair",
    "ammo",
    "rounds",
    "magazine",
    "motion tracker",
)
ACTION_COMPONENTS = {
    "left_x": (0, "joystick"),
    "left_y": (1, "joystick"),
    "right_x": (2, "joystick"),
    "right_y": (3, "joystick"),
    "right_trigger": (0, "button"),
    "left_trigger": (1, "button"),
    "right_thumb": (2, "button"),
    "south": (3, "button"),
    "west": (4, "button"),
    "north": (5, "button"),
    "A": (0, "canonical_button"),
    "B": (1, "canonical_button"),
    "X": (2, "canonical_button"),
    "Y": (3, "canonical_button"),
    "LB": (4, "canonical_button"),
    "RB": (5, "canonical_button"),
    "LT": (6, "canonical_button"),
    "RT": (7, "canonical_button"),
    **{
        button: (index, "crossfps_unified_button")
        for index, button in enumerate(CROSS_FPS_UNIFIED_BUTTONS)
    },
}


@dataclass
class EvalActionConditioning:
    mode: str
    model_button_dim: int
    semantic_vocab: list[dict[str, str]]
    semantic_index_by_game_button: dict[tuple[str, str], int]
    timeline_prompt: dict[str, Any]
    control_schemas: dict[str, dict[str, dict[str, Any]]]
    use_scene_prompt: bool


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _guider_params_to_dict(params: MultiModalGuiderParams) -> dict[str, Any]:
    return asdict(params)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_manifest_row_paths(row: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    ''

    resolved = dict(row)
    base_dir = manifest_path.expanduser().resolve().parent
    for field in MANIFEST_DATA_PATH_FIELDS:
        value = resolved.get(field)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        resolved[field] = str(path.resolve())
    return resolved


def read_manifest_jsonl(path: Path) -> list[dict[str, Any]]:
    manifest_path = path.expanduser().resolve()
    return [resolve_manifest_row_paths(row, manifest_path) for row in read_jsonl(manifest_path)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_step(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.name)
    if not match:
        return -1
    return int(match.group(1))


def find_latest_checkpoint(config: dict[str, Any], explicit: str | None) -> Path:
    if explicit:
        ckpt = Path(explicit).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {ckpt}")
        return ckpt

    output_dir = Path(config["output_dir"]).expanduser()
    ckpt_dir = output_dir / "checkpoints"
    candidates = sorted(ckpt_dir.glob("model_weights_step_*.safetensors"), key=parse_step)
    if not candidates:
        raise FileNotFoundError(f"No model_weights_step_*.safetensors found under {ckpt_dir}")
    return candidates[-1].resolve()


def resolve_manifest(config: dict[str, Any], explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    manifest = None
    dataset = config.get("dataset", {}) or {}
    if isinstance(dataset, dict):
        manifest = dataset.get("manifest_path")
    data = config.get("data", {}) or {}
    for source in data.get("sources", []) or []:
        if source.get("kind") == "crossfps_audio_action" and source.get("manifest_path"):
            manifest = source["manifest_path"]
            break
    if manifest is None:
        for source in data.get("sources", []) or []:
            if source.get("manifest_path"):
                manifest = source["manifest_path"]
                break
    if not manifest:
        raise ValueError("Missing dataset.manifest_path in config; pass --manifest-path explicitly.")
    return Path(manifest).expanduser().resolve()


def resolve_train_membership_manifest(config: dict[str, Any], explicit: str | None) -> Path | None:
    if explicit:
        if explicit.lower() in {"none", "off", "disable", "disabled"}:
            return None
        return Path(explicit).expanduser().resolve()
    try:
        return resolve_manifest(config, None)
    except Exception:
        return None


def _membership_ids(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("sample_id", "clip_id", "anchor_id"):
        value = row.get(key)
        if value is not None:
            text = str(value)
            if text:
                ids.append(text)
    clip_dir = row.get("clip_dir")
    if clip_dir:
        ids.append(Path(str(clip_dir)).name)
    for key in ("video_path", "audio_path", "action_path", "meta_path"):
        value = row.get(key)
        if value:
            parent = Path(str(value)).parent.name
            if parent:
                ids.append(parent)
    deduped: list[str] = []
    seen: set[str] = set()
    for value in ids:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def load_manifest_membership_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for row in read_jsonl(path):
        ids.update(_membership_ids(row))
    return ids


def annotate_train_membership(
    cases: list[dict[str, Any]],
    *,
    membership_manifest_path: Path,
    membership_ids: set[str],
) -> dict[str, Any]:
    counts = {"in_train_manifest": 0, "filtered_out_of_train_manifest": 0}
    for case in cases:
        candidate_ids = _membership_ids(case)
        is_member = any(candidate in membership_ids for candidate in candidate_ids)
        label = "in_train_manifest" if is_member else "filtered_out_of_train_manifest"
        counts[label] += 1
        case["train_manifest_member"] = bool(is_member)
        case["train_membership"] = label
        case["train_membership_key"] = candidate_ids[0] if candidate_ids else str(case.get("case_id") or "")
        case["train_membership_manifest_path"] = str(membership_manifest_path)
    return {
        "membership_manifest_path": str(membership_manifest_path),
        "membership_id_count": len(membership_ids),
        "num_cases": len(cases),
        "counts": counts,
    }


def resolve_base_model(config: dict[str, Any], explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    model = config.get("model", {})
    model_path = model.get("model_path") or model.get("pretrained_model_path")
    if not model_path:
        raise ValueError("Missing model.model_path in config; pass --base-model explicitly.")
    return Path(model_path).expanduser().resolve()


def resolve_text_encoder(config: dict[str, Any], explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser().resolve()
    model = config.get("model", {})
    text_encoder = model.get("text_encoder_path")
    if not text_encoder:
        return None
    return Path(text_encoder).expanduser().resolve()


def resolve_config_path(value: str | None, config_path: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [
        (Path.cwd() / path).resolve(),
        (REPO_ROOT / path).resolve(),
        (config_path.parent / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _action_binding_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("action_binding") or {}
    return cfg if isinstance(cfg, dict) else {}


def resolve_eval_action_mode(
    *,
    config: dict[str, Any],
    cli_mode: str,
    model_button_dim: int,
    explicit_semantic_vocab_path: str | None,
) -> str:
    if cli_mode != "auto":
        return cli_mode

    action_cfg = _action_binding_config(config)
    if action_cfg.get("enabled", False):
        mode = str(action_cfg.get("button_conditioning_mode") or "text_binding")
        if mode in {
            "text_binding",
            "semantic_vocab",
            "prompt_timeline",
            "crossfps_unified",
            "crossfps_legacy6",
        }:
            return mode

    if explicit_semantic_vocab_path:
        return "semantic_vocab"
    if model_button_dim == len(CROSS_FPS_UNIFIED_BUTTONS):
        return "crossfps_unified"
    if model_button_dim == len(LEGACY_CROSSFPS_BUTTONS):
        return "crossfps_legacy6"
    if model_button_dim != len(CANONICAL_BUTTONS):
        return "semantic_vocab"
    return "text_binding"


def _semantic_vocab_path_from_config(
    config: dict[str, Any],
    config_path: Path,
    explicit_path: str | None,
) -> Path | None:
    if explicit_path:
        return resolve_config_path(explicit_path, config_path)
    action_cfg = _action_binding_config(config)
    semantic_cfg = action_cfg.get("semantic_vocab") or {}
    if isinstance(semantic_cfg, dict):
        return resolve_config_path(semantic_cfg.get("path"), config_path)
    return None


def _schema_path_from_config(config: dict[str, Any], config_path: Path) -> Path | None:
    action_cfg = _action_binding_config(config)
    value = action_cfg.get("schema_path")
    return resolve_config_path(value, config_path) if value else None


def _scheduler_timestep_shift_from_config(config: dict[str, Any]) -> float | None:
    flow_cfg = config.get("flow_matching") or {}
    if not isinstance(flow_cfg, Mapping):
        return None
    sigma_cfg = flow_cfg.get("sigma_sampling") or {}
    if not isinstance(sigma_cfg, Mapping):
        return None
    if str(sigma_cfg.get("mode") or "") != "timestep_shift":
        return None
    value = sigma_cfg.get("timestep_shift")
    return None if value is None else float(value)


def _action_button_temporal_alignment_from_config(config: dict[str, Any], cli_value: str) -> str:
    if cli_value != "auto":
        return cli_value
    strategy_cfg = config.get("training_strategy") or {}
    if isinstance(strategy_cfg, Mapping):
        value = str(strategy_cfg.get("action_button_temporal_alignment") or "global")
        if value in {"global", "vae_window", "causal_prefix_rope"}:
            return value
    return "global"


def build_eval_action_conditioning(
    *,
    config: dict[str, Any],
    config_path: Path,
    cli_mode: str,
    model_button_dim: int,
    explicit_semantic_vocab_path: str | None,
) -> EvalActionConditioning:
    mode = resolve_eval_action_mode(
        config=config,
        cli_mode=cli_mode,
        model_button_dim=model_button_dim,
        explicit_semantic_vocab_path=explicit_semantic_vocab_path,
    )
    action_cfg = _action_binding_config(config)
    timeline_prompt = action_cfg.get("timeline_prompt") if isinstance(action_cfg.get("timeline_prompt"), dict) else {}
    use_scene_prompt = bool(action_cfg.get("use_scene_prompt", False))

    semantic_vocab: list[dict[str, str]] = []
    semantic_index: dict[tuple[str, str], int] = {}
    if mode == "crossfps_unified":
        if int(model_button_dim) != len(CROSS_FPS_UNIFIED_BUTTONS):
            raise ValueError(
                "crossfps_unified eval requires a checkpoint with action_button_dim="
                f"{len(CROSS_FPS_UNIFIED_BUTTONS)}, got {model_button_dim}"
            )
    elif mode == "crossfps_legacy6":
        if int(model_button_dim) != len(LEGACY_CROSSFPS_BUTTONS):
            raise ValueError(
                "crossfps_legacy6 eval requires a checkpoint with action_button_dim="
                f"{len(LEGACY_CROSSFPS_BUTTONS)}, got {model_button_dim}"
            )
    elif mode == "semantic_vocab":
        vocab_path = _semantic_vocab_path_from_config(config, config_path, explicit_semantic_vocab_path)
        if vocab_path is None:
            raise ValueError(
                "semantic_vocab eval requires --semantic-action-vocab-path or action_binding.semantic_vocab.path"
            )
        semantic_vocab = load_semantic_action_vocab(vocab_path)
        if len(semantic_vocab) != int(model_button_dim):
            raise ValueError(
                "Semantic action vocab size does not match model action_button_dim: "
                f"vocab={len(semantic_vocab)} model={model_button_dim} path={vocab_path}"
            )
        semantic_index = {
            (item["game_id"], normalize_button_name(item["button"]) or item["button"]): index
            for index, item in enumerate(semantic_vocab)
        }

    schema_path = _schema_path_from_config(config, config_path)
    control_schemas = load_control_schemas(schema_path) if schema_path is not None else load_control_schemas(None)
    return EvalActionConditioning(
        mode=mode,
        model_button_dim=int(model_button_dim),
        semantic_vocab=semantic_vocab,
        semantic_index_by_game_button=semantic_index,
        timeline_prompt=dict(timeline_prompt or {}),
        control_schemas=control_schemas,
        use_scene_prompt=use_scene_prompt,
    )


def _case_raw_game_id(case: dict[str, Any]) -> str:
    for key in ("game_id", "selected_game", "game", "source_game", "control_profile_id"):
        value = case.get(key)
        if value:
            return str(value)
    return ""


def map_eval_buttons_to_semantic_vocab(
    case: dict[str, Any],
    buttons: torch.Tensor,
    action_context: EvalActionConditioning,
) -> torch.Tensor:
    canonical_buttons = canonicalize_button_tensor(buttons)
    if canonical_buttons.ndim != 2 or canonical_buttons.shape[-1] != len(CANONICAL_BUTTONS):
        raise ValueError(
            "semantic_vocab eval expects canonical button tensor "
            f"[frames, {len(CANONICAL_BUTTONS)}], got {tuple(canonical_buttons.shape)}"
        )

    raw_game_id = _case_raw_game_id(case)
    game_id = normalize_game_id(raw_game_id)
    if not game_id:
        raise ValueError(f"semantic_vocab eval case {case.get('case_id')} is missing a usable game id")

    semantic_buttons = torch.zeros(
        canonical_buttons.shape[0],
        len(action_context.semantic_vocab),
        dtype=canonical_buttons.dtype,
        device=canonical_buttons.device,
    )
    matched = 0
    for button_index, button in enumerate(CANONICAL_BUTTONS):
        vocab_index = action_context.semantic_index_by_game_button.get((game_id, button))
        if vocab_index is None:
            continue
        matched += 1
        semantic_buttons[:, vocab_index] = torch.maximum(
            semantic_buttons[:, vocab_index],
            canonical_buttons[:, button_index],
        )
    if matched == 0:
        raise ValueError(
            f"semantic_vocab eval case {case.get('case_id')} game={raw_game_id!r} normalized={game_id!r} "
            "matched zero vocab channels"
        )
    return semantic_buttons


def validate_crossfps_unified_eval_buttons(
    buttons: torch.Tensor,
    action_context: EvalActionConditioning,
) -> torch.Tensor:
    expected_dim = len(CROSS_FPS_UNIFIED_BUTTONS)
    if action_context.model_button_dim != expected_dim:
        raise ValueError(
            "crossfps_unified checkpoint button dimension mismatch: "
            f"expected {expected_dim}, got {action_context.model_button_dim}"
        )
    if buttons.ndim != 2 or buttons.shape[-1] != expected_dim:
        raise ValueError(
            "crossfps_unified eval expects fixed button tensor "
            f"[frames, {expected_dim}] ({', '.join(CROSS_FPS_UNIFIED_BUTTONS)}), "
            f"got {tuple(buttons.shape)}"
        )
    return buttons


def _normalize_button_meaning_texts(values: Mapping[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in values.items():
        button = normalize_button_name(key)
        if button is None:
            continue
        normalized[button] = str(value or "")
    return normalized


def case_button_meanings_for_timeline(
    case: dict[str, Any],
    action_context: EvalActionConditioning,
) -> dict[str, Any]:
    explicit = case.get("button_meaning_texts")
    if isinstance(explicit, Mapping) and explicit:
        return _normalize_button_meaning_texts(explicit)

    sample = dict(case)
    sample.setdefault("game_id", case.get("game"))
    return resolve_button_meanings(sample, action_context.control_schemas, null_meaning_text="")


def build_eval_prompt_timeline(
    case: dict[str, Any],
    buttons: torch.Tensor,
    source_fps: float,
    action_context: EvalActionConditioning,
) -> tuple[str, str]:
    canonical_buttons = canonicalize_button_tensor(buttons)
    meanings = case_button_meanings_for_timeline(case, action_context)
    timeline_cfg = action_context.timeline_prompt
    timeline_text = build_button_action_timeline_text(
        canonical_buttons,
        meanings,
        fps=source_fps,
        active_threshold=float(timeline_cfg.get("active_threshold", 0.1)),
        quantize_sec=float(timeline_cfg.get("quantize_sec", 0.1)),
        max_events=int(timeline_cfg.get("max_events", 24)),
        min_event_frames=int(timeline_cfg.get("min_event_frames", 1)),
        point_event_max_frames=int(timeline_cfg.get("point_event_max_frames", 2)),
        prefix=str(timeline_cfg.get("prefix", "Actions:")),
    )
    timeline_override = (
        case.get("prompt_timeline_text_override")
        or case.get("action_timeline_text_override")
        or case.get("timeline_text_override")
    )
    if isinstance(timeline_override, str):
        timeline_text = timeline_override.strip()
    elif timeline_text:
        action_details = case.get("timeline_action_details")
        if isinstance(action_details, Mapping):
            for meaning, detail in sorted(action_details.items(), key=lambda item: len(str(item[0])), reverse=True):
                meaning_text = str(meaning or "").strip()
                detail_text = str(detail or "").strip()
                if not meaning_text or not detail_text:
                    continue
                timeline_text = timeline_text.replace(
                    f"] {meaning_text}.",
                    f"] {meaning_text}: {detail_text}.",
                )
    scene_prompt = str(case.get("prompt") or "")
    if action_context.use_scene_prompt:
        prompt = "\n".join(part for part in (scene_prompt, timeline_text) if part)
    else:
        prompt = timeline_text
    return prompt, timeline_text


def _pair_to_xy(value: Any) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    if isinstance(value, dict):
        return float(value.get("x", 0.0) or 0.0), float(value.get("y", 0.0) or 0.0)
    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            return float(value[0] or 0.0), float(value[1] or 0.0)
        if len(value) == 1:
            return float(value[0] or 0.0), 0.0
    return 0.0, 0.0


def _parse_fps(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            den = float(denominator)
            return float(numerator) / den if den else float(default)
        except ValueError:
            return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _action_row_scale_from_row(row: dict[str, Any]) -> float:
    raw_rows = int(row.get("raw_action_chunk_size") or 0)
    standard_rows = int(row.get("standard_chunk_action_rows") or 0)
    if raw_rows > 0 and standard_rows > 0:
        scale = float(raw_rows) / float(standard_rows)
        nearest_integer_scale = round(scale)
        if nearest_integer_scale > 0 and abs(scale - nearest_integer_scale) < 0.02:
            return float(nearest_integer_scale)
        return scale
    return 1.0


def _source_prompt_from_manifest_row(row: dict[str, Any]) -> str:
    for key in ("training_prompt_static", "caption", "prompt_text", "prompt", "source_prompt"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def _row_game_id(row: dict[str, Any]) -> str:
    for key in ("game_id", "selected_game", "game", "source_game", "control_profile_id"):
        value = row.get(key)
        if value:
            return str(value)
    return "unknown"


def _manifest_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        video_path = str(row.get("video_path") or "")
        segment_id = str(row.get("segment_id") or "")
        if video_path:
            index[(video_path, segment_id)] = row
            index.setdefault((video_path, ""), row)
    return index


def _convert_text_binding_case(
    case: dict[str, Any],
    manifest_rows_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    num_frames: int,
    fps: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    if "first_frame_source" not in case or "action_trace_override" not in case:
        return case

    source = case.get("first_frame_source") or {}
    source_video_path = str(source.get("video_path") or case.get("first_frame") or "")
    source_segment_id = str(source.get("segment_id") or "")
    manifest_row = manifest_rows_by_key.get((source_video_path, source_segment_id)) or manifest_rows_by_key.get(
        (source_video_path, "")
    ) or {}

    video_path = str(manifest_row.get("video_path") or source_video_path)
    audio_path = str(manifest_row.get("audio_path") or source.get("audio_path") or "")
    action_path = str(
        source.get("actions_raw_path")
        or manifest_row.get("actions_raw_path")
        or manifest_row.get("action_path")
        or source.get("action_path")
        or ""
    )
    prompt_override = (
        case.get("scene_prompt_override")
        or case.get("prompt_override")
        or case.get("refined_scene_prompt")
    )
    scene_prompt = str(prompt_override or "").strip() or _source_prompt_from_manifest_row(manifest_row) or str(
        case.get("scene_prompt") or ""
    )
    video_offset_sec = float(source.get("video_offset_sec") or manifest_row.get("video_offset_sec") or 0.0)
    action_row_start = int(
        source.get("segment_action_row_start")
        or manifest_row.get("segment_action_row_start")
        or manifest_row.get("action_row_start")
        or 0
    )
    action_row_scale = _action_row_scale_from_row(manifest_row)
    source_fps = _parse_fps(manifest_row.get("standard_action_hz") or manifest_row.get("source_fps"), float(fps))
    video_source_fps = _parse_fps(
        manifest_row.get("video_source_fps") or manifest_row.get("video_fps"),
        source_fps,
    )
    override = case["action_trace_override"]
    overrides = case.get("action_trace_overrides")
    if isinstance(overrides, Sequence) and not isinstance(overrides, (str, bytes)):
        trace_overrides = [item for item in overrides if isinstance(item, Mapping)]
    else:
        trace_overrides = [override]

    def _synthetic_from_override(item: Mapping[str, Any]) -> dict[str, Any]:
        component = item.get("component")
        if component is None:
            component_name = str(item.get("button") or "").strip().upper()
        else:
            component_name = str(component or "").strip()
            if component_name.upper() in CANONICAL_BUTTONS:
                component_name = component_name.upper()
        start = int(round(float(item.get("t_down", 0.0)) * float(fps)))
        end = int(round(float(item.get("t_up", item.get("t_down", 0.0))) * float(fps)))
        return {
            "component": component_name,
            "start_frame": max(0, start),
            "end_frame": max(start + 1, min(num_frames, end)),
            "value": float(item.get("value", 1.0)),
        }

    synthetic_actions = [_synthetic_from_override(item) for item in trace_overrides]
    synthetic_action: dict[str, Any]
    if len(synthetic_actions) == 1:
        synthetic_action = synthetic_actions[0]
    else:
        synthetic_action = {"actions": synthetic_actions}

    converted = dict(case)
    converted.update(
        {
            "case_id": str(case["case_id"]),
            "group": str(case.get("eval_type") or case.get("group") or "text_binding"),
            "variant": "text_binding_synthetic_button",
            "sample_id": str(case.get("case_id")),
            "game": str(case.get("game_id") or _row_game_id(manifest_row)),
            "game_id": str(case.get("game_id") or _row_game_id(manifest_row)),
            "selected_game": manifest_row.get("selected_game"),
            "control_profile_id": manifest_row.get("control_profile_id"),
            "video_path": video_path,
            "audio_path": audio_path,
            "action_path": action_path,
            "crop_start": 0,
            "action_row_start": action_row_start,
            "action_row_scale": action_row_scale,
            "raw_action_chunk_size": manifest_row.get("raw_action_chunk_size"),
            "standard_chunk_action_rows": manifest_row.get("standard_chunk_action_rows"),
            "video_offset_sec": video_offset_sec,
            "audio_offset_sec": video_offset_sec,
            "source_fps": source_fps,
            "video_source_fps": video_source_fps,
            "fps": int(fps),
            "num_frames": int(num_frames),
            "height": int(height),
            "width": int(width),
            "prompt": scene_prompt,
            "source_prompt": scene_prompt,
            "control_binding_text": case.get("control_binding_text") or case.get("prompt") or "",
            "button_meaning_texts": {
                str(key).strip().upper(): str(value)
                for key, value in (case.get("button_meanings") or {}).items()
                if str(key).strip()
            },
            "button_schema": "canonical",
            "synthetic_action": synthetic_action,
            "synthetic_actions": synthetic_actions,
        }
    )
    return converted


def prepare_text_binding_cases(
    cases: list[dict[str, Any]],
    *,
    manifest_path: Path,
    num_frames: int,
    fps: int,
    height: int,
    width: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not any("first_frame_source" in case and "action_trace_override" in case for case in cases):
        return cases, None

    manifest_rows = read_manifest_jsonl(manifest_path)
    index = _manifest_index(manifest_rows)
    converted = [
        _convert_text_binding_case(
            case,
            index,
            num_frames=num_frames,
            fps=fps,
            height=height,
            width=width,
        )
        for case in cases
    ]
    summary = {
        "mode": "text_binding_case_conversion",
        "source_manifest": str(manifest_path),
        "input_cases": len(cases),
        "converted_cases": sum(
            1 for case in cases if "first_frame_source" in case and "action_trace_override" in case
        ),
        "prompt_policy": "scene_prompt_from_manifest",
        "meaning_policy": "button_meaning_texts_to_action_button_meaning_embeds",
        "action_policy": "synthetic_button_trace_override",
    }
    return converted, summary


def _clamp_crop_start(peak_frame: int, source_frames: int, num_frames: int) -> int:
    max_start = max(0, int(source_frames) - num_frames)
    return max(0, min(max_start, int(peak_frame) - num_frames // 2))


def _source_prompt(row: dict[str, Any]) -> str:
    prompt = row.get("prompt_text") or row.get("prompt") or row.get("source_prompt") or ""
    return str(prompt).strip()


def _is_primary_fps_eval_row(row: dict[str, Any]) -> bool:
    prompt = _source_prompt(row).lower()
    if not prompt:
        return False
    if any(term in prompt for term in LONG_TAIL_EVAL_PROMPT_TERMS):
        return False
    return any(term in prompt for term in PRIMARY_FPS_EVAL_PROMPT_TERMS)


def _stats_worker(row: dict[str, Any]) -> dict[str, Any] | None:
    action_path = row.get("action_path")
    if not action_path:
        return None
    try:
        table = pq.read_table(
            action_path,
            columns=["j_left", "j_right", *BUTTON_COLUMNS],
        )
        records = table.to_pylist()
    except Exception as exc:
        return {
            "sample_id": row.get("sample_id"),
            "game": _row_game_id(row),
            "game_id": _row_game_id(row),
            "selected_game": row.get("selected_game"),
            "control_profile_id": row.get("control_profile_id"),
            "error": repr(exc),
        }

    if not records:
        return None

    left = np.zeros((len(records), 2), dtype=np.float32)
    right = np.zeros((len(records), 2), dtype=np.float32)
    buttons = np.zeros((len(records), len(BUTTON_COLUMNS)), dtype=np.float32)

    for i, record in enumerate(records):
        left[i] = _pair_to_xy(record.get("j_left"))
        right[i] = _pair_to_xy(record.get("j_right"))
        for j, column in enumerate(BUTTON_COLUMNS):
            buttons[i, j] = float(record.get(column, 0.0) or 0.0)

    left_mag = np.linalg.norm(left, axis=1)
    right_mag = np.linalg.norm(right, axis=1)
    button_active = (buttons > 0.05).any(axis=1)
    trigger_frames = ((buttons[:, 0] > 0.10) | (buttons[:, 1] > 0.10)).sum()
    combined = left_mag + right_mag + button_active.astype(np.float32) * 0.5
    peak_frame = int(combined.argmax()) if combined.size else 0
    source_frames = int(row.get("source_frames") or len(records))

    return {
        "sample_id": row.get("sample_id"),
        "game": _row_game_id(row),
        "game_id": _row_game_id(row),
        "selected_game": row.get("selected_game"),
        "control_profile_id": row.get("control_profile_id"),
        "video_path": row.get("video_path"),
        "audio_path": row.get("audio_path"),
        "action_path": row.get("action_path"),
        "meta_path": row.get("meta_path"),
        "source_prompt": _source_prompt(row),
        "source_frames": source_frames,
        "left_mean": float(left_mag.mean()),
        "left_max": float(left_mag.max(initial=0.0)),
        "right_mean": float(right_mag.mean()),
        "right_max": float(right_mag.max(initial=0.0)),
        "trigger_frames": int(trigger_frames),
        "button_frames": int(button_active.sum()),
        "button_max": float(buttons.max(initial=0.0)),
        "combined_score": float(left_mag.mean() + right_mag.mean() + button_active.mean()),
        "idle_score": float(left_mag.mean() + right_mag.mean() + button_active.mean()),
        "peak_frame": peak_frame,
    }


def _best_unused(
    stats: list[dict[str, Any]],
    used: set[str],
    score_key: str,
    reverse: bool = True,
    game: str | None = None,
    min_value: float | None = None,
) -> dict[str, Any] | None:
    pool = [s for s in stats if s["sample_id"] not in used and (game is None or s["game"] == game)]
    if min_value is not None:
        pool = [s for s in pool if float(s.get(score_key, 0.0)) >= min_value]
    if not pool:
        return None
    return sorted(pool, key=lambda s: float(s.get(score_key, 0.0)), reverse=reverse)[0]


def _short_game(game: str) -> str:
    mapping = {
        "halo_infinite": "halo",
        "call_of_duty__modern_warfare_iii": "mwiii",
        "call_of_duty__warzone": "warzone",
        "call_of_duty": "cod",
    }
    return mapping.get(game, re.sub(r"[^a-z0-9]+", "_", game.lower()).strip("_")[:16] or "game")


def _case_prompt(game: str, category: str, variant: str) -> str:
    game_name = game.replace("__", " ").replace("_", " ")
    if variant == "idle":
        return f"{game_name} first-person gameplay, no player input, camera mostly stable."
    if variant == "remove_joysticks":
        return f"{game_name} first-person gameplay, button presses preserved, joystick movement removed."
    if variant == "mirror_right_joystick":
        return f"{game_name} first-person gameplay, mirrored look control compared with the reference action."
    if variant == "invert_joysticks":
        return f"{game_name} first-person gameplay with joystick directions inverted compared with the reference action."
    if variant == "remove_buttons":
        return f"{game_name} first-person gameplay, movement and looking preserved, button presses removed."
    return f"{game_name} first-person gameplay with {category} controls."


def _eval_prompt(anchor: dict[str, Any], category: str, variant: str) -> str:
    return str(anchor.get("source_prompt") or "").strip() or _case_prompt(anchor["game"], category, variant)


def _make_case(
    anchor: dict[str, Any],
    group: str,
    category: str,
    variant: str,
    index: int,
    num_frames: int,
    fps: int,
    seed: int,
) -> dict[str, Any]:
    crop_start = _clamp_crop_start(int(anchor.get("peak_frame", 0)), int(anchor["source_frames"]), num_frames)
    case_id = f"{group}_{index:03d}_{_short_game(anchor['game'])}_{category}_{variant}"
    return {
        "case_id": case_id,
        "group": group,
        "anchor_id": anchor["sample_id"],
        "sample_id": anchor["sample_id"],
        "game": anchor["game"],
        "game_id": anchor.get("game_id") or anchor["game"],
        "selected_game": anchor.get("selected_game"),
        "control_profile_id": anchor.get("control_profile_id"),
        "category": category,
        "variant": variant,
        "video_path": anchor["video_path"],
        "audio_path": anchor["audio_path"],
        "action_path": anchor["action_path"],
        "meta_path": anchor.get("meta_path"),
        "source_frames": int(anchor["source_frames"]),
        "crop_start": int(crop_start),
        "num_frames": int(num_frames),
        "fps": int(fps),
        "width": DEFAULT_WIDTH,
        "height": DEFAULT_HEIGHT,
        "seed": int(seed + index),
        "prompt": _eval_prompt(anchor, category, variant),
        "source_prompt": str(anchor.get("source_prompt") or ""),
        "synthetic_prompt": _case_prompt(anchor["game"], category, variant),
        "action_stats": {
            key: anchor[key]
            for key in (
                "left_mean",
                "left_max",
                "right_mean",
                "right_max",
                "trigger_frames",
                "button_frames",
                "button_max",
                "combined_score",
                "idle_score",
                "peak_frame",
            )
            if key in anchor
        },
    }


def _row_source_frames(row: dict[str, Any], num_frames: int) -> int:
    action_stats = row.get("action_stats")
    if isinstance(action_stats, dict):
        frames = action_stats.get("num_action_frames")
        if frames:
            return int(frames)
    return int(row.get("source_frames") or num_frames)


def _row_prompt_flags(row: dict[str, Any]) -> list[str]:
    flags = row.get("prompt_flags")
    if isinstance(flags, str):
        return [flag for flag in flags.split(",") if flag]
    if isinstance(flags, (list, tuple)):
        return [str(flag) for flag in flags if flag]
    prompt = _source_prompt(row).lower()
    return [term for term in LONG_TAIL_EVAL_PROMPT_TERMS if term in prompt]


def _joystick_activity_from_row(row: dict[str, Any]) -> float:
    action_stats = row.get("action_stats")
    if isinstance(action_stats, dict):
        values = [
            action_stats.get("left_abs_mean"),
            action_stats.get("right_abs_mean"),
            action_stats.get("left_mean"),
            action_stats.get("right_mean"),
            action_stats.get("left_abs_max"),
            action_stats.get("right_abs_max"),
            action_stats.get("left_max"),
            action_stats.get("right_max"),
        ]
        score = float(sum(float(value or 0.0) for value in values))
        if score > 0.0:
            return score

    stat = _stats_worker(row)
    if not stat or "error" in stat:
        return 0.0
    return float(stat.get("left_mean", 0.0)) + float(stat.get("right_mean", 0.0))


def _make_random_case(
    row: dict[str, Any],
    group: str,
    category: str,
    variant: str,
    index: int,
    crop_start: int,
    num_frames: int,
    fps: int,
    seed: int,
) -> dict[str, Any]:
    game = _row_game_id(row)
    sample_id = str(row.get("sample_id") or row.get("clip_id") or f"sample_{index:03d}")
    source_frames = _row_source_frames(row, num_frames)
    case_id = f"{group}_{index:03d}_{_short_game(game)}_{category}_{variant}"
    prompt = _source_prompt(row) or _case_prompt(game, category, variant)
    return {
        "case_id": case_id,
        "group": group,
        "anchor_id": sample_id,
        "sample_id": sample_id,
        "game": game,
        "game_id": game,
        "selected_game": row.get("selected_game"),
        "control_profile_id": row.get("control_profile_id"),
        "category": category,
        "variant": variant,
        "video_path": row.get("video_path"),
        "audio_path": row.get("audio_path"),
        "action_path": row.get("action_path"),
        "meta_path": row.get("meta_path"),
        "source_frames": int(source_frames),
        "crop_start": int(crop_start),
        "source_fps": float(row.get("source_fps") or fps),
        "video_source_fps": float(row.get("video_source_fps") or row.get("source_fps") or fps),
        "video_offset_sec": float(row.get("video_offset_sec") or 0.0),
        "audio_offset_sec": float(row.get("audio_offset_sec") or row.get("video_offset_sec") or 0.0),
        "action_row_start": int(row.get("action_row_start") or 0),
        "action_row_scale": _action_row_scale_from_row(row),
        "num_frames": int(num_frames),
        "fps": int(fps),
        "width": DEFAULT_WIDTH,
        "height": DEFAULT_HEIGHT,
        "seed": int(seed + index),
        "prompt": prompt,
        "source_prompt": prompt,
        "synthetic_prompt": _case_prompt(game, category, variant),
        "review_index": row.get("review_index"),
        "prompt_flags": _row_prompt_flags(row),
        "action_stats": row.get("action_stats") or {},
    }


def _build_random_eval_cases(
    *,
    manifest_path: Path,
    cases_path: Path,
    num_frames: int,
    fps: int,
    seed: int,
    allow_long_tail: bool,
    random_cases_source: Path | None,
    num_random_real_cases: int,
    num_joystick_reverse_cases: int,
    exclude_prompt_flagged: bool,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    source_path = random_cases_source or manifest_path
    raw_rows = [
        row for row in read_manifest_jsonl(source_path)
        if int(row.get("source_frames") or (row.get("action_stats") or {}).get("num_action_frames") or 0) >= num_frames
        and row.get("action_path")
        and row.get("video_path")
    ]

    rows = raw_rows
    prompt_flag_excluded = 0
    if exclude_prompt_flagged:
        rows = [row for row in rows if not _row_prompt_flags(row)]
        prompt_flag_excluded = len(raw_rows) - len(rows)
    domain_excluded = 0
    if not allow_long_tail:
        before_domain_filter = len(rows)
        rows = [row for row in rows if _is_primary_fps_eval_row(row)]
        domain_excluded = before_domain_filter - len(rows)

    if not rows:
        raise RuntimeError(f"No rows available for random eval case sampling from {source_path}")

    real_count = min(num_random_real_cases, len(rows))
    real_indices = rng.choice(len(rows), size=real_count, replace=False)
    real_rows = [rows[int(index)] for index in real_indices]

    cases: list[dict[str, Any]] = []
    crop_by_sample: dict[str, int] = {}
    for index, row in enumerate(real_rows, start=1):
        sample_id = str(row.get("sample_id") or row.get("clip_id") or f"sample_{index:03d}")
        max_start = max(0, _row_source_frames(row, num_frames) - num_frames)
        crop_start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        crop_by_sample[sample_id] = crop_start
        cases.append(
            _make_random_case(
                row=row,
                group="real_replay",
                category="random",
                variant="real",
                index=index,
                crop_start=crop_start,
                num_frames=num_frames,
                fps=fps,
                seed=seed,
            )
        )

    joystick_pool = [
        row for row in real_rows
        if _joystick_activity_from_row(row) > 0.05
    ]
    if len(joystick_pool) < min(num_joystick_reverse_cases, len(real_rows)):
        seen = {str(row.get("sample_id") or row.get("clip_id")) for row in real_rows}
        extra_rows = [
            row for row in rows
            if str(row.get("sample_id") or row.get("clip_id")) not in seen
            and _joystick_activity_from_row(row) > 0.05
        ]
        if extra_rows:
            needed = min(num_joystick_reverse_cases, len(real_rows) + len(extra_rows)) - len(joystick_pool)
            extra_take = min(max(0, needed), len(extra_rows))
            if extra_take:
                extra_indices = rng.choice(len(extra_rows), size=extra_take, replace=False)
                joystick_pool.extend(extra_rows[int(index)] for index in extra_indices)

    reverse_count = min(num_joystick_reverse_cases, len(joystick_pool))
    if reverse_count:
        reverse_indices = rng.choice(len(joystick_pool), size=reverse_count, replace=False)
        reverse_rows = [joystick_pool[int(index)] for index in reverse_indices]
    else:
        reverse_rows = []

    for row in reverse_rows:
        index = len(cases) + 1
        sample_id = str(row.get("sample_id") or row.get("clip_id") or f"sample_{index:03d}")
        if sample_id in crop_by_sample:
            crop_start = crop_by_sample[sample_id]
        else:
            max_start = max(0, _row_source_frames(row, num_frames) - num_frames)
            crop_start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        cases.append(
            _make_random_case(
                row=row,
                group="joystick_reverse",
                category="joystick",
                variant="invert_joysticks",
                index=index,
                crop_start=crop_start,
                num_frames=num_frames,
                fps=fps,
                seed=seed + 1000,
            )
        )

    write_jsonl(cases_path, cases)
    summary_path = cases_path.with_suffix(".summary.json")
    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "sampling_mode": "random",
        "manifest_path": str(manifest_path),
        "random_cases_source": str(source_path),
        "cases_path": str(cases_path),
        "num_cases": len(cases),
        "num_random_real_cases": sum(case["group"] == "real_replay" for case in cases),
        "num_joystick_reverse_cases": sum(case["group"] == "joystick_reverse" for case in cases),
        "num_source_rows_raw": len(raw_rows),
        "num_source_rows_scanned": len(rows),
        "num_prompt_flag_excluded": prompt_flag_excluded,
        "num_domain_excluded": domain_excluded,
        "exclude_prompt_flagged": exclude_prompt_flagged,
        "domain_filter": (
            "all" if allow_long_tail
            else ("source_review_primary_fps" if random_cases_source else "primary_fps")
        ),
        "groups": {group: sum(c["group"] == group for c in cases) for group in sorted({c["group"] for c in cases})},
        "games": {game: sum(c["game"] == game for c in cases) for game in sorted({c["game"] for c in cases})},
        "prompt_flagged_cases": sum(bool(case.get("prompt_flags")) for case in cases),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(f"[eval] Wrote {len(cases)} random eval cases to {cases_path}")
    print(f"[eval] Case summary: {summary_path}")
    return cases


def build_cases(
    manifest_path: Path,
    cases_path: Path,
    num_frames: int,
    fps: int,
    seed: int,
    workers: int,
    allow_long_tail: bool = False,
    sampling_mode: str = "random",
    random_cases_source: Path | None = None,
    num_random_real_cases: int = DEFAULT_RANDOM_REAL_CASES,
    num_joystick_reverse_cases: int = DEFAULT_JOYSTICK_REVERSE_CASES,
    exclude_prompt_flagged_random_cases: bool = False,
) -> list[dict[str, Any]]:
    if sampling_mode == "random":
        return _build_random_eval_cases(
            manifest_path=manifest_path,
            cases_path=cases_path,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            allow_long_tail=allow_long_tail,
            random_cases_source=random_cases_source,
            num_random_real_cases=num_random_real_cases,
            num_joystick_reverse_cases=num_joystick_reverse_cases,
            exclude_prompt_flagged=exclude_prompt_flagged_random_cases,
        )
    if sampling_mode != "action_extreme":
        raise ValueError(f"Unknown eval sampling mode: {sampling_mode}")

    raw_manifest = [
        row for row in read_manifest_jsonl(manifest_path)
        if int(row.get("source_frames") or 0) >= num_frames and row.get("action_path") and row.get("video_path")
    ]
    if allow_long_tail:
        manifest = raw_manifest
        domain_excluded = 0
    else:
        manifest = [row for row in raw_manifest if _is_primary_fps_eval_row(row)]
        domain_excluded = len(raw_manifest) - len(manifest)
    if not manifest:
        raise RuntimeError(
            f"No manifest rows after eval-domain filtering: {manifest_path}. "
            "Use --allow-long-tail-cases to disable the filter."
        )

    print(
        f"[eval] Scanning action stats from {len(manifest)} manifest rows "
        f"(raw={len(raw_manifest)}, long_tail_excluded={domain_excluded})..."
    )
    stats: list[dict[str, Any]] = []
    errors = 0
    max_workers = max(1, workers)
    with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for item in executor.map(_stats_worker, manifest, chunksize=64):
            if not item:
                continue
            if "error" in item:
                errors += 1
                continue
            if int(item.get("source_frames") or 0) >= num_frames:
                stats.append(item)

    if not stats:
        raise RuntimeError("No readable action rows found while building eval cases.")

    preferred_games = [
        "halo_infinite",
        "call_of_duty__modern_warfare_iii",
        "call_of_duty__warzone",
    ]
    games = sorted({s["game"] for s in stats})
    if allow_long_tail:
        game_order = [g for g in preferred_games if g in games] + [g for g in games if g not in preferred_games]
    else:
        game_order = [g for g in preferred_games if g in games]
        if not game_order:
            game_order = games

    target_real_replay_cases = 12
    game_order = game_order[:target_real_replay_cases]
    base_quota = target_real_replay_cases // max(1, len(game_order))
    quota_remainder = target_real_replay_cases % max(1, len(game_order))
    game_quota = {game: base_quota for game in game_order}
    for game in game_order[:quota_remainder]:
        game_quota[game] += 1

    used: set[str] = set()
    anchors: list[tuple[dict[str, Any], str]] = []
    category_specs = [
        ("look", "right_mean", True, 0.08),
        ("move", "left_mean", True, 0.08),
        ("fire", "trigger_frames", True, 1.0),
        ("idle", "idle_score", False, None),
        ("mixed", "combined_score", True, 0.12),
    ]
    for game in game_order:
        quota = game_quota.get(game, 2)
        for i in range(quota):
            category, score_key, reverse, min_value = category_specs[i % len(category_specs)]
            pick = _best_unused(stats, used, score_key, reverse=reverse, game=game, min_value=min_value)
            if pick is None:
                pick = _best_unused(stats, used, "combined_score", reverse=True, game=game)
            if pick is not None:
                used.add(pick["sample_id"])
                anchors.append((pick, category))

    cases: list[dict[str, Any]] = []
    for index, (anchor, category) in enumerate(anchors, start=1):
        cases.append(_make_case(anchor, "real_replay", category, "real", index, num_frames, fps, seed))

                                                                                  
                                                                         
                                                               
    counter_pool: list[tuple[dict[str, Any], str]] = [
        (anchor, category) for anchor, category in anchors if category != "idle"
    ][:6]
    if len(counter_pool) < 6:
        fallback = sorted(
            [s for s in stats if s["sample_id"] not in used],
            key=lambda s: float(s.get("combined_score", 0.0)),
            reverse=True,
        )
        seen_counter = {anchor["sample_id"] for anchor, _ in counter_pool}
        for anchor in fallback:
            if anchor["sample_id"] in seen_counter:
                continue
            counter_pool.append((anchor, "mixed"))
            seen_counter.add(anchor["sample_id"])
            if len(counter_pool) >= 6:
                break

    for anchor_index, (anchor, category) in enumerate(counter_pool, start=1):
        for variant in ("real", "idle", "mirror_right_joystick", "remove_buttons"):
            index = len(cases) + 1
            cases.append(_make_case(anchor, "counterfactual", category, variant, index, num_frames, fps, seed + anchor_index * 17))

    audio_pool = sorted(
        [s for s in stats if s["sample_id"] not in used],
        key=lambda s: (int(s.get("trigger_frames", 0)), float(s.get("button_max", 0.0))),
        reverse=True,
    )[:4]
    for anchor in audio_pool:
        index = len(cases) + 1
        cases.append(_make_case(anchor, "audio_event", "fire", "real", index, num_frames, fps, seed + 1000))

    write_jsonl(cases_path, cases)
    summary_path = cases_path.with_suffix(".summary.json")
    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "sampling_mode": "action_extreme",
        "manifest_path": str(manifest_path),
        "cases_path": str(cases_path),
        "num_cases": len(cases),
        "domain_filter": "all" if allow_long_tail else "primary_fps",
        "num_manifest_rows_raw": len(raw_manifest),
        "num_manifest_rows_scanned": len(manifest),
        "num_long_tail_excluded": domain_excluded,
        "long_tail_prompt_terms": [] if allow_long_tail else list(LONG_TAIL_EVAL_PROMPT_TERMS),
        "num_stats": len(stats),
        "num_action_errors": errors,
        "primary_fps_prompt_terms": [] if allow_long_tail else list(PRIMARY_FPS_EVAL_PROMPT_TERMS),
        "groups": {group: sum(c["group"] == group for c in cases) for group in sorted({c["group"] for c in cases})},
        "games": {game: sum(c["game"] == game for c in cases) for game in sorted({c["game"] for c in cases})},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(f"[eval] Wrote {len(cases)} cases to {cases_path}")
    print(f"[eval] Case summary: {summary_path}")
    return cases


def _unit_anchor(source_cases: list[dict[str, Any]]) -> dict[str, Any]:
    for case in source_cases:
        if case.get("group") == "real_replay" and case.get("category") == "idle":
            return case
    for case in source_cases:
        if case.get("group") == "real_replay":
            return case
    if not source_cases:
        raise RuntimeError("Cannot build action-unit cases from an empty source case file.")
    return source_cases[0]


def _unit_specs() -> list[dict[str, Any]]:
    return [
        {"name": "idle", "component": None, "value": 0.0, "label": "idle baseline"},
        {"name": "left_forward", "component": "left_y", "value": -1.0, "label": "left stick forward"},
        {"name": "left_back", "component": "left_y", "value": 1.0, "label": "left stick back"},
        {"name": "left_left", "component": "left_x", "value": -1.0, "label": "left stick left"},
        {"name": "left_right", "component": "left_x", "value": 1.0, "label": "left stick right"},
        {"name": "right_look_left", "component": "right_x", "value": -1.0, "label": "right stick look left"},
        {"name": "right_look_right", "component": "right_x", "value": 1.0, "label": "right stick look right"},
        {"name": "right_look_up", "component": "right_y", "value": -1.0, "label": "right stick look up"},
        {"name": "right_look_down", "component": "right_y", "value": 1.0, "label": "right stick look down"},
        {"name": "button_rt", "component": "right_trigger", "value": 1.0, "label": "right trigger"},
        {"name": "button_lt", "component": "left_trigger", "value": 1.0, "label": "left trigger"},
        {"name": "button_r3", "component": "right_thumb", "value": 1.0, "label": "right thumb"},
        {"name": "button_south", "component": "south", "value": 1.0, "label": "south button"},
        {"name": "button_west", "component": "west", "value": 1.0, "label": "west button"},
        {"name": "button_north", "component": "north", "value": 1.0, "label": "north button"},
    ]


def build_action_unit_cases(
    source_cases_path: Path,
    cases_path: Path,
    num_frames: int,
    fps: int,
    seed: int,
) -> list[dict[str, Any]]:
    source_cases = read_jsonl(source_cases_path)
    anchor = _unit_anchor(source_cases)
    specs = _unit_specs()
    cases: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        case = dict(anchor)
        case["case_id"] = f"action_unit_{index:03d}_{_short_game(anchor['game'])}_{spec['name']}"
        case["group"] = "action_unit"
        case["anchor_id"] = anchor["sample_id"]
        case["category"] = "unit"
        case["variant"] = "synthetic"
        case["num_frames"] = int(num_frames)
        case["fps"] = int(fps)
        case["seed"] = int(seed + 2000 + index)
        case["synthetic_prompt"] = (
            f"{anchor['game'].replace('__', ' ').replace('_', ' ')} first-person gameplay, {spec['label']} only."
        )
        case["prompt"] = str(anchor.get("source_prompt") or "").strip() or case["synthetic_prompt"]
        case["synthetic_action"] = {
            "name": spec["name"],
            "component": spec["component"],
            "value": spec["value"],
            "start_frame": 1,
            "end_frame": int(num_frames),
            "description": spec["label"],
        }
        cases.append(case)

    write_jsonl(cases_path, cases)
    summary_path = cases_path.with_suffix(".summary.json")
    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_cases_path": str(source_cases_path),
        "cases_path": str(cases_path),
        "anchor_case_id": anchor["case_id"],
        "anchor_sample_id": anchor["sample_id"],
        "num_cases": len(cases),
        "actions": [case["synthetic_action"]["name"] for case in cases],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(f"[eval] Wrote {len(cases)} action-unit cases to {cases_path}")
    print(f"[eval] Action-unit summary: {summary_path}")
    return cases


def _resize_center_crop(video: torch.Tensor, height: int, width: int) -> torch.Tensor:
                                            
    _, _, h, w = video.shape
    scale = max(height / h, width / w)
    new_h = max(height, int(round(h * scale)))
    new_w = max(width, int(round(w * scale)))
    video = F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)
    top = max(0, (new_h - height) // 2)
    left = max(0, (new_w - width) // 2)
    return video[:, :, top:top + height, left:left + width].contiguous()


def load_video_window(
    video_path: str,
    crop_start: int,
    num_frames: int,
    height: int,
    width: int,
    *,
    source_fps: float | None = None,
    video_source_fps: float | None = None,
    video_offset_sec: float = 0.0,
) -> torch.Tensor:
    source_fps = float(source_fps or video_source_fps or DEFAULT_FPS)
    video_source_fps = float(video_source_fps or source_fps)
    action_indices = torch.arange(crop_start, crop_start + num_frames, dtype=torch.float64)
    frame_indices = ((action_indices / source_fps + float(video_offset_sec)) * video_source_fps).round().to(torch.int64)
    target_map: dict[int, list[int]] = {}
    for out_index, source_index in enumerate(frame_indices.tolist()):
        target_map.setdefault(int(source_index), []).append(out_index)

    frames: list[torch.Tensor | None] = [None] * num_frames
    first_needed = min(target_map)
    last_needed = max(target_map)
    container = av.open(video_path)
    try:
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
        for frame in container.decode(stream):
            if frame.time is not None and video_source_fps > 0:
                source_index = int(round(float(frame.time) * video_source_fps))
            else:
                source_index = fallback_index
                fallback_index += 1
            if source_index > last_needed:
                break
            if source_index not in target_map:
                continue
            array = frame.to_ndarray(format="rgb24")
            tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
            for out_index in target_map[source_index]:
                frames[out_index] = tensor
    finally:
        container.close()

    first_valid = next((frame for frame in frames if frame is not None), None)
    if first_valid is None:
        raise RuntimeError(f"No frames decoded from {video_path} at crop_start={crop_start}")
    for idx in range(len(frames)):
        if frames[idx] is None:
            frames[idx] = frames[idx - 1] if idx > 0 and frames[idx - 1] is not None else first_valid
    video = torch.stack([frame for frame in frames if frame is not None])
    return _resize_center_crop(video, height, width)


def load_audio_window(
    audio_path: str,
    crop_start: int,
    num_frames: int,
    fps: int,
    *,
    audio_offset_sec: float = 0.0,
) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    start_sample = int(round((float(audio_offset_sec) + crop_start / fps) * sample_rate))
    target_samples = int(round((num_frames / fps) * sample_rate))
    end_sample = start_sample + target_samples
    if start_sample >= waveform.shape[1]:
        clipped = waveform.new_zeros((2, target_samples))
    else:
        clipped = waveform[:, start_sample:min(end_sample, waveform.shape[1])]
        if clipped.shape[1] < target_samples:
            clipped = F.pad(clipped, (0, target_samples - clipped.shape[1]))
    return clipped.contiguous(), int(sample_rate)


def load_action_window(
    action_path: str,
    crop_start: int,
    num_frames: int,
    variant: str,
    synthetic_action: dict[str, Any] | None = None,
    *,
    action_row_start: int = 0,
    action_row_scale: float = 1.0,
    button_resampling: str = "point",
    button_schema: str = "legacy",
    game_id: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    use_canonical_buttons = button_schema == "canonical"
    use_crossfps_unified_buttons = button_schema == "crossfps_unified"
    if use_crossfps_unified_buttons:
        raw_button_columns = crossfps_unified_raw_button_columns(game_id)
        button_labels = CROSS_FPS_UNIFIED_BUTTONS
    elif use_canonical_buttons:
        raw_button_columns = CANONICAL_BUTTONS
        button_labels = CANONICAL_BUTTONS
    else:
        raw_button_columns = BUTTON_COLUMNS
        button_labels = BUTTON_COLUMNS

    if synthetic_action is not None:
        joystick = torch.zeros((num_frames, 4), dtype=torch.float32)
        buttons = torch.zeros((num_frames, len(button_labels)), dtype=torch.float32)
        if isinstance(synthetic_action, Mapping) and isinstance(synthetic_action.get("actions"), Sequence):
            synthetic_actions = [item for item in synthetic_action["actions"] if isinstance(item, Mapping)]
        elif isinstance(synthetic_action, Sequence) and not isinstance(synthetic_action, (str, bytes)):
            synthetic_actions = [item for item in synthetic_action if isinstance(item, Mapping)]
        else:
            synthetic_actions = [synthetic_action]
        for action in synthetic_actions:
            component = action.get("component")
            if not component:
                continue
            component = str(component)
            if use_crossfps_unified_buttons and component in raw_button_columns:
                index = raw_button_columns.index(component)
                kind = "crossfps_unified_button"
            elif use_crossfps_unified_buttons and (normalized := normalize_button_name(component)) is not None:
                aliases = BUTTON_COLUMN_ALIASES.get(normalized, ())
                matching_alias = next((alias for alias in aliases if alias in raw_button_columns), None)
                if matching_alias is None:
                    raise ValueError(
                        f"Synthetic action component {component!r} is not part of crossfps_unified"
                    )
                index = raw_button_columns.index(matching_alias)
                kind = "crossfps_unified_button"
            else:
                if component not in ACTION_COMPONENTS:
                    raise ValueError(f"Unknown synthetic action component: {component}")
                index, kind = ACTION_COMPONENTS[component]
            start = max(0, int(action.get("start_frame", 0)))
            end = min(num_frames, int(action.get("end_frame", num_frames)))
            value = float(action.get("value", 0.0))
            if kind == "joystick":
                joystick[start:end, index] = value
            elif kind in {"button", "crossfps_unified_button"}:
                buttons[start:end, index] = value
            elif kind == "canonical_button":
                buttons[start:end, index] = value
        return joystick.clamp(-1.0, 1.0), buttons.clamp(0.0, 1.0)

    if button_resampling not in {"point", "causal_run_max"}:
        raise ValueError(f"Unknown button resampling mode: {button_resampling}")

    table = pq.read_table(action_path)
    records = table.to_pylist()
    if not records:
        raise RuntimeError(f"No action rows in {action_path}")
    indices = [
        int(round((int(action_row_start) + int(crop_start) + frame_index) * float(action_row_scale)))
        for frame_index in range(num_frames)
    ]

    def button_values(record: Mapping[str, Any]) -> list[float]:
        if use_canonical_buttons:
            values: list[float] = []
            for column in raw_button_columns:
                value = 0.0
                for alias in BUTTON_COLUMN_ALIASES.get(column, ()):
                    if alias in record:
                        value = float(record.get(alias, 0.0) or 0.0)
                        break
                values.append(value)
            return values
        return [float(record.get(column, 0.0) or 0.0) for column in raw_button_columns]

    joystick = torch.zeros((num_frames, 4), dtype=torch.float32)
    buttons = torch.zeros((num_frames, len(button_labels)), dtype=torch.float32)
    for i, row_index in enumerate(indices):
        record = records[min(max(row_index, 0), len(records) - 1)]
        joystick[i, 0:2] = torch.tensor(_pair_to_xy(record.get("j_left")), dtype=torch.float32)
        joystick[i, 2:4] = torch.tensor(_pair_to_xy(record.get("j_right")), dtype=torch.float32)
        buttons[i] = torch.tensor(button_values(record), dtype=torch.float32)

    if button_resampling == "causal_run_max" and action_row_scale > 1.0:
                                                                                 
                                                                                
                                                                                
        raw_values = [button_values(record) for record in records]
        buttons.zero_()

        def first_standard_frame_at_or_after(raw_row_index: int) -> int:
            frame_index = int(math.floor(float(raw_row_index) / action_row_scale - action_row_start))

            def mapped(index: int) -> int:
                return int(round((action_row_start + index) * action_row_scale))

            while mapped(frame_index) < raw_row_index:
                frame_index += 1
            while mapped(frame_index - 1) >= raw_row_index:
                frame_index -= 1
            return frame_index

        target_frames = [crop_start + index for index in range(num_frames)]
        for channel in range(len(button_labels)):
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
                target_start = first_standard_frame_at_or_after(run_start)
                target_end = max(first_standard_frame_at_or_after(run_end), target_start + 1)
                for output_index, frame_index in enumerate(target_frames):
                    if target_start <= frame_index < target_end:
                        buttons[output_index, channel] = max(
                            float(buttons[output_index, channel]),
                            run_peak,
                        )

    joystick = joystick.clamp(-1.0, 1.0)
    buttons = buttons.clamp(0.0, 1.0)

    if variant == "idle":
        joystick.zero_()
        buttons.zero_()
    elif variant == "mirror_right_joystick":
        joystick[:, 2] *= -1.0
    elif variant == "invert_joysticks":
        joystick.mul_(-1.0)
    elif variant == "invert_joysticks_flip_triggers":
        joystick.mul_(-1.0)
        buttons[:, 0:2] = 1.0 - buttons[:, 0:2]
    elif variant == "invert_joysticks_swap_triggers":
        joystick.mul_(-1.0)
        buttons[:, 0:2] = buttons[:, [1, 0]]
    elif variant == "remove_joysticks":
        joystick.zero_()
    elif variant == "remove_buttons":
        buttons.zero_()
    elif variant != "real":
        raise ValueError(f"Unknown action variant: {variant}")
    return joystick, buttons


def joystick_to_display_units(
    joystick: torch.Tensor,
    display_scales: tuple[float, float, float, float] | None,
) -> torch.Tensor:
    ''
    if display_scales is None:
        return joystick
    scales = torch.as_tensor(display_scales, dtype=joystick.dtype, device=joystick.device)
    if scales.shape != (4,) or not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError(f"Joystick display scales must be four finite positive values, got {display_scales}")
    return joystick / scales


def save_action_plot(
    path: Path,
    joystick: torch.Tensor,
    buttons: torch.Tensor,
    title: str,
    joystick_display_scales: tuple[float, float, float, float] | None = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[eval] matplotlib unavailable, skipping action plot: {exc}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(joystick.shape[0])
    joy = joystick_to_display_units(joystick, joystick_display_scales).cpu().numpy()
    btn = buttons.cpu().numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t, joy[:, 0], label="left_x")
    axes[0].plot(t, joy[:, 1], label="left_y")
    axes[0].plot(t, joy[:, 2], label="right_x")
    axes[0].plot(t, joy[:, 3], label="right_y")
    joy_limit = max(1.05, float(np.abs(joy).max(initial=0.0)) * 1.05)
    axes[0].set_ylim(-joy_limit, joy_limit)
    axes[0].set_ylabel(
        "train-scale units" if joystick_display_scales is not None else "model input"
    )
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].set_title(title)
    labels = _button_labels_for_width(buttons.shape[-1])
    for i, column in enumerate(labels):
        axes[1].plot(t, btn[:, i], label=column)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(ncol=4, fontsize=8)
    axes[1].set_xlabel("raw frame")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _ensure_cfhw(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 4:
        raise ValueError(f"Expected video tensor [C,F,H,W] or [F,C,H,W], got {tuple(video.shape)}")
    if video.shape[0] in (1, 3):
        return video.detach().cpu().float().clamp(0, 1)
    if video.shape[1] in (1, 3):
        return video.detach().cpu().float().clamp(0, 1).permute(1, 0, 2, 3)
    raise ValueError(f"Cannot infer video layout from shape {tuple(video.shape)}")


def _draw_stick(
    draw: Any,
    center: tuple[int, int],
    radius: int,
    raw_xy: tuple[float, float],
    display_xy: tuple[float, float],
    label: str,
) -> None:
    cx, cy = center
    raw_x = float(raw_xy[0])
    raw_y = float(raw_xy[1])
    x = float(np.clip(display_xy[0], -1.0, 1.0))
    y = float(np.clip(display_xy[1], -1.0, 1.0))
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(230, 235, 240, 255), width=2)
    draw.line((cx - radius, cy, cx + radius, cy), fill=(110, 118, 130, 255), width=1)
    draw.line((cx, cy - radius, cx, cy + radius), fill=(110, 118, 130, 255), width=1)
    dot_x = int(round(cx + x * radius))
    dot_y = int(round(cy + y * radius))
    draw.ellipse((dot_x - 5, dot_y - 5, dot_x + 5, dot_y + 5), fill=(64, 214, 146, 255))
    draw.text(
        (cx - radius, cy + radius + 5),
        f"{label} {raw_x:+.2f},{raw_y:+.2f}",
        fill=(235, 238, 242, 255),
    )


def _button_labels_for_width(width: int) -> tuple[str, ...]:
    if width == len(CANONICAL_BUTTONS):
        return CANONICAL_BUTTONS
    if width == len(CROSS_FPS_UNIFIED_BUTTONS):
        return CROSS_FPS_UNIFIED_BUTTONS
    if width == len(LEGACY_BUTTON_LABELS):
        return LEGACY_BUTTON_LABELS
    return tuple(f"button_{index}" for index in range(width))


def _draw_button_row(draw: Any, origin: tuple[int, int], values: np.ndarray) -> None:
    labels = _button_labels_for_width(values.shape[0])
    x0, y0 = origin
    box_w = 42 if len(labels) > 6 else (62 if labels == CROSS_FPS_UNIFIED_BUTTONS else 52)
    box_h = 30
    gap = 6 if len(labels) > 6 else 8
    for i, label in enumerate(labels):
        value = float(np.clip(values[i], 0.0, 1.0))
        x = x0 + i * (box_w + gap)
        fill = (58, 69, 82, 230)
        outline = (125, 136, 150, 255)
        if value > 0.05:
            fill = (57, 172, 104, 245)
            outline = (166, 242, 195, 255)
        draw.rounded_rectangle((x, y0, x + box_w, y0 + box_h), radius=5, fill=fill, outline=outline, width=1)
        if 0.05 < value < 0.98:
            draw.rectangle((x + 3, y0 + box_h - 6, x + 3 + int((box_w - 6) * value), y0 + box_h - 3), fill=(238, 244, 250, 255))
        draw.text((x + 6, y0 + 8), label, fill=(245, 248, 252, 255))


def _draw_action_hud(
    draw: Any,
    *,
    panel_x: int,
    panel_y: int,
    panel_w: int,
    joystick_values: np.ndarray,
    joystick_display_values: np.ndarray,
    button_values: np.ndarray,
    title: str,
) -> None:
    panel_h = 112
    draw.rounded_rectangle(
        (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
        radius=8,
        fill=(18, 22, 28, 205),
        outline=(78, 88, 102, 230),
        width=1,
    )
    draw.text((panel_x + 14, panel_y + 10), title, fill=(245, 248, 252, 255))
    _draw_stick(
        draw,
        (panel_x + 58, panel_y + 58),
        25,
        (joystick_values[0], joystick_values[1]),
        (joystick_display_values[0], joystick_display_values[1]),
        "L",
    )
    _draw_stick(
        draw,
        (panel_x + 152, panel_y + 58),
        25,
        (joystick_values[2], joystick_values[3]),
        (joystick_display_values[2], joystick_display_values[3]),
        "R",
    )
    _draw_button_row(draw, (panel_x + 220, panel_y + 40), button_values)


def add_action_overlay(
    video: torch.Tensor,
    joystick: torch.Tensor,
    buttons: torch.Tensor,
    joystick_display_scales: tuple[float, float, float, float] | None = None,
    joystick_display_label: str | None = None,
    action_usage_label: str = "fed action",
) -> torch.Tensor:
    ''
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"[eval] PIL unavailable, skipping action overlay: {exc}")
        cfhw = _ensure_cfhw(video)
        return cfhw.permute(1, 0, 2, 3)

    cfhw = _ensure_cfhw(video)
    _, frames, height, width = cfhw.shape
    joystick_np = joystick.detach().cpu().float().numpy()
    joystick_display_np = (
        joystick_to_display_units(joystick, joystick_display_scales)
        .detach()
        .cpu()
        .float()
        .numpy()
    )
    buttons_np = buttons.detach().cpu().float().numpy()

    panel_w = min(width - 24, 640)
    panel_h = 112
    panel_x = 12
    panel_y = max(8, height - panel_h - 12)
    rendered: list[np.ndarray] = []

    for frame_index in range(frames):
        frame = (cfhw[:, frame_index].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        image = Image.fromarray(frame).convert("RGBA")
        hud = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(hud)
        idx = min(frame_index, joystick_np.shape[0] - 1)
        _draw_action_hud(
            draw,
            panel_x=panel_x,
            panel_y=panel_y,
            panel_w=panel_w,
            joystick_values=joystick_np[idx],
            joystick_display_values=joystick_display_np[idx],
            button_values=buttons_np[min(frame_index, buttons_np.shape[0] - 1)],
            title=(
                f"{action_usage_label} frame {frame_index:03d} | "
                f"dot={joystick_display_label or 'train scale'}, value=model raw"
                if joystick_display_scales is not None
                else f"{action_usage_label} frame {frame_index:03d}"
            ),
        )

        image = Image.alpha_composite(image, hud).convert("RGB")
        rendered.append(np.asarray(image))

    return torch.from_numpy(np.stack(rendered, axis=0)).permute(0, 3, 1, 2).float().div_(255.0)


def add_dual_action_overlay(
    video: torch.Tensor,
    left_joystick: torch.Tensor,
    left_buttons: torch.Tensor,
    right_joystick: torch.Tensor,
    right_buttons: torch.Tensor,
    joystick_display_scales: tuple[float, float, float, float] | None = None,
) -> torch.Tensor:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"[eval] PIL unavailable, skipping dual action overlay: {exc}")
        cfhw = _ensure_cfhw(video)
        return cfhw.permute(1, 0, 2, 3)

    cfhw = _ensure_cfhw(video)
    _, frames, height, width = cfhw.shape
    half_width = width // 2
    panel_w = min(half_width - 24, 640)
    panel_h = 112
    panel_y = max(8, height - panel_h - 12)
    left_np = left_joystick.detach().cpu().float().numpy()
    left_display_np = (
        joystick_to_display_units(left_joystick, joystick_display_scales)
        .detach()
        .cpu()
        .float()
        .numpy()
    )
    left_btn_np = left_buttons.detach().cpu().float().numpy()
    right_np = right_joystick.detach().cpu().float().numpy()
    right_display_np = (
        joystick_to_display_units(right_joystick, joystick_display_scales)
        .detach()
        .cpu()
        .float()
        .numpy()
    )
    right_btn_np = right_buttons.detach().cpu().float().numpy()
    rendered: list[np.ndarray] = []

    for frame_index in range(frames):
        frame = (cfhw[:, frame_index].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        image = Image.fromarray(frame).convert("RGBA")
        hud = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(hud)
        left_idx = min(frame_index, left_np.shape[0] - 1)
        right_idx = min(frame_index, right_np.shape[0] - 1)
        _draw_action_hud(
            draw,
            panel_x=12,
            panel_y=panel_y,
            panel_w=panel_w,
            joystick_values=left_np[left_idx],
            joystick_display_values=left_display_np[left_idx],
            button_values=left_btn_np[min(frame_index, left_btn_np.shape[0] - 1)],
            title=(
                f"GT real {frame_index:03d} | dot=train scale, value=raw"
                if joystick_display_scales is not None
                else f"GT real action {frame_index:03d}"
            ),
        )
        _draw_action_hud(
            draw,
            panel_x=half_width + 12,
            panel_y=panel_y,
            panel_w=panel_w,
            joystick_values=right_np[right_idx],
            joystick_display_values=right_display_np[right_idx],
            button_values=right_btn_np[min(frame_index, right_btn_np.shape[0] - 1)],
            title=(
                f"Pred fed {frame_index:03d} | dot=train scale, value=raw"
                if joystick_display_scales is not None
                else f"Pred fed action {frame_index:03d}"
            ),
        )
        image = Image.alpha_composite(image, hud).convert("RGB")
        rendered.append(np.asarray(image))

    return torch.from_numpy(np.stack(rendered, axis=0)).permute(0, 3, 1, 2).float().div_(255.0)


def save_video_with_action_overlay(
    video_tensor: torch.Tensor,
    output_path: Path,
    fps: int,
    joystick: torch.Tensor,
    buttons: torch.Tensor,
    audio: torch.Tensor | None = None,
    audio_sample_rate: int | None = None,
    joystick_display_scales: tuple[float, float, float, float] | None = None,
    joystick_display_label: str | None = None,
    action_usage_label: str = "fed action",
) -> None:
    video_with_overlay = add_action_overlay(
        video_tensor,
        joystick,
        buttons,
        joystick_display_scales=joystick_display_scales,
        joystick_display_label=joystick_display_label,
        action_usage_label=action_usage_label,
    )
    save_video(
        video_tensor=video_with_overlay,
        output_path=output_path,
        fps=fps,
        audio=audio,
        audio_sample_rate=audio_sample_rate,
    )


def infer_action_button_dim_from_state(state: dict[str, torch.Tensor]) -> int | None:
    pattern = re.compile(r"(?:^|.*\.)transformer_blocks\.0\.action_button_attn\.to_v\.weight$")
    for key, value in state.items():
        if pattern.match(key) and torch.is_tensor(value) and value.ndim == 2:
            return int(value.shape[1])
    return None


def infer_action_button_state_dim_from_state(state: dict[str, torch.Tensor]) -> int | None:
    pattern = re.compile(r"(?:^|.*\.)transformer_blocks\.0\.action_button_attn\.to_k\.weight$")
    for key, value in state.items():
        if pattern.match(key) and torch.is_tensor(value) and value.ndim == 2:
            return int(value.shape[1])
    return None


def load_state_dict_streaming(
    module: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    mmap_path: Path | None = None,
    discard_interval: int = 32,
) -> tuple[list[str], list[str]]:
    ''

    target_state = module.state_dict()
    missing = [key for key in target_state if key not in state]
    unexpected = [key for key in state if key not in target_state]
    shape_mismatches: list[str] = []

    with torch.no_grad():
        for index, (key, target) in enumerate(target_state.items(), start=1):
            source = state.pop(key, None)
            if source is None:
                continue
            if tuple(source.shape) != tuple(target.shape):
                shape_mismatches.append(
                    f"{key}: checkpoint={tuple(source.shape)} model={tuple(target.shape)}"
                )
                del source
                continue
            target.copy_(source, non_blocking=False)
            del source
            if index % discard_interval == 0:
                gc.collect()
                if mmap_path is not None:
                    _discard_file_backed_pages(mmap_path)

    if shape_mismatches:
        preview = "; ".join(shape_mismatches[:8])
        raise RuntimeError(f"Checkpoint tensor shape mismatch: {preview}")

    state.clear()
    gc.collect()
    if mmap_path is not None:
        _discard_file_backed_pages(mmap_path)
    return missing, unexpected


def _discard_file_backed_pages(path: Path) -> None:
    ''

    target = str(path.resolve())
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        madvise = libc.madvise
        madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        madvise.restype = ctypes.c_int
        for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) < 6 or fields[5] != target:
                continue
            start_text, end_text = fields[0].split("-", maxsplit=1)
            start = int(start_text, 16)
            end = int(end_text, 16)
                                                                          
                                                                      
            madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start), ctypes.c_int(4))
    except (OSError, ValueError):
        return


def load_eval_components(
    base_model: Path,
    text_encoder: Path | None,
    checkpoint: Path,
    dtype: torch.dtype,
    with_audio: bool,
    transformer_device_after_load: torch.device | str | None = None,
) -> tuple[Any, Any, int]:
    defer_auxiliary_components = transformer_device_after_load is not None
    components = load_model(
        checkpoint_path=base_model,
        text_encoder_path=text_encoder,
        device=(transformer_device_after_load if defer_auxiliary_components else "cpu"),
        dtype=dtype,
        with_video_vae_encoder=not defer_auxiliary_components,
        with_video_vae_decoder=not defer_auxiliary_components,
        with_audio_vae_decoder=with_audio and not defer_auxiliary_components,
        with_vocoder=with_audio and not defer_auxiliary_components,
        with_text_encoder=not defer_auxiliary_components,
    )
    components.transformer.eval()

    if checkpoint.resolve() != base_model.resolve():
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
            if not isinstance(state, dict):
                raise TypeError(f"Expected checkpoint state dict, got {type(state).__name__}")
        except Exception as torch_load_error:
            print(f"[eval] torch.load failed for {checkpoint}: {torch_load_error}; trying safetensors.")
            state = load_file(str(checkpoint), device="cpu")
        model_keys = set(components.transformer.state_dict())
        total_params = len(model_keys)
        best_prefix: str | None = None
        best_score: tuple[int, int] | None = None
        for prefix in ("", "module.", "_orig_mod.", "diffusion_model.", "transformer."):
            candidate_keys = {
                key[len(prefix):] if prefix and key.startswith(prefix) else key
                for key in state
            }
            missing_count = len(model_keys - candidate_keys)
            unexpected_count = len(candidate_keys - model_keys)
            score = (missing_count + unexpected_count, unexpected_count)
            if best_score is None or score < best_score:
                best_score = score
                best_prefix = prefix

        assert best_prefix is not None
        if best_prefix:
            state = {
                key[len(best_prefix):] if key.startswith(best_prefix) else key: value
                for key, value in state.items()
            }
        action_button_dim = infer_action_button_dim_from_state(state)
        action_button_state_dim = infer_action_button_state_dim_from_state(state)
        if action_button_dim is not None and hasattr(components.transformer, "set_action_button_dim"):
            components.transformer.set_action_button_dim(action_button_dim, state_dim=action_button_state_dim)
                                                                               
                                                                                
                                                                                
                                                                               
                                                                             
            action_module_device = (
                transformer_device_after_load
                if defer_auxiliary_components
                else "cpu"
            )
            components.transformer.to(device=action_module_device, dtype=dtype)
            model_keys = set(components.transformer.state_dict())
            total_params = len(model_keys)
            print(
                "[eval] action dimensions inferred from checkpoint: "
                f"button_dim={action_button_dim} state_dim={action_button_state_dim}"
            )
        if defer_auxiliary_components:
                                                                               
                                                                              
                                           
            components.transformer.to(
                device=transformer_device_after_load,
                dtype=dtype,
            ).eval()
        print(f"[eval] Loading transformer checkpoint with prefix={best_prefix!r}")
        missing, unexpected = load_state_dict_streaming(
            components.transformer,
            state,
            mmap_path=checkpoint,
            discard_interval=32,
        )
        del state
        if missing or unexpected:
            print(
                "[eval] Transformer checkpoint loaded with non-strict key match: "
                f"missing={len(missing)}/{total_params}, unexpected={len(unexpected)}"
            )
            legacy_optional_missing = [
                key
                for key in missing
                if key.startswith("action_button_meaning_proj.")
                or ".action_button_attn.meaning_concat_mlp." in key
            ]
            required_missing = [key for key in missing if key not in legacy_optional_missing]
            if legacy_optional_missing:
                print(
                    "[eval] Legacy checkpoint has no button-meaning modules; "
                    f"ignoring {len(legacy_optional_missing)} unused missing tensors."
                )
            if len(required_missing) > max(4, int(total_params * 0.02)):
                raise RuntimeError(
                    "Checkpoint does not look compatible with the transformer. "
                    f"First required missing keys: {required_missing[:5]}, "
                    f"first unexpected keys: {unexpected[:5]}"
                )

    if defer_auxiliary_components:
                                                                                
                                                                                  
                                                                                   
                                                                            
        components.transformer.eval()
        components.video_vae_encoder = load_video_vae_encoder(
            checkpoint_path=base_model, device="cpu", dtype=dtype
        )
        components.video_vae_decoder = load_video_vae_decoder(
            checkpoint_path=base_model, device="cpu", dtype=dtype
        )
        if with_audio:
            components.audio_vae_decoder = load_audio_vae_decoder(
                checkpoint_path=base_model, device="cpu", dtype=dtype
            )
            components.vocoder = load_vocoder(
                checkpoint_path=base_model, device="cpu", dtype=dtype
            )
        if text_encoder is None:
            raise ValueError("text_encoder is required for evaluation")
        components.text_encoder = load_text_encoder(
            gemma_model_path=text_encoder, device="cpu", dtype=dtype
        )

    embeddings_processor = load_embeddings_processor(
        checkpoint_path=base_model,
        device="cpu",
        dtype=dtype,
    )
    return components, embeddings_processor, int(getattr(components.transformer, "action_button_dim", len(CANONICAL_BUTTONS)))


def build_guider_params(config: dict[str, Any], base_model: Path) -> tuple[MultiModalGuiderParams, MultiModalGuiderParams]:
    validation = ValidationConfig(**(config.get("validation", {}) or {}))
    return validation.resolved_guider_params(base_model)


def compute_video_metrics(pred: torch.Tensor, gt_fchw: torch.Tensor) -> dict[str, float]:
                                               
    gt = gt_fchw.permute(1, 0, 2, 3).to(pred.device, pred.dtype)
    frames = min(pred.shape[1], gt.shape[1])
    pred = pred[:, :frames].clamp(0, 1).float()
    gt = gt[:, :frames].clamp(0, 1).float()
    if frames > 1:
        pred = pred[:, 1:]
        gt = gt[:, 1:]
    mse = torch.mean((pred - gt) ** 2).item()
    l1 = torch.mean(torch.abs(pred - gt)).item()
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    return {
        "frame_l1_excluding_condition": float(l1),
        "frame_mse_excluding_condition": float(mse),
        "frame_psnr_excluding_condition": float(psnr),
    }


def save_side_by_side(path: Path, gt: torch.Tensor, pred: torch.Tensor, fps: int) -> None:
    gt_cfhw = gt.permute(1, 0, 2, 3).clamp(0, 1)
    frames = min(gt_cfhw.shape[1], pred.shape[1])
    side = torch.cat([gt_cfhw[:, :frames], pred[:, :frames].detach().cpu().clamp(0, 1)], dim=-1)
    save_video(video_tensor=side, output_path=path, fps=fps)


def save_side_by_side_with_action_overlay(
    path: Path,
    gt: torch.Tensor,
    pred: torch.Tensor,
    fps: int,
    gt_joystick: torch.Tensor,
    gt_buttons: torch.Tensor,
    pred_joystick: torch.Tensor,
    pred_buttons: torch.Tensor,
    audio: torch.Tensor | None = None,
    audio_sample_rate: int | None = None,
    joystick_display_scales: tuple[float, float, float, float] | None = None,
) -> None:
    gt_cfhw = gt.permute(1, 0, 2, 3).clamp(0, 1)
    frames = min(gt_cfhw.shape[1], pred.shape[1])
    side = torch.cat([gt_cfhw[:, :frames], pred[:, :frames].detach().cpu().clamp(0, 1)], dim=-1)
    side = add_dual_action_overlay(
        video=side,
        left_joystick=gt_joystick[:frames],
        left_buttons=gt_buttons[:frames],
        right_joystick=pred_joystick[:frames],
        right_buttons=pred_buttons[:frames],
        joystick_display_scales=joystick_display_scales,
    )
    save_video(
        video_tensor=side,
        output_path=path,
        fps=fps,
        audio=audio,
        audio_sample_rate=audio_sample_rate,
    )


def run_case(
    case: dict[str, Any],
    sampler: ValidationSampler,
    output_dir: Path,
    action_context: EvalActionConditioning,
    num_inference_steps: int,
    seed: int | None,
    skip_audio: bool,
    skip_existing: bool,
    device: str,
    video_guider_params: MultiModalGuiderParams,
    audio_guider_params: MultiModalGuiderParams | None,
    scheduler_timestep_shift: float | None,
    action_guidance_scale: float,
    joystick_guidance_scale: float,
    action_button_temporal_alignment: str,
    generated_audio_sample_rate: int | None,
    tiled_decoding: TiledDecodingConfig | bool | None,
    joystick_input_space: str,
    joystick_profile: JoystickUnitScaleProfile | None,
    joystick_display_profile: JoystickUnitScaleProfile | None,
) -> dict[str, Any]:
    case_dir = output_dir / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    pred_path = case_dir / "pred.mp4"
    pred_actions_path = case_dir / "pred_actions.mp4"
    side_actions_path = case_dir / "side_by_side_actions.mp4"
    metrics_path = case_dir / "metrics.json"
    if skip_existing and pred_path.exists() and pred_actions_path.exists() and side_actions_path.exists() and metrics_path.exists():
        print(f"[eval] skip existing {case['case_id']}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    height = int(case.get("height", DEFAULT_HEIGHT))
    width = int(case.get("width", DEFAULT_WIDTH))
    fps = int(case.get("fps", DEFAULT_FPS))
    num_frames = int(case.get("num_frames", DEFAULT_NUM_FRAMES))
    crop_start = int(case["crop_start"])
    source_fps = float(case.get("source_fps") or fps)
    video_source_fps = float(case.get("video_source_fps") or source_fps)
    video_offset_sec = float(case.get("video_offset_sec") or 0.0)
    audio_offset_sec = float(case.get("audio_offset_sec") or video_offset_sec)
    action_row_start = int(case.get("action_row_start") or 0)
    action_row_scale = float(case.get("action_row_scale") or 1.0)
    button_schema = str(case.get("button_schema") or "legacy")
    if action_context.mode == "crossfps_unified":
        button_schema = "crossfps_unified"
    elif action_context.mode == "crossfps_legacy6":
        button_schema = "legacy"
    raw_game_id = _case_raw_game_id(case)

    gt_video = load_video_window(
        case["video_path"],
        crop_start,
        num_frames,
        height,
        width,
        source_fps=source_fps,
        video_source_fps=video_source_fps,
        video_offset_sec=video_offset_sec,
    )
    gt_audio, gt_audio_sr = load_audio_window(
        case["audio_path"],
        crop_start,
        num_frames,
        fps,
        audio_offset_sec=audio_offset_sec,
    )
    gt_joystick, gt_buttons = load_action_window(
        case["action_path"],
        crop_start,
        num_frames,
        "real",
        action_row_start=action_row_start,
        action_row_scale=action_row_scale,
        button_schema=button_schema,
        game_id=raw_game_id,
    )
    fed_joystick, fed_buttons = load_action_window(
        case["action_path"],
        crop_start,
        num_frames,
        case["variant"],
        synthetic_action=case.get("synthetic_action"),
        action_row_start=action_row_start,
        action_row_scale=action_row_scale,
        button_schema=button_schema,
        game_id=raw_game_id,
    )
    source_gt_joystick = gt_joystick
    source_fed_joystick = fed_joystick
    gt_joystick = adapt_eval_joystick_input(
        source_gt_joystick,
        input_space=joystick_input_space,
        profile=joystick_profile,
    )
    fed_joystick = adapt_eval_joystick_input(
        source_fed_joystick,
        input_space=joystick_input_space,
        profile=joystick_profile,
    )

    save_video(
        video_tensor=gt_video.permute(1, 0, 2, 3),
        output_path=case_dir / "gt.mp4",
        fps=fps,
        audio=gt_audio,
        audio_sample_rate=gt_audio_sr,
    )
    save_video_with_action_overlay(
        video_tensor=gt_video.permute(1, 0, 2, 3),
        output_path=case_dir / "gt_actions.mp4",
        fps=fps,
        joystick=gt_joystick,
        buttons=gt_buttons,
        audio=gt_audio,
        audio_sample_rate=gt_audio_sr,
        joystick_display_scales=(
            None if joystick_display_profile is None else joystick_display_profile.scales
        ),
    )
    save_action_plot(
        case_dir / "gt_action.png",
        gt_joystick,
        gt_buttons,
        f"{case['case_id']} gt real",
        joystick_display_scales=(
            None if joystick_display_profile is None else joystick_display_profile.scales
        ),
    )
    save_action_plot(
        case_dir / "action.png",
        fed_joystick,
        fed_buttons,
        f"{case['case_id']} fed",
        joystick_display_scales=(
            None if joystick_display_profile is None else joystick_display_profile.scales
        ),
    )

    model_prompt = str(case["prompt"])
    model_buttons: torch.Tensor | None = fed_buttons
    model_button_meaning_texts: dict[str, str] | None = case.get("button_meaning_texts")
    timeline_text = ""
    if action_context.mode == "semantic_vocab":
        model_buttons = map_eval_buttons_to_semantic_vocab(case, fed_buttons, action_context)
        model_button_meaning_texts = None
    elif action_context.mode == "crossfps_unified":
        model_buttons = validate_crossfps_unified_eval_buttons(fed_buttons, action_context)
        model_button_meaning_texts = None
    elif action_context.mode == "crossfps_legacy6":
        if fed_buttons.shape[-1] != len(LEGACY_CROSSFPS_BUTTONS):
            raise ValueError(
                f"crossfps_legacy6 eval expects {len(LEGACY_CROSSFPS_BUTTONS)} buttons, "
                f"got {fed_buttons.shape[-1]}"
            )
        model_button_meaning_texts = None
    elif action_context.mode == "prompt_timeline":
        if action_guidance_scale > 0.0:
            raise ValueError("prompt_timeline eval does not support action_guidance_scale > 0 because buttons are not fed")
        model_prompt, timeline_text = build_eval_prompt_timeline(
            case,
            fed_buttons,
            source_fps,
            action_context,
        )
        model_buttons = None
        model_button_meaning_texts = None

    negative_prompt = str(case.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT)
    config = GenerationConfig(
        prompt=model_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=fps,
        num_inference_steps=num_inference_steps,
        generate_audio=not skip_audio,
        condition_image=gt_video[0],
        action_joystick=fed_joystick,
        action_buttons=model_buttons,
        action_guidance_scale=action_guidance_scale,
        joystick_guidance_scale=joystick_guidance_scale,
        action_button_temporal_alignment=action_button_temporal_alignment,
        button_meaning_texts=model_button_meaning_texts,
        action_button_meaning_fusion=str(case.get("action_button_meaning_fusion") or "add"),
        seed=int(seed if seed is not None else case.get("seed", 0)),
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params or MultiModalGuiderParams(),
        scheduler_timestep_shift=scheduler_timestep_shift,
        tiled_decoding=tiled_decoding,
    )

    with torch.inference_mode():
        video, audio, _ = sampler.generate(config=config, device=device)

    save_video(
        video_tensor=video.detach().cpu().clamp(0, 1),
        output_path=pred_path,
        fps=fps,
        audio=None if skip_audio or audio is None else audio.detach().cpu(),
        audio_sample_rate=None if skip_audio or audio is None else generated_audio_sample_rate,
    )
    save_video_with_action_overlay(
        video_tensor=video.detach().cpu().clamp(0, 1),
        output_path=pred_actions_path,
        fps=fps,
        joystick=fed_joystick,
        buttons=fed_buttons,
        audio=None if skip_audio or audio is None else audio.detach().cpu(),
        audio_sample_rate=None if skip_audio or audio is None else generated_audio_sample_rate,
        joystick_display_scales=(
            None if joystick_display_profile is None else joystick_display_profile.scales
        ),
    )
    save_side_by_side(case_dir / "side_by_side.mp4", gt_video, video.detach().cpu(), fps)
    save_side_by_side_with_action_overlay(
        path=side_actions_path,
        gt=gt_video,
        pred=video.detach().cpu(),
        fps=fps,
        gt_joystick=gt_joystick,
        gt_buttons=gt_buttons,
        pred_joystick=fed_joystick,
        pred_buttons=fed_buttons,
        audio=None if skip_audio or audio is None else audio.detach().cpu(),
        audio_sample_rate=None if skip_audio or audio is None else generated_audio_sample_rate,
        joystick_display_scales=(
            None if joystick_display_profile is None else joystick_display_profile.scales
        ),
    )

    metrics = {
        "case_id": case["case_id"],
        "group": case["group"],
        "variant": case["variant"],
        "sample_id": case["sample_id"],
        "game": case["game"],
        "train_manifest_member": case.get("train_manifest_member"),
        "train_membership": case.get("train_membership"),
        "train_membership_key": case.get("train_membership_key"),
        "train_membership_manifest_path": case.get("train_membership_manifest_path"),
        "crop_start": crop_start,
        "prompt": model_prompt,
        "negative_prompt": negative_prompt,
        "source_prompt": case.get("source_prompt"),
        "control_binding_text": case.get("control_binding_text"),
        "button_meaning_texts": model_button_meaning_texts,
        "action_conditioning": {
            "mode": action_context.mode,
            "model_button_dim": None if model_buttons is None else int(model_buttons.shape[-1]),
            "checkpoint_button_dim": int(action_context.model_button_dim),
            "prompt_timeline_text": timeline_text,
        },
        "num_inference_steps": num_inference_steps,
        "action_guidance_scale": action_guidance_scale,
        "joystick_guidance_scale": joystick_guidance_scale,
        "joystick_input_space": joystick_input_space,
        "joystick_normalization": (
            None
            if joystick_profile is None
            else {
                "profile_id": joystick_profile.profile_id,
                "profile_path": str(joystick_profile.path),
                "operation": "model_input = normalized_unit * scale",
                "scales": list(joystick_profile.scales),
            }
        ),
        "skip_audio": bool(skip_audio),
        "video": compute_video_metrics(video.detach().cpu(), gt_video),
        "paths": {
            "gt": str(case_dir / "gt.mp4"),
            "gt_actions": str(case_dir / "gt_actions.mp4"),
            "pred": str(pred_path),
            "pred_actions": str(pred_actions_path),
            "side_by_side": str(case_dir / "side_by_side.mp4"),
            "side_by_side_actions": str(side_actions_path),
            "action_plot": str(case_dir / "action.png"),
            "gt_action_plot": str(case_dir / "gt_action.png"),
        },
        "source_gt_action_stats": action_tensor_stats(source_gt_joystick, gt_buttons),
        "source_fed_action_stats": action_tensor_stats(source_fed_joystick, fed_buttons),
        "gt_action_stats": action_tensor_stats(gt_joystick, gt_buttons),
        "fed_action_stats": action_tensor_stats(fed_joystick, fed_buttons),
    }
    saved_case = dict(case)
    saved_case["eval_prompt"] = model_prompt
    saved_case["eval_action_conditioning"] = metrics["action_conditioning"]
    saved_case["eval_joystick_input_space"] = joystick_input_space
    saved_case["eval_joystick_normalization"] = metrics["joystick_normalization"]
    (case_dir / "case.json").write_text(json.dumps(saved_case, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[eval] {case['case_id']} "
        f"l1={metrics['video']['frame_l1_excluding_condition']:.4f} "
        f"psnr={metrics['video']['frame_psnr_excluding_condition']:.2f}"
    )
    return metrics


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({row["group"] for row in rows})
    summary: dict[str, Any] = {"num_cases": len(rows), "groups": {}}
    memberships = [row.get("train_membership") for row in rows if row.get("train_membership")]
    if memberships:
        summary["train_membership"] = {
            membership: sum(item == membership for item in memberships)
            for membership in sorted(set(memberships))
        }
    for group in groups:
        group_rows = [row for row in rows if row["group"] == group]
        l1 = [r["video"]["frame_l1_excluding_condition"] for r in group_rows]
        psnr = [r["video"]["frame_psnr_excluding_condition"] for r in group_rows]
        summary["groups"][group] = {
            "num_cases": len(group_rows),
            "mean_l1": float(np.mean(l1)) if l1 else None,
            "mean_psnr": float(np.mean(psnr)) if psnr else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ltx2_p0_crossfps_action_audio.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--text-encoder", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument(
        "--train-membership-manifest",
        default=None,
        help=(
            "Manifest used to label sampled eval cases as in_train_manifest or "
            "filtered_out_of_train_manifest. Defaults to the manifest in --config; "
            "pass 'none' to disable."
        ),
    )
    parser.add_argument("--cases-path", default="eval/crossfps_action_audio_eval_cases.jsonl")
    parser.add_argument("--unit-anchor-cases-path", default="eval/crossfps_action_audio_eval_cases.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--action-guidance-scale", type=float, default=0.0)
    parser.add_argument("--joystick-guidance-scale", type=float, default=0.0)
    parser.add_argument(
        "--joystick-input-space",
        choices=("model_raw", "normalized_unit"),
        default="model_raw",
        help=(
            "model_raw feeds dataset values unchanged. normalized_unit requires "
            "--joystick-normalization-profile and applies an inference-only per-axis scale."
        ),
    )
    parser.add_argument(
        "--joystick-normalization-profile",
        type=Path,
        help="Versioned per-axis scale profile used only with --joystick-input-space normalized_unit.",
    )
    parser.add_argument(
        "--joystick-display-profile",
        type=Path,
        help=(
            "Display-only per-axis train scale. HUD dots and action plots use model_input / scale; "
            "the numeric HUD values and model conditioning remain unchanged."
        ),
    )
    parser.add_argument(
        "--action-button-temporal-alignment",
        choices=("auto", "global", "vae_window", "causal_prefix_rope"),
        default="auto",
        help="auto reads training_strategy.action_button_temporal_alignment from --config.",
    )
    parser.add_argument(
        "--action-conditioning-mode",
        choices=(
            "auto",
            "text_binding",
            "semantic_vocab",
            "prompt_timeline",
            "crossfps_unified",
            "crossfps_legacy6",
        ),
        default="auto",
        help=(
            "How eval should feed button actions. auto first uses "
            "action_binding.button_conditioning_mode from --config."
        ),
    )
    parser.add_argument(
        "--semantic-action-vocab-path",
        default=None,
        help="Semantic action vocab YAML used when --action-conditioning-mode=semantic_vocab.",
    )
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--case-groups", nargs="*", default=None)
    parser.add_argument("--rebuild-cases", action="store_true")
    parser.add_argument("--build-action-unit-cases", action="store_true")
    parser.add_argument("--cases-only", action="store_true")
    parser.add_argument("--case-build-workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument(
        "--eval-sampling-mode",
        choices=("random", "action_extreme"),
        default="random",
        help="random is the main eval mode; action_extreme preserves the old stress-set sampler.",
    )
    parser.add_argument("--random-cases-source", default=None)
    parser.add_argument("--num-random-real-cases", type=int, default=DEFAULT_RANDOM_REAL_CASES)
    parser.add_argument("--num-joystick-reverse-cases", type=int, default=DEFAULT_JOYSTICK_REVERSE_CASES)
    parser.add_argument(
        "--exclude-prompt-flagged-random-cases",
        dest="exclude_prompt_flagged_random_cases",
        action="store_true",
        default=False,
        help="Exclude random-source cases flagged as non-primary FPS by prompt or raw review metadata.",
    )
    parser.add_argument(
        "--include-prompt-flagged-random-cases",
        dest="exclude_prompt_flagged_random_cases",
        action="store_false",
        help="Allow flagged random-source cases for explicit long-tail/debug evaluations.",
    )
    parser.add_argument(
        "--allow-long-tail-cases",
        action="store_true",
        help="Disable the default eval-domain filter that removes vehicle/aerial/third-person long-tail clips.",
    )
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--vae-decoding-mode",
        choices=("trainer_default", "core_default", "full"),
        default="trainer_default",
        help=(
            "Video VAE decoding mode. trainer_default preserves the historical "
            "192px/48f validation tiles; core_default uses ltx-core's "
            "512px/64f tiles; full disables tiling."
        ),
    )
    args = parser.parse_args()
    if args.joystick_input_space == "normalized_unit" and args.joystick_normalization_profile is None:
        parser.error("--joystick-input-space normalized_unit requires --joystick-normalization-profile")
    if args.joystick_input_space == "model_raw" and args.joystick_normalization_profile is not None:
        parser.error("--joystick-normalization-profile requires --joystick-input-space normalized_unit")

    joystick_profile = (
        load_joystick_unit_scale_profile(args.joystick_normalization_profile)
        if args.joystick_input_space == "normalized_unit"
        else None
    )
    joystick_display_profile = (
        load_joystick_unit_scale_profile(args.joystick_display_profile)
        if args.joystick_display_profile is not None
        else None
    )

    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    manifest_path = resolve_manifest(config, args.manifest_path)
    train_membership_manifest = resolve_train_membership_manifest(config, args.train_membership_manifest)
    cases_path = Path(args.cases_path).expanduser().resolve()

    if args.build_action_unit_cases or (not cases_path.exists() and "action_unit" in cases_path.name):
        cases = build_action_unit_cases(
            source_cases_path=Path(args.unit_anchor_cases_path).expanduser().resolve(),
            cases_path=cases_path,
            num_frames=args.num_frames,
            fps=args.fps,
            seed=args.seed,
        )
    elif args.rebuild_cases or not cases_path.exists():
        cases = build_cases(
            manifest_path=manifest_path,
            cases_path=cases_path,
            num_frames=args.num_frames,
            fps=args.fps,
            seed=args.seed,
            workers=args.case_build_workers,
            allow_long_tail=args.allow_long_tail_cases,
            sampling_mode=args.eval_sampling_mode,
            random_cases_source=Path(args.random_cases_source).expanduser().resolve() if args.random_cases_source else None,
            num_random_real_cases=args.num_random_real_cases,
            num_joystick_reverse_cases=args.num_joystick_reverse_cases,
            exclude_prompt_flagged_random_cases=args.exclude_prompt_flagged_random_cases,
        )
    else:
        cases = read_jsonl(cases_path)
        print(f"[eval] Loaded {len(cases)} cases from {cases_path}")

    cases, case_conversion_summary = prepare_text_binding_cases(
        cases,
        manifest_path=manifest_path,
        num_frames=args.num_frames,
        fps=args.fps,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
    )
    if case_conversion_summary is not None:
        print(f"[eval] Case conversion: {json.dumps(case_conversion_summary, sort_keys=True)}")

    train_membership_summary = None
    if train_membership_manifest is not None:
        if not train_membership_manifest.exists():
            raise FileNotFoundError(f"Train membership manifest does not exist: {train_membership_manifest}")
        train_membership_ids = load_manifest_membership_ids(train_membership_manifest)
        train_membership_summary = annotate_train_membership(
            cases,
            membership_manifest_path=train_membership_manifest,
            membership_ids=train_membership_ids,
        )
        if case_conversion_summary is None:
            write_jsonl(cases_path, cases)
        membership_summary_path = cases_path.with_suffix(".train_membership.summary.json")
        membership_summary_path.write_text(
            json.dumps(train_membership_summary, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        print(
            "[eval] Train membership: "
            f"{train_membership_summary['counts']} via {train_membership_manifest}"
        )

    if args.case_groups:
        allowed = set(args.case_groups)
        cases = [case for case in cases if case["group"] in allowed]
        print(f"[eval] Filtered to {len(cases)} cases by groups: {sorted(allowed)}")
    if args.case_offset:
        if args.case_offset < 0:
            raise ValueError("--case-offset must be >= 0")
        cases = cases[args.case_offset:]
        print(f"[eval] Applied case_offset={args.case_offset}; remaining cases={len(cases)}")
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
        print(f"[eval] Truncated to max_cases={args.max_cases}")
    if args.cases_only:
        return
    if not cases:
        raise RuntimeError("No evaluation cases selected.")

    checkpoint = find_latest_checkpoint(config, args.checkpoint)
    base_model = resolve_base_model(config, args.base_model)
    text_encoder = resolve_text_encoder(config, args.text_encoder)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    default_output_dir = Path(config["output_dir"]) / "eval"
    output_root = Path(args.output_dir).expanduser() if args.output_dir else default_output_dir
    run_name = f"step_{parse_step(checkpoint):08d}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = (output_root / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] Config: {config_path}")
    print(f"[eval] Manifest: {manifest_path}")
    if train_membership_manifest is not None:
        print(f"[eval] Train membership manifest: {train_membership_manifest}")
    print(f"[eval] Cases: {cases_path}")
    print(f"[eval] Checkpoint: {checkpoint}")
    print(f"[eval] Base model: {base_model}")
    print(f"[eval] Output: {output_dir}")
    if joystick_profile is None:
        print("[eval] Joystick adapter: none input_space=model_raw")
    else:
        print(
            f"[eval] Joystick adapter: {joystick_profile.profile_id} "
            f"scales={','.join(f'{value:.8f}' for value in joystick_profile.scales)}"
        )
    if joystick_display_profile is None:
        print("[eval] Joystick display: raw [-1,1] HUD scale")
    else:
        print(
            f"[eval] Joystick display only: {joystick_display_profile.profile_id} "
            f"scales={','.join(f'{value:.8f}' for value in joystick_display_profile.scales)}"
        )

    components, embeddings_processor, model_button_dim = load_eval_components(
        base_model=base_model,
        text_encoder=text_encoder,
        checkpoint=checkpoint,
        dtype=dtype,
        with_audio=not args.skip_audio,
    )
    action_context = build_eval_action_conditioning(
        config=config,
        config_path=config_path,
        cli_mode=args.action_conditioning_mode,
        model_button_dim=model_button_dim,
        explicit_semantic_vocab_path=args.semantic_action_vocab_path,
    )
    print(
        "[eval] Action conditioning: "
        f"mode={action_context.mode} model_button_dim={action_context.model_button_dim} "
        f"semantic_vocab={len(action_context.semantic_vocab)}"
    )
    video_guider, audio_guider = build_guider_params(config, base_model)
    video_guider_info = _guider_params_to_dict(video_guider)
    audio_guider_info = _guider_params_to_dict(audio_guider)
    scheduler_timestep_shift = _scheduler_timestep_shift_from_config(config)
    tiled_decoding: TiledDecodingConfig | bool | None
    if args.vae_decoding_mode == "trainer_default":
        tiled_decoding = None
    elif args.vae_decoding_mode == "core_default":
        tiled_decoding = TiledDecodingConfig(
            tile_size_pixels=512,
            tile_overlap_pixels=64,
            tile_size_frames=64,
            tile_overlap_frames=24,
        )
    else:
        tiled_decoding = False
    action_button_temporal_alignment = _action_button_temporal_alignment_from_config(
        config,
        args.action_button_temporal_alignment,
    )
    print(f"[eval] Video guider: {json.dumps(video_guider_info, sort_keys=True)}")
    print(
        "[eval] Audio guider: "
        f"{json.dumps(audio_guider_info, sort_keys=True)}"
        f"{' (resolved but disabled by --skip-audio)' if args.skip_audio else ''}"
    )
    print(f"[eval] Scheduler timestep_shift: {scheduler_timestep_shift}")
    print(f"[eval] Action button temporal alignment: {action_button_temporal_alignment}")
    sampler = ValidationSampler(
        transformer=components.transformer,
        vae_decoder=components.video_vae_decoder,
        vae_encoder=components.video_vae_encoder,
        text_encoder=components.text_encoder,
        audio_decoder=components.audio_vae_decoder if not args.skip_audio else None,
        vocoder=components.vocoder if not args.skip_audio else None,
        embeddings_processor=embeddings_processor,
    )
    generated_audio_sample_rate = (
        int(components.vocoder.output_sampling_rate)
        if not args.skip_audio and components.vocoder is not None
        else None
    )

    shutil.copy2(config_path, output_dir / "train_config.yaml")
    write_jsonl(output_dir / "cases.jsonl", cases)
    run_info = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "config": str(config_path),
        "manifest_path": str(manifest_path),
        "train_membership": train_membership_summary,
        "case_conversion": case_conversion_summary,
        "cases_path": str(cases_path),
        "checkpoint": str(checkpoint),
        "base_model": str(base_model),
        "text_encoder": str(text_encoder) if text_encoder else None,
        "num_cases": len(cases),
        "num_inference_steps": args.num_inference_steps,
        "scheduler": {
            "source": "flow_matching.sigma_sampling.timestep_shift",
            "timestep_shift": scheduler_timestep_shift,
        },
        "action_guidance_scale": args.action_guidance_scale,
        "joystick_guidance_scale": args.joystick_guidance_scale,
        "joystick_input_space": args.joystick_input_space,
        "joystick_normalization": (
            None
            if joystick_profile is None
            else {
                "profile_id": joystick_profile.profile_id,
                "profile_path": str(joystick_profile.path),
                "operation": "model_input = normalized_unit * scale",
                "scales": list(joystick_profile.scales),
            }
        ),
        "joystick_display": (
            None
            if joystick_display_profile is None
            else {
                "profile_id": joystick_display_profile.profile_id,
                "profile_path": str(joystick_display_profile.path),
                "operation": "dot = clip(model_input / scale, -1, 1); numeric value = model_input",
                "model_input_unchanged": True,
                "scales": list(joystick_display_profile.scales),
            }
        ),
        "action_button_temporal_alignment": action_button_temporal_alignment,
        "action_conditioning": {
            "mode": action_context.mode,
            "model_button_dim": action_context.model_button_dim,
            "semantic_vocab_size": len(action_context.semantic_vocab),
            "timeline_prompt": action_context.timeline_prompt,
            "use_scene_prompt": action_context.use_scene_prompt,
        },
        "device": args.device,
        "dtype": args.dtype,
        "skip_audio": args.skip_audio,
        "vae_decoding_mode": args.vae_decoding_mode,
        "tiled_decoding": asdict(tiled_decoding) if isinstance(tiled_decoding, TiledDecodingConfig) else tiled_decoding,
        "guidance": {
            "source": "ValidationConfig.resolved_guider_params(base_model)",
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "video_guider": video_guider_info,
            "audio_guider": None if args.skip_audio else audio_guider_info,
            "audio_guider_resolved": audio_guider_info,
            "audio_guidance_used": not args.skip_audio,
        },
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics_rows: list[dict[str, Any]] = []
    for case in cases:
        try:
            metrics_rows.append(
                run_case(
                    case=case,
                    sampler=sampler,
                    output_dir=output_dir,
                    action_context=action_context,
                    num_inference_steps=args.num_inference_steps,
                    seed=None,
                    skip_audio=args.skip_audio,
                    skip_existing=args.skip_existing,
                    device=args.device,
                    video_guider_params=video_guider,
                    audio_guider_params=None if args.skip_audio else audio_guider,
                    scheduler_timestep_shift=scheduler_timestep_shift,
                    action_guidance_scale=args.action_guidance_scale,
                    joystick_guidance_scale=args.joystick_guidance_scale,
                    action_button_temporal_alignment=action_button_temporal_alignment,
                    generated_audio_sample_rate=generated_audio_sample_rate,
                    tiled_decoding=tiled_decoding,
                    joystick_input_space=args.joystick_input_space,
                    joystick_profile=joystick_profile,
                    joystick_display_profile=joystick_display_profile,
                )
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    write_jsonl(output_dir / "metrics.jsonl", metrics_rows)
    summary = aggregate_metrics(metrics_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval] Done. Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
