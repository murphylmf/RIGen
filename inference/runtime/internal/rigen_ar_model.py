                      
''

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
for _relative_source in (
    "packages/ltx-trainer/src",
    "packages/ltx-core/src",
    "packages/ltx-pipelines/src",
):
    _source_path = REPO_ROOT / _relative_source
    if str(_source_path) not in sys.path:
        sys.path.insert(0, str(_source_path))

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.transformer.attention import (
    configure_attention_backend,
)
from ltx_core.model.transformer.rope import apply_rotary_emb
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape
from ltx_core.utils import to_denoised
from ltx_trainer.ar_kv_cache import (
    ARTeacherKVCacheRunner,
    estimate_cache_bank_bytes,
    slice_ar_model_inputs,
)
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.fa4_mask_execution import configure_exact_segmented_fa4
from ltx_trainer.training_strategies.ar_teacher import (
    ARTeacherConfig,
    ARTeacherStrategy,
)
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.validation_sampler import (
    GenerationConfig,
    TiledDecodingConfig,
    ValidationSampler,
)
from ltx_trainer.video_utils import save_video
from ltx_trainer.vlm_agent import (
    AIM_TARGET_RING_NEUTRAL_PROMPT,
    FPSBlockActionPlan,
    FPSNavigationObservation,
    InteractionMode,
    JoystickPixelLookup,
    OpenAIVLMObserver,
    TwoConfirmedHitKillRule,
    VLMObservation,
    audit_interaction_guidance,
    build_duel_lifecycle_prompts,
    build_interaction_prompts,
    center_crosshair_crop,
    draw_aim_center_ring,
    draw_center_crosshair,
    draw_direct_crosshair,
    extract_prompt_section,
    freeze_opponent_motion,
    plan_fps_block_actions,
    set_player_killed_outcome,
    shift_image_for_camera_projection,
    strict_consensus_observation,
)

from rigen_generation import (
    compute_video_metrics,
    load_action_window,
    load_audio_window,
    load_eval_components,
    load_video_window,
    save_action_plot,
    save_video_with_action_overlay,
)


DEFAULT_CONFIG = (
    REPO_ROOT / "configs/ltx2_ar_teacher_crossfps_c25k_ti2av_chunk2_20fps_2x4_8gpu.yaml"
)
DEFAULT_RUN = (
    REPO_ROOT
    / "outputs/ltx2_ar_teacher_crossfps_c25k_ti2av_chunk2_20fps_3x4_12gpu_fullft"
)
DEFAULT_CHECKPOINT = DEFAULT_RUN / "checkpoints/model_weights_step_05000.safetensors"
DEFAULT_MANIFEST_ROOT = (
    REPO_ROOT
    / "outputs/data_reviews/crossfps_single_enemy_pass2_timeline_v0_8"
    / "full_pass15_accept4870_luna_low_v01_20260719"
    / "training_manifest_heldout100_v01"
)
DEFAULT_HELDOUT_MANIFEST = (
    DEFAULT_MANIFEST_ROOT / "bidirectional_training_manifest_heldout.jsonl"
)
DEFAULT_VARIABLE_MANIFEST = (
    DEFAULT_MANIFEST_ROOT / "ar_teacher_chunk2_training_manifest_train.jsonl"
)
DEFAULT_NEGATIVE_PROMPT = "low quality, distorted, blurry, text, watermark"


class FixedSigmaSampler:
    ''

    def __init__(self, sigma: Tensor) -> None:
        self._sigma = sigma

    def sample(
        self,
        *,
        batch_size: int,
        seq_length: int,
        device: torch.device,
    ) -> Tensor:
        del seq_length
        return self._sigma.to(device=device, dtype=torch.float32).expand(batch_size)


@dataclass(frozen=True)
class VLMTwoHitConfig:
    observer: OpenAIVLMObserver
    min_hit_confidence: float = 0.55
    required_post_shot_frames: int = 6
    decision_mode: str = "posthoc"
    initial_hp: int = 100
    confirmed_hit_damage: int = 50
    pre_action_votes: int = 1
    pre_action_grounding: bool = False
    joystick_pixel_lookup: JoystickPixelLookup | None = None
    freeze_opponent_motion: bool = False
    direct_crosshair_crop_size: int | None = None
    aim_target_ring_v9: bool = False
    duel_loop: bool = False
    duel_respawn_seconds: float = 3.0
    duel_player_initial_hp: int = 100
    duel_enemy_shot_damage: int = 1
    deferred_healthcheck_timeout_seconds: float = 0.0


@dataclass(frozen=True)
class VLMActionControllerConfig:
    observer: OpenAIVLMObserver
    move_magnitude: float = 0.20
    strafe_magnitude: float = 0.18
    turn_magnitude: float = 0.025
    aim_magnitude: float = 0.035
    vertical_aim_magnitude: float = 0.018
    fire_pulse_offset: int = 12
    deferred_healthcheck_timeout_seconds: float = 0.0


def wait_for_vlm_healthcheck(
    observer: OpenAIVLMObserver,
    *,
    timeout_seconds: float,
) -> None:
    ''

    if timeout_seconds <= 0:
        return
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    print(
        "[vlm] video model is on device; waiting for deferred VLM service "
        f"(timeout={timeout_seconds:.0f}s)",
        flush=True,
    )
    while time.monotonic() < deadline:
        try:
            observer.healthcheck()
            print("[vlm] deferred healthcheck passed", flush=True)
            return
        except Exception as error:                                                    
            last_error = error
            time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
    raise RuntimeError(
        f"VLM service did not become healthy within {timeout_seconds:.0f}s"
    ) from last_error


@dataclass(frozen=True)
class SlidingWindowInputs:
    ''

    inputs: ModelInputs
    video_current: slice
    audio_current: slice
    retained_clean_blocks: tuple[int, ...]


def fire_pulse_frames(action_buttons: Tensor, *, threshold: float = 0.5) -> list[int]:
    ''

    if action_buttons.ndim != 2 or action_buttons.shape[1] < 1:
        raise ValueError(
            f"Expected [frames, buttons] action tensor, got {tuple(action_buttons.shape)}"
        )
    active = action_buttons[:, 0] > threshold
    previous = torch.cat([torch.zeros(1, dtype=torch.bool), active[:-1].cpu()])
    rising = active.cpu() & ~previous
    return torch.nonzero(rising, as_tuple=False).flatten().tolist()


def resolve_vlm_decision_fire_frames(
    action_buttons: Tensor,
    configured_frames: Any | None,
) -> tuple[list[int], list[int]]:
    ''

    action_frames = fire_pulse_frames(action_buttons)
    if configured_frames is None:
        if len(action_frames) != 2:
            raise ValueError(
                "--vlm-two-hit-kill requires exactly two right-trigger pulses unless "
                "vlm_decision_fire_frames explicitly selects the decision pulses; "
                f"found {action_frames}"
            )
        return action_frames, action_frames

    if not isinstance(configured_frames, list) or not configured_frames:
        raise ValueError("vlm_decision_fire_frames must be a non-empty JSON list")
    if any(
        isinstance(frame, bool) or not isinstance(frame, int)
        for frame in configured_frames
    ):
        raise ValueError(
            "vlm_decision_fire_frames entries must be integer frame indices"
        )
    decision_frames = list(configured_frames)
    if decision_frames != sorted(set(decision_frames)):
        raise ValueError("vlm_decision_fire_frames must be sorted and unique")
    missing = [frame for frame in decision_frames if frame not in action_frames]
    if missing:
        raise ValueError(
            "Every VLM decision frame must be a right-trigger rising edge; "
            f"missing={missing}, action_fire_frames={action_frames}"
        )
    return action_frames, decision_frames


def event_frame_indices(
    *,
    event_kind: str,
    available_frames: int,
    shot_frame: int | None = None,
    chunk_start: int | None = None,
    chunk_end: int | None = None,
) -> list[int]:
    if event_kind == "shot":
        if shot_frame is None:
            raise ValueError("shot_frame is required for shot sampling")
        candidates = [shot_frame + offset for offset in (-4, -2, -1, 0, 1, 3, 6, 9)]
    elif event_kind == "state_check":
        if chunk_start is None or chunk_end is None:
            raise ValueError("chunk bounds are required for state sampling")
        candidates = [
            chunk_start - 1,
            chunk_start,
            chunk_start + 3,
            chunk_start + 7,
            chunk_start + 11,
            chunk_end,
        ]
    else:
        raise ValueError(f"Unknown event kind: {event_kind}")
    return sorted({min(max(value, 0), available_frames - 1) for value in candidates})


def pre_action_frame_indices(available_frames: int) -> list[int]:
    ''

    if available_frames <= 0:
        raise ValueError(
            "Pre-action prediction requires at least one causal history frame"
        )
    last = available_frames - 1
    return sorted({max(0, last - offset) for offset in (8, 4, 0)})


def summarize_pre_action_plan(
    action_joystick: Tensor,
    *,
    interval_start_frame: int,
    interval_end_frame: int,
    shot_frame: int,
) -> str:
    ''

    if action_joystick.ndim != 2 or action_joystick.shape[1] != 4:
        raise ValueError(
            f"Expected [frames, 4] joystick actions, got {tuple(action_joystick.shape)}"
        )
    if (
        not 0
        <= interval_start_frame
        <= shot_frame
        <= interval_end_frame
        < action_joystick.shape[0]
    ):
        raise ValueError(
            f"Invalid pre-action frame interval [{interval_start_frame}, {interval_end_frame}] "
            f"with shot={shot_frame} for {action_joystick.shape[0]} action frames"
        )
    before_shot = (
        action_joystick[interval_start_frame : shot_frame + 1].detach().float().cpu()
    )
    means = before_shot.mean(dim=0)
    max_abs = before_shot.abs().amax(dim=0)
    camera_steady = bool(max_abs[2:].max().item() <= 1e-6)
    movement_steady = bool(max_abs[:2].max().item() <= 1e-6)
    relative_frame = shot_frame - interval_start_frame
    return (
        f"The fire pulse occurs {relative_frame} frames after the interval starts. "
        f"Before and through the pulse, mean joystick values are "
        f"movement_left_x={means[0].item():+.6f}, movement_left_y={means[1].item():+.6f}, "
        f"camera_right_x={means[2].item():+.6f}, camera_right_y={means[3].item():+.6f}; "
        f"maximum absolute values are movement_left_x={max_abs[0].item():.6f}, "
        f"movement_left_y={max_abs[1].item():.6f}, camera_right_x={max_abs[2].item():.6f}, "
        f"camera_right_y={max_abs[3].item():.6f}. "
        f"Camera aim is {'held exactly steady' if camera_steady else 'changed by the right stick'} "
        f"and player movement is {'zero' if movement_steady else 'nonzero'}. "
        "Only the right stick rotates aim: camera_right_x + turns right and - turns left; "
        "camera_right_y + looks down and - looks up. The left stick only translates the player "
        "and must never be interpreted as camera rotation."
    )


def pre_action_prompt_mode(
    transition: Any, *, min_confidence: float
) -> InteractionMode:
    ''

    if transition.counted_hit:
        return "collapse" if transition.hp_after == 0 else "hit_alive"
    observation = transition.observation
    if observation.shot_result == "miss" and observation.confidence >= min_confidence:
        return "miss_alive"
    return "active"


def video_frames_as_images(
    video: Tensor, frame_indices: list[int]
) -> list[Image.Image]:
    ''

    images: list[Image.Image] = []
    for frame_index in frame_indices:
        array = (
            video[:, frame_index]
            .detach()
            .float()
            .clamp(0, 1)
            .permute(1, 2, 0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        images.append(Image.fromarray(array, mode="RGB"))
    return images


def vlm_images_as_video_tensor(images: list[Image.Image]) -> Tensor:
    ''

    if not images:
        raise ValueError("VLM evidence images must be non-empty")
    arrays = [
        np.asarray(image.convert("RGB"), dtype=np.uint8).copy() for image in images
    ]
    return (
        torch.stack(
            [torch.from_numpy(array).permute(2, 0, 1) for array in arrays], dim=0
        )
        .float()
        .div_(255.0)
    )


def set_prompt_schedule_block(
    *,
    schedule: dict[str, Tensor],
    bank: EncodedPromptBank,
    prompt: str,
    block_index: int,
) -> None:
    video, audio, mask = bank[prompt]
    schedule["video_prompt_schedule_embeds"][0, block_index].copy_(video)
    schedule["audio_prompt_schedule_embeds"][0, block_index].copy_(audio)
    schedule["prompt_schedule_attention_mask"][0, block_index].copy_(mask)
    if block_index == 0:
        schedule["video_prompt_embeds"][0].copy_(video)
        schedule["audio_prompt_embeds"][0].copy_(audio)
        schedule["prompt_attention_mask"][0].copy_(mask)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_prompt_schedule_contract(path: Path) -> set[str]:
    ''

    prompts: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            schedule = row.get("prompt_schedule")
            if not isinstance(schedule, list) or not all(
                isinstance(prompt, str) and prompt for prompt in schedule
            ):
                raise ValueError(
                    f"Invalid prompt_schedule in contract manifest {path} "
                    f"at line {line_number}"
                )
            prompts.update(schedule)
    if not prompts:
        raise ValueError(f"Prompt contract manifest has no prompts: {path}")
    return prompts


def load_custom_condition_image(path: Path, *, height: int, width: int) -> Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (width, height):
            raise ValueError(
                f"Custom condition image must be exactly {width}x{height}, got {rgb.width}x{rgb.height}: {path}"
            )
        array = np.asarray(rgb, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def save_condition_image(path: Path, frame_chw: Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        frame_chw.detach()
        .cpu()
        .float()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    Image.fromarray(array).save(path)


def load_custom_action_array(
    path: Path,
    *,
    num_frames: int,
    feature_dim: int,
    name: str,
) -> Tensor:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != feature_dim:
        raise ValueError(
            f"{name} must have shape [F, {feature_dim}], got {array.shape}: {path}"
        )
    if array.shape[0] < num_frames:
        raise ValueError(
            f"{name} has {array.shape[0]} frames, need at least {num_frames}: {path}"
        )
    return torch.from_numpy(np.asarray(array[:num_frames], dtype=np.float32).copy())


def compute_temporal_metrics(pred: Tensor) -> dict[str, float]:
    video = pred.detach().float().clamp(0, 1)
    if video.shape[1] < 2:
        return {
            "adjacent_frame_l1_mean": 0.0,
            "ar_boundary_frame_l1_mean": 0.0,
            "ar_boundary_to_overall_ratio": 0.0,
        }
    frame_l1 = torch.mean(torch.abs(video[:, 1:] - video[:, :-1]), dim=(0, 2, 3))
    overall = float(frame_l1.mean().item())
    boundary_indices = torch.arange(
        16,
        frame_l1.shape[0],
        16,
        device=frame_l1.device,
    )
    boundary = (
        float(frame_l1.index_select(0, boundary_indices).mean().item())
        if boundary_indices.numel() > 0
        else 0.0
    )
    return {
        "adjacent_frame_l1_mean": overall,
        "ar_boundary_frame_l1_mean": boundary,
        "ar_boundary_to_overall_ratio": boundary / max(overall, 1e-12),
    }


def load_training_config(path: Path) -> tuple[dict[str, Any], LtxTrainerConfig]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return raw, LtxTrainerConfig(**raw)


def prompt_from_row(row: dict[str, Any]) -> str:
    for key in (
        "training_prompt_static",
        "prompt_text",
        "caption",
        "prompt",
        "source_prompt",
    ):
        value = row.get(key)
        if value:
            return str(value).strip()
    raise ValueError(
        f"Manifest row has no prompt: {row.get('sample_id') or row.get('clip_id')}"
    )


def action_row_scale_from_row(row: dict[str, Any]) -> float:
    explicit = row.get("action_row_scale")
    if explicit is not None:
        return float(explicit)
    raw_action_chunk_size = int(row.get("raw_action_chunk_size") or 0)
    standard_chunk_action_rows = int(row.get("standard_chunk_action_rows") or 0)
    if raw_action_chunk_size <= 0 or standard_chunk_action_rows <= 0:
        return 1.0
    scale = float(raw_action_chunk_size) / float(standard_chunk_action_rows)
    nearest_integer = round(scale)
    if nearest_integer > 0 and abs(scale - nearest_integer) < 0.02:
        return float(nearest_integer)
    return scale


def build_case(
    *,
    row: dict[str, Any],
    case_id: str,
    use_variable_schedule: bool,
    num_generated_blocks: int,
    seed: int,
) -> dict[str, Any]:
    static_prompt = prompt_from_row(row)
    raw_schedule = row.get("prompt_schedule") if use_variable_schedule else None
    if raw_schedule is None:
        schedule = [static_prompt] * num_generated_blocks
    else:
        if not isinstance(raw_schedule, list) or not all(
            isinstance(item, str) for item in raw_schedule
        ):
            raise TypeError(f"{case_id}: prompt_schedule must be a list of strings")
        if len(raw_schedule) != num_generated_blocks:
            raise ValueError(
                f"{case_id}: expected {num_generated_blocks} prompt schedule entries, got {len(raw_schedule)}"
            )
        schedule = [item.strip() for item in raw_schedule]

    is_custom_image_case = bool(row.get("condition_image_path"))
    required_paths = (
        ("condition_image_path", "action_joystick_path", "action_buttons_path")
        if is_custom_image_case
        else ("video_path", "audio_path", "action_path")
    )
    for key in required_paths:
        value = row.get(key)
        if not value or not Path(value).exists():
            raise FileNotFoundError(f"{case_id}: missing {key}: {value}")

    negative_prompt = str(row.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT).strip()
    negative_prompt_file = row.get("negative_prompt_file")
    if negative_prompt_file:
        negative_path = Path(str(negative_prompt_file))
        if not negative_path.exists():
            raise FileNotFoundError(
                f"{case_id}: missing negative_prompt_file: {negative_path}"
            )
        negative_prompt = negative_path.read_text(encoding="utf-8").strip()

    return {
        "case_id": str(row.get("eval_case_id") or case_id),
        "row": row,
        "prompt_schedule": schedule,
        "negative_prompt": negative_prompt,
        "seed": int(row.get("eval_seed") or seed),
        "is_variable_prompt": bool(use_variable_schedule),
        "is_custom_image_case": is_custom_image_case,
    }


EncodedPromptBank = dict[str, tuple[Tensor, Tensor, Tensor]]


@torch.inference_mode()
def encode_prompt_bank(
    *,
    sampler: ValidationSampler,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    offload_after_encode: bool = True,
) -> EncodedPromptBank:
    text_encoder = sampler._text_encoder
    embeddings_processor = sampler._embeddings_processor
    if text_encoder is None or embeddings_processor is None:
        raise RuntimeError(
            "AR evaluation requires the text encoder and embeddings processor"
        )

    prompt_cache: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
    text_encoder.to(device)
    embeddings_processor.to(device)
    unique_prompts = list(dict.fromkeys(prompts))
    for prompt in unique_prompts:
        hidden_states, attention_mask = text_encoder.encode(prompt)
        processed = embeddings_processor.process_hidden_states(
            hidden_states, attention_mask
        )
        video_encoding = processed.video_encoding.squeeze(0)
        audio_encoding = (
            processed.audio_encoding.squeeze(0)
            if processed.audio_encoding is not None
            else video_encoding
        )
        binary_mask = processed.attention_mask.squeeze(0).to(dtype=torch.bool)
        prompt_cache[prompt] = (video_encoding, audio_encoding, binary_mask)

    encoded = [prompt_cache[prompt] for prompt in unique_prompts]
    video = pad_sequence([item[0] for item in encoded], batch_first=True).to(
        dtype=dtype
    )
    audio = pad_sequence([item[1] for item in encoded], batch_first=True).to(
        dtype=dtype
    )
    mask = pad_sequence(
        [item[2] for item in encoded], batch_first=True, padding_value=False
    )
    bank = {
        prompt: (video[index], audio[index], mask[index])
        for index, prompt in enumerate(unique_prompts)
    }

    if offload_after_encode:
        text_encoder.model.to("cpu")
        embeddings_processor.to("cpu")
        torch.cuda.empty_cache()
    return bank


def materialize_prompt_schedule(
    bank: EncodedPromptBank,
    prompts: list[str],
) -> dict[str, Tensor]:
    if not prompts:
        raise ValueError("Prompt schedule cannot be empty")
    encoded = [bank[prompt] for prompt in prompts]
    video = torch.stack([item[0] for item in encoded])
    audio = torch.stack([item[1] for item in encoded])
    mask = torch.stack([item[2] for item in encoded])
    return {
        "video_prompt_schedule_embeds": video.unsqueeze(0),
        "audio_prompt_schedule_embeds": audio.unsqueeze(0),
        "prompt_schedule_attention_mask": mask.unsqueeze(0),
        "video_prompt_embeds": video[:1],
        "audio_prompt_embeds": audio[:1],
        "prompt_attention_mask": mask[:1],
    }


@torch.inference_mode()
def encode_prompt_schedules(
    *,
    sampler: ValidationSampler,
    positive_prompts: list[str],
    negative_prompt: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    offload_after_encode: bool = True,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    bank = encode_prompt_bank(
        sampler=sampler,
        prompts=positive_prompts + [negative_prompt],
        device=device,
        dtype=dtype,
        offload_after_encode=offload_after_encode,
    )

    return (
        materialize_prompt_schedule(bank, positive_prompts),
        materialize_prompt_schedule(bank, [negative_prompt] * len(positive_prompts)),
    )


def transformer_forward(
    transformer: torch.nn.Module,
    inputs: ModelInputs,
    *,
    action_button_meaning_fusion: str,
    perturbations: BatchedPerturbationConfig | None = None,
) -> tuple[Tensor, Tensor | None]:
    video_pred, audio_pred = transformer(
        video=inputs.video,
        audio=inputs.audio,
        perturbations=perturbations,
        action_joystick_tokens=inputs.action_joystick_tokens,
        action_joystick_condition_mask=inputs.action_joystick_condition_mask,
        action_button_state=inputs.action_button_state,
        action_button_values=inputs.action_button_values,
        action_button_condition_mask=inputs.action_button_condition_mask,
        action_button_meaning_embeds=inputs.action_button_meaning_embeds,
        action_button_meaning_fusion=action_button_meaning_fusion,
        action_button_temporal_alignment=inputs.action_button_temporal_alignment,
        action_button_prefix_tokens=inputs.action_button_prefix_tokens,
        action_button_frame_rate=inputs.action_button_frame_rate,
        action_button_query_latent_indices=inputs.action_button_query_latent_indices,
    )
    if video_pred is None:
        raise RuntimeError("AR transformer returned no video prediction")
    return video_pred, audio_pred


def build_stg_perturbations(
    *,
    video_params: MultiModalGuiderParams,
    audio_params: MultiModalGuiderParams,
    batch_size: int,
) -> BatchedPerturbationConfig | None:
    perturbations: list[Perturbation] = []
    if not math.isclose(video_params.stg_scale, 0.0):
        perturbations.append(
            Perturbation(
                type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                blocks=video_params.stg_blocks,
            )
        )
    if not math.isclose(audio_params.stg_scale, 0.0):
        perturbations.append(
            Perturbation(
                type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                blocks=audio_params.stg_blocks,
            )
        )
    if not perturbations:
        return None
    config = PerturbationConfig(perturbations)
    return BatchedPerturbationConfig([config for _ in range(batch_size)])


def build_modality_perturbations(
    *,
    video_params: MultiModalGuiderParams,
    audio_params: MultiModalGuiderParams,
    batch_size: int,
) -> BatchedPerturbationConfig | None:
    video_uses_modality_guidance = not math.isclose(video_params.modality_scale, 1.0)
    audio_uses_modality_guidance = not math.isclose(audio_params.modality_scale, 1.0)
    if not video_uses_modality_guidance and not audio_uses_modality_guidance:
        return None
    config = PerturbationConfig(
        [
            Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
            Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
        ]
    )
    return BatchedPerturbationConfig([config for _ in range(batch_size)])


def inject_current_block(
    inputs: ModelInputs,
    *,
    block_index: int,
    current_video: Tensor,
    current_audio: Tensor,
) -> ModelInputs:
    if (
        inputs.ar_video_block_token_ranges is None
        or inputs.ar_audio_block_token_ranges is None
    ):
        raise ValueError("AR block token ranges are missing")
    if inputs.audio is None:
        raise ValueError("TI2AV AR evaluation requires the audio modality")

    video_half = inputs.video.latent.shape[1] // 2
    video_start, video_end = inputs.ar_video_block_token_ranges[block_index]
    video_latent = inputs.video.latent.clone()
    video_latent[:, video_half + video_start : video_half + video_end] = current_video

    audio_half = inputs.audio.latent.shape[1] // 2
    audio_start, audio_end = inputs.ar_audio_block_token_ranges[block_index]
    audio_latent = inputs.audio.latent.clone()
    audio_latent[:, audio_half + audio_start : audio_half + audio_end] = current_audio

    return replace(
        inputs,
        video=replace(inputs.video, latent=video_latent),
        audio=replace(inputs.audio, latent=audio_latent),
    )


def current_block_x0(
    *,
    inputs: ModelInputs,
    video_pred: Tensor,
    audio_pred: Tensor | None,
    block_index: int,
) -> tuple[Tensor, Tensor]:
    if (
        inputs.ar_video_block_token_ranges is None
        or inputs.ar_audio_block_token_ranges is None
    ):
        raise ValueError("AR block token ranges are missing")
    if inputs.audio is None or audio_pred is None:
        raise RuntimeError("TI2AV AR evaluation requires an audio prediction")

    video_half = inputs.video.latent.shape[1] // 2
    video_start, video_end = inputs.ar_video_block_token_ranges[block_index]
    video_x0 = to_denoised(inputs.video.latent, video_pred, inputs.video.timesteps)
    current_video_x0 = video_x0[:, video_half + video_start : video_half + video_end]

    audio_half = inputs.audio.latent.shape[1] // 2
    audio_start, audio_end = inputs.ar_audio_block_token_ranges[block_index]
    audio_x0 = to_denoised(inputs.audio.latent, audio_pred, inputs.audio.timesteps)
    current_audio_x0 = audio_x0[:, audio_half + audio_start : audio_half + audio_end]
    return current_video_x0, current_audio_x0


def current_block_model_pass(
    *,
    transformer: torch.nn.Module,
    inputs: ModelInputs,
    block_index: int,
    action_button_meaning_fusion: str,
    perturbations: BatchedPerturbationConfig | None = None,
) -> tuple[Tensor, Tensor]:
    video_pred, audio_pred = transformer_forward(
        transformer,
        inputs,
        action_button_meaning_fusion=action_button_meaning_fusion,
        perturbations=perturbations,
    )
    return current_block_x0(
        inputs=inputs,
        video_pred=video_pred,
        audio_pred=audio_pred,
        block_index=block_index,
    )


def sliding_window_token_indices(
    block_ranges: tuple[tuple[int, int], ...],
    *,
    total_tokens: int,
    block_index: int,
    history_blocks: int,
    device: torch.device,
) -> tuple[Tensor, slice, tuple[int, ...]]:
    ''

    if history_blocks < 0:
        raise ValueError("history_blocks must be non-negative")
    if not 1 <= block_index < len(block_ranges):
        raise ValueError(
            f"block_index must be in [1, {len(block_ranges) - 1}], got {block_index}"
        )
    if total_tokens != block_ranges[-1][1]:
        raise ValueError(
            f"total_tokens={total_tokens} disagrees with block ranges ending at {block_ranges[-1][1]}"
        )

    first_history_block = max(1, block_index - history_blocks)
    retained_clean_blocks = (0, *range(first_history_block, block_index))
    clean_parts = [
        torch.arange(block_ranges[index][0], block_ranges[index][1], device=device)
        for index in retained_clean_blocks
    ]
    clean_indices = torch.cat(clean_parts)
    noisy_start, noisy_end = block_ranges[block_index]
    noisy_indices = torch.arange(
        total_tokens + noisy_start,
        total_tokens + noisy_end,
        device=device,
    )
    indices = torch.cat([clean_indices, noisy_indices])
    current = slice(clean_indices.numel(), indices.numel())
    return indices, current, retained_clean_blocks


def _index_dense_mask(
    mask: Tensor, query_indices: Tensor, key_indices: Tensor
) -> Tensor:
    if mask.ndim not in (2, 3):
        raise ValueError(
            f"Expected a 2D or 3D dense attention mask, got {tuple(mask.shape)}"
        )
    return mask.index_select(-2, query_indices).index_select(-1, key_indices)


def _crop_sliding_modality(
    modality: Any,
    *,
    query_indices: Tensor,
    cross_key_indices: Tensor,
) -> Any:
    if (
        modality.attention_mask is None
        or modality.self_attention_block_mask is not None
    ):
        raise ValueError(
            "Sliding-history inference requires dense structural masks; "
            "construct the AR strategy with mask_backend='dense'."
        )
    context_mask = modality.context_mask
    if context_mask is not None:
        context_mask = context_mask.index_select(1, query_indices)
    cross_timesteps = modality.cross_attention_timesteps
    if cross_timesteps is not None:
        cross_timesteps = cross_timesteps.index_select(1, query_indices)
    cross_mask = modality.cross_attention_mask
    if cross_mask is not None:
        cross_mask = _index_dense_mask(cross_mask, query_indices, cross_key_indices)
    return replace(
        modality,
        latent=modality.latent.index_select(1, query_indices),
        timesteps=modality.timesteps.index_select(1, query_indices),
        positions=modality.positions.index_select(2, query_indices),
        context_mask=context_mask,
        attention_mask=_index_dense_mask(
            modality.attention_mask,
            query_indices,
            query_indices,
        ),
        self_attention_block_mask=None,
        cross_attention_mask=cross_mask,
        cross_attention_timesteps=cross_timesteps,
    )


def crop_sliding_window_inputs(
    inputs: ModelInputs,
    *,
    block_index: int,
    history_blocks: int,
) -> SlidingWindowInputs:
    ''

    if (
        inputs.ar_video_block_token_ranges is None
        or inputs.ar_audio_block_token_ranges is None
    ):
        raise ValueError("AR block token ranges are missing")
    if inputs.audio is None:
        raise ValueError("TI2AV sliding-history inference requires audio")
    video_total = inputs.video.latent.shape[1] // 2
    audio_total = inputs.audio.latent.shape[1] // 2
    video_indices, video_current, retained_video_blocks = sliding_window_token_indices(
        inputs.ar_video_block_token_ranges,
        total_tokens=video_total,
        block_index=block_index,
        history_blocks=history_blocks,
        device=inputs.video.latent.device,
    )
    audio_indices, audio_current, retained_audio_blocks = sliding_window_token_indices(
        inputs.ar_audio_block_token_ranges,
        total_tokens=audio_total,
        block_index=block_index,
        history_blocks=history_blocks,
        device=inputs.audio.latent.device,
    )
    if retained_video_blocks != retained_audio_blocks:
        raise ValueError("Video/audio sliding windows selected different AR blocks")

    action_joystick = inputs.action_joystick_tokens
    if action_joystick is not None:
        if action_joystick.shape[1] != 2 * video_total:
            raise ValueError(
                f"Expected doubled joystick tokens ({2 * video_total}), got {action_joystick.shape[1]}"
            )
        action_joystick = action_joystick.index_select(1, video_indices)
    query_indices = inputs.action_button_query_latent_indices
    if query_indices is not None:
        if query_indices.ndim != 1 or query_indices.numel() != 2 * video_total:
            raise ValueError(
                f"Expected {2 * video_total} action query indices, got {tuple(query_indices.shape)}"
            )
        query_indices = query_indices.index_select(0, video_indices)

    cropped = replace(
        inputs,
        video=_crop_sliding_modality(
            inputs.video,
            query_indices=video_indices,
            cross_key_indices=audio_indices,
        ),
        audio=_crop_sliding_modality(
            inputs.audio,
            query_indices=audio_indices,
            cross_key_indices=video_indices,
        ),
        action_joystick_tokens=action_joystick,
        action_button_query_latent_indices=query_indices,
    )
    return SlidingWindowInputs(
        inputs=cropped,
        video_current=video_current,
        audio_current=audio_current,
        retained_clean_blocks=retained_video_blocks,
    )


def sliding_window_model_pass(
    *,
    transformer: torch.nn.Module,
    inputs: ModelInputs,
    block_index: int,
    history_blocks: int,
    action_button_meaning_fusion: str,
    perturbations: BatchedPerturbationConfig | None = None,
) -> tuple[Tensor, Tensor]:
    cropped = crop_sliding_window_inputs(
        inputs,
        block_index=block_index,
        history_blocks=history_blocks,
    )
    video_pred, audio_pred = transformer_forward(
        transformer,
        cropped.inputs,
        action_button_meaning_fusion=action_button_meaning_fusion,
        perturbations=perturbations,
    )
    if cropped.inputs.audio is None or audio_pred is None:
        raise RuntimeError(
            "TI2AV sliding-history inference requires an audio prediction"
        )
    video_x0 = to_denoised(
        cropped.inputs.video.latent,
        video_pred,
        cropped.inputs.video.timesteps,
    )
    audio_x0 = to_denoised(
        cropped.inputs.audio.latent,
        audio_pred,
        cropped.inputs.audio.timesteps,
    )
    return video_x0[:, cropped.video_current], audio_x0[:, cropped.audio_current]


def apply_multimodal_guidance(
    *,
    video_params: MultiModalGuiderParams,
    audio_params: MultiModalGuiderParams,
    pass_outputs: dict[str, tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    cond_video, cond_audio = pass_outputs["cond"]
    uncond_video, uncond_audio = pass_outputs.get("uncond", (0.0, 0.0))
    perturbed_video, perturbed_audio = pass_outputs.get("ptb", (0.0, 0.0))
    modality_video, modality_audio = pass_outputs.get("mod", (0.0, 0.0))
    denoised_video = MultiModalGuider(params=video_params).calculate(
        cond_video,
        uncond_video,
        perturbed_video,
        modality_video,
    )
    denoised_audio = MultiModalGuider(params=audio_params).calculate(
        cond_audio,
        uncond_audio,
        perturbed_audio,
        modality_audio,
    )
    return denoised_video, denoised_audio


def compare_kv_cache_outputs(
    cached: tuple[Tensor, Tensor],
    reference: tuple[Tensor, Tensor],
) -> dict[str, dict[str, float]]:
    ''

    result: dict[str, dict[str, float]] = {}
    for name, cached_tensor, reference_tensor in zip(
        ("video", "audio"),
        cached,
        reference,
        strict=True,
    ):
        error = (cached_tensor.float() - reference_tensor.float()).abs()
        reference_abs = reference_tensor.float().abs()
        result[name] = {
            "max_abs": float(error.max().item()),
            "mean_abs": float(error.mean().item()),
            "reference_mean_abs": float(reference_abs.mean().item()),
            "relative_mean_abs": float(
                error.mean().div(reference_abs.mean().clamp(min=1e-8)).item()
            ),
        }
    return result


def compare_tensor_outputs(cached: Tensor, reference: Tensor) -> dict[str, float]:
    ''

    if cached.shape != reference.shape:
        raise ValueError(
            "KV tensor parity requires matching shapes: "
            f"cached={tuple(cached.shape)} reference={tuple(reference.shape)}"
        )
    error = (cached.float() - reference.float()).abs()
    reference_abs = reference.float().abs()
    return {
        "max_abs": float(error.max().item()),
        "mean_abs": float(error.mean().item()),
        "reference_mean_abs": float(reference_abs.mean().item()),
        "relative_mean_abs": float(
            error.mean().div(reference_abs.mean().clamp(min=1e-8)).item()
        ),
    }


def register_full_projected_kv_parity_hooks(
    *,
    transformer: torch.nn.Module,
    cache_bank: Any,
    video_total: int,
    audio_total: int,
    video_start: int,
    video_end: int,
    audio_start: int,
    audio_end: int,
    selected_layers: tuple[int, ...] = (
        0,
        1,
        7,
        15,
        23,
        31,
        35,
        36,
        37,
        38,
        39,
        42,
        47,
    ),
) -> tuple[list[Any], list[dict[str, Any]]]:
    ''

    records: list[dict[str, Any]] = []
    handles: list[Any] = []
    selected = set(selected_layers)

    def make_hook(
        *,
        layer_index: int,
        attention_name: str,
        projected_cache: Any,
        key_total: int,
        prefix_end: int,
        current_start: int,
        current_end: int,
    ):
        def hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            _output: Tensor,
        ) -> None:
            if projected_cache.k is None or projected_cache.v is None:
                raise RuntimeError(
                    f"Missing compact cache for layer={layer_index} attention={attention_name}"
                )
            if projected_cache.valid_end != current_end:
                raise RuntimeError(
                    "Unexpected compact cache extent before full-prefix comparison: "
                    f"layer={layer_index} attention={attention_name} "
                    f"valid_end={projected_cache.valid_end} expected={current_end}"
                )

            x = kwargs.get("x", args[0] if args else None)
            if x is None:
                raise RuntimeError("Attention hook could not resolve its query input")
            context = kwargs.get("context")
            if context is None:
                context = x

            reference_v = module.to_v(context)
            reference_k = module.k_norm(module.to_k(context))
            pe = kwargs.get("pe")
            if pe is not None:
                k_pe = kwargs.get("k_pe")
                reference_k = apply_rotary_emb(
                    reference_k,
                    pe if k_pe is None else k_pe,
                    module.rope_type,
                )

            noisy_start = key_total + current_start
            noisy_end = key_total + current_end
            if reference_k.shape[1] < noisy_end or reference_v.shape[1] < noisy_end:
                raise RuntimeError(
                    "Full-prefix K/V is shorter than the requested current-noisy slice: "
                    f"attention={attention_name} k={tuple(reference_k.shape)} "
                    f"v={tuple(reference_v.shape)} slice=[{noisy_start}:{noisy_end}]"
                )

            records.append(
                {
                    "layer": layer_index,
                    "attention": attention_name,
                    "cache_valid_end": int(projected_cache.valid_end),
                    "prefix_k": compare_tensor_outputs(
                        projected_cache.k[:, :prefix_end],
                        reference_k[:, :prefix_end],
                    ),
                    "prefix_v": compare_tensor_outputs(
                        projected_cache.v[:, :prefix_end],
                        reference_v[:, :prefix_end],
                    ),
                    "current_k": compare_tensor_outputs(
                        projected_cache.k[:, current_start:current_end],
                        reference_k[:, noisy_start:noisy_end],
                    ),
                    "current_v": compare_tensor_outputs(
                        projected_cache.v[:, current_start:current_end],
                        reference_v[:, noisy_start:noisy_end],
                    ),
                }
            )

        return hook

    for layer_index, (block, layer_cache) in enumerate(
        zip(transformer.transformer_blocks, cache_bank.layers, strict=True)
    ):
        if layer_index not in selected:
            continue
        specs = (
            (
                "video_self",
                block.attn1,
                layer_cache.video_self,
                video_total,
                video_start,
                video_start,
                video_end,
            ),
            (
                "audio_self",
                block.audio_attn1,
                layer_cache.audio_self,
                audio_total,
                audio_start,
                audio_start,
                audio_end,
            ),
            (
                "audio_to_video",
                block.audio_to_video_attn,
                layer_cache.audio_to_video,
                audio_total,
                audio_start,
                audio_start,
                audio_end,
            ),
            (
                "video_to_audio",
                block.video_to_audio_attn,
                layer_cache.video_to_audio,
                video_total,
                video_start,
                video_start,
                video_end,
            ),
        )
        for (
            attention_name,
            module,
            projected_cache,
            key_total,
            prefix_end,
            start,
            end,
        ) in specs:
            handles.append(
                module.register_forward_hook(
                    make_hook(
                        layer_index=layer_index,
                        attention_name=attention_name,
                        projected_cache=projected_cache,
                        key_total=key_total,
                        prefix_end=prefix_end,
                        current_start=start,
                        current_end=end,
                    ),
                    with_kwargs=True,
                )
            )
    return handles, records


class CurrentSublayerParityRecorder:
    ''

    def __init__(
        self,
        *,
        transformer: torch.nn.Module,
        video_total: int,
        audio_total: int,
        video_start: int,
        video_end: int,
        audio_start: int,
        audio_end: int,
        selected_layers: tuple[int, ...] = (0, 1, 35, 36, 37, 38, 39, 47),
    ) -> None:
        self.phase: str | None = None
        self.compact_outputs: dict[tuple[int, str], Tensor] = {}
        self.records: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        self._ranges = {
            "video": (video_total, video_start, video_end),
            "audio": (audio_total, audio_start, audio_end),
        }

        selected = set(selected_layers)
        for layer_index, block in enumerate(transformer.transformer_blocks):
            if layer_index not in selected:
                continue
            specs = (
                ("video_self_q", block.attn1.q_norm, "video"),
                ("video_self_out", block.attn1.to_out, "video"),
                ("video_text_out", block.attn2.to_out, "video"),
                ("video_joy_scale", block.action_joy_scale_layer, "video"),
                ("video_joy_shift", block.action_joy_shift_layer, "video"),
                ("video_button_out", block.action_button_attn, "video"),
                ("audio_self_q", block.audio_attn1.q_norm, "audio"),
                ("audio_self_out", block.audio_attn1.to_out, "audio"),
                ("audio_text_out", block.audio_attn2.to_out, "audio"),
                ("audio_to_video_out", block.audio_to_video_attn.to_out, "video"),
                ("video_to_audio_out", block.video_to_audio_attn.to_out, "audio"),
                ("video_ff_out", block.ff, "video"),
                ("audio_ff_out", block.audio_ff, "audio"),
            )
            for name, module, modality in specs:
                self.handles.append(
                    module.register_forward_hook(
                        self._make_hook(
                            layer_index=layer_index,
                            name=name,
                            modality=modality,
                        )
                    )
                )

    def _make_hook(self, *, layer_index: int, name: str, modality: str):
        key = (layer_index, name)

        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            if self.phase is None:
                return
            if not isinstance(output, Tensor):
                raise RuntimeError(
                    f"Expected tensor output for sublayer parity {key}, got {type(output).__name__}"
                )
            total, start, end = self._ranges[modality]
            if self.phase == "compact":
                expected = end - start
                if output.shape[1] != expected:
                    raise RuntimeError(
                        f"Compact sublayer output {key} has {output.shape[1]} tokens, expected {expected}"
                    )
                if key in self.compact_outputs:
                    raise RuntimeError(f"Duplicate compact sublayer output for {key}")
                self.compact_outputs[key] = output.detach().clone()
                return
            if self.phase != "full":
                raise RuntimeError(f"Unknown sublayer parity phase {self.phase!r}")
            if key not in self.compact_outputs:
                raise RuntimeError(f"Missing compact sublayer output for {key}")
            noisy_start = total + start
            noisy_end = total + end
            if output.shape[1] < noisy_end:
                raise RuntimeError(
                    f"Full sublayer output {key} has {output.shape[1]} tokens, "
                    f"cannot take [{noisy_start}:{noisy_end}]"
                )
            compact = self.compact_outputs.pop(key)
            self.records.append(
                {
                    "layer": layer_index,
                    "sublayer": name,
                    "modality": modality,
                    **compare_tensor_outputs(compact, output[:, noisy_start:noisy_end]),
                }
            )

        return hook

    def set_phase(self, phase: str | None) -> None:
        if phase not in (None, "compact", "full"):
            raise ValueError(f"Unsupported sublayer parity phase: {phase}")
        self.phase = phase

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def assert_complete(self) -> None:
        if self.compact_outputs:
            missing = sorted(self.compact_outputs)
            raise RuntimeError(
                f"Full sublayer parity did not consume compact outputs: {missing}"
            )


def required_guidance_passes(
    video_params: MultiModalGuiderParams,
    audio_params: MultiModalGuiderParams,
) -> list[str]:
    video_guider = MultiModalGuider(params=video_params)
    audio_guider = MultiModalGuider(params=audio_params)
    passes = ["cond"]
    if (
        video_guider.do_unconditional_generation()
        or audio_guider.do_unconditional_generation()
    ):
        passes.append("uncond")
    if video_guider.do_perturbed_generation() or audio_guider.do_perturbed_generation():
        passes.append("ptb")
    if (
        video_guider.do_isolated_modality_generation()
        or audio_guider.do_isolated_modality_generation()
    ):
        passes.append("mod")
    return passes


def resolve_inference_settings(
    *,
    training_config: LtxTrainerConfig,
    base_model: Path,
    num_inference_steps_override: int | None,
) -> tuple[int, MultiModalGuiderParams, MultiModalGuiderParams]:
    num_inference_steps = (
        num_inference_steps_override
        if num_inference_steps_override is not None
        else training_config.validation.resolved_inference_steps(base_model)
    )
    video_params, audio_params = training_config.validation.resolved_guider_params(
        base_model
    )
    if video_params.skip_step != 0 or audio_params.skip_step != 0:
        raise ValueError(
            "AR full-prefix evaluation currently requires video/audio skip_step=0; "
            f"got video={video_params.skip_step}, audio={audio_params.skip_step}"
        )
    return num_inference_steps, video_params, audio_params


def validate_denoising_sigmas(values: list[float] | None) -> tuple[float, ...] | None:
    ''
    if values is None:
        return None
    sigmas = tuple(float(value) for value in values)
    if len(sigmas) < 2:
        raise ValueError("--denoising-sigmas requires at least two endpoints")
    if not all(math.isfinite(value) for value in sigmas):
        raise ValueError("--denoising-sigmas must contain only finite values")
    if sigmas[0] <= 0.0:
        raise ValueError("--denoising-sigmas must start above zero")
    if sigmas[-1] != 0.0:
        raise ValueError("--denoising-sigmas must end at exactly 0.0")
    if any(left <= right for left, right in zip(sigmas, sigmas[1:])):
        raise ValueError("--denoising-sigmas must be strictly descending")
    return sigmas


def classify_xonotic_ar_geometry(
    *,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    allow_bootstrap_prefix_h4: bool,
    allow_experimental_long_horizon: bool,
) -> str:
    ''
    fixed_spatial_temporal = fps == 20 and (width, height) == (832, 480)
    if fixed_spatial_temporal and num_frames == 81:
        return "formal_81f"
    if fixed_spatial_temporal and num_frames == 65 and allow_bootstrap_prefix_h4:
        return "bootstrap_prefix_h4_65f"
    if (
        fixed_spatial_temporal
        and allow_experimental_long_horizon
        and 81 < num_frames <= 609
        and (num_frames - 1) % 16 == 0
    ):
        return "experimental_long_horizon"
    raise ValueError(
        "Formal Xonotic AR inference is fixed to 832x480 at 81 frames / 20 FPS. "
        "The exact sink+H1-H4 65-frame diagnostic requires "
        "--allow-bootstrap-prefix-h4. Longer 16k+1-frame rollouts require "
        "--allow-experimental-long-horizon and remain capped at 609 frames."
    )


def make_strategy(
    training_config: LtxTrainerConfig,
    *,
    mask_backend_override: str | None = None,
) -> ARTeacherStrategy:
    config_payload = training_config.training_strategy.model_dump()
    config_payload.update(
        {
            "cfg_training_enabled": False,
            "conditioning_dropout_p": 0.0,
            "joint_uncond_prob": 0.0,
            "action_dropout_prob": 0.0,
            "action_button_uncond_prob": 0.0,
            "action_joystick_uncond_prob": 0.0,
        }
    )
    if mask_backend_override is not None:
        config_payload["mask_backend"] = mask_backend_override
    strategy = ARTeacherStrategy(ARTeacherConfig(**config_payload))
    strategy.set_action_binding_config(training_config.action_binding)
    return strategy


def build_strategy_batch(
    *,
    strategy: ARTeacherStrategy,
    video_tokens: Tensor,
    audio_tokens: Tensor,
    video_shape: VideoLatentShape,
    audio_shape: AudioLatentShape,
    fps: float,
    action_joystick: Tensor,
    action_buttons: Tensor,
    prompt_schedule: dict[str, Tensor],
) -> dict[str, Any]:
    video_latents = strategy._video_patchifier.unpatchify(
        video_tokens, output_shape=video_shape
    )
    audio_latents = strategy._audio_patchifier.unpatchify(
        audio_tokens, output_shape=audio_shape
    )
    batch: dict[str, Any] = {
        "video_latents": video_latents,
        "audio_latents": audio_latents,
        "fps": torch.tensor([fps], device=video_tokens.device, dtype=torch.float32),
        "has_audio": torch.ones(1, device=video_tokens.device, dtype=torch.bool),
        "audio_loss_eligible": torch.ones(
            1, device=video_tokens.device, dtype=torch.bool
        ),
        "action_joystick": action_joystick.unsqueeze(0).to(device=video_tokens.device),
        "action_buttons": action_buttons.unsqueeze(0).to(device=video_tokens.device),
    }
    batch.update(
        {
            key: value.to(device=video_tokens.device)
            for key, value in prompt_schedule.items()
        }
    )
    return batch


@torch.inference_mode()
def generate_case(
    *,
    case: dict[str, Any],
    sampler: ValidationSampler,
    strategy: ARTeacherStrategy,
    transformer: torch.nn.Module,
    training_config: LtxTrainerConfig,
    device: torch.device,
    dtype: torch.dtype,
    component_dtype: torch.dtype,
    num_inference_steps: int,
    video_guider_params: MultiModalGuiderParams,
    audio_guider_params: MultiModalGuiderParams,
    scheduler_timestep_shift: float,
    denoising_sigmas: tuple[float, ...] | None,
    guidance_mode: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    output_dir: Path,
    generated_audio_sample_rate: int,
    tiled_decoding: TiledDecodingConfig | bool | None,
    save_latent_cache: bool,
    use_kv_cache: bool = False,
    kv_cache_attention_backend: str = "native",
    kv_cache_full_prefix_interval: int = 0,
    verify_kv_cache_first_step: bool = False,
    verify_kv_cache_projected_kv: bool = False,
    verify_kv_cache_sublayers: bool = False,
    verify_kv_cache_cold_recompute: bool = False,
    history_blocks: int | None = None,
    vlm_two_hit_config: VLMTwoHitConfig | None = None,
    vlm_action_controller_config: VLMActionControllerConfig | None = None,
    denoising_update_mode: str = "euler",
    latent_only: bool = False,
    keep_text_encoder_on_device: bool = False,
    persistent_prompt_embedding_cache: EncodedPromptBank | None = None,
) -> dict[str, Any]:
    row = case["row"]
    case_dir = output_dir / case["case_id"]
    if not latent_only:
        case_dir.mkdir(parents=True, exist_ok=False)
    crop_start = 0
    source_fps = float(row.get("source_fps") or fps)
    video_source_fps = float(row.get("video_source_fps") or source_fps)
    video_offset_sec = float(row.get("video_offset_sec") or 0.0)
    audio_offset_sec = float(row.get("audio_offset_sec") or video_offset_sec)
    action_row_start = int(row.get("action_row_start") or 0)
    action_row_scale = action_row_scale_from_row(row)
    button_resampling = str(row.get("button_resampling") or "point")

    if case["is_custom_image_case"]:
        condition_image = load_custom_condition_image(
            Path(row["condition_image_path"]),
            height=height,
            width=width,
        )
        gt_video = condition_image.unsqueeze(0).expand(num_frames, -1, -1, -1).clone()
        gt_audio = None
        gt_audio_sample_rate = None
        action_joystick = load_custom_action_array(
            Path(row["action_joystick_path"]),
            num_frames=num_frames,
            feature_dim=4,
            name="action_joystick",
        )
        action_buttons = load_custom_action_array(
            Path(row["action_buttons_path"]),
            num_frames=num_frames,
            feature_dim=6,
            name="action_buttons",
        )
    else:
                                                                                 
                                                                                
                                                    
        loaded_video_frames = 1 if latent_only else num_frames
        gt_video = load_video_window(
            str(row["video_path"]),
            crop_start,
            loaded_video_frames,
            height,
            width,
            source_fps=source_fps,
            video_source_fps=video_source_fps,
            video_offset_sec=video_offset_sec,
        )
        if latent_only:
            gt_audio, gt_audio_sample_rate = None, None
        else:
            gt_audio, gt_audio_sample_rate = load_audio_window(
                str(row["audio_path"]),
                crop_start,
                num_frames,
                fps,
                audio_offset_sec=audio_offset_sec,
            )
        action_joystick, action_buttons = load_action_window(
            str(row["action_path"]),
            crop_start,
            num_frames,
            "real",
            action_row_start=action_row_start,
            action_row_scale=action_row_scale,
            button_resampling=button_resampling,
            button_schema="legacy",
            game_id=str(row.get("game") or "unknown"),
        )
        condition_image = gt_video[0]

    runtime_prompt_schedule = list(case["prompt_schedule"])
    interaction_prompt_schedule: list[dict[InteractionMode, str]] | None = None
    duel_lifecycle_prompt_schedule: list[dict[str, str]] | None = None
    duel_reentry_interaction_prompt_schedule: (
        list[dict[str, dict[InteractionMode, str]]] | None
    ) = None
    encoded_prompt_bank: EncodedPromptBank | None = None
    two_hit_rule: TwoConfirmedHitKillRule | None = None
    agent_trace: list[dict[str, Any]] = []
    observed_fire_frames: set[int] = set()
    forecasted_fire_frames: set[int] = set()
    pending_pre_action_events: dict[int, dict[str, Any]] = {}
    death_prompt_block_index: int | None = None
    action_fire_frames: list[int] = []
    fire_frames: list[int] = []
    vlm_weapon_context: str | None = None
    vlm_enemy_context: str | None = None
    vlm_opponent_distance: str | None = None
    duel_enemy_attack_blocks: set[int] = set()
    duel_respawn_delay_blocks: int | None = None
    duel_death_confirmed_block: int | None = None
    duel_generation = 0
    duel_player_hp: int | None = None
    duel_suppressed_fire_frames: set[int] = set()
    duel_last_strafe_direction: str | None = None
    duel_current_reentry_key = "reenter"
    duel_prompt_contract: set[str] | None = None
    duel_prompt_contract_path: Path | None = None
    if vlm_two_hit_config is not None:
        if vlm_two_hit_config.freeze_opponent_motion:
            runtime_prompt_schedule = [
                freeze_opponent_motion(prompt) for prompt in runtime_prompt_schedule
            ]
        if vlm_two_hit_config.duel_loop:
                                                                           
                                                                              
                                                                               
                                                       
            runtime_prompt_schedule = [
                set_player_killed_outcome(prompt, killed=False)
                for prompt in runtime_prompt_schedule
            ]
        try:
            action_fire_frames, fire_frames = resolve_vlm_decision_fire_frames(
                action_buttons,
                row.get("vlm_decision_fire_frames"),
            )
        except ValueError as exc:
            raise ValueError(f"{case['case_id']}: {exc}") from exc
        interaction_prompt_schedule = [
            build_interaction_prompts(prompt) for prompt in runtime_prompt_schedule
        ]
        if vlm_two_hit_config.duel_loop:
            duel_lifecycle_prompt_schedule = [
                build_duel_lifecycle_prompts(prompt)
                for prompt in runtime_prompt_schedule
            ]
            duel_reentry_interaction_prompt_schedule = [
                {
                    key: build_interaction_prompts(variants[key])
                    for key in ("reenter", "reenter_left", "reenter_right")
                }
                for variants in duel_lifecycle_prompt_schedule
            ]
            raw_enemy_attack_blocks = row.get("vlm_enemy_attack_blocks")
            if not isinstance(raw_enemy_attack_blocks, list) or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_enemy_attack_blocks
            ):
                raise ValueError(
                    f"{case['case_id']}: duel loop requires integer "
                    "vlm_enemy_attack_blocks"
                )
            duel_enemy_attack_blocks = set(raw_enemy_attack_blocks)
            if any(
                value < 0 or value >= len(runtime_prompt_schedule)
                for value in duel_enemy_attack_blocks
            ):
                raise ValueError(
                    f"{case['case_id']}: enemy attack block lies outside prompt schedule"
                )
            block_seconds = 16.0 / float(fps)
            duel_respawn_delay_blocks = max(
                1,
                int(math.ceil(vlm_two_hit_config.duel_respawn_seconds / block_seconds)),
            )
            duel_player_hp = vlm_two_hit_config.duel_player_initial_hp
            maximum_player_damage = (
                len(duel_enemy_attack_blocks)
                * vlm_two_hit_config.duel_enemy_shot_damage
            )
            if maximum_player_damage >= duel_player_hp:
                raise ValueError(
                    f"{case['case_id']}: this duel-loop pilot does not model the "
                    "first-person player's post-death/respawn state; configure "
                    "player HP above the maximum scheduled damage"
                )
            raw_contract_path = row.get("vlm_duel_prompt_contract_manifest_path")
            if not isinstance(raw_contract_path, str) or not raw_contract_path:
                raise ValueError(
                    f"{case['case_id']}: duel loop requires "
                    "vlm_duel_prompt_contract_manifest_path"
                )
            duel_prompt_contract_path = Path(raw_contract_path).resolve()
            if not duel_prompt_contract_path.is_file():
                raise FileNotFoundError(duel_prompt_contract_path)
            duel_prompt_contract = load_prompt_schedule_contract(
                duel_prompt_contract_path
            )
        runtime_prompt_schedule = [
            variants["active"] for variants in interaction_prompt_schedule
        ]
        if duel_prompt_contract is not None:
            for block_index, prompt in enumerate(runtime_prompt_schedule):
                if prompt not in duel_prompt_contract:
                    raise ValueError(
                        f"{case['case_id']}: initial runtime prompt block "
                        f"{block_index} is outside the exact Duel-v12 training "
                        "prompt contract"
                    )
        vlm_weapon_context = (
            str(row["vlm_weapon_context"]).strip()
            if row.get("vlm_weapon_context")
            else extract_prompt_section(runtime_prompt_schedule[0], "ACTIVE WEAPON")
        )
        vlm_enemy_context = (
            str(row["vlm_enemy_context"]).strip()
            if row.get("vlm_enemy_context")
            else extract_prompt_section(
                runtime_prompt_schedule[0], "TRACKED ENEMY IDENTITY"
            )
        )
        vlm_opponent_distance = str(row.get("opponent_distance") or "unspecified")
        two_hit_rule = TwoConfirmedHitKillRule(
            min_hit_confidence=vlm_two_hit_config.min_hit_confidence,
            initial_hp=vlm_two_hit_config.initial_hp,
            confirmed_hit_damage=vlm_two_hit_config.confirmed_hit_damage,
        )

    if case["is_custom_image_case"]:
        cache_condition_image_path = Path(row["condition_image_path"]).resolve()
        refiner_condition_image_path = Path(
            row.get("refiner_condition_image_path") or row["condition_image_path"]
        ).resolve()
    else:
        cache_condition_image_path = (case_dir / "condition_image.png").resolve()
        if not latent_only:
            save_condition_image(cache_condition_image_path, condition_image)
        refiner_condition_image_path = cache_condition_image_path

    if not latent_only:
        save_video(
            video_tensor=gt_video.permute(1, 0, 2, 3),
            output_path=case_dir / "gt.mp4",
            fps=fps,
            audio=gt_audio,
            audio_sample_rate=gt_audio_sample_rate,
        )
        save_video_with_action_overlay(
            video_tensor=gt_video.permute(1, 0, 2, 3),
            output_path=case_dir / "gt_actions.mp4",
            fps=fps,
            joystick=action_joystick,
            buttons=action_buttons,
            audio=gt_audio,
            audio_sample_rate=gt_audio_sample_rate,
        )
        save_action_plot(
            case_dir / "actions.png", action_joystick, action_buttons, case["case_id"]
        )

    required_prompts = runtime_prompt_schedule + [case["negative_prompt"]]
    if interaction_prompt_schedule is not None:
        required_prompts += [
            prompt
            for variants in interaction_prompt_schedule
            for prompt in variants.values()
        ]
    if duel_lifecycle_prompt_schedule is not None:
        required_prompts += [
            prompt
            for variants in duel_lifecycle_prompt_schedule
            for prompt in variants.values()
        ]
    if duel_reentry_interaction_prompt_schedule is not None:
        required_prompts += [
            prompt
            for direction_variants in duel_reentry_interaction_prompt_schedule
            for variants in direction_variants.values()
            for prompt in variants.values()
        ]
    if duel_prompt_contract is not None:
                                                                       
                                                                             
                                                                        
                                                        
                                               
        required_prompts = [
            prompt
            for prompt in required_prompts
            if prompt == case["negative_prompt"] or prompt in duel_prompt_contract
        ]
    required_prompts = list(dict.fromkeys(required_prompts))

    if persistent_prompt_embedding_cache is not None:
        missing_prompts = [
            prompt
            for prompt in required_prompts
            if prompt not in persistent_prompt_embedding_cache
        ]
        if missing_prompts:
            newly_encoded = encode_prompt_bank(
                sampler=sampler,
                prompts=missing_prompts,
                device=device,
                dtype=dtype,
                offload_after_encode=not keep_text_encoder_on_device,
            )
            persistent_prompt_embedding_cache.update(
                {
                    prompt: tuple(tensor.detach().cpu() for tensor in encoded)
                    for prompt, encoded in newly_encoded.items()
                }
            )
        encoded_prompt_bank = {
            prompt: tuple(
                tensor.to(device=device, non_blocking=True)
                for tensor in persistent_prompt_embedding_cache[prompt]
            )
            for prompt in required_prompts
        }
        positive_schedule = materialize_prompt_schedule(
            encoded_prompt_bank,
            runtime_prompt_schedule,
        )
        negative_schedule = materialize_prompt_schedule(
            encoded_prompt_bank,
            [case["negative_prompt"]] * len(runtime_prompt_schedule),
        )
    elif interaction_prompt_schedule is None:
        positive_schedule, negative_schedule = encode_prompt_schedules(
            sampler=sampler,
            positive_prompts=runtime_prompt_schedule,
            negative_prompt=case["negative_prompt"],
            device=device,
            dtype=dtype,
            offload_after_encode=not keep_text_encoder_on_device,
        )
    else:
        encoded_prompt_bank = encode_prompt_bank(
            sampler=sampler,
            prompts=required_prompts,
            device=device,
            dtype=dtype,
            offload_after_encode=not keep_text_encoder_on_device,
        )
        positive_schedule = materialize_prompt_schedule(
            encoded_prompt_bank,
            runtime_prompt_schedule,
        )
        negative_schedule = materialize_prompt_schedule(
            encoded_prompt_bank,
            [case["negative_prompt"]] * len(runtime_prompt_schedule),
        )

    def apply_runtime_prompt_block(*, prompt: str, block_index: int) -> None:
        ''

        if encoded_prompt_bank is None:
            raise RuntimeError("Runtime prompt routing requires an encoded prompt bank")
        if duel_prompt_contract is not None and prompt not in duel_prompt_contract:
            opponent_body = extract_prompt_section(
                prompt, "CURRENT OPPONENT MOTION AND OUTCOME"
            )
            raise ValueError(
                f"{case['case_id']}: runtime prompt block {block_index} is "
                "outside the exact Duel-v12 training contract; "
                f"opponent_body={opponent_body!r}"
            )
        if prompt not in encoded_prompt_bank:
            raise RuntimeError(
                f"{case['case_id']}: runtime prompt was not pre-encoded for "
                f"block {block_index}"
            )
        set_prompt_schedule_block(
            schedule=positive_schedule,
            bank=encoded_prompt_bank,
            prompt=prompt,
            block_index=block_index,
        )
        runtime_prompt_schedule[block_index] = prompt

    generation_config = GenerationConfig(
        prompt=runtime_prompt_schedule[0],
        negative_prompt=case["negative_prompt"],
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=fps,
        num_inference_steps=num_inference_steps,
        condition_image=condition_image,
        generate_audio=True,
        use_audio_modality=True,
        seed=case["seed"],
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        scheduler_timestep_shift=scheduler_timestep_shift,
        capture_intermediates=False,
        tiled_decoding=tiled_decoding,
    )
    video_tools = sampler._create_video_latent_tools(generation_config)
    audio_tools = sampler._create_audio_latent_tools(generation_config)
    video_state = video_tools.create_initial_state(device=device, dtype=component_dtype)
    video_state = sampler._apply_image_conditioning(
        video_state,
        condition_image,
        generation_config,
        device,
    )
    audio_state = audio_tools.create_initial_state(device=device, dtype=component_dtype)
    video_tokens = video_state.clean_latent.clone().to(dtype=dtype)
    audio_tokens = audio_state.clean_latent.clone().to(dtype=dtype)
    video_shape = video_tools.target_shape
    audio_shape = audio_tools.target_shape

    expected_raw_frames = (video_shape.frames - 1) * 8 + 1
    if action_joystick.shape[0] != expected_raw_frames:
        raise ValueError(
            f"{case['case_id']}: actions have {action_joystick.shape[0]} frames, expected {expected_raw_frames}"
        )

    if denoising_sigmas is None:
        sigmas = (
            LTX2Scheduler()
            .execute(
                steps=num_inference_steps,
                max_shift=scheduler_timestep_shift,
                base_shift=scheduler_timestep_shift,
            )
            .to(device=device, dtype=torch.float32)
        )
        sigma_schedule_source = "ltx2_scheduler"
    else:
        sigmas = torch.tensor(denoising_sigmas, device=device, dtype=torch.float32)
        sigma_schedule_source = "explicit"

    from ltx_trainer.training_strategies.ar_teacher_masks import compute_ti2av_blocks

    ar_blocks = compute_ti2av_blocks(
        video_shape.frames,
        audio_shape.frames,
        chunk_size=strategy.config.chunk_size,
        video_fps=float(fps),
        video_temporal_scale=strategy.config.video_temporal_scale,
        audio_latent_fps=strategy.config.audio_latent_fps,
    )
    if len(ar_blocks) - 1 != len(runtime_prompt_schedule):
        raise ValueError(
            f"{case['case_id']}: geometry has {len(ar_blocks) - 1} generated blocks but "
            f"{len(runtime_prompt_schedule)} prompts"
        )

    generator = torch.Generator(device=device).manual_seed(case["seed"])
    action_button_meaning_fusion = (
        training_config.action_binding.button_meaning_embedding.fusion
    )
    video_guider = MultiModalGuider(params=video_guider_params)
    audio_guider = MultiModalGuider(params=audio_guider_params)
    needs_unconditional_pass = (
        video_guider.do_unconditional_generation()
        or audio_guider.do_unconditional_generation()
    )
    stg_perturbations = build_stg_perturbations(
        video_params=video_guider_params,
        audio_params=audio_guider_params,
        batch_size=1,
    )
    modality_perturbations = build_modality_perturbations(
        video_params=video_guider_params,
        audio_params=audio_guider_params,
        batch_size=1,
    )
    transformer.to(device=device, dtype=dtype).eval()
    vlm_healthchecks: dict[int, tuple[OpenAIVLMObserver, float]] = {}
    for vlm_config in (vlm_two_hit_config, vlm_action_controller_config):
        if vlm_config is not None:
            vlm_healthchecks[id(vlm_config.observer)] = (
                vlm_config.observer,
                vlm_config.deferred_healthcheck_timeout_seconds,
            )
    for observer, timeout_seconds in vlm_healthchecks.values():
        wait_for_vlm_healthcheck(observer, timeout_seconds=timeout_seconds)
    compact_prompt_context = strategy.config.prompt_execution_mode == "compact_current"
    if compact_prompt_context and not use_kv_cache:
        raise ValueError(
            "compact_current prompt execution requires --use-kv-cache during evaluation"
        )
    if compact_prompt_context and kv_cache_full_prefix_interval > 0:
        raise ValueError(
            "compact_current/FA4 evaluation cannot run masked full-prefix anchor steps"
        )
    kv_runner: ARTeacherKVCacheRunner | None = None
    kv_cache_banks: dict[str, Any] = {}
    kv_cache_estimated_bytes = 0
    kv_parity: dict[str, Any] = {}
    if use_kv_cache:
        max_video_tokens = int(video_tokens.shape[1])
        max_audio_tokens = int(audio_tokens.shape[1])
        kv_runner = ARTeacherKVCacheRunner(
            transformer,
            max_video_tokens=max_video_tokens,
            max_audio_tokens=max_audio_tokens,
            require_mask_free_attention=compact_prompt_context,
            attention_backend=kv_cache_attention_backend,
        )
        guidance_passes = required_guidance_passes(
            video_guider_params, audio_guider_params
        )
        kv_cache_banks = {name: kv_runner.new_cache_bank() for name in guidance_passes}
        bytes_per_bank = estimate_cache_bank_bytes(
            transformer,
            max_video_tokens=max_video_tokens,
            max_audio_tokens=max_audio_tokens,
            dtype=dtype,
        )
        kv_cache_estimated_bytes = bytes_per_bank * len(guidance_passes)
        print(
            "[ar-eval] KV cache enabled: "
            f"branches={guidance_passes} attention_backend={kv_runner.attention_backend} "
            f"estimated_capacity_gib={kv_cache_estimated_bytes / 2**30:.2f}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    case_start = time.perf_counter()
    block_times: list[float] = []

    def decode_video_tokens(tokens: Tensor, config: GenerationConfig) -> Tensor:
        return sampler._decode_video_latent(
            tokens.to(dtype=component_dtype),
            config,
            device,
        )

    def run_current_block_pass(
        inputs: ModelInputs,
        *,
        block_index: int,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[Tensor, Tensor]:
        if history_blocks is None:
            return current_block_model_pass(
                transformer=transformer,
                inputs=inputs,
                block_index=block_index,
                action_button_meaning_fusion=action_button_meaning_fusion,
                perturbations=perturbations,
            )
        return sliding_window_model_pass(
            transformer=transformer,
            inputs=inputs,
            block_index=block_index,
            history_blocks=history_blocks,
            action_button_meaning_fusion=action_button_meaning_fusion,
            perturbations=perturbations,
        )

    def prepare_full_inputs(
        prompt_schedule: dict[str, Tensor],
        sigma: Tensor,
        *,
        block_index: int,
        current_video: Tensor | None = None,
        current_audio: Tensor | None = None,
    ) -> ModelInputs:
        batch = build_strategy_batch(
            strategy=strategy,
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            video_shape=video_shape,
            audio_shape=audio_shape,
            fps=float(fps),
            action_joystick=action_joystick,
            action_buttons=action_buttons,
            prompt_schedule=prompt_schedule,
        )
        prepared = strategy.prepare_training_inputs(batch, FixedSigmaSampler(sigma))
        if current_video is not None or current_audio is not None:
            if current_video is None or current_audio is None:
                raise ValueError("Current video/audio blocks must be supplied together")
            prepared = inject_current_block(
                prepared,
                block_index=block_index,
                current_video=current_video,
                current_audio=current_audio,
            )
        return prepared

    def prepare_inputs(
        prompt_schedule: dict[str, Tensor],
        sigma: Tensor,
        *,
        block_index: int,
        current_video: Tensor | None = None,
        current_audio: Tensor | None = None,
        noisy: bool,
    ) -> ModelInputs:
        prepared = prepare_full_inputs(
            prompt_schedule,
            sigma,
            block_index=block_index,
            current_video=current_video,
            current_audio=current_audio,
        )
        return slice_ar_model_inputs(
            prepared,
            block_index=block_index,
            noisy=noisy,
            compact_prompt_context=compact_prompt_context,
        )

    trace_path = case_dir / "agent_trace.jsonl"

    def record_agent_event(payload: dict[str, Any]) -> None:
        agent_trace.append(payload)
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def observe_navigation_and_plan(
        *,
        block: Any,
        video_token_end: int,
    ) -> tuple[FPSNavigationObservation, FPSBlockActionPlan, dict[str, Any]]:
        if vlm_action_controller_config is None:
            raise RuntimeError(
                "Navigation observation requested without an action-controller config"
            )
        prefix_num_frames = (block.video_end - 1) * 8 + 1
        prefix_config = replace(generation_config, num_frames=prefix_num_frames)
        decode_start = time.perf_counter()
        preview = decode_video_tokens(video_tokens[:, :video_token_end], prefix_config)
        decode_seconds = time.perf_counter() - decode_start
        last_frame = preview.shape[1] - 1
        frame_indices = list(dict.fromkeys((max(0, last_frame - 16), last_frame)))
        images = video_frames_as_images(preview, frame_indices)
        event_dir = case_dir / "vlm_action_frames" / f"after_block_{block.index:03d}"
        event_dir.mkdir(parents=True, exist_ok=False)
        frame_paths: list[str] = []
        for frame_index, image in zip(frame_indices, images, strict=True):
            frame_path = event_dir / f"frame_{frame_index:03d}.png"
            draw_center_crosshair(image).save(frame_path)
            frame_paths.append(str(frame_path.resolve()))

        vlm_start = time.perf_counter()
        observation = vlm_action_controller_config.observer.observe_navigation(
            images=images,
            frame_indices=frame_indices,
        )
        vlm_seconds = time.perf_counter() - vlm_start
        fallback_turn = "left" if block.index % 2 == 0 else "right"
        plan = plan_fps_block_actions(
            observation,
            move_magnitude=vlm_action_controller_config.move_magnitude,
            strafe_magnitude=vlm_action_controller_config.strafe_magnitude,
            turn_magnitude=vlm_action_controller_config.turn_magnitude,
            aim_magnitude=vlm_action_controller_config.aim_magnitude,
            vertical_aim_magnitude=(
                vlm_action_controller_config.vertical_aim_magnitude
            ),
            fallback_turn=fallback_turn,
        )
        metadata = {
            "observation": observation.to_dict(),
            "action_plan": plan.to_dict(),
            "frame_indices": frame_indices,
            "frame_paths": frame_paths,
            "prefix_decode_seconds": decode_seconds,
            "vlm_seconds": vlm_seconds,
        }
        print(
            f"[ar-vlm-action] {case['case_id']} after_block={block.index} "
            f"path={observation.path_ahead} move={plan.movement} "
            f"enemy={observation.enemy_side}/{observation.enemy_vertical} "
            f"fire={plan.fire}",
            flush=True,
        )
        return observation, plan, metadata

    def observe_generated_prefix(
        *,
        block: Any,
        video_token_end: int,
        event_id: str,
        event_kind: str,
        shot_frame: int | None = None,
        chunk_start: int | None = None,
        chunk_end: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if vlm_two_hit_config is None:
            raise RuntimeError("VLM observation requested without a VLM configuration")
        prefix_num_frames = (block.video_end - 1) * 8 + 1
        prefix_config = replace(generation_config, num_frames=prefix_num_frames)
        decode_start = time.perf_counter()
        preview = decode_video_tokens(video_tokens[:, :video_token_end], prefix_config)
        decode_seconds = time.perf_counter() - decode_start
        verified_fire_geometry = (
            vlm_two_hit_config.decision_mode == "pre_action_verified"
            and event_kind == "shot"
        )
        if verified_fire_geometry:
            if shot_frame is None:
                raise ValueError("Verified fire geometry requires an exact shot frame")
            last_frame = preview.shape[1] - 1
            frame_indices = [shot_frame]
            frame_indices.extend(
                frame
                for frame in (shot_frame + 1, shot_frame + 3, shot_frame + 6)
                if frame <= last_frame
            )
            frame_indices = list(dict.fromkeys(frame_indices))
        elif event_kind == "state_check":
                                                                            
                                                                            
                                                                              
                                                            
            frame_indices = [preview.shape[1] - 1]
        else:
            frame_indices = event_frame_indices(
                event_kind=event_kind,
                available_frames=preview.shape[1],
                shot_frame=shot_frame,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
            )
        decoded_images = video_frames_as_images(preview, frame_indices)
        images = [draw_center_crosshair(image) for image in decoded_images]
        event_dir = case_dir / "vlm_frames" / event_id
        event_dir.mkdir(parents=True, exist_ok=False)
        frame_paths: list[str] = []
        for frame_index, image in zip(frame_indices, images, strict=True):
            frame_path = event_dir / f"frame_{frame_index:03d}.png"
            image.save(frame_path)
            frame_paths.append(str(frame_path.resolve()))
        save_video(
            video_tensor=vlm_images_as_video_tensor(images),
            output_path=event_dir / "sampled_frames.mp4",
            fps=4,
        )
        direct_fire_crop_path = None
        direct_crop_size = vlm_two_hit_config.direct_crosshair_crop_size
        if verified_fire_geometry and direct_crop_size is not None:
            crop_prefix = (
                "aim_target_ring_v9_crop"
                if vlm_two_hit_config.aim_target_ring_v9
                else "direct_fire_crop"
            )
            direct_fire_crop_path = event_dir / (
                f"{crop_prefix}_{direct_crop_size}px_frame_{int(shot_frame):03d}.png"
            )
            focused_fire_image = center_crosshair_crop(
                decoded_images[0], crop_size=direct_crop_size
            )
            (
                draw_aim_center_ring(focused_fire_image)
                if vlm_two_hit_config.aim_target_ring_v9
                else draw_direct_crosshair(focused_fire_image)
            ).save(direct_fire_crop_path)
        vlm_start = time.perf_counter()
        aim_geometry = None
        aim_target_observation = None
        posture_observation = None
        if verified_fire_geometry:
            if vlm_two_hit_config.aim_target_ring_v9:
                if direct_crop_size is None:
                    raise RuntimeError("V9 aim-target observer requires a crop size")
                aim_target_observation = vlm_two_hit_config.observer.observe_aim_target(
                    image=decoded_images[0],
                    crop_size=direct_crop_size,
                )
                aim_observation = aim_target_observation.to_vlm_observation()
            else:
                geometry_images = (
                    [decoded_images[0]]
                    if direct_crop_size is not None
                    else decoded_images
                )
                geometry_indices = (
                    [int(shot_frame)] if direct_crop_size is not None else frame_indices
                )
                aim_observation, aim_geometry = (
                    vlm_two_hit_config.observer.observe_aim_geometry(
                        images=geometry_images,
                        frame_indices=geometry_indices,
                        shot_frame=int(shot_frame),
                        weapon_context=vlm_weapon_context,
                        enemy_context=vlm_enemy_context,
                        projected=False,
                        focus_crop_size=direct_crop_size,
                    )
                )
            if direct_crop_size is not None:
                posture_images = [decoded_images[-1]]
                posture_indices = [frame_indices[-1]]
                posture_observation = vlm_two_hit_config.observer.observe(
                    images=posture_images,
                    frame_indices=posture_indices,
                    event_kind="state_check",
                    weapon_context=vlm_weapon_context,
                    enemy_context=vlm_enemy_context,
                )
                observation = VLMObservation(
                    enemy_visible=posture_observation.enemy_visible,
                    shot_result=aim_observation.shot_result,
                    enemy_posture=posture_observation.enemy_posture,
                    confidence=min(
                        aim_observation.confidence,
                        posture_observation.confidence,
                    ),
                    evidence=(
                        f"{'Aim target' if vlm_two_hit_config.aim_target_ring_v9 else 'Aim geometry'}: "
                        f"{aim_observation.evidence} "
                        f"Post-shot posture: {posture_observation.evidence}"
                    ),
                    raw_response=json.dumps(
                        {
                            (
                                "aim_target_ring_v9"
                                if vlm_two_hit_config.aim_target_ring_v9
                                else "aim_geometry"
                            ): (
                                aim_target_observation.to_dict()
                                if aim_target_observation is not None
                                else aim_observation.to_dict()
                            ),
                            "post_shot_posture": posture_observation.to_dict(),
                        },
                        ensure_ascii=False,
                    ),
                )
            else:
                observation = aim_observation
        else:
            observation = vlm_two_hit_config.observer.observe(
                images=decoded_images,
                frame_indices=frame_indices,
                event_kind=event_kind,
                shot_frame=shot_frame,
                weapon_context=vlm_weapon_context,
                enemy_context=vlm_enemy_context,
            )
        vlm_seconds = time.perf_counter() - vlm_start
        metadata = {
            "prefix_num_frames": prefix_num_frames,
            "sampled_frames": frame_indices,
            "frame_paths": frame_paths,
            "sampled_video": str((event_dir / "sampled_frames.mp4").resolve()),
            "direct_fire_crop": (
                str(direct_fire_crop_path.resolve())
                if direct_fire_crop_path is not None
                else None
            ),
            "direct_crosshair_crop_size": direct_crop_size,
            "crosshair_overlay": (
                "vlm_only_lime_hollow_ring_v9"
                if vlm_two_hit_config.aim_target_ring_v9
                else "vlm_only_exact_image_center"
            ),
            "prefix_decode_seconds": decode_seconds,
            "vlm_seconds": vlm_seconds,
            "weapon_context": vlm_weapon_context,
            "enemy_context": vlm_enemy_context,
            "decision_method": (
                "actual_fire_frame_aim_target_ring_v9_plus_posture"
                if verified_fire_geometry and vlm_two_hit_config.aim_target_ring_v9
                else "actual_fire_frame_direct_crosshair_focus_v2_plus_posture"
                if verified_fire_geometry and direct_crop_size is not None
                else (
                    "actual_fire_frame_aim_geometry"
                    if verified_fire_geometry
                    else "generated_visual_event"
                )
            ),
            "aim_geometry": aim_geometry.to_dict()
            if aim_geometry is not None
            else None,
            "aim_target_observation": (
                aim_target_observation.to_dict()
                if aim_target_observation is not None
                else None
            ),
            "aim_target_prompt": (
                AIM_TARGET_RING_NEUTRAL_PROMPT
                if vlm_two_hit_config.aim_target_ring_v9
                else None
            ),
            "posture_observation": (
                posture_observation.to_dict()
                if posture_observation is not None
                else None
            ),
        }
        print(
            f"[ar-vlm] {case['case_id']} event={event_id} "
            f"result={observation.shot_result} posture={observation.enemy_posture} "
            f"confidence={observation.confidence:.2f}",
            flush=True,
        )
        return observation, metadata

    def predict_upcoming_shot(
        *,
        block: Any,
        video_token_start: int,
        event_id: str,
        shot_frame: int,
        interval_start_frame: int,
        interval_end_frame: int,
    ) -> tuple[Any, dict[str, Any]]:
        if vlm_two_hit_config is None:
            raise RuntimeError(
                "Pre-action VLM prediction requested without a VLM configuration"
            )
        prefix_num_frames = interval_start_frame
        if prefix_num_frames <= 0:
            raise ValueError(f"Invalid pre-action prefix length: {prefix_num_frames}")
        decode_start = time.perf_counter()
        if prefix_num_frames == 1:
            preview = condition_image.unsqueeze(1)
        else:
            prefix_config = replace(generation_config, num_frames=prefix_num_frames)
            preview = decode_video_tokens(
                video_tokens[:, :video_token_start], prefix_config
            )
        decode_seconds = time.perf_counter() - decode_start
        frame_indices = pre_action_frame_indices(preview.shape[1])
        decoded_images = video_frames_as_images(preview, frame_indices)
        images = [draw_center_crosshair(image) for image in decoded_images]
        action_plan = summarize_pre_action_plan(
            action_joystick,
            interval_start_frame=interval_start_frame,
            interval_end_frame=interval_end_frame,
            shot_frame=shot_frame,
        )
        lookup_projection = (
            vlm_two_hit_config.decision_mode == "pre_action_verified"
            and vlm_two_hit_config.joystick_pixel_lookup is not None
        )
        pixel_shift = None
        projected_fire_image = None
        if lookup_projection:
            joystick_frames = (
                action_joystick[interval_start_frame : shot_frame + 1, :4]
                .detach()
                .float()
                .cpu()
                .tolist()
            )
            pixel_shift = vlm_two_hit_config.joystick_pixel_lookup.estimate(
                joystick_frames
            )
            projected_fire_image = shift_image_for_camera_projection(
                decoded_images[-1],
                dx_pixels=pixel_shift.dx_pixels,
                dy_pixels=pixel_shift.dy_pixels,
            )
            multivariate_projection = (
                vlm_two_hit_config.joystick_pixel_lookup.linear_dx_coefficients
                is not None
            )
            action_plan += (
                " Empirical controller lookup projects a stationary target and scene from "
                f"causal frame {frame_indices[-1]} to fire frame {shot_frame} by "
                f"dx={pixel_shift.dx_pixels:+.2f} pixels and "
                f"dy={pixel_shift.dy_pixels:+.2f} pixels at {width}x{height}. "
                "Positive dx/dy means right/down. "
                + (
                    "This calibration includes all four joystick axes but still ignores "
                    "enemy self-motion."
                    if multivariate_projection
                    else "This legacy calibration ignores left-stick parallax and enemy self-motion."
                )
            )
        event_dir = case_dir / "vlm_frames" / event_id
        event_dir.mkdir(parents=True, exist_ok=False)
        frame_paths: list[str] = []
        for frame_index, image in zip(frame_indices, images, strict=True):
            frame_path = event_dir / f"frame_{frame_index:03d}.png"
            image.save(frame_path)
            frame_paths.append(str(frame_path.resolve()))
        save_video(
            video_tensor=vlm_images_as_video_tensor(images),
            output_path=event_dir / "sampled_frames.mp4",
            fps=4,
        )
        projected_frame_path = None
        direct_projected_crop_path = None
        vlm_input_frames_path = None
        direct_crop_size = vlm_two_hit_config.direct_crosshair_crop_size
        if projected_fire_image is not None:
            projected_frame_path = (
                event_dir / f"projected_fire_frame_{shot_frame:03d}.png"
            )
            draw_center_crosshair(projected_fire_image).save(projected_frame_path)
            if direct_crop_size is not None:
                focused_projected_image = center_crosshair_crop(
                    projected_fire_image, crop_size=direct_crop_size
                )
                direct_projected_crop = (
                    draw_aim_center_ring(focused_projected_image)
                    if vlm_two_hit_config.aim_target_ring_v9
                    else draw_direct_crosshair(focused_projected_image)
                )
                crop_prefix = (
                    "aim_target_ring_v9_projected_crop"
                    if vlm_two_hit_config.aim_target_ring_v9
                    else "direct_projected_fire_crop"
                )
                direct_projected_crop_path = event_dir / (
                    f"{crop_prefix}_{direct_crop_size}px_frame_{shot_frame:03d}.png"
                )
                direct_projected_crop.save(direct_projected_crop_path)
            vlm_input_frames_path = event_dir / "vlm_input_frames.mp4"
            save_video(
                video_tensor=vlm_images_as_video_tensor(
                    [direct_projected_crop]
                    if direct_crop_size is not None
                    else [*images, draw_center_crosshair(projected_fire_image)]
                ),
                output_path=vlm_input_frames_path,
                fps=4,
            )
        vlm_start = time.perf_counter()
        vote_observations = []
        vote_geometries: list[dict[str, Any]] = []
        for _ in range(vlm_two_hit_config.pre_action_votes):
            if lookup_projection:
                if projected_fire_image is None or pixel_shift is None:
                    raise RuntimeError("Missing projected fire image or pixel shift")
                if vlm_two_hit_config.aim_target_ring_v9:
                    if direct_crop_size is None:
                        raise RuntimeError(
                            "V9 aim-target observer requires a crop size"
                        )
                    aim_target = vlm_two_hit_config.observer.observe_aim_target(
                        image=projected_fire_image,
                        crop_size=direct_crop_size,
                    )
                    vote_observations.append(aim_target.to_vlm_observation())
                    vote_geometries.append(
                        {
                            "observer": "aim_target_ring_v9",
                            **aim_target.to_dict(),
                        }
                    )
                else:
                    projected_observation, geometry = (
                        vlm_two_hit_config.observer.observe_aim_geometry(
                            images=(
                                [projected_fire_image]
                                if direct_crop_size is not None
                                else [*decoded_images, projected_fire_image]
                            ),
                            frame_indices=(
                                [shot_frame]
                                if direct_crop_size is not None
                                else [*frame_indices, shot_frame]
                            ),
                            shot_frame=shot_frame,
                            weapon_context=vlm_weapon_context,
                            enemy_context=vlm_enemy_context,
                            projected=True,
                            projection_note=(
                                f"dx={pixel_shift.dx_pixels:+.2f}px, "
                                f"dy={pixel_shift.dy_pixels:+.2f}px, "
                                f"lookup={vlm_two_hit_config.joystick_pixel_lookup.calibration_id}"
                            ),
                            focus_crop_size=direct_crop_size,
                        )
                    )
                    vote_observations.append(projected_observation)
                    vote_geometries.append(geometry.to_dict())
            elif vlm_two_hit_config.pre_action_grounding:
                grounded_observation, geometry = (
                    vlm_two_hit_config.observer.predict_shot_grounded(
                        images=decoded_images,
                        frame_indices=frame_indices,
                        shot_frame=shot_frame,
                        weapon_context=vlm_weapon_context or "",
                        enemy_context=vlm_enemy_context or "",
                    )
                )
                vote_observations.append(grounded_observation)
                vote_geometries.append(geometry)
            else:
                vote_observations.append(
                    vlm_two_hit_config.observer.predict_shot(
                        images=decoded_images,
                        frame_indices=frame_indices,
                        shot_frame=shot_frame,
                        interval_start_frame=interval_start_frame,
                        interval_end_frame=interval_end_frame,
                        action_plan=action_plan,
                        weapon_context=vlm_weapon_context,
                        enemy_context=vlm_enemy_context,
                        opponent_distance=vlm_opponent_distance,
                    )
                )
        observation = strict_consensus_observation(vote_observations)
        vlm_seconds = time.perf_counter() - vlm_start
        metadata = {
            "decision_timing": "before_ar_block",
            "prefix_num_frames": prefix_num_frames,
            "sampled_frames": frame_indices,
            "frame_paths": frame_paths,
            "sampled_video": str((event_dir / "sampled_frames.mp4").resolve()),
            "vlm_input_frames": (
                str(vlm_input_frames_path.resolve())
                if vlm_input_frames_path is not None
                else str((event_dir / "sampled_frames.mp4").resolve())
            ),
            "vlm_input_frame_indices": (
                [shot_frame]
                if projected_fire_image is not None and direct_crop_size is not None
                else [*frame_indices, shot_frame]
                if projected_fire_image is not None
                else frame_indices
            ),
            "projected_fire_frame": (
                str(projected_frame_path.resolve())
                if projected_frame_path is not None
                else None
            ),
            "direct_projected_fire_crop": (
                str(direct_projected_crop_path.resolve())
                if direct_projected_crop_path is not None
                else None
            ),
            "direct_crosshair_crop_size": direct_crop_size,
            "crosshair_overlay": (
                "vlm_only_lime_hollow_ring_v9"
                if vlm_two_hit_config.aim_target_ring_v9
                else "vlm_only_exact_image_center"
            ),
            "vlm_vote_policy": (
                "single_vote"
                if vlm_two_hit_config.pre_action_votes == 1
                else "strict_unanimous"
            ),
            "vlm_votes": [value.to_dict() for value in vote_observations],
            "vlm_grounding_enabled": vlm_two_hit_config.pre_action_grounding,
            "vlm_grounding_votes": vote_geometries,
            "decision_method": (
                "lookup_projected_fire_frame_aim_target_ring_v9"
                if lookup_projection and vlm_two_hit_config.aim_target_ring_v9
                else "lookup_projected_fire_frame_direct_crosshair_focus_v2"
                if lookup_projection and direct_crop_size is not None
                else "lookup_projected_fire_frame_aim_geometry_v2_history_context"
                if lookup_projection
                else "legacy_pre_action"
            ),
            "joystick_pixel_lookup": (
                vlm_two_hit_config.joystick_pixel_lookup.calibration_id
                if lookup_projection
                else None
            ),
            "joystick_pixel_shift": (
                pixel_shift.to_dict() if pixel_shift is not None else None
            ),
            "action_plan": action_plan,
            "prefix_decode_seconds": decode_seconds,
            "vlm_seconds": vlm_seconds,
            "weapon_context": vlm_weapon_context,
            "enemy_context": vlm_enemy_context,
            "opponent_distance": vlm_opponent_distance,
            "aim_target_prompt": (
                AIM_TARGET_RING_NEUTRAL_PROMPT
                if vlm_two_hit_config.aim_target_ring_v9
                else None
            ),
        }
        print(
            f"[ar-vlm-pre] {case['case_id']} event={event_id} "
            f"prediction={observation.shot_result} confidence={observation.confidence:.2f}",
            flush=True,
        )
        return observation, metadata

    if kv_runner is not None:
        zero_sigma = torch.zeros((), device=device, dtype=torch.float32)
        positive_prefix = prepare_inputs(
            positive_schedule,
            zero_sigma,
            block_index=0,
            noisy=False,
        )
        kv_runner.forward(
            positive_prefix,
            cache=kv_cache_banks["cond"],
            video_start=0,
            audio_start=0,
            action_button_meaning_fusion=action_button_meaning_fusion,
        )
        if "ptb" in kv_cache_banks:
            kv_runner.forward(
                positive_prefix,
                cache=kv_cache_banks["ptb"],
                video_start=0,
                audio_start=0,
                action_button_meaning_fusion=action_button_meaning_fusion,
                perturbations=stg_perturbations,
            )
        if "mod" in kv_cache_banks:
            kv_runner.forward(
                positive_prefix,
                cache=kv_cache_banks["mod"],
                video_start=0,
                audio_start=0,
                action_button_meaning_fusion=action_button_meaning_fusion,
                perturbations=modality_perturbations,
            )
        if "uncond" in kv_cache_banks:
            negative_prefix = prepare_inputs(
                negative_schedule,
                zero_sigma,
                block_index=0,
                noisy=False,
            )
            kv_runner.forward(
                negative_prefix,
                cache=kv_cache_banks["uncond"],
                video_start=0,
                audio_start=0,
                action_button_meaning_fusion=action_button_meaning_fusion,
            )

    for block in ar_blocks[1:]:
        block_start = time.perf_counter()
        video_token_start = block.video_start * video_shape.height * video_shape.width
        video_token_end = block.video_end * video_shape.height * video_shape.width
        duel_phase = "active"
        duel_current_reentry_key = "reenter"
        if vlm_two_hit_config is not None and vlm_two_hit_config.duel_loop:
            if (
                two_hit_rule is None
                or encoded_prompt_bank is None
                or interaction_prompt_schedule is None
                or duel_lifecycle_prompt_schedule is None
                or duel_reentry_interaction_prompt_schedule is None
                or duel_respawn_delay_blocks is None
                or duel_player_hp is None
            ):
                raise RuntimeError("Incomplete VLM duel-loop runtime state")
            current_prompt_block = block.index - 1
            planned_active_prompt = interaction_prompt_schedule[current_prompt_block][
                "active"
            ]
            if "toward the player's left" in planned_active_prompt:
                planned_strafe_direction = "left"
            elif "toward the player's right" in planned_active_prompt:
                planned_strafe_direction = "right"
            else:
                planned_strafe_direction = None
            if duel_death_confirmed_block is None and planned_strafe_direction:
                duel_last_strafe_direction = planned_strafe_direction
            if duel_death_confirmed_block is not None:
                elapsed_dead_blocks = current_prompt_block - duel_death_confirmed_block
                if elapsed_dead_blocks < duel_respawn_delay_blocks:
                    duel_phase = "corpse"
                elif elapsed_dead_blocks == duel_respawn_delay_blocks:
                                                                           
                                                                                
                                                                                 
                    duel_phase = "respawn_offscreen"
                else:
                    duel_phase = "reenter"
                    if duel_last_strafe_direction in {"left", "right"}:
                        duel_current_reentry_key = (
                            f"reenter_{duel_last_strafe_direction}"
                        )
                    duel_generation += 1
                    two_hit_rule = TwoConfirmedHitKillRule(
                        min_hit_confidence=vlm_two_hit_config.min_hit_confidence,
                        initial_hp=vlm_two_hit_config.initial_hp,
                        confirmed_hit_damage=vlm_two_hit_config.confirmed_hit_damage,
                    )
                    duel_death_confirmed_block = None
                    death_prompt_block_index = None

                if duel_phase == "reenter":
                    lifecycle_prompt = duel_reentry_interaction_prompt_schedule[
                        current_prompt_block
                    ][duel_current_reentry_key]["active"]
                else:
                    lifecycle_prompt = duel_lifecycle_prompt_schedule[
                        current_prompt_block
                    ][duel_phase]
                apply_runtime_prompt_block(
                    prompt=lifecycle_prompt,
                    block_index=current_prompt_block,
                )

            interval_start_frame = current_prompt_block * 16 + 1
            interval_end_frame = (current_prompt_block + 1) * 16
            if duel_phase in {"corpse", "respawn_offscreen", "reenter"}:
                suppressed = [
                    frame
                    for frame in fire_frames
                    if interval_start_frame <= frame <= interval_end_frame
                ]
                action_buttons[interval_start_frame : interval_end_frame + 1, 0] = 0.0
                duel_suppressed_fire_frames.update(suppressed)
            if (
                duel_phase not in {"corpse", "respawn_offscreen"}
                and current_prompt_block in duel_enemy_attack_blocks
            ):
                hp_before = duel_player_hp
                duel_player_hp = max(
                    0,
                    duel_player_hp - vlm_two_hit_config.duel_enemy_shot_damage,
                )
                record_agent_event(
                    {
                        "event_id": f"enemy_attack_block_{current_prompt_block:03d}",
                        "before_ar_block": block.index,
                        "event_kind": "enemy_attack",
                        "duel_generation": duel_generation,
                        "duel_phase": duel_phase,
                        "player_hp_before": hp_before,
                        "player_hp_after": duel_player_hp,
                        "rendered_into_video_prompt": False,
                    }
                )
            if duel_phase != "active":
                record_agent_event(
                    {
                        "event_id": (
                            f"duel_lifecycle_block_{current_prompt_block:03d}"
                        ),
                        "before_ar_block": block.index,
                        "event_kind": "duel_lifecycle",
                        "duel_generation": duel_generation,
                        "duel_phase": duel_phase,
                        "death_confirmed_prompt_block": duel_death_confirmed_block,
                        "respawn_delay_blocks": duel_respawn_delay_blocks,
                        "player_fire_frames_suppressed": suppressed
                        if duel_phase in {"corpse", "respawn_offscreen", "reenter"}
                        else [],
                    }
                )
        if vlm_two_hit_config is not None and vlm_two_hit_config.decision_mode in {
            "pre_action",
            "pre_action_verified",
        }:
            if (
                two_hit_rule is None
                or interaction_prompt_schedule is None
                or encoded_prompt_bank is None
            ):
                raise RuntimeError("Incomplete pre-action VLM two-hit runtime state")
            current_prompt_block = block.index - 1
            interval_start_frame = current_prompt_block * 16 + 1
            interval_end_frame = (current_prompt_block + 1) * 16
            upcoming_fire_frames = [
                frame
                for frame in fire_frames
                if interval_start_frame <= frame <= interval_end_frame
                and frame not in duel_suppressed_fire_frames
                and frame
                not in (
                    forecasted_fire_frames
                    if vlm_two_hit_config.decision_mode == "pre_action_verified"
                    else observed_fire_frames
                )
            ]
            if (
                vlm_two_hit_config.decision_mode == "pre_action_verified"
                and two_hit_rule.hp == 0
            ):
                upcoming_fire_frames = []
            if len(upcoming_fire_frames) > 1:
                raise ValueError(
                    "The pre-action two-hit pilot requires at most one fire pulse per AR block; "
                    f"block={block.index} frames={upcoming_fire_frames}"
                )
            for fire_frame in upcoming_fire_frames:
                shot_ordinal = fire_frames.index(fire_frame) + 1
                event_id = f"pre_action_shot_{shot_ordinal}_frame_{fire_frame:03d}"
                observation, observation_metadata = predict_upcoming_shot(
                    block=block,
                    video_token_start=video_token_start,
                    event_id=event_id,
                    shot_frame=fire_frame,
                    interval_start_frame=interval_start_frame,
                    interval_end_frame=interval_end_frame,
                )
                verified_mode = (
                    vlm_two_hit_config.decision_mode == "pre_action_verified"
                )
                transition = (
                    two_hit_rule.preview_shot(event_id, observation)
                    if verified_mode
                    else two_hit_rule.observe_shot(event_id, observation)
                )
                if verified_mode:
                    forecasted_fire_frames.add(fire_frame)
                else:
                    observed_fire_frames.add(fire_frame)
                applied_mode = pre_action_prompt_mode(
                    transition,
                    min_confidence=vlm_two_hit_config.min_hit_confidence,
                )
                current_variants = (
                    duel_reentry_interaction_prompt_schedule[current_prompt_block][
                        duel_current_reentry_key
                    ]
                    if duel_phase == "reenter"
                    and duel_reentry_interaction_prompt_schedule is not None
                    else interaction_prompt_schedule[current_prompt_block]
                )
                current_prompt = current_variants[applied_mode]
                apply_runtime_prompt_block(
                    prompt=current_prompt,
                    block_index=current_prompt_block,
                )
                future_mode: InteractionMode | None = None
                if transition.hp_after == 0:
                    if verified_mode:
                                                                               
                                                                             
                                                                       
                                                                             
                                                                    
                        death_prompt_block_index = (
                            None
                            if vlm_two_hit_config.duel_loop
                            else current_prompt_block
                        )
                    else:
                        future_mode = "persist_killed"
                        for future_prompt_block in range(
                            current_prompt_block + 1,
                            len(runtime_prompt_schedule),
                        ):
                            future_prompt = interaction_prompt_schedule[
                                future_prompt_block
                            ][future_mode]
                            apply_runtime_prompt_block(
                                prompt=future_prompt,
                                block_index=future_prompt_block,
                            )
                if verified_mode:
                    pending_pre_action_events[fire_frame] = {
                        "forecast_event_id": event_id,
                        "block_index": block.index,
                        "prompt_block": current_prompt_block,
                        "applied_mode": applied_mode,
                        "forecast_transition": transition,
                    }
                record_agent_event(
                    {
                        "event_id": event_id,
                        "before_ar_block": block.index,
                        "current_prompt_block": current_prompt_block,
                        "shot_frame": fire_frame,
                        "transition": transition.to_dict(),
                        "applied_prompt_block": current_prompt_block,
                        "applied_prompt_mode": applied_mode,
                        "future_prompt_mode": future_mode,
                        "provisional_until_visual_audit": verified_mode,
                        **observation_metadata,
                    }
                )
        current_video = torch.randn(
            (1, video_token_end - video_token_start, video_tokens.shape[-1]),
            generator=generator,
            device=device,
            dtype=component_dtype,
        ).to(dtype=dtype)
        current_audio = torch.randn(
            (1, block.audio_end - block.audio_start, audio_tokens.shape[-1]),
            generator=generator,
            device=device,
            dtype=component_dtype,
        ).to(dtype=dtype)

        for step_index, sigma in enumerate(sigmas[:-1]):
            full_prefix_anchor_step = (
                kv_runner is not None
                and kv_cache_full_prefix_interval > 0
                and step_index % kv_cache_full_prefix_interval == 0
            )
            positive_batch = build_strategy_batch(
                strategy=strategy,
                video_tokens=video_tokens,
                audio_tokens=audio_tokens,
                video_shape=video_shape,
                audio_shape=audio_shape,
                fps=float(fps),
                action_joystick=action_joystick,
                action_buttons=action_buttons,
                prompt_schedule=positive_schedule,
            )
            positive_inputs = strategy.prepare_training_inputs(
                positive_batch,
                FixedSigmaSampler(sigma),
            )
            positive_inputs = inject_current_block(
                positive_inputs,
                block_index=block.index,
                current_video=current_video,
                current_audio=current_audio,
            )
            if kv_runner is None or full_prefix_anchor_step:
                cond_output = run_current_block_pass(
                    positive_inputs,
                    block_index=block.index,
                )
            else:
                positive_current_inputs = slice_ar_model_inputs(
                    positive_inputs,
                    block_index=block.index,
                    noisy=True,
                    compact_prompt_context=compact_prompt_context,
                )
                capture_cond_layers = (
                    verify_kv_cache_first_step and block.index == 1 and step_index == 0
                )
                cached_layer_states: list[tuple[Tensor, Tensor]] = []
                video_total = positive_inputs.video.latent.shape[1] // 2
                if positive_inputs.audio is None:
                    raise RuntimeError("TI2AV layer parity requires audio")
                audio_total = positive_inputs.audio.latent.shape[1] // 2
                sublayer_recorder: CurrentSublayerParityRecorder | None = None
                if capture_cond_layers and verify_kv_cache_sublayers:
                    sublayer_recorder = CurrentSublayerParityRecorder(
                        transformer=transformer,
                        video_total=video_total,
                        audio_total=audio_total,
                        video_start=video_token_start,
                        video_end=video_token_end,
                        audio_start=block.audio_start,
                        audio_end=block.audio_end,
                    )
                    sublayer_recorder.set_phase("compact")

                def capture_cached_layer(
                    _index: int, video_args: Any, audio_args: Any
                ) -> None:
                    if video_args is None or audio_args is None:
                        raise RuntimeError(
                            "TI2AV layer parity requires both modalities"
                        )
                    cached_layer_states.append(
                        (
                            video_args.x.detach().clone(),
                            audio_args.x.detach().clone(),
                        )
                    )

                try:
                    cond_output = kv_runner.predict_x0(
                        positive_current_inputs,
                        cache=kv_cache_banks["cond"],
                        video_start=video_token_start,
                        audio_start=block.audio_start,
                        action_button_meaning_fusion=action_button_meaning_fusion,
                        layer_state_callback=capture_cached_layer
                        if capture_cond_layers
                        else None,
                    )
                finally:
                    if sublayer_recorder is not None:
                        sublayer_recorder.set_phase(None)
                if verify_kv_cache_cold_recompute and step_index == 0:
                    cold_cache = kv_runner.new_cache_bank()
                    zero_sigma = torch.zeros((), device=device, dtype=torch.float32)
                    for prefix_block in ar_blocks[: block.index]:
                        prefix_inputs = prepare_inputs(
                            positive_schedule,
                            zero_sigma,
                            block_index=prefix_block.index,
                            noisy=False,
                        )
                        kv_runner.forward(
                            prefix_inputs,
                            cache=cold_cache,
                            video_start=(
                                prefix_block.video_start
                                * video_shape.height
                                * video_shape.width
                            ),
                            audio_start=prefix_block.audio_start,
                            action_button_meaning_fusion=action_button_meaning_fusion,
                        )
                    cold_output = kv_runner.predict_x0(
                        positive_current_inputs,
                        cache=cold_cache,
                        video_start=video_token_start,
                        audio_start=block.audio_start,
                        action_button_meaning_fusion=action_button_meaning_fusion,
                    )
                    cold_parity = {
                        "block": block.index,
                        **compare_kv_cache_outputs(cond_output, cold_output),
                    }
                    kv_parity.setdefault("cond_cold_recompute", []).append(cold_parity)
                    print(
                        f"[ar-eval] KV cold-recompute parity cond: {cold_parity}",
                        flush=True,
                    )
                    del cold_cache, cold_output
                if capture_cond_layers:
                    layer_parity: list[dict[str, Any]] = []
                    handles = []
                    projected_kv_records: list[dict[str, Any]] = []

                    def full_layer_hook(layer_index: int):
                        def hook(
                            _module: Any,
                            _args: tuple[Any, ...],
                            output: tuple[Any, Any],
                        ) -> None:
                            full_video_args, full_audio_args = output
                            cached_video, cached_audio = cached_layer_states[
                                layer_index
                            ]
                            reference_video = full_video_args.x[
                                :,
                                video_total + video_token_start : video_total
                                + video_token_end,
                            ]
                            reference_audio = full_audio_args.x[
                                :,
                                audio_total + block.audio_start : audio_total
                                + block.audio_end,
                            ]
                            layer_parity.append(
                                {
                                    "layer": layer_index,
                                    **compare_kv_cache_outputs(
                                        (cached_video, cached_audio),
                                        (reference_video, reference_audio),
                                    ),
                                }
                            )

                        return hook

                    for layer_index, transformer_block in enumerate(
                        transformer.transformer_blocks
                    ):
                        handles.append(
                            transformer_block.register_forward_hook(
                                full_layer_hook(layer_index)
                            )
                        )
                    if verify_kv_cache_projected_kv:
                        projected_handles, projected_kv_records = (
                            register_full_projected_kv_parity_hooks(
                                transformer=transformer,
                                cache_bank=kv_cache_banks["cond"],
                                video_total=video_total,
                                audio_total=audio_total,
                                video_start=video_token_start,
                                video_end=video_token_end,
                                audio_start=block.audio_start,
                                audio_end=block.audio_end,
                            )
                        )
                        handles.extend(projected_handles)
                    try:
                        if sublayer_recorder is not None:
                            sublayer_recorder.set_phase("full")
                        reference = current_block_model_pass(
                            transformer=transformer,
                            inputs=positive_inputs,
                            block_index=block.index,
                            action_button_meaning_fusion=action_button_meaning_fusion,
                        )
                    finally:
                        if sublayer_recorder is not None:
                            sublayer_recorder.set_phase(None)
                            sublayer_recorder.remove()
                        for handle in handles:
                            handle.remove()
                    kv_parity["cond"] = compare_kv_cache_outputs(cond_output, reference)
                    kv_parity["cond_layers"] = layer_parity
                    if verify_kv_cache_projected_kv:
                        kv_parity["cond_projected_kv"] = projected_kv_records
                        projected_summary = [
                            {
                                "layer": record["layer"],
                                "attention": record["attention"],
                                "prefix_k_relative": record["prefix_k"][
                                    "relative_mean_abs"
                                ],
                                "prefix_v_relative": record["prefix_v"][
                                    "relative_mean_abs"
                                ],
                                "current_k_relative": record["current_k"][
                                    "relative_mean_abs"
                                ],
                                "current_v_relative": record["current_v"][
                                    "relative_mean_abs"
                                ],
                            }
                            for record in projected_kv_records
                            if record["attention"] == "video_self"
                        ]
                        print(
                            f"[ar-eval] KV projected parity cond/video_self: {projected_summary}",
                            flush=True,
                        )
                    if sublayer_recorder is not None:
                        sublayer_recorder.assert_complete()
                        kv_parity["cond_sublayers"] = sublayer_recorder.records
                        sublayer_summary = [
                            {
                                "layer": record["layer"],
                                "sublayer": record["sublayer"],
                                "relative": record["relative_mean_abs"],
                                "max_abs": record["max_abs"],
                            }
                            for record in sublayer_recorder.records
                            if record["layer"] in (0, 36, 37, 38)
                        ]
                        print(
                            f"[ar-eval] KV sublayer parity cond: {sublayer_summary}",
                            flush=True,
                        )
                    selected_layers = [
                        layer_parity[index]
                        for index in (0, 1, 3, 7, 15, 23, 31, 39, len(layer_parity) - 1)
                        if index < len(layer_parity)
                    ]
                    print(
                        f"[ar-eval] KV layer parity cond: {selected_layers}", flush=True
                    )
                    print(f"[ar-eval] KV parity cond: {kv_parity['cond']}", flush=True)
            pass_outputs = {"cond": cond_output}

            if needs_unconditional_pass:
                negative_batch = build_strategy_batch(
                    strategy=strategy,
                    video_tokens=video_tokens,
                    audio_tokens=audio_tokens,
                    video_shape=video_shape,
                    audio_shape=audio_shape,
                    fps=float(fps),
                    action_joystick=action_joystick,
                    action_buttons=action_buttons,
                    prompt_schedule=negative_schedule,
                )
                negative_inputs = strategy.prepare_training_inputs(
                    negative_batch,
                    FixedSigmaSampler(sigma),
                )
                negative_inputs = inject_current_block(
                    negative_inputs,
                    block_index=block.index,
                    current_video=current_video,
                    current_audio=current_audio,
                )
                if kv_runner is None or full_prefix_anchor_step:
                    uncond_output = run_current_block_pass(
                        negative_inputs,
                        block_index=block.index,
                    )
                else:
                    negative_current_inputs = slice_ar_model_inputs(
                        negative_inputs,
                        block_index=block.index,
                        noisy=True,
                        compact_prompt_context=compact_prompt_context,
                    )
                    uncond_output = kv_runner.predict_x0(
                        negative_current_inputs,
                        cache=kv_cache_banks["uncond"],
                        video_start=video_token_start,
                        audio_start=block.audio_start,
                        action_button_meaning_fusion=action_button_meaning_fusion,
                    )
                    if (
                        verify_kv_cache_first_step
                        and block.index == 1
                        and step_index == 0
                    ):
                        reference = current_block_model_pass(
                            transformer=transformer,
                            inputs=negative_inputs,
                            block_index=block.index,
                            action_button_meaning_fusion=action_button_meaning_fusion,
                        )
                        kv_parity["uncond"] = compare_kv_cache_outputs(
                            uncond_output, reference
                        )
                        print(
                            f"[ar-eval] KV parity uncond: {kv_parity['uncond']}",
                            flush=True,
                        )
                pass_outputs["uncond"] = uncond_output

            if stg_perturbations is not None:
                if kv_runner is None or full_prefix_anchor_step:
                    ptb_output = run_current_block_pass(
                        positive_inputs,
                        block_index=block.index,
                        perturbations=stg_perturbations,
                    )
                else:
                    ptb_output = kv_runner.predict_x0(
                        positive_current_inputs,
                        cache=kv_cache_banks["ptb"],
                        video_start=video_token_start,
                        audio_start=block.audio_start,
                        action_button_meaning_fusion=action_button_meaning_fusion,
                        perturbations=stg_perturbations,
                    )
                    if (
                        verify_kv_cache_first_step
                        and block.index == 1
                        and step_index == 0
                    ):
                        reference = current_block_model_pass(
                            transformer=transformer,
                            inputs=positive_inputs,
                            block_index=block.index,
                            action_button_meaning_fusion=action_button_meaning_fusion,
                            perturbations=stg_perturbations,
                        )
                        kv_parity["ptb"] = compare_kv_cache_outputs(
                            ptb_output, reference
                        )
                        print(
                            f"[ar-eval] KV parity ptb: {kv_parity['ptb']}", flush=True
                        )
                pass_outputs["ptb"] = ptb_output

            if modality_perturbations is not None:
                if kv_runner is None or full_prefix_anchor_step:
                    mod_output = run_current_block_pass(
                        positive_inputs,
                        block_index=block.index,
                        perturbations=modality_perturbations,
                    )
                else:
                    mod_output = kv_runner.predict_x0(
                        positive_current_inputs,
                        cache=kv_cache_banks["mod"],
                        video_start=video_token_start,
                        audio_start=block.audio_start,
                        action_button_meaning_fusion=action_button_meaning_fusion,
                        perturbations=modality_perturbations,
                    )
                    if (
                        verify_kv_cache_first_step
                        and block.index == 1
                        and step_index == 0
                    ):
                        reference = current_block_model_pass(
                            transformer=transformer,
                            inputs=positive_inputs,
                            block_index=block.index,
                            action_button_meaning_fusion=action_button_meaning_fusion,
                            perturbations=modality_perturbations,
                        )
                        kv_parity["mod"] = compare_kv_cache_outputs(
                            mod_output, reference
                        )
                        print(
                            f"[ar-eval] KV parity mod: {kv_parity['mod']}", flush=True
                        )
                pass_outputs["mod"] = mod_output

            denoised_video, denoised_audio = apply_multimodal_guidance(
                video_params=video_guider_params,
                audio_params=audio_guider_params,
                pass_outputs=pass_outputs,
            )

            next_sigma = sigmas[step_index + 1]
            if denoising_update_mode == "euler":
                                                                                  
                                                                     
                if float(next_sigma.item()) > 0.0:
                    delta = next_sigma - sigma
                    current_video = (
                        current_video.float()
                        + (current_video.float() - denoised_video.float())
                        / sigma.float()
                        * delta.float()
                    ).to(dtype=dtype)
                    current_audio = (
                        current_audio.float()
                        + (current_audio.float() - denoised_audio.float())
                        / sigma.float()
                        * delta.float()
                    ).to(dtype=dtype)
                else:
                    current_video = denoised_video.to(dtype=dtype)
                    current_audio = denoised_audio.to(dtype=dtype)
            elif denoising_update_mode == "consistency_renoise":
                                                                                
                                                                           
                from ltx_trainer.causal_rcm_inference import consistency_renoise_step

                current_video = consistency_renoise_step(
                    denoised=denoised_video,
                    next_sigma=next_sigma,
                    generator=generator,
                    output_dtype=dtype,
                )
                current_audio = consistency_renoise_step(
                    denoised=denoised_audio,
                    next_sigma=next_sigma,
                    generator=generator,
                    output_dtype=dtype,
                )
            else:
                raise ValueError(
                    f"Unsupported denoising_update_mode={denoising_update_mode!r}."
                )

            print(
                f"[ar-eval] {case['case_id']} block={block.index}/{len(ar_blocks) - 1} "
                f"step={step_index + 1}/{num_inference_steps} sigma={float(sigma):.5f}",
                flush=True,
            )

        video_tokens[:, video_token_start:video_token_end] = current_video
        audio_tokens[:, block.audio_start : block.audio_end] = current_audio
        if kv_runner is not None:
            zero_sigma = torch.zeros((), device=device, dtype=torch.float32)
            positive_clean = prepare_inputs(
                positive_schedule,
                zero_sigma,
                block_index=block.index,
                noisy=False,
            )
            kv_runner.forward(
                positive_clean,
                cache=kv_cache_banks["cond"],
                video_start=video_token_start,
                audio_start=block.audio_start,
                action_button_meaning_fusion=action_button_meaning_fusion,
            )
            if "ptb" in kv_cache_banks:
                kv_runner.forward(
                    positive_clean,
                    cache=kv_cache_banks["ptb"],
                    video_start=video_token_start,
                    audio_start=block.audio_start,
                    action_button_meaning_fusion=action_button_meaning_fusion,
                    perturbations=stg_perturbations,
                )
            if "mod" in kv_cache_banks:
                kv_runner.forward(
                    positive_clean,
                    cache=kv_cache_banks["mod"],
                    video_start=video_token_start,
                    audio_start=block.audio_start,
                    action_button_meaning_fusion=action_button_meaning_fusion,
                    perturbations=modality_perturbations,
                )
            if "uncond" in kv_cache_banks:
                negative_clean = prepare_inputs(
                    negative_schedule,
                    zero_sigma,
                    block_index=block.index,
                    noisy=False,
                )
                kv_runner.forward(
                    negative_clean,
                    cache=kv_cache_banks["uncond"],
                    video_start=video_token_start,
                    audio_start=block.audio_start,
                    action_button_meaning_fusion=action_button_meaning_fusion,
                )

        if (
            vlm_two_hit_config is not None
            and vlm_two_hit_config.decision_mode == "pre_action_verified"
        ):
            if (
                two_hit_rule is None
                or interaction_prompt_schedule is None
                or encoded_prompt_bank is None
            ):
                raise RuntimeError("Incomplete verified pre-action VLM runtime state")
            current_prompt_block = block.index - 1
            next_prompt_block = block.index
            pending_frames = sorted(
                frame
                for frame, pending in pending_pre_action_events.items()
                if int(pending["block_index"]) == block.index
            )
            for fire_frame in pending_frames:
                pending = pending_pre_action_events.pop(fire_frame)
                shot_ordinal = fire_frames.index(fire_frame) + 1
                event_id = f"guidance_audit_shot_{shot_ordinal}_frame_{fire_frame:03d}"
                observation, observation_metadata = observe_generated_prefix(
                    block=block,
                    video_token_end=video_token_end,
                    event_id=event_id,
                    event_kind="shot",
                    shot_frame=fire_frame,
                )
                actual_transition = two_hit_rule.observe_shot(event_id, observation)
                observed_fire_frames.add(fire_frame)
                lethal_hit_verified = bool(
                    vlm_two_hit_config.duel_loop
                    and actual_transition.counted_hit
                    and actual_transition.hp_after == 0
                )
                if lethal_hit_verified:
                                                                         
                                                                            
                                                                               
                                                                             
                                                                             
                    duel_death_confirmed_block = current_prompt_block
                    death_prompt_block_index = None
                expected_mode = pending["applied_mode"]
                audit = audit_interaction_guidance(
                    expected_mode=expected_mode,
                    observation=observation,
                    min_confidence=vlm_two_hit_config.min_hit_confidence,
                )
                next_mode: InteractionMode | None = None
                if next_prompt_block < len(runtime_prompt_schedule):
                    if lethal_hit_verified:
                        next_mode = "persist_killed"
                        if duel_lifecycle_prompt_schedule is None:
                            raise RuntimeError(
                                "Duel lifecycle prompt schedule is unavailable"
                            )
                        next_prompt = duel_lifecycle_prompt_schedule[next_prompt_block][
                            "corpse"
                        ]
                    else:
                        next_mode = "active"
                        next_prompt = interaction_prompt_schedule[next_prompt_block][
                            next_mode
                        ]
                    apply_runtime_prompt_block(
                        prompt=next_prompt,
                        block_index=next_prompt_block,
                    )
                    if not vlm_two_hit_config.duel_loop:
                        death_prompt_block_index = (
                            next_prompt_block if next_mode == "collapse" else None
                        )
                else:
                    death_prompt_block_index = None
                record_agent_event(
                    {
                        "event_id": event_id,
                        "after_ar_block": block.index,
                        "current_prompt_block": current_prompt_block,
                        "shot_frame": fire_frame,
                        "forecast_event_id": pending["forecast_event_id"],
                        "forecast_transition": pending["forecast_transition"].to_dict(),
                        "applied_prompt_mode": expected_mode,
                        "guidance_audit": audit.to_dict(),
                        "actual_transition": actual_transition.to_dict(),
                        "next_prompt_block": (
                            next_prompt_block
                            if next_prompt_block < len(runtime_prompt_schedule)
                            else None
                        ),
                        "next_prompt_mode": next_mode,
                        **observation_metadata,
                    }
                )

            if (
                not vlm_two_hit_config.duel_loop
                and death_prompt_block_index == current_prompt_block
                and not pending_frames
            ):
                event_id = f"guidance_audit_death_after_block_{block.index}"
                chunk_start = current_prompt_block * 16 + 1
                chunk_end = (current_prompt_block + 1) * 16
                observation, observation_metadata = observe_generated_prefix(
                    block=block,
                    video_token_end=video_token_end,
                    event_id=event_id,
                    event_kind="state_check",
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )
                transition = two_hit_rule.observe_state(event_id, observation)
                if vlm_two_hit_config.duel_loop and transition.death_visually_confirmed:
                    duel_death_confirmed_block = current_prompt_block
                audit = audit_interaction_guidance(
                    expected_mode="collapse",
                    observation=observation,
                    min_confidence=vlm_two_hit_config.min_hit_confidence,
                )
                next_mode = None
                if next_prompt_block < len(runtime_prompt_schedule):
                    next_mode = transition.next_mode
                    next_prompt = interaction_prompt_schedule[next_prompt_block][
                        next_mode
                    ]
                    apply_runtime_prompt_block(
                        prompt=next_prompt,
                        block_index=next_prompt_block,
                    )
                    death_prompt_block_index = (
                        next_prompt_block if next_mode == "collapse" else None
                    )
                else:
                    death_prompt_block_index = None
                record_agent_event(
                    {
                        "event_id": event_id,
                        "after_ar_block": block.index,
                        "current_prompt_block": current_prompt_block,
                        "guidance_audit": audit.to_dict(),
                        "transition": transition.to_dict(),
                        "next_prompt_block": (
                            next_prompt_block
                            if next_prompt_block < len(runtime_prompt_schedule)
                            else None
                        ),
                        "next_prompt_mode": next_mode,
                        **observation_metadata,
                    }
                )

        if (
            vlm_two_hit_config is not None
            and vlm_two_hit_config.decision_mode == "posthoc"
        ):
            if (
                two_hit_rule is None
                or interaction_prompt_schedule is None
                or encoded_prompt_bank is None
            ):
                raise RuntimeError("Incomplete VLM two-hit runtime state")
            generated_end_frame = (block.video_end - 1) * 8
            current_prompt_block = block.index - 1
            next_prompt_block = block.index

            for shot_ordinal, fire_frame in enumerate(fire_frames, start=1):
                if fire_frame in observed_fire_frames:
                    continue
                if (
                    generated_end_frame - fire_frame
                    < vlm_two_hit_config.required_post_shot_frames
                ):
                    continue
                event_id = f"shot_{shot_ordinal}_frame_{fire_frame:03d}"
                observation, observation_metadata = observe_generated_prefix(
                    block=block,
                    video_token_end=video_token_end,
                    event_id=event_id,
                    event_kind="shot",
                    shot_frame=fire_frame,
                )
                transition = two_hit_rule.observe_shot(event_id, observation)
                observed_fire_frames.add(fire_frame)
                applied_mode: InteractionMode | None = None
                if next_prompt_block < len(runtime_prompt_schedule):
                    applied_mode = transition.next_mode
                    next_prompt = interaction_prompt_schedule[next_prompt_block][
                        applied_mode
                    ]
                    apply_runtime_prompt_block(
                        prompt=next_prompt,
                        block_index=next_prompt_block,
                    )
                    if applied_mode == "collapse":
                        death_prompt_block_index = next_prompt_block
                    elif applied_mode == "persist_killed":
                        death_prompt_block_index = None
                record_agent_event(
                    {
                        "event_id": event_id,
                        "after_ar_block": block.index,
                        "current_prompt_block": current_prompt_block,
                        "shot_frame": fire_frame,
                        "transition": transition.to_dict(),
                        "next_prompt_block": (
                            next_prompt_block
                            if next_prompt_block < len(runtime_prompt_schedule)
                            else None
                        ),
                        "next_prompt_mode": applied_mode,
                        **observation_metadata,
                    }
                )

            if death_prompt_block_index == current_prompt_block:
                event_id = f"death_check_after_block_{block.index}"
                chunk_start = current_prompt_block * 16 + 1
                chunk_end = (current_prompt_block + 1) * 16
                observation, observation_metadata = observe_generated_prefix(
                    block=block,
                    video_token_end=video_token_end,
                    event_id=event_id,
                    event_kind="state_check",
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )
                transition = two_hit_rule.observe_state(event_id, observation)
                applied_mode = None
                if next_prompt_block < len(runtime_prompt_schedule):
                    applied_mode = transition.next_mode
                    next_prompt = interaction_prompt_schedule[next_prompt_block][
                        applied_mode
                    ]
                    apply_runtime_prompt_block(
                        prompt=next_prompt,
                        block_index=next_prompt_block,
                    )
                    death_prompt_block_index = (
                        next_prompt_block if applied_mode == "collapse" else None
                    )
                else:
                    death_prompt_block_index = None
                record_agent_event(
                    {
                        "event_id": event_id,
                        "after_ar_block": block.index,
                        "current_prompt_block": current_prompt_block,
                        "transition": transition.to_dict(),
                        "next_prompt_block": (
                            next_prompt_block
                            if next_prompt_block < len(runtime_prompt_schedule)
                            else None
                        ),
                        "next_prompt_mode": applied_mode,
                        **observation_metadata,
                    }
                )
        if (
            vlm_action_controller_config is not None
            and block.index < len(ar_blocks) - 1
        ):
            observation, action_plan, action_metadata = observe_navigation_and_plan(
                block=block,
                video_token_end=video_token_end,
            )
            next_action_block = block.index
            interval_start_frame = 1 + next_action_block * 16
            interval_end_frame = (next_action_block + 1) * 16
            action_joystick[interval_start_frame : interval_end_frame + 1] = (
                torch.tensor(
                    [
                        action_plan.left_x,
                        action_plan.left_y,
                        action_plan.right_x,
                        action_plan.right_y,
                    ],
                    dtype=action_joystick.dtype,
                    device=action_joystick.device,
                )
            )
            action_buttons[interval_start_frame : interval_end_frame + 1].zero_()
            action_fire_frames[:] = [
                frame
                for frame in action_fire_frames
                if not interval_start_frame <= frame <= interval_end_frame
            ]
            fire_frames[:] = [
                frame
                for frame in fire_frames
                if not interval_start_frame <= frame <= interval_end_frame
            ]
            fire_allowed = (
                not (two_hit_rule is not None and two_hit_rule.hp == 0)
                and duel_death_confirmed_block is None
            )
            if action_plan.fire and not fire_allowed:
                action_plan = replace(action_plan, fire=False)
            applied_fire_frame: int | None = None
            if action_plan.fire:
                applied_fire_frame = min(
                    interval_end_frame,
                    interval_start_frame
                    + vlm_action_controller_config.fire_pulse_offset,
                )
                action_buttons[applied_fire_frame, 0] = 1.0
                action_fire_frames.append(applied_fire_frame)
                fire_frames.append(applied_fire_frame)
                action_fire_frames.sort()
                fire_frames.sort()
                                                                             
                                                                              
                                                            
                action_joystick[
                    applied_fire_frame + 1 : interval_end_frame + 1, 2:4
                ].zero_()
            record_agent_event(
                {
                    "event_id": f"action_controller_after_block_{block.index:03d}",
                    "after_ar_block": block.index,
                    "event_kind": "next_block_action_control",
                    "next_action_block": next_action_block,
                    "next_frame_interval": [
                        interval_start_frame,
                        interval_end_frame,
                    ],
                    "fire_frame": applied_fire_frame,
                    "right_stick_zero_after_fire": applied_fire_frame is not None,
                    "fire_allowed_by_combat_lifecycle": fire_allowed,
                    **action_metadata,
                    "action_plan": action_plan.to_dict(),
                }
            )
        block_times.append(time.perf_counter() - block_start)
        torch.cuda.empty_cache()

    sampling_seconds = time.perf_counter() - case_start
    kv_cache_allocated_bytes = sum(
        bank.allocated_bytes for bank in kv_cache_banks.values()
    )
    kv_peak_cuda_allocated_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if use_kv_cache and device.type == "cuda"
        else None
    )
    video_latent = strategy._video_patchifier.unpatchify(
        video_tokens, output_shape=video_shape
    )
    if latent_only:
        if save_latent_cache:
            raise ValueError(
                "latent_only returns the latent directly; disable save_latent_cache"
            )
        return {
            "case_id": case["case_id"],
            "sample_id": row.get("sample_id"),
            "video_latent": video_latent.detach().cpu().to(torch.bfloat16),
            "sampling_seconds": sampling_seconds,
            "denoising_sigmas": [float(value) for value in sigmas.tolist()],
            "denoising_update_mode": denoising_update_mode,
            "kv_cache_enabled": kv_runner is not None,
            "kv_cache_attention_backend": (
                kv_runner.attention_backend if kv_runner is not None else None
            ),
            "kv_cache_full_prefix_interval": kv_cache_full_prefix_interval,
            "kv_cache_attention_route_stats": (
                kv_runner.attention_route_stats if kv_runner is not None else None
            ),
        }
    video = decode_video_tokens(video_tokens, generation_config)
    audio_latent = strategy._audio_patchifier.unpatchify(
        audio_tokens, output_shape=audio_shape
    )

    if save_latent_cache:
        cache_path = case_dir / "stage1_cache.pt"
        temporary_cache_path = cache_path.with_suffix(".pt.tmp")
        torch.save(
            {
                                                                       
                                                                            
                                                                        
                                                 
                "schema_version": "ar_teacher_stage1_refiner_cache_v2",
                "case_id": case["case_id"],
                "seed": case["seed"],
                "generator_state": generator.get_state().detach().cpu(),
                "video_latent": video_latent.detach().cpu(),
                "audio_latent": audio_latent.detach().cpu(),
                "video_prompt_schedule_embeds": positive_schedule[
                    "video_prompt_schedule_embeds"
                ]
                .detach()
                .cpu(),
                "audio_prompt_schedule_embeds": positive_schedule[
                    "audio_prompt_schedule_embeds"
                ]
                .detach()
                .cpu(),
                "prompt_schedule_attention_mask": positive_schedule[
                    "prompt_schedule_attention_mask"
                ]
                .detach()
                .cpu(),
                "prompt_schedule": runtime_prompt_schedule,
                "condition_image_path": str(cache_condition_image_path),
                "refiner_condition_image_path": str(refiner_condition_image_path),
                                                                             
                                                                               
                                                                              
                "action_joystick": action_joystick.detach().cpu(),
                "action_buttons": action_buttons.detach().cpu(),
                "action_joystick_path": str(Path(row["action_joystick_path"]).resolve())
                if case["is_custom_image_case"]
                else None,
                "action_buttons_path": str(Path(row["action_buttons_path"]).resolve())
                if case["is_custom_image_case"]
                else None,
                "source_action_path": str(Path(row["action_path"]).resolve())
                if not case["is_custom_image_case"]
                else None,
                "stage1_geometry": {
                    "width": width,
                    "height": height,
                    "num_frames": num_frames,
                    "fps": fps,
                    "video_latent_frames": video_shape.frames,
                    "video_latent_height": video_shape.height,
                    "video_latent_width": video_shape.width,
                    "audio_latent_frames": audio_shape.frames,
                },
                "ar_chunk_size": strategy.config.chunk_size,
                "video_temporal_scale": strategy.config.video_temporal_scale,
                "audio_latent_fps": strategy.config.audio_latent_fps,
            },
            temporary_cache_path,
        )
        temporary_cache_path.replace(cache_path)
    audio_decode_latent = audio_latent.to(dtype=component_dtype)
    decoded_audio_state = LatentState(
        latent=audio_decode_latent,
        denoise_mask=torch.ones_like(audio_decode_latent[:, :1, :, :1]),
        positions=torch.empty(0, device=device),
        clean_latent=audio_decode_latent,
    )
    audio = sampler._decode_audio(decoded_audio_state, device)

    save_video(
        video_tensor=video,
        output_path=case_dir / "pred.mp4",
        fps=fps,
        audio=audio,
        audio_sample_rate=generated_audio_sample_rate,
    )
    save_video_with_action_overlay(
        video_tensor=video,
        output_path=case_dir / "pred_actions.mp4",
        fps=fps,
        joystick=action_joystick,
        buttons=action_buttons,
        audio=audio,
        audio_sample_rate=generated_audio_sample_rate,
    )
    if vlm_action_controller_config is not None:
                                                                             
                                                        
        save_action_plot(
            case_dir / "actions.png", action_joystick, action_buttons, case["case_id"]
        )
    comparison = torch.cat([gt_video.permute(1, 0, 2, 3), video], dim=3)
    save_video(
        video_tensor=comparison,
        output_path=case_dir / "side_by_side.mp4",
        fps=fps,
    )

    metrics = (
        compute_temporal_metrics(video)
        if case["is_custom_image_case"]
        else compute_video_metrics(video, gt_video)
    )
    metrics.update(
        {
            "case_id": case["case_id"],
            "sample_id": row.get("sample_id"),
            "clip_id": row.get("clip_id"),
            "game": row.get("game"),
            "is_variable_prompt": case["is_variable_prompt"],
            "is_custom_image_case": case["is_custom_image_case"],
            "prompt_schedule_unique_count": len(set(runtime_prompt_schedule)),
            "num_inference_steps": num_inference_steps,
            "transformer_dtype": str(dtype).removeprefix("torch."),
            "component_dtype": str(component_dtype).removeprefix("torch."),
            "video_guider_params": asdict(video_guider_params),
            "audio_guider_params": asdict(audio_guider_params),
            "guidance_mode": guidance_mode,
            "denoising_sigmas": [float(value) for value in sigmas.tolist()],
            "sigma_schedule_source": sigma_schedule_source,
            "scheduler_timestep_shift": scheduler_timestep_shift,
            "sampling_seconds": sampling_seconds,
            "block_seconds": block_times,
            "kv_cache_enabled": use_kv_cache,
            "kv_cache_attention_backend": kv_cache_attention_backend,
            "kv_cache_attention_route_stats": (
                kv_runner.attention_route_stats if kv_runner is not None else None
            ),
            "kv_cache_full_prefix_interval": kv_cache_full_prefix_interval,
            "kv_cache_estimated_capacity_bytes": kv_cache_estimated_bytes,
            "kv_cache_allocated_bytes": kv_cache_allocated_bytes,
            "kv_cache_peak_cuda_allocated_bytes": kv_peak_cuda_allocated_bytes,
            "kv_cache_first_step_parity": kv_parity,
            "history_generated_blocks": history_blocks,
            "vlm_two_hit_kill_enabled": vlm_two_hit_config is not None,
            "vlm_two_hit_decision_mode": (
                vlm_two_hit_config.decision_mode
                if vlm_two_hit_config is not None
                else None
            ),
            "vlm_two_hit_final_hp": two_hit_rule.hp
            if two_hit_rule is not None
            else None,
            "vlm_two_hit_death_requested": (
                two_hit_rule.death_requested if two_hit_rule is not None else None
            ),
            "vlm_two_hit_death_visually_confirmed": (
                two_hit_rule.death_visually_confirmed
                if two_hit_rule is not None
                else None
            ),
            "vlm_two_hit_observed_fire_frames": sorted(observed_fire_frames),
            "vlm_two_hit_forecasted_fire_frames": sorted(forecasted_fire_frames),
            "vlm_two_hit_pending_forecast_frames": sorted(pending_pre_action_events),
            "vlm_joystick_pixel_lookup": (
                vlm_two_hit_config.joystick_pixel_lookup.calibration_id
                if vlm_two_hit_config is not None
                and vlm_two_hit_config.joystick_pixel_lookup is not None
                else None
            ),
            "vlm_two_hit_action_fire_frames": action_fire_frames,
            "vlm_two_hit_decision_fire_frames": fire_frames,
            "vlm_duel_loop_enabled": bool(
                vlm_two_hit_config is not None and vlm_two_hit_config.duel_loop
            ),
            "vlm_duel_generation": duel_generation,
            "vlm_duel_player_final_hp": duel_player_hp,
            "vlm_duel_enemy_attack_blocks": sorted(duel_enemy_attack_blocks),
            "vlm_duel_respawn_delay_blocks": duel_respawn_delay_blocks,
            "vlm_duel_death_confirmed_block": duel_death_confirmed_block,
            "vlm_duel_suppressed_fire_frames": sorted(duel_suppressed_fire_frames),
            "vlm_duel_prompt_contract_manifest": (
                str(duel_prompt_contract_path)
                if duel_prompt_contract_path is not None
                else None
            ),
            "vlm_duel_player_outcome_contract": (
                {
                    "viewpoint": "first_person_player",
                    "death_sentence": (
                        "The opponent kills the player during this block."
                    ),
                    "survival_representation": "death_sentence_absent",
                    "player_death_enabled": False,
                }
                if vlm_two_hit_config is not None and vlm_two_hit_config.duel_loop
                else None
            ),
            "vlm_agent_event_count": len(agent_trace),
            "vlm_action_controller_enabled": (vlm_action_controller_config is not None),
            "vlm_action_controller_event_count": sum(
                event.get("event_kind") == "next_block_action_control"
                for event in agent_trace
            ),
            "seed": case["seed"],
            "stage1_latent_cache": str((case_dir / "stage1_cache.pt").resolve())
            if save_latent_cache
            else None,
        }
    )
    if denoising_update_mode != "euler":
        metrics["denoising_update_mode"] = denoising_update_mode
    (case_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (case_dir / "prompts.json").write_text(
        json.dumps(runtime_prompt_schedule, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--heldout-manifest", type=Path, default=DEFAULT_HELDOUT_MANIFEST
    )
    parser.add_argument(
        "--variable-manifest", type=Path, default=DEFAULT_VARIABLE_MANIFEST
    )
    parser.add_argument("--heldout-index", type=int, default=0)
    parser.add_argument(
        "--variable-index",
        type=int,
        action="append",
        dest="variable_indices",
        default=None,
        help=(
            "Variable-manifest row to evaluate. Repeat this option to evaluate multiple "
            "rows with one model load. Defaults to index 2 when omitted."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Optional diagnostic override. By default this is resolved from the base LTX checkpoint.",
    )
    parser.add_argument(
        "--denoising-sigmas",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional endpoint-inclusive sigma schedule. When supplied, its interval count "
            "defines the inference-step count and bypasses LTX2Scheduler construction."
        ),
    )
    parser.add_argument(
        "--guidance-mode",
        choices=("config", "conditional_only"),
        default="config",
        help=(
            "Use guidance from the training config, or run only the conditional branch. "
            "Few-step DMD generators normally require conditional_only because teacher "
            "guidance is already distilled into the generator."
        ),
    )
    parser.add_argument("--scheduler-timestep-shift", type=float, default=2.05)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--allow-bootstrap-prefix-h4",
        action="store_true",
        help=(
            "Allow the exact 65-frame sink+H1-H4 causal-prefix bootstrap diagnostic. "
            "This opt-in geometry is restricted to full-prefix inference without KV cache "
            "or a cropped sliding-history window."
        ),
    )
    parser.add_argument(
        "--allow-experimental-long-horizon",
        action="store_true",
        help=(
            "Allow an opt-in 16k+1-frame Xonotic AR rollout longer than the formal 81-frame "
            "training geometry, up to 609 frames (30.4 generated seconds). Width, height, "
            "and FPS remain fixed."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--low-host-memory",
        action="store_true",
        help=(
            "Move the loaded transformer to the target GPU before loading text/VAE "
            "components. This is opt-in and leaves the established evaluator path unchanged."
        ),
    )
    parser.add_argument(
        "--attention-backend",
        choices=(
            "default",
            "pytorch",
            "xformers",
            "flash_attention_2",
            "flash_attention_3",
            "flash_attention_4",
        ),
        default=None,
        help=(
            "Transformer attention override. Defaults to model.attention_backend from the "
            "training config, so sequential-KV FA2 training and inference stay aligned."
        ),
    )
    parser.add_argument(
        "--fa4-mask-execution",
        choices=("native_mask_mod", "segmented_exact"),
        default="native_mask_mod",
        help=(
            "How explicit boolean masks are executed when FlashAttention4 is selected. "
            "segmented_exact partitions equal query-mask rows and runs only mask-free FA4 "
            "on their selected KV tokens; it is an exact, fail-closed SM103 workaround."
        ),
    )
    parser.add_argument(
        "--mask-backend",
        choices=("auto", "dense", "flex"),
        default=None,
        help=(
            "Evaluation-only AR structural-mask override. The default preserves the training "
            "config (and therefore the canonical full-prefix path). Use 'dense' only for a "
            "matched-SDPA KV parity diagnostic."
        ),
    )
    parser.add_argument(
        "--kv-cache-transformer-dtype",
        choices=("match", "fp16"),
        default="match",
        help=(
            "With --use-kv-cache, optionally run only the compact Transformer and KV banks "
            "in FP16. Text encoders, VAEs, decoders, and initial RNG sampling keep --dtype."
        ),
    )
    parser.add_argument(
        "--only", choices=("both", "heldout", "variable"), default="both"
    )
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
    parser.add_argument("--cases-only", action="store_true")
    parser.add_argument(
        "--save-latent-cache",
        action="store_true",
        help=(
            "Save the native unpatchified Stage-1 video/audio latents, per-block positive "
            "prompt embeddings, and final RNG state for a native two-stage refiner run."
        ),
    )
    parser.add_argument(
        "--use-kv-cache",
        action="store_true",
        help=(
            "Opt in to the action-aware incremental KV-cache path. The default remains the "
            "established full-prefix evaluator."
        ),
    )
    parser.add_argument(
        "--kv-cache-attention-backend",
        choices=("native", "flex"),
        default="native",
        help=(
            "Backend for compact cached self/A2V/V2A attention only. 'native' preserves the "
            "existing unmasked configured backend; 'flex' is a fail-closed matched-kernel "
            "diagnostic using an all-real-keys BlockMask."
        ),
    )
    parser.add_argument(
        "--kv-cache-full-prefix-interval",
        type=int,
        default=0,
        help=(
            "In KV-cache mode, rerun the canonical full-prefix path every N denoising steps "
            "to limit accumulated numerical drift. Zero keeps the pure KV-cache path."
        ),
    )
    parser.add_argument(
        "--history-blocks",
        type=int,
        default=None,
        help=(
            "Experimental sliding visual/audio history. Keep the image condition plus this "
            "many completed generated chunks; physically crop older tokens. The formal default "
            "is the unchanged full prefix."
        ),
    )
    parser.add_argument(
        "--verify-kv-cache-first-step",
        action="store_true",
        help=(
            "With --use-kv-cache, also run the legacy full-prefix path for the first block/step "
            "and record per-guidance numerical differences."
        ),
    )
    parser.add_argument(
        "--verify-kv-cache-projected-kv",
        action="store_true",
        help=(
            "With --verify-kv-cache-first-step, compare compact cached projected K/V against "
            "the matching clean-prefix and current-noisy K/V from selected full-prefix layers."
        ),
    )
    parser.add_argument(
        "--verify-kv-cache-sublayers",
        action="store_true",
        help=(
            "With --verify-kv-cache-first-step, compare selected compact/full current-query "
            "q_norm, attention, action, and feed-forward submodule outputs."
        ),
    )
    parser.add_argument(
        "--verify-kv-cache-cold-recompute",
        action="store_true",
        help=(
            "With --use-kv-cache, compare each block's first cond pass against a fresh compact "
            "cache rebuilt sequentially from all completed clean blocks."
        ),
    )
    parser.add_argument(
        "--vlm-two-hit-kill",
        action="store_true",
        help=(
            "Enable the opt-in closed loop: an OpenAI-compatible VLM observes the two "
            "right-trigger pulses and deterministic 100-HP/50-damage logic selects future prompts."
        ),
    )
    parser.add_argument(
        "--vlm-two-hit-mode",
        choices=("posthoc", "pre_action", "pre_action_verified"),
        default="posthoc",
        help=(
            "posthoc preserves the original observe-then-guide-next-block pilot. pre_action "
            "predicts each upcoming hit from the causal prefix and known current-block actions, "
            "then applies the aligned outcome prompt to that same block. pre_action_verified "
            "keeps that alignment, treats the prediction as provisional, and commits HP only "
            "after a one-vote visual audit of the generated block."
        ),
    )
    parser.add_argument("--vlm-base-url", default="http://127.0.0.1:8012/v1")
    parser.add_argument("--vlm-model", default="internvl3-8b")
    parser.add_argument(
        "--vlm-action-controller",
        action="store_true",
        help=(
            "Opt in to a second block-boundary VLM role that observes the latest "
            "generated view and writes normalized movement, camera, and short-pulse "
            "primary-fire actions for the next AR block."
        ),
    )
    parser.add_argument("--vlm-action-move-magnitude", type=float, default=0.20)
    parser.add_argument("--vlm-action-strafe-magnitude", type=float, default=0.18)
    parser.add_argument("--vlm-action-turn-magnitude", type=float, default=0.025)
    parser.add_argument("--vlm-action-aim-magnitude", type=float, default=0.035)
    parser.add_argument(
        "--vlm-action-vertical-aim-magnitude", type=float, default=0.018
    )
    parser.add_argument(
        "--vlm-action-fire-pulse-offset",
        type=int,
        default=12,
        help="Frame offset [0,15] for one primary-fire pulse in a VLM-planned block.",
    )
    parser.add_argument("--vlm-min-hit-confidence", type=float, default=0.55)
    parser.add_argument("--vlm-required-post-shot-frames", type=int, default=6)
    parser.add_argument("--vlm-initial-hp", type=int, default=100)
    parser.add_argument("--vlm-confirmed-hit-damage", type=int, default=50)
    parser.add_argument(
        "--vlm-pre-action-votes",
        type=int,
        default=1,
        help=(
            "Repeat each pre-action VLM prediction and require strict consensus. "
            "The default of 1 preserves existing evaluations."
        ),
    )
    parser.add_argument(
        "--vlm-pre-action-grounding",
        action="store_true",
        help=(
            "Use VLM enemy bounding boxes plus deterministic crosshair geometry instead "
            "of asking the VLM to directly classify the future hit."
        ),
    )
    parser.add_argument(
        "--vlm-joystick-pixel-lookup",
        type=Path,
        help=(
            "Measured controller-to-pixel calibration used by pre_action_verified. The "
            "controller projects the latest causal frame to the fire frame, then the VLM "
            "directly judges crosshair geometry using the real history plus projected frame."
        ),
    )
    parser.add_argument(
        "--vlm-direct-crosshair-crop-size",
        type=int,
        help=(
            "Opt in to the direct-crosshair V2 observer. The VLM receives only a fixed "
            "aim-centered square crop of this many native pixels for hit geometry; no "
            "enemy boxes are drawn. Post-shot posture is observed in a separate call. "
            "Use 320 for the 832x480 Xonotic evaluation geometry."
        ),
    )
    parser.add_argument(
        "--vlm-aim-target-ring-v9",
        action="store_true",
        help=(
            "Use the finalized V9 neutral hollow-ring observer for both the projected "
            "pre-action shot decision and the actual fire-frame verification. Requires "
            "pre_action_verified and --vlm-direct-crosshair-crop-size. No weapon, enemy, "
            "action, HP, history, or full-frame context is sent to this hit observer."
        ),
    )
    parser.add_argument(
        "--vlm-freeze-opponent-motion",
        action="store_true",
        help=(
            "For the closed-loop geometry ablation, rewrite only the latest opponent "
            "section so the opponent remains stationary and visible. HP and controller "
            "state remain private and are never inserted into the model prompt."
        ),
    )
    parser.add_argument(
        "--vlm-duel-loop",
        action="store_true",
        help=(
            "Enable the opt-in duel lifecycle controller on top of verified pre-action "
            "hit decisions: keep a confirmed corpse, respawn off-screen after the configured "
            "delay, then re-enter with a fresh two-hit enemy."
        ),
    )
    parser.add_argument("--vlm-duel-respawn-seconds", type=float, default=3.0)
    parser.add_argument("--vlm-duel-player-initial-hp", type=int, default=100)
    parser.add_argument("--vlm-duel-enemy-shot-damage", type=int, default=1)
    parser.add_argument(
        "--vlm-deferred-healthcheck-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Opt in to checking the VLM service only after the video Transformer is on "
            "GPU, then wait up to this many seconds. Zero preserves the eager check."
        ),
    )
    return parser.parse_args()


def main(
    *,
    denoising_update_mode: str = "euler",
    allow_kv_cache_with_consistency: bool = False,
) -> None:
    args = parse_args()
    if denoising_update_mode not in {"euler", "consistency_renoise"}:
        raise ValueError(
            f"Unsupported denoising_update_mode={denoising_update_mode!r}."
        )
    if args.num_inference_steps is not None and args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive")
    denoising_sigmas = validate_denoising_sigmas(args.denoising_sigmas)
    if denoising_sigmas is not None:
        explicit_step_count = len(denoising_sigmas) - 1
        if (
            args.num_inference_steps is not None
            and args.num_inference_steps != explicit_step_count
        ):
            raise ValueError(
                "--num-inference-steps does not match --denoising-sigmas: "
                f"{args.num_inference_steps} != {explicit_step_count}"
            )
        inference_steps_override = explicit_step_count
    else:
        inference_steps_override = args.num_inference_steps
    if denoising_update_mode == "consistency_renoise":
        from ltx_trainer.causal_rcm import FOUR_STEP_SIGMAS

        if denoising_sigmas != FOUR_STEP_SIGMAS:
            raise ValueError(
                "The isolated Causal-rCM evaluator requires exactly "
                f"--denoising-sigmas {' '.join(str(value) for value in FOUR_STEP_SIGMAS)}."
            )
        if args.guidance_mode != "conditional_only":
            raise ValueError(
                "The isolated Causal-rCM evaluator requires --guidance-mode conditional_only."
            )
        if (
            args.use_kv_cache and not allow_kv_cache_with_consistency
        ) or args.history_blocks is not None:
            raise ValueError(
                "The initial Causal-rCM evaluator is full-prefix only; KV cache and sliding history "
                "remain separate experiments."
            )
    if args.verify_kv_cache_first_step and not args.use_kv_cache:
        raise ValueError("--verify-kv-cache-first-step requires --use-kv-cache")
    if args.kv_cache_attention_backend != "native" and not args.use_kv_cache:
        raise ValueError("--kv-cache-attention-backend flex requires --use-kv-cache")
    if args.fa4_mask_execution != "native_mask_mod" and args.attention_backend not in (
        None,
        "flash_attention_4",
    ):
        raise ValueError(
            "--fa4-mask-execution segmented_exact requires FlashAttention4"
        )
    if args.verify_kv_cache_projected_kv and not args.verify_kv_cache_first_step:
        raise ValueError(
            "--verify-kv-cache-projected-kv requires --verify-kv-cache-first-step"
        )
    if args.verify_kv_cache_sublayers and not args.verify_kv_cache_first_step:
        raise ValueError(
            "--verify-kv-cache-sublayers requires --verify-kv-cache-first-step"
        )
    if args.verify_kv_cache_cold_recompute and not args.use_kv_cache:
        raise ValueError("--verify-kv-cache-cold-recompute requires --use-kv-cache")
    if args.kv_cache_full_prefix_interval < 0:
        raise ValueError("--kv-cache-full-prefix-interval must be non-negative")
    if args.kv_cache_full_prefix_interval > 0 and not args.use_kv_cache:
        raise ValueError("--kv-cache-full-prefix-interval requires --use-kv-cache")
    if args.kv_cache_full_prefix_interval > 0 and args.verify_kv_cache_first_step:
        raise ValueError(
            "--kv-cache-full-prefix-interval and --verify-kv-cache-first-step are incompatible"
        )
    if (
        args.kv_cache_full_prefix_interval > 0
        and args.kv_cache_transformer_dtype == "fp16"
    ):
        raise ValueError(
            "--kv-cache-full-prefix-interval requires the BF16 Transformer because full-prefix FP16 is unstable"
        )
    if args.kv_cache_transformer_dtype == "fp16" and not args.use_kv_cache:
        raise ValueError("--kv-cache-transformer-dtype fp16 requires --use-kv-cache")
    if args.kv_cache_transformer_dtype == "fp16" and args.dtype != "bf16":
        raise ValueError("--kv-cache-transformer-dtype fp16 requires --dtype bf16")
    if args.history_blocks is not None and args.history_blocks < 0:
        raise ValueError("--history-blocks must be non-negative")
    if args.history_blocks is not None and args.use_kv_cache:
        raise ValueError(
            "--history-blocks and --use-kv-cache are separate, incompatible ablations"
        )
    if args.history_blocks is not None and args.mask_backend not in (None, "dense"):
        raise ValueError(
            "Sliding-history inference requires the dense structural-mask backend"
        )
    if args.vlm_two_hit_mode != "posthoc" and not args.vlm_two_hit_kill:
        raise ValueError("--vlm-two-hit-mode requires --vlm-two-hit-kill")
    if (
        args.vlm_two_hit_kill
        and args.vlm_two_hit_mode == "pre_action_verified"
        and args.vlm_joystick_pixel_lookup is None
    ):
        raise ValueError("pre_action_verified requires --vlm-joystick-pixel-lookup")
    if args.vlm_joystick_pixel_lookup is not None and not args.vlm_two_hit_kill:
        raise ValueError("--vlm-joystick-pixel-lookup requires --vlm-two-hit-kill")
    if args.vlm_direct_crosshair_crop_size is not None:
        if args.vlm_two_hit_mode != "pre_action_verified":
            raise ValueError(
                "--vlm-direct-crosshair-crop-size requires pre_action_verified"
            )
        if args.vlm_direct_crosshair_crop_size <= 0:
            raise ValueError("--vlm-direct-crosshair-crop-size must be positive")
        if args.vlm_direct_crosshair_crop_size > min(args.width, args.height):
            raise ValueError(
                "--vlm-direct-crosshair-crop-size exceeds the output frame"
            )
        if args.vlm_pre_action_grounding:
            raise ValueError(
                "Direct-crosshair V2 and bounding-box grounding are mutually exclusive"
            )
    if args.vlm_aim_target_ring_v9:
        if not args.vlm_two_hit_kill:
            raise ValueError("--vlm-aim-target-ring-v9 requires --vlm-two-hit-kill")
        if args.vlm_two_hit_mode != "pre_action_verified":
            raise ValueError("--vlm-aim-target-ring-v9 requires pre_action_verified")
        if args.vlm_direct_crosshair_crop_size is None:
            raise ValueError(
                "--vlm-aim-target-ring-v9 requires --vlm-direct-crosshair-crop-size"
            )
    if args.vlm_freeze_opponent_motion and not args.vlm_two_hit_kill:
        raise ValueError("--vlm-freeze-opponent-motion requires --vlm-two-hit-kill")
    if args.vlm_duel_loop:
        if not args.vlm_two_hit_kill:
            raise ValueError("--vlm-duel-loop requires --vlm-two-hit-kill")
        if args.vlm_two_hit_mode != "pre_action_verified":
            raise ValueError(
                "--vlm-duel-loop requires --vlm-two-hit-mode pre_action_verified"
            )
        if args.vlm_freeze_opponent_motion:
            raise ValueError(
                "--vlm-duel-loop and --vlm-freeze-opponent-motion are incompatible"
            )
        if args.vlm_duel_respawn_seconds <= 0:
            raise ValueError("--vlm-duel-respawn-seconds must be positive")
        if args.vlm_duel_player_initial_hp <= 0:
            raise ValueError("--vlm-duel-player-initial-hp must be positive")
        if args.vlm_duel_enemy_shot_damage <= 0:
            raise ValueError("--vlm-duel-enemy-shot-damage must be positive")
    if args.vlm_deferred_healthcheck_timeout_seconds < 0:
        raise ValueError(
            "--vlm-deferred-healthcheck-timeout-seconds cannot be negative"
        )
    if args.vlm_action_controller:
        controller_magnitudes = {
            "move": args.vlm_action_move_magnitude,
            "strafe": args.vlm_action_strafe_magnitude,
            "turn": args.vlm_action_turn_magnitude,
            "aim": args.vlm_action_aim_magnitude,
            "vertical_aim": args.vlm_action_vertical_aim_magnitude,
        }
        invalid = {
            name: value
            for name, value in controller_magnitudes.items()
            if not 0.0 < value <= 1.0
        }
        if invalid:
            raise ValueError(
                f"VLM action-controller magnitudes must lie in (0,1]: {invalid}"
            )
        if not 0 <= args.vlm_action_fire_pulse_offset <= 15:
            raise ValueError("--vlm-action-fire-pulse-offset must lie in [0,15]")
    if not 0.0 <= args.vlm_min_hit_confidence <= 1.0:
        raise ValueError("--vlm-min-hit-confidence must be in [0, 1]")
    if not 1 <= args.vlm_pre_action_votes <= 5:
        raise ValueError("--vlm-pre-action-votes must be in [1, 5]")
    if args.vlm_required_post_shot_frames < 1:
        raise ValueError("--vlm-required-post-shot-frames must be positive")
    if args.vlm_initial_hp < 1:
        raise ValueError("--vlm-initial-hp must be positive")
    if args.vlm_confirmed_hit_damage < 1:
        raise ValueError("--vlm-confirmed-hit-damage must be positive")
    geometry_contract = classify_xonotic_ar_geometry(
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
        allow_bootstrap_prefix_h4=args.allow_bootstrap_prefix_h4,
        allow_experimental_long_horizon=args.allow_experimental_long_horizon,
    )
    bootstrap_prefix_h4_geometry = geometry_contract == "bootstrap_prefix_h4_65f"
    if bootstrap_prefix_h4_geometry and (
        args.use_kv_cache or args.history_blocks is not None
    ):
        raise ValueError(
            "--allow-bootstrap-prefix-h4 requires full-prefix inference without "
            "--use-kv-cache or --history-blocks"
        )
    if not args.config.exists() or not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Missing config/checkpoint: {args.config}, {args.checkpoint}"
        )

    raw_config, training_config = load_training_config(args.config.resolve())
    if training_config.training_strategy.name != "ar_teacher":
        raise ValueError(
            f"Expected ar_teacher config, got {training_config.training_strategy.name}"
        )
    base_model = Path(training_config.model.model_path).resolve()
    mask_backend_override = (
        args.mask_backend
        if args.mask_backend is not None
        else "dense"
        if args.history_blocks is not None
        else None
    )
    effective_mask_backend = (
        mask_backend_override
        if mask_backend_override is not None
        else training_config.training_strategy.mask_backend
    )
    effective_attention_backend = (
        args.attention_backend or training_config.model.attention_backend
    )
    prompt_execution_mode = training_config.training_strategy.prompt_execution_mode
    if prompt_execution_mode == "compact_current":
        if effective_attention_backend != "flash_attention_4":
            raise ValueError(
                "compact_current evaluation is locked to fail-closed flash_attention_4"
            )
        if not args.use_kv_cache:
            raise ValueError("compact_current evaluation requires --use-kv-cache")
        if args.kv_cache_full_prefix_interval > 0:
            raise ValueError(
                "compact_current evaluation forbids masked full-prefix anchor steps"
            )
    num_inference_steps, video_guider_params, audio_guider_params = (
        resolve_inference_settings(
            training_config=training_config,
            base_model=base_model,
            num_inference_steps_override=inference_steps_override,
        )
    )
    if args.guidance_mode == "conditional_only":
        video_guider_params = MultiModalGuiderParams()
        audio_guider_params = MultiModalGuiderParams()
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
    generated_blocks = (
        args.num_frames // 8 + 1 - 1
    ) // training_config.training_strategy.chunk_size

    cases: list[dict[str, Any]] = []
    if args.only in {"both", "heldout"}:
        heldout_rows = read_jsonl(args.heldout_manifest.resolve())
        cases.append(
            build_case(
                row=heldout_rows[args.heldout_index],
                case_id=f"heldout_static_index{args.heldout_index:04d}",
                use_variable_schedule=False,
                num_generated_blocks=generated_blocks,
                seed=args.seed,
            )
        )
    if args.only in {"both", "variable"}:
        variable_rows = read_jsonl(args.variable_manifest.resolve())
        variable_indices = args.variable_indices or [2]
        if len(variable_indices) != len(set(variable_indices)):
            raise ValueError(f"Duplicate --variable-index values: {variable_indices}")
        for case_offset, variable_index in enumerate(variable_indices, start=1):
            if not 0 <= variable_index < len(variable_rows):
                raise IndexError(
                    f"--variable-index {variable_index} is outside manifest range "
                    f"[0, {len(variable_rows) - 1}]"
                )
            cases.append(
                build_case(
                    row=variable_rows[variable_index],
                    case_id=f"train_variable_index{variable_index:04d}",
                    use_variable_schedule=True,
                    num_generated_blocks=generated_blocks,
                    seed=args.seed + case_offset,
                )
            )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (DEFAULT_RUN / "eval_ar_teacher" / f"step_00005000_{timestamp}").resolve()
    )
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing eval output: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    case_summary = [
        {
            "case_id": case["case_id"],
            "sample_id": case["row"].get("sample_id"),
            "clip_id": case["row"].get("clip_id"),
            "game": case["row"].get("game"),
            "is_variable_prompt": case["is_variable_prompt"],
            "is_custom_image_case": case["is_custom_image_case"],
            "prompt_schedule_unique_count": len(set(case["prompt_schedule"])),
            "seed": case["seed"],
        }
        for case in cases
    ]
    run_info = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "base_model": str(base_model),
        "text_encoder": str(Path(training_config.model.text_encoder_path).resolve()),
        "evaluation_semantics": (
            "blockwise_hybrid_kv_cache_full_prefix_anchors"
            if args.use_kv_cache and args.kv_cache_full_prefix_interval > 0
            else "blockwise_incremental_kv_cache"
            if args.use_kv_cache
            else "blockwise_cropped_sliding_history_rerun"
            if args.history_blocks is not None
            else "blockwise_full_prefix_rerun"
        ),
        "kv_cache_enabled": args.use_kv_cache,
        "kv_cache_attention_backend": args.kv_cache_attention_backend,
        "kv_cache_full_prefix_interval": args.kv_cache_full_prefix_interval,
        "verify_kv_cache_first_step": args.verify_kv_cache_first_step,
        "verify_kv_cache_projected_kv": args.verify_kv_cache_projected_kv,
        "verify_kv_cache_sublayers": args.verify_kv_cache_sublayers,
        "verify_kv_cache_cold_recompute": args.verify_kv_cache_cold_recompute,
        "component_dtype": args.dtype,
        "low_host_memory": args.low_host_memory,
        "transformer_dtype": (
            args.dtype
            if args.kv_cache_transformer_dtype == "match"
            else args.kv_cache_transformer_dtype
        ),
        "history_generated_blocks": args.history_blocks,
        "mask_backend_requested": args.mask_backend,
        "mask_backend_effective": effective_mask_backend,
        "attention_backend_requested": args.attention_backend,
        "attention_backend_effective": effective_attention_backend,
        "fa4_mask_execution": args.fa4_mask_execution,
        "prompt_execution_mode": prompt_execution_mode,
        "geometry_contract": geometry_contract,
        "bootstrap_prefix_h4": bootstrap_prefix_h4_geometry,
        "experimental_long_horizon": geometry_contract == "experimental_long_horizon",
        "chunk_size": training_config.training_strategy.chunk_size,
        "num_inference_steps": num_inference_steps,
        "guidance_mode": args.guidance_mode,
        "video_guider_params": asdict(video_guider_params),
        "audio_guider_params": asdict(audio_guider_params),
        "action_guidance_scale": 0.0,
        "joystick_guidance_scale": 0.0,
        "action_button_temporal_alignment": training_config.training_strategy.action_button_temporal_alignment,
        "guidance_passes": required_guidance_passes(
            video_guider_params, audio_guider_params
        ),
        "denoising_sigmas": list(denoising_sigmas)
        if denoising_sigmas is not None
        else None,
        "sigma_schedule_source": "explicit"
        if denoising_sigmas is not None
        else "ltx2_scheduler",
        "scheduler_timestep_shift": args.scheduler_timestep_shift,
        "vae_decoding_mode": args.vae_decoding_mode,
        "save_latent_cache": args.save_latent_cache,
        "vlm_two_hit_kill": {
            "enabled": args.vlm_two_hit_kill,
            "decision_mode": args.vlm_two_hit_mode if args.vlm_two_hit_kill else None,
            "base_url": args.vlm_base_url if args.vlm_two_hit_kill else None,
            "model": args.vlm_model if args.vlm_two_hit_kill else None,
            "initial_hp": args.vlm_initial_hp if args.vlm_two_hit_kill else None,
            "confirmed_hit_damage": (
                args.vlm_confirmed_hit_damage if args.vlm_two_hit_kill else None
            ),
            "min_hit_confidence": (
                args.vlm_min_hit_confidence if args.vlm_two_hit_kill else None
            ),
            "required_post_shot_frames": (
                args.vlm_required_post_shot_frames if args.vlm_two_hit_kill else None
            ),
            "pre_action_votes": (
                args.vlm_pre_action_votes if args.vlm_two_hit_kill else None
            ),
            "pre_action_grounding": (
                args.vlm_pre_action_grounding if args.vlm_two_hit_kill else None
            ),
            "direct_crosshair_crop_size": (
                args.vlm_direct_crosshair_crop_size if args.vlm_two_hit_kill else None
            ),
            "aim_target_ring_v9": (
                args.vlm_aim_target_ring_v9 if args.vlm_two_hit_kill else None
            ),
            "aim_target_prompt": (
                AIM_TARGET_RING_NEUTRAL_PROMPT
                if args.vlm_two_hit_kill and args.vlm_aim_target_ring_v9
                else None
            ),
            "joystick_pixel_lookup": (
                str(args.vlm_joystick_pixel_lookup.resolve())
                if args.vlm_two_hit_kill and args.vlm_joystick_pixel_lookup is not None
                else None
            ),
            "enemy_behavior_policy": (
                "duel_prompt_rule_approach_seeded_strafe_respawn"
                if args.vlm_two_hit_kill and args.vlm_duel_loop
                else "stationary_visible"
                if args.vlm_two_hit_kill and args.vlm_freeze_opponent_motion
                else "manifest_schedule"
                if args.vlm_two_hit_kill
                else None
            ),
            "duel_loop": args.vlm_duel_loop if args.vlm_two_hit_kill else False,
            "duel_respawn_seconds": (
                args.vlm_duel_respawn_seconds if args.vlm_duel_loop else None
            ),
            "duel_player_initial_hp": (
                args.vlm_duel_player_initial_hp if args.vlm_duel_loop else None
            ),
            "duel_enemy_shot_damage": (
                args.vlm_duel_enemy_shot_damage if args.vlm_duel_loop else None
            ),
            "deferred_healthcheck_timeout_seconds": (
                args.vlm_deferred_healthcheck_timeout_seconds
            ),
        },
        "vlm_action_controller": {
            "enabled": args.vlm_action_controller,
            "base_url": args.vlm_base_url if args.vlm_action_controller else None,
            "model": args.vlm_model if args.vlm_action_controller else None,
            "move_magnitude": (
                args.vlm_action_move_magnitude if args.vlm_action_controller else None
            ),
            "strafe_magnitude": (
                args.vlm_action_strafe_magnitude if args.vlm_action_controller else None
            ),
            "turn_magnitude": (
                args.vlm_action_turn_magnitude if args.vlm_action_controller else None
            ),
            "aim_magnitude": (
                args.vlm_action_aim_magnitude if args.vlm_action_controller else None
            ),
            "vertical_aim_magnitude": (
                args.vlm_action_vertical_aim_magnitude
                if args.vlm_action_controller
                else None
            ),
            "fire_pulse_offset": (
                args.vlm_action_fire_pulse_offset
                if args.vlm_action_controller
                else None
            ),
            "policy": (
                "vlm_categorical_navigation_python_bounded_controls_v1"
                if args.vlm_action_controller
                else None
            ),
        },
        "tiled_decoding": asdict(tiled_decoding)
        if isinstance(tiled_decoding, TiledDecodingConfig)
        else tiled_decoding,
        "geometry": {
            "width": args.width,
            "height": args.height,
            "num_frames": args.num_frames,
            "fps": args.fps,
        },
        "cases": case_summary,
        "raw_training_run_name": raw_config.get("run_name"),
    }
    if denoising_update_mode != "euler":
        run_info["denoising_update_mode"] = denoising_update_mode
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_info, indent=2, ensure_ascii=False), flush=True)
    print(f"[ar-eval] output={output_dir}", flush=True)
    if args.cases_only:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("AR evaluation requires CUDA")
    device = torch.device(args.device)
    component_dtype = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    transformer_dtype = (
        torch.float16 if args.kv_cache_transformer_dtype == "fp16" else component_dtype
    )
    if effective_attention_backend in {
        "flash_attention_2",
        "flash_attention_3",
        "flash_attention_4",
    } and transformer_dtype not in {
        torch.float16,
        torch.bfloat16,
    }:
        raise ValueError(
            f"{effective_attention_backend} evaluation requires a BF16 or FP16 transformer dtype"
        )
    components, embeddings_processor, model_button_dim = load_eval_components(
        base_model=base_model,
        text_encoder=Path(training_config.model.text_encoder_path).resolve(),
        checkpoint=args.checkpoint.resolve(),
        dtype=component_dtype,
        with_audio=True,
        transformer_device_after_load=device if args.low_host_memory else None,
    )
    configured_attention_modules = configure_attention_backend(
        components.transformer,
        effective_attention_backend,
    )
    if configured_attention_modules == 0:
        raise RuntimeError(
            "Loaded transformer contains no configurable Attention modules"
        )
    if args.fa4_mask_execution == "segmented_exact":
        if effective_attention_backend != "flash_attention_4":
            raise ValueError(
                "--fa4-mask-execution segmented_exact resolved a non-FA4 backend: "
                f"{effective_attention_backend}"
            )
        segmented_modules = configure_exact_segmented_fa4(components.transformer)
        if segmented_modules != configured_attention_modules:
            raise RuntimeError(
                "Segmented FA4 did not replace every configured attention module: "
                f"segmented={segmented_modules} configured={configured_attention_modules}"
            )
    print(
        f"[ar-eval] attention={effective_attention_backend}; "
        f"fa4_mask_execution={args.fa4_mask_execution}; modules={configured_attention_modules}",
        flush=True,
    )
    expected_button_dim = 6
    if model_button_dim != expected_button_dim:
        raise ValueError(
            f"AR C25k contract expects {expected_button_dim} raw buttons, checkpoint has {model_button_dim}"
        )
    sampler = ValidationSampler(
        transformer=components.transformer,
        vae_decoder=components.video_vae_decoder,
        vae_encoder=components.video_vae_encoder,
        text_encoder=components.text_encoder,
        audio_decoder=components.audio_vae_decoder,
        vocoder=components.vocoder,
        embeddings_processor=embeddings_processor,
    )
    if components.vocoder is None:
        raise RuntimeError("Audio eval requested but the model has no vocoder")
    generated_audio_sample_rate = int(components.vocoder.output_sampling_rate)
    strategy = make_strategy(
        training_config,
        mask_backend_override=mask_backend_override,
    )
    vlm_observer: OpenAIVLMObserver | None = None
    if args.vlm_two_hit_kill or args.vlm_action_controller:
        vlm_observer = OpenAIVLMObserver(
            base_url=args.vlm_base_url,
            model=args.vlm_model,
        )
        if args.vlm_deferred_healthcheck_timeout_seconds == 0:
            vlm_observer.healthcheck()
        else:
            print(
                "[vlm] deferring healthcheck until the video model is on device",
                flush=True,
            )
    vlm_two_hit_config: VLMTwoHitConfig | None = None
    if args.vlm_two_hit_kill:
        if vlm_observer is None:
            raise RuntimeError("VLM observer initialization failed")
        joystick_pixel_lookup = (
            JoystickPixelLookup.from_path(args.vlm_joystick_pixel_lookup)
            if args.vlm_joystick_pixel_lookup is not None
            else None
        )
        if joystick_pixel_lookup is not None:
            joystick_pixel_lookup.validate_geometry(
                width=args.width,
                height=args.height,
                fps=args.fps,
            )
            if args.vlm_action_controller:
                controller_axis_max = (
                    args.vlm_action_strafe_magnitude,
                    args.vlm_action_move_magnitude,
                    max(
                        args.vlm_action_turn_magnitude,
                        1.5 * args.vlm_action_aim_magnitude,
                    ),
                    args.vlm_action_vertical_aim_magnitude,
                )
                if joystick_pixel_lookup.linear_abs_ranges is not None:
                    violations = {
                        axis: (requested, calibrated)
                        for axis, requested, calibrated in zip(
                            ("left_x", "left_y", "right_x", "right_y"),
                            controller_axis_max,
                            joystick_pixel_lookup.linear_abs_ranges,
                            strict=True,
                        )
                        if requested > calibrated + 1e-7
                    }
                else:
                    violations = {}
                    for axis, requested, points in (
                        (
                            "right_x",
                            controller_axis_max[2],
                            joystick_pixel_lookup.right_x_points,
                        ),
                        (
                            "right_y",
                            controller_axis_max[3],
                            joystick_pixel_lookup.right_y_points,
                        ),
                    ):
                        if points and requested > points[-1][0] + 1e-7:
                            violations[axis] = (requested, points[-1][0])
                if violations:
                    raise ValueError(
                        "VLM action-controller amplitudes exceed the fail-closed "
                        f"joystick pixel calibration: {violations}"
                    )
        vlm_two_hit_config = VLMTwoHitConfig(
            observer=vlm_observer,
            min_hit_confidence=args.vlm_min_hit_confidence,
            required_post_shot_frames=args.vlm_required_post_shot_frames,
            decision_mode=args.vlm_two_hit_mode,
            initial_hp=args.vlm_initial_hp,
            confirmed_hit_damage=args.vlm_confirmed_hit_damage,
            pre_action_votes=args.vlm_pre_action_votes,
            pre_action_grounding=args.vlm_pre_action_grounding,
            joystick_pixel_lookup=joystick_pixel_lookup,
            freeze_opponent_motion=args.vlm_freeze_opponent_motion,
            direct_crosshair_crop_size=args.vlm_direct_crosshair_crop_size,
            aim_target_ring_v9=args.vlm_aim_target_ring_v9,
            duel_loop=args.vlm_duel_loop,
            duel_respawn_seconds=args.vlm_duel_respawn_seconds,
            duel_player_initial_hp=args.vlm_duel_player_initial_hp,
            duel_enemy_shot_damage=args.vlm_duel_enemy_shot_damage,
            deferred_healthcheck_timeout_seconds=(
                args.vlm_deferred_healthcheck_timeout_seconds
            ),
        )
    vlm_action_controller_config: VLMActionControllerConfig | None = None
    if args.vlm_action_controller:
        if vlm_observer is None:
            raise RuntimeError("VLM observer initialization failed")
        vlm_action_controller_config = VLMActionControllerConfig(
            observer=vlm_observer,
            move_magnitude=args.vlm_action_move_magnitude,
            strafe_magnitude=args.vlm_action_strafe_magnitude,
            turn_magnitude=args.vlm_action_turn_magnitude,
            aim_magnitude=args.vlm_action_aim_magnitude,
            vertical_aim_magnitude=args.vlm_action_vertical_aim_magnitude,
            fire_pulse_offset=args.vlm_action_fire_pulse_offset,
            deferred_healthcheck_timeout_seconds=(
                args.vlm_deferred_healthcheck_timeout_seconds
            ),
        )

    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            results.append(
                generate_case(
                    case=case,
                    sampler=sampler,
                    strategy=strategy,
                    transformer=components.transformer,
                    training_config=training_config,
                    device=device,
                    dtype=transformer_dtype,
                    component_dtype=component_dtype,
                    num_inference_steps=num_inference_steps,
                    video_guider_params=video_guider_params,
                    audio_guider_params=audio_guider_params,
                    scheduler_timestep_shift=args.scheduler_timestep_shift,
                    denoising_sigmas=denoising_sigmas,
                    guidance_mode=args.guidance_mode,
                    width=args.width,
                    height=args.height,
                    num_frames=args.num_frames,
                    fps=args.fps,
                    output_dir=output_dir,
                    generated_audio_sample_rate=generated_audio_sample_rate,
                    tiled_decoding=tiled_decoding,
                    save_latent_cache=args.save_latent_cache,
                    use_kv_cache=args.use_kv_cache,
                    kv_cache_attention_backend=args.kv_cache_attention_backend,
                    kv_cache_full_prefix_interval=args.kv_cache_full_prefix_interval,
                    verify_kv_cache_first_step=args.verify_kv_cache_first_step,
                    verify_kv_cache_projected_kv=args.verify_kv_cache_projected_kv,
                    verify_kv_cache_sublayers=args.verify_kv_cache_sublayers,
                    verify_kv_cache_cold_recompute=args.verify_kv_cache_cold_recompute,
                    history_blocks=args.history_blocks,
                    vlm_two_hit_config=vlm_two_hit_config,
                    vlm_action_controller_config=vlm_action_controller_config,
                    denoising_update_mode=denoising_update_mode,
                )
            )
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[ar-eval] done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
