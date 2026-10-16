''

from __future__ import annotations

import ctypes
import gc
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from ltx_core.loader import (
    LoraPathStrengthAndSDOps,
    LoraStateDictWithStrength,
    SafetensorsStateDictLoader,
    StateDict,
)
from ltx_core.loader.fuse_loras import fuse_lora_weights
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.model.transformer.model_configurator import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
)

ACTION_TEMPORAL_COMPRESSION = 8
JOYSTICK_CHANNELS = 4
JOYSTICK_LATENT_DIM = ACTION_TEMPORAL_COMPRESSION * JOYSTICK_CHANNELS


@dataclass(frozen=True)
class ActionCheckpointLoadReport:
    button_dim: int
    button_state_dim: int
    checkpoint_prefix: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    fused_lora_paths: tuple[str, ...]


@dataclass(frozen=True)
class PreparedActionConditioning:
    joystick_tokens: Tensor
    joystick_condition_mask: Tensor
    button_state: Tensor
    button_values: Tensor
    button_condition_mask: Tensor
    prefix_tokens: int
    frame_rate: float


def infer_action_button_dim(state: dict[str, Tensor]) -> int | None:
    pattern = re.compile(r"(?:^|.*\.)transformer_blocks\.0\.action_button_attn\.to_v\.weight$")
    for key, value in state.items():
        if pattern.match(key) and torch.is_tensor(value) and value.ndim == 2:
            return int(value.shape[1])
    return None


def infer_action_button_state_dim(state: dict[str, Tensor]) -> int | None:
    pattern = re.compile(r"(?:^|.*\.)transformer_blocks\.0\.action_button_attn\.to_k\.weight$")
    for key, value in state.items():
        if pattern.match(key) and torch.is_tensor(value) and value.ndim == 2:
            return int(value.shape[1])
    return None


def _strip_best_checkpoint_prefix(
    state: dict[str, Tensor],
    model_keys: set[str],
) -> tuple[dict[str, Tensor], str]:
    best_prefix = ""
    best_score: tuple[int, int] | None = None
    for prefix in ("", "module.", "_orig_mod.", "diffusion_model.", "transformer."):
        candidate_keys = {
            key[len(prefix) :] if prefix and key.startswith(prefix) else key
            for key in state
        }
        missing_count = len(model_keys - candidate_keys)
        unexpected_count = len(candidate_keys - model_keys)
        score = (missing_count + unexpected_count, unexpected_count)
        if best_score is None or score < best_score:
            best_score = score
            best_prefix = prefix

    if not best_prefix:
        return state, ""
    stripped = {
        key[len(best_prefix) :] if key.startswith(best_prefix) else key: value
        for key, value in state.items()
    }
    state.clear()
    return stripped, best_prefix


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
            madvise(
                ctypes.c_void_p(start),
                ctypes.c_size_t(end - start),
                ctypes.c_int(4),                       
            )
    except (OSError, ValueError):
        return


def _load_action_checkpoint(path: Path) -> tuple[dict[str, Tensor], bool]:
    try:
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(state, dict):
            raise TypeError(f"Expected a state dict, got {type(state).__name__}")
        return state, True
    except Exception as torch_load_error:
        try:
            state = load_file(str(path), device="cpu")
        except Exception as safetensors_error:
            raise RuntimeError(
                f"Could not load action checkpoint {path}: "
                f"torch.load={torch_load_error}; safetensors={safetensors_error}"
            ) from safetensors_error
        return state, False


def inspect_action_checkpoint(path: str | Path) -> tuple[int, int, int]:
    ''

    checkpoint = Path(path).expanduser().resolve()
    state, is_mmap = _load_action_checkpoint(checkpoint)
    button_dim = infer_action_button_dim(state)
    state_dim = infer_action_button_state_dim(state)
    tensor_count = len(state)
    del state
    gc.collect()
    if is_mmap:
        _discard_file_backed_pages(checkpoint)
    if button_dim is None or state_dim is None:
        raise ValueError(f"Checkpoint has no C action-button attention weights: {checkpoint}")
    return button_dim, state_dim, tensor_count


def _stream_checkpoint_into_module(
    module: nn.Module,
    state: dict[str, Tensor],
) -> tuple[list[str], list[str]]:
    ''

    target_state = module.state_dict()
    missing = [key for key in target_state if key not in state]
    unexpected = [key for key in state if key not in target_state]
    shape_mismatches: list[str] = []
    with torch.no_grad():
        for key, target in target_state.items():
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
    if shape_mismatches:
        raise RuntimeError(
            "Action checkpoint tensor shape mismatch: " + "; ".join(shape_mismatches[:8])
        )
    state.clear()
    return missing, unexpected


def _fuse_loras_in_place(
    module: nn.Module,
    loras: tuple[LoraPathStrengthAndSDOps, ...],
    *,
    dtype: torch.dtype,
) -> None:
    active_loras = tuple(lora for lora in loras if float(lora.strength) != 0.0)
    if not active_loras:
        return

    loader = SafetensorsStateDictLoader()
    loaded = [
        loader.load(
            lora.path,
            sd_ops=lora.sd_ops,
            device=torch.device("cpu"),
        )
        for lora in active_loras
    ]
    weighted = [
        LoraStateDictWithStrength(state_dict, float(lora.strength))
        for state_dict, lora in zip(loaded, active_loras, strict=True)
    ]
    target_state = module.state_dict()
    state_view = StateDict(
        sd=target_state,
        device=next(module.parameters()).device,
        size=sum(tensor.nbytes for tensor in target_state.values()),
        dtype={tensor.dtype for tensor in target_state.values()},
    )
    fused_count = 0
    with torch.no_grad():
        for key, fused in fuse_lora_weights(state_view, weighted, dtype=dtype):
            target = target_state.get(key)
            if target is None:
                raise KeyError(f"LoRA fusion produced an unknown model key: {key}")
            target.copy_(fused)
            fused_count += 1
            del fused
    del weighted, loaded, target_state, state_view
    gc.collect()
    if fused_count == 0:
        raise RuntimeError(
            "Configured stage-2 LoRA(s) matched zero transformer weights: "
            + ", ".join(lora.path for lora in active_loras)
        )


def load_action_aware_transformer(
    *,
    base_model: str | Path,
    action_checkpoint: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    frozen_loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
) -> tuple[nn.Module, ActionCheckpointLoadReport]:
    ''

    base_model = Path(base_model).expanduser().resolve()
    action_checkpoint = Path(action_checkpoint).expanduser().resolve()
    transformer = SingleGPUModelBuilder(
        model_path=str(base_model),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    ).build(device=device, dtype=dtype)

    state, is_mmap = _load_action_checkpoint(action_checkpoint)
    state, prefix = _strip_best_checkpoint_prefix(
        state,
        set(transformer.state_dict()),
    )
    button_dim = infer_action_button_dim(state)
    state_dim = infer_action_button_state_dim(state)
    if button_dim is None or state_dim is None:
        raise ValueError(
            f"Full checkpoint does not contain C action attention: {action_checkpoint}"
        )
    if not hasattr(transformer, "set_action_button_dim"):
        raise TypeError("Loaded LTX transformer cannot configure action button dimensions")
    transformer.set_action_button_dim(button_dim, state_dim=state_dim)

    missing, unexpected = _stream_checkpoint_into_module(transformer, state)
    del state
    gc.collect()
    if is_mmap:
        _discard_file_backed_pages(action_checkpoint)

    optional_missing = {
        key
        for key in missing
        if key.startswith("action_button_meaning_proj.")
        or ".action_button_attn.meaning_concat_mlp." in key
    }
    required_missing = [key for key in missing if key not in optional_missing]
    total_keys = max(1, len(transformer.state_dict()))
    if len(required_missing) > max(4, int(total_keys * 0.02)):
        raise RuntimeError(
            "C checkpoint is not compatible with the current transformer: "
            f"required_missing={required_missing[:8]}, unexpected={unexpected[:8]}"
        )

    _fuse_loras_in_place(transformer, frozen_loras, dtype=dtype)
                                                                           
                                                                         
                                                                              
                                                                       
                                                                              
    transformer = transformer.to(device=device, dtype=dtype)
    unexpected_float_dtypes = {
        parameter.dtype
        for parameter in transformer.parameters()
        if parameter.is_floating_point() and parameter.dtype != dtype
    }
    if unexpected_float_dtypes:
        raise RuntimeError(
            "Action-aware transformer contains mixed floating dtypes after load: "
            f"expected={dtype}, unexpected={sorted(map(str, unexpected_float_dtypes))}"
        )
    report = ActionCheckpointLoadReport(
        button_dim=button_dim,
        button_state_dim=state_dim,
        checkpoint_prefix=prefix,
        missing_keys=tuple(missing),
        unexpected_keys=tuple(unexpected),
        fused_lora_paths=tuple(lora.path for lora in frozen_loras if float(lora.strength) != 0.0),
    )
    return transformer, report


def prepare_crossfps_action_conditioning(
    *,
    joystick: Tensor,
    buttons: Tensor,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    model_sequence_length: int,
    button_state_dim: int,
    frame_rate: float,
    device: torch.device,
    dtype: torch.dtype,
) -> PreparedActionConditioning:
    ''

    joystick = joystick.to(device=device, dtype=dtype)
    buttons = buttons.to(device=device, dtype=dtype)
    if joystick.ndim != 3 or joystick.shape[-1] != JOYSTICK_CHANNELS:
        raise ValueError(
            f"action_joystick must be [B,F,{JOYSTICK_CHANNELS}], got {tuple(joystick.shape)}"
        )
    if buttons.ndim != 3:
        raise ValueError(f"action_buttons must be [B,F,D], got {tuple(buttons.shape)}")
    if joystick.shape[:2] != buttons.shape[:2]:
        raise ValueError(
            f"Joystick/button frame mismatch: {tuple(joystick.shape)} vs {tuple(buttons.shape)}"
        )

    expected_raw_frames = (
        (latent_frames - 1) * ACTION_TEMPORAL_COMPRESSION + 1
    )
    if joystick.shape[1] != expected_raw_frames:
        raise ValueError(
            f"Action frame count {joystick.shape[1]} does not match latent_frames={latent_frames}; "
            f"expected {expected_raw_frames}"
        )
    batch_size = joystick.shape[0]
    latent_joystick = torch.zeros(
        batch_size,
        latent_frames,
        JOYSTICK_LATENT_DIM,
        device=device,
        dtype=dtype,
    )
    for latent_index in range(1, latent_frames):
        start = ACTION_TEMPORAL_COMPRESSION * (latent_index - 1) + 1
        end = ACTION_TEMPORAL_COMPRESSION * latent_index + 1
        latent_joystick[:, latent_index] = joystick[:, start:end].reshape(
            batch_size,
            JOYSTICK_LATENT_DIM,
        )

    target_tokens = latent_frames * latent_height * latent_width
    prefix_tokens = int(model_sequence_length) - target_tokens
    if prefix_tokens < 0:
        raise ValueError(
            f"Model sequence {model_sequence_length} is shorter than action target {target_tokens}"
        )
    joystick_tokens = (
        latent_joystick[:, :, None, None, :]
        .expand(
            batch_size,
            latent_frames,
            latent_height,
            latent_width,
            JOYSTICK_LATENT_DIM,
        )
        .reshape(batch_size, target_tokens, JOYSTICK_LATENT_DIM)
        .contiguous()
    )
    if prefix_tokens:
        joystick_tokens = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    prefix_tokens,
                    JOYSTICK_LATENT_DIM,
                    device=device,
                    dtype=dtype,
                ),
                joystick_tokens,
            ],
            dim=1,
        )

    active = buttons.abs().sum(dim=-1, keepdim=True) > 0
    idle = (~active).to(dtype=dtype)
    button_dim = int(buttons.shape[-1])
    if button_state_dim == button_dim + 1:
        button_state = torch.cat([buttons, idle], dim=-1).contiguous()
    elif button_state_dim == button_dim + 2:
        button_state = torch.cat(
            [buttons, active.to(dtype=dtype), idle],
            dim=-1,
        ).contiguous()
    else:
        raise ValueError(
            f"Unsupported action state layout: buttons={button_dim}, state={button_state_dim}"
        )
    button_values = torch.where(active, buttons, torch.zeros_like(buttons)).contiguous()
    present = torch.ones(batch_size, 1, 1, device=device, dtype=dtype)
    return PreparedActionConditioning(
        joystick_tokens=joystick_tokens,
        joystick_condition_mask=present,
        button_state=button_state,
        button_values=button_values,
        button_condition_mask=present.clone(),
        prefix_tokens=prefix_tokens,
        frame_rate=float(frame_rate),
    )
