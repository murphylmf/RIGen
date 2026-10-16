from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


JOYSTICK_CHANNEL_ORDER: tuple[str, ...] = (
    "j_left_x",
    "j_left_y",
    "j_right_x",
    "j_right_y",
)


@dataclass(frozen=True)
class JoystickUnitScaleProfile:
    profile_id: str
    scales: tuple[float, float, float, float]
    path: Path
    payload: dict[str, Any]


def load_joystick_unit_scale_profile(path: str | Path) -> JoystickUnitScaleProfile:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Joystick normalization profile does not exist: {resolved_path}")

    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "joystick_unit_to_model_scale_v1":
        raise ValueError(
            "Unsupported joystick normalization profile schema: "
            f"{payload.get('schema_version')!r} in {resolved_path}"
        )
    channel_order = tuple(payload.get("channel_order") or ())
    if channel_order != JOYSTICK_CHANNEL_ORDER:
        raise ValueError(
            f"Joystick profile channel order must be {JOYSTICK_CHANNEL_ORDER}, got {channel_order}"
        )
    if payload.get("operation") != "model_input = normalized_unit * scale":
        raise ValueError(f"Unsupported joystick profile operation in {resolved_path}")

    raw_scales = payload.get("scales")
    if not isinstance(raw_scales, list) or len(raw_scales) != len(JOYSTICK_CHANNEL_ORDER):
        raise ValueError(f"Joystick profile scales must contain four values: {resolved_path}")
    scales = (
        float(raw_scales[0]),
        float(raw_scales[1]),
        float(raw_scales[2]),
        float(raw_scales[3]),
    )
    if not all(math.isfinite(value) and value > 0.0 for value in scales):
        raise ValueError(f"Joystick profile scales must be finite and positive: {scales}")

    profile_id = str(payload.get("profile_id") or "").strip()
    if not profile_id:
        raise ValueError(f"Joystick profile_id must be non-empty: {resolved_path}")
    return JoystickUnitScaleProfile(
        profile_id=profile_id,
        scales=scales,
        path=resolved_path,
        payload=payload,
    )


def normalized_unit_joystick_to_model_input(
    joystick: Tensor,
    profile: JoystickUnitScaleProfile,
    *,
    range_tolerance: float = 1e-6,
) -> Tensor:
    if joystick.ndim < 1 or joystick.shape[-1] != len(JOYSTICK_CHANNEL_ORDER):
        raise ValueError(f"Joystick tensor must end in four channels, got {tuple(joystick.shape)}")
    if not torch.isfinite(joystick).all():
        raise ValueError("Normalized-unit joystick tensor contains non-finite values")
    max_abs = float(joystick.detach().abs().max().item()) if joystick.numel() else 0.0
    if max_abs > 1.0 + float(range_tolerance):
        raise ValueError(
            "Normalized-unit joystick values must stay in [-1, 1] without clipping; "
            f"observed abs max {max_abs:.8f}"
        )

    scales = torch.tensor(profile.scales, device=joystick.device, dtype=joystick.dtype)
    return joystick * scales
