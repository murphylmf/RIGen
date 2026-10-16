''

from __future__ import annotations

import base64
import bisect
import io
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from PIL import Image, ImageDraw


def _validate_navigation_payload(value, schema):
    ''
    if not isinstance(value, dict):
        raise ValueError('Navigation result must be an object')
    if set(value) != set(schema['required']):
        raise ValueError('Navigation fields do not match the required schema')
    for key, spec in schema['properties'].items():
        item = value[key]
        kind = spec['type']
        if kind == 'boolean' and type(item) is not bool:
            raise ValueError(f'{key} must be boolean')
        if kind == 'string' and not isinstance(item, str):
            raise ValueError(f'{key} must be string')
        if 'enum' in spec and item not in spec['enum']:
            raise ValueError(f'{key} is not an allowed value')
        if kind == 'number':
            if type(item) not in (int, float) or not math.isfinite(item):
                raise ValueError(f'{key} must be finite numeric')
            if not spec.get('minimum', -math.inf) <= item <= spec.get('maximum', math.inf):
                raise ValueError(f'{key} out of range')


InteractionMode = Literal[
    "active",
    "hit_alive",
    "miss_alive",
    "collapse",
    "persist_killed",
]
ShotResult = Literal["hit", "miss", "uncertain", "not_applicable"]
EnemyPosture = Literal["standing", "falling", "down", "not_visible", "uncertain"]
EnemySide = Literal[
    "far_left",
    "left",
    "center",
    "right",
    "far_right",
    "not_visible",
    "uncertain",
]
NavigationPathState = Literal["clear", "blocked", "corner", "uncertain"]
NavigationMovement = Literal[
    "forward",
    "strafe_left",
    "strafe_right",
    "backward",
]
NavigationTurn = Literal["left", "right", "none"]
EnemyVertical = Literal["above", "center", "below", "not_visible", "uncertain"]


ACTIVE_INTERACTION = "The tracked enemy remains active throughout this interval."
HIT_ALIVE_INTERACTION = (
    "One or more direct hits damage the tracked enemy during this interval. "
    "The tracked enemy remains active through the end of the interval."
)
MISS_ALIVE_INTERACTION = (
    "One or more attacks miss the tracked enemy during this interval. "
    "The tracked enemy remains active throughout the interval."
)
COLLAPSE_INTERACTION = (
    "The tracked enemy is killed during this interval, falls, and is lying on "
    "the ground by the end of the interval."
)
PERSIST_KILLED_INTERACTION = (
    "The tracked enemy remains killed and lying on the ground throughout this interval."
)

ACTIVE_OPPONENT_OUTCOME = "The opponent is not damaged during this block."
HIT_ALIVE_OPPONENT_OUTCOME = (
    "The opponent is hit during this block and remains alive."
)
MISS_ALIVE_OPPONENT_OUTCOME = ACTIVE_OPPONENT_OUTCOME
COLLAPSE_OPPONENT_OUTCOME = (
    "The opponent is killed during this block, is visibly knocked to the ground, and "
    "remains dead through the end of the block."
)
PERSIST_KILLED_OPPONENT_OUTCOME = (
    "The opponent starts this block dead and remains dead throughout the block."
)
PLAYER_KILLED_OUTCOME = "The opponent kills the player during this block."

_INTERACTION_SECTION = re.compile(
    r"(?ms)(^\[CURRENT INTERACTION CONDITION\]\n).*?(?=\n\[[A-Z][A-Z _]+\]\n|\Z)"
)
_OPPONENT_MOTION_OUTCOME_SECTION = re.compile(
    r"(?ms)(^\[CURRENT OPPONENT MOTION AND OUTCOME\]\n)(.*?)(?=\n\[[A-Z][A-Z _]+\]\n|\Z)"
)
_PROMPT_SECTION = re.compile(
    r"(?ms)^\[(?P<header>[A-Z][A-Z _]+)\]\n(?P<body>.*?)(?=\n\[[A-Z][A-Z _]+\]\n|\Z)"
)

_OUTCOME_MARKERS = (
    "not damaged",
    "is hit during",
    "takes one or more direct hits",
    "one or more direct hits damage",
    "is killed during",
    "starts this block dead",
    "remains killed",
    "remains dead",
)


AIM_TARGET_CENTER_GAP_PROMPT = (
    "Look at the small magenta hollow crosshair at the exact center of this image. "
    "Judge only the tiny unpainted center gap, not the magenta arms. "
    "Is the scene visible in that center gap the enemy or the background/environment? "
    "Ignore an enemy visible elsewhere. The player's foreground weapon is not the enemy."
)


AIM_TARGET_CENTER_POINT_PROMPT = (
    "Look at the bright lime-green hollow circle centered exactly in this image. "
    "The circle is only a location marker. Judge the single point at its exact center, "
    "not the ring and not the whole enclosed area. At that center point, is the scene "
    "the enemy or the background/environment? Ignore an enemy visible elsewhere. "
    "The player's foreground weapon is not the enemy."
)


AIM_TARGET_RING_OVERLAP_PROMPT = (
    "Look at the bright lime-green hollow circle at the center of this image. "
    "Is this aiming circle on the enemy? Answer enemy if any visible part of the "
    "enemy's body lies inside or overlaps the circle; exact center alignment is not "
    "required. Answer background only if the entire circle is off the enemy and covers "
    "background/environment. Ignore an enemy visible elsewhere. The player's foreground "
    "weapon is not the enemy."
)


AIM_TARGET_RING_NEUTRAL_PROMPT = (
    "Ignore muzzle flash and motion blur. The decoded opponent may be soft or lack "
    "fine texture: a coherent armored humanoid silhouette or body-colored region still "
    "counts as the enemy. Is any part of that humanoid inside or touching the bright "
    "lime-green circle? Answer enemy if yes, otherwise background. Do not count isolated "
    "wall texture, smoke, blood, or the player's foreground weapon as the enemy."
)


def draw_center_crosshair(image: Image.Image) -> Image.Image:
    ''

    canvas = image.convert("RGB").copy()
    width, height = canvas.size
    center_x, center_y = width // 2, height // 2
    scale = max(1, min(width, height))
    gap = max(3, round(scale * 0.008))
    arm = max(gap + 7, round(scale * 0.032))
    foreground_width = max(2, round(scale * 0.0045))
    outline_width = foreground_width + max(2, round(scale * 0.0045))
    segments = (
        ((center_x - arm, center_y), (center_x - gap, center_y)),
        ((center_x + gap, center_y), (center_x + arm, center_y)),
        ((center_x, center_y - arm), (center_x, center_y - gap)),
        ((center_x, center_y + gap), (center_x, center_y + arm)),
    )
    draw = ImageDraw.Draw(canvas)
    for start, end in segments:
        draw.line((start, end), fill=(0, 0, 0), width=outline_width)
    for start, end in segments:
        draw.line((start, end), fill=(255, 255, 255), width=foreground_width)
    canvas.info["vlm_center_crosshair"] = "hollow_image_center_v2"
    return canvas


def center_crosshair_crop(image: Image.Image, *, crop_size: int) -> Image.Image:
    ''

    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    rgb = image.convert("RGB")
    side = min(int(crop_size), rgb.width, rgb.height)
    center_x, center_y = rgb.width // 2, rgb.height // 2
    left = center_x - side // 2
    top = center_y - side // 2
    return rgb.crop((left, top, left + side, top + side))


def draw_direct_crosshair(image: Image.Image) -> Image.Image:
    ''

    canvas = image.convert("RGB").copy()
    width, height = canvas.size
    center_x, center_y = width // 2, height // 2
    scale = max(1, min(width, height))
    gap = max(4, round(scale * 0.032))
    arm = max(gap + 11, round(scale * 0.14))
    foreground_width = max(2, round(scale * 0.016))
    outline_width = foreground_width + max(2, round(scale * 0.012))
    segments = (
        ((center_x - arm, center_y), (center_x - gap, center_y)),
        ((center_x + gap, center_y), (center_x + arm, center_y)),
        ((center_x, center_y - arm), (center_x, center_y - gap)),
        ((center_x, center_y + gap), (center_x, center_y + arm)),
    )
    draw = ImageDraw.Draw(canvas)
    for start, end in segments:
        draw.line((start, end), fill=(0, 0, 0), width=outline_width)
    for start, end in segments:
        draw.line((start, end), fill=(255, 0, 255), width=foreground_width)
    canvas.info["vlm_center_crosshair"] = "direct_hollow_center_v3"
    return canvas


def draw_precise_aim_crosshair(image: Image.Image) -> Image.Image:
    ''

    canvas = image.convert("RGB").copy()
    width, height = canvas.size
    center_x, center_y = width // 2, height // 2
    scale = max(1, min(width, height))
    gap = max(3, round(scale * 0.0125))
    arm = max(gap + 6, round(scale * 0.04375))
    foreground_width = max(2, round(scale * 0.00625))
    outline_width = foreground_width + max(2, round(scale * 0.00625))
    segments = (
        ((center_x - arm, center_y), (center_x - gap, center_y)),
        ((center_x + gap, center_y), (center_x + arm, center_y)),
        ((center_x, center_y - arm), (center_x, center_y - gap)),
        ((center_x, center_y + gap), (center_x, center_y + arm)),
    )
    draw = ImageDraw.Draw(canvas)
    for start, end in segments:
        draw.line((start, end), fill=(0, 0, 0), width=outline_width)
    for start, end in segments:
        draw.line((start, end), fill=(255, 0, 255), width=foreground_width)
    canvas.info["vlm_center_crosshair"] = "precise_center_gap_v4"
    return canvas


def draw_aim_center_ring(image: Image.Image) -> Image.Image:
    ''

    canvas = image.convert("RGB").copy()
    width, height = canvas.size
    center_x, center_y = width // 2, height // 2
    scale = max(1, min(width, height))
    radius = max(8, round(scale * 0.05))
    foreground_width = max(2, round(scale * 0.009375))
    outline_width = foreground_width + max(2, round(scale * 0.00625))
    bounds = (
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
    )
    draw = ImageDraw.Draw(canvas)
    draw.ellipse(bounds, outline=(0, 0, 0), width=outline_width)
    draw.ellipse(bounds, outline=(57, 255, 20), width=foreground_width)
    canvas.info["vlm_center_marker"] = "aim_center_ring_lime_v1"
    return canvas


def shift_image_for_camera_projection(
    image: Image.Image, *, dx_pixels: float, dy_pixels: float
) -> Image.Image:
    ''

    rgb = image.convert("RGB")
    projected = Image.new("RGB", rgb.size, (0, 0, 0))
    projected.paste(rgb, (round(dx_pixels), round(dy_pixels)))
    return projected


@dataclass(frozen=True)
class JoystickPixelShift:
    dx_pixels: float
    dy_pixels: float
    num_action_frames: int
    mean_left_x: float
    mean_left_y: float
    mean_right_x: float
    mean_right_y: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JoystickPixelLookup:
    ''

    width: int
    height: int
    fps: int
    calibration_id: str
    source: dict[str, Any]
    right_x_points: tuple[tuple[float, float], ...] = ()
    right_y_points: tuple[tuple[float, float], ...] = ()
    linear_dx_coefficients: tuple[float, float, float, float, float] | None = None
    linear_dy_coefficients: tuple[float, float, float, float, float] | None = None
    linear_abs_ranges: tuple[float, float, float, float] | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "JoystickPixelLookup":
        resolved = Path(path).resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))

        def parse_points(axis: str) -> tuple[tuple[float, float], ...]:
            raw_points = payload["axes"][axis]["positive_magnitude_lookup"]
            points = tuple(
                (float(item["joystick_abs"]), float(item["pixels_per_frame_abs"]))
                for item in raw_points
            )
            if len(points) < 2 or points[0] != (0.0, 0.0):
                raise ValueError(f"{resolved}: {axis} lookup must begin at (0, 0)")
            if any(
                current[0] <= previous[0] or current[1] < previous[1]
                for previous, current in zip(points, points[1:], strict=False)
            ):
                raise ValueError(
                    f"{resolved}: {axis} lookup magnitudes must increase monotonically"
                )
            return points

        common = {
            "width": int(payload["geometry"]["width"]),
            "height": int(payload["geometry"]["height"]),
            "fps": int(payload["geometry"]["fps"]),
            "calibration_id": str(payload["calibration_id"]),
            "source": dict(payload.get("source") or {}),
        }
        model = payload.get("model")
        if isinstance(model, dict) and model.get("type") == "multivariate_linear":
            feature_order = list(model.get("feature_order") or [])
            expected_order = ["left_x", "left_y", "right_x", "right_y", "intercept"]
            if feature_order != expected_order:
                raise ValueError(
                    f"{resolved}: linear feature_order must be {expected_order}, got {feature_order}"
                )

            def parse_coefficients(name: str) -> tuple[float, float, float, float, float]:
                values = tuple(float(value) for value in model[name])
                if len(values) != 5:
                    raise ValueError(f"{resolved}: {name} must contain five coefficients")
                return values                              

            ranges = model.get("calibrated_abs_range")
            if not isinstance(ranges, dict):
                raise ValueError(f"{resolved}: linear calibration requires calibrated_abs_range")
            abs_ranges = tuple(
                float(ranges[axis]) for axis in ("left_x", "left_y", "right_x", "right_y")
            )
            if any(value <= 0 for value in abs_ranges):
                raise ValueError(f"{resolved}: calibrated_abs_range values must be positive")
            return cls(
                **common,
                linear_dx_coefficients=parse_coefficients("dx_pixels_per_frame"),
                linear_dy_coefficients=parse_coefficients("dy_pixels_per_frame"),
                linear_abs_ranges=abs_ranges,                          
            )
        if model is not None:
            raise ValueError(f"{resolved}: unsupported lookup model {model!r}")
        return cls(
            **common,
            right_x_points=parse_points("right_x"),
            right_y_points=parse_points("right_y"),
        )

    def validate_geometry(self, *, width: int, height: int, fps: int) -> None:
        actual = (int(width), int(height), int(fps))
        expected = (self.width, self.height, self.fps)
        if actual != expected:
            raise ValueError(
                f"Joystick pixel lookup {self.calibration_id!r} is for {expected}, got {actual}"
            )

    @staticmethod
    def _pixels_per_frame(
        value: float, points: tuple[tuple[float, float], ...], *, axis: str
    ) -> float:
        magnitude = abs(float(value))
        max_magnitude = points[-1][0]
        if magnitude > max_magnitude + 1e-7:
            raise ValueError(
                f"right_{axis}={value:+.6f} exceeds calibrated magnitude {max_magnitude:.6f}"
            )
        if magnitude == 0.0:
            return 0.0
        knots = [point[0] for point in points]
        upper = min(bisect.bisect_left(knots, magnitude), len(points) - 1)
        lower = max(0, upper - 1)
        x0, y0 = points[lower]
        x1, y1 = points[upper]
        if x1 == x0:
            pixels = y1
        else:
            weight = (magnitude - x0) / (x1 - x0)
            pixels = y0 + weight * (y1 - y0)
                                                                            
                                                                          
        return -pixels if value > 0 else pixels

    def estimate(
        self,
        joystick_frames: Sequence[Sequence[float]],
        *,
        enforce_calibrated_range: bool = True,
    ) -> JoystickPixelShift:
        if not joystick_frames:
            raise ValueError("At least one joystick frame is required")
        if any(len(frame) not in (2, 4) for frame in joystick_frames):
            raise ValueError("Each joystick frame must contain two right axes or all four axes")
        values = [
            (
                float(frame[0]) if len(frame) == 4 else 0.0,
                float(frame[1]) if len(frame) == 4 else 0.0,
                float(frame[2]) if len(frame) == 4 else float(frame[0]),
                float(frame[3]) if len(frame) == 4 else float(frame[1]),
            )
            for frame in joystick_frames
        ]
        if self.linear_dx_coefficients is not None:
            if self.linear_dy_coefficients is None or self.linear_abs_ranges is None:
                raise RuntimeError("Incomplete multivariate joystick calibration")
            if any(len(frame) != 4 for frame in joystick_frames):
                raise ValueError("Multivariate calibration requires all four joystick axes")
            axis_names = ("left_x", "left_y", "right_x", "right_y")
            if enforce_calibrated_range:
                for value in values:
                    for axis, magnitude, limit in zip(
                        axis_names,
                        (abs(item) for item in value),
                        self.linear_abs_ranges,
                        strict=True,
                    ):
                        if magnitude > limit + 1e-7:
                            raise ValueError(
                                f"{axis}={value[axis_names.index(axis)]:+.6f} exceeds "
                                f"calibrated magnitude {limit:.6f}"
                            )

            def linear_sum(coefficients: tuple[float, float, float, float, float]) -> float:
                return sum(
                    sum(coefficient * axis for coefficient, axis in zip(coefficients[:4], value, strict=True))
                    + coefficients[4]
                    for value in values
                )

            dx = linear_sum(self.linear_dx_coefficients)
            dy = linear_sum(self.linear_dy_coefficients)
        else:
            if not self.right_x_points or not self.right_y_points:
                raise RuntimeError("Incomplete legacy right-stick calibration")
            dx = sum(
                self._pixels_per_frame(value[2], self.right_x_points, axis="x")
                for value in values
            )
            dy = sum(
                self._pixels_per_frame(value[3], self.right_y_points, axis="y")
                for value in values
            )
        return JoystickPixelShift(
            dx_pixels=dx,
            dy_pixels=dy,
            num_action_frames=len(values),
            mean_left_x=sum(value[0] for value in values) / len(values),
            mean_left_y=sum(value[1] for value in values) / len(values),
            mean_right_x=sum(value[2] for value in values) / len(values),
            mean_right_y=sum(value[3] for value in values) / len(values),
        )

    def solve_right_stick_for_target(
        self,
        *,
        target_x_norm: float,
        target_y_norm: float,
        left_x: float,
        left_y: float,
        num_action_frames: int,
        gain: float = 1.0,
        max_right_x: float = 0.08,
        max_right_y: float = 0.05,
    ) -> tuple[float, float, JoystickPixelShift]:
        ''

        if self.linear_dx_coefficients is None or self.linear_dy_coefficients is None:
            raise ValueError("Target inversion requires a multivariate linear calibration")
        if not 0.0 <= target_x_norm <= 1.0 or not 0.0 <= target_y_norm <= 1.0:
            raise ValueError("Normalized target coordinates must lie in [0,1]")
        if num_action_frames <= 0:
            raise ValueError("num_action_frames must be positive")
        if not 0.0 < gain <= 1.5:
            raise ValueError("gain must lie in (0,1.5]")
        if not 0.0 < max_right_x <= 1.0 or not 0.0 < max_right_y <= 1.0:
            raise ValueError("Right-stick safety bounds must lie in (0,1]")

        dx = self.linear_dx_coefficients
        dy = self.linear_dy_coefficients
        desired_dx_per_frame = (
            gain * (0.5 - target_x_norm) * self.width / num_action_frames
        )
        desired_dy_per_frame = (
            gain * (0.5 - target_y_norm) * self.height / num_action_frames
        )
        rhs_x = desired_dx_per_frame - (dx[0] * left_x + dx[1] * left_y + dx[4])
        rhs_y = desired_dy_per_frame - (dy[0] * left_x + dy[1] * left_y + dy[4])
        determinant = dx[2] * dy[3] - dx[3] * dy[2]
        if abs(determinant) < 1e-8:
            raise ValueError("Right-stick calibration matrix is singular")
        right_x = (rhs_x * dy[3] - dx[3] * rhs_y) / determinant
        right_y = (dx[2] * rhs_y - rhs_x * dy[2]) / determinant
        right_x = max(-max_right_x, min(max_right_x, right_x))
        right_y = max(-max_right_y, min(max_right_y, right_y))
        action = [left_x, left_y, right_x, right_y]
        shift = self.estimate(
            [action for _ in range(num_action_frames)],
            enforce_calibrated_range=False,
        )
        return right_x, right_y, shift


def extract_prompt_section(prompt: str, header: str) -> str | None:
    ''

    normalized = header.strip().strip("[]").upper()
    for match in _PROMPT_SECTION.finditer(prompt):
        if match.group("header") == normalized:
            return match.group("body").strip()
    return None


def freeze_opponent_motion(prompt: str) -> str:
    ''

    match = _OPPONENT_MOTION_OUTCOME_SECTION.search(prompt)
    if match is None:
        raise ValueError(
            "Prompt has no [CURRENT OPPONENT MOTION AND OUTCOME] section"
        )
    sentences = _sentences(match.group(2))
    outcomes = [
        sentence
        for sentence in sentences
        if any(marker in sentence.lower() for marker in _OUTCOME_MARKERS)
    ]
    if not outcomes:
        raise ValueError(
            "Unable to preserve an outcome while freezing opponent motion"
        )
    body = " ".join(
        (
            "The opponent remains stationary.",
            *outcomes,
            "The opponent remains in the player's view during this block.",
        )
    )
    return prompt[: match.start()] + match.group(1) + body + prompt[match.end() :]


def _sentences(text: str) -> list[str]:
    return [value.strip() for value in re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", text) if value.strip()]


def _replace_modern_opponent_outcome(
    prompt: str,
    *,
    outcome: str,
    persistent: bool = False,
) -> str:
    ''

    match = _OPPONENT_MOTION_OUTCOME_SECTION.search(prompt)
    if match is None:
        raise ValueError("Prompt has no [CURRENT OPPONENT MOTION AND OUTCOME] section")
    if persistent:
        body = outcome
    else:
        sentences = _sentences(match.group(2))
        outcome_indices = [
            index
            for index, sentence in enumerate(sentences)
            if any(marker in sentence.lower() for marker in _OUTCOME_MARKERS)
        ]
        if not outcome_indices:
            raise ValueError(
                "Unable to locate the outcome sentence in "
                "[CURRENT OPPONENT MOTION AND OUTCOME]"
            )
        first_outcome = outcome_indices[0]
        retained = [
            sentence
            for index, sentence in enumerate(sentences)
            if index not in set(outcome_indices)
        ]
        insert_at = sum(index < first_outcome for index in range(len(sentences)) if index not in set(outcome_indices))
        retained.insert(insert_at, outcome)
        body = " ".join(retained)
    return prompt[: match.start()] + match.group(1) + body.strip() + prompt[match.end() :]


def set_player_killed_outcome(prompt: str, *, killed: bool) -> str:
    ''

    match = _OPPONENT_MOTION_OUTCOME_SECTION.search(prompt)
    if match is None:
        raise ValueError(
            "Player outcome routing requires "
            "[CURRENT OPPONENT MOTION AND OUTCOME]"
        )
    original_body = match.group(2)
    leading_whitespace = original_body[: len(original_body) - len(original_body.lstrip())]
    trailing_whitespace = original_body[len(original_body.rstrip()) :]
    sentences = [
        sentence
        for sentence in _sentences(original_body)
        if sentence != PLAYER_KILLED_OUTCOME
    ]
    if killed:
        visibility_markers = (
            "player's view",
            "off-screen",
            "goes off-screen",
        )
        insert_at = next(
            (
                index
                for index, sentence in enumerate(sentences)
                if any(marker in sentence for marker in visibility_markers)
            ),
            len(sentences),
        )
        sentences.insert(insert_at, PLAYER_KILLED_OUTCOME)
    body = leading_whitespace + " ".join(sentences) + trailing_whitespace
    return prompt[: match.start()] + match.group(1) + body + prompt[match.end() :]


def replace_interaction_condition(prompt: str, condition: str) -> str:
    ''

    if not _INTERACTION_SECTION.search(prompt):
        raise ValueError("Prompt has no [CURRENT INTERACTION CONDITION] section")
    return _INTERACTION_SECTION.sub(r"\1" + condition.strip(), prompt, count=1)


def build_interaction_prompts(prompt: str) -> dict[InteractionMode, str]:
    ''

    modern_match = _OPPONENT_MOTION_OUTCOME_SECTION.search(prompt)
    if modern_match:
        return {
            "active": _replace_modern_opponent_outcome(
                prompt, outcome=ACTIVE_OPPONENT_OUTCOME
            ),
            "hit_alive": _replace_modern_opponent_outcome(
                prompt, outcome=HIT_ALIVE_OPPONENT_OUTCOME
            ),
            "miss_alive": _replace_modern_opponent_outcome(
                prompt, outcome=MISS_ALIVE_OPPONENT_OUTCOME
            ),
            "collapse": _replace_modern_opponent_outcome(
                prompt,
                outcome=COLLAPSE_OPPONENT_OUTCOME,
            ),
            "persist_killed": _replace_modern_opponent_outcome(
                prompt,
                outcome=PERSIST_KILLED_OPPONENT_OUTCOME,
                persistent=True,
            ),
        }

    return {
        "active": replace_interaction_condition(prompt, ACTIVE_INTERACTION),
        "hit_alive": replace_interaction_condition(prompt, HIT_ALIVE_INTERACTION),
        "miss_alive": replace_interaction_condition(prompt, MISS_ALIVE_INTERACTION),
        "collapse": replace_interaction_condition(prompt, COLLAPSE_INTERACTION),
        "persist_killed": replace_interaction_condition(prompt, PERSIST_KILLED_INTERACTION),
    }


def build_duel_lifecycle_prompts(prompt: str) -> dict[str, str]:
    ''

    match = _OPPONENT_MOTION_OUTCOME_SECTION.search(prompt)
    if match is None:
        raise ValueError(
            "Duel lifecycle prompts require "
            "[CURRENT OPPONENT MOTION AND OUTCOME]"
        )
    sentences = _sentences(match.group(2))
    movement = next(
        (
            sentence
            for sentence in sentences
            if sentence.startswith("The opponent moves ")
            or sentence == "The opponent remains stationary."
        ),
        "The opponent moves toward the player.",
    )
    if movement == "The opponent remains stationary.":
        movement = "The opponent moves toward the player."
    fire = next(
        (
            sentence
            for sentence in sentences
            if "fires the machine gun toward the player" in sentence
            or "does not fire during this block" in sentence
        ),
        "The opponent does not fire during this block.",
    )

    def replace_body(body: str) -> str:
        return prompt[: match.start()] + match.group(1) + body + prompt[match.end() :]

    def reentry_body(reentry_movement: str) -> str:
        return " ".join(
            (
                reentry_movement,
                fire,
                "The opponent is not damaged during this block.",
                "The opponent starts this block off-screen and re-enters the player's view during this block.",
            )
        )

    return {
        "corpse": replace_body(
            "The opponent remains stationary. "
            "The opponent does not fire during this block. "
            "The opponent starts this block dead and remains dead throughout the block. "
            "The opponent remains in the player's view during this block."
        ),
        "respawn_offscreen": replace_body(
            "The opponent remains stationary. "
            "The opponent does not fire during this block. "
            "The opponent is not damaged during this block. "
            "The opponent respawns outside the player's view during this block. "
            "The opponent leaves the player's view and goes off-screen during this block."
        ),
        "reenter": replace_body(reentry_body(movement)),
        "reenter_left": replace_body(
            reentry_body(
                "The opponent moves toward the player's left while approaching the player."
            )
        ),
        "reenter_right": replace_body(
            reentry_body(
                "The opponent moves toward the player's right while approaching the player."
            )
        ),
    }


@dataclass(frozen=True)
class VLMObservation:
    enemy_visible: bool
    shot_result: ShotResult
    enemy_posture: EnemyPosture
    confidence: float
    evidence: str
    raw_response: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any], *, raw_response: str = "") -> "VLMObservation":
        shot_result = str(value.get("shot_result", ""))
        enemy_posture = str(value.get("enemy_posture", ""))
        if shot_result not in {"hit", "miss", "uncertain", "not_applicable"}:
            raise ValueError(f"Invalid VLM shot_result: {shot_result!r}")
        if enemy_posture not in {"standing", "falling", "down", "not_visible", "uncertain"}:
            raise ValueError(f"Invalid VLM enemy_posture: {enemy_posture!r}")
        confidence = float(value.get("confidence"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid VLM confidence: {confidence}")
        return cls(
            enemy_visible=bool(value.get("enemy_visible")),
            shot_result=shot_result,                          
            enemy_posture=enemy_posture,                          
            confidence=confidence,
            evidence=str(value.get("evidence", "")).strip(),
            raw_response=raw_response,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HitLikelihoodObservation:
    ''

    can_hit: bool
    confidence: float
    evidence: str
    raw_response: str = ""

    @classmethod
    def from_mapping(
        cls, value: dict[str, Any], *, raw_response: str = ""
    ) -> "HitLikelihoodObservation":
        confidence = float(value.get("confidence"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid VLM confidence: {confidence}")
        return cls(
            can_hit=bool(value.get("can_hit")),
            confidence=confidence,
            evidence=str(value.get("evidence", "")).strip(),
            raw_response=raw_response,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AimTargetObservation:
    ''

    aim_target: Literal["enemy", "background"]
    confidence: float
    evidence: str
    raw_response: str = ""

    @classmethod
    def from_mapping(
        cls, value: dict[str, Any], *, raw_response: str = ""
    ) -> "AimTargetObservation":
        aim_target = str(value.get("aim_target", ""))
        if aim_target not in {"enemy", "background"}:
            raise ValueError(f"Invalid VLM aim_target: {aim_target!r}")
        confidence = float(value.get("confidence"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid VLM confidence: {confidence}")
        return cls(
            aim_target=aim_target,                          
            confidence=confidence,
            evidence=str(value.get("evidence", "")).strip(),
            raw_response=raw_response,
        )

    @property
    def shot_result(self) -> Literal["hit", "miss"]:
        return "hit" if self.aim_target == "enemy" else "miss"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "shot_result": self.shot_result}

    def to_vlm_observation(
        self,
        *,
        enemy_visible: bool = True,
        enemy_posture: EnemyPosture = "standing",
    ) -> VLMObservation:
        ''

        return VLMObservation(
            enemy_visible=enemy_visible,
            shot_result=self.shot_result,
            enemy_posture=enemy_posture,
            confidence=self.confidence,
            evidence=self.evidence,
            raw_response=self.raw_response,
        )


@dataclass(frozen=True)
class CombatGuidanceRealizationObservation:
    ''

    realized: bool
    outcome: Literal[
        "knocked_down_or_disappeared", "still_upright", "uncertain"
    ]
    confidence: float
    evidence: str
    raw_response: str = ""

    @classmethod
    def from_mapping(
        cls, value: dict[str, Any], *, raw_response: str = ""
    ) -> "CombatGuidanceRealizationObservation":
        outcome = str(value.get("outcome", ""))
        if outcome not in {
            "knocked_down_or_disappeared",
            "still_upright",
            "uncertain",
        }:
            raise ValueError(f"Invalid combat-guidance outcome: {outcome!r}")
        confidence = float(value.get("confidence"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid combat-guidance confidence: {confidence}")
        return cls(
            realized=bool(value.get("realized")),
            outcome=outcome,                          
            confidence=confidence,
            evidence=str(value.get("evidence", "")).strip(),
            raw_response=raw_response,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AimGeometryObservation:
    ''

    enemy_visible: bool
    enemy_side: EnemySide
    center_dot_overlaps_enemy: bool
    within_weapon_spread: bool
    visible_projectile_contact: bool
    enemy_posture: EnemyPosture
    confidence: float
    evidence: str
    raw_response: str = ""

    @classmethod
    def from_mapping(
        cls, value: dict[str, Any], *, raw_response: str = ""
    ) -> "AimGeometryObservation":
        enemy_side = str(value.get("enemy_side", ""))
        enemy_posture = str(value.get("enemy_posture", ""))
        if enemy_side not in {
            "far_left",
            "left",
            "center",
            "right",
            "far_right",
            "not_visible",
            "uncertain",
        }:
            raise ValueError(f"Invalid VLM enemy_side: {enemy_side!r}")
        if enemy_posture not in {
            "standing",
            "falling",
            "down",
            "not_visible",
            "uncertain",
        }:
            raise ValueError(f"Invalid VLM enemy_posture: {enemy_posture!r}")
        confidence = float(value.get("confidence"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid VLM confidence: {confidence}")
        return cls(
            enemy_visible=bool(value.get("enemy_visible")),
            enemy_side=enemy_side,                          
            center_dot_overlaps_enemy=bool(
                value.get("center_dot_overlaps_enemy")
            ),
            within_weapon_spread=bool(value.get("within_weapon_spread")),
            visible_projectile_contact=bool(
                value.get("visible_projectile_contact")
            ),
            enemy_posture=enemy_posture,                          
            confidence=confidence,
            evidence=str(value.get("evidence", "")).strip(),
            raw_response=raw_response,
        )

    def to_observation(self, *, spread_weapon: bool, projectile_weapon: bool) -> VLMObservation:
        hit = self.center_dot_overlaps_enemy
        if spread_weapon:
            hit = hit or self.within_weapon_spread
        if projectile_weapon:
            hit = hit or self.visible_projectile_contact
        if hit:
            shot_result: ShotResult = "hit"
        elif self.enemy_visible:
            shot_result = "miss"
        else:
            shot_result = "uncertain"
        return VLMObservation(
            enemy_visible=self.enemy_visible,
            shot_result=shot_result,
            enemy_posture=self.enemy_posture,
            confidence=self.confidence,
            evidence=self.evidence,
            raw_response=self.raw_response,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FPSNavigationObservation:
    ''

    enemy_visible: bool
    enemy_side: EnemySide
    enemy_vertical: EnemyVertical
    enemy_posture: EnemyPosture
    path_ahead: NavigationPathState
    recommended_movement: NavigationMovement
    recommended_turn: NavigationTurn
    confidence: float
    evidence: str
    raw_response: str = ""
    enemy_x_norm: float | None = None
    enemy_y_norm: float | None = None

    @classmethod
    def from_mapping(
        cls, value: dict[str, Any], *, raw_response: str = ""
    ) -> "FPSNavigationObservation":
        enemy_side = str(value.get("enemy_side", ""))
        enemy_vertical = str(value.get("enemy_vertical", ""))
        enemy_posture = str(value.get("enemy_posture", ""))
        path_ahead = str(value.get("path_ahead", ""))
        movement = str(value.get("recommended_movement", ""))
        turn = str(value.get("recommended_turn", ""))
        if enemy_side not in {
            "far_left",
            "left",
            "center",
            "right",
            "far_right",
            "not_visible",
            "uncertain",
        }:
            raise ValueError(f"Invalid navigation enemy_side: {enemy_side!r}")
        if enemy_vertical not in {
            "above",
            "center",
            "below",
            "not_visible",
            "uncertain",
        }:
            raise ValueError(
                f"Invalid navigation enemy_vertical: {enemy_vertical!r}"
            )
        if enemy_posture not in {
            "standing",
            "falling",
            "down",
            "not_visible",
            "uncertain",
        }:
            raise ValueError(
                f"Invalid navigation enemy_posture: {enemy_posture!r}"
            )
        if path_ahead not in {"clear", "blocked", "corner", "uncertain"}:
            raise ValueError(f"Invalid navigation path_ahead: {path_ahead!r}")
        if movement not in {
            "forward",
            "strafe_left",
            "strafe_right",
            "backward",
        }:
            raise ValueError(f"Invalid navigation movement: {movement!r}")
        if turn not in {"left", "right", "none"}:
            raise ValueError(f"Invalid navigation turn: {turn!r}")
        confidence = float(value.get("confidence"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid navigation confidence: {confidence}")
        enemy_visible = bool(value.get("enemy_visible"))
        if not enemy_visible and (
            enemy_side not in {"not_visible", "uncertain"}
            or enemy_vertical not in {"not_visible", "uncertain"}
        ):
            raise ValueError(
                "Invisible enemy must use not_visible/uncertain position labels"
            )
        if enemy_visible and (
            enemy_side == "not_visible"
            or enemy_vertical == "not_visible"
            or enemy_posture == "not_visible"
        ):
            raise ValueError(
                "Visible enemy cannot use a not_visible position or posture label"
            )
        has_x = "enemy_x_norm" in value
        has_y = "enemy_y_norm" in value
        if has_x != has_y:
            raise ValueError("Navigation coordinates must provide both x and y")
        enemy_x_norm = float(value["enemy_x_norm"]) if has_x else None
        enemy_y_norm = float(value["enemy_y_norm"]) if has_y else None
        if enemy_x_norm is not None and not 0.0 <= enemy_x_norm <= 1.0:
            raise ValueError(f"Invalid normalized enemy x coordinate: {enemy_x_norm}")
        if enemy_y_norm is not None and not 0.0 <= enemy_y_norm <= 1.0:
            raise ValueError(f"Invalid normalized enemy y coordinate: {enemy_y_norm}")
        return cls(
            enemy_visible=enemy_visible,
            enemy_side=enemy_side,                          
            enemy_vertical=enemy_vertical,                          
            enemy_posture=enemy_posture,                          
            path_ahead=path_ahead,                          
            recommended_movement=movement,                          
            recommended_turn=turn,                          
            confidence=confidence,
            evidence=str(value.get("evidence", "")).strip(),
            raw_response=raw_response,
            enemy_x_norm=enemy_x_norm,
            enemy_y_norm=enemy_y_norm,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FPSBlockActionPlan:
    ''

    left_x: float
    left_y: float
    right_x: float
    right_y: float
    fire: bool
    movement: NavigationMovement
    turn: NavigationTurn

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_fps_block_actions(
    observation: FPSNavigationObservation,
    *,
    move_magnitude: float = 0.20,
    strafe_magnitude: float = 0.18,
    turn_magnitude: float = 0.025,
    aim_magnitude: float = 0.035,
    vertical_aim_magnitude: float = 0.018,
    fallback_turn: Literal["left", "right"] = "right",
) -> FPSBlockActionPlan:
    ''

    magnitudes = (
        move_magnitude,
        strafe_magnitude,
        turn_magnitude,
        aim_magnitude,
        vertical_aim_magnitude,
    )
    if any(not 0.0 < value <= 1.0 for value in magnitudes):
        raise ValueError("Action-controller magnitudes must lie in (0, 1]")

    movement: NavigationMovement = observation.recommended_movement
    turn: NavigationTurn = observation.recommended_turn
    if observation.path_ahead in {"blocked", "corner"} and movement == "forward":
        if turn == "left":
            movement = "strafe_left"
        elif turn == "right":
            movement = "strafe_right"
        else:
            movement = "backward"
            turn = fallback_turn

    if movement == "forward":
        left_x, left_y = 0.0, -move_magnitude
    elif movement == "strafe_left":
        left_x, left_y = -strafe_magnitude, -0.55 * move_magnitude
    elif movement == "strafe_right":
        left_x, left_y = strafe_magnitude, -0.55 * move_magnitude
    else:
        left_x, left_y = 0.0, 0.80 * move_magnitude

    if observation.enemy_visible:
        right_x = {
            "far_left": -1.5 * aim_magnitude,
            "left": -aim_magnitude,
            "center": 0.0,
            "right": aim_magnitude,
            "far_right": 1.5 * aim_magnitude,
            "not_visible": 0.0,
            "uncertain": 0.0,
        }[observation.enemy_side]
        right_y = {
            "above": -vertical_aim_magnitude,
            "center": 0.0,
            "below": vertical_aim_magnitude,
            "not_visible": 0.0,
            "uncertain": 0.0,
        }[observation.enemy_vertical]
    else:
        right_x = (
            -turn_magnitude
            if turn == "left"
            else turn_magnitude
            if turn == "right"
            else 0.0
        )
        right_y = 0.0

    fire = observation.enemy_visible and observation.enemy_posture not in {
        "falling",
        "down",
        "not_visible",
    }
    values = (left_x, left_y, right_x, right_y)
    if any(not -1.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Action plan exceeds [-1,1]: {values}")
    if left_x == 0.0 and left_y == 0.0:
        raise RuntimeError("Navigation controller must never emit idle movement")
    return FPSBlockActionPlan(
        left_x=left_x,
        left_y=left_y,
        right_x=right_x,
        right_y=right_y,
        fire=fire,
        movement=movement,
        turn=turn,
    )

def strict_consensus_observation(
    observations: Sequence[VLMObservation],
) -> VLMObservation:
    ''

    if not observations:
        raise ValueError("At least one VLM observation is required")
    if len(observations) == 1:
        return observations[0]

    shot_votes = [value.shot_result for value in observations]
    posture_votes = [value.enemy_posture for value in observations]
    visibility_votes = [value.enemy_visible for value in observations]
    shot_result: ShotResult = (
        shot_votes[0] if len(set(shot_votes)) == 1 else "uncertain"
    )
    enemy_posture: EnemyPosture = (
        posture_votes[0] if len(set(posture_votes)) == 1 else "uncertain"
    )
    enemy_visible = sum(visibility_votes) > len(visibility_votes) / 2
    confidence = min(value.confidence for value in observations)
    if len(set(shot_votes)) == 1:
        evidence = observations[0].evidence
    else:
        evidence = (
            f"Strict VLM consensus rejected a combat outcome because votes were "
            f"{shot_votes}; no hit is applied."
        )
    return VLMObservation(
        enemy_visible=enemy_visible,
        shot_result=shot_result,
        enemy_posture=enemy_posture,
        confidence=confidence,
        evidence=evidence,
        raw_response=json.dumps(
            [value.to_dict() for value in observations], ensure_ascii=False
        ),
    )


def classify_grounded_shot(
    boxes: Sequence[dict[str, Any]],
    *,
    shot_frame: int,
    weapon_context: str,
) -> tuple[VLMObservation, dict[str, Any]]:
    ''

    visible = [box for box in boxes if bool(box.get("enemy_visible"))]
    if len(visible) < 2:
        observation = VLMObservation(
            enemy_visible=bool(visible),
            shot_result="uncertain",
            enemy_posture="uncertain" if visible else "not_visible",
            confidence=0.0,
            evidence="Fewer than two visible grounded boxes; no hit is applied.",
        )
        return observation, {
            "boxes": list(boxes),
            "predicted_box_at_shot": None,
            "crosshair": [500.0, 500.0],
            "classification": "uncertain",
        }

    frames = [float(box["frame"]) for box in visible]

    def extrapolate(field: str) -> float:
        values = [float(box[field]) for box in visible]
        frame_mean = sum(frames) / len(frames)
        value_mean = sum(values) / len(values)
        denominator = sum((frame - frame_mean) ** 2 for frame in frames)
        if denominator == 0:
            return values[-1]
        slope = sum(
            (frame - frame_mean) * (value - value_mean)
            for frame, value in zip(frames, values, strict=True)
        ) / denominator
        return value_mean + slope * (float(shot_frame) - frame_mean)

    predicted = {
        field: max(0.0, min(1000.0, extrapolate(field)))
        for field in ("x1", "y1", "x2", "y2")
    }
    if predicted["x1"] >= predicted["x2"] or predicted["y1"] >= predicted["y2"]:
        raise ValueError(f"Invalid extrapolated enemy box: {predicted}")

    width = predicted["x2"] - predicted["x1"]
    height = predicted["y2"] - predicted["y1"]
    weapon_lower = weapon_context.lower()
    spread_weapon = "shotgun" in weapon_lower or "pellet spread" in weapon_lower
    margin_ratio = 0.45 if spread_weapon else 0.05
    margin_x = max(4.0, width * margin_ratio)
    margin_y = max(4.0, height * margin_ratio)
    crosshair_x = 500.0
    crosshair_y = 500.0
    hit = (
        predicted["x1"] - margin_x
        <= crosshair_x
        <= predicted["x2"] + margin_x
        and predicted["y1"] - margin_y
        <= crosshair_y
        <= predicted["y2"] + margin_y
    )
    result: ShotResult = "hit" if hit else "miss"
    confidence = 0.95 if len(visible) == len(boxes) else 0.80
    evidence = (
        f"Grounded box at frame {shot_frame} is "
        f"[{predicted['x1']:.1f},{predicted['y1']:.1f},"
        f"{predicted['x2']:.1f},{predicted['y2']:.1f}]; "
        f"crosshair=[500,500], margin=[{margin_x:.1f},{margin_y:.1f}], result={result}."
    )
    observation = VLMObservation(
        enemy_visible=True,
        shot_result=result,
        enemy_posture="standing",
        confidence=confidence,
        evidence=evidence,
    )
    return observation, {
        "boxes": list(boxes),
        "predicted_box_at_shot": predicted,
        "crosshair": [crosshair_x, crosshair_y],
        "margin": [margin_x, margin_y],
        "margin_ratio": margin_ratio,
        "spread_weapon": spread_weapon,
        "classification": result,
    }
class OpenAIVLMObserver:
    ''

    _JSON_SCHEMA = {
        "name": "fps_visual_observation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "enemy_visible": {"type": "boolean"},
                "shot_result": {
                    "type": "string",
                    "enum": ["hit", "miss", "uncertain", "not_applicable"],
                },
                "enemy_posture": {
                    "type": "string",
                    "enum": ["standing", "falling", "down", "not_visible", "uncertain"],
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence": {"type": "string"},
            },
            "required": [
                "enemy_visible",
                "shot_result",
                "enemy_posture",
                "confidence",
                "evidence",
            ],
            "additionalProperties": False,
        },
    }

    _AIM_GEOMETRY_SCHEMA = {
        "name": "fps_aim_geometry",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "enemy_visible": {"type": "boolean"},
                "enemy_side": {
                    "type": "string",
                    "enum": [
                        "far_left",
                        "left",
                        "center",
                        "right",
                        "far_right",
                        "not_visible",
                        "uncertain",
                    ],
                },
                "center_dot_overlaps_enemy": {"type": "boolean"},
                "within_weapon_spread": {"type": "boolean"},
                "visible_projectile_contact": {"type": "boolean"},
                "enemy_posture": {
                    "type": "string",
                    "enum": [
                        "standing",
                        "falling",
                        "down",
                        "not_visible",
                        "uncertain",
                    ],
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence": {"type": "string"},
            },
            "required": [
                "enemy_visible",
                "enemy_side",
                "center_dot_overlaps_enemy",
                "within_weapon_spread",
                "visible_projectile_contact",
                "enemy_posture",
                "confidence",
                "evidence",
            ],
            "additionalProperties": False,
        },
    }

    _FOCUS_AIM_GEOMETRY_SCHEMA = {
        "name": "fps_rough_crosshair_alignment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "enemy_visible": {"type": "boolean"},
                "crosshair_roughly_on_enemy": {"type": "boolean"},
                "within_weapon_spread": {"type": "boolean"},
                "enemy_posture": {
                    "type": "string",
                    "enum": [
                        "standing",
                        "falling",
                        "down",
                        "not_visible",
                        "uncertain",
                    ],
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence": {"type": "string"},
            },
            "required": [
                "enemy_visible",
                "crosshair_roughly_on_enemy",
                "within_weapon_spread",
                "enemy_posture",
                "confidence",
                "evidence",
            ],
            "additionalProperties": False,
        },
    }

    _HIT_LIKELIHOOD_SCHEMA = {
        "name": "fps_hit_likelihood",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "can_hit": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence": {"type": "string"},
            },
            "required": ["can_hit", "confidence", "evidence"],
            "additionalProperties": False,
        },
    }

    _AIM_TARGET_SCHEMA = {
        "name": "fps_crosshair_target",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "aim_target": {
                    "type": "string",
                    "enum": ["enemy", "background"],
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence": {"type": "string"},
            },
            "required": ["aim_target", "confidence", "evidence"],
            "additionalProperties": False,
        },
    }

    _COMBAT_GUIDANCE_REALIZATION_SCHEMA = {
        "name": "fps_combat_guidance_realization",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "realized": {"type": "boolean"},
                "outcome": {
                    "type": "string",
                    "enum": [
                        "knocked_down_or_disappeared",
                        "still_upright",
                        "uncertain",
                    ],
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence": {"type": "string"},
            },
            "required": ["realized", "outcome", "confidence", "evidence"],
            "additionalProperties": False,
        },
    }

    _FPS_NAVIGATION_SCHEMA = {
        "name": "fps_next_block_navigation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "enemy_visible": {"type": "boolean"},
                "enemy_side": {
                    "type": "string",
                    "enum": [
                        "far_left",
                        "left",
                        "center",
                        "right",
                        "far_right",
                        "not_visible",
                        "uncertain",
                    ],
                },
                "enemy_vertical": {
                    "type": "string",
                    "enum": ["above", "center", "below", "not_visible", "uncertain"],
                },
                "enemy_posture": {
                    "type": "string",
                    "enum": [
                        "standing",
                        "falling",
                        "down",
                        "not_visible",
                        "uncertain",
                    ],
                },
                "path_ahead": {
                    "type": "string",
                    "enum": ["clear", "blocked", "corner", "uncertain"],
                },
                "recommended_movement": {
                    "type": "string",
                    "enum": ["forward", "strafe_left", "strafe_right", "backward"],
                },
                "recommended_turn": {
                    "type": "string",
                    "enum": ["left", "right", "none"],
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence": {"type": "string"},
            },
            "required": [
                "enemy_visible",
                "enemy_side",
                "enemy_vertical",
                "enemy_posture",
                "path_ahead",
                "recommended_movement",
                "recommended_turn",
                "confidence",
                "evidence",
            ],
            "additionalProperties": False,
        },
    }

    _FPS_NAVIGATION_V2_SCHEMA = {
        "name": "fps_next_block_navigation_v2",
        "strict": True,
        "schema": {
            **_FPS_NAVIGATION_SCHEMA["schema"],
            "properties": {
                **_FPS_NAVIGATION_SCHEMA["schema"]["properties"],
                "enemy_x_norm": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "enemy_y_norm": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": [
                *_FPS_NAVIGATION_SCHEMA["schema"]["required"],
                "enemy_x_norm",
                "enemy_y_norm",
            ],
        },
    }

    _SYSTEM_PROMPT = """You are a conservative visual event detector for generated first-person shooter video. The supplied frames are chronological and each has an explicit frame number. Judge only visible evidence in those frames. Every supplied image has an artificial high-contrast crosshair drawn at the exact center solely as an observation aid; it is not a projectile, impact, muzzle flash, or native game object.

Use the supplied weapon description. Hitscan damage is instantaneous along the aim line; shotgun pellets form a spread around that line; projectile or energy shots may have visible travel time. Bright brass ejected to the side is a shell casing, not a projectile. A muzzle flash, tracer, energy bolt, or lingering generated effect alone is not evidence of a hit. For a shot event, output hit only when the enemy visibly reacts, an impact is visible on the enemy, or the described firing geometry unambiguously intersects the enemy. Output miss when the attack path is clearly off the enemy or obstructed. Otherwise output uncertain. Generation artifacts are not evidence. Do not infer an intended outcome or hidden game rule.

enemy_posture describes the tracked enemy in the last supplied frame. For a state_check event, shot_result must be not_applicable. Keep evidence to one concrete sentence of at most 35 words."""

    _PRE_ACTION_SYSTEM_PROMPT = """You are a conservative pre-action hit predictor for first-person shooter gameplay. The supplied images are causal history only: they end before the upcoming generated interval and contain no future evidence. Every supplied image has an artificial high-contrast crosshair drawn at the exact center solely as an observation aid; use that crosshair as the current aim line and do not mistake it for a projectile, impact, or native game object.

Use the supplied tracked-enemy and weapon descriptions to identify the same target and the primary-fire geometry across images. Hitscan fire is instantaneous along the center aim line. A shotgun can hit when its center line or plausible pellet spread overlaps the target at the stated range. A direct projectile or energy shot can hit only when its projected path is unobstructed and plausibly reaches the target. Use the visible center crosshair, target position, stated distance, and supplied upcoming controller plan. Do not infer a desired outcome or hidden health rule. Predict hit only when the planned camera motion and the described fire geometry make contact likely at the stated fire frame; predict miss when they clearly do not. Otherwise output uncertain.

Locate the central reticle region and compare it with the tracked enemy's whole visible body, including limbs. Judge practical gameplay alignment rather than exact-pixel segmentation: overlap with any body part counts, and an immediately adjacent offset no larger than about one apparent target-body width also counts for this coarse controller. Output miss only when the target remains separated from the central reticle neighborhood by a clear margin at the shot. Account for a described shotgun spread, but do not treat a wide visual muzzle flash as the damage cone. Use uncertain only when target identity, occlusion, projectile travel, or aim geometry is genuinely ambiguous.

There is no aim assist, target snapping, or automatic correction. Treat the visible crosshair-to-torso displacement and its measured trend across the chronological history frames as the primary geometric evidence. The normalized right-stick numbers provide direction but are not pixel displacements: never assume a small nonzero value can close a large visible gap. Extrapolate the observed screen-space trend only across the few listed frames up to the shot. If the torso would still be clearly outside the crosshair or plausible shotgun spread at the shot, output miss.

For a stationary world target, turning the camera right with right-stick +x makes the target slide left in the image relative to the fixed center line; turning left with -x makes it slide right. Likewise, looking down with +y makes the target slide up, and looking up with -y makes it slide down. Use chronological history frames to estimate whether that visible trend reaches the center at the fire frame. Do not reverse camera-turn direction and target screen-motion direction.

Only right-stick x/y rotate the camera and change the aim line. Left-stick x/y translate the player's body; never describe a left-stick value as looking up/down or turning left/right. Treat any parallax from left-stick translation conservatively instead of converting it into camera rotation.

enemy_posture describes the tracked enemy in the latest supplied history image. Keep evidence to one concrete sentence of at most 35 words and explicitly state the visible aim geometry and planned camera motion. Never claim to have seen the future shot frame; only predict it from the listed history frames."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8012/v1",
        model: str = "internvl3-8b",
        timeout_seconds: float = 120.0,
        max_tokens: int = 384,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    @staticmethod
    def _prepare_vlm_image(image: Image.Image, *, max_side: int = 896) -> Image.Image:
        ''

        rgb = image.convert("RGB")
        if max(rgb.size) > max_side:
            scale = max_side / max(rgb.size)
            rgb = rgb.resize(
                (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
                Image.Resampling.LANCZOS,
            )
        return draw_center_crosshair(rgb)

    @classmethod
    def _image_data_url(cls, image: Image.Image) -> str:
        buffer = io.BytesIO()
                                                                         
                                                                         
                                                
        cls._prepare_vlm_image(image).save(buffer, format="PNG", compress_level=3)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @classmethod
    def _single_tile_image_data_url(cls, image: Image.Image) -> str:
        ''

        rgb = image.convert("RGB")
        scale = min(448.0 / rgb.width, 448.0 / rgb.height)
        resized = rgb.resize(
            (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
            Image.Resampling.LANCZOS,
        )
        canvas = Image.new("RGB", (448, 448), (0, 0, 0))
        canvas.paste(
            resized,
            ((448 - resized.width) // 2, (448 - resized.height) // 2),
        )
        canvas = draw_center_crosshair(canvas)
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG", compress_level=3)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _direct_focus_image_data_url(
        image: Image.Image, *, crop_size: int
    ) -> str:
        ''

        focused = center_crosshair_crop(image, crop_size=crop_size)
        focused = draw_direct_crosshair(focused)
        buffer = io.BytesIO()
        focused.save(buffer, format="PNG", compress_level=3)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _aim_target_image_data_url(
        image: Image.Image, *, crop_size: int
    ) -> str:
        ''

        focused = center_crosshair_crop(image, crop_size=crop_size)
        focused = draw_aim_center_ring(focused)
        buffer = io.BytesIO()
        focused.save(buffer, format="PNG", compress_level=3)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _weapon_geometry_flags(weapon_context: str | None) -> tuple[bool, bool]:
        lowered = (weapon_context or "").lower()
        spread_weapon = "shotgun" in lowered or "pellet spread" in lowered
        projectile_weapon = any(
            marker in lowered
            for marker in (
                "projectile",
                "energy bolt",
                "plasma bolt",
                "rocket",
                "grenade",
            )
        ) and "hitscan" not in lowered
        return spread_weapon, projectile_weapon

    def observe_aim_geometry(
        self,
        *,
        images: Sequence[Image.Image],
        frame_indices: Sequence[int],
        shot_frame: int,
        weapon_context: str | None,
        enemy_context: str | None,
        projected: bool,
        projection_note: str | None = None,
        focus_crop_size: int | None = None,
    ) -> tuple[VLMObservation, AimGeometryObservation]:
        ''

        import requests

        if not images or len(images) != len(frame_indices):
            raise ValueError(
                "images and frame_indices must be non-empty and have equal length"
            )
        if shot_frame not in frame_indices:
            raise ValueError(
                f"Exact fire frame {shot_frame} is missing from {list(frame_indices)}"
            )
        if focus_crop_size is not None:
            if focus_crop_size <= 0:
                raise ValueError("focus_crop_size must be positive")
            if len(images) != 1 or list(frame_indices) != [shot_frame]:
                raise ValueError(
                    "Direct focus-crop geometry requires exactly the fire frame"
                )
        weapon_text = weapon_context or (
            "The active weapon is a hitscan machine gun using primary fire."
        )
        enemy_text = enemy_context or (
            "The tracked enemy is the same humanoid opponent across images."
        )
        spread_weapon, projectile_weapon = self._weapon_geometry_flags(weapon_text)
        if focus_crop_size is not None and projected:
            timing_text = (
                "This is the controller-projected fire frame. The measured four-axis "
                "controller displacement has already been applied; do not apply it again. "
                f"Projection details: {projection_note or 'not supplied'}."
            )
        elif focus_crop_size is not None:
            timing_text = "This is the actually generated fire frame."
        elif projected:
            timing_text = (
                "Images before the fire-frame index are real causal history. The image at "
                "the fire-frame index is a controller-produced approximation: the latest "
                "real causal frame has already been translated by the measured four-axis "
                "controller-to-pixel calibration. Use history to preserve target identity "
                "and reject an implausible projection, but judge crosshair overlap directly "
                "in the projected fire frame. Do not apply the joystick a second time. "
                f"Projection details: {projection_note or 'not supplied'}."
            )
        else:
            timing_text = (
                "The image labeled as the fire frame is the actually generated fire frame. "
                "Any later images are actual post-shot frames and are used only for visible "
                "projectile contact and the final enemy posture."
            )
        view_text = (
            f"The sole image is a fixed {focus_crop_size}x{focus_crop_size} crop centered "
            "exactly on the full-frame aim point, enlarged to one 448x448 tile. It is not an "
            "enemy detector crop and contains no box. Four magenta crosshair arms point toward "
            "the exact center, while the small central gap is deliberately left unpainted so "
            "the underlying scene remains visible."
            if focus_crop_size is not None
            else (
                "Every image is letterboxed into one full 448x448 tile, and the artificial "
                "hollow black/white crosshair marks the exact center of the original full "
                "frame without covering the underlying aim-point pixels. Black bars are "
                "padding, not game content."
            )
        )
        aim_instruction = (
            "Judge practical gameplay alignment, not exact-pixel segmentation. Set "
            "crosshair_roughly_on_enemy=true when the reticle center or arms lie on any visible "
            "part of the tracked enemy, including the head, torso, arms, or legs. Also set it "
            "true when the reticle is immediately beside the body with a gap no larger than "
            "about one apparent target-body width: this controller intentionally uses a coarse "
            "'enemy near the crosshair' hit test. Do not restrict a hit to the torso. Set it "
            "false only when the enemy is absent from this aim crop or is separated from the "
            "central reticle neighborhood by a clear margin. For a described shotgun or pellet weapon, "
            "within_weapon_spread may cover a visibly wider near-reticle offset. Do not require "
            "the exact center pixel to lie inside the silhouette."
            if focus_crop_size is not None
            else (
                "The compatibility field center_dot_overlaps_enemy means that the central "
                "reticle region is on or closely aligned with the tracked enemy's main body; "
                "it does not refer to a painted dot and does not require exact-pixel overlap. "
                "A small edge offset counts when the shot line would plausibly intersect the "
                "enemy. enemy_side is relative to the hollow reticle in the fire frame. "
                "within_weapon_spread may be true only for a described pellet/spread weapon "
                "whose plausible spread reaches the enemy despite a wider offset. "
                "visible_projectile_contact may be true only when a later supplied frame "
                "visibly shows the described projectile contacting the enemy; muzzle flash, "
                "tracers, and lingering effects alone are not contact."
            )
        )
        text = (
            "Inspect explicit aim geometry in first-person shooter frames. "
            f"{view_text} The player's foreground gun, scope, optic, hands, and arms are never "
            "the tracked enemy.\n\n"
            f"Tracked enemy: {enemy_text}\n"
            f"Weapon: {weapon_text}\n"
            f"{timing_text}\n"
            f"The primary-fire pulse occurs at frame {shot_frame}. {aim_instruction} "
            "enemy_posture describes the tracked "
            "enemy in the final supplied image. Keep evidence to one concrete sentence."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for frame_index, image in zip(frame_indices, images, strict=True):
            if frame_index == shot_frame:
                label = "PROJECTED FIRE FRAME" if projected else "EXACT FIRE FRAME"
            else:
                label = "causal history frame" if projected else "post-shot frame"
            content.extend(
                [
                    {"type": "text", "text": f"{label}: frame {frame_index}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                self._direct_focus_image_data_url(
                                    image, crop_size=focus_crop_size
                                )
                                if focus_crop_size is not None
                                else self._single_tile_image_data_url(image)
                            )
                        },
                    },
                ]
            )
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": (
                    self._FOCUS_AIM_GEOMETRY_SCHEMA
                    if focus_crop_size is not None
                    else self._AIM_GEOMETRY_SCHEMA
                ),
            },
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=request,
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise RuntimeError(
                f"Aim-geometry VLM request failed with HTTP {response.status_code}: "
                f"{response.text}"
            )
        payload = response.json()
        raw = str(payload["choices"][0]["message"]["content"])
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"VLM returned invalid aim-geometry JSON: {raw}") from exc
        if focus_crop_size is not None:
            enemy_visible = bool(parsed.get("enemy_visible"))
            roughly_aligned = bool(parsed.get("crosshair_roughly_on_enemy"))
            geometry = AimGeometryObservation.from_mapping(
                {
                    "enemy_visible": enemy_visible,
                    "enemy_side": (
                        "center"
                        if roughly_aligned
                        else "not_visible"
                        if not enemy_visible
                        else "uncertain"
                    ),
                    "center_dot_overlaps_enemy": roughly_aligned,
                    "within_weapon_spread": bool(
                        parsed.get("within_weapon_spread")
                    ),
                    "visible_projectile_contact": False,
                    "enemy_posture": parsed.get("enemy_posture"),
                    "confidence": parsed.get("confidence"),
                    "evidence": parsed.get("evidence", ""),
                },
                raw_response=raw,
            )
        else:
            geometry = AimGeometryObservation.from_mapping(parsed, raw_response=raw)
        observation = geometry.to_observation(
            spread_weapon=spread_weapon,
            projectile_weapon=projectile_weapon,
        )
        if (
            focus_crop_size is not None
            and observation.shot_result == "uncertain"
            and not geometry.enemy_visible
            and not geometry.center_dot_overlaps_enemy
            and not geometry.within_weapon_spread
        ):
            observation = VLMObservation(
                enemy_visible=False,
                shot_result="miss",
                enemy_posture=geometry.enemy_posture,
                confidence=geometry.confidence,
                evidence=(
                    geometry.evidence
                    or "The tracked enemy is absent from the aim-centered crop."
                ),
                raw_response=geometry.raw_response,
            )
        return observation, geometry

    def observe_hit_likelihood(
        self,
        *,
        image: Image.Image,
        crop_size: int,
        weapon_context: str | None,
        enemy_context: str | None,
        projected: bool,
    ) -> HitLikelihoodObservation:
        ''

        import requests

        if crop_size <= 0:
            raise ValueError("crop_size must be positive")
        weapon_text = weapon_context or (
            "The active weapon is a hitscan machine gun using primary fire."
        )
        enemy_text = enemy_context or (
            "The tracked enemy is the visible hostile character."
        )
        frame_text = (
            "This is the controller-projected moment when the player will fire. "
            "The controller motion has already been applied to the image."
            if projected
            else "This is the decoded moment when the player fires."
        )
        text = (
            "Will this primary-fire shot likely hit the tracked enemy?\n\n"
            f"The image is a {crop_size}x{crop_size} crop centered on the player's aim at the "
            "firing moment. The hollow crosshair marks the aim and does not cover the center "
            "pixels. The crop includes surrounding scene context.\n"
            f"Tracked enemy: {enemy_text}\n"
            f"Weapon: {weapon_text}\n"
            f"{frame_text}\n"
            "Use normal gameplay judgment and the described weapon behavior; pixel-perfect "
            "overlap is not required. Return can_hit=false when the crosshair points at the "
            "environment rather than the enemy, even if the enemy is visible elsewhere in "
            "the crop. Do not infer a desired outcome, health, or death."
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": text},
            {"type": "text", "text": "AIM-CENTERED FIRING CROP"},
            {
                "type": "image_url",
                "image_url": {
                    "url": self._direct_focus_image_data_url(
                        image, crop_size=crop_size
                    )
                },
            },
        ]
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": self._HIT_LIKELIHOOD_SCHEMA,
            },
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=request,
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise RuntimeError(
                f"Hit-likelihood VLM request failed with HTTP {response.status_code}: "
                f"{response.text}"
            )
        payload = response.json()
        raw = str(payload["choices"][0]["message"]["content"])
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"VLM returned invalid hit-likelihood JSON: {raw}") from exc
        return HitLikelihoodObservation.from_mapping(parsed, raw_response=raw)

    def observe_aim_target(
        self,
        *,
        image: Image.Image,
        crop_size: int,
    ) -> AimTargetObservation:
        ''

        import requests

        if crop_size <= 0:
            raise ValueError("crop_size must be positive")
        text = AIM_TARGET_RING_NEUTRAL_PROMPT
        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._aim_target_image_data_url(
                                    image, crop_size=crop_size
                                )
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": self._AIM_TARGET_SCHEMA,
            },
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=request,
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise RuntimeError(
                f"Aim-target VLM request failed with HTTP {response.status_code}: "
                f"{response.text}"
            )
        payload = response.json()
        raw = str(payload["choices"][0]["message"]["content"])
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"VLM returned invalid aim-target JSON: {raw}") from exc
        return AimTargetObservation.from_mapping(parsed, raw_response=raw)

    def observe_combat_guidance_realization(
        self,
        *,
        before_image: Image.Image,
        before_frame_index: int,
        after_images: Sequence[Image.Image],
        after_frame_indices: Sequence[int],
    ) -> CombatGuidanceRealizationObservation:
        ''

        import requests

        if not after_images or len(after_images) != len(after_frame_indices):
            raise ValueError(
                "after_images and after_frame_indices must be non-empty and equal length"
            )
        if any(index <= before_frame_index for index in after_frame_indices):
            raise ValueError("Combat-guidance after frames must follow the before frame")
        text = (
            "Check whether the previous combat guidance was visibly realized in this "
            "generated first-person shooter sequence. The first image is immediately "
            "before one short hitscan shot; the later images are chronological after it. "
            "Track the living humanoid opponent visible before the shot. Stage-1 decoding "
            "is blurry and may smear a falling body, so use a lenient temporal judgment: "
            "realized=true when that same opponent clearly starts falling, is knocked down, "
            "disintegrates, or disappears from the same location immediately after the shot. "
            "Do not require a crisp corpse. realized=false only when the opponent remains "
            "visibly upright and unaffected. Use uncertain only when the temporal change "
            "cannot be distinguished from camera motion or obstruction. The player's "
            "foreground weapon is not the opponent. Keep evidence to one concrete sentence."
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": text},
            {"type": "text", "text": f"BEFORE SHOT frame {before_frame_index}"},
            {
                "type": "image_url",
                "image_url": {"url": self._image_data_url(before_image)},
            },
        ]
        for frame_index, image in zip(
            after_frame_indices, after_images, strict=True
        ):
            content.extend(
                [
                    {"type": "text", "text": f"AFTER SHOT frame {frame_index}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_data_url(image)},
                    },
                ]
            )
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": self._COMBAT_GUIDANCE_REALIZATION_SCHEMA,
            },
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=request,
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise RuntimeError(
                "Combat-guidance VLM request failed with HTTP "
                f"{response.status_code}: {response.text}"
            )
        payload = response.json()
        raw = str(payload["choices"][0]["message"]["content"])
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"VLM returned invalid combat-guidance JSON: {raw}"
            ) from exc
        return CombatGuidanceRealizationObservation.from_mapping(
            parsed, raw_response=raw
        )

    def healthcheck(self) -> dict[str, Any]:
        import requests

        response = requests.get(f"{self.base_url}/models", timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        served = [item.get("id") for item in payload.get("data", [])]
        if self.model not in served:
            raise RuntimeError(f"VLM model {self.model!r} not in served models: {served}")
        return payload

    def observe_navigation(
        self,
        *,
        images: Sequence[Image.Image],
        frame_indices: Sequence[int],
        controller_v2: bool = False,
        verified_enemy_state: dict | None = None,
    ) -> FPSNavigationObservation:
        ''

        import requests

        if not images or len(images) != len(frame_indices):
            raise ValueError(
                "Navigation images and frame_indices must be non-empty and equal length"
            )
        if controller_v2:
            navigation_instruction = (
                "Analyze two chronological full-frame views of a first-person shooter. The "
                "final frame is the current view. Use the preceding frame both for opponent "
                "identity and for detecting approach, lack of progress, or an enlarging wall; "
                "all reported coordinates must still describe the final frame. First look carefully "
                "for a living person-sized opponent. The large gun and hands at the bottom "
                "belong to the player, not the enemy. The generated Stage-1 preview can make "
                "the opponent blurry, low-detail, or mildly distorted: accept a coherent "
                "person-like or armored silhouette with a head-and-torso body layout even "
                "when facial detail, exact limbs, armor texture, or the held weapon is unclear. "
                "Do not require all those details at once. Do not invent an enemy from an "
                "isolated wall pattern, shadow, blood mark, muzzle flash, or particle effect. "
                "If a plausible living opponent is visible in the final frame, locate the "
                "approximate center of its upper torso and return enemy_x_norm and enemy_y_norm "
                "in full-image coordinates, with left/top=0 and right/bottom=1. Aim at the "
                "chest, not feet, weapon, shadow, blood, or the player's foreground gun. If "
                "no plausible living opponent is visible in the final frame, use not_visible "
                "labels and return 0.5 for both coordinates.\n\n"
                "Act as a conservative collision-safety controller for the next 0.8 seconds. "
                "Forward is clear ONLY when a continuous navigable floor corridor extends "
                "from the bottom-center toward the middle distance. A wall panel spanning the "
                "image center, a wall base crossing the lower-center route, a flat wall surface "
                "filling the central view, or a rapidly enlarging wall means blocked even if a "
                "small strip of floor is visible at the player's feet. If depth is ambiguous, "
                "do not choose clear; choose blocked or corner. If the two frames show little "
                "useful progress or a larger wall after forward motion, do not repeat forward. "
                "For blocked or corner, recommend strafe_left, strafe_right, or backward and "
                "turn toward the visibly more open side; never recommend forward. The player's "
                "foreground gun is not an obstacle. If the enemy is visible, aiming will be "
                "handled separately and must not make you call a wall clear. Keep evidence to "
                "one short sentence describing both the opponent, if any, and the floor corridor "
                "or wall base."
            )
            selected_images = images[-2:]
            selected_indices = frame_indices[-2:]
        else:
            navigation_instruction = (
                "You control the next 0.8 seconds of a generated first-person shooter "
                "rollout. The supplied images are chronological; the last image is the "
                "current player view. A small artificial crosshair marks exact image "
                "center only for aiming and is not a world object. Do not mistake the "
                "player's foreground weapon for an enemy.\n\n"
                "Keep the player moving through open space without walking into a nearby "
                "wall, obstacle, or corner. Classify the central forward path from the "
                "latest image. Recommend forward only when the lower-center route is open; "
                "otherwise choose the safer strafe/backward movement and a turn toward the "
                "more open side. The previous interval always had nonzero movement; if the "
                "chronological images show almost no scene progress, treat that as evidence "
                "of being blocked and do not repeat forward. Never recommend standing still.\n\n"
                "If a living humanoid enemy is visible, report its torso position relative "
                "to the image-center crosshair. left/right means a moderate offset; "
                "far_left/far_right means near that side of the image. above/below means "
                "the torso center is above/below the crosshair. The deterministic controller "
                "will aim and fire; do not reason about hidden health or desired outcomes. "
                "Use one short concrete evidence sentence."
            )
            selected_images = images
            selected_indices = frame_indices
        if len(selected_images) == 1:
            navigation_instruction = (
                "This is a cold start with ONE initial image and NO previous movement. "
                "Choose the first 0.8 seconds of movement and aim using this image. "
                "Do not infer motion, approach, lack of progress or a stuck condition "
                "from temporal comparisons: no earlier frame exists. Report visible "
                "enemy torso coordinates relative to image center, posture and visibility. "
                "The foreground gun and hands belong to the player, not an enemy. "
                "Assess path_ahead from visible floor and obstacles only. Recommend "
                "safe movement and turn; do not infer HP, death or desired outcomes. "
                "Give one concrete evidence sentence."
            )
        if verified_enemy_state is not None:
            navigation_instruction += (
                ("\nCold-start initial rule state; no preceding block exists or has been audited: "
                 if len(selected_images) == 1 else
                 "\nThe preceding block has already been audited. Authoritative enemy state: ")
                + json.dumps(verified_enemy_state, sort_keys=True)
                + ". You control PLAYER movement and aiming only, not enemy movement, "
                "HP, death, respawn or story outcomes. Do not override this state. "
                "For corpse/despawn phases do not target or attack the corpse; navigate safely. "
                "For an active but offscreen enemy, search by turning toward open space. "
                "Visibility is not a vote on death or respawn."
            )
        response_schema = (self._FPS_NAVIGATION_V2_SCHEMA if controller_v2
                           else self._FPS_NAVIGATION_SCHEMA)
        navigation_instruction += (
            "\nReturn exactly one compact JSON object matching the complete schema below. "
            "Include every required field, then close the object and stop. No markdown or "
            "whitespace padding. Keep evidence under 20 words. Coordinates describe the "
            "visible torso in the full image: x=0 left, x=1 right, y=0 top, y=1 bottom. "
            "Do not default a visible enemy to (0.5,0.5); estimate its actual position. "
            "When no enemy is visible, use not_visible labels and coordinates (0.5,0.5); "
            "do not invent its offscreen position or posture. Schema: "
            + json.dumps(response_schema['schema'], separators=(',', ':'))
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": navigation_instruction,
            }
        ]
        for frame_index, image in zip(selected_indices, selected_images, strict=True):
            content.extend(
                [
                    {"type": "text", "text": f"Chronological frame {frame_index}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_data_url(image)},
                    },
                ]
            )
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": (
                    self._FPS_NAVIGATION_V2_SCHEMA
                    if controller_v2
                    else self._FPS_NAVIGATION_SCHEMA
                ),
            },
        }
        for attempt in range(2):
            response = requests.post(
                f"{self.base_url}/chat/completions", json=request,
                timeout=self.timeout_seconds,
            )
            if not response.ok:
                raise RuntimeError(f"Navigation VLM HTTP {response.status_code}: {response.text}")
            choice = response.json()['choices'][0]
            raw = str(choice['message']['content'])
            finish_reason = choice.get('finish_reason')
            try:
                if finish_reason != 'stop':
                    raise ValueError(f'Incomplete navigation response: finish_reason={finish_reason}')
                parsed = json.loads(raw)
                _validate_navigation_payload(parsed, response_schema['schema'])
                break
            except ValueError as exc:
                print(f'[vlm-navigation] attempt={attempt + 1} finish_reason={finish_reason} '
                      f'error={exc} raw={raw[:600]!r}', flush=True)
                if attempt == 1:
                    raise ValueError('Navigation VLM invalid after two attempts; no action committed') from exc
                                                                                        
                content[0]['text'] += (
                    '\nYour previous response was incomplete or invalid. Output ALL required '
                    'fields in one compact JSON object, including both coordinates when required. '
                    'Close the object immediately after the last field; do not append whitespace.'
                )
        if controller_v2 and not bool(parsed.get("enemy_visible")):
            parsed.update(
                {
                    "enemy_side": "not_visible",
                    "enemy_vertical": "not_visible",
                    "enemy_posture": "not_visible",
                    "enemy_x_norm": 0.5,
                    "enemy_y_norm": 0.5,
                }
            )
        elif controller_v2:
            x_norm = float(parsed.get("enemy_x_norm", 0.5))
            y_norm = float(parsed.get("enemy_y_norm", 0.5))
            parsed["enemy_side"] = (
                "far_left"
                if x_norm < 0.18
                else "left"
                if x_norm < 0.42
                else "center"
                if x_norm <= 0.58
                else "right"
                if x_norm <= 0.82
                else "far_right"
            )
            parsed["enemy_vertical"] = (
                "above" if y_norm < 0.42 else "below" if y_norm > 0.58 else "center"
            )
            if parsed.get("enemy_posture") == "not_visible":
                parsed["enemy_posture"] = "uncertain"
        return FPSNavigationObservation.from_mapping(parsed, raw_response=raw)

    def observe(
        self,
        *,
        images: Sequence[Image.Image],
        frame_indices: Sequence[int],
        event_kind: Literal["shot", "state_check"],
        shot_frame: int | None = None,
        weapon_context: str | None = None,
        enemy_context: str | None = None,
    ) -> VLMObservation:
        import requests

        if not images or len(images) != len(frame_indices):
            raise ValueError("images and frame_indices must be non-empty and have equal length")
        if event_kind == "shot" and shot_frame is None:
            raise ValueError("shot events require shot_frame")
        if event_kind == "state_check" and shot_frame is not None:
            raise ValueError("state_check events cannot have shot_frame")

        weapon_text = weapon_context or (
            "The active weapon is a hitscan machine gun using primary fire."
        )
        enemy_text = enemy_context or "The tracked enemy is the same humanoid robot across frames."
        event_text = (
            "A single short primary-fire pulse from the described weapon occurs at "
            f"frame {shot_frame}. Determine whether that pulse visibly hit the tracked enemy."
            if event_kind == "shot"
            else "This is a state check with no shot to classify. Determine the tracked enemy's final posture."
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    self._SYSTEM_PROMPT
                    + "\n\n"
                    f"event_kind={event_kind}. {event_text} "
                    f"Weapon description: {weapon_text}\n"
                    f"Tracked enemy description: {enemy_text}\n"
                    f"The following images correspond in order to frame numbers {list(frame_indices)}."
                ),
            }
        ]
        for frame_index, image in zip(frame_indices, images, strict=True):
            content.append(
                {
                    "type": "text",
                    "text": f"Decoded frame {frame_index}; the artificial crosshair marks exact image center.",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(image)},
                }
            )

        request = {
            "model": self.model,
                                                                             
                                                                            
                                                                              
                                                                  
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_schema", "json_schema": self._JSON_SCHEMA},
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=request,
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise RuntimeError(
                f"VLM request failed with HTTP {response.status_code}: {response.text}"
            )
        payload = response.json()
        raw = str(payload["choices"][0]["message"]["content"])
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"VLM returned invalid structured JSON: {raw}") from exc
        return VLMObservation.from_mapping(parsed, raw_response=raw)

    def predict_shot(
        self,
        *,
        images: Sequence[Image.Image],
        frame_indices: Sequence[int],
        shot_frame: int,
        interval_start_frame: int,
        interval_end_frame: int,
        action_plan: str,
        weapon_context: str | None = None,
        enemy_context: str | None = None,
        opponent_distance: str | None = None,
    ) -> VLMObservation:
        ''

        import requests

        if not images or len(images) != len(frame_indices):
            raise ValueError("images and frame_indices must be non-empty and have equal length")
        if interval_start_frame > shot_frame or shot_frame > interval_end_frame:
            raise ValueError(
                f"shot frame {shot_frame} is outside interval "
                f"[{interval_start_frame}, {interval_end_frame}]"
            )
        weapon_text = weapon_context or (
            "The active weapon is a hitscan machine gun using primary fire."
        )
        enemy_text = enemy_context or "The tracked enemy is the same humanoid robot across images."
        distance_text = opponent_distance or "unspecified"
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    self._PRE_ACTION_SYSTEM_PROMPT
                    + "\n\n"
                    f"Weapon description: {weapon_text}\n"
                    f"Tracked enemy description: {enemy_text}\n"
                    f"Approximate opponent distance: {distance_text}.\n"
                    f"The latest causal history frames are {list(frame_indices)}. "
                    f"The upcoming generated interval is frames {interval_start_frame} through "
                    f"{interval_end_frame}, and one short primary-fire pulse occurs at absolute "
                    f"frame {shot_frame}.\n"
                    f"Upcoming controller plan: {action_plan}"
                ),
            }
        ]
        for frame_index, image in zip(frame_indices, images, strict=True):
            content.append(
                {
                    "type": "text",
                    "text": f"Causal history frame {frame_index}; the artificial crosshair marks exact image center.",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(image)},
                }
            )

        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_schema", "json_schema": self._JSON_SCHEMA},
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=request,
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise RuntimeError(
                f"VLM request failed with HTTP {response.status_code}: {response.text}"
            )
        payload = response.json()
        raw = str(payload["choices"][0]["message"]["content"])
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"VLM returned invalid structured JSON: {raw}") from exc
        return VLMObservation.from_mapping(parsed, raw_response=raw)

    def predict_shot_grounded(
        self,
        *,
        images: Sequence[Image.Image],
        frame_indices: Sequence[int],
        shot_frame: int,
        weapon_context: str,
        enemy_context: str,
    ) -> tuple[VLMObservation, dict[str, Any]]:
        ''

        import requests

        if len(images) < 2 or len(images) != len(frame_indices):
            raise ValueError(
                "Grounded shot prediction requires at least two equally indexed images"
            )
        box_schema = {
            "type": "object",
            "properties": {
                "frame": {"type": "integer"},
                "enemy_visible": {"type": "boolean"},
                "x1": {"type": "integer", "minimum": 0, "maximum": 1000},
                "y1": {"type": "integer", "minimum": 0, "maximum": 1000},
                "x2": {"type": "integer", "minimum": 0, "maximum": 1000},
                "y2": {"type": "integer", "minimum": 0, "maximum": 1000},
            },
            "required": ["frame", "enemy_visible", "x1", "y1", "x2", "y2"],
            "additionalProperties": False,
        }
        response_schema = {
            "name": "fps_enemy_track_boxes",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "boxes": {
                        "type": "array",
                        "minItems": len(images),
                        "maxItems": len(images),
                        "items": box_schema,
                    }
                },
                "required": ["boxes"],
                "additionalProperties": False,
            },
        }
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Track the same described enemy across exactly the supplied chronological "
                    "decoded FPS frames. For each frame return a tight full-body bounding box "
                    "in normalized full-image coordinates [0,1000], origin top-left. Exclude "
                    "the player's foreground weapon. The artificial center crosshair is only "
                    "a coordinate guide at (500,500), not part of the enemy. Do not classify "
                    "hit or miss and do not invent future motion. "
                    f"Tracked enemy: {enemy_context}"
                ),
            }
        ]
        for frame_index, image in zip(frame_indices, images, strict=True):
            content.extend(
                [
                    {"type": "text", "text": f"Frame {frame_index}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_data_url(image)},
                    },
                ]
            )
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": max(256, 80 * len(images)),
            "response_format": {"type": "json_schema", "json_schema": response_schema},
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=request,
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise RuntimeError(
                f"Grounding VLM request failed with HTTP {response.status_code}: {response.text}"
            )
        payload = response.json()
        raw = str(payload["choices"][0]["message"]["content"])
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Grounding VLM returned invalid structured JSON: {raw}") from exc
        boxes = parsed.get("boxes")
        if not isinstance(boxes, list) or len(boxes) != len(frame_indices):
            raise ValueError(
                f"Grounding VLM returned wrong box count: expected={len(frame_indices)} raw={raw}"
            )
        normalized_boxes: list[dict[str, Any]] = []
        for expected_frame, box in zip(frame_indices, boxes, strict=True):
            if not isinstance(box, dict) or int(box.get("frame", -1)) != expected_frame:
                raise ValueError(
                    f"Grounding VLM frame mismatch: expected={expected_frame} box={box}"
                )
            normalized = {
                "frame": expected_frame,
                "enemy_visible": bool(box["enemy_visible"]),
                **{field: int(box[field]) for field in ("x1", "y1", "x2", "y2")},
            }
            if normalized["enemy_visible"] and (
                normalized["x1"] >= normalized["x2"]
                or normalized["y1"] >= normalized["y2"]
            ):
                raise ValueError(f"Grounding VLM returned invalid visible box: {normalized}")
            normalized_boxes.append(normalized)

        observation, geometry = classify_grounded_shot(
            normalized_boxes,
            shot_frame=shot_frame,
            weapon_context=weapon_context,
        )
        observation = VLMObservation(
            **{**observation.to_dict(), "raw_response": raw}
        )
        geometry.update(
            {
                "schema_version": "vlm_grounded_shot_geometry_v1",
                "model": self.model,
                "shot_frame": shot_frame,
                "raw_response": raw,
            }
        )
        return observation, geometry


@dataclass(frozen=True)
class RuleTransition:
    event_id: str
    event_kind: Literal["shot", "state_check"]
    hp_before: int
    hp_after: int
    counted_hit: bool
    next_mode: InteractionMode
    death_visually_confirmed: bool
    violations: tuple[str, ...]
    observation: VLMObservation

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "observation": self.observation.to_dict(),
        }


@dataclass(frozen=True)
class GuidanceAudit:
    ''

    expected_mode: InteractionMode
    fulfilled: bool | None
    actual_shot_result: ShotResult
    actual_enemy_posture: EnemyPosture
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_interaction_guidance(
    *,
    expected_mode: InteractionMode,
    observation: VLMObservation,
    min_confidence: float,
) -> GuidanceAudit:
    ''

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    if (
        observation.confidence < min_confidence
        or observation.shot_result == "uncertain"
        or observation.enemy_posture == "uncertain"
    ):
        fulfilled: bool | None = None
        reason = "Generated evidence is uncertain; leave the guidance audit unresolved."
    elif expected_mode == "hit_alive":
        fulfilled = (
            observation.shot_result == "hit"
            and observation.enemy_posture == "standing"
        )
        reason = (
            "A nonlethal hit and a still-standing enemy were both observed."
            if fulfilled
            else "The generated shot/contact or surviving posture did not match hit_alive."
        )
    elif expected_mode == "miss_alive":
        fulfilled = (
            observation.shot_result == "miss"
            and observation.enemy_posture == "standing"
        )
        reason = (
            "A miss and a still-standing enemy were both observed."
            if fulfilled
            else "The generated shot/contact or surviving posture did not match miss_alive."
        )
    elif expected_mode == "collapse":
        fulfilled = (
            observation.shot_result in {"hit", "not_applicable"}
            and observation.enemy_posture in {"falling", "down"}
        )
        reason = (
            "A hit and the requested falling/down posture were both observed."
            if fulfilled
            else "The generated shot/contact or death animation did not match collapse."
        )
    elif expected_mode == "persist_killed":
        fulfilled = observation.enemy_posture == "down"
        reason = (
            "The enemy remained down as requested."
            if fulfilled
            else "The generated posture did not preserve the killed/down state."
        )
    else:
        fulfilled = observation.enemy_posture == "standing"
        reason = (
            "The enemy remained active as requested."
            if fulfilled
            else "The generated posture did not preserve the active state."
        )
    return GuidanceAudit(
        expected_mode=expected_mode,
        fulfilled=fulfilled,
        actual_shot_result=observation.shot_result,
        actual_enemy_posture=observation.enemy_posture,
        confidence=observation.confidence,
        reason=reason,
    )


class TwoConfirmedHitKillRule:
    ''

    def __init__(
        self,
        *,
        min_hit_confidence: float = 0.55,
        initial_hp: int = 100,
        confirmed_hit_damage: int = 50,
    ) -> None:
        if not 0.0 <= min_hit_confidence <= 1.0:
            raise ValueError("min_hit_confidence must be in [0, 1]")
        if initial_hp <= 0:
            raise ValueError("initial_hp must be positive")
        if confirmed_hit_damage <= 0:
            raise ValueError("confirmed_hit_damage must be positive")
        self.hp = initial_hp
        self.initial_hp = initial_hp
        self.confirmed_hit_damage = confirmed_hit_damage
        self.min_hit_confidence = min_hit_confidence
        self._observed_event_ids: set[str] = set()
        self.death_requested = False
        self.death_visually_confirmed = False

    def _claim_event(self, event_id: str) -> None:
        if event_id in self._observed_event_ids:
            raise ValueError(f"Event already applied: {event_id}")
        self._observed_event_ids.add(event_id)

    def preview_shot(
        self, event_id: str, observation: VLMObservation
    ) -> RuleTransition:
        ''

        hp_before = self.hp
        counted_hit = (
            observation.shot_result == "hit"
            and observation.confidence >= self.min_hit_confidence
        )
        hp_after = (
            max(0, hp_before - self.confirmed_hit_damage)
            if counted_hit
            else hp_before
        )
        next_mode: InteractionMode = "collapse" if hp_after == 0 else "active"
        return RuleTransition(
            event_id=event_id,
            event_kind="shot",
            hp_before=hp_before,
            hp_after=hp_after,
            counted_hit=counted_hit,
            next_mode=next_mode,
            death_visually_confirmed=False,
            violations=(),
            observation=observation,
        )

    def observe_shot(self, event_id: str, observation: VLMObservation) -> RuleTransition:
        self._claim_event(event_id)
        hp_before = self.hp
        counted_hit = (
            observation.shot_result == "hit"
            and observation.confidence >= self.min_hit_confidence
        )
        if counted_hit:
            self.hp = max(0, self.hp - self.confirmed_hit_damage)

        violations: list[str] = []
        visibly_down = observation.enemy_posture == "down"
        visibly_falling = observation.enemy_posture == "falling"
        if (visibly_down or visibly_falling) and self.hp > 0:
            violations.append("enemy_fell_before_rule_hp_reached_zero")

        if self.hp == 0:
            self.death_requested = True
            self.death_visually_confirmed = visibly_down
            next_mode: InteractionMode = "persist_killed" if visibly_down else "collapse"
        else:
            next_mode = "active"

        return RuleTransition(
            event_id=event_id,
            event_kind="shot",
            hp_before=hp_before,
            hp_after=self.hp,
            counted_hit=counted_hit,
            next_mode=next_mode,
            death_visually_confirmed=self.death_visually_confirmed,
            violations=tuple(violations),
            observation=observation,
        )

    def observe_state(self, event_id: str, observation: VLMObservation) -> RuleTransition:
        self._claim_event(event_id)
        hp_before = self.hp
        violations: list[str] = []
        if observation.shot_result != "not_applicable":
            violations.append("state_check_returned_shot_classification")

        if self.death_requested:
                                                                              
                                                                                
                                                                                
                                                                           
                                                                            
                                                                            
            self.death_visually_confirmed = observation.enemy_posture in {
                "down",
                "not_visible",
            }
            next_mode: InteractionMode = (
                "persist_killed" if self.death_visually_confirmed else "collapse"
            )
        else:
            if observation.enemy_posture in {"falling", "down"}:
                violations.append("enemy_fell_before_rule_hp_reached_zero")
            next_mode = "active"

        return RuleTransition(
            event_id=event_id,
            event_kind="state_check",
            hp_before=hp_before,
            hp_after=self.hp,
            counted_hit=False,
            next_mode=next_mode,
            death_visually_confirmed=self.death_visually_confirmed,
            violations=tuple(violations),
            observation=observation,
        )
