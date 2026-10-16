''
from __future__ import annotations

import base64
import io
import json
from typing import Any

PLAN_PROMPT = """You control only the PLAYER's actions for the next 0.8-second block.
Use only the supplied game image. Aim toward the visible enemy. If the aim is
close to the enemy, fire repeated short pulses while making small aim corrections.
Do not hold the fire button continuously. If no enemy is visible, turn to search
instead of firing blindly. Avoid moving into obstacles.
Do not plan enemy behavior, predict damage, or decide HP, death, or respawn.
Return a JSON object with exactly these scalar fields:
movement: hold, forward, backward, strafe_left, or strafe_right;
turn: hold, left, or right; vertical: hold, up, or down;
fire_mode: pulses to shoot now in short pulses, or off to not shoot.
Do not return lists or explanations. Shooting near an enemy does not require
confirmation of a previous hit."""

AIM_OVERLAP_PROMPT = (
    "Is any part of the enemy character's body inside or touching the lime-green circle? "
    'Return only JSON: {"aim_target":"enemy"} for yes, '
    'or {"aim_target":"background"} for no.'
)
                                                                            
                                                                
FORECAST_PROMPT = AIM_OVERLAP_PROMPT
AUDIT_PROMPT = AIM_OVERLAP_PROMPT
AUDIT_POLICY = "actual_fire_full_frame_binary_aim_hit_v6"

PLAN_FIELDS = {
    "movement": ["hold", "forward", "backward", "strafe_left", "strafe_right"],
    "turn": ["hold", "left", "right"], "vertical": ["hold", "up", "down"],
    "fire_mode": ["off", "pulses"],
}
TARGETS = ["enemy", "background"]
DEFAULT_INITIAL_HP = 100
HP_TEST_LEVELS = (50, 100, 200)


def initial_aim_state(hp: int = DEFAULT_INITIAL_HP) -> dict[str, Any]:
    ''
    if hp not in HP_TEST_LEVELS:
        raise ValueError(f"Unsupported initial HP: {hp}; expected {HP_TEST_LEVELS}")
    return dict(hp=hp, last_alive_hp=hp, phase="active", corpse_blocks=0)


def enemy_motion_direction(block: int, first_direction: str = "left") -> str:
    ''
    if block < 0 or first_direction not in {"left", "right"}:
        raise ValueError("Invalid enemy motion schedule")
    opposite = "right" if first_direction == "left" else "left"
    return first_direction if (block // 3) % 2 == 0 else opposite


def independent_guidance_variants(variants: dict[str, str], weapon_name: str) -> dict[str, str]:
    ''
    legacy = f"The player fires the {weapon_name} while moving"
    neutral = f"The player fires the {weapon_name}"
    return {name: text.replace(legacy, neutral) for name, text in variants.items()}


def single_enemy_guidance_variants(variants: dict[str, str]) -> dict[str, str]:
    ''
    allowed = {"active_left", "active_right", "hit_alive_left", "hit_alive_right",
               "killed_left", "killed_right", "corpse"}
    if not allowed <= variants.keys():
        raise ValueError("Missing single-enemy prompt variants")
    result = {}
    for name, text in variants.items():
        if name not in allowed:
            continue
        text += " There is only this one opponent in the entire scene. No additional opponent enters."
        if name == "corpse" or name.startswith("killed_"):
            text += " The defeated opponent stays down and does not get up or reappear."
        result[name] = text
    return result


def request(observer: Any, prompt: str, image_url: str,
            fields: dict[str, list[str]]) -> dict[str, str]:
    import requests
    schema = {"type": "object", "properties": {
        k: {"type": "string", "enum": v} for k, v in fields.items()},
        "required": list(fields), "additionalProperties": False}
    r = requests.post(f"{observer.base_url}/chat/completions", json={
        "model": observer.model, "temperature": 0, "max_tokens": 256,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}]}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "fire_protocol", "strict": True, "schema": schema}},
    }, timeout=observer.timeout_seconds)
    r.raise_for_status()
    choice = r.json()["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Incomplete fire-protocol response; no result committed")
    result = json.loads(choice["message"]["content"])
    if set(result) != set(fields) or any(result[k] not in values for k, values in fields.items()):
        raise ValueError(f"Invalid fire-protocol response: {result}")
    return result


def plan(observer: Any, image: Any) -> dict[str, str]:
                                                                                 
    return request(observer, PLAN_PROMPT, observer._image_data_url(image), PLAN_FIELDS)


def actions(proposal: dict[str, str], frames: int = 16) -> tuple[list[list[float]], list[bool]]:
    x, y = {"hold": (0., 0.), "forward": (0., -.15), "backward": (0., .15),
            "strafe_left": (-.1, 0.), "strafe_right": (.1, 0.)}[proposal["movement"]]
    rx = {"hold": 0., "left": -.015, "right": .015}[proposal["turn"]]
    ry = {"hold": 0., "up": -.01, "down": .01}[proposal["vertical"]]
                                                                       
    fire = [proposal["fire_mode"] == "pulses" and i % 8 == 2 for i in range(frames)]
    return [[x, y, rx, ry] for _ in range(frames)], fire


def calibrated_actions(proposal: dict[str, str], lookup: Any,
                       frames: int = 16) -> tuple[list[list[float]], list[bool]]:
    ''
    joystick, fire = actions(proposal, frames)
    limits = lookup.linear_abs_ranges
    if limits is None:
        limits = (1.0, 1.0, lookup.right_x_points[-1][0], lookup.right_y_points[-1][0])
    bounded = [[max(-limit, min(limit, value)) for value, limit in zip(row, limits, strict=True)]
               for row in joystick]
    return bounded, fire


def select_fire_frames(fire: list[bool], period: int = 4) -> list[int]:
    ''
    if period < 1:
        raise ValueError("Fire sampling period must be positive")
    indices: list[int] = []
    start = -1
    for i, value in enumerate(fire):
        if not value:
            start = -1
        else:
            if start < 0:
                start = i
            if (i - start) % period == 0:
                indices.append(i)
    return indices


def forecast(observer: Any, image: Any, crop_size: int) -> dict[str, str]:
    return request(observer, FORECAST_PROMPT,
                   observer._aim_target_image_data_url(image, crop_size=crop_size),
                   {"aim_target": TARGETS})


def audit(observer: Any, image: Any) -> dict[str, str]:
    ''
    from ltx_trainer.vlm_agent import draw_aim_center_ring
    buffer = io.BytesIO()
    draw_aim_center_ring(image).save(buffer, format="PNG", compress_level=3)
    image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return request(observer, AUDIT_PROMPT, image_url, {"aim_target": TARGETS})


def hp_after_aim_audit(hp: int, phase: str, targets: list[str]) -> int:
    ''
    if phase == "active" and "enemy" in targets:
        return max(0, hp - 50)
    return hp


def advance_aim_state(state: dict[str, Any], targets: list[str], *,
                      single_enemy: bool = False) -> dict[str, Any]:
    ''
    result = dict(state)
    phase = state["phase"]
    if single_enemy and phase not in {"active", "corpse"}:
        raise ValueError(f"Unexpected single-enemy lifecycle phase: {phase}")
    hp = hp_after_aim_audit(state["hp"], phase, targets)
    if hp < state["hp"]:
        if hp == 0:
            result.update(hp=0, phase="corpse", corpse_blocks=0)
        else:
            result.update(hp=hp, phase="active", last_alive_hp=hp)
    elif phase == "corpse":
        count = state["corpse_blocks"] + 1
        result.update(corpse_blocks=count,
                      phase="corpse" if single_enemy or count < 2 else "despawn")
    elif phase == "despawn":
        result.update(phase="reentry")
    elif phase == "reentry":
        result.update(hp=100, last_alive_hp=100, phase="active", corpse_blocks=0)
    return result
