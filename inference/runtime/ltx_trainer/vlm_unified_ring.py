''
from __future__ import annotations

import json
from typing import Any

from ltx_trainer.vlm_agent import FPSBlockActionPlan

CHOICES = {
    "movement": ["hold", "forward", "backward", "strafe_left", "strafe_right"],
    "turn": ["hold", "left", "right"],
    "vertical": ["hold", "up", "down"],
    "enemy_motion": ["left", "right"],
    "enemy_visible": [True, False],
    "fire_requested": [True, False],
}


def propose(observer: Any, image: Any, state: dict[str, Any], ring: dict[str, Any],
            previous_audit: dict[str, Any] | None) -> dict[str, Any]:
    import requests

    schema = {"type": "object", "properties": {
        k: {"type": "boolean" if isinstance(v[0], bool) else "string", "enum": v}
        for k, v in CHOICES.items()}, "required": list(CHOICES), "additionalProperties": False}
    text = (
        "You are the SINGLE planner for the NEXT 0.8-second FPS video block. "
        "The image is the latest committed video result. Previous audit has ALREADY "
        "reconciled the previous guidance against its video. Do not reinterpret that "
        "audit as a future event. Plan PLAYER movement/aim and ENEMY lateral movement "
        "together. Keep the enemy visible for this short evaluation; do not deliberately "
        "send it offscreen. If visible but not under the ring, turn toward its apparent "
        "side and correct vertical aim; do not just hold indefinitely. Use small "
        "categorical corrections, no coordinates. If enemy "
        "is offscreen, search by turning, not firing. Avoid walking into walls. When "
        "ring observes enemy, hold movement/aim and request fire. Never invent HP, "
        "death or respawn: rules own those. Non-active lifecycle takes precedence. "
        "Return compact JSON only, exactly these keys, no evidence or explanation. "
        f"Allowed fields: {json.dumps(CHOICES)}. "
        f"Verified state: {json.dumps(state)}. Ring observation (NOT a hit): "
        f"{json.dumps({k: ring[k] for k in ('aim_target', 'confidence')})}. "
        f"Previous block audit: {json.dumps(previous_audit)}."
    )
    response = requests.post(f"{observer.base_url}/chat/completions", json={
        "model": observer.model, "temperature": 0, "max_tokens": 256,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": observer._image_data_url(image)}},
        ]}], "response_format": {"type": "json_schema", "json_schema": {
            "name": "unified_discrete_plan", "strict": True, "schema": schema}},
    }, timeout=observer.timeout_seconds)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Unified planner response did not finish; no action committed")
    value = json.loads(choice["message"]["content"])
    if set(value) != set(CHOICES) or any(
        type(value[k]) is not type(v[0]) or value[k] not in v for k, v in CHOICES.items()
    ):
        raise ValueError(f"Invalid unified planner output: {value}")
    return value


def execute(proposal: dict[str, Any], state: dict[str, Any], ring: dict[str, Any],
            threshold: float, block: int) -> tuple[FPSBlockActionPlan, str]:
    ''
    active = state["phase"] == "active"
    aligned = ring["aim_target"] == "enemy" and ring["confidence"] >= threshold
    movement, turn, vertical = proposal["movement"], proposal["turn"], proposal["vertical"]
    if not proposal["enemy_visible"]:
        movement, vertical = "hold", "hold"
        if turn == "hold":
            turn = "left" if (block // 3) % 2 == 0 else "right"
    fire = active and proposal["enemy_visible"] and aligned and proposal["fire_requested"]
    if aligned or not active:
        movement, turn, vertical = "hold", "hold", "hold"
    x, y = {"hold": (0., 0.), "forward": (0., -.15), "backward": (0., .15),
            "strafe_left": (-.1, 0.), "strafe_right": (.1, 0.)}[movement]
    action = FPSBlockActionPlan(x, y, {"hold": 0., "left": -.015, "right": .015}[turn],
                               {"hold": 0., "up": -.01, "down": .01}[vertical], fire,
                               movement, turn)
    direction = proposal["enemy_motion"]
    if active:
                                                                               
        mode = (f"{'killed' if state['hp'] <= 50 else 'hit_alive'}_{direction}" if fire
                else f"active_{direction}" if proposal["enemy_visible"] else "active_reacquire")
    else:
        mode = f"reentry_{direction}" if state["phase"] == "reentry" else state["phase"]
    return action, mode
