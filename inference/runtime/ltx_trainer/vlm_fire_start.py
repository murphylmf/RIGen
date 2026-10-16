''
from __future__ import annotations

from typing import Any


def apply_fire_start(
    fire: list[bool], *, target_block: int, first_fire_block: int = 1,
) -> tuple[list[bool], dict[str, Any]]:
    ''
    if type(target_block) is not int or target_block < 0:
        raise ValueError("target_block must be a non-negative integer")
    if type(first_fire_block) is not int or first_fire_block < 0:
        raise ValueError("first_fire_block must be a non-negative integer")
    allowed = target_block >= first_fire_block
    committed = list(fire) if allowed else [False] * len(fire)
    return committed, {
        "policy": "episode_start_fire_delay_v1",
        "first_fire_block": first_fire_block,
        "target_block": target_block,
        "fire_allowed": allowed,
        "planner_requested_fire_offsets": [i for i, value in enumerate(fire) if value],
        "committed_fire_offsets": [i for i, value in enumerate(committed) if value],
        "suppressed": any(fire) and not allowed,
    }
