from __future__ import annotations

import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

CANONICAL_BUTTONS: tuple[str, ...] = ("A", "B", "X", "Y", "LB", "RB", "LT", "RT")
CROSS_FPS_ALL_BUTTONS: tuple[str, ...] = CANONICAL_BUTTONS
CROSS_FPS_UNIFIED_BUTTONS: tuple[str, ...] = ("aim", "fire", "reload", "melee", "jump")
CROSS_FPS_FOUR_ACTION_BUTTONS: tuple[str, ...] = ("aim", "fire", "reload", "jump")
CROSS_FPS_FIXED_BUTTON_MODES: frozenset[str] = frozenset(
    {"crossfps_all_buttons", "crossfps_unified", "crossfps_4action", "crossfps_legacy6"}
)
CROSS_FPS_ALL_BUTTON_RAW_COLUMNS: dict[str, str] = {
    "A": "south",
    "B": "east",
    "X": "west",
    "Y": "north",
    "LB": "left_shoulder",
    "RB": "right_shoulder",
    "LT": "left_trigger",
    "RT": "right_trigger",
}
CROSS_FPS_UNIFIED_RAW_COLUMNS: dict[str, str] = {
    "aim": "left_trigger",
    "fire": "right_trigger",
    "reload": "west",
    "melee": "right_thumb",
    "jump": "south",
}
CROSS_FPS_UNIFIED_GAME_IDS: frozenset[str] = frozenset(
    {
        "call_of_duty",
        "call_of_duty__modern_warfare_iii",
        "call_of_duty__warzone",
        "halo_infinite",
    }
)
LEGACY_CROSSFPS_BUTTONS: tuple[str, ...] = (
    "right_trigger",
    "left_trigger",
    "right_thumb",
    "south",
    "west",
    "north",
)

BUTTON_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "A": ("south", "a"),
    "B": ("east", "b"),
    "X": ("west", "x"),
    "Y": ("north", "y"),
    "LB": ("left_shoulder", "left_bumper", "lb"),
    "RB": ("right_shoulder", "right_bumper", "rb"),
    "LT": ("left_trigger", "lt"),
    "RT": ("right_trigger", "rt"),
}

BUTTON_NAME_ALIASES: dict[str, str] = {}
for _button, _aliases in BUTTON_COLUMN_ALIASES.items():
    BUTTON_NAME_ALIASES[_button.lower()] = _button
    for _alias in _aliases:
        BUTTON_NAME_ALIASES[_alias.lower()] = _button
BUTTON_NAME_ALIASES.update(
    {
        "l1": "LB",
        "r1": "RB",
        "l2": "LT",
        "r2": "RT",
        "left trigger": "LT",
        "right trigger": "RT",
        "left bumper": "LB",
        "right bumper": "RB",
    }
)

_VOCAB_ID_SAFE_PATTERN = re.compile(r"[^a-z0-9]+")

DEFAULT_CONTROL_SCHEMAS: dict[str, dict[str, dict[str, Any]]] = {
    "resident_evil": {
        "LT": {"meaning": "aim", "aliases": ["aim weapon", "aim the gun"]},
        "RT": {"meaning": "shoot", "aliases": ["fire weapon", "shoot the weapon"]},
        "A": {"meaning": "interact", "aliases": ["interact", "use nearby object"]},
        "X": {"meaning": "reload", "aliases": ["reload weapon"]},
        "Y": {"meaning": "open inventory", "aliases": ["inventory", "open inventory"]},
        "B": {"meaning": "dodge", "aliases": ["evade", "dodge"]},
        "LB": {"meaning": "quick turn", "aliases": ["quick turn"]},
    },
    "lies_of_p": {
        "RB": {"meaning": "light attack", "aliases": ["quick attack", "normal attack"]},
        "RT": {"meaning": "heavy attack", "aliases": ["strong attack", "charged attack"]},
        "LB": {"meaning": "guard", "aliases": ["block", "defend"]},
        "B": {
            "meaning": "dodge",
            "aliases": ["evade", "sidestep"],
            "timeline_meanings": [
                {"max_duration_sec": 0.3, "meaning": "dodge"},
                {"min_duration_sec": 0.3, "meaning": "sprint"},
            ],
        },
        "A": {"meaning": "interact", "aliases": ["interact", "use nearby object"]},
        "X": {"meaning": "use item", "aliases": ["use item", "consume item"]},
        "Y": {"meaning": "switch weapon", "aliases": ["change weapon", "switch weapon"]},
        "LT": {"meaning": "special arm attack", "aliases": ["use arm weapon", "special arm attack"]},
    },
    "sekiro": {
        "RB": {"meaning": "attack", "aliases": ["sword attack", "slash"]},
        "LB": {
            "meaning": "deflect",
            "aliases": ["parry", "guard deflect"],
            "timeline_meanings": [
                {"max_duration_sec": 0.3, "meaning": "deflect"},
                {"min_duration_sec": 0.3, "meaning": "guard"},
            ],
        },
        "B": {
            "meaning": "dodge",
            "aliases": ["evade", "step dodge"],
            "timeline_meanings": [
                {"max_duration_sec": 0.3, "meaning": "dodge"},
                {"min_duration_sec": 0.3, "meaning": "sprint"},
            ],
        },
        "A": {"meaning": "jump", "aliases": ["jump"]},
        "X": {"meaning": "use item", "aliases": ["use item", "consume item"]},
        "RT": {"meaning": "use prosthetic tool", "aliases": ["use prosthetic tool", "special tool"]},
        "LT": {"meaning": "grapple", "aliases": ["grappling hook", "grapple"]},
        "Y": {"meaning": "switch prosthetic tool", "aliases": ["switch tool", "change prosthetic tool"]},
    },
}

CONTROL_TEXT_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("Controls: {pairs}.", "{button} means {meaning}"),
    ("Button bindings: {pairs}.", "pressing {button} performs {meaning}"),
    ("Input mapping: {pairs}.", "{button} -> {meaning}"),
    ("Controller layout: {pairs}.", "{button} is bound to {meaning}"),
    ("Use these controls: {pairs}.", "{button} triggers {meaning}"),
)
PAIR_SEPARATORS: tuple[str, ...] = (". ", "; ", ", ")


def normalize_button_name(name: Any) -> str | None:
    if name is None:
        return None
    normalized = str(name).strip().replace("-", "_").replace(" ", "_").lower()
    if normalized in BUTTON_NAME_ALIASES:
        return BUTTON_NAME_ALIASES[normalized]
    normalized = normalized.upper()
    return normalized if normalized in CANONICAL_BUTTONS else None


def normalize_game_id(game_id: Any) -> str | None:
    if game_id is None:
        return None
    value = str(game_id).strip().lower().replace("-", "_").replace(" ", "_")
    if not value:
        return None
    if value in DEFAULT_CONTROL_SCHEMAS:
        return value
    if value.startswith("resident_evil"):
        return "resident_evil"
    if value.startswith("lies_of_p"):
        return "lies_of_p"
    if value.startswith("sekiro"):
        return "sekiro"
    return value


def is_crossfps_fixed_button_mode(mode: Any) -> bool:
    return str(mode or "") in CROSS_FPS_FIXED_BUTTON_MODES


def crossfps_fixed_button_names(mode: Any) -> tuple[str, ...]:
    normalized_mode = str(mode or "")
    if normalized_mode == "crossfps_all_buttons":
        return CROSS_FPS_ALL_BUTTONS
    if normalized_mode == "crossfps_unified":
        return CROSS_FPS_UNIFIED_BUTTONS
    if normalized_mode == "crossfps_4action":
        return CROSS_FPS_FOUR_ACTION_BUTTONS
    if normalized_mode == "crossfps_legacy6":
        return LEGACY_CROSSFPS_BUTTONS
    supported = ", ".join(sorted(CROSS_FPS_FIXED_BUTTON_MODES))
    raise ValueError(f"Unsupported fixed CrossFPS button mode {mode!r}; supported: {supported}")


def crossfps_fixed_raw_button_columns(game_id: Any, mode: Any) -> tuple[str, ...]:
    ''

    normalized_game_id = normalize_game_id(game_id)
    normalized_mode = str(mode or "")
    if normalized_game_id == "xonotic":
        if normalized_mode == "crossfps_legacy6":
            return LEGACY_CROSSFPS_BUTTONS
        raise ValueError(
            f"{mode} has no audited fixed button mapping for game_id={game_id!r} "
            f"(normalized={normalized_game_id!r}); Xonotic supports: crossfps_legacy6"
        )
    if normalized_game_id not in CROSS_FPS_UNIFIED_GAME_IDS:
        supported = ", ".join(sorted(CROSS_FPS_UNIFIED_GAME_IDS))
        raise ValueError(
            f"{mode} has no audited fixed button mapping for "
            f"game_id={game_id!r} (normalized={normalized_game_id!r}); supported: {supported}"
        )
    if normalized_mode == "crossfps_legacy6":
        return LEGACY_CROSSFPS_BUTTONS
    mapping = (
        CROSS_FPS_ALL_BUTTON_RAW_COLUMNS
        if normalized_mode == "crossfps_all_buttons"
        else CROSS_FPS_UNIFIED_RAW_COLUMNS
    )
    return tuple(mapping[button] for button in crossfps_fixed_button_names(mode))


def crossfps_fixed_button_state_dim(mode: Any) -> int:
    ''

    button_dim = len(crossfps_fixed_button_names(mode))
    return button_dim + (2 if str(mode or "") == "crossfps_legacy6" else 1)


def crossfps_unified_raw_button_columns(game_id: Any) -> tuple[str, ...]:
    ''

    return crossfps_fixed_raw_button_columns(game_id, "crossfps_unified")


def load_control_schemas(schema_path: str | Path | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    if schema_path is None:
        return deepcopy(DEFAULT_CONTROL_SCHEMAS)

    path = Path(schema_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return deepcopy(DEFAULT_CONTROL_SCHEMAS)

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Control schema must be a mapping: {path}")

    schemas: dict[str, dict[str, dict[str, Any]]] = {}
    for game_id, buttons in payload.items():
        normalized_game = normalize_game_id(game_id)
        if not normalized_game or not isinstance(buttons, dict):
            continue
        normalized_buttons: dict[str, dict[str, Any]] = {}
        for button, entry in buttons.items():
            normalized_button = normalize_button_name(button)
            if normalized_button is None:
                continue
            normalized_buttons[normalized_button] = _normalize_meaning_entry(entry)
        schemas[normalized_game] = normalized_buttons
    return schemas or deepcopy(DEFAULT_CONTROL_SCHEMAS)


def _normalize_meaning_entry(entry: Any) -> dict[str, Any]:
    if isinstance(entry, str):
        return {"meaning": entry, "aliases": []}
    if isinstance(entry, Mapping):
        meaning = str(entry.get("meaning", "") or "")
        aliases = entry.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases = [str(alias) for alias in aliases if str(alias).strip()]
        timeline_meanings = entry.get("timeline_meanings") or []
        if not isinstance(timeline_meanings, list):
            timeline_meanings = []
        normalized_timeline_meanings: list[dict[str, Any]] = []
        for rule in timeline_meanings:
            if not isinstance(rule, Mapping):
                continue
            rule_meaning = str(rule.get("meaning", "") or "").strip()
            if not rule_meaning:
                continue
            normalized_rule: dict[str, Any] = {"meaning": rule_meaning}
            for key in ("min_duration_sec", "max_duration_sec", "min_frames", "max_frames"):
                if key in rule and rule[key] is not None:
                    normalized_rule[key] = rule[key]
            normalized_timeline_meanings.append(normalized_rule)
        return {"meaning": meaning, "aliases": aliases, "timeline_meanings": normalized_timeline_meanings}
    return {"meaning": "", "aliases": []}


def resolve_button_meanings(
    sample: Mapping[str, Any],
    schemas: Mapping[str, Mapping[str, Any]],
    *,
    null_meaning_text: str = "",
) -> dict[str, dict[str, Any]]:
    sample_meanings = sample.get("button_meanings")
    if isinstance(sample_meanings, Mapping) and sample_meanings:
        source = sample_meanings
    else:
        game_id = normalize_game_id(
            sample.get("game_id")
            or sample.get("game")
            or sample.get("source_game")
            or sample.get("selected_game")
            or sample.get("control_profile_id")
        )
        source = schemas.get(game_id or "", {})

    meanings: dict[str, dict[str, Any]] = {}
    for button in CANONICAL_BUTTONS:
        meanings[button] = {"meaning": null_meaning_text, "aliases": []}

    for button, entry in source.items():
        normalized_button = normalize_button_name(button)
        if normalized_button is None:
            continue
        normalized_entry = _normalize_meaning_entry(entry)
        if not normalized_entry["meaning"]:
            normalized_entry["meaning"] = null_meaning_text
        meanings[normalized_button] = normalized_entry
    return meanings


def resolved_button_meaning_texts(
    button_meanings: Mapping[str, Any],
    *,
    null_meaning_text: str = "",
) -> dict[str, str]:
    texts: dict[str, str] = {}
    for button in CANONICAL_BUTTONS:
        entry = button_meanings.get(button, {})
        normalized_entry = _normalize_meaning_entry(entry)
        meaning = normalized_entry["meaning"] or null_meaning_text
        texts[button] = str(meaning or "")
    return texts


def _safe_vocab_id_part(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = _VOCAB_ID_SAFE_PATTERN.sub("_", text).strip("_")
    return text or "unknown"


def make_semantic_action_vocab_id(game_id: Any, button: Any, meaning: Any) -> str:
    return "__".join(
        (
            _safe_vocab_id_part(normalize_game_id(game_id) or game_id),
            _safe_vocab_id_part(normalize_button_name(button) or button),
            _safe_vocab_id_part(meaning),
        )
    )


def load_semantic_action_vocab(vocab_path: str | Path) -> list[dict[str, str]]:
    path = Path(vocab_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"Semantic action vocab path does not exist: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    channels = payload.get("channels") if isinstance(payload, Mapping) else None
    if not isinstance(channels, list) or not channels:
        raise ValueError(f"Semantic action vocab must contain a non-empty `channels` list: {path}")

    normalized: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, str]] = set()
    for index, item in enumerate(channels):
        if not isinstance(item, Mapping):
            raise ValueError(f"Semantic action vocab channel #{index} is not a mapping: {path}")
        game_id = normalize_game_id(item.get("game_id") or item.get("game") or item.get("selected_game"))
        button = normalize_button_name(item.get("button"))
        meaning = str(item.get("meaning") or "").strip()
        if not game_id or not button or not meaning:
            raise ValueError(
                f"Semantic action vocab channel #{index} must define game_id, button, and meaning: {path}"
            )
        channel_id = str(item.get("id") or make_semantic_action_vocab_id(game_id, button, meaning))
        key = (game_id, button)
        if channel_id in seen_ids:
            raise ValueError(f"Duplicate semantic action vocab id `{channel_id}` in {path}")
        if key in seen_keys:
            raise ValueError(f"Duplicate semantic action vocab game/button key `{game_id}:{button}` in {path}")
        seen_ids.add(channel_id)
        seen_keys.add(key)
        normalized.append(
            {
                "id": channel_id,
                "game_id": game_id,
                "button": button,
                "meaning": meaning,
            }
        )
    return normalized


def format_timeline_seconds(seconds: float, quantize_sec: float) -> str:
    if quantize_sec > 0:
        seconds = round(seconds / quantize_sec) * quantize_sec
    seconds = max(float(seconds), 0.0)
    decimals = 1 if quantize_sec >= 0.1 else 2
    return f"{seconds:.{decimals}f}s"


def build_button_action_timeline_text(
    buttons: torch.Tensor,
    button_meaning_texts: Mapping[str, Any],
    *,
    fps: float,
    active_threshold: float = 0.1,
    quantize_sec: float = 0.1,
    max_events: int = 24,
    min_event_frames: int = 1,
    point_event_max_frames: int = 2,
    prefix: str = "Actions:",
) -> str:
    if buttons.ndim != 2 or buttons.shape[-1] != len(CANONICAL_BUTTONS):
        raise ValueError(
            f"buttons must have shape [frames, {len(CANONICAL_BUTTONS)}] for timeline text, got {tuple(buttons.shape)}"
        )
    if fps <= 0:
        raise ValueError(f"fps must be positive for action timeline text, got {fps}")

    active = buttons.detach().abs().cpu() > float(active_threshold)
    events: list[tuple[int, int, Any]] = []
    num_frames = int(active.shape[0])
    for button_index, button in enumerate(CANONICAL_BUTTONS):
        meaning_entry = button_meaning_texts.get(button, "")
        base_meaning = _timeline_base_meaning(meaning_entry)
        if not base_meaning:
            continue
        start: int | None = None
        for frame_index in range(num_frames + 1):
            is_active = frame_index < num_frames and bool(active[frame_index, button_index].item())
            if is_active and start is None:
                start = frame_index
            elif (not is_active) and start is not None:
                end = frame_index
                if end - start >= int(min_event_frames):
                    events.append((start, end, meaning_entry))
                start = None

    if not events:
        return ""

    events.sort(key=lambda item: (item[0], item[1], _timeline_base_meaning(item[2])))
    if max_events > 0:
        events = events[: int(max_events)]

    phrases: list[str] = []
    for start, end, meaning_entry in events:
        duration_frames = end - start
        meaning = _timeline_meaning_for_event(meaning_entry, duration_frames=duration_frames, fps=fps)
        if not meaning:
            continue
        if duration_frames <= int(point_event_max_frames):
            timestamp = format_timeline_seconds(start / fps, quantize_sec)
            phrases.append(f"[{timestamp}] {meaning}.")
        else:
            start_text = format_timeline_seconds(start / fps, quantize_sec)
            end_text = format_timeline_seconds(end / fps, quantize_sec)
            phrases.append(f"[{start_text}-{end_text}] {meaning}.")

    prefix = prefix.strip()
    return f"{prefix} {' '.join(phrases)}" if prefix else " ".join(phrases)


def _timeline_base_meaning(entry: Any) -> str:
    if isinstance(entry, Mapping):
        return str(entry.get("meaning", "") or "").strip()
    return str(entry or "").strip()


def _timeline_meaning_for_event(entry: Any, *, duration_frames: int, fps: float) -> str:
    normalized = _normalize_meaning_entry(entry)
    duration_sec = float(duration_frames) / max(float(fps), 1e-6)
    for rule in normalized.get("timeline_meanings", []):
        try:
            min_duration_sec = rule.get("min_duration_sec")
            max_duration_sec = rule.get("max_duration_sec")
            min_frames = rule.get("min_frames")
            max_frames = rule.get("max_frames")
            if min_duration_sec is not None and duration_sec < float(min_duration_sec):
                continue
            if max_duration_sec is not None and duration_sec > float(max_duration_sec):
                continue
            if min_frames is not None and duration_frames < int(min_frames):
                continue
            if max_frames is not None and duration_frames > int(max_frames):
                continue
        except (TypeError, ValueError):
            continue
        return str(rule.get("meaning", "") or "").strip()
    return str(normalized.get("meaning", "") or "").strip()


def build_control_binding_text(
    button_meanings: Mapping[str, Any],
    rng: random.Random | Any,
    *,
    use_alias: bool = True,
    alias_sampling_prob: float = 0.3,
    deterministic: bool = False,
) -> tuple[str, dict[str, str]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for button, entry in button_meanings.items():
        normalized_button = normalize_button_name(button)
        if normalized_button is None:
            continue
        normalized_entry = _normalize_meaning_entry(entry)
        if normalized_entry["meaning"]:
            items.append((normalized_button, normalized_entry))

    if deterministic:
        order = {button: idx for idx, button in enumerate(CANONICAL_BUTTONS)}
        items.sort(key=lambda item: order.get(item[0], 999))
        wrapper, pair_template = CONTROL_TEXT_TEMPLATES[0]
        separator = ". "
    else:
        rng.shuffle(items)
        wrapper, pair_template = rng.choice(CONTROL_TEXT_TEMPLATES)
        separator = rng.choice(PAIR_SEPARATORS)

    phrase_by_button: dict[str, str] = {}
    pairs: list[str] = []
    for button, entry in items:
        meaning = str(entry.get("meaning", "") or "")
        aliases = [str(alias) for alias in entry.get("aliases", []) if str(alias).strip()]
        if use_alias and aliases and not deterministic and rng.random() < alias_sampling_prob:
            meaning = rng.choice(aliases)
        phrase_by_button[button] = meaning
        pairs.append(pair_template.format(button=button, meaning=meaning))

    if not pairs:
        return "", phrase_by_button
    return wrapper.format(pairs=separator.join(pairs)), phrase_by_button


def apply_button_remap(
    buttons: torch.Tensor,
    button_meanings: Mapping[str, Any],
    *,
    groups: list[list[str]],
    rng: random.Random | Any,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, str]]:
    if buttons.ndim != 2 or buttons.shape[-1] != len(CANONICAL_BUTTONS):
        raise ValueError(
            f"buttons must have shape [frames, {len(CANONICAL_BUTTONS)}], got {tuple(buttons.shape)}"
        )

    normalized_meanings = {
        normalized: deepcopy(value)
        for button, value in button_meanings.items()
        if (normalized := normalize_button_name(button)) is not None
    }
    remapped_buttons = buttons.clone()
    full_mapping = {button: button for button in CANONICAL_BUTTONS}

    for group in groups:
        normalized_group = [button for item in group if (button := normalize_button_name(item)) in CANONICAL_BUTTONS]
        if len(normalized_group) < 2:
            continue
        permuted = list(normalized_group)
        rng.shuffle(permuted)
        for old_button, new_button in zip(normalized_group, permuted, strict=True):
            old_idx = CANONICAL_BUTTONS.index(old_button)
            new_idx = CANONICAL_BUTTONS.index(new_button)
            remapped_buttons[:, new_idx] = buttons[:, old_idx]
            full_mapping[old_button] = new_button

    remapped_meanings = {}
    for old_button, value in normalized_meanings.items():
        remapped_meanings[full_mapping.get(old_button, old_button)] = deepcopy(value)
    return remapped_buttons, remapped_meanings, full_mapping


def canonicalize_button_tensor(buttons: torch.Tensor) -> torch.Tensor:
    if buttons.shape[-1] == len(CANONICAL_BUTTONS):
        return buttons
    if buttons.shape[-1] != len(LEGACY_CROSSFPS_BUTTONS):
        raise ValueError(
            f"Unsupported button tensor width {buttons.shape[-1]}; expected {len(CANONICAL_BUTTONS)} "
            f"or legacy {len(LEGACY_CROSSFPS_BUTTONS)}"
        )

    canonical = torch.zeros(*buttons.shape[:-1], len(CANONICAL_BUTTONS), dtype=buttons.dtype, device=buttons.device)
    legacy_map = {
        "RT": 0,
        "LT": 1,
        "A": 3,
        "X": 4,
        "Y": 5,
    }
    for button, legacy_idx in legacy_map.items():
        canonical[..., CANONICAL_BUTTONS.index(button)] = buttons[..., legacy_idx]
    return canonical
