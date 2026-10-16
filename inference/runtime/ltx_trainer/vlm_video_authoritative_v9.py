                                                                              

                                                                          
                                                                     
   
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

from ltx_trainer.vlm_fire_protocol import (
    AIM_OVERLAP_PROMPT,
    TARGETS,
)

POLICY = "projected_fire_reference_crop_video_authoritative_v9"
FORECAST_PROMPT = (
    "Image 1 is an appearance reference of the one enemy, not the current scene. "
    "Image 2 is the aiming image. Identify that enemy in image 2, allowing different "
    "poses and lighting. Answer ONLY about the circle in image 2. "
    + AIM_OVERLAP_PROMPT
)
AUDIT_PROMPT = FORECAST_PROMPT
STATE_PROMPT = (
    "Audit the enemy's physical state in the just-generated block, not the intended story. "
    "The REFERENCE shows identity only; it is not a current observation. "
    "The remaining images are chronological: BEFORE, LATE, END. "
    "Report the same enemy's state at END using the sequence for context: "
    "alive = visibly living, upright or actively moving; "
    "dead = a visibly defeated body/corpse, supported by its posture and the sequence; "
    "not_visible = the enemy cannot be visually identified at END, including occlusion "
    "or an unrecognizable remnant. Disappearance, shadows, blur, or falling alone do not prove death. "
    "A clearly living enemy at END is alive even if an earlier image looks dead. "
    "The supplied requested outcome describes THIS block's instruction, not evidence. "
    "Let the images override that instruction. Do not infer HP or infer death from gunfire. "
    'Return only JSON: {"status":"alive"}, {"status":"dead"}, or {"status":"not_visible"}.'
)
STATUSES = ["alive", "dead", "not_visible"]


def image_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=3)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def request_images(
    observer: Any,
    prompt: str,
    images: list[tuple[str, str]],
    fields: dict[str, list[str]],
) -> dict[str, str]:
                                                                                    
    import requests

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for label, url in images:
        content.extend([
            {"type": "text", "text": label},
            {"type": "image_url", "image_url": {"url": url}},
        ])
    schema = {
        "type": "object",
        "properties": {key: {"type": "string", "enum": values} for key, values in fields.items()},
        "required": list(fields),
        "additionalProperties": False,
    }
    response = requests.post(
        f"{observer.base_url}/chat/completions",
        json={
            "model": observer.model, "temperature": 0, "max_tokens": 256,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "video_authoritative_v9", "strict": True, "schema": schema,
            }},
        },
        timeout=observer.timeout_seconds,
    )
    if not response.ok:
        raise RuntimeError(f"VLM HTTP {response.status_code}: {response.text[:1200]}")
    choice = response.json()["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Incomplete VLM response; do not commit state")
    result = json.loads(choice["message"]["content"])
    if not isinstance(result, dict) or set(result) != set(fields):
        raise ValueError(f"Unexpected VLM response fields: {result}")
    if any(result[key] not in values for key, values in fields.items()):
        raise ValueError(f"Invalid VLM response: {result}")
    return result


def load_enemy_reference(
    source: Path, appearance_id: str, crop_xyxy: list[int] | tuple[int, int, int, int] | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Load the reviewed appearance crop from the case image.

    Crop geometry is part of the case manifest; no external registry or source
    hash is required for inference.
    """
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with Image.open(source) as handle:
        image = handle.convert("RGB")
    if crop_xyxy is None:
        box = [0, 0, image.width, image.height]
    else:
        box = [int(value) for value in crop_xyxy]
    if len(box) != 4:
        raise ValueError("Reference crop must contain four coordinates")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError(f"Out-of-bounds enemy crop: {box}")
    crop = image.crop((x0, y0, x1, y1))
    spec = {
        "source_path": str(source),
        "appearance_id": appearance_id,
        "source_size": list(image.size),
        "crop_xyxy": box,
    }
    return crop, spec


def forecast_input(observer: Any, projected: Image.Image, crop_size: int) -> str:
                                                                             
    return observer._aim_target_image_data_url(projected, crop_size=crop_size)


def forecast(
    observer: Any, projected: Image.Image, crop_size: int, reference: Image.Image,
) -> dict[str, str]:
    return request_images(observer, FORECAST_PROMPT, [
        ("Image 1: REFERENCE ONLY", image_url(reference)),
        ("Image 2: PROJECTED AIM", forecast_input(observer, projected, crop_size)),
    ], {"aim_target": TARGETS})


def audit_fire(observer: Any, actual: Image.Image, reference: Image.Image) -> dict[str, str]:
    from ltx_trainer.vlm_agent import draw_aim_center_ring

    return request_images(observer, AUDIT_PROMPT, [
        ("Image 1: REFERENCE ONLY", image_url(reference)),
        ("Image 2: ACTUAL FIRE", image_url(draw_aim_center_ring(actual))),
    ], {"aim_target": TARGETS})


def requested_status(prompt_mode: str) -> str:
    if prompt_mode == "corpse" or prompt_mode.startswith("killed_"):
        return "dead"
    if prompt_mode.startswith(("active_", "hit_alive_")):
        return "alive"
    raise ValueError(f"Unsupported single-enemy guidance: {prompt_mode}")


def observe_state(
    observer: Any, *, reference: Image.Image, before: Image.Image,
    late: Image.Image, end: Image.Image, prompt_mode: str,
) -> dict[str, str]:
                                                                        
    expected = requested_status(prompt_mode)

    def padded(image: Image.Image, ratio: int) -> Image.Image:
                                                                                 
                                                                            
        height = max(image.height, (image.width + ratio - 1) // ratio)
        canvas = Image.new("RGB", (height * ratio, height), (24, 24, 24))
        canvas.paste(image, ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2))
        return canvas

    return request_images(
        observer, STATE_PROMPT + " Requested outcome: " + json.dumps({"status": expected}),
        [(label, image_url(padded(image, ratio))) for label, image, ratio in [
            ("REFERENCE ONLY", reference, 1), ("BEFORE", before, 2),
            ("LATE", late, 2), ("END", end, 2),
        ]], {"status": STATUSES},
    )


def audit_block(
    observer: Any, *, reference: Image.Image, before: Image.Image,
    fire_frames: list[tuple[int, Image.Image]], late: Image.Image,
    end: Image.Image, prompt_mode: str,
) -> dict[str, str]:
                                                                                
    from ltx_trainer.vlm_agent import draw_aim_center_ring

    if len({index for index, _ in fire_frames}) != len(fire_frames):
        raise ValueError("Duplicate fire frame")
    if fire_frames != sorted(fire_frames, key=lambda entry: entry[0]):
        raise ValueError("Fire frames must be chronological")
    fields = {f"frame_{index}": TARGETS for index, _ in fire_frames}
    fields["status"] = STATUSES
    prompt = (
        "Audit ONE generated block. REFERENCE is identity only, never current evidence. "
        "Images follow chronological order: BEFORE, selected ACTUAL FIRE frames, LATE, END. "
        "For each frame_N field, classify ONLY the body overlap with the marked circle in "
        "that corresponding ACTUAL FIRE image. Enemy means the reference enemy's visible "
        "body overlaps the circle interior; background means no identifiable enemy body "
        "overlaps it. A nearby body outside the circle, weapon held by the player, muzzle "
        "flash, scenery or shadow is background. No damage animation is required. "
        "Never use another image to fill a missing body or count a hit. "
        "Separately report status at END: alive = visibly living; dead = visibly defeated "
        "body/corpse supported by posture and the sequence; not_visible = cannot identify "
        "the enemy at END. Disappearance or an unrecognizable shadow is not death. "
        "Use BEFORE/LATE/END only for physical state, not additional hit votes. "
        "A clearly living enemy at END is alive even after an earlier collapse. "
        "This block requested " + json.dumps({"status": requested_status(prompt_mode)}) +
        "; that is an instruction being checked, not evidence. Images override it. "
        "Do not infer numeric HP. Return only the requested structured JSON fields."
    )

    def padded(image: Image.Image, ratio: int) -> Image.Image:
        height = max(image.height, (image.width + ratio - 1) // ratio)
        canvas = Image.new("RGB", (height * ratio, height), (24, 24, 24))
        canvas.paste(image, ((canvas.width-image.width)//2, (canvas.height-image.height)//2))
        return canvas

    images = [("REFERENCE ONLY", image_url(padded(reference, 1))),
              ("BEFORE", image_url(padded(before, 2)))]
    images.extend((f"ACTUAL FIRE frame_{index}", image_url(padded(draw_aim_center_ring(frame), 2)))
                  for index, frame in fire_frames)
    images.extend([(label, image_url(padded(frame, 2))) for label, frame in [("LATE", late), ("END", end)]])
    return request_images(observer, prompt, images, fields)


def guidance_mode(state: dict[str, Any], checks: list[dict[str, Any]], direction: str) -> str:
                                                                                      
    if direction not in {"left", "right"}:
        raise ValueError(direction)
    if state["phase"] == "corpse" and state["hp"] == 0:
        return "corpse"
    if state["phase"] != "active" or state["hp"] <= 0:
        raise ValueError(f"Invalid single-enemy state: {state}")
    if any(event["predicted_aim_target"] == "enemy" for event in checks):
        return f'{"killed" if state["hp"] <= 50 else "hit_alive"}_{direction}'
    return f"active_{direction}"


def reconcile(
    state: dict[str, Any], status: str, targets: list[str], prompt_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
                                                                              

                                                                                
                                                                             
                                                                              
       
    if status not in STATUSES or any(target not in TARGETS for target in targets):
        raise ValueError("Invalid structured state/aim observation")
    prior_hp = state["hp"]
    if type(prior_hp) is not int or prior_hp < 0 or state["phase"] not in {"active", "corpse"}:
        raise ValueError(f"Invalid rule state: {state}")
    if (state["phase"] == "corpse") != (prior_hp == 0) or state["last_alive_hp"] <= 0:
        raise ValueError(f"Inconsistent HP/lifecycle: {state}")
    result = dict(state)
    hit = "enemy" in targets
    corrections: list[str] = []
    if status == "dead":
        result.update(hp=0, phase="corpse", corpse_blocks=(
            state["corpse_blocks"] + 1 if state["phase"] == "corpse" else 0))
        if prior_hp > 0:
            result["last_alive_hp"] = prior_hp
            corrections.append("visual_death_overrides_aim_ledger")
    elif status == "alive" and prior_hp == 0:
        result.update(hp=state["last_alive_hp"], phase="active", corpse_blocks=0)
        corrections.append("visible_alive_overrides_recorded_death_not_respawn")
    elif prior_hp > 0 and hit:
        if prior_hp > 50:
            result.update(hp=prior_hp - 50, last_alive_hp=prior_hp - 50)
        else:
            corrections.append("lethal_aim_vote_not_committed_without_visual_death")
    result["visual_status"] = status
    expected = requested_status(prompt_mode)
    report = {
        "observed_status": status,
        "requested_status": expected,
        "lifecycle_guidance_match": ("not_observed" if status == "not_visible" else
                                     "matches" if status == expected else "contradicts"),
        "aim_hit": hit,
        "corrections": corrections,
        "hp_source": "visual_zero_or_rule_estimated_positive_not_vlm_numeric_hp",
    }
    return result, report
