                                                                                 

                                                                            
                                                                             
   
from __future__ import annotations

import json
from typing import Any

from PIL import Image

from ltx_trainer.vlm_video_authoritative_v9 import (
    STATUSES, request_images, image_url, requested_status,
    load_enemy_reference, guidance_mode, reconcile, forecast,
    audit_block as prior_audit_block,
)

POLICY = "end_first_state_pass_preserved_projection_and_fire_audit_v11"
AIM_PROMPT = (
    "Image 1 CURRENT AIM is the ONLY image to classify. Image 2 REFERENCE ONLY shows "
    "what the enemy looks like, NOT where it is now or whether it is present. "
    "Answer enemy only when that enemy's identifiable body is inside or touching the lime-green circle "
    "in Image 1. Answer background when no identifiable enemy body overlaps that lime-green circle. "
    "A nearby body entirely outside the lime-green circle, the player's weapon, muzzle flash, "
    "scenery and a shadow do not count. Use Image 2 only to recognize identity in Image 1; "
    "never answer about a body or its position in Image 2. No damage animation is required. "
    'Return only JSON {"aim_target":"enemy"} or {"aim_target":"background"}.'
)
STATE_PROMPT = (
    "Image 1 END is the current observation you must classify. Other images provide "
    "context only: BEFORE precedes this block, LATE precedes END in this block, and "
    "REFERENCE ONLY is an older appearance reference, NOT current presence evidence. "
    "Report physical status in END ONLY. alive means an identifiable living enemy is "
    "visibly present in END. dead means an identifiable defeated body/corpse is visibly "
    "present in END, with posture and temporal context supporting that interpretation. "
    "A stagger, lean, flinch or a fall still in progress while partly upright is not by "
    "itself proof of death; do not label that a settled corpse. "
    "not_visible means no identifiable enemy is visible in END, including full occlusion "
    "behind the player's gun, leaving the view or unrecognizable remnants. A shadow, "
    "gun, scope, reticle or scenery is not an enemy. Do not carry alive/dead from BEFORE "
    "or REFERENCE when the enemy is not identifiable in END. If END clearly shows a "
    "living enemy, answer alive even after an earlier collapse. Requested outcome is "
    "the instruction being checked, never evidence. Do not infer numeric HP. "
    'Return only JSON with status: alive, dead, or not_visible.'
)


def padded(image: Image.Image, ratio: int) -> Image.Image:
    height = max(image.height, (image.width+ratio-1)//ratio)
    canvas = Image.new("RGB", (height*ratio, height), (24,24,24))
    canvas.paste(image, ((canvas.width-image.width)//2, (canvas.height-image.height)//2))
    return canvas


def audit_block(observer: Any, *, reference: Image.Image, before: Image.Image,
                fire_frames: list[tuple[int,Image.Image]], late: Image.Image,
                end: Image.Image, prompt_mode: str) -> dict[str,str]:
    if len({frame for frame, _ in fire_frames}) != len(fire_frames):
        raise ValueError("Duplicate fire frame")
    result = (prior_audit_block(observer, reference=reference, before=before,
        fire_frames=fire_frames, late=late, end=end, prompt_mode=prompt_mode)
        if fire_frames else {})
    state = request_images(observer, STATE_PROMPT+" Requested outcome: "+json.dumps({"status":requested_status(prompt_mode)}),
        [(label,image_url(padded(image,ratio))) for label,image,ratio in [
            ("Image 1 END - classify this image only",end,2),
            ("Image 2 BEFORE - earlier context only",before,2),
            ("Image 3 LATE - just before END",late,2),
            ("Image 4 REFERENCE ONLY - identity, not current presence",reference,1),
        ]], {"status":STATUSES})
    result["status"] = state["status"]
    return result
