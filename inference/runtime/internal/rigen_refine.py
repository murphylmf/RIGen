                      
''

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for relative in (
    "packages/ltx-trainer/src",
    "packages/ltx-core/src",
    "packages/ltx-pipelines/src",
    "packages/ltx-trainer/scripts",
):
    source = REPO_ROOT / relative
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from ltx_core.components.noisers import GaussianNoiser              
from ltx_core.conditioning import VideoConditionByLatentIndex              
from ltx_core.loader import (              
    LTXV_LORA_COMFY_RENAMING_MAP,
    LoraPathStrengthAndSDOps,
)
from ltx_core.model.video_vae import (              
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
)
from ltx_pipelines.utils.args import ImageConditioningInput              
from ltx_pipelines.utils.blocks import (              
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.helpers import combined_image_conditionings              
from ltx_pipelines.utils.types import ModalitySpec              
from ltx_trainer.video_utils import save_video              
from rigen_components import CachedOriginalDiffusionStage, FixedVideoNoiser, SimpleDenoiserNoCamTokens, cleanup_cuda


BASE_MODEL = Path("weights/ltx-2.3-22b-dev.safetensors")
GEMMA_ROOT = Path(
    "weights/gemma-3-12b-it-qat-q4_0-unquantized"
)
DISTILLED_LORA = Path(
    "weights/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
)
SPATIAL_UPSAMPLER = Path(
    "weights/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
)
SIGMAS = (0.725, 0.421875, 0.0)
HISTORY_BLOCKS = 4
BLOCK_FRAMES = 16
LATENTS_PER_BLOCK = 2
WINDOW_FRAMES = 1 + (HISTORY_BLOCKS + 1) * BLOCK_FRAMES
WINDOW_LATENTS = 1 + (HISTORY_BLOCKS + 1) * LATENTS_PER_BLOCK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--gemma-root", type=Path, default=GEMMA_ROOT)
    parser.add_argument("--distilled-lora", type=Path, default=DISTILLED_LORA)
    parser.add_argument("--spatial-upsampler", type=Path, default=SPATIAL_UPSAMPLER)
    parser.add_argument(
        "--pre-upscaled-latent",
        type=Path,
        default=None,
        help=(
            "Diagnostic-only override: consume an already x2-upscaled Stage-1 "
            "latent. The default full-clip upsampler path is unchanged."
        ),
    )
    parser.add_argument("--first-target-block", type=int, default=4)
    parser.add_argument(
        "--last-target-block",
        type=int,
        help="Inclusive final target block. Defaults to the final cache block.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--decoder-mode",
        choices=("untiled", "tiled"),
        default="untiled",
        help="No automatic fallback: an untiled OOM is raised to the caller.",
    )
    parser.add_argument(
        "--handoff-latents",
        type=int,
        default=0,
        help=(
            "Enable refined-latent handoff with this many clean bridge latents. "
            "Zero preserves the original independent-window RGB append path."
        ),
    )
    parser.add_argument(
        "--save-refined-latent",
        action="store_true",
        help="Persist the assembled full refined latent in handoff mode.",
    )
    parser.add_argument(
        "--pairwise-target-only",
        action="store_true",
        help=(
            "After the bootstrap window, give the refiner only the clean "
            "handoff prefix plus the two new AR latents. Requires "
            "--handoff-latents > 0."
        ),
    )
    parser.add_argument(
        "--precomputed-noise",
        type=Path,
        default=None,
        help=(
            "Optional full-clip [B,C,T,H,W] noise. Each local refiner call "
            "uses the exact matching global temporal slice."
        ),
    )
    parser.add_argument(
        "--defer-decode",
        action="store_true",
        help=(
            "Save the assembled refined latent and exit before VAE decode. "
            "This keeps the 22B refiner and long-video decoder in separate processes."
        ),
    )
    return parser.parse_args()


def run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def extract_anchor_frames(
    source_video: Path,
    destination: Path,
    *,
    count: int,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    pattern = destination / "anchor_%03d.png"
    expected = [destination / f"anchor_{index:03d}.png" for index in range(count)]
    if not all(path.is_file() for path in expected):
        for path in destination.glob("anchor_*.png"):
            path.unlink()
        run_checked(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source_video),
                "-vf",
                r"select=not(mod(n\,16))",
                "-fps_mode",
                "vfr",
                "-frames:v",
                str(count),
                "-start_number",
                "0",
                str(pattern),
            ]
        )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError("Failed to extract sliding-window anchors: " + "; ".join(missing))
    return expected


def encode_prompt_contexts(
    *,
    prompts: list[str],
    base_model: Path,
    gemma_root: Path,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    unique = list(dict.fromkeys(prompts))
    encoder = PromptEncoder(
        str(base_model.resolve()),
        str(gemma_root.resolve()),
        dtype,
        device,
    )
    with torch.inference_mode():
        outputs = encoder(unique)
    contexts = {
        prompt: output.video_encoding.detach().to(device="cpu", dtype=dtype)
        for prompt, output in zip(unique, outputs, strict=True)
    }
    del encoder, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return contexts


def decode_video_latent(
    decoder: torch.nn.Module,
    latent: torch.Tensor,
    tiling_config: TilingConfig | None,
) -> torch.Tensor:
    chunks = list(decoder.decode_video(latent, tiling_config=tiling_config))
    if not chunks:
        raise RuntimeError("Video decoder returned no chunks")
    decoded = torch.cat(chunks, dim=0)
    return decoded.permute(3, 0, 1, 2).contiguous().cpu()


def tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().contiguous().cpu()
    return hashlib.sha256(contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()


def mux_source_audio(silent: Path, source: Path, output: Path) -> None:
    run_checked(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(silent),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_source_cache(
    payload: dict[str, Any],
    cache_path: Path,
) -> tuple[dict[str, Any], str]:
    ''
    schema = str(payload.get("schema_version"))
    if schema == "ar_sliding_ref_refiner_cache_v1":
        return payload, schema
    if schema != "ar_teacher_stage1_refiner_cache_v2":
        raise ValueError(f"Unsupported cache schema: {schema!r}")

    prompt_schedule = [str(value) for value in payload["prompt_schedule"]]
    if not prompt_schedule:
        raise ValueError("Stage-2 AR cache has an empty prompt schedule")
    stage1_geometry = payload["stage1_geometry"]
    source_raw_video = cache_path.resolve().parent / "pred.mp4"
    normalized = {
        "schema_version": "ar_sliding_ref_refiner_cache_v1",
        "sample_id": str(payload.get("case_id", cache_path.stem)),
        "seed": int(payload["seed"]),
        "generator_state": payload["generator_state"],
        "video_latent": payload["video_latent"],
        "audio_latent": payload["audio_latent"],
        "video_context": payload["video_prompt_schedule_embeds"][:, 0].detach().clone(),
        "audio_context": payload["audio_prompt_schedule_embeds"][:, 0].detach().clone(),
        "context_mask": payload["prompt_schedule_attention_mask"][:, 0].detach().clone(),
        "refiner_prompt": prompt_schedule[0],
        "prompt_schedule": prompt_schedule,
        "condition_image_path": str(payload["refiner_condition_image_path"]),
        "action_joystick": payload["action_joystick"],
        "action_buttons": payload["action_buttons"],
        "geometry": {
            "stage1_width": int(stage1_geometry["width"]),
            "stage1_height": int(stage1_geometry["height"]),
            "final_width": 2 * int(stage1_geometry["width"]),
            "final_height": 2 * int(stage1_geometry["height"]),
            "num_frames": int(stage1_geometry["num_frames"]),
            "fps": int(stage1_geometry["fps"]),
        },
        "source_raw_video": str(source_raw_video),
    }
    return normalized, schema


def main() -> None:
    args = parse_args()
    required = [args.cache, args.base_model, args.gemma_root, args.distilled_lora]
    if args.pre_upscaled_latent is None:
        required.append(args.spatial_upsampler)
    if args.precomputed_noise is not None:
        required.append(args.precomputed_noise)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing append-only refiner inputs: " + "; ".join(missing))

    loaded_payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    payload, source_cache_schema = normalize_source_cache(loaded_payload, args.cache)
    del loaded_payload
    geometry = payload["geometry"]
    if int(geometry["fps"]) != 20:
        raise ValueError(f"This contract expects 20 fps, got {geometry['fps']}")
    if not (0 <= int(args.handoff_latents) <= WINDOW_LATENTS - LATENTS_PER_BLOCK):
        raise ValueError(
            "--handoff-latents must be between 0 and the nine-latent window overlap; "
            f"got {args.handoff_latents}"
        )
    if args.pairwise_target_only and int(args.handoff_latents) <= 0:
        raise ValueError("--pairwise-target-only requires --handoff-latents > 0")
    if args.defer_decode and not args.save_refined_latent:
        raise ValueError("--defer-decode requires --save-refined-latent")
    if int(args.first_target_block) != HISTORY_BLOCKS:
        raise ValueError(
            "The first append-only window must contain exactly H4 + target1; "
            f"got first target {args.first_target_block}"
        )
    prompt_schedule = list(payload["prompt_schedule"])
    total_blocks = (int(geometry["num_frames"]) - 1) // BLOCK_FRAMES
    if len(prompt_schedule) != total_blocks:
        raise ValueError(
            f"Prompt schedule/block mismatch: {len(prompt_schedule)} vs {total_blocks}"
        )
    last_target = (
        total_blocks - 1
        if args.last_target_block is None
        else int(args.last_target_block)
    )
    if not (args.first_target_block <= last_target < total_blocks):
        raise ValueError(
            f"Invalid target range {args.first_target_block}:{last_target}; total={total_blocks}"
        )

    video_latent = payload["video_latent"]
    expected_stage1_shape = (
        1,
        128,
        1 + total_blocks * LATENTS_PER_BLOCK,
        int(geometry["stage1_height"]) // 32,
        int(geometry["stage1_width"]) // 32,
    )
    if tuple(video_latent.shape) != expected_stage1_shape:
        raise ValueError(
            f"Unexpected Stage-1 latent shape {tuple(video_latent.shape)}; "
            f"expected {expected_stage1_shape}"
        )
    source_raw_video = Path(payload["source_raw_video"]).resolve()
    if not source_raw_video.is_file():
        raise FileNotFoundError(source_raw_video)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "commit_log.jsonl").unlink(missing_ok=True)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    target_blocks = list(range(args.first_target_block, last_target + 1))
    window_starts = [target - HISTORY_BLOCKS for target in target_blocks]
    handoff_mode = int(args.handoff_latents) > 0
    conditioning_starts = [window_starts[0]] if handoff_mode else window_starts
    anchor_paths = extract_anchor_frames(
        source_raw_video,
        output_dir / "window_anchors",
        count=max(conditioning_starts) + 1,
    )

    start_time = time.perf_counter()
    target_prompts = [prompt_schedule[target] for target in target_blocks]
    if set(target_prompts) == {str(payload["refiner_prompt"])}:
        prompt_contexts = {
            str(payload["refiner_prompt"]): payload["video_context"]
            .detach()
            .to(device="cpu", dtype=dtype)
        }
        prompt_context_source = "exact context cached by Stage-1 evaluator"
    else:
        prompt_contexts = encode_prompt_contexts(
            prompts=target_prompts,
            base_model=args.base_model,
            gemma_root=args.gemma_root,
            dtype=dtype,
            device=device,
        )
        prompt_context_source = "Gemma plus original LTX embeddings processor"
    print(
        f"[append-only-refiner] prepared {len(prompt_contexts)} unique target prompts "
        f"from {prompt_context_source}",
        flush=True,
    )

    if args.pre_upscaled_latent is None:
        upsampler = VideoUpsampler(
            str(args.base_model.resolve()),
            str(args.spatial_upsampler.resolve()),
            dtype,
            device,
        )
        with torch.inference_mode():
            upscaled_full = upsampler(video_latent.to(device=device, dtype=dtype))
        del upsampler
    else:
        if not args.pre_upscaled_latent.is_file():
            raise FileNotFoundError(args.pre_upscaled_latent)
        upscaled_full = torch.load(
            args.pre_upscaled_latent,
            map_location=device,
            weights_only=True,
        ).to(dtype=dtype)
    expected_upscaled_shape = (
        1,
        128,
        expected_stage1_shape[2],
        int(geometry["final_height"]) // 32,
        int(geometry["final_width"]) // 32,
    )
    if tuple(upscaled_full.shape) != expected_upscaled_shape:
        raise ValueError(
            f"Unexpected upscaled shape {tuple(upscaled_full.shape)}; "
            f"expected {expected_upscaled_shape}"
        )
    precomputed_noise = None
    if args.precomputed_noise is not None:
        precomputed_noise = torch.load(
            args.precomputed_noise,
            map_location="cpu",
            weights_only=True,
        )
        if tuple(precomputed_noise.shape) != expected_upscaled_shape:
            raise ValueError(
                f"Unexpected precomputed noise shape {tuple(precomputed_noise.shape)}; "
                f"expected {expected_upscaled_shape}"
            )

    conditioner = ImageConditioner(str(args.base_model.resolve()), dtype, device)
    with torch.inference_mode():
        conditioning_by_start = conditioner(
            lambda image_encoder: {
                start: combined_image_conditionings(
                    images=[
                        ImageConditioningInput(
                            path=str(anchor_paths[start]),
                            frame_idx=0,
                            strength=1.0,
                            crf=33,
                        )
                    ],
                    height=int(geometry["final_height"]),
                    width=int(geometry["final_width"]),
                    video_encoder=image_encoder,
                    dtype=dtype,
                    device=device,
                )
                for start in conditioning_starts
            }
        )
    del conditioner
    gc.collect()
    torch.cuda.empty_cache()

    loras = (
        LoraPathStrengthAndSDOps(
            str(args.distilled_lora.resolve()),
            1.0,
            LTXV_LORA_COMFY_RENAMING_MAP,
        ),
    )
    stage2 = CachedOriginalDiffusionStage(
        str(args.base_model.resolve()), dtype, device, loras=loras
    )
    decoder_owner = VideoDecoder(str(args.base_model.resolve()), dtype, device)
    decoder: torch.nn.Module | None = None
    stage2_discarded = False
    sigmas = torch.tensor(SIGMAS, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.set_state(payload["generator_state"].cpu())
    noiser = GaussianNoiser(generator=generator)
    committed_chunks: list[torch.Tensor] = []
    committed_latent: torch.Tensor | None = None
    committed_chunk_sha256: list[str] = []
    records: list[dict[str, Any]] = []
    committed_frames = 0
    prior_prefix_sha256: str | None = None

    try:
        if not handoff_mode:
            decoder = decoder_owner._decoder_builder.build(device=device, dtype=dtype).to(device).eval()
        window_tiling = TilingConfig.default() if args.decoder_mode == "tiled" else None
        for sequence_index, target_block in enumerate(target_blocks):
            window_start = target_block - HISTORY_BLOCKS
            latent_end = 1 + (target_block + 1) * LATENTS_PER_BLOCK
            history_end = latent_end - LATENTS_PER_BLOCK
            pairwise_increment = bool(
                args.pairwise_target_only and handoff_mode and sequence_index > 0
            )
                                                                       
                                                                              
                                                                           
                                                                           
            if pairwise_increment:
                latent_start = history_end - int(args.handoff_latents)
            else:
                latent_start = window_start * LATENTS_PER_BLOCK
            local_initial = upscaled_full[:, :, latent_start:latent_end]
            expected_local_latents = (
                int(args.handoff_latents) + LATENTS_PER_BLOCK
                if pairwise_increment
                else WINDOW_LATENTS
            )
            if int(local_initial.shape[2]) != expected_local_latents:
                raise RuntimeError(
                    f"Window {target_block} has {local_initial.shape[2]} latents; "
                    f"expected {expected_local_latents}"
                )
            local_frames = 1 + (expected_local_latents - 1) * 8
            prompt = prompt_schedule[target_block]
            video_context = prompt_contexts[prompt].to(device=device, dtype=dtype)
            if video_context.ndim == 2:
                video_context = video_context.unsqueeze(0)

            handoff_bridge = None
            handoff_global_range = None
            handoff_local_range = None
            if handoff_mode and sequence_index > 0:
                if committed_latent is None:
                    raise RuntimeError("Missing committed latent before handoff window")
                bridge_start = history_end - int(args.handoff_latents)
                if bridge_start < latent_start:
                    raise RuntimeError(
                        f"Handoff bridge {bridge_start}:{history_end} falls outside "
                        f"window {latent_start}:{latent_end}"
                    )
                if int(committed_latent.shape[2]) != history_end:
                    raise RuntimeError(
                        "Committed-latent cursor drift: "
                        f"have={committed_latent.shape[2]} expected={history_end}"
                    )
                handoff_bridge = committed_latent[
                    :, :, bridge_start:history_end
                ].detach()
                handoff_local_start = bridge_start - latent_start
                local_conditionings = [
                    VideoConditionByLatentIndex(
                        latent=handoff_bridge,
                        strength=1.0,
                        latent_idx=handoff_local_start,
                    )
                ]
                handoff_global_range = [bridge_start, history_end]
                handoff_local_range = [
                    handoff_local_start,
                    handoff_local_start + int(args.handoff_latents),
                ]
                anchor_image = None
            else:
                local_conditionings = conditioning_by_start[window_start]
                anchor_image = str(anchor_paths[window_start].resolve())

            conditioned_local_latents = (
                int(args.handoff_latents)
                if handoff_mode and sequence_index > 0
                else 1
            )
            mutable_local_latents = expected_local_latents - conditioned_local_latents

            if precomputed_noise is None:
                local_noiser = noiser
                noise_global_range = None
            else:
                local_noiser = FixedVideoNoiser(
                    precomputed_noise[:, :, latent_start:latent_end]
                )
                noise_global_range = [latent_start, latent_end]

            block_start = time.perf_counter()
            with torch.inference_mode():
                refined, _ = stage2(
                    denoiser=SimpleDenoiserNoCamTokens(
                        v_context=video_context,
                        a_context=None,
                    ),
                    sigmas=sigmas,
                    noiser=local_noiser,
                    width=int(geometry["final_width"]),
                    height=int(geometry["final_height"]),
                    frames=local_frames,
                    fps=float(geometry["fps"]),
                    video=ModalitySpec(
                        context=video_context,
                        conditionings=local_conditionings,
                        noise_scale=float(SIGMAS[0]),
                        initial_latent=local_initial,
                    ),
                    audio=None,
                )
                if refined is None:
                    raise RuntimeError(f"Refiner returned no video for block {target_block}")
                if not handoff_mode:
                    if decoder is None:
                        raise AssertionError("Window decoder was not initialized")
                    local_video = decode_video_latent(
                        decoder, refined.latent, window_tiling
                    )

            handoff_preserved_exactly = None
            committed_latent_range = None
            if handoff_mode:
                if sequence_index == 0:
                    committed_piece = refined.latent.detach().clone()
                    committed_latent = committed_piece
                    committed_latent_range = [0, latent_end]
                    expected_start_frame = 0
                else:
                    if handoff_bridge is None or handoff_local_range is None:
                        raise AssertionError("Missing handoff bridge metadata")
                    handoff_preserved_exactly = torch.equal(
                        refined.latent[
                            :, :, handoff_local_range[0] : handoff_local_range[1]
                        ],
                        handoff_bridge,
                    )
                    if not handoff_preserved_exactly:
                        raise RuntimeError(
                            f"Refiner changed clean handoff latents for target block {target_block}"
                        )
                    committed_piece = refined.latent[
                        :, :, -LATENTS_PER_BLOCK:
                    ].detach().clone()
                    committed_latent = torch.cat(
                        (committed_latent, committed_piece), dim=2
                    )
                    committed_latent_range = [
                        latent_end - LATENTS_PER_BLOCK,
                        latent_end,
                    ]
                    expected_start_frame = 1 + target_block * BLOCK_FRAMES
                if int(committed_latent.shape[2]) != latent_end:
                    raise RuntimeError(
                        f"Committed latent count {committed_latent.shape[2]}; expected {latent_end}"
                    )
                committed_frames = 1 + (target_block + 1) * BLOCK_FRAMES
                committed_value = committed_piece
                newly_committed_frames = (
                    WINDOW_FRAMES if sequence_index == 0 else BLOCK_FRAMES
                )
            else:
                if tuple(local_video.shape[:2]) != (3, WINDOW_FRAMES):
                    raise RuntimeError(
                        f"Unexpected decoded window shape {tuple(local_video.shape)}"
                    )
                if sequence_index == 0:
                    newly_committed = local_video
                    expected_start_frame = 0
                else:
                    newly_committed = local_video[:, -BLOCK_FRAMES:]
                    expected_start_frame = 1 + target_block * BLOCK_FRAMES
                if committed_frames != expected_start_frame:
                    raise RuntimeError(
                        f"Commit cursor drift before block {target_block}: "
                        f"cursor={committed_frames}, expected={expected_start_frame}"
                    )
                committed_uint8 = newly_committed.to(torch.uint8).contiguous()
                committed_chunks.append(committed_uint8)
                committed_frames += int(committed_uint8.shape[1])
                committed_value = committed_uint8
                newly_committed_frames = int(committed_uint8.shape[1])

            new_chunk_sha256 = tensor_sha256(committed_value)
            committed_chunk_sha256.append(new_chunk_sha256)
            new_prefix_sha256 = hashlib.sha256(
                "\n".join(committed_chunk_sha256).encode("ascii")
            )
            new_prefix_sha256 = new_prefix_sha256.hexdigest()
            seconds = time.perf_counter() - block_start
            record = {
                "target_block": target_block,
                "window_blocks": (
                    None if pairwise_increment else [window_start, target_block]
                ),
                "pairwise_target_only": pairwise_increment,
                "local_frames": local_frames,
                "local_latents": expected_local_latents,
                "conditioned_local_latents": conditioned_local_latents,
                "mutable_local_latents": mutable_local_latents,
                "window_global_frame_range_inclusive": [
                    latent_start * 8,
                    (latent_end - 1) * 8,
                ],
                "window_latent_range": [latent_start, latent_end],
                "noise_global_latent_range_half_open": noise_global_range,
                "prompt_schedule_index": target_block,
                "anchor_image": anchor_image,
                "commit_domain": "latent" if handoff_mode else "rgb",
                "committed_latent_range_half_open": committed_latent_range,
                "handoff_global_latent_range_half_open": handoff_global_range,
                "handoff_local_latent_range_half_open": handoff_local_range,
                "handoff_preserved_exactly": handoff_preserved_exactly,
                "committed_frame_range_half_open": [
                    expected_start_frame,
                    committed_frames,
                ],
                "committed_frames_after": committed_frames,
                "commit_chain_sha256_before": prior_prefix_sha256,
                "commit_chain_sha256_after": new_prefix_sha256,
                "new_chunk_sha256": new_chunk_sha256,
                "rollback_frames": 0,
                "seconds": seconds,
            }
            records.append(record)
            with (output_dir / "commit_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            prior_prefix_sha256 = new_prefix_sha256
            print(
                f"[{'pairwise' if pairwise_increment else ('latent-handoff' if handoff_mode else 'append-only')}-refiner] "
                f"block={target_block}/{last_target} "
                f"latents={latent_start}:{latent_end} mutable={mutable_local_latents} "
                f"append_frames={newly_committed_frames} "
                f"total={committed_frames} seconds={seconds:.1f}",
                flush=True,
            )
            del local_initial, video_context, refined, committed_value
            if handoff_bridge is not None:
                del handoff_bridge
            if precomputed_noise is not None:
                del local_noiser
            if handoff_mode:
                del committed_piece
            else:
                del local_video, newly_committed, committed_uint8
            gc.collect()
            torch.cuda.empty_cache()

        expected_frames = 1 + (last_target + 1) * BLOCK_FRAMES
        if committed_frames != expected_frames:
            raise RuntimeError(
                f"Final committed frame count {committed_frames}; expected {expected_frames}"
            )
        refined_latent_path = None
        final_video = None
        silent = None
        final_path = None
        if handoff_mode:
            expected_latents = 1 + (last_target + 1) * LATENTS_PER_BLOCK
            if committed_latent is None or int(committed_latent.shape[2]) != expected_latents:
                raise RuntimeError(
                    "Final committed latent mismatch: "
                    f"have={None if committed_latent is None else committed_latent.shape[2]} "
                    f"expected={expected_latents}"
                )
            if args.save_refined_latent:
                refined_latent_path = output_dir / "sliding_refiner0p7_latent_handoff.pt"
                torch.save(committed_latent.detach().cpu(), refined_latent_path)
            stage2._cached_transformer = None
            stage2_discarded = True
            cleanup_cuda()
            if not args.defer_decode:
                decoder = decoder_owner._decoder_builder.build(device=device, dtype=dtype).to(device).eval()
                temporal_tile_frames = ((expected_frames + 7) // 8) * 8
                full_tiling = TilingConfig(
                    spatial_config=SpatialTilingConfig(512, 64),
                    temporal_config=TemporalTilingConfig(temporal_tile_frames, 0),
                )
                final_video = decode_video_latent(decoder, committed_latent, full_tiling)
                silent = output_dir / "sliding_refiner0p7_latent_handoff_silent.mp4"
                final_path = output_dir / "sliding_refiner0p7_latent_handoff.mp4"
        else:
            final_video = torch.cat(committed_chunks, dim=1)
            silent = output_dir / "sliding_refiner0p7_append_only_silent.mp4"
            final_path = output_dir / "sliding_refiner0p7_append_only.mp4"
        if final_video is not None:
            if silent is None or final_path is None:
                raise AssertionError("Missing output paths for decoded video")
            save_video(final_video, silent, fps=float(geometry["fps"]))
            mux_source_audio(silent, source_raw_video, final_path)
        elapsed = time.perf_counter() - start_time
        run_info = {
            "schema_version": (
                "xonotic_pairwise_refiner_latent_handoff_v1"
                if args.pairwise_target_only
                else "xonotic_sliding_refiner_latent_handoff_v1"
                if handoff_mode
                else "xonotic_sliding_refiner_append_only_v1"
            ),
            "source_cache": str(args.cache.resolve()),
            "source_cache_schema": source_cache_schema,
            "source_raw_video": str(source_raw_video),
            "refiner": "official_ltx23_original_non_action",
            "refiner_input_contract": {
                "stage1_upscaled_latent": True,
                "single_text_context_per_window": True,
                "standard_image_conditioning_at_local_frame_zero": (
                    "first_window_only" if handoff_mode else True
                ),
                "action_conditioned": False,
                "enemy_reference_conditioned": False,
                "causal_or_custom_attention_mask": False,
                "frozen_history_latent": handoff_mode,
                "refined_history_feedback": handoff_mode,
                "global_rope_override": False,
            },
            "append_only": True,
            "commit_domain": "latent" if handoff_mode else "rgb",
            "handoff_latents": int(args.handoff_latents),
            "pairwise_target_only": bool(args.pairwise_target_only),
            "mutable_latents_per_incremental_call": (
                LATENTS_PER_BLOCK
                if args.pairwise_target_only
                else WINDOW_LATENTS - int(args.handoff_latents)
                if handoff_mode
                else WINDOW_LATENTS - 1
            ),
            "incremental_refiner_latents": (
                int(args.handoff_latents) + LATENTS_PER_BLOCK
                if args.pairwise_target_only
                else WINDOW_LATENTS
            ),
            "rollback_frames": 0,
            "history_blocks": HISTORY_BLOCKS,
            "target_blocks_per_window": 1,
            "bootstrap_window_frames": WINDOW_FRAMES,
            "incremental_window_frames": (
                1 + (int(args.handoff_latents) + LATENTS_PER_BLOCK - 1) * 8
                if args.pairwise_target_only
                else WINDOW_FRAMES
            ),
            "first_target_block": int(args.first_target_block),
            "last_target_block": last_target,
            "committed_frames": committed_frames,
            "sigma_schedule": list(SIGMAS),
            "noise_contract": (
                "precomputed_exact_global_temporal_slice"
                if precomputed_noise is not None
                else "sequential_gaussian_generator"
            ),
            "precomputed_noise": (
                None
                if args.precomputed_noise is None
                else str(args.precomputed_noise.resolve())
            ),
            "upsampler_execution": (
                "full_clip_internal"
                if args.pre_upscaled_latent is None
                else "precomputed_diagnostic_override"
            ),
            "pre_upscaled_latent": (
                None
                if args.pre_upscaled_latent is None
                else str(args.pre_upscaled_latent.resolve())
            ),
            "prompt_policy": "target block prompt only",
            "prompt_context_source": prompt_context_source,
            "anchor_policy": (
                "RGB frame zero for first window, then exact recent refined-latent bridge"
                if handoff_mode
                else "raw Stage-1 decoded frame at local window frame zero"
            ),
            "decoder_mode": (
                "deferred_to_fresh_process"
                if args.defer_decode
                else "single full-temporal spatial-tiled decode"
                if handoff_mode
                else args.decoder_mode
            ),
            "seconds": elapsed,
            "final_commit_chain_sha256": prior_prefix_sha256,
            "video": None if final_path is None else str(final_path.resolve()),
            "refined_latent": (
                None
                if refined_latent_path is None
                else str(refined_latent_path.resolve())
            ),
            "commit_log": str((output_dir / "commit_log.jsonl").resolve()),
            "records": records,
        }
        info_name = "refine_run_info.json" if args.defer_decode else "run_info.json"
        write_json(output_dir / info_name, run_info)
        print(json.dumps(run_info, indent=2, ensure_ascii=False), flush=True)
    finally:
        if not stage2_discarded:
            stage2.release()
        if decoder is not None:
            decoder.to("cpu")
        cleanup_cuda()


if __name__ == "__main__":
    main()
