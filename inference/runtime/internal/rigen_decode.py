                      
''

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for relative in (
    "packages/ltx-trainer/src",
    "packages/ltx-core/src",
    "packages/ltx-pipelines/src",
):
    source = REPO_ROOT / relative
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from ltx_core.model.video_vae import (              
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
)
from ltx_pipelines.utils.blocks import VideoDecoder              
from ltx_trainer.video_utils import save_video              
from rigen_components import cleanup_cuda, collect_decoded_video


BASE_MODEL = Path("weights/ltx-2.3-22b-dev.safetensors")
FULL_SHAPE = (1, 128, 41, 30, 52)
HALF_SHAPE = (1, 128, 21, 30, 52)


def _hash(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _decode(
    latent: torch.Tensor,
    *,
    generator_state: torch.Tensor,
    tiling: TilingConfig,
    model: Path,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    generator = torch.Generator(device=device)
    generator.set_state(generator_state.cpu())
    decoder = VideoDecoder(str(model.resolve()), dtype, device)
    start = time.perf_counter()
    with torch.inference_mode():
        video = collect_decoded_video(
            decoder(
                latent.to(device=device, dtype=dtype),
                tiling_config=tiling,
                generator=generator,
            )
        ).clamp(0, 1)
    elapsed = time.perf_counter() - start
    cleanup_cuda()
    return video, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-latent", required=True, type=Path)
    parser.add_argument("--first8-latent", required=True, type=Path)
    parser.add_argument("--last8-latent", required=True, type=Path)
    parser.add_argument(
        "--midanchor-latent",
        type=Path,
        default=None,
        help="Optional full-length refined latent conditioned again at the 8 s midpoint.",
    )
    parser.add_argument("--source-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()

    required_paths = [
        args.full_latent,
        args.first8_latent,
        args.last8_latent,
        args.source_cache,
        args.base_model,
    ]
    if args.midanchor_latent is not None:
        required_paths.append(args.midanchor_latent)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    full = torch.load(args.full_latent, map_location="cpu", weights_only=True)
    first = torch.load(args.first8_latent, map_location="cpu", weights_only=True)
    last = torch.load(args.last8_latent, map_location="cpu", weights_only=True)
    midanchor = (
        None
        if args.midanchor_latent is None
        else torch.load(args.midanchor_latent, map_location="cpu", weights_only=True)
    )
    if tuple(full.shape) != FULL_SHAPE:
        raise ValueError(f"Expected full latent {FULL_SHAPE}, got {tuple(full.shape)}")
    if tuple(first.shape) != HALF_SHAPE or tuple(last.shape) != HALF_SHAPE:
        raise ValueError(
            f"Expected half latents {HALF_SHAPE}, got {tuple(first.shape)} and {tuple(last.shape)}"
        )
    if midanchor is not None and tuple(midanchor.shape) != FULL_SHAPE:
        raise ValueError(f"Expected midpoint-anchor latent {FULL_SHAPE}, got {tuple(midanchor.shape)}")

                                                                          
                                                                                  
    stitched = torch.cat((first[:, :, :20], last), dim=2).contiguous()
    if tuple(stitched.shape) != FULL_SHAPE:
        raise AssertionError(tuple(stitched.shape))
    stitched_path = args.output_dir / "refined_split8x2_stitched_latent.pt"
    torch.save(stitched, stitched_path)

    cache = torch.load(args.source_cache, map_location="cpu", weights_only=False)
    generator_state = cache["generator_state"].cpu()
    temporal_frames = 328
    tiling = TilingConfig(
        spatial_config=SpatialTilingConfig(512, 64),
        temporal_config=TemporalTilingConfig(temporal_frames, 0),
    )
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device(args.device)
    full_video, full_seconds = _decode(
        full,
        generator_state=generator_state,
        tiling=tiling,
        model=args.base_model,
        dtype=dtype,
        device=device,
    )
    split_video, split_seconds = _decode(
        stitched,
        generator_state=generator_state,
        tiling=tiling,
        model=args.base_model,
        dtype=dtype,
        device=device,
    )
    full_video_path = args.output_dir / "full16_refiner_matched_decode.mp4"
    split_video_path = args.output_dir / "split8x2_refiner_matched_decode.mp4"
    save_video(full_video, full_video_path, fps=20)
    save_video(split_video, split_video_path, fps=20)
    midanchor_video_path = None
    midanchor_seconds = None
    if midanchor is not None:
        midanchor_video, midanchor_seconds = _decode(
            midanchor,
            generator_state=generator_state,
            tiling=tiling,
            model=args.base_model,
            dtype=dtype,
            device=device,
        )
        midanchor_video_path = args.output_dir / "full16_midanchor_refiner_matched_decode.mp4"
        save_video(midanchor_video, midanchor_video_path, fps=20)

    direct_pixel_mae = float((full_video.float() - split_video.float()).abs().mean())
    report = {
        "schema_version": "alpine-refiner-matched-length-decode-v1",
        "stitch_contract": "first8 latent indices 0:20 plus all last8 indices 0:21; standalone last8 owns shared anchor",
        "decode_contract": {
            "same_base_model": str(args.base_model.resolve()),
            "same_generator_state": True,
            "generator_state_sha256": _hash(generator_state),
            "same_full_temporal_decode": [temporal_frames, 0],
            "same_spatial_tiling": [512, 64],
        },
        "latents": {
            "full16": {"path": str(args.full_latent.resolve()), "sha256": _hash(full)},
            "first8": {"path": str(args.first8_latent.resolve()), "sha256": _hash(first)},
            "last8": {"path": str(args.last8_latent.resolve()), "sha256": _hash(last)},
            "split8x2_stitched": {"path": str(stitched_path.resolve()), "sha256": _hash(stitched)},
            "full16_midanchor": (
                None
                if midanchor is None
                else {"path": str(args.midanchor_latent.resolve()), "sha256": _hash(midanchor)}
            ),
        },
        "videos": {
            "full16": str(full_video_path.resolve()),
            "split8x2": str(split_video_path.resolve()),
            "full16_midanchor": (
                None if midanchor_video_path is None else str(midanchor_video_path.resolve())
            ),
        },
        "direct_full_vs_split_pixel_mae": direct_pixel_mae,
        "seconds": {
            "full16_decode": full_seconds,
            "split8x2_decode": split_seconds,
            "full16_midanchor_decode": midanchor_seconds,
        },
    }
    (args.output_dir / "run_info.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
