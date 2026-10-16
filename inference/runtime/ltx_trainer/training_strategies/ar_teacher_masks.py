''

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from ltx_core.model.transformer.flex_attention import (
    create_structured_block_mask,
    flex_attention_available,
)


ModalityKind = Literal["video", "audio"]
MaskBackend = Literal["auto", "flex", "dense"]


@dataclass(frozen=True)
class AVBlock:
    ''

    index: int
    video_start: int
    video_end: int
    audio_start: int
    audio_end: int

    def range_for(self, kind: ModalityKind) -> tuple[int, int]:
        if kind == "video":
            return self.video_start, self.video_end
        return self.audio_start, self.audio_end


def compute_ti2av_blocks(
    total_video_latents: int,
    total_audio_latents: int | None,
    *,
    chunk_size: int,
    video_fps: float,
    video_temporal_scale: int = 8,
    audio_latent_fps: float = 25.0,
) -> list[AVBlock]:
    ''

    total_video_latents = int(total_video_latents)
    total_audio_latents = int(total_audio_latents) if total_audio_latents is not None else None
    chunk_size = int(chunk_size)
    video_fps = float(video_fps)
    if total_video_latents < 2:
        raise ValueError(f"TI2AV requires at least two video latents, got {total_video_latents}.")
    if total_audio_latents is not None and total_audio_latents < 1:
        raise ValueError(f"TI2AV requires at least one audio latent, got {total_audio_latents}.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if (total_video_latents - 1) % chunk_size:
        raise ValueError(
            "The generated video tail must contain complete chunks: "
            f"(video_latents - 1)={total_video_latents - 1} is not divisible by chunk_size={chunk_size}."
        )
    if not math.isfinite(video_fps) or video_fps <= 0:
        raise ValueError(f"video_fps must be finite and positive, got {video_fps}.")
    if video_temporal_scale <= 0 or audio_latent_fps <= 0:
        raise ValueError("video_temporal_scale and audio_latent_fps must be positive.")

    def aligned_audio_end(video_end: int) -> int:
        elapsed_after_first_frame = (video_end - 1) * video_temporal_scale / video_fps
        return 1 + round(elapsed_after_first_frame * audio_latent_fps)

    blocks = [AVBlock(index=0, video_start=0, video_end=1, audio_start=0, audio_end=1)]
    video_start = 1
    audio_start = 1
    block_index = 1
    while video_start < total_video_latents:
        video_end = video_start + chunk_size
        audio_end = aligned_audio_end(video_end)
        blocks.append(
            AVBlock(
                index=block_index,
                video_start=video_start,
                video_end=video_end,
                audio_start=audio_start,
                audio_end=audio_end,
            )
        )
        video_start = video_end
        audio_start = audio_end
        block_index += 1

    if total_audio_latents is not None and blocks[-1].audio_end != total_audio_latents:
        raise ValueError(
            "Audio/video latent geometry is not time-aligned: "
            f"video_latents={total_video_latents}, fps={video_fps}, chunk_size={chunk_size} "
            f"imply {blocks[-1].audio_end} audio latents, but the batch contains {total_audio_latents}."
        )
    return blocks


def _token_ranges(
    blocks: list[AVBlock],
    kind: ModalityKind,
    tokens_per_latent: int,
) -> list[tuple[int, int]]:
    if tokens_per_latent <= 0:
        raise ValueError(f"tokens_per_latent must be positive, got {tokens_per_latent}.")
    return [
        (start * tokens_per_latent, end * tokens_per_latent)
        for start, end in (block.range_for(kind) for block in blocks)
    ]


def _dense_teacher_forcing_self_mask(
    token_ranges: list[tuple[int, int]],
    *,
    device: torch.device,
) -> Tensor:
    total = token_ranges[-1][1]
    mask = torch.zeros(2 * total, 2 * total, dtype=torch.bool, device=device)
    for clean_start, clean_end in token_ranges:
        mask[clean_start:clean_end, :clean_end] = True
        noisy_start = total + clean_start
        noisy_end = total + clean_end
        if clean_start:
            mask[noisy_start:noisy_end, :clean_start] = True
        mask[noisy_start:noisy_end, noisy_start:noisy_end] = True
    return mask


def _flex_teacher_forcing_self_mask(
    token_ranges: list[tuple[int, int]],
    *,
    device: torch.device,
) -> object:
    total = token_ranges[-1][1]
    total_length = 2 * total
    padded_length = math.ceil(total_length / 128) * 128

    clean_ends = torch.zeros(padded_length, device=device, dtype=torch.long)
    noisy_context_ends = torch.zeros_like(clean_ends)
    noisy_self_starts = torch.zeros_like(clean_ends)
    noisy_self_ends = torch.zeros_like(clean_ends)
    for clean_start, clean_end in token_ranges:
        clean_ends[clean_start:clean_end] = clean_end
        noisy_start = total + clean_start
        noisy_end = total + clean_end
        noisy_context_ends[noisy_start:noisy_end] = clean_start
        noisy_self_starts[noisy_start:noisy_end] = noisy_start
        noisy_self_ends[noisy_start:noisy_end] = noisy_end

                                                                            
                                                                                
    clean_ends[total_length:] = total_length
    noisy_self_ends[total_length:] = total_length

    def mask_fn(_batch: Tensor, _head: Tensor, query_index: Tensor, key_index: Tensor) -> Tensor:
        clean_query = (query_index < total) & (key_index < clean_ends[query_index])
        noisy_context = (query_index >= total) & (key_index < noisy_context_ends[query_index])
        noisy_self = (
            (query_index >= total)
            & (key_index >= noisy_self_starts[query_index])
            & (key_index < noisy_self_ends[query_index])
        )
        return (query_index == key_index) | clean_query | noisy_context | noisy_self

    return create_structured_block_mask(
        mask_fn,
        query_length=total_length,
        key_length=total_length,
        device=device,
    )


def build_teacher_forcing_self_mask(
    blocks: list[AVBlock],
    *,
    kind: ModalityKind,
    tokens_per_latent: int,
    device: torch.device,
    backend: MaskBackend,
) -> Tensor | object:
    ''

    resolved_backend = backend
    if resolved_backend == "auto":
        resolved_backend = "flex" if device.type == "cuda" else "dense"
    token_ranges = _token_ranges(blocks, kind, tokens_per_latent)
    if resolved_backend == "dense":
        return _dense_teacher_forcing_self_mask(token_ranges, device=device)
    if resolved_backend != "flex":
        raise ValueError(f"Unknown AR mask backend: {backend!r}")
    if not flex_attention_available():
        raise RuntimeError("mask_backend='flex' requires a PyTorch build with FlexAttention.")
    if device.type != "cuda":
        raise RuntimeError("mask_backend='flex' is intended for CUDA training; use 'dense' for CPU tests.")
    return _flex_teacher_forcing_self_mask(token_ranges, device=device)


def build_teacher_forcing_cross_mask(
    blocks: list[AVBlock],
    *,
    query_kind: ModalityKind,
    key_kind: ModalityKind,
    query_tokens_per_latent: int,
    key_tokens_per_latent: int,
    device: torch.device,
) -> Tensor:
    ''

    query_ranges = _token_ranges(blocks, query_kind, query_tokens_per_latent)
    key_ranges = _token_ranges(blocks, key_kind, key_tokens_per_latent)
    query_total = query_ranges[-1][1]
    key_total = key_ranges[-1][1]
    mask = torch.zeros(2 * query_total, 2 * key_total, dtype=torch.bool, device=device)
    for query_range, key_range in zip(query_ranges, key_ranges, strict=True):
        query_start, query_end = query_range
        key_start, key_end = key_range
        mask[query_start:query_end, :key_end] = True

        noisy_query_start = query_total + query_start
        noisy_query_end = query_total + query_end
        if key_start:
            mask[noisy_query_start:noisy_query_end, :key_start] = True
        mask[
            noisy_query_start:noisy_query_end,
            key_total + key_start : key_total + key_end,
        ] = True
    return mask


def expand_block_values_to_tokens(
    block_values: Tensor,
    blocks: list[AVBlock],
    *,
    kind: ModalityKind,
    tokens_per_latent: int,
) -> Tensor:
    ''

    if block_values.ndim != 2 or block_values.shape[1] != len(blocks):
        raise ValueError(f"block_values must have shape [B, {len(blocks)}], got {tuple(block_values.shape)}.")
    total_tokens = blocks[-1].range_for(kind)[1] * tokens_per_latent
    expanded = block_values.new_zeros(block_values.shape[0], total_tokens)
    for block, (start, end) in zip(
        blocks,
        _token_ranges(blocks, kind, tokens_per_latent),
        strict=True,
    ):
        expanded[:, start:end] = block_values[:, block.index : block.index + 1]
    return expanded


def build_action_query_latent_indices(
    *,
    total_video_latents: int,
    spatial_tokens: int,
    device: torch.device,
) -> Tensor:
    ''

    one_half = torch.arange(total_video_latents, device=device, dtype=torch.long).repeat_interleave(spatial_tokens)
    return one_half.repeat(2)


def build_prompt_context_mask(
    blocks: list[AVBlock],
    *,
    kind: ModalityKind,
    tokens_per_latent: int,
    prompt_attention_mask: Tensor,
) -> Tensor:
    ''

    if prompt_attention_mask.ndim != 3:
        raise ValueError(
            "prompt_attention_mask must have shape [B, generated_blocks, prompt_tokens], "
            f"got {tuple(prompt_attention_mask.shape)}."
        )
    batch, prompt_count, prompt_length = prompt_attention_mask.shape
    expected_prompt_count = len(blocks) - 1
    if prompt_count != expected_prompt_count:
        raise ValueError(f"Expected {expected_prompt_count} per-chunk prompts, got {prompt_count}.")

    token_ranges = _token_ranges(blocks, kind, tokens_per_latent)
    total_tokens = token_ranges[-1][1]
    half_context = prompt_count * prompt_length
    mask = torch.zeros(
        batch,
        2 * total_tokens,
        2 * half_context,
        dtype=torch.bool,
        device=prompt_attention_mask.device,
    )
    valid = prompt_attention_mask.to(dtype=torch.bool)
    for block, (start, end) in zip(blocks, token_ranges, strict=True):
        prompt_index = max(block.index - 1, 0)
        clean_context_start = prompt_index * prompt_length
        clean_context_end = clean_context_start + prompt_length
        mask[:, start:end, clean_context_start:clean_context_end] = valid[:, prompt_index, None, :]

        noisy_start = total_tokens + start
        noisy_end = total_tokens + end
        if block.index == 0:
            context_start = clean_context_start
        else:
            context_start = half_context + clean_context_start
        context_end = context_start + prompt_length
        mask[:, noisy_start:noisy_end, context_start:context_end] = valid[:, prompt_index, None, :]
    return mask


def build_prompt_context_timesteps(
    block_sigmas: Tensor,
    prompt_attention_mask: Tensor,
) -> Tensor:
    ''

    if block_sigmas.ndim != 2:
        raise ValueError(f"block_sigmas must be 2D, got {tuple(block_sigmas.shape)}.")
    batch, prompt_count, prompt_length = prompt_attention_mask.shape
    if block_sigmas.shape != (batch, prompt_count + 1):
        raise ValueError(f"block_sigmas must have shape {(batch, prompt_count + 1)}, got {tuple(block_sigmas.shape)}.")
    clean = block_sigmas.new_zeros(batch, prompt_count, prompt_length)
    noisy = block_sigmas[:, 1:, None].expand(-1, -1, prompt_length)
    return torch.cat([clean.flatten(1), noisy.flatten(1)], dim=1)


__all__ = [
    "AVBlock",
    "MaskBackend",
    "build_action_query_latent_indices",
    "build_prompt_context_mask",
    "build_prompt_context_timesteps",
    "build_teacher_forcing_cross_mask",
    "build_teacher_forcing_self_mask",
    "compute_ti2av_blocks",
    "expand_block_values_to_tokens",
]
