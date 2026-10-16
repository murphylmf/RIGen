''

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.training_strategies.ar_teacher_masks import AVBlock, ModalityKind


def _token_ranges(
    blocks: list[AVBlock],
    kind: ModalityKind,
    tokens_per_latent: int,
) -> list[tuple[int, int]]:
    if not blocks:
        raise ValueError("At least one AV block is required.")
    if tokens_per_latent <= 0:
        raise ValueError(f"tokens_per_latent must be positive, got {tokens_per_latent}.")
    return [
        (start * tokens_per_latent, end * tokens_per_latent)
        for start, end in (block.range_for(kind) for block in blocks)
    ]


def build_sliding_ref_self_mask(
    blocks: list[AVBlock],
    *,
    kind: ModalityKind,
    tokens_per_latent: int,
    reference_tokens: int = 0,
    device: torch.device,
) -> Tensor:
    ''

    if reference_tokens < 0:
        raise ValueError(f"reference_tokens must be non-negative, got {reference_tokens}.")
    ranges = _token_ranges(blocks, kind, tokens_per_latent)
    world_tokens = ranges[-1][1]
    total_tokens = world_tokens + reference_tokens
    mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)
    for start, end in ranges:
        mask[start:end, :end] = True

    if reference_tokens:
        reference_start = world_tokens
        target_start, target_end = ranges[-1]
        mask[target_start:target_end, reference_start:] = True
        mask[reference_start:, reference_start:] = True
    return mask


def build_sliding_ref_cross_mask(
    blocks: list[AVBlock],
    *,
    query_kind: ModalityKind,
    key_kind: ModalityKind,
    query_tokens_per_latent: int,
    key_tokens_per_latent: int,
    query_reference_tokens: int = 0,
    key_reference_tokens: int = 0,
    device: torch.device,
) -> Tensor:
    ''

    if query_reference_tokens < 0 or key_reference_tokens < 0:
        raise ValueError("Reference token counts must be non-negative.")
    query_ranges = _token_ranges(blocks, query_kind, query_tokens_per_latent)
    key_ranges = _token_ranges(blocks, key_kind, key_tokens_per_latent)
    query_world_tokens = query_ranges[-1][1]
    key_world_tokens = key_ranges[-1][1]
    mask = torch.zeros(
        query_world_tokens + query_reference_tokens,
        key_world_tokens + key_reference_tokens,
        dtype=torch.bool,
        device=device,
    )
    for (query_start, query_end), (_key_start, key_end) in zip(
        query_ranges,
        key_ranges,
        strict=True,
    ):
        mask[query_start:query_end, :key_end] = True
    return mask


def build_sliding_ref_prompt_context_mask(
    blocks: list[AVBlock],
    *,
    kind: ModalityKind,
    tokens_per_latent: int,
    prompt_attention_mask: Tensor,
    reference_tokens: int = 0,
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
    if reference_tokens < 0:
        raise ValueError(f"reference_tokens must be non-negative, got {reference_tokens}.")

    ranges = _token_ranges(blocks, kind, tokens_per_latent)
    world_tokens = ranges[-1][1]
    context_tokens = prompt_count * prompt_length
    mask = torch.zeros(
        batch,
        world_tokens + reference_tokens,
        context_tokens,
        dtype=torch.bool,
        device=prompt_attention_mask.device,
    )
    valid = prompt_attention_mask.to(dtype=torch.bool)
                                                                       
    for block, (start, end) in zip(blocks[1:], ranges[1:], strict=True):
        prompt_index = block.index - 1
        context_start = prompt_index * prompt_length
        context_end = context_start + prompt_length
        mask[:, start:end, context_start:context_end] = valid[:, prompt_index, None, :]
    return mask


def build_sliding_ref_prompt_context_timesteps(
    block_sigmas: Tensor,
    prompt_attention_mask: Tensor,
) -> Tensor:
    ''

    if block_sigmas.ndim != 2:
        raise ValueError(f"block_sigmas must be 2D, got {tuple(block_sigmas.shape)}.")
    batch, prompt_count, prompt_length = prompt_attention_mask.shape
    if block_sigmas.shape != (batch, prompt_count + 1):
        raise ValueError(f"block_sigmas must have shape {(batch, prompt_count + 1)}, got {tuple(block_sigmas.shape)}.")
    return block_sigmas[:, 1:, None].expand(-1, -1, prompt_length).flatten(1)


__all__ = [
    "build_sliding_ref_cross_mask",
    "build_sliding_ref_prompt_context_mask",
    "build_sliding_ref_prompt_context_timesteps",
    "build_sliding_ref_self_mask",
]
