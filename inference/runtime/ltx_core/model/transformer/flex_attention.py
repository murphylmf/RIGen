''

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask,
        flex_attention as _flex_attention,
    )
except ImportError:                                                          
    BlockMask = None                                 
    create_block_mask = None
    _flex_attention = None


_compiled_flex_attention: Callable[..., torch.Tensor] | None = None
_FLEX_ATTENTION_COMPILE_MODE_ENV = "LTX_FLEX_ATTENTION_COMPILE_MODE"


def _raise_dynamo_recompile_limits() -> None:
    ''

    dynamo = getattr(torch, "_dynamo", None)
    config = getattr(dynamo, "config", None)
    if config is None:
        return
    for name, minimum in (
        ("recompile_limit", 256),
        ("cache_size_limit", 256),
        ("accumulated_recompile_limit", 1024),
        ("accumulated_cache_size_limit", 1024),
    ):
        if not hasattr(config, name):
            continue
        try:
            setattr(config, name, max(int(getattr(config, name)), minimum))
        except (TypeError, ValueError):
            continue


def flex_attention_available() -> bool:
    ''

    return BlockMask is not None and create_block_mask is not None and _flex_attention is not None


def is_block_mask(value: object) -> bool:
    ''

    return BlockMask is not None and isinstance(value, BlockMask)


def _get_compiled_flex_attention() -> Callable[..., torch.Tensor]:
    global _compiled_flex_attention
    if _flex_attention is None:
        raise RuntimeError("A structured attention BlockMask requires PyTorch FlexAttention support.")
    if _compiled_flex_attention is None:
        _raise_dynamo_recompile_limits()
                                                                           
                                                                             
        compile_mode = os.environ.get(_FLEX_ATTENTION_COMPILE_MODE_ENV, "default")
        _compiled_flex_attention = torch.compile(
            _flex_attention,
            dynamic=True,
            mode=compile_mode,
        )
    return _compiled_flex_attention


def _padded_length(length: int, multiple: int = 128) -> int:
    return math.ceil(length / multiple) * multiple


def flex_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    heads: int,
    block_mask: object,
) -> torch.Tensor:
    ''

    if not is_block_mask(block_mask):
        raise TypeError(f"Expected a FlexAttention BlockMask, got {type(block_mask).__name__}.")

    batch, query_length, projected_dim = q.shape
    if projected_dim % heads:
        raise ValueError(f"Projected attention dim {projected_dim} is not divisible by heads={heads}.")
    key_length = int(k.shape[1])
    if k.shape != v.shape:
        raise ValueError(f"FlexAttention key/value shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")

    head_dim = projected_dim // heads
    q_heads = q.view(batch, query_length, heads, head_dim).transpose(1, 2)
    k_heads = k.view(batch, key_length, heads, head_dim).transpose(1, 2)
    v_heads = v.view(batch, key_length, heads, head_dim).transpose(1, 2)

    query_padding = _padded_length(query_length) - query_length
    key_padding = _padded_length(key_length) - key_length
    if query_padding:
        q_heads = F.pad(q_heads, (0, 0, 0, query_padding))
    if key_padding:
        k_heads = F.pad(k_heads, (0, 0, 0, key_padding))
        v_heads = F.pad(v_heads, (0, 0, 0, key_padding))

    out = _get_compiled_flex_attention()(
        q_heads,
        k_heads,
        v_heads,
        block_mask=block_mask,
    )
    out = out[:, :, :query_length].transpose(1, 2).reshape(batch, query_length, projected_dim)
    return out


def create_structured_block_mask(
    mask_fn: Callable[[Any, Any, Any, Any], Any],
    *,
    query_length: int,
    key_length: int,
    device: torch.device | str,
    compile_mask: bool = True,
) -> object:
    ''

    if create_block_mask is None:
        raise RuntimeError("PyTorch FlexAttention is unavailable; use the dense mask backend for tests.")
    return create_block_mask(
        mask_fn,
        B=None,
        H=None,
        Q_LEN=_padded_length(query_length),
        KV_LEN=_padded_length(key_length),
        device=device,
        _compile=compile_mask,
    )


__all__ = [
    "create_structured_block_mask",
    "flex_attention_available",
    "flex_attention_forward",
    "is_block_mask",
]
