''

from __future__ import annotations

import weakref

import torch

from ltx_core.model.transformer.attention import (
    FlashAttention4,
    FlashAttention4BooleanMask,
)


MaskPlanSegment = tuple[
    int,
    int,
    int | None,
    int | None,
    torch.Tensor | None,
]


class ExactSegmentedFlashAttention4(FlashAttention4):
    ''

    _plan_cache: dict[
        int,
        tuple[
            weakref.ReferenceType[torch.Tensor],
            int,
            tuple[MaskPlanSegment, ...],
        ],
    ] = {}

    @classmethod
    def clear_plan_cache(cls) -> None:
        ''

        cls._plan_cache.clear()

    @classmethod
    def _mask_plan(
        cls,
        mask: FlashAttention4BooleanMask,
    ) -> tuple[MaskPlanSegment, ...]:
        boolean_mask = mask.tensor
        source = mask.source_tensor if mask.source_tensor is not None else boolean_mask
        cache_key = id(source)
        try:
            source_version = int(source._version)
        except RuntimeError:
                                                                            
                                                                            
                                      
            source_version = -1
        cached = cls._plan_cache.get(cache_key)
        if cached is not None and cached[0]() is source and cached[1] == source_version:
            return cached[2]

        query_length, key_length = boolean_mask.shape
        if query_length == 1:
            starts = [0]
        else:
            row_changed = torch.empty(
                query_length,
                dtype=torch.bool,
                device=boolean_mask.device,
            )
            row_changed[0] = True
            row_changed[1:] = torch.any(
                boolean_mask[1:] != boolean_mask[:-1],
                dim=1,
            )
            starts = torch.nonzero(row_changed, as_tuple=False).flatten().cpu().tolist()
        ends = starts[1:] + [query_length]

        plan: list[MaskPlanSegment] = []
        for query_start, query_end in zip(starts, ends, strict=True):
            key_indices = torch.nonzero(
                boolean_mask[query_start],
                as_tuple=False,
            ).flatten()
            selected_keys = int(key_indices.numel())
            if selected_keys == 0:
                plan.append((query_start, query_end, None, None, key_indices))
                continue
            key_start = int(key_indices[0].item())
            key_end = int(key_indices[-1].item()) + 1
            if key_end - key_start == selected_keys:
                plan.append((query_start, query_end, key_start, key_end, None))
            else:
                plan.append((query_start, query_end, None, None, key_indices))

        frozen_plan = tuple(plan)

        def remove_stale(
            reference: weakref.ReferenceType[torch.Tensor],
            *,
            key: int = cache_key,
        ) -> None:
            current = cls._plan_cache.get(key)
            if current is not None and current[0] is reference:
                cls._plan_cache.pop(key, None)

        reference = weakref.ref(source, remove_stale)
        cls._plan_cache[cache_key] = (reference, source_version, frozen_plan)
        return frozen_plan

    @staticmethod
    def _validate_inputs(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: FlashAttention4BooleanMask,
    ) -> None:
        if not q.is_cuda or not k.is_cuda or not v.is_cuda:
            raise RuntimeError(
                "Exact segmented FA4 requires CUDA q/k/v tensors; "
                "no CPU fallback is allowed."
            )
        if v.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "Exact segmented FA4 requires float16 or bfloat16 values; "
                f"got v={v.dtype}."
            )
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or k.shape != v.shape:
            raise ValueError(
                "Exact segmented FA4 expects q/k/v shaped [B, T, H*D] with matching k/v; "
                f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
            )
        if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
            raise ValueError(
                "Exact segmented FA4 q/k batch and hidden dimensions must match; "
                f"got q={tuple(q.shape)}, k={tuple(k.shape)}."
            )
        if q.shape[1] == 0 or k.shape[1] == 0:
            raise ValueError("Exact segmented FA4 does not accept empty q/k sequences.")
        if q.shape[-1] % heads:
            raise ValueError(
                f"Exact segmented FA4 hidden dim {q.shape[-1]} is not divisible by {heads} heads."
            )
        boolean_mask = mask.tensor
        expected_shape = (q.shape[1], k.shape[1])
        if boolean_mask.dtype != torch.bool or tuple(boolean_mask.shape) != expected_shape:
            raise ValueError(
                "Exact segmented FA4 mask must be bool [Tq, Tk]; "
                f"expected={expected_shape}, got={boolean_mask.dtype} "
                f"{tuple(boolean_mask.shape)}."
            )
        if not boolean_mask.is_cuda or boolean_mask.device != q.device:
            raise RuntimeError(
                "Exact segmented FA4 mask must share q's CUDA device; "
                f"mask={boolean_mask.device}, q={q.device}."
            )
        if not boolean_mask.is_contiguous():
            raise ValueError(
                "Exact segmented FA4 mask must be contiguous; refusing an implicit copy."
            )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | FlashAttention4BooleanMask | None = None,
    ) -> torch.Tensor:
        if mask is None:
            return super().__call__(q, k, v, heads, None)
        if not isinstance(mask, FlashAttention4BooleanMask):
            raise TypeError(
                "Exact segmented FA4 requires FlashAttention4BooleanMask; "
                "no alternate attention backend is allowed."
            )
        self._validate_inputs(q, k, v, heads, mask)

        outputs: list[torch.Tensor] = []
        for query_start, query_end, key_start, key_end, key_indices in self._mask_plan(mask):
            query = q[:, query_start:query_end]
            if key_indices is not None:
                if key_indices.numel() == 0:
                    outputs.append(torch.zeros_like(query))
                    continue
                key = k.index_select(1, key_indices)
                value = v.index_select(1, key_indices)
            else:
                if key_start is None or key_end is None:
                    raise RuntimeError(
                        "Segmented FA4 received an invalid nonempty KV plan."
                    )
                key = k[:, key_start:key_end]
                value = v[:, key_start:key_end]
            outputs.append(super().__call__(query, key, value, heads, None))
        return torch.cat(outputs, dim=1)


def configure_exact_segmented_fa4(root: torch.nn.Module) -> int:
    ''

    configured = 0
    for module in root.modules():
        attention_function = getattr(module, "attention_function", None)
        if isinstance(attention_function, FlashAttention4):
            module.attention_function = ExactSegmentedFlashAttention4()
            configured += 1
    return configured


__all__ = ["ExactSegmentedFlashAttention4", "configure_exact_segmented_fa4"]
