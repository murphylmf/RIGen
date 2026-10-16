''

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.modality import Modality


def _split_perturbations(config: BatchedPerturbationConfig, sizes: list[int]) -> list[BatchedPerturbationConfig]:
    ''
    it = iter(config.perturbations)
    return [BatchedPerturbationConfig([next(it) for _ in range(s)]) for s in sizes]


def _merge_tensors(tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
    ''
    non_none = [t for t in tensors if t is not None]
    if not non_none:
        return None
    return torch.cat(non_none, dim=0)


class BatchSplitAdapter(nn.Module):
    ''

    def __init__(self, model: nn.Module, max_batch_size: int) -> None:
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        super().__init__()
        self._model = model
        self._max_batch_size = max_batch_size

    def _get_chunk_sizes(self, batch_size: int) -> list[int]:
        full, remainder = divmod(batch_size, self._max_batch_size)
        sizes = [self._max_batch_size] * full
        if remainder:
            sizes.append(remainder)
        return sizes

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        action_joystick_tokens: torch.Tensor | None = None,
        action_joystick_condition_mask: torch.Tensor | None = None,
        action_button_state: torch.Tensor | None = None,
        action_button_values: torch.Tensor | None = None,
        action_button_condition_mask: torch.Tensor | None = None,
        action_button_meaning_embeds: torch.Tensor | None = None,
        action_button_meaning_fusion: str = "add",
        action_button_temporal_alignment: str = "global",
        action_button_prefix_tokens: int = 0,
        action_button_frame_rate: float = 20.0,
        action_button_query_latent_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        batch_size = (video or audio).latent.shape[0]

        if batch_size <= self._max_batch_size:
            return self._model(
                video=video,
                audio=audio,
                perturbations=perturbations,
                action_joystick_tokens=action_joystick_tokens,
                action_joystick_condition_mask=action_joystick_condition_mask,
                action_button_state=action_button_state,
                action_button_values=action_button_values,
                action_button_condition_mask=action_button_condition_mask,
                action_button_meaning_embeds=action_button_meaning_embeds,
                action_button_meaning_fusion=action_button_meaning_fusion,
                action_button_temporal_alignment=action_button_temporal_alignment,
                action_button_prefix_tokens=action_button_prefix_tokens,
                action_button_frame_rate=action_button_frame_rate,
                action_button_query_latent_indices=action_button_query_latent_indices,
            )

        sizes = self._get_chunk_sizes(batch_size)
        n = len(sizes)

        v_chunks = video.split(sizes) if video is not None else [None] * n
        a_chunks = audio.split(sizes) if audio is not None else [None] * n
        p_chunks = _split_perturbations(perturbations, sizes)
        joy_chunks = (
            list(action_joystick_tokens.split(sizes, dim=0)) if action_joystick_tokens is not None else [None] * n
        )
        joy_mask_chunks = (
            list(action_joystick_condition_mask.split(sizes, dim=0))
            if action_joystick_condition_mask is not None
            else [None] * n
        )
        button_state_chunks = (
            list(action_button_state.split(sizes, dim=0)) if action_button_state is not None else [None] * n
        )
        button_value_chunks = (
            list(action_button_values.split(sizes, dim=0)) if action_button_values is not None else [None] * n
        )
        button_mask_chunks = (
            list(action_button_condition_mask.split(sizes, dim=0))
            if action_button_condition_mask is not None
            else [None] * n
        )
        button_meaning_chunks = (
            list(action_button_meaning_embeds.split(sizes, dim=0))
            if action_button_meaning_embeds is not None
            else [None] * n
        )

        chunk_results = [
            self._model(
                video=vc,
                audio=ac,
                perturbations=pc,
                action_joystick_tokens=jc,
                action_joystick_condition_mask=jmc,
                action_button_state=bsc,
                action_button_values=bvc,
                action_button_condition_mask=bmaskc,
                action_button_meaning_embeds=bmc,
                action_button_meaning_fusion=action_button_meaning_fusion,
                action_button_temporal_alignment=action_button_temporal_alignment,
                action_button_prefix_tokens=action_button_prefix_tokens,
                action_button_frame_rate=action_button_frame_rate,
                action_button_query_latent_indices=action_button_query_latent_indices,
            )
            for vc, ac, pc, jc, jmc, bsc, bvc, bmaskc, bmc in zip(
                v_chunks,
                a_chunks,
                p_chunks,
                joy_chunks,
                joy_mask_chunks,
                button_state_chunks,
                button_value_chunks,
                button_mask_chunks,
                button_meaning_chunks,
                strict=True,
            )
        ]

        results_v, results_a = zip(*chunk_results, strict=True)
        return _merge_tensors(list(results_v)), _merge_tensors(list(results_a))

    def __getattr__(self, name: str) -> Any:                
        ''
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)
