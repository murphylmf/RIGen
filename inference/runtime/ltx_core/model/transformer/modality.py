from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Modality:
    ''

    latent: (
        torch.Tensor
    )                                                                                                   
    sigma: torch.Tensor                                                                                    
    timesteps: torch.Tensor                                                    
    positions: (
        torch.Tensor
    )                                                                                                 
    context: torch.Tensor
    enabled: bool = True
    context_mask: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    self_attention_block_mask: object | None = None
    cross_attention_mask: torch.Tensor | None = None
    cross_attention_timesteps: torch.Tensor | None = None
    context_timesteps: torch.Tensor | None = None

    def split(self, sizes: list[int]) -> list[Modality]:
        ''
        n = len(sizes)
        split_fields: dict[str, list[object]] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                split_fields[f.name] = list(value.split(sizes, dim=0))
            elif value is None or isinstance(value, bool):
                split_fields[f.name] = [value] * n
            elif f.name == "self_attention_block_mask":
                                                                       
                                                                             
                                                             
                split_fields[f.name] = [value] * n
            else:
                raise TypeError(f"Cannot split field {f.name!r}: unsupported type {type(value)}")
        return [Modality(**{name: parts[i] for name, parts in split_fields.items()}) for i in range(n)]
