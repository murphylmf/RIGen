from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

import torch

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.model_protocol import ModelType

if TYPE_CHECKING:
    from ltx_core.loader.registry import Registry


@dataclass(frozen=True)
class StateDict:
    ''

    sd: dict
    device: torch.device
    size: int
    dtype: set[torch.dtype]

    def footprint(self) -> tuple[int, torch.device]:
        return self.size, self.device


class StateDictLoader(Protocol):
    ''

    def metadata(self, path: str) -> dict:
        ''

    def load(self, path: str | list[str], sd_ops: SDOps | None = None, device: torch.device | None = None) -> StateDict:
        ''


class ModelBuilderProtocol(Protocol[ModelType]):
    ''

    model_sd_ops: SDOps | None
    module_ops: tuple[ModuleOps, ...]
    loras: tuple["LoraPathStrengthAndSDOps", ...]
    registry: "Registry"

    def meta_model(self, config: dict, module_ops: list[ModuleOps] | None = None) -> ModelType:
        ''
        ...

    def with_sd_ops(self, sd_ops: SDOps | None) -> "ModelBuilderProtocol[ModelType]":
        ''
        ...

    def with_module_ops(self, module_ops: tuple[ModuleOps, ...]) -> "ModelBuilderProtocol[ModelType]":
        ''
        ...

    def with_loras(self, loras: tuple["LoraPathStrengthAndSDOps", ...]) -> "ModelBuilderProtocol[ModelType]":
        ''
        ...

    def with_registry(self, registry: "Registry") -> "ModelBuilderProtocol[ModelType]":
        ''
        ...

    def with_lora_load_device(self, device: torch.device) -> "ModelBuilderProtocol[ModelType]":
        ''
        ...

    def build(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs: object
    ) -> ModelType:
        ''
        ...

    def model_config(self) -> dict:
        ''
        ...


class LoRAAdaptableProtocol(Protocol):
    ''

    def lora(self, lora_path: str, strength: float) -> "LoRAAdaptableProtocol":
        pass


class LoraPathStrengthAndSDOps(NamedTuple):
    ''

    path: str
    strength: float
    sd_ops: SDOps


class LoraStateDictWithStrength(NamedTuple):
    ''

    state_dict: StateDict
    strength: float
