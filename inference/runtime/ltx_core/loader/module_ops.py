from typing import Callable, NamedTuple

import torch


class ModuleOps(NamedTuple):
    ''

    name: str
    matcher: Callable[[torch.nn.Module], bool]
    mutator: Callable[[torch.nn.Module], torch.nn.Module]
