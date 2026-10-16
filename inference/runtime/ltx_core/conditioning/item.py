from typing import Protocol

from ltx_core.tools import LatentTools
from ltx_core.types import LatentState


class ConditioningItem(Protocol):
    ''

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        ''
        ...
