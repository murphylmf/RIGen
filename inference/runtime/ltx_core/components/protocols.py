from typing import Protocol, Tuple

import torch

from ltx_core.types import AudioLatentShape, VideoLatentShape


class Patchifier(Protocol):
    ''

    def patchify(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        ...
        """
        Convert latent tensors into flattened patch tokens.
        Args:
            latents: Latent tensor to patchify.
        Returns:
            Flattened patch tokens tensor.
        """

    def unpatchify(
        self,
        latents: torch.Tensor,
        output_shape: AudioLatentShape | VideoLatentShape,
    ) -> torch.Tensor:
        ''

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        ...
        """
        Returns the patch size as a tuple of (temporal, height, width) dimensions
        """

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape | VideoLatentShape,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        ...
        """
        Compute metadata describing where each latent patch resides within the
        grid specified by `output_shape`.
        Args:
            output_shape: Target grid layout for the patches.
            device: Target device for the returned tensor.
        Returns:
            Tensor containing patch coordinate metadata such as spatial or temporal intervals.
        """


class SchedulerProtocol(Protocol):
    ''

    def execute(self, steps: int, **kwargs) -> torch.FloatTensor: ...


class GuiderProtocol(Protocol):
    ''

    scale: float

    def delta(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor: ...

    def enabled(self) -> bool:
        ''
        ...


class DiffusionStepProtocol(Protocol):
    ''

    def step(
        self, sample: torch.Tensor, denoised_sample: torch.Tensor, sigmas: torch.Tensor, step_index: int, **kwargs
    ) -> torch.Tensor: ...
