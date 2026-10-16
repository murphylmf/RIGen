import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

from ltx_core.components.protocols import GuiderProtocol


@dataclass(frozen=True)
class CFGGuider(GuiderProtocol):
    ''

    scale: float

    def delta(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        return (self.scale - 1) * (cond - uncond)

    def enabled(self) -> bool:
        return self.scale != 1.0


@dataclass(frozen=True)
class CFGStarRescalingGuider(GuiderProtocol):
    ''

    scale: float

    def delta(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        rescaled_neg = projection_coef(cond, uncond) * uncond
        return (self.scale - 1) * (cond - rescaled_neg)

    def enabled(self) -> bool:
        return self.scale != 1.0


@dataclass(frozen=True)
class STGGuider(GuiderProtocol):
    ''

    scale: float

    def delta(self, pos_denoised: torch.Tensor, perturbed_denoised: torch.Tensor) -> torch.Tensor:
        return self.scale * (pos_denoised - perturbed_denoised)

    def enabled(self) -> bool:
        return self.scale != 0.0


@dataclass(frozen=True)
class LtxAPGGuider(GuiderProtocol):
    ''

    scale: float
    eta: float = 1.0
    norm_threshold: float = 0.0

    def delta(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        guidance = cond - uncond
        if self.norm_threshold > 0:
            ones = torch.ones_like(guidance)
            guidance_norm = guidance.norm(p=2, dim=[-1, -2, -3], keepdim=True)
            scale_factor = torch.minimum(ones, self.norm_threshold / guidance_norm)
            guidance = guidance * scale_factor
        proj_coeff = projection_coef(guidance, cond)
        g_parallel = proj_coeff * cond
        g_orth = guidance - g_parallel
        g_apg = g_parallel * self.eta + g_orth

        return g_apg * (self.scale - 1)

    def enabled(self) -> bool:
        return self.scale != 1.0


@dataclass(frozen=False)
class LegacyStatefulAPGGuider(GuiderProtocol):
    ''

    scale: float
    eta: float
    norm_threshold: float = 5.0
    momentum: float = 0.0
                                                                                                          
                                                                                         
    running_avg: torch.Tensor | None = None

    def delta(self, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        guidance = cond - uncond
        if self.momentum != 0:
            if self.running_avg is None:
                self.running_avg = guidance.clone()
            else:
                self.running_avg = self.momentum * self.running_avg + guidance
            guidance = self.running_avg

        if self.norm_threshold > 0:
            ones = torch.ones_like(guidance)
            guidance_norm = guidance.norm(p=2, dim=[-1, -2, -3], keepdim=True)
            scale_factor = torch.minimum(ones, self.norm_threshold / guidance_norm)
            guidance = guidance * scale_factor

        proj_coeff = projection_coef(guidance, cond)
        g_parallel = proj_coeff * cond
        g_orth = guidance - g_parallel
        g_apg = g_parallel * self.eta + g_orth

        return g_apg * self.scale

    def enabled(self) -> bool:
        return self.scale != 0.0


@dataclass(frozen=True)
class MultiModalGuiderParams:
    ''

    cfg_scale: float = 1.0
    "CFG (Classifier-free guidance) scale controlling how strongly the model adheres to the prompt."
    stg_scale: float = 0.0
    "STG (Spatio-Temporal Guidance) scale controls how strongly the model reacts to the perturbation of the modality."
    stg_blocks: list[int] | None = field(default_factory=list)
    "Which transformer blocks to perturb for STG."
    rescale_scale: float = 0.0
    "Rescale scale controlling how strongly the model rescales the modality after applying other guidance."
    modality_scale: float = 1.0
    "Modality scale controlling how strongly the model reacts to the perturbation of the modality."
    skip_step: int = 0
    "Skip step controlling how often the model skips the step."


def _params_for_sigma_from_sorted_dict(
    sigma: float, params_by_sigma: Sequence[tuple[float, MultiModalGuiderParams]]
) -> MultiModalGuiderParams:
    ''
    if not params_by_sigma:
        raise ValueError("params_by_sigma must be non-empty")
    sigma = float(sigma)
    keys_desc = [k for k, _ in params_by_sigma]
    keys_ge_sigma = [k for k in keys_desc if k >= sigma]
                                                   
    key = keys_ge_sigma[-1] if keys_ge_sigma else keys_desc[0]
    return next(p for k, p in params_by_sigma if k == key)


@dataclass(frozen=True)
class MultiModalGuider:
    ''

    params: MultiModalGuiderParams
    negative_context: torch.Tensor | None = None

    def calculate(
        self,
        cond: torch.Tensor,
        uncond_text: torch.Tensor | float,
        uncond_perturbed: torch.Tensor | float,
        uncond_modality: torch.Tensor | float,
    ) -> torch.Tensor:
        ''
        pred = (
            cond
            + (self.params.cfg_scale - 1) * (cond - uncond_text)
            + self.params.stg_scale * (cond - uncond_perturbed)
            + (self.params.modality_scale - 1) * (cond - uncond_modality)
        )

        if self.params.rescale_scale != 0:
            factor = cond.std() / pred.std()
            factor = self.params.rescale_scale * factor + (1 - self.params.rescale_scale)
            pred = pred * factor

        return pred

    def do_unconditional_generation(self) -> bool:
        ''
        return not math.isclose(self.params.cfg_scale, 1.0)

    def do_perturbed_generation(self) -> bool:
        ''
        return not math.isclose(self.params.stg_scale, 0.0)

    def do_isolated_modality_generation(self) -> bool:
        ''
        return not math.isclose(self.params.modality_scale, 1.0)

    def should_skip_step(self, step: int) -> bool:
        ''
        if self.params.skip_step == 0:
            return False
        return step % (self.params.skip_step + 1) != 0


@dataclass(frozen=True)
class MultiModalGuiderFactory:
    ''

    negative_context: torch.Tensor | None = None
    _params_by_sigma: tuple[tuple[float, MultiModalGuiderParams], ...] = ()

    @classmethod
    def constant(
        cls,
        params: MultiModalGuiderParams,
        negative_context: torch.Tensor | None = None,
    ) -> "MultiModalGuiderFactory":
        ''
        return cls(
            negative_context=negative_context,
            _params_by_sigma=((float("inf"), params),),
        )

    @classmethod
    def from_dict(
        cls,
        sigma_to_params: Mapping[float, MultiModalGuiderParams],
        negative_context: torch.Tensor | None = None,
    ) -> "MultiModalGuiderFactory":
        ''
        if not sigma_to_params:
            raise ValueError("sigma_to_params must be non-empty")
        sorted_items = tuple(sorted(sigma_to_params.items(), key=lambda x: x[0], reverse=True))
        return cls(negative_context=negative_context, _params_by_sigma=sorted_items)

    def params(self, sigma: float | torch.Tensor) -> MultiModalGuiderParams:
        ''
        sigma_val = float(sigma.item() if isinstance(sigma, torch.Tensor) else sigma)
        return _params_for_sigma_from_sorted_dict(sigma_val, self._params_by_sigma)

    def build_from_sigma(self, sigma: float | torch.Tensor) -> MultiModalGuider:
        ''
        return MultiModalGuider(
            params=self.params(sigma),
            negative_context=self.negative_context,
        )


def create_multimodal_guider_factory(
    params: MultiModalGuiderParams | MultiModalGuiderFactory,
    negative_context: torch.Tensor | None = None,
) -> MultiModalGuiderFactory:
    ''
    if isinstance(params, MultiModalGuiderFactory):
        if negative_context is not None and params.negative_context is not negative_context:
            return MultiModalGuiderFactory.from_dict(dict(params._params_by_sigma), negative_context=negative_context)
        return params
    return MultiModalGuiderFactory.constant(params, negative_context=negative_context)


def projection_coef(to_project: torch.Tensor, project_onto: torch.Tensor) -> torch.Tensor:
    batch_size = to_project.shape[0]
    positive_flat = to_project.reshape(batch_size, -1)
    negative_flat = project_onto.reshape(batch_size, -1)
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    return dot_product / squared_norm
