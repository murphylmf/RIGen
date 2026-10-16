import logging
from dataclasses import replace
from functools import partial
from typing import Callable

import torch
from tqdm import tqdm

from ltx_core.components.diffusion_steps import Res2sDiffusionStep
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.model.transformer import X0Model
from ltx_core.utils import to_denoised, to_velocity
from ltx_pipelines.utils.helpers import post_process_latent, timesteps_from_mask
from ltx_pipelines.utils.res2s import get_res2s_coefficients
from ltx_pipelines.utils.types import Denoiser, LatentState

logger = logging.getLogger(__name__)


def _step_state(
    state: LatentState | None,
    denoised: torch.Tensor | None,
    stepper: DiffusionStepProtocol,
    sigmas: torch.Tensor,
    step_idx: int,
) -> LatentState | None:
    ''
    if state is None or denoised is None:
        return state
    denoised = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
    return replace(state, latent=stepper.step(state.latent, denoised, sigmas, step_idx))


def euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    stepper: DiffusionStepProtocol,
    transformer: X0Model,
    denoiser: Denoiser,
) -> tuple[LatentState | None, LatentState | None]:
    ''
    for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
        denoised_video, denoised_audio = denoiser(transformer, video_state, audio_state, sigmas, step_idx)

        video_state = _step_state(video_state, denoised_video, stepper, sigmas, step_idx)
        audio_state = _step_state(audio_state, denoised_audio, stepper, sigmas, step_idx)

    return (video_state, audio_state)


def gradient_estimating_euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    stepper: DiffusionStepProtocol,
    transformer: X0Model,
    denoiser: Denoiser,
    ge_gamma: float = 2.0,
) -> tuple[LatentState | None, LatentState | None]:
    ''

    previous_audio_velocity = None
    previous_video_velocity = None

    def update_velocity_and_sample(
        noisy_sample: torch.Tensor, denoised_sample: torch.Tensor, sigma: float, previous_velocity: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_velocity = to_velocity(noisy_sample, sigma, denoised_sample)
        if previous_velocity is not None:
            delta_v = current_velocity - previous_velocity
            total_velocity = ge_gamma * delta_v + previous_velocity
            denoised_sample = to_denoised(noisy_sample, total_velocity, sigma)
        return current_velocity, denoised_sample

    for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
        denoised_video, denoised_audio = denoiser(transformer, video_state, audio_state, sigmas, step_idx)

        if video_state is not None and denoised_video is not None:
            denoised_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
        if audio_state is not None and denoised_audio is not None:
            denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)

        if sigmas[step_idx + 1] == 0:
            if video_state is not None and denoised_video is not None:
                video_state = replace(video_state, latent=denoised_video)
            if audio_state is not None and denoised_audio is not None:
                audio_state = replace(audio_state, latent=denoised_audio)
            return video_state, audio_state

        if video_state is not None and denoised_video is not None:
            previous_video_velocity, denoised_video = update_velocity_and_sample(
                video_state.latent, denoised_video, sigmas[step_idx], previous_video_velocity
            )
            video_state = replace(
                video_state, latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx)
            )

        if audio_state is not None and denoised_audio is not None:
            previous_audio_velocity, denoised_audio = update_velocity_and_sample(
                audio_state.latent, denoised_audio, sigmas[step_idx], previous_audio_velocity
            )
            audio_state = replace(
                audio_state, latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx)
            )

    return (video_state, audio_state)


def _channelwise_normalize(x: torch.Tensor) -> torch.Tensor:
    return x.sub_(x.mean(dim=(-2, -1), keepdim=True)).div_(x.std(dim=(-2, -1), keepdim=True))


def _get_new_noise(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(x.shape, generator=generator, dtype=torch.float64, device=generator.device)
    noise = (noise - noise.mean()) / noise.std()
    return _channelwise_normalize(noise)


def _inject_sde_noise(
    state: LatentState,
    sample: torch.Tensor,
    denoised_sample: torch.Tensor,
    step_noise_generator: torch.Generator,
    new_noise_fn: Callable[[torch.Tensor, torch.Generator], torch.Tensor],
    stepper: DiffusionStepProtocol,
    sigmas: torch.Tensor,
    step_idx: int,
    legacy_mode: bool = False,
    eta: float = 0.5,
) -> torch.Tensor:
    sigmas_copy = sigmas.clone()
    new_noise = new_noise_fn(state.latent, step_noise_generator)
    if not legacy_mode:
        timesteps = timesteps_from_mask(state.denoise_mask.double(), sigmas_copy[step_idx].double())
        next_timesteps = timesteps_from_mask(state.denoise_mask.double(), sigmas_copy[step_idx + 1].double())
        sigmas = torch.stack([timesteps, next_timesteps])
        step_idx = 0
    x_next = stepper.step(
        sample=sample,
        denoised_sample=denoised_sample,
        sigmas=sigmas,
        step_index=step_idx,
        noise=new_noise,
        eta=eta,
    )

    if legacy_mode:
        x_next = post_process_latent(x_next, state.denoise_mask, state.clean_latent)

    return x_next


def res2s_audio_video_denoising_loop(                                 
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    stepper: DiffusionStepProtocol,
    transformer: X0Model,
    denoiser: Denoiser,
    noise_seed: int = -1,
    noise_seed_substep: int | None = None,
    eta: float = 0.5,
    bongmath: bool = True,
    bongmath_max_iter: int = 100,
    new_noise_fn: Callable[[torch.Tensor, torch.Generator], torch.Tensor] = _get_new_noise,
    model_dtype: torch.dtype = torch.bfloat16,
    legacy_mode: bool = True,
) -> tuple[LatentState | None, LatentState | None]:
    ''
                                                      
    present_state = video_state or audio_state
    if present_state is None:
        raise ValueError("At least one of video_state or audio_state must be provided")
    state_device = present_state.latent.device

                                                      
    if noise_seed_substep is None:
        noise_seed_substep = noise_seed + 10000                                    
    step_noise_generator = torch.Generator(device=state_device).manual_seed(noise_seed)
    substep_noise_generator = torch.Generator(device=state_device).manual_seed(noise_seed_substep)
    sde_noise_injecting_fn = partial(
        _inject_sde_noise, stepper=stepper, new_noise_fn=new_noise_fn, legacy_mode=legacy_mode
    )
    step_noise_injecting_fn = partial(sde_noise_injecting_fn, step_noise_generator=step_noise_generator, eta=eta)
                                                                                       
    substep_noise_injecting_fn = partial(sde_noise_injecting_fn, step_noise_generator=substep_noise_generator, eta=0.5)

    if not isinstance(stepper, Res2sDiffusionStep):
        raise ValueError("stepper must be an instance of Res2sDiffusionStep")

    n_full_steps = len(sigmas) - 1
                                                          
    if sigmas[-1] == 0:
        sigmas = torch.cat([sigmas[:-1], torch.tensor([0.0011, 0.0], device=sigmas.device)], dim=0)
                                            
    hs = -torch.log(sigmas[1:].double().cpu() / (sigmas[:-1].double().cpu()))

                                                           
    phi_cache = {}
    c2 = 0.5                       

    for step_idx in tqdm(range(n_full_steps)):
        sigma = sigmas[step_idx].double()
        sigma_next = sigmas[step_idx + 1].double()

                                 
        x_anchor_video = video_state.latent.clone().double() if video_state is not None else None
        x_anchor_audio = audio_state.latent.clone().double() if audio_state is not None else None

                                                                              
                                            
                                                                              
        denoised_video_1, denoised_audio_1 = denoiser(transformer, video_state, audio_state, sigmas, step_idx)
        if video_state is not None and denoised_video_1 is not None:
            denoised_video_1 = post_process_latent(denoised_video_1, video_state.denoise_mask, video_state.clean_latent)
        if audio_state is not None and denoised_audio_1 is not None:
            denoised_audio_1 = post_process_latent(denoised_audio_1, audio_state.denoise_mask, audio_state.clean_latent)

        h = hs[step_idx].item()

                                                              
        a21, b1, b2 = get_res2s_coefficients(h, phi_cache, c2)

                                                                
        sub_sigma = torch.sqrt(sigma * sigma_next)

                                                                              
                                                    
                                                                              
        if x_anchor_video is not None and denoised_video_1 is not None:
            eps_1_video = denoised_video_1.double() - x_anchor_video
            x_mid_video = x_anchor_video.double() + h * a21 * eps_1_video
        else:
            eps_1_video = None
            x_mid_video = None

        if x_anchor_audio is not None and denoised_audio_1 is not None:
            eps_1_audio = denoised_audio_1.double() - x_anchor_audio
            x_mid_audio = x_anchor_audio.double() + h * a21 * eps_1_audio
        else:
            eps_1_audio = None
            x_mid_audio = None

                                                                              
                                        
                                                                              
        if x_mid_video is not None and video_state is not None:
            x_mid_video = substep_noise_injecting_fn(
                state=video_state,
                sample=x_anchor_video,
                denoised_sample=x_mid_video,
                sigmas=torch.stack([sigma, sub_sigma]),
                step_idx=0,
            )
        if x_mid_audio is not None and audio_state is not None:
            x_mid_audio = substep_noise_injecting_fn(
                state=audio_state,
                sample=x_anchor_audio,
                denoised_sample=x_mid_audio,
                sigmas=torch.stack([sigma, sub_sigma]),
                step_idx=0,
            )

                                                                              
                                               
                                                                              
        if bongmath and h < 0.5 and sigma > 0.03:
            for _ in range(bongmath_max_iter):
                if x_mid_video is not None and eps_1_video is not None:
                    x_anchor_video = x_mid_video - h * a21 * eps_1_video
                    eps_1_video = denoised_video_1.double() - x_anchor_video
                if x_mid_audio is not None and eps_1_audio is not None:
                    x_anchor_audio = x_mid_audio - h * a21 * eps_1_audio
                    eps_1_audio = denoised_audio_1.double() - x_anchor_audio

                                                                              
                                                         
                                                                              
        mid_video_state = (
            replace(video_state, latent=x_mid_video.to(model_dtype))
            if video_state is not None and x_mid_video is not None
            else None
        )
        mid_audio_state = (
            replace(audio_state, latent=x_mid_audio.to(model_dtype))
            if audio_state is not None and x_mid_audio is not None
            else None
        )

        denoised_video_2, denoised_audio_2 = denoiser(
            transformer,
            video_state=mid_video_state,
            audio_state=mid_audio_state,
            sigmas=torch.stack([sub_sigma]).to(sigmas.device),
            step_index=0,
        )
        if video_state is not None and denoised_video_2 is not None:
            denoised_video_2 = post_process_latent(denoised_video_2, video_state.denoise_mask, video_state.clean_latent)
        if audio_state is not None and denoised_audio_2 is not None:
            denoised_audio_2 = post_process_latent(denoised_audio_2, audio_state.denoise_mask, audio_state.clean_latent)

                                                                              
                                                                 
                                                                              
        if x_anchor_video is not None and eps_1_video is not None and denoised_video_2 is not None:
            eps_2_video = denoised_video_2.double() - x_anchor_video
            x_next_video = x_anchor_video + h * (b1 * eps_1_video + b2 * eps_2_video)
        else:
            x_next_video = None

        if x_anchor_audio is not None and eps_1_audio is not None and denoised_audio_2 is not None:
            eps_2_audio = denoised_audio_2.double() - x_anchor_audio
            x_next_audio = x_anchor_audio + h * (b1 * eps_1_audio + b2 * eps_2_audio)
        else:
            x_next_audio = None

                                                                              
                                           
                                                                              
        if x_next_video is not None and video_state is not None:
            x_next_video = step_noise_injecting_fn(
                state=video_state,
                sample=x_anchor_video,
                denoised_sample=x_next_video,
                sigmas=sigmas,
                step_idx=step_idx,
            )
        if x_next_audio is not None and audio_state is not None:
            x_next_audio = step_noise_injecting_fn(
                state=audio_state,
                sample=x_anchor_audio,
                denoised_sample=x_next_audio,
                sigmas=sigmas,
                step_idx=step_idx,
            )

                       
        if video_state is not None and x_next_video is not None:
            video_state = replace(video_state, latent=x_next_video.to(model_dtype))
        if audio_state is not None and x_next_audio is not None:
            audio_state = replace(audio_state, latent=x_next_audio.to(model_dtype))

                                                     
    if sigmas[-1] == 0:
        denoised_video_1, denoised_audio_1 = denoiser(transformer, video_state, audio_state, sigmas, n_full_steps)
        if video_state is not None and denoised_video_1 is not None:
            denoised_video_1 = post_process_latent(denoised_video_1, video_state.denoise_mask, video_state.clean_latent)
            video_state = replace(video_state, latent=denoised_video_1.to(model_dtype))
        if audio_state is not None and denoised_audio_1 is not None:
            denoised_audio_1 = post_process_latent(denoised_audio_1, audio_state.denoise_mask, audio_state.clean_latent)
            audio_state = replace(audio_state, latent=denoised_audio_1.to(model_dtype))

    return video_state, audio_state
