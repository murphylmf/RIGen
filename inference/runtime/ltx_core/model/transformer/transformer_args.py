from dataclasses import dataclass, replace

import torch

from ltx_core.model.transformer.adaln import AdaLayerNormSingle
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import (
    LTXRopeType,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)


@dataclass(frozen=True)
class TransformerArgs:
    x: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    timesteps: torch.Tensor
    embedded_timestep: torch.Tensor
    positional_embeddings: torch.Tensor
    cross_positional_embeddings: torch.Tensor | None
    cross_scale_shift_timestep: torch.Tensor | None
    cross_gate_timestep: torch.Tensor | None
    enabled: bool
    prompt_timestep: torch.Tensor | None = None
    action_joystick_tokens: torch.Tensor | None = None
    action_joystick_condition_mask: torch.Tensor | None = None
    action_button_state: torch.Tensor | None = None
    action_button_values: torch.Tensor | None = None
    action_button_condition_mask: torch.Tensor | None = None
    action_button_meaning_embeds: torch.Tensor | None = None
    action_button_meaning_fusion: str = "add"
    action_button_temporal_alignment: str = "global"
    action_button_prefix_tokens: int = 0
    action_button_causal_attention_mask: torch.Tensor | None = None
    action_button_causal_query_cos: torch.Tensor | None = None
    action_button_causal_query_sin: torch.Tensor | None = None
    action_button_causal_key_cos: torch.Tensor | None = None
    action_button_causal_key_sin: torch.Tensor | None = None
    self_attention_mask: torch.Tensor | object | None = (
        None                                                                              
    )
    cross_attention_mask: torch.Tensor | None = None


class TransformerArgsPreprocessor:
    def __init__(                 
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        caption_projection: torch.nn.Module | None = None,
        prompt_adaln: AdaLayerNormSingle | None = None,
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.double_precision_rope = double_precision_rope
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        self.caption_projection = caption_projection
        self.prompt_adaln = prompt_adaln

    def _prepare_timestep(
        self, timestep: torch.Tensor, adaln: AdaLayerNormSingle, batch_size: int, hidden_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ''
        timestep_scaled = timestep * self.timestep_scale_multiplier
        timestep, embedded_timestep = adaln(
            timestep_scaled.flatten(),
            hidden_dtype=hidden_dtype,
        )
                                                                           
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])

        return timestep, embedded_timestep

    def _prepare_context(
        self,
        context: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        ''
        if self.caption_projection is not None:
            context = self.caption_projection(context)
        batch_size = x.shape[0]
        return context.view(batch_size, -1, x.shape[-1])

    def _prepare_attention_mask(self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype) -> torch.Tensor | None:
        ''
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask

        if attention_mask.shape[-1] == 0:
            raise ValueError("Cross-attention mask must contain at least one key token.")
        if torch.all(attention_mask == attention_mask[..., :1]):
            return None

        return (attention_mask.to(x_dtype) - 1).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max

    def _prepare_self_attention_mask(
        self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype
    ) -> torch.Tensor | None:
        ''
        if attention_mask is None:
            return None

                                                                   
                                                            
                                                  
        finfo = torch.finfo(x_dtype)
        eps = finfo.tiny

        bias = torch.full_like(attention_mask, finfo.min, dtype=x_dtype)
        positive = attention_mask > 0
        if positive.any():
            bias[positive] = torch.log(attention_mask[positive].clamp(min=eps)).to(x_dtype)

        return bias.unsqueeze(1)                                   

    def _prepare_positional_embeddings(
        self,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
    ) -> torch.Tensor:
        ''
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        pe = precompute_freqs_cis(
            positions,
            dim=inner_dim,
            out_dtype=x_dtype,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )
        return pe

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,                
        action_joystick_tokens: torch.Tensor | None = None,
        action_joystick_condition_mask: torch.Tensor | None = None,
        action_button_state: torch.Tensor | None = None,
        action_button_values: torch.Tensor | None = None,
        action_button_condition_mask: torch.Tensor | None = None,
        action_button_meaning_embeds: torch.Tensor | None = None,
        action_button_meaning_fusion: str = "add",
        action_button_temporal_alignment: str = "global",
        action_button_prefix_tokens: int = 0,
    ) -> TransformerArgs:
        x = self.patchify_proj(modality.latent)
        batch_size = x.shape[0]
        timestep, embedded_timestep = self._prepare_timestep(
            modality.timesteps, self.adaln, batch_size, modality.latent.dtype
        )
        prompt_timestep = None
        if self.prompt_adaln is not None:
            context_timesteps = (
                modality.context_timesteps if modality.context_timesteps is not None else modality.sigma
            )
            prompt_timestep, _ = self._prepare_timestep(
                context_timesteps, self.prompt_adaln, batch_size, modality.latent.dtype
            )
        context = self._prepare_context(modality.context, x)
        attention_mask = self._prepare_attention_mask(modality.context_mask, modality.latent.dtype)
        pe = self._prepare_positional_embeddings(
            positions=modality.positions,
            inner_dim=self.inner_dim,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )
        self_attention_mask = (
            modality.self_attention_block_mask
            if modality.self_attention_block_mask is not None
            else self._prepare_self_attention_mask(modality.attention_mask, modality.latent.dtype)
        )
        return TransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
            prompt_timestep=prompt_timestep,
            action_joystick_tokens=action_joystick_tokens,
            action_joystick_condition_mask=action_joystick_condition_mask,
            action_button_state=action_button_state,
            action_button_values=action_button_values,
            action_button_condition_mask=action_button_condition_mask,
            action_button_meaning_embeds=action_button_meaning_embeds,
            action_button_meaning_fusion=action_button_meaning_fusion,
            action_button_temporal_alignment=action_button_temporal_alignment,
            action_button_prefix_tokens=action_button_prefix_tokens,
            self_attention_mask=self_attention_mask,
        )


class MultiModalTransformerArgsPreprocessor:
    def __init__(                 
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        av_ca_timestep_scale_multiplier: int,
        caption_projection: torch.nn.Module | None = None,
        prompt_adaln: AdaLayerNormSingle | None = None,
    ) -> None:
        self.simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=patchify_proj,
            adaln=adaln,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            caption_projection=caption_projection,
            prompt_adaln=prompt_adaln,
        )
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,
        action_joystick_tokens: torch.Tensor | None = None,
        action_joystick_condition_mask: torch.Tensor | None = None,
        action_button_state: torch.Tensor | None = None,
        action_button_values: torch.Tensor | None = None,
        action_button_condition_mask: torch.Tensor | None = None,
        action_button_meaning_embeds: torch.Tensor | None = None,
        action_button_meaning_fusion: str = "add",
        action_button_temporal_alignment: str = "global",
        action_button_prefix_tokens: int = 0,
    ) -> TransformerArgs:
        transformer_args = self.simple_preprocessor.prepare(
            modality,
            action_joystick_tokens=action_joystick_tokens,
            action_joystick_condition_mask=action_joystick_condition_mask,
            action_button_state=action_button_state,
            action_button_values=action_button_values,
            action_button_condition_mask=action_button_condition_mask,
            action_button_meaning_embeds=action_button_meaning_embeds,
            action_button_meaning_fusion=action_button_meaning_fusion,
            action_button_temporal_alignment=action_button_temporal_alignment,
            action_button_prefix_tokens=action_button_prefix_tokens,
        )
        if cross_modality is None:
            return transformer_args

        if modality.cross_attention_timesteps is not None:
            cross_timestep = modality.cross_attention_timesteps
            if cross_timestep.shape != modality.timesteps.shape:
                raise ValueError(
                    "Per-token cross-attention timesteps must match modality timesteps: "
                    f"{tuple(cross_timestep.shape)} vs {tuple(modality.timesteps.shape)}"
                )
        else:
            if cross_modality.sigma.numel() > 1:
                if cross_modality.sigma.shape[0] != modality.timesteps.shape[0]:
                    raise ValueError("Cross modality sigma must have the same batch size as the modality")
                if cross_modality.sigma.ndim != 1:
                    raise ValueError("Cross modality sigma must be a 1D tensor")

            cross_timestep = cross_modality.sigma.view(
                modality.timesteps.shape[0], 1, *[1] * len(modality.timesteps.shape[2:])
            )

        cross_pe = self.simple_preprocessor._prepare_positional_embeddings(
            positions=modality.positions[:, 0:1, :],
            inner_dim=self.audio_cross_attention_dim,
            max_pos=[self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )

        cross_scale_shift_timestep, cross_gate_timestep = self._prepare_cross_attention_timestep(
            timestep=cross_timestep,
            timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
            batch_size=transformer_args.x.shape[0],
            hidden_dtype=modality.latent.dtype,
        )

        return replace(
            transformer_args,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
            cross_attention_mask=self.simple_preprocessor._prepare_attention_mask(
                modality.cross_attention_mask,
                modality.latent.dtype,
            ),
        )

    def _prepare_cross_attention_timestep(
        self,
        timestep: torch.Tensor | None,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ''
        timestep = timestep * timestep_scale_multiplier

        av_ca_factor = self.av_ca_timestep_scale_multiplier / timestep_scale_multiplier

        scale_shift_timestep, _ = self.cross_scale_shift_adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )
        scale_shift_timestep = scale_shift_timestep.view(batch_size, -1, scale_shift_timestep.shape[-1])
        gate_noise_timestep, _ = self.cross_gate_adaln(
            timestep.flatten() * av_ca_factor,
            hidden_dtype=hidden_dtype,
        )
        gate_noise_timestep = gate_noise_timestep.view(batch_size, -1, gate_noise_timestep.shape[-1])

        return scale_shift_timestep, gate_noise_timestep
