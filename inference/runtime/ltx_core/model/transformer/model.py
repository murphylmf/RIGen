from collections.abc import Callable
from dataclasses import replace
from enum import Enum

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.adaln import AdaLayerNormSingle, adaln_embedding_coefficient
from ltx_core.model.transformer.action import ActionButtonCrossAttention, prepare_causal_prefix_attention_metadata
from ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction
from ltx_core.model.transformer.flex_attention import is_block_mask
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from ltx_core.utils import to_denoised


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(torch.nn.Module):
    ''

    def __init__(                 
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        attention_type: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        caption_projection: torch.nn.Module | None = None,
        audio_caption_projection: torch.nn.Module | None = None,
        cross_attention_adaln: bool = False,
        action_button_dim: int = 8,
    ):
        super().__init__()
        self._enable_gradient_checkpointing = False
        self.cross_attention_adaln = cross_attention_adaln
        self.action_button_dim = int(action_button_dim)
        self.action_button_state_dim = self.action_button_dim + 1
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_eps=norm_eps,
                caption_projection=caption_projection,
                cross_attention_dim=cross_attention_dim,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                norm_eps=norm_eps,
                caption_projection=audio_caption_projection,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
                                       
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            attention_type=attention_type,
            apply_gated_attention=apply_gated_attention,
        )

    @property
    def _adaln_embedding_coefficient(self) -> int:
        return adaln_embedding_coefficient(self.cross_attention_adaln)

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
        cross_attention_dim: int = 4096,
    ) -> None:
        ''
                                
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        self.action_joystick_proj = torch.nn.Linear(32, self.inner_dim, bias=True)
        self.action_button_meaning_proj = torch.nn.Linear(cross_attention_dim, self.inner_dim, bias=True)
        if caption_projection is not None:
            self.caption_projection = caption_projection

        self.adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=self._adaln_embedding_coefficient)

        self.prompt_adaln_single = (
            AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

                                 
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        ''

                                
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        if caption_projection is not None:
            self.audio_caption_projection = caption_projection

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=self._adaln_embedding_coefficient,
        )

        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

                                 
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        ''
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        ''

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        attention_type: AttentionFunction | AttentionCallable,
        apply_gated_attention: bool,
    ) -> None:
        ''
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
                action_button_dim=self.action_button_dim,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_function=attention_type,
                )
                for idx in range(num_layers)
            ]
        )

    def set_action_button_dim(self, button_dim: int, state_dim: int | None = None) -> None:
        button_dim = int(button_dim)
        state_dim = button_dim + 1 if state_dim is None else int(state_dim)
        if button_dim <= 0:
            raise ValueError(f"action button dim must be positive, got {button_dim}")
        if state_dim <= button_dim:
            raise ValueError(f"action button state dim must exceed button dim, got {state_dim} <= {button_dim}")
        if (
            button_dim == int(getattr(self, "action_button_dim", 8))
            and state_dim == int(getattr(self, "action_button_state_dim", button_dim + 1))
        ):
            return
        if not self.model_type.is_video_enabled():
            raise ValueError("Cannot set action button dim on a model without video blocks")

        for block in self.transformer_blocks:
            old_attn = getattr(block, "action_button_attn", None)
            if old_attn is None:
                continue
            weight = old_attn.to_q.weight
            new_attn = ActionButtonCrossAttention(
                self.inner_dim,
                old_attn.heads,
                old_attn.dim_head,
                state_dim=state_dim,
                button_dim=button_dim,
                norm_eps=block.norm_eps,
                attention_function=old_attn.attention_function,
            )
            new_attn = new_attn.to(device=weight.device, dtype=weight.dtype)
            block.action_button_attn = new_attn
        self.action_button_dim = button_dim
        self.action_button_state_dim = state_dim

    def set_gradient_checkpointing(self, enable: bool) -> None:
        ''
        self._enable_gradient_checkpointing = enable

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[TransformerArgs, TransformerArgs]:
        ''

                                    
        for block in self.transformer_blocks:
            if self._enable_gradient_checkpointing and self.training:
                                                                            
                                                                              
                                                                                
                checkpoint_kwargs = {"use_reentrant": False}
                if (
                    (video is not None and is_block_mask(video.self_attention_mask))
                    or (audio is not None and is_block_mask(audio.self_attention_mask))
                ):
                                                                         
                                                                        
                    checkpoint_kwargs["determinism_check"] = "none"
                video, audio = torch.utils.checkpoint.checkpoint(
                    block,
                    video,
                    audio,
                    perturbations,
                    **checkpoint_kwargs,
                )
            else:
                video, audio = block(
                    video=video,
                    audio=audio,
                    perturbations=perturbations,
                )

        return video, audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        ''
                                      
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig | None,
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
        forward_override: Callable[..., tuple[torch.Tensor | None, torch.Tensor | None]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ''
                                                                        
                                                                            
                                                                            
                                  
        if forward_override is not None:
            return forward_override(
                self,
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

        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        prepared_action_joystick = None
        if action_joystick_tokens is not None:
            if video is None:
                raise ValueError("Action joystick tokens require a video modality")
            action_joystick_tokens = action_joystick_tokens.to(device=video.latent.device, dtype=video.latent.dtype)
            if action_joystick_tokens.ndim != 3 or action_joystick_tokens.shape[-1] != 32:
                raise ValueError(
                    "Action joystick tokens must have shape [B, video_tokens, 32], "
                    f"got {tuple(action_joystick_tokens.shape)}"
                )
            prepared_action_joystick = self.action_joystick_proj(action_joystick_tokens)

        prepared_action_joystick_condition_mask = self._prepare_action_condition_mask(
            action_joystick_condition_mask,
            condition=prepared_action_joystick,
            video=video,
            name="action_joystick_condition_mask",
        )

        prepared_action_button_state = None
        if action_button_state is not None:
            if video is None:
                raise ValueError("Action button state requires a video modality")
            action_button_state = action_button_state.to(device=video.latent.device, dtype=video.latent.dtype)
            expected_state_dim = int(
                getattr(self, "action_button_state_dim", int(getattr(self, "action_button_dim", 8)) + 1)
            )
            if action_button_state.ndim != 3 or action_button_state.shape[-1] != expected_state_dim:
                raise ValueError(
                    f"Action button state must have shape [B, raw_action_frames, {expected_state_dim}], "
                    f"got {tuple(action_button_state.shape)}"
                )
            prepared_action_button_state = action_button_state

        prepared_action_button_values = None
        if action_button_values is not None:
            if video is None:
                raise ValueError("Action button values require a video modality")
            action_button_values = action_button_values.to(device=video.latent.device, dtype=video.latent.dtype)
            expected_button_dim = int(getattr(self, "action_button_dim", 8))
            if action_button_values.ndim != 3 or action_button_values.shape[-1] != expected_button_dim:
                raise ValueError(
                    f"Action button values must have shape [B, raw_action_frames, {expected_button_dim}], "
                    f"got {tuple(action_button_values.shape)}"
                )
            prepared_action_button_values = action_button_values

        prepared_action_button_condition_mask = self._prepare_action_condition_mask(
            action_button_condition_mask,
            condition=prepared_action_button_state,
            video=video,
            name="action_button_condition_mask",
        )

        prepared_action_button_meaning_embeds = None
        if action_button_meaning_embeds is not None:
            if video is None:
                raise ValueError("Action button meaning embeddings require a video modality")
            action_button_meaning_embeds = action_button_meaning_embeds.to(
                device=video.latent.device,
                dtype=video.latent.dtype,
            )
            expected_button_dim = int(getattr(self, "action_button_dim", 8))
            if action_button_meaning_embeds.ndim != 3 or action_button_meaning_embeds.shape[1] != expected_button_dim:
                raise ValueError(
                    f"Action button meaning embeddings must have shape [B, {expected_button_dim}, D], "
                    f"got {tuple(action_button_meaning_embeds.shape)}"
                )
            prepared_action_button_meaning_embeds = self.action_button_meaning_proj(action_button_meaning_embeds)

        video_args = (
            self.video_args_preprocessor.prepare(
                video,
                audio,
                action_joystick_tokens=prepared_action_joystick,
                action_joystick_condition_mask=prepared_action_joystick_condition_mask,
                action_button_state=prepared_action_button_state,
                action_button_values=prepared_action_button_values,
                action_button_condition_mask=prepared_action_button_condition_mask,
                action_button_meaning_embeds=prepared_action_button_meaning_embeds,
                action_button_meaning_fusion=action_button_meaning_fusion,
                action_button_temporal_alignment=action_button_temporal_alignment,
                action_button_prefix_tokens=action_button_prefix_tokens,
            )
            if video is not None
            else None
        )
        if (
            video_args is not None
            and prepared_action_button_state is not None
            and action_button_temporal_alignment == "causal_prefix_rope"
        ):
            metadata = prepare_causal_prefix_attention_metadata(
                query_len=int(video_args.x.shape[1]),
                key_len=int(prepared_action_button_state.shape[1]),
                prefix_tokens=action_button_prefix_tokens,
                dim_head=self.inner_dim // self.num_attention_heads,
                frame_rate=action_button_frame_rate,
                device=video_args.x.device,
                dtype=video_args.x.dtype,
                query_latent_indices=action_button_query_latent_indices,
            )
            video_args = replace(
                video_args,
                action_button_causal_attention_mask=metadata.attention_mask,
                action_button_causal_query_cos=metadata.query_cos,
                action_button_causal_query_sin=metadata.query_sin,
                action_button_causal_key_cos=metadata.key_cos,
                action_button_causal_key_sin=metadata.key_sin,
            )
        audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None
                                    
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
        )

                        
        vx = (
            self._process_output(
                self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep
            )
            if video_out is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax

    @staticmethod
    def _prepare_action_condition_mask(
        mask: torch.Tensor | None,
        *,
        condition: torch.Tensor | None,
        video: Modality | None,
        name: str,
    ) -> torch.Tensor | None:
        if condition is None:
            if mask is not None:
                raise ValueError(f"{name} was provided without its conditioning branch")
            return None
        if video is None:
            raise ValueError(f"{name} requires a video modality")
        batch_size = int(video.latent.shape[0])
        video_tokens = int(video.latent.shape[1])
        if mask is None:
            return torch.ones(batch_size, 1, 1, device=video.latent.device, dtype=video.latent.dtype)
        mask = mask.to(device=video.latent.device, dtype=video.latent.dtype)
        if mask.ndim == 1:
            mask = mask.view(batch_size, 1, 1)
        elif mask.ndim == 2 and mask.shape == (batch_size, 1):
            mask = mask.unsqueeze(-1)
        elif mask.ndim == 2 and mask.shape == (batch_size, video_tokens):
            mask = mask.unsqueeze(-1)
        allowed_shapes = {(batch_size, 1, 1), (batch_size, video_tokens, 1)}
        if tuple(mask.shape) not in allowed_shapes:
            raise ValueError(
                f"{name} must have shape [B], [B, 1], [B, 1, 1], [B, T], or [B, T, 1] "
                f"with T={video_tokens}; got {tuple(mask.shape)}"
            )
        return mask


class LegacyX0Model(torch.nn.Module):
    ''

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        sigma: float,
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
        ''
        vx, ax = self.velocity_model(
            video,
            audio,
            perturbations,
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
        denoised_video = to_denoised(video.latent, vx, sigma) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, sigma) if ax is not None else None
        return denoised_video, denoised_audio


class X0Model(torch.nn.Module):
    ''

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

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
        ''
        vx, ax = self.velocity_model(
            video,
            audio,
            perturbations,
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
        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        return denoised_video, denoised_audio
