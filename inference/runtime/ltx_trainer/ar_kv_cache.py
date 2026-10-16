''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import torch
from torch import Tensor

from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.transformer.action import prepare_causal_prefix_attention_metadata
from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.flex_attention import (
    create_structured_block_mask,
    flex_attention_forward,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import apply_rotary_emb
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.utils import rms_norm, to_denoised
from ltx_trainer.training_strategies.base_strategy import ModelInputs


@dataclass
class ProjectedKVCache:
    ''

    max_tokens: int
    k: Tensor | None = None
    v: Tensor | None = None
    valid_end: int = 0

    def update(self, k: Tensor, v: Tensor, *, start: int) -> tuple[Tensor, Tensor]:
        if k.shape != v.shape or k.ndim != 3:
            raise ValueError(f"Projected K/V must be matching [B, T, D] tensors: {k.shape} vs {v.shape}")
        start = int(start)
        end = start + int(k.shape[1])
        if start < 0 or end > self.max_tokens:
            raise ValueError(f"KV write [{start}:{end}] exceeds cache capacity {self.max_tokens}")
        if start > self.valid_end:
            raise ValueError(
                f"KV cache cannot skip an uncommitted prefix: start={start}, valid_end={self.valid_end}"
            )
        if start < self.valid_end and end != self.valid_end:
            raise ValueError(
                "KV cache may only append a new block or overwrite the current tail: "
                f"write=[{start}:{end}], valid_end={self.valid_end}"
            )

        if self.k is None or self.v is None:
            shape = (int(k.shape[0]), self.max_tokens, int(k.shape[2]))
            self.k = torch.empty(shape, device=k.device, dtype=k.dtype)
            self.v = torch.empty(shape, device=v.device, dtype=v.dtype)
        elif (
            self.k.shape[0] != k.shape[0]
            or self.k.shape[2] != k.shape[2]
            or self.k.device != k.device
            or self.k.dtype != k.dtype
            or self.v.device != v.device
            or self.v.dtype != v.dtype
        ):
            raise ValueError(
                "KV cache allocation no longer matches the current projection: "
                f"cache={tuple(self.k.shape)} {self.k.dtype} {self.k.device}, "
                f"current={tuple(k.shape)} {k.dtype} {k.device}"
            )

                                                                           
                                                                     
        self.k[:, start:end].copy_(k.detach())
        self.v[:, start:end].copy_(v.detach())
        self.valid_end = max(self.valid_end, end)
        return self.k[:, :end], self.v[:, :end]

    def prefix(self, *, end: int) -> tuple[Tensor, Tensor]:
        ''

        end = int(end)
        if end < 0 or end > self.valid_end:
            raise ValueError(f"Requested KV prefix [0:{end}] with valid_end={self.valid_end}")
        if self.k is None or self.v is None:
            if end == 0:
                raise ValueError("An empty, unallocated KV prefix has no shape metadata")
            raise ValueError("KV cache is unallocated despite a non-empty committed prefix")
        return self.k[:, :end], self.v[:, :end]

    @property
    def allocated_bytes(self) -> int:
        tensors = (self.k, self.v)
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)


@dataclass
class LayerAVKVCache:
    video_self: ProjectedKVCache
    audio_self: ProjectedKVCache
    audio_to_video: ProjectedKVCache
    video_to_audio: ProjectedKVCache

    @property
    def allocated_bytes(self) -> int:
        return sum(
            entry.allocated_bytes
            for entry in (
                self.video_self,
                self.audio_self,
                self.audio_to_video,
                self.video_to_audio,
            )
        )


@dataclass
class AVKVCacheBank:
    ''

    layers: list[LayerAVKVCache]
    max_video_tokens: int
    max_audio_tokens: int

    @classmethod
    def create(cls, model: torch.nn.Module, *, max_video_tokens: int, max_audio_tokens: int) -> AVKVCacheBank:
        if max_video_tokens <= 0 or max_audio_tokens <= 0:
            raise ValueError("KV cache token capacities must be positive")
        return cls(
            layers=[
                LayerAVKVCache(
                    video_self=ProjectedKVCache(max_video_tokens),
                    audio_self=ProjectedKVCache(max_audio_tokens),
                    audio_to_video=ProjectedKVCache(max_audio_tokens),
                    video_to_audio=ProjectedKVCache(max_video_tokens),
                )
                for _ in model.transformer_blocks
            ],
            max_video_tokens=int(max_video_tokens),
            max_audio_tokens=int(max_audio_tokens),
        )

    @property
    def allocated_bytes(self) -> int:
        return sum(layer.allocated_bytes for layer in self.layers)


def estimate_cache_bank_bytes(
    model: torch.nn.Module,
    *,
    max_video_tokens: int,
    max_audio_tokens: int,
    dtype: torch.dtype,
) -> int:
    ''

    element_size = torch.empty((), dtype=dtype).element_size()
    total = 0
    for block in model.transformer_blocks:
        total += 2 * max_video_tokens * int(block.attn1.to_k.out_features) * element_size
        total += 2 * max_audio_tokens * int(block.audio_attn1.to_k.out_features) * element_size
        total += 2 * max_audio_tokens * int(block.audio_to_video_attn.to_k.out_features) * element_size
        total += 2 * max_video_tokens * int(block.video_to_audio_attn.to_k.out_features) * element_size
    return total


class _KVCacheAttentionDispatcher:
    ''

    _SUPPORTED_BACKENDS = frozenset(("native", "flex"))

    def __init__(self, backend: str = "native") -> None:
        backend = str(backend)
        if backend not in self._SUPPORTED_BACKENDS:
            supported = ", ".join(sorted(self._SUPPORTED_BACKENDS))
            raise ValueError(f"Unsupported KV-cache attention backend {backend!r}; expected one of: {supported}")
        self.backend = backend
        self.call_count = 0
        self._flex_block_masks: dict[tuple[int, int, torch.device], object] = {}

    @property
    def flex_mask_count(self) -> int:
        return len(self._flex_block_masks)

    def __call__(self, attn: Attention, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        self.call_count += 1
        if self.backend == "native":
            return attn.attention_function(q, k, v, attn.heads, None)

        query_length = int(q.shape[1])
        key_length = int(k.shape[1])
        if query_length <= 0 or key_length <= 0:
            raise ValueError(
                "Flex KV-cache attention requires positive query/key lengths, "
                f"got query={query_length}, key={key_length}"
            )
        cache_key = (query_length, key_length, q.device)
        block_mask = self._flex_block_masks.get(cache_key)
        if block_mask is None:
                                                                              
                                                                            
                                                       
            def all_real_keys_visible(
                _batch: Tensor,
                _head: Tensor,
                _query_index: Tensor,
                key_index: Tensor,
            ) -> Tensor:
                return key_index < key_length

            block_mask = create_structured_block_mask(
                all_real_keys_visible,
                query_length=query_length,
                key_length=key_length,
                device=q.device,
            )
            self._flex_block_masks[cache_key] = block_mask
        return flex_attention_forward(q, k, v, heads=attn.heads, block_mask=block_mask)


def _attention_with_kv_cache(
    attn: Attention,
    x: Tensor,
    *,
    cache: ProjectedKVCache,
    write_start: int,
    context: Tensor | None = None,
    pe: Tensor | None = None,
    k_pe: Tensor | None = None,
    perturbation_mask: Tensor | None = None,
    all_perturbed: bool = False,
    commit_current: bool = True,
    attention_dispatcher: _KVCacheAttentionDispatcher | None = None,
    static_suffix_tokens: int = 0,
) -> Tensor:
    ''

    context = x if context is None else context
    v_current = attn.to_v(context)

    if all_perturbed:
                                                                               
                                                                        
        out = v_current
    else:
        q = attn.q_norm(attn.to_q(x))
        k_current = attn.k_norm(attn.to_k(context))
        if pe is not None:
            q = apply_rotary_emb(q, pe, attn.rope_type)
            k_current = apply_rotary_emb(k_current, pe if k_pe is None else k_pe, attn.rope_type)
        if commit_current:
            k_all, v_all = cache.update(k_current, v_current, start=write_start)
        else:
            if write_start != cache.valid_end:
                raise ValueError(
                    "A differentiable current block must begin exactly after the committed history: "
                    f"write_start={write_start}, valid_end={cache.valid_end}"
                )
            if write_start == 0:
                k_all, v_all = k_current, v_current
            else:
                k_prefix, v_prefix = cache.prefix(end=write_start)
                                                                            
                                                                          
                                                                  
                k_all = torch.cat((k_prefix, k_current), dim=1)
                v_all = torch.cat((v_prefix, v_current), dim=1)
        if static_suffix_tokens:
                                                                             
                                                                              
                                                                              
                                                                               
            suffix_start = write_start - static_suffix_tokens
            if suffix_start < 0:
                raise ValueError("Static KV suffix is longer than committed prefix")
            k_all = torch.cat((k_all[:, :suffix_start], k_all[:, write_start:],
                               k_all[:, suffix_start:write_start]), dim=1)
            v_all = torch.cat((v_all[:, :suffix_start], v_all[:, write_start:],
                               v_all[:, suffix_start:write_start]), dim=1)
        if attention_dispatcher is None:
                                                                           
                                                                    
            out = attn.attention_function(q, k_all, v_all, attn.heads, None)
        else:
            out = attention_dispatcher(attn, q, k_all, v_all)
        if perturbation_mask is not None:
            if v_current.shape != out.shape:
                raise ValueError(
                    "A partial perturbation requires matching query/value lengths, "
                    f"got output={tuple(out.shape)} value={tuple(v_current.shape)}"
                )
            out = out * perturbation_mask + v_current * (1 - perturbation_mask)

    if attn.to_gate_logits is not None:
        gate_logits = attn.to_gate_logits(x)
        batch, tokens, _ = out.shape
        out = out.view(batch, tokens, attn.heads, attn.dim_head)
        out = out * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
        out = out.view(batch, tokens, attn.heads * attn.dim_head)
    return attn.to_out(out)


def _forward_cached_block(                 
    block: torch.nn.Module,
    video: TransformerArgs | None,
    audio: TransformerArgs | None,
    *,
    cache: LayerAVKVCache,
    video_start: int,
    audio_start: int,
    video_self_start: int | None = None,
    video_cross_start: int | None = None,
    perturbations: BatchedPerturbationConfig | None,
    commit_current: bool = True,
    require_mask_free_attention: bool = False,
    attention_dispatcher: _KVCacheAttentionDispatcher | None = None,
    preserve_reference_key_order: bool = False,
) -> tuple[TransformerArgs | None, TransformerArgs | None]:
    ''

    if video is None and audio is None:
        raise ValueError("At least one of video or audio must be provided")
    video_self_start = video_start if video_self_start is None else int(video_self_start)
    video_cross_start = video_start if video_cross_start is None else int(video_cross_start)
    if require_mask_free_attention:
        for name, args in (("video", video), ("audio", audio)):
            if args is None:
                continue
            residual_masks = {
                "context_mask": args.context_mask,
                "self_attention_mask": args.self_attention_mask,
                "cross_attention_mask": args.cross_attention_mask,
            }
            present = [mask_name for mask_name, value in residual_masks.items() if value is not None]
            if present:
                raise RuntimeError(
                    f"Sequential flash-attention {name} block still carries attention masks {present}. "
                    "No fallback backend is allowed."
                )
            if args.context.ndim != 3 or args.context.shape[1] == 0:
                raise RuntimeError(
                    f"Sequential flash-attention {name} block requires a non-empty physical current prompt; "
                    f"got context={tuple(args.context.shape)}."
                )
    batch_size = (video or audio).x.shape[0]
    if perturbations is None:
        perturbations = BatchedPerturbationConfig.empty(batch_size)

    vx = video.x if video is not None else None
    ax = audio.x if audio is not None else None
    run_vx = video is not None and video.enabled and vx.numel() > 0
    run_ax = audio is not None and audio.enabled and ax.numel() > 0
    run_a2v = run_vx and audio is not None and ax.numel() > 0
    run_v2a = run_ax and video is not None and vx.numel() > 0

    if run_vx:
        vshift_msa, vscale_msa, vgate_msa = block.get_ada_values(
            block.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
        )
        norm_vx = rms_norm(vx, eps=block.norm_eps) * (1 + vscale_msa) + vshift_msa
        all_perturbed = perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, block.idx)
        none_perturbed = not perturbations.any_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, block.idx)
        v_mask = (
            perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, block.idx, vx)
            if not all_perturbed and not none_perturbed
            else None
        )
        vx = vx + _attention_with_kv_cache(
            block.attn1,
            norm_vx,
            cache=cache.video_self,
            write_start=video_self_start,
            pe=video.positional_embeddings,
            perturbation_mask=v_mask,
            all_perturbed=all_perturbed,
            commit_current=commit_current,
            attention_dispatcher=attention_dispatcher,
            static_suffix_tokens=(video_self_start - video_start if preserve_reference_key_order else 0),
        ) * vgate_msa
        vx = vx + block._apply_text_cross_attention(
            vx,
            video.context,
            block.attn2,
            block.scale_shift_table,
            getattr(block, "prompt_scale_shift_table", None),
            video.timesteps,
            video.prompt_timestep,
            video.context_mask,
            cross_attention_adaln=block.cross_attention_adaln,
        )
        if video.action_joystick_tokens is not None:
            hidden_before_action = vx
            joy_h = block.action_joy_injector_layer1(video.action_joystick_tokens)
            joy_h = block.action_joy_hidden_states_layer1(joy_h) + joy_h
            joy_h = block.action_joy_injector_layer2(joy_h)
            joy_h = block.action_joy_hidden_states_layer2(joy_h) + joy_h
            joy_scale = block.action_joy_scale_layer(joy_h).to(dtype=vx.dtype)
            joy_shift = block.action_joy_shift_layer(joy_h).to(dtype=vx.dtype)
            if video.action_joystick_condition_mask is not None:
                joy_scale = joy_scale * video.action_joystick_condition_mask
                joy_shift = joy_shift * video.action_joystick_condition_mask
            if joy_scale.shape != vx.shape or joy_shift.shape != vx.shape:
                raise ValueError(
                    f"Action joystick FiLM shape mismatch: scale={tuple(joy_scale.shape)} "
                    f"shift={tuple(joy_shift.shape)} video={tuple(vx.shape)}"
                )
            vx = hidden_before_action + joy_scale * hidden_before_action + joy_shift

        if video.action_button_state is not None or video.action_button_values is not None:
            if video.action_button_state is None or video.action_button_values is None:
                raise ValueError("Both action_button_state and action_button_values must be provided together.")
            button_update = block.action_button_attn(
                video_tokens=rms_norm(vx, eps=block.norm_eps),
                button_state=video.action_button_state,
                button_values=video.action_button_values,
                button_meaning_embeds=video.action_button_meaning_embeds,
                meaning_fusion=video.action_button_meaning_fusion,
                temporal_alignment=video.action_button_temporal_alignment,
                prefix_tokens=video.action_button_prefix_tokens,
                causal_attention_mask=video.action_button_causal_attention_mask,
                causal_query_cos=video.action_button_causal_query_cos,
                causal_query_sin=video.action_button_causal_query_sin,
                causal_key_cos=video.action_button_causal_key_cos,
                causal_key_sin=video.action_button_causal_key_sin,
            )
            if video.action_button_condition_mask is not None:
                button_update = button_update * video.action_button_condition_mask
            vx = vx + button_update

    if run_ax:
        ashift_msa, ascale_msa, agate_msa = block.get_ada_values(
            block.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
        )
        norm_ax = rms_norm(ax, eps=block.norm_eps) * (1 + ascale_msa) + ashift_msa
        all_perturbed = perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, block.idx)
        none_perturbed = not perturbations.any_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, block.idx)
        a_mask = (
            perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, block.idx, ax)
            if not all_perturbed and not none_perturbed
            else None
        )
        ax = ax + _attention_with_kv_cache(
            block.audio_attn1,
            norm_ax,
            cache=cache.audio_self,
            write_start=audio_start,
            pe=audio.positional_embeddings,
            perturbation_mask=a_mask,
            all_perturbed=all_perturbed,
            commit_current=commit_current,
            attention_dispatcher=attention_dispatcher,
        ) * agate_msa
        ax = ax + block._apply_text_cross_attention(
            ax,
            audio.context,
            block.audio_attn2,
            block.audio_scale_shift_table,
            getattr(block, "audio_prompt_scale_shift_table", None),
            audio.timesteps,
            audio.prompt_timestep,
            audio.context_mask,
            cross_attention_adaln=block.cross_attention_adaln,
        )

    if run_a2v or run_v2a:
        vx_norm3 = rms_norm(vx, eps=block.norm_eps)
        ax_norm3 = rms_norm(ax, eps=block.norm_eps)

        if run_a2v and not perturbations.all_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, block.idx):
            scale_v, shift_v, gate_a2v = block.get_av_ca_ada_values(
                block.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
                slice(0, 2),
            )
            vx_scaled = vx_norm3 * (1 + scale_v) + shift_v
            scale_a, shift_a, _ = block.get_av_ca_ada_values(
                block.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
                slice(0, 2),
            )
            ax_scaled = ax_norm3 * (1 + scale_a) + shift_a
            a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, block.idx, vx)
            vx = vx + _attention_with_kv_cache(
                block.audio_to_video_attn,
                vx_scaled,
                context=ax_scaled,
                cache=cache.audio_to_video,
                write_start=audio_start,
                pe=video.cross_positional_embeddings,
                k_pe=audio.cross_positional_embeddings,
                commit_current=commit_current,
                attention_dispatcher=attention_dispatcher,
            ) * gate_a2v * a2v_mask

        if run_v2a and not perturbations.all_in_batch(PerturbationType.SKIP_V2A_CROSS_ATTN, block.idx):
            scale_a, shift_a, gate_v2a = block.get_av_ca_ada_values(
                block.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
                slice(2, 4),
            )
            ax_scaled = ax_norm3 * (1 + scale_a) + shift_a
            scale_v, shift_v, _ = block.get_av_ca_ada_values(
                block.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
                slice(2, 4),
            )
            vx_scaled = vx_norm3 * (1 + scale_v) + shift_v
            v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, block.idx, ax)
            ax = ax + _attention_with_kv_cache(
            block.video_to_audio_attn,
            ax_scaled,
            context=vx_scaled,
            cache=cache.video_to_audio,
            write_start=video_cross_start,
                pe=audio.cross_positional_embeddings,
                k_pe=video.cross_positional_embeddings,
                commit_current=commit_current,
                attention_dispatcher=attention_dispatcher,
            ) * gate_v2a * v2a_mask

    if run_vx:
        vshift_mlp, vscale_mlp, vgate_mlp = block.get_ada_values(
            block.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
        )
        vx_scaled = rms_norm(vx, eps=block.norm_eps) * (1 + vscale_mlp) + vshift_mlp
        vx = vx + block.ff(vx_scaled) * vgate_mlp

    if run_ax:
        ashift_mlp, ascale_mlp, agate_mlp = block.get_ada_values(
            block.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
        )
        ax_scaled = rms_norm(ax, eps=block.norm_eps) * (1 + ascale_mlp) + ashift_mlp
        ax = ax + block.audio_ff(ax_scaled) * agate_mlp

    return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None


def _compact_current_prompt(
    modality: Modality,
    *,
    query_start: int,
    query_end: int,
    block_index: int,
    noisy: bool,
    generated_blocks: int,
) -> tuple[Tensor, Tensor | None]:
    ''

    context = modality.context
    context_mask = modality.context_mask
    if context.ndim != 3:
        raise ValueError(f"Prompt context must be [B, K, D], got {tuple(context.shape)}")
    if context.shape[0] != 1:
        raise ValueError(
            "compact_current prompt execution requires per-rank batch size 1; "
            f"got batch={context.shape[0]}."
        )
    if generated_blocks <= 0 or context.shape[1] % (2 * generated_blocks):
        raise ValueError(
            "Global AR prompt context must contain two equal copies of every generated-block prompt: "
            f"context_tokens={context.shape[1]}, generated_blocks={generated_blocks}."
        )
    if context_mask is None or context_mask.ndim != 3:
        raise ValueError(
            "compact_current requires the original [B, query, context] routing mask so prompt isolation "
            "can be validated before the mask is removed."
        )
    if context_mask.shape != (context.shape[0], modality.latent.shape[1], context.shape[1]):
        raise ValueError(
            "Prompt routing mask does not match the unsliced modality/context geometry: "
            f"mask={tuple(context_mask.shape)}, latent={tuple(modality.latent.shape)}, "
            f"context={tuple(context.shape)}."
        )
    if not 0 <= query_start < query_end <= modality.latent.shape[1]:
        raise ValueError(
            f"Invalid prompt query slice [{query_start}:{query_end}] for {modality.latent.shape[1]} tokens."
        )

    prompt_length = context.shape[1] // (2 * generated_blocks)
    prompt_index = max(block_index - 1, 0)
    if prompt_index >= generated_blocks:
        raise ValueError(
            f"AR block {block_index} maps to prompt {prompt_index}, outside {generated_blocks} prompts."
        )
    half_context = generated_blocks * prompt_length
    use_noisy_copy = noisy and block_index > 0
    context_start = (half_context if use_noisy_copy else 0) + prompt_index * prompt_length
    context_end = context_start + prompt_length

    query_mask = context_mask[:, query_start:query_end]
    outside = query_mask.clone()
    outside[:, :, context_start:context_end] = False
    if torch.any(outside):
        raise ValueError(
            "Prompt routing is not isolated to the expected current prompt; refusing to compact a potentially "
            f"leaky schedule for block={block_index}, noisy={noisy}."
        )
    selected_mask = query_mask[:, :, context_start:context_end].to(dtype=torch.bool)
    if not torch.equal(selected_mask, selected_mask[:, :1].expand_as(selected_mask)):
        raise ValueError("Current prompt validity differs across query tokens; it cannot become mask-free.")
    valid_tokens = selected_mask[0, 0]
    if not torch.any(valid_tokens):
        raise ValueError(
            "Current prompt has no valid tokens. compact_current/mask-free FA does not silently substitute an "
            "unconditional or SDPA path."
        )

    compact_context = context[:, context_start:context_end, :][:, valid_tokens, :]
    context_timesteps = modality.context_timesteps
    compact_context_timesteps = None
    if context_timesteps is not None:
        if context_timesteps.shape != context.shape[:2]:
            raise ValueError(
                "Prompt AdaLN timesteps do not match global context: "
                f"timesteps={tuple(context_timesteps.shape)}, context={tuple(context.shape)}."
            )
        compact_context_timesteps = context_timesteps[:, context_start:context_end][:, valid_tokens]
    return compact_context, compact_context_timesteps


def _slice_modality(
    modality: Modality,
    *,
    start: int,
    end: int,
    block_index: int,
    noisy: bool,
    generated_blocks: int,
    compact_prompt_context: bool,
) -> Modality:
    context_mask = modality.context_mask
    context = modality.context
    context_timesteps = modality.context_timesteps
    if compact_prompt_context:
        context, context_timesteps = _compact_current_prompt(
            modality,
            query_start=start,
            query_end=end,
            block_index=block_index,
            noisy=noisy,
            generated_blocks=generated_blocks,
        )
        context_mask = None
    if context_mask is not None:
        if context_mask.ndim != 3 or context_mask.shape[1] < end:
            raise ValueError(f"Unexpected context mask shape {tuple(context_mask.shape)} for slice [{start}:{end}]")
        context_mask = context_mask[:, start:end]
    cross_timesteps = modality.cross_attention_timesteps
    if cross_timesteps is not None:
        cross_timesteps = cross_timesteps[:, start:end]
    return replace(
        modality,
        latent=modality.latent[:, start:end],
        timesteps=modality.timesteps[:, start:end],
        positions=modality.positions[:, :, start:end],
        context=context,
        context_mask=context_mask,
        attention_mask=None,
        self_attention_block_mask=None,
                                                                             
                                                                           
                                                               
        cross_attention_mask=None,
        cross_attention_timesteps=cross_timesteps,
        context_timesteps=context_timesteps,
    )


def slice_ar_model_inputs(
    inputs: ModelInputs,
    *,
    block_index: int,
    noisy: bool,
    compact_prompt_context: bool = False,
) -> ModelInputs:
    ''

    if inputs.ar_video_block_token_ranges is None or inputs.ar_audio_block_token_ranges is None:
        raise ValueError("AR block token ranges are required for KV-cache inference")
    if inputs.audio is None:
        raise ValueError("TI2AV KV-cache inference requires audio")
    generated_blocks = len(inputs.ar_video_block_token_ranges) - 1
    if generated_blocks != len(inputs.ar_audio_block_token_ranges) - 1:
        raise ValueError("Video/audio AR block counts differ")
    video_total = inputs.video.latent.shape[1] // 2
    audio_total = inputs.audio.latent.shape[1] // 2
    video_start, video_end = inputs.ar_video_block_token_ranges[block_index]
    audio_start, audio_end = inputs.ar_audio_block_token_ranges[block_index]
    video_offset = video_total if noisy else 0
    audio_offset = audio_total if noisy else 0
    absolute_video_start = video_offset + video_start
    absolute_video_end = video_offset + video_end
    absolute_audio_start = audio_offset + audio_start
    absolute_audio_end = audio_offset + audio_end

    action_joystick = inputs.action_joystick_tokens
    if action_joystick is not None:
        if action_joystick.shape[1] != 2 * video_total:
            raise ValueError(
                f"Expected doubled joystick tokens ({2 * video_total}), got {action_joystick.shape[1]}"
            )
        action_joystick = action_joystick[:, absolute_video_start:absolute_video_end]
    query_indices = inputs.action_button_query_latent_indices
    if query_indices is not None:
        if query_indices.ndim != 1 or query_indices.numel() != 2 * video_total:
            raise ValueError(
                f"Expected {2 * video_total} action query indices, got {tuple(query_indices.shape)}"
            )
        query_indices = query_indices[absolute_video_start:absolute_video_end]

    return replace(
        inputs,
        video=_slice_modality(
            inputs.video,
            start=absolute_video_start,
            end=absolute_video_end,
            block_index=block_index,
            noisy=noisy,
            generated_blocks=generated_blocks,
            compact_prompt_context=compact_prompt_context,
        ),
        audio=_slice_modality(
            inputs.audio,
            start=absolute_audio_start,
            end=absolute_audio_end,
            block_index=block_index,
            noisy=noisy,
            generated_blocks=generated_blocks,
            compact_prompt_context=compact_prompt_context,
        ),
        action_joystick_tokens=action_joystick,
        action_button_query_latent_indices=query_indices,
    )


class ARTeacherKVCacheRunner:
    ''

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        max_video_tokens: int,
        max_audio_tokens: int,
        require_mask_free_attention: bool = False,
        attention_backend: str = "native",
        rebuild_full_causal_action_metadata: bool = True,
        preserve_reference_key_order: bool = False,
    ) -> None:
        if not hasattr(model, "transformer_blocks"):
            raise TypeError("ARTeacherKVCacheRunner expects an LTXModel-like transformer")
        self.model = model
        self.max_video_tokens = int(max_video_tokens)
        self.max_audio_tokens = int(max_audio_tokens)
        self.require_mask_free_attention = bool(require_mask_free_attention)
        self.rebuild_full_causal_action_metadata = bool(rebuild_full_causal_action_metadata)
        self.preserve_reference_key_order = bool(preserve_reference_key_order)
        self.attention_dispatcher = _KVCacheAttentionDispatcher(attention_backend)

    @property
    def attention_backend(self) -> str:
        return self.attention_dispatcher.backend

    @property
    def attention_route_stats(self) -> dict[str, int | str]:
        return {
            "backend": self.attention_dispatcher.backend,
            "call_count": self.attention_dispatcher.call_count,
            "flex_mask_count": self.attention_dispatcher.flex_mask_count,
        }

    def new_cache_bank(self) -> AVKVCacheBank:
        return AVKVCacheBank.create(
            self.model,
            max_video_tokens=self.max_video_tokens,
            max_audio_tokens=self.max_audio_tokens,
        )

    def forward(
        self,
        inputs: ModelInputs,
        *,
        cache: AVKVCacheBank,
        video_start: int,
        audio_start: int,
        video_self_start: int | None = None,
        video_cross_start: int | None = None,
        action_button_meaning_fusion: str,
        perturbations: BatchedPerturbationConfig | None = None,
        layer_state_callback: Callable[[int, TransformerArgs | None, TransformerArgs | None], None] | None = None,
    ) -> tuple[Tensor, Tensor]:
        model = self.model
        video = inputs.video
        audio = inputs.audio
        if audio is None:
            raise ValueError("TI2AV KV-cache inference requires audio")
        if len(cache.layers) != len(model.transformer_blocks):
            raise ValueError("KV cache layer count does not match the transformer")
        resolved_video_self_start = (
            video_start if video_self_start is None else int(video_self_start)
        )
        resolved_video_cross_start = (
            video_start if video_cross_start is None else int(video_cross_start)
        )
        if max(resolved_video_self_start, resolved_video_cross_start) + video.latent.shape[1] > cache.max_video_tokens:
            raise ValueError("Video block exceeds KV cache capacity")
        if audio_start + audio.latent.shape[1] > cache.max_audio_tokens:
            raise ValueError("Audio block exceeds KV cache capacity")

        prepared_joystick = None
        if inputs.action_joystick_tokens is not None:
            joystick = inputs.action_joystick_tokens.to(device=video.latent.device, dtype=video.latent.dtype)
            if joystick.ndim != 3 or joystick.shape[-1] != 32:
                raise ValueError(f"Action joystick tokens must be [B, T, 32], got {tuple(joystick.shape)}")
            prepared_joystick = model.action_joystick_proj(joystick)
        prepared_joystick_mask = model._prepare_action_condition_mask(
            inputs.action_joystick_condition_mask,
            condition=prepared_joystick,
            video=video,
            name="action_joystick_condition_mask",
        )

        button_state = inputs.action_button_state
        if button_state is not None:
            button_state = button_state.to(device=video.latent.device, dtype=video.latent.dtype)
            expected_state_dim = int(model.action_button_state_dim)
            if button_state.ndim != 3 or button_state.shape[-1] != expected_state_dim:
                raise ValueError(
                    f"Action button state must be [B, raw_frames, {expected_state_dim}], "
                    f"got {tuple(button_state.shape)}"
                )
        button_values = inputs.action_button_values
        if button_values is not None:
            button_values = button_values.to(device=video.latent.device, dtype=video.latent.dtype)
            expected_button_dim = int(model.action_button_dim)
            if button_values.ndim != 3 or button_values.shape[-1] != expected_button_dim:
                raise ValueError(
                    f"Action button values must be [B, raw_frames, {expected_button_dim}], "
                    f"got {tuple(button_values.shape)}"
                )
        button_mask = model._prepare_action_condition_mask(
            inputs.action_button_condition_mask,
            condition=button_state,
            video=video,
            name="action_button_condition_mask",
        )
        meaning_embeds = inputs.action_button_meaning_embeds
        if meaning_embeds is not None:
            meaning_embeds = model.action_button_meaning_proj(
                meaning_embeds.to(device=video.latent.device, dtype=video.latent.dtype)
            )

        video_args = model.video_args_preprocessor.prepare(
            video,
            audio,
            action_joystick_tokens=prepared_joystick,
            action_joystick_condition_mask=prepared_joystick_mask,
            action_button_state=button_state,
            action_button_values=button_values,
            action_button_condition_mask=button_mask,
            action_button_meaning_embeds=meaning_embeds,
            action_button_meaning_fusion=action_button_meaning_fusion,
            action_button_temporal_alignment=inputs.action_button_temporal_alignment,
            action_button_prefix_tokens=inputs.action_button_prefix_tokens,
        )
        if button_state is not None and inputs.action_button_temporal_alignment == "causal_prefix_rope":
            if inputs.action_button_prefix_tokens:
                raise ValueError("Incremental causal-prefix action attention currently requires prefix_tokens=0")
            if self.rebuild_full_causal_action_metadata:
                key_len = int(button_state.shape[1])
                latent_frames = (key_len - 1) // 8 + 1
                if self.max_video_tokens % latent_frames:
                    raise ValueError(
                        f"Full video token capacity {self.max_video_tokens} is not divisible by "
                        f"the {latent_frames} action-aligned latent frames"
                    )
                spatial_tokens = self.max_video_tokens // latent_frames
                full_query_indices = torch.arange(
                    latent_frames,
                    device=video_args.x.device,
                    dtype=torch.long,
                ).repeat_interleave(spatial_tokens)
                current_query_indices = inputs.action_button_query_latent_indices
                query_end = video_start + int(video_args.x.shape[1])
                if current_query_indices is not None:
                    full_query_indices[video_start:query_end] = current_query_indices.to(
                        device=video_args.x.device,
                        dtype=torch.long,
                    )
                metadata = prepare_causal_prefix_attention_metadata(
                    query_len=self.max_video_tokens,
                    key_len=key_len,
                    prefix_tokens=0,
                    dim_head=model.inner_dim // model.num_attention_heads,
                    frame_rate=inputs.action_button_frame_rate,
                    device=video_args.x.device,
                    dtype=video_args.x.dtype,
                    query_latent_indices=full_query_indices,
                )
                query_slice = slice(video_start, query_end)
            else:
                query_indices = inputs.action_button_query_latent_indices
                if query_indices is not None:
                    query_indices = query_indices.to(device=video_args.x.device, dtype=torch.long)
                metadata = prepare_causal_prefix_attention_metadata(
                    query_len=int(video_args.x.shape[1]),
                    key_len=int(button_state.shape[1]),
                    prefix_tokens=0,
                    dim_head=model.inner_dim // model.num_attention_heads,
                    frame_rate=inputs.action_button_frame_rate,
                    device=video_args.x.device,
                    dtype=video_args.x.dtype,
                    query_latent_indices=query_indices,
                )
                query_slice = slice(None)
            video_args = replace(
                video_args,
                action_button_causal_attention_mask=metadata.attention_mask[query_slice],
                action_button_causal_query_cos=metadata.query_cos[:, :, query_slice],
                action_button_causal_query_sin=metadata.query_sin[:, :, query_slice],
                action_button_causal_key_cos=metadata.key_cos,
                action_button_causal_key_sin=metadata.key_sin,
            )
        audio_args = model.audio_args_preprocessor.prepare(audio, video)

        for layer_index, (block, layer_cache) in enumerate(
            zip(model.transformer_blocks, cache.layers, strict=True)
        ):
            video_args, audio_args = _forward_cached_block(
                block,
                video_args,
                audio_args,
                cache=layer_cache,
                video_start=video_start,
                audio_start=audio_start,
                video_self_start=resolved_video_self_start,
                video_cross_start=resolved_video_cross_start,
                perturbations=perturbations,
                require_mask_free_attention=self.require_mask_free_attention,
                attention_dispatcher=self.attention_dispatcher,
                preserve_reference_key_order=self.preserve_reference_key_order,
            )
            if layer_state_callback is not None:
                layer_state_callback(layer_index, video_args, audio_args)

        video_pred = model._process_output(
            model.scale_shift_table,
            model.norm_out,
            model.proj_out,
            video_args.x,
            video_args.embedded_timestep,
        )
        audio_pred = model._process_output(
            model.audio_scale_shift_table,
            model.audio_norm_out,
            model.audio_proj_out,
            audio_args.x,
            audio_args.embedded_timestep,
        )
        return video_pred, audio_pred

    def predict_x0(
        self,
        inputs: ModelInputs,
        *,
        cache: AVKVCacheBank,
        video_start: int,
        audio_start: int,
        video_self_start: int | None = None,
        video_cross_start: int | None = None,
        action_button_meaning_fusion: str,
        perturbations: BatchedPerturbationConfig | None = None,
        layer_state_callback: Callable[[int, TransformerArgs | None, TransformerArgs | None], None] | None = None,
    ) -> tuple[Tensor, Tensor]:
        video_pred, audio_pred = self.forward(
            inputs,
            cache=cache,
            video_start=video_start,
            audio_start=audio_start,
            video_self_start=video_self_start,
            video_cross_start=video_cross_start,
            action_button_meaning_fusion=action_button_meaning_fusion,
            perturbations=perturbations,
            layer_state_callback=layer_state_callback,
        )
        return (
            to_denoised(inputs.video.latent, video_pred, inputs.video.timesteps),
            to_denoised(inputs.audio.latent, audio_pred, inputs.audio.timesteps),
        )


__all__ = [
    "ARTeacherKVCacheRunner",
    "AVKVCacheBank",
    "estimate_cache_bank_bytes",
    "slice_ar_model_inputs",
]
