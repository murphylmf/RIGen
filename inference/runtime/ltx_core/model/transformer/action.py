from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn

from ltx_core.model.transformer.attention import (
    AttentionCallable,
    AttentionFunction,
    FlashAttention4,
    FlashAttention4BooleanMask,
    PytorchAttention,
)


@dataclass(frozen=True)
class CausalPrefixAttentionMetadata:
    ''

    attention_mask: torch.Tensor
    query_cos: torch.Tensor
    query_sin: torch.Tensor
    key_cos: torch.Tensor
    key_sin: torch.Tensor


def _temporal_rope_cos_sin(
    positions: torch.Tensor,
    *,
    rope_dim: int,
    dtype: torch.dtype,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rope_dim <= 0 or rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be a positive even integer, got {rope_dim}")
    positions = positions.to(dtype=torch.float32)
    exponent = torch.arange(0, rope_dim, 2, device=positions.device, dtype=torch.float32) / rope_dim
    inv_freq = torch.pow(torch.tensor(theta, device=positions.device, dtype=torch.float32), -exponent)
    angles = positions[:, None] * inv_freq[None, :]
    cos = angles.cos().repeat_interleave(2, dim=-1).to(dtype=dtype)[None, None]
    sin = angles.sin().repeat_interleave(2, dim=-1).to(dtype=dtype)[None, None]
    return cos, sin


def prepare_causal_prefix_attention_metadata(
    *,
    query_len: int,
    key_len: int,
    prefix_tokens: int,
    dim_head: int,
    frame_rate: float,
    device: torch.device,
    dtype: torch.dtype,
    query_latent_indices: torch.Tensor | None = None,
    temporal_scale: int = 8,
    rope_units_per_second: float = 20.0,
) -> CausalPrefixAttentionMetadata:
    ''

    query_len = int(query_len)
    key_len = int(key_len)
    prefix_tokens = int(prefix_tokens)
    dim_head = int(dim_head)
    frame_rate = float(frame_rate)
    if temporal_scale <= 0:
        raise ValueError(f"temporal_scale must be positive, got {temporal_scale}")
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(f"frame_rate must be finite and positive, got {frame_rate}")
    if not math.isfinite(rope_units_per_second) or rope_units_per_second <= 0:
        raise ValueError(f"rope_units_per_second must be finite and positive, got {rope_units_per_second}")
    if key_len < 1 or (key_len - 1) % temporal_scale != 0:
        raise ValueError(
            f"Raw action frame count {key_len} is incompatible with causal VAE scale {temporal_scale}; "
            "expected (latent_frames - 1) * temporal_scale + 1"
        )
    if prefix_tokens < 0 or prefix_tokens > query_len:
        raise ValueError(f"prefix_tokens must be in [0, {query_len}], got {prefix_tokens}")

    latent_frames = (key_len - 1) // temporal_scale + 1
    target_query_len = query_len - prefix_tokens
    if target_query_len <= 0:
        raise ValueError(f"Video target token count must be positive, got {target_query_len}")
    if query_latent_indices is None and target_query_len % latent_frames != 0:
        raise ValueError(
            f"Video target token count {target_query_len} cannot be divided across "
            f"{latent_frames} action-aligned latent frames"
        )
    if query_latent_indices is None:
        spatial_tokens = target_query_len // latent_frames
        latent_end_frames = torch.arange(latent_frames, device=device, dtype=torch.long) * temporal_scale
        query_end_frames = latent_end_frames.repeat_interleave(spatial_tokens)
    else:
        if query_latent_indices.ndim != 1 or query_latent_indices.numel() != target_query_len:
            raise ValueError(
                "query_latent_indices must be a 1D tensor with one entry per non-prefix query token; "
                f"expected {target_query_len}, got {tuple(query_latent_indices.shape)}"
            )
        if query_latent_indices.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError(f"query_latent_indices must have integer dtype, got {query_latent_indices.dtype}")
        query_latent_indices = query_latent_indices.to(device=device, dtype=torch.long)
        if torch.any(query_latent_indices < 0) or torch.any(query_latent_indices >= latent_frames):
            raise ValueError(
                f"query_latent_indices must lie in [0, {latent_frames - 1}], "
                f"got min={int(query_latent_indices.min())} max={int(query_latent_indices.max())}"
            )
        query_end_frames = query_latent_indices * temporal_scale
    key_frames = torch.arange(key_len, device=device, dtype=torch.long)
    attention_mask = key_frames.unsqueeze(0) <= query_end_frames.unsqueeze(1)

                                                                              
    rope_dim = (dim_head // 2) // 2 * 2
    if rope_dim < 2:
        raise ValueError(f"dim_head={dim_head} is too small for partial temporal RoPE")
    position_scale = rope_units_per_second / frame_rate
    query_positions = query_end_frames.to(dtype=torch.float32) * position_scale
    key_positions = key_frames.to(dtype=torch.float32) * position_scale
    query_cos, query_sin = _temporal_rope_cos_sin(query_positions, rope_dim=rope_dim, dtype=dtype)
    key_cos, key_sin = _temporal_rope_cos_sin(key_positions, rope_dim=rope_dim, dtype=dtype)
    return CausalPrefixAttentionMetadata(
        attention_mask=attention_mask,
        query_cos=query_cos,
        query_sin=query_sin,
        key_cos=key_cos,
        key_sin=key_sin,
    )


def _apply_partial_temporal_rope(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    rope_dim = int(cos.shape[-1])
    if cos.shape != sin.shape:
        raise ValueError(f"Temporal RoPE cos/sin shape mismatch: {tuple(cos.shape)} vs {tuple(sin.shape)}")
    if tensor.shape[-1] < rope_dim or rope_dim % 2 != 0:
        raise ValueError(f"Temporal RoPE dim {rope_dim} is incompatible with tensor shape {tuple(tensor.shape)}")
    rotary = tensor[..., :rope_dim]
    passthrough = tensor[..., rope_dim:]
    paired = rotary.reshape(*rotary.shape[:-1], rope_dim // 2, 2)
    rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1).flatten(-2)
    rotary = rotary * cos + rotated * sin
    return torch.cat([rotary, passthrough], dim=-1)


class ActionButtonCrossAttention(nn.Module):
    ''

    def __init__(
        self,
        inner_dim: int,
        heads: int,
        dim_head: int,
        state_dim: int = 9,
        button_dim: int = 8,
        norm_eps: float = 1e-6,
        attention_function: AttentionFunction | AttentionCallable | None = None,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.button_dim = button_dim
        if attention_function is None or attention_function is AttentionFunction.DEFAULT:
                                                                            
                                              
            self.attention_function: AttentionCallable = PytorchAttention()
        elif isinstance(attention_function, AttentionFunction):
            self.attention_function = attention_function.to_callable()
        else:
            self.attention_function = attention_function
        attn_dim = heads * dim_head
        self.to_q = nn.Linear(inner_dim, attn_dim, bias=True)
        self.to_k = nn.Linear(state_dim, attn_dim, bias=True)
        self.to_v = nn.Linear(button_dim, attn_dim, bias=False)
        self.meaning_concat_mlp = nn.Sequential(
            nn.Linear(attn_dim * 2, attn_dim),
            nn.SiLU(),
            nn.Linear(attn_dim, attn_dim),
        )
        self.q_norm = nn.RMSNorm(attn_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(attn_dim, eps=norm_eps)
        self.action_modality_embed = nn.Parameter(torch.zeros(attn_dim))
                                                                           
                                                                             
                                                                             
                                                                   
        self.relative_time_key_embed = nn.Parameter(torch.zeros(9, attn_dim))
        self.relative_time_value_embed = nn.Parameter(torch.zeros(9, attn_dim))
        self.to_out = nn.Linear(attn_dim, inner_dim, bias=True)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.action_modality_embed)
        nn.init.zeros_(self.relative_time_key_embed)
        nn.init.zeros_(self.relative_time_value_embed)
        for module in self.meaning_concat_mlp:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def _run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | FlashAttention4BooleanMask | None = None,
    ) -> torch.Tensor:
        ''

        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
            raise ValueError(
                "Action attention expects q/k/v shaped [B, H, T, D] with matching k/v; "
                f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
            )
        batch, heads, query_len, dim_head = q.shape
        if heads != self.heads or dim_head != self.dim_head:
            raise ValueError(
                f"Action query head geometry {(heads, dim_head)} does not match "
                f"configured {(self.heads, self.dim_head)}."
            )
        if k.shape[0] != batch or k.shape[1] != heads or k.shape[3] != dim_head:
            raise ValueError("Action q/k batch and head geometry must match.")
        key_len = int(k.shape[2])
        q_flat = q.transpose(1, 2).reshape(batch, query_len, heads * dim_head)
        k_flat = k.transpose(1, 2).reshape(batch, key_len, heads * dim_head)
        v_flat = v.transpose(1, 2).reshape(batch, key_len, heads * dim_head)
        out = self.attention_function(q_flat, k_flat, v_flat, heads, mask)
        return out.view(batch, query_len, heads, dim_head).transpose(1, 2)

    def forward(
        self,
        video_tokens: torch.Tensor,
        button_state: torch.Tensor,
        button_values: torch.Tensor,
        button_meaning_embeds: torch.Tensor | None = None,
        meaning_fusion: str = "add",
        temporal_alignment: str = "global",
        prefix_tokens: int = 0,
        temporal_frame_rate: float = 20.0,
        causal_attention_mask: torch.Tensor | None = None,
        causal_query_cos: torch.Tensor | None = None,
        causal_query_sin: torch.Tensor | None = None,
        causal_key_cos: torch.Tensor | None = None,
        causal_key_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, query_len, _ = video_tokens.shape
        key_len = button_state.shape[1]
        q = self.q_norm(self.to_q(video_tokens))
        k_base = self.to_k(button_state)
        if button_meaning_embeds is not None:
            if button_meaning_embeds.ndim != 3 or button_meaning_embeds.shape[:2] != (batch, self.button_dim):
                raise ValueError(
                    "button_meaning_embeds must have shape "
                    f"[B, {self.button_dim}, {k_base.shape[-1]}], got {tuple(button_meaning_embeds.shape)}"
                )
            if button_meaning_embeds.shape[-1] != k_base.shape[-1]:
                raise ValueError(
                    "button_meaning_embeds hidden dim must match attention dim "
                    f"{k_base.shape[-1]}, got {button_meaning_embeds.shape[-1]}"
                )
            weighted_meaning = torch.bmm(
                button_values.to(dtype=k_base.dtype),
                button_meaning_embeds.to(dtype=k_base.dtype),
            )
            if meaning_fusion == "add":
                k_base = k_base + weighted_meaning
            elif meaning_fusion == "concat_mlp":
                k_base = self.meaning_concat_mlp(torch.cat([k_base, weighted_meaning], dim=-1))
            else:
                raise ValueError(f"Unsupported action button meaning fusion: {meaning_fusion}")
        v = self.to_v(button_values)
        if temporal_alignment == "vae_window":
            relative_slots = self._vae_relative_slots(key_len, device=k_base.device)
            k_base = k_base + self.relative_time_key_embed[relative_slots].to(dtype=k_base.dtype).unsqueeze(0)
                                                                               
                                                                            
            active = button_values.abs().amax(dim=-1, keepdim=True).clamp(max=1.0).to(dtype=v.dtype)
            v = v + active * self.relative_time_value_embed[relative_slots].to(dtype=v.dtype).unsqueeze(0)
        k = self.k_norm(k_base + self.action_modality_embed.view(1, 1, -1))

        q = q.view(batch, query_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch, key_len, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch, key_len, self.heads, self.dim_head).transpose(1, 2)
        if temporal_alignment == "global":
            out = self._run_attention(q, k, v)
        elif temporal_alignment == "vae_window":
            out = self._vae_window_attention(q, k, v, prefix_tokens=prefix_tokens)
        elif temporal_alignment == "causal_prefix_rope":
            metadata_values = (
                causal_attention_mask,
                causal_query_cos,
                causal_query_sin,
                causal_key_cos,
                causal_key_sin,
            )
            if all(value is None for value in metadata_values):
                metadata = prepare_causal_prefix_attention_metadata(
                    query_len=query_len,
                    key_len=key_len,
                    prefix_tokens=prefix_tokens,
                    dim_head=self.dim_head,
                    frame_rate=temporal_frame_rate,
                    device=q.device,
                    dtype=q.dtype,
                )
                causal_attention_mask = metadata.attention_mask
                causal_query_cos = metadata.query_cos
                causal_query_sin = metadata.query_sin
                causal_key_cos = metadata.key_cos
                causal_key_sin = metadata.key_sin
            elif any(value is None for value in metadata_values):
                raise ValueError("Causal-prefix attention metadata must be provided together or omitted together")
            out = self._causal_prefix_attention(
                q,
                k,
                v,
                prefix_tokens=prefix_tokens,
                attention_mask=causal_attention_mask,
                query_cos=causal_query_cos,
                query_sin=causal_query_sin,
                key_cos=causal_key_cos,
                key_sin=causal_key_sin,
            )
        else:
            raise ValueError(f"Unsupported action button temporal alignment: {temporal_alignment!r}")
        out = out.transpose(1, 2).reshape(batch, query_len, self.heads * self.dim_head)
        out = self.to_out(out)
        if temporal_alignment in {"vae_window", "causal_prefix_rope"} and prefix_tokens:
            out = torch.cat([torch.zeros_like(out[:, :prefix_tokens]), out[:, prefix_tokens:]], dim=1)
        return out

    def _causal_prefix_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        prefix_tokens: int,
        attention_mask: torch.Tensor,
        query_cos: torch.Tensor,
        query_sin: torch.Tensor,
        key_cos: torch.Tensor,
        key_sin: torch.Tensor,
    ) -> torch.Tensor:
        ''

        query_len = int(q.shape[2])
        key_len = int(k.shape[2])
        prefix_tokens = int(prefix_tokens)
        if prefix_tokens < 0 or prefix_tokens > query_len:
            raise ValueError(f"prefix_tokens must be in [0, {query_len}], got {prefix_tokens}")
        target_query_len = query_len - prefix_tokens
        if attention_mask.dtype != torch.bool or attention_mask.shape != (target_query_len, key_len):
            raise ValueError(
                "Causal-prefix attention mask must be bool with shape "
                f"[{target_query_len}, {key_len}], got {attention_mask.dtype} {tuple(attention_mask.shape)}"
            )
        expected_query_rope = (1, 1, target_query_len)
        expected_key_rope = (1, 1, key_len)
        if query_cos.shape[:3] != expected_query_rope or query_sin.shape != query_cos.shape:
            raise ValueError(
                f"Causal-prefix query RoPE must start with {expected_query_rope}, "
                f"got cos={tuple(query_cos.shape)} sin={tuple(query_sin.shape)}"
            )
        if key_cos.shape[:3] != expected_key_rope or key_sin.shape != key_cos.shape:
            raise ValueError(
                f"Causal-prefix key RoPE must start with {expected_key_rope}, "
                f"got cos={tuple(key_cos.shape)} sin={tuple(key_sin.shape)}"
            )
        if query_cos.shape[-1] != key_cos.shape[-1]:
            raise ValueError(
                f"Causal-prefix query/key RoPE dims differ: {query_cos.shape[-1]} vs {key_cos.shape[-1]}"
            )

        q_target = _apply_partial_temporal_rope(q[:, :, prefix_tokens:], query_cos, query_sin)
        k = _apply_partial_temporal_rope(k, key_cos, key_sin)
        backend_mask: torch.Tensor | FlashAttention4BooleanMask = attention_mask
        if isinstance(self.attention_function, FlashAttention4):
            backend_mask = FlashAttention4BooleanMask(attention_mask)
        target_out = self._run_attention(
            q_target,
            k,
            v,
            mask=backend_mask,
        )
        if not prefix_tokens:
            return target_out
        return torch.cat([torch.zeros_like(q[:, :, :prefix_tokens]), target_out], dim=2)

    @staticmethod
    def _vae_relative_slots(key_len: int, *, device: torch.device) -> torch.Tensor:
        ''

        if key_len < 1 or (key_len - 1) % 8 != 0:
            raise ValueError(
                f"Raw action frame count {key_len} is incompatible with causal VAE scale 8; "
                "expected (latent_frames - 1) * 8 + 1"
            )
        slots = torch.empty(key_len, device=device, dtype=torch.long)
        slots[0] = 0
        if key_len > 1:
            slots[1:] = torch.arange(key_len - 1, device=device) % 8 + 1
        return slots

    def _vae_window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        prefix_tokens: int,
        temporal_scale: int = 8,
    ) -> torch.Tensor:
        ''

        query_len = int(q.shape[2])
        key_len = int(k.shape[2])
        prefix_tokens = int(prefix_tokens)
        if temporal_scale <= 0:
            raise ValueError(f"temporal_scale must be positive, got {temporal_scale}")
        if key_len < 1 or (key_len - 1) % temporal_scale != 0:
            raise ValueError(
                f"Raw action frame count {key_len} is incompatible with causal VAE scale {temporal_scale}; "
                "expected (latent_frames - 1) * temporal_scale + 1"
            )
        if prefix_tokens < 0 or prefix_tokens > query_len:
            raise ValueError(f"prefix_tokens must be in [0, {query_len}], got {prefix_tokens}")

        latent_frames = (key_len - 1) // temporal_scale + 1
        target_query_len = query_len - prefix_tokens
        if target_query_len <= 0 or target_query_len % latent_frames != 0:
            raise ValueError(
                f"Video target token count {target_query_len} cannot be divided across "
                f"{latent_frames} action-aligned latent frames"
            )
        spatial_tokens = target_query_len // latent_frames
        outputs: list[torch.Tensor] = []
        if prefix_tokens:
            outputs.append(torch.zeros_like(q[:, :, :prefix_tokens]))

        for latent_index in range(latent_frames):
            query_start = prefix_tokens + latent_index * spatial_tokens
            query_end = query_start + spatial_tokens
            if latent_index == 0:
                key_start, key_end = 0, 1
            else:
                key_start = temporal_scale * (latent_index - 1) + 1
                key_end = temporal_scale * latent_index + 1
            outputs.append(
                self._run_attention(
                    q[:, :, query_start:query_end],
                    k[:, :, key_start:key_end],
                    v[:, :, key_start:key_end],
                )
            )
        return torch.cat(outputs, dim=2)


def build_per_block_action_modules(
    inner_dim: int,
    heads: int,
    dim_head: int,
    norm_eps: float,
    *,
    button_dim: int = 8,
    attention_function: AttentionFunction | AttentionCallable | None = None,
) -> dict[str, nn.Module]:
    modules: OrderedDict[str, nn.Module] = OrderedDict(
        [
            ("action_joy_injector_layer1", nn.Linear(inner_dim, inner_dim)),
            ("action_joy_hidden_states_layer1", nn.Linear(inner_dim, inner_dim)),
            ("action_joy_injector_layer2", nn.Linear(inner_dim, inner_dim)),
            ("action_joy_hidden_states_layer2", nn.Linear(inner_dim, inner_dim)),
            ("action_joy_scale_layer", nn.Linear(inner_dim, inner_dim)),
            ("action_joy_shift_layer", nn.Linear(inner_dim, inner_dim)),
            (
                "action_button_attn",
                ActionButtonCrossAttention(
                    inner_dim,
                    heads,
                    dim_head,
                    state_dim=button_dim + 1,
                    button_dim=button_dim,
                    norm_eps=norm_eps,
                    attention_function=attention_function,
                ),
            ),
        ]
    )

    nn.init.zeros_(modules["action_joy_scale_layer"].weight)
    nn.init.zeros_(modules["action_joy_scale_layer"].bias)
    nn.init.zeros_(modules["action_joy_shift_layer"].weight)
    nn.init.zeros_(modules["action_joy_shift_layer"].bias)
    return dict(modules)
