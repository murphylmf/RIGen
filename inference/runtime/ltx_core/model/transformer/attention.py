import importlib
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch

from ltx_core.model.transformer.flex_attention import flex_attention_forward, is_block_mask
from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb

memory_efficient_attention = None
flash_attn_2_func = None
flash_attn_2_import_error: Exception | None = None
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None
try:
                                                                        
                                                                             
    from flash_attn import flash_attn_func as flash_attn_2_func
except Exception as exc:                                                      
    flash_attn_2_func = None
    flash_attn_2_import_error = exc

flash_attn_3_interface = None
flash_attn_3_import_error: Exception | None = None
flash_attn_3_import_attempted = False
flash_attn_4_func = None
flash_attn_4_import_error: Exception | None = None
flash_attn_4_import_attempted = False
flash_attn_4_boolean_mask_mod = None
flash_attn_4_boolean_mask_mod_error: Exception | None = None


def _load_flash_attention_3():
    ''

    global flash_attn_3_import_attempted, flash_attn_3_import_error, flash_attn_3_interface
    if not flash_attn_3_import_attempted:
        flash_attn_3_import_attempted = True
        try:
            flash_attn_3_interface = importlib.import_module("flash_attn_interface")
        except Exception as exc:                                                    
            flash_attn_3_import_error = exc
    if flash_attn_3_interface is None:
        detail = f" ({flash_attn_3_import_error})" if flash_attn_3_import_error is not None else ""
        raise RuntimeError(
            "FlashAttention3 was selected but the `flash_attn_interface` CUDA extension "
            f"is unavailable{detail}. No fallback backend will be used."
        )
    return flash_attn_3_interface


def _load_flash_attention_4():
    ''

    global flash_attn_4_func, flash_attn_4_import_attempted, flash_attn_4_import_error
    if not flash_attn_4_import_attempted:
        flash_attn_4_import_attempted = True
        try:
            module = importlib.import_module("flash_attn.cute")
            flash_attn_4_func = module.flash_attn_func
        except Exception as exc:                                                          
            flash_attn_4_import_error = exc
    if flash_attn_4_func is None:
        detail = f" ({flash_attn_4_import_error})" if flash_attn_4_import_error is not None else ""
        raise RuntimeError(
            "FlashAttention4 was selected but `flash_attn.cute.flash_attn_func` is unavailable"
            f"{detail}. No fallback backend will be used."
        )
    return flash_attn_4_func


def _load_flash_attention_4_boolean_mask_mod():
    ''

    global flash_attn_4_boolean_mask_mod, flash_attn_4_boolean_mask_mod_error
    if flash_attn_4_boolean_mask_mod is None and flash_attn_4_boolean_mask_mod_error is None:
        try:
            cutlass = importlib.import_module("cutlass")
            cute = importlib.import_module("cutlass.cute")
            utils = importlib.import_module("flash_attn.cute.utils")

            @cute.jit
            def boolean_mask_mod(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):                
                attention_mask = aux_tensors[0]
                keep = attention_mask[q_idx[0], kv_idx[0]]
                return utils.scalar_to_ssa(keep, cutlass.Boolean)

            flash_attn_4_boolean_mask_mod = boolean_mask_mod
        except Exception as exc:                                                          
            flash_attn_4_boolean_mask_mod_error = exc
    if flash_attn_4_boolean_mask_mod is None:
        detail = (
            f" ({flash_attn_4_boolean_mask_mod_error})"
            if flash_attn_4_boolean_mask_mod_error is not None
            else ""
        )
        raise RuntimeError(
            "FlashAttention4 boolean mask_mod could not be constructed"
            f"{detail}. No fallback backend will be used."
        )
    return flash_attn_4_boolean_mask_mod


@dataclass(frozen=True)
class FlashAttention4BooleanMask:
    ''

    tensor: torch.Tensor
                                                                            
                                                                             
                                                                                
    source_tensor: torch.Tensor | None = None


def _convert_mask_for_flash_attention_4(
    mask: torch.Tensor,
    *,
    query_length: int,
    key_length: int,
) -> FlashAttention4BooleanMask:
    ''

    source_tensor = mask
    if mask.ndim == 4:
        if mask.shape[1] != 1:
            raise ValueError(f"FA4 mask conversion requires a singleton head axis, got {tuple(mask.shape)}")
        mask = mask[:, 0]
    if mask.ndim == 3:
        if mask.shape[0] < 1:
            raise ValueError("FA4 mask conversion received an empty batch axis")
        first = mask[0]
        if mask.shape[0] > 1 and not torch.equal(mask, first.unsqueeze(0).expand_as(mask)):
            raise ValueError(
                "FA4 mask_mod currently requires the same structural mask for every sample; "
                "per-sample masks are not supported and no fallback is allowed."
            )
        mask = first
    if mask.ndim != 2 or tuple(mask.shape) != (query_length, key_length):
        raise ValueError(
            "FA4 mask conversion requires [Tq, Tk] after batch/head normalization; "
            f"expected={(query_length, key_length)}, got={tuple(mask.shape)}"
        )

    if mask.dtype == torch.bool:
        keep = mask
    elif torch.is_floating_point(mask):
        zero = mask == 0
        masked = torch.isneginf(mask) | (mask <= torch.finfo(mask.dtype).min / 2)
        if not torch.all(zero | masked):
            raise ValueError(
                "FA4 mask conversion accepts only exact binary additive masks (0 or finfo.min/-inf); "
                "soft attention biases are unsupported and no fallback is allowed."
            )
        keep = zero
    else:
        unique = torch.unique(mask)
        if not torch.all((unique == 0) | (unique == 1)):
            raise ValueError(f"FA4 integer mask must contain only 0/1 values, got {unique.tolist()[:8]}")
        keep = mask != 0

                                                                               
                                                                              
                                                                          
    return FlashAttention4BooleanMask(keep.contiguous(), source_tensor=source_tensor)


class AttentionCallable(Protocol):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


class PytorchAttention(AttentionCallable):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
                                                              
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
                                                              
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class XFormersAttention(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_efficient_attention is None:
            raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

                                       
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
                                             
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
                                             
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
                                    
            pad = 8 - mask.shape[-1] % 8
                                                                                          
                                                                                
                                                         
                                                                                                                    
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
            )

            mask_out[..., : mask.shape[-1]] = mask
                                                     
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention2(AttentionCallable):
    ''

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            raise NotImplementedError(
                "FlashAttention2 received an attention mask. No SDPA fallback is allowed; "
                "compact or remove the mask before calling attention."
            )
        if flash_attn_2_func is None:
            detail = f" ({flash_attn_2_import_error})" if flash_attn_2_import_error is not None else ""
            raise RuntimeError(f"FlashAttention2 was selected but `flash-attn` is not available{detail}.")
        if not q.is_cuda:
            raise RuntimeError("FlashAttention2 requires CUDA tensors.")
        if v.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "FlashAttention2 requires the value projection to use float16 or bfloat16; "
                f"got v={v.dtype}."
            )

        b, _, inner_dim = q.shape
        dim_head = inner_dim // heads
        q, k, v = (tensor.view(b, -1, heads, dim_head) for tensor in (q, k, v))
                                                                           
                                                                           
                                                                               
        out = flash_attn_2_func(q.to(v.dtype), k.to(v.dtype), v, dropout_p=0.0, causal=False)
        return out.reshape(b, -1, heads * dim_head)


class FlashAttention3(AttentionCallable):
    ''

    def __init__(self) -> None:
        self._interface = _load_flash_attention_3()

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            raise NotImplementedError(
                "FlashAttention3 received an attention mask. No fallback backend is allowed; "
                "compact or remove the mask before calling attention."
            )
        if not q.is_cuda or not k.is_cuda or not v.is_cuda:
            raise RuntimeError("FlashAttention3 requires CUDA q/k/v tensors; no CPU fallback is allowed.")
        if v.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "FlashAttention3 requires the value projection to use float16 or bfloat16; "
                f"got v={v.dtype}."
            )
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or k.shape != v.shape:
            raise ValueError(
                "FlashAttention3 expects q/k/v shaped [B, T, H*D] with matching k/v; "
                f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
            )
        if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
            raise ValueError(
                "FlashAttention3 q/k batch and hidden dimensions must match; "
                f"got q={tuple(q.shape)}, k={tuple(k.shape)}."
            )
        if q.shape[1] == 0 or k.shape[1] == 0:
            raise ValueError("FlashAttention3 does not accept an empty query or key sequence in this pipeline.")

        b, _, inner_dim = q.shape
        if inner_dim % heads:
            raise ValueError(f"FlashAttention3 hidden dim {inner_dim} is not divisible by {heads} heads.")
        dim_head = inner_dim // heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        out = self._interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention4(AttentionCallable):
    ''

    def __init__(self) -> None:
        self._func = _load_flash_attention_4()

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | FlashAttention4BooleanMask | None = None,
    ) -> torch.Tensor:
        if mask is not None and not isinstance(mask, FlashAttention4BooleanMask):
            raise NotImplementedError(
                "FlashAttention4 received a generic attention mask. No fallback backend is allowed; "
                "use the explicit FA4-native mask type or compact the mask before calling attention."
            )
        if not q.is_cuda or not k.is_cuda or not v.is_cuda:
            raise RuntimeError("FlashAttention4 requires CUDA q/k/v tensors; no CPU fallback is allowed.")
        if v.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "FlashAttention4 requires the value projection to use float16 or bfloat16; "
                f"got v={v.dtype}."
            )
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or k.shape != v.shape:
            raise ValueError(
                "FlashAttention4 expects q/k/v shaped [B, T, H*D] with matching k/v; "
                f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
            )
        if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
            raise ValueError(
                "FlashAttention4 q/k batch and hidden dimensions must match; "
                f"got q={tuple(q.shape)}, k={tuple(k.shape)}."
            )
        if q.shape[1] == 0 or k.shape[1] == 0:
            raise ValueError("FlashAttention4 does not accept an empty query or key sequence in this pipeline.")

        boolean_mask = mask.tensor if isinstance(mask, FlashAttention4BooleanMask) else None
        if boolean_mask is not None:
            expected_mask_shape = (q.shape[1], k.shape[1])
            if boolean_mask.dtype != torch.bool or tuple(boolean_mask.shape) != expected_mask_shape:
                raise ValueError(
                    "FlashAttention4BooleanMask must be bool [Tq, Tk]; "
                    f"expected={expected_mask_shape}, got={boolean_mask.dtype} {tuple(boolean_mask.shape)}."
                )
            if not boolean_mask.is_cuda or boolean_mask.device != q.device:
                raise RuntimeError(
                    "FlashAttention4BooleanMask must be on the same CUDA device as q; "
                    f"mask={boolean_mask.device}, q={q.device}."
                )
            if not boolean_mask.is_contiguous():
                raise ValueError("FlashAttention4BooleanMask must be contiguous; refusing an implicit copy.")

        batch, _, inner_dim = q.shape
        if inner_dim % heads:
            raise ValueError(f"FlashAttention4 hidden dim {inner_dim} is not divisible by {heads} heads.")
        dim_head = inner_dim // heads
        q, k, v = (tensor.view(batch, -1, heads, dim_head) for tensor in (q, k, v))
        call_kwargs = {"causal": False}
        if boolean_mask is not None:
            call_kwargs.update(
                mask_mod=_load_flash_attention_4_boolean_mask_mod(),
                aux_tensors=[boolean_mask],
            )
        result = self._func(
            q.to(v.dtype).contiguous(),
            k.to(v.dtype).contiguous(),
            v.contiguous(),
            **call_kwargs,
        )
        if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], torch.Tensor):
            raise RuntimeError(
                "FlashAttention4 returned an unexpected API result; refusing to substitute another backend."
            )
        return result[0].reshape(batch, -1, heads * dim_head)


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_2 = "flash_attention_2"
    FLASH_ATTENTION_3 = "flash_attention_3"
    FLASH_ATTENTION_4 = "flash_attention_4"
    DEFAULT = "default"

    def to_callable(self) -> AttentionCallable:
        ''
        if self is AttentionFunction.PYTORCH:
            return PytorchAttention()
        elif self is AttentionFunction.XFORMERS:
            return XFormersAttention()
        elif self is AttentionFunction.FLASH_ATTENTION_2:
            return FlashAttention2()
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            return FlashAttention3()
        elif self is AttentionFunction.FLASH_ATTENTION_4:
            return FlashAttention4()
        else:
                                                                    
            return XFormersAttention() if memory_efficient_attention is not None else PytorchAttention()


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = (
            attention_function.to_callable()
            if isinstance(attention_function, AttentionFunction)
            else attention_function
        )

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

                                  
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | object | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        ''
        context = x if context is None else context
        use_attention = not all_perturbed

        v = self.to_v(context)

        if not use_attention:
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(context)

            q = self.q_norm(q)
            k = self.k_norm(k)

            if pe is not None:
                q = apply_rotary_emb(q, pe, self.rope_type)
                k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

            if is_block_mask(mask):
                if isinstance(
                    self.attention_function,
                    (FlashAttention2, FlashAttention3, FlashAttention4),
                ):
                    raise NotImplementedError(
                        f"{type(self.attention_function).__name__} received a BlockMask. "
                        "Refusing to bypass the configured FlashAttention backend through FlexAttention."
                    )
                out = flex_attention_forward(q, k, v, heads=self.heads, block_mask=mask)
            else:
                if isinstance(self.attention_function, FlashAttention4) and isinstance(mask, torch.Tensor):
                    mask = _convert_mask_for_flash_attention_4(
                        mask,
                        query_length=q.shape[1],
                        key_length=k.shape[1],
                    )
                out = self.attention_function(q, k, v, self.heads, mask)               

            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

                                          
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)             
            b, t, _ = out.shape
                                                         
            out = out.view(b, t, self.heads, self.dim_head)
                                                                                           
            gates = 2.0 * torch.sigmoid(gate_logits)             
            out = out * gates.unsqueeze(-1)                               
                                         
            out = out.view(b, t, self.heads * self.dim_head)

        return self.to_out(out)


def configure_attention_backend(
    root: torch.nn.Module,
    backend: AttentionFunction | str,
) -> int:
    ''

                                                                         
                                                          
    from ltx_core.model.transformer.action import ActionButtonCrossAttention

    resolved = AttentionFunction(backend) if isinstance(backend, str) else backend
    configured = 0
    for module in root.modules():
        if isinstance(module, Attention):
            module.attention_function = resolved.to_callable()
            configured += 1
        elif isinstance(module, ActionButtonCrossAttention):
            module.attention_function = resolved.to_callable()
            configured += 1
    return configured
