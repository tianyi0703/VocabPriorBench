"""Small causal student with two independent token-code tables.

The donor contributes only ``input_codes`` and ``output_codes``.  Every other
parameter is random student state and is included in ``common_state_dict``.
There is no vocabulary bias, no unigram bias, and no implicit weight tying.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CODE_PARAMETER_NAMES = frozenset({"input_codes", "output_codes"})


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        # Bias-free Q/K/V avoids a mathematically redundant key-bias parameter.
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.projection = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        allowed = None
        if attention_mask is not None:
            causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
            allowed = causal[None, None] & attention_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attention_mask is None,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, dim)
        return self.projection(attended)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = CausalSelfAttention(dim, heads, dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        x = x + self.residual_dropout(self.attention(self.attention_norm(x), attention_mask))
        return x + self.residual_dropout(self.ff(self.ff_norm(x)))


class Student(nn.Module):
    """Pre-LN causal Transformer behind a shared low-dimensional interface.

    For code matrices ``C_in, C_out`` of shape ``[vocabulary, code_width]``:

    ``x = C_in[ids] @ W_in + learned_position``
    ``h = final_LN(causal_blocks(x))``
    ``logits = (h @ W_out) @ C_out.T``

    The two Linear modules use PyTorch's transposed weight convention. Codes
    are cloned into distinct Parameters even when the caller passes one Tensor
    twice. ``trainable_codes`` changes only their requires_grad flags. Scaling,
    PCA fitting, and sign binarization belong in source preparation, not here.
    """

    def __init__(
        self,
        code_in: Tensor,
        code_out: Tensor,
        trainable_codes: bool,
        *,
        dim: int = 128,
        layers: int = 2,
        heads: int = 4,
        ff_dim: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if code_in.ndim != 2 or code_out.ndim != 2 or code_in.shape != code_out.shape:
            raise ValueError("input and output codes must have the same [vocabulary, width] shape")
        if not code_in.is_floating_point() or not code_out.is_floating_point():
            raise ValueError("codes must have floating storage; no bit-packed kernel is implemented")
        if code_in.dtype != code_out.dtype or code_in.device != code_out.device:
            raise ValueError("input and output codes must have the same dtype and device")
        if not torch.isfinite(code_in).all() or not torch.isfinite(code_out).all():
            raise ValueError("codes must be finite")
        if min(*code_in.shape, dim, layers, heads, ff_dim, max_seq_len) < 1:
            raise ValueError("all architecture dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.vocab_size, self.code_width = code_in.shape
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.input_codes = nn.Parameter(code_in.detach().clone(), requires_grad=trainable_codes)
        self.output_codes = nn.Parameter(code_out.detach().clone(), requires_grad=trainable_codes)
        self.input_projection = nn.Linear(self.code_width, dim, bias=False)
        self.position = nn.Embedding(max_seq_len, dim)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, heads, ff_dim, dropout) for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, self.code_width, bias=False)
        # This also makes non-default floating dtypes/devices coherent at init.
        self.to(device=code_in.device, dtype=code_in.dtype)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.max_seq_len or input_ids.shape[1] < 1:
            raise ValueError("sequence length must be in [1, max_seq_len]")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids")
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.input_projection(F.embedding(input_ids, self.input_codes))
        x = x + self.position(position_ids)[None]
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.output_projection(self.final_norm(x))
        return F.linear(x, self.output_codes)


def next_token_labels(
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """Align next-token targets with full logits, ignoring pads and the last slot.

    Both source and target must be valid; this handles left or right padding and
    holes in a mask. It does not insert EOS or infer document boundaries.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    labels = torch.full_like(input_ids, ignore_index)
    labels[:, :-1] = input_ids[:, 1:]
    if attention_mask is not None:
        valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        pair_valid = valid[:, :-1] & valid[:, 1:]
        labels[:, :-1].masked_fill_(~pair_valid, ignore_index)
    return labels


def common_state_dict(model: nn.Module) -> OrderedDict[str, Tensor]:
    """Detached copies of every student parameter except the two code tables."""
    return OrderedDict(
        (name, tensor.detach().clone())
        for name, tensor in model.state_dict().items()
        if name not in CODE_PARAMETER_NAMES
    )


def load_common_state_dict(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    """Load exactly all common parameters, rejecting missing/extra keys."""
    expected = set(common_state_dict(model))
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise ValueError(f"common state mismatch: missing={missing}, unexpected={unexpected}")
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != CODE_PARAMETER_NAMES or result.unexpected_keys:
        raise RuntimeError(f"unexpected common-state load result: {result}")


def tensor_sha256(tensor: Tensor) -> str:
    """Canonical SHA-256 including dtype, shape and raw CPU tensor bytes."""
    value = tensor.detach().cpu().contiguous()
    metadata = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True
    ).encode("utf-8")
    digest = hashlib.sha256(metadata + b"\0")
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_sha256(state: Mapping[str, Tensor]) -> str:
    """Order-independent state hash; names and each full tensor hash are included."""
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor_sha256(state[name]).encode("ascii") + b"\0")
    return digest.hexdigest()


def unique_parameter_count(model: nn.Module, trainable_only: bool = False) -> int:
    """Count parameter objects once, even if a module has aliases."""
    parameters = {
        id(parameter): parameter
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    }
    return sum(parameter.numel() for parameter in parameters.values())
