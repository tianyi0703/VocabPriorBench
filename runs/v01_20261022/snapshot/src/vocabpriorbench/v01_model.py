"""V0.1 output-only intervention; the V0 model and numerics remain unchanged."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .model import Student


NON_COMMON_STATE_NAMES = frozenset({"input_codes", "output_codes", "output_bias"})


class OutputStudent(Student):
    """Freeze continuous input codes and vary only output codes and fixed bias.

    Both zero and log-unigram bias pass through the same FP32 addition. The
    buffer is cloned and detached, is not an optimizer parameter, and is saved
    in the complete state_dict. Under bf16 autocast the base matrix products
    follow V0, then logits are promoted to float32 for the addition and loss.
    Public backbone/projection keys are exactly those of the V0 Student.
    """

    def __init__(
        self,
        code_in: Tensor,
        code_out: Tensor,
        trainable_output: bool,
        output_bias: Tensor,
        **basekwargs,
    ) -> None:
        super().__init__(code_in, code_out, trainable_codes=False, **basekwargs)
        self.input_codes.requires_grad_(False)
        self.output_codes.requires_grad_(trainable_output)
        bias = torch.as_tensor(
            output_bias, dtype=torch.float32, device=self.output_codes.device
        ).detach().clone()
        if bias.ndim != 1 or bias.numel() != self.vocab_size:
            raise ValueError("output_bias must have shape [vocabulary]")
        if not torch.isfinite(bias).all():
            raise ValueError("output_bias must be finite")
        self.register_buffer("output_bias", bias, persistent=True)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return super().forward(input_ids, attention_mask).float() + self.output_bias


def common_state_dict(model: nn.Module) -> OrderedDict[str, Tensor]:
    """Snapshot public student state, excluding both tables and the fixed bias.

    These keys and shapes also accept an existing base Student common snapshot.
    The extra V0.1 bias buffer must never be loaded from another condition.
    """
    return OrderedDict(
        (name, tensor.detach().clone())
        for name, tensor in model.state_dict().items()
        if name not in NON_COMMON_STATE_NAMES
    )


def load_common_state_dict(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    """Strictly require all and only public keys, preserving per-condition state."""
    expected = set(common_state_dict(model))
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise ValueError(f"common state mismatch: missing={missing}, unexpected={unexpected}")
    excluded = set(model.state_dict()) & NON_COMMON_STATE_NAMES
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != excluded or result.unexpected_keys:
        raise RuntimeError(f"unexpected V0.1 common-state load result: {result}")
