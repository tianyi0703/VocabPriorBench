"""CPU-only V0.2 row-norm/direction construction and descriptive checks.

This module opens no files, reads no corpus, fits nothing, and changes no table
scale after construction. The model applies the fixed output bias separately.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from vocabpriorbench.model import tensor_sha256


WIDTH = 128
RTOL = 2e-6
ATOL = 2e-7
TINY_NORM = 1e-12
CONDITIONS = ("C", "U", "RS", "S")


def _validate_source(C: Tensor) -> None:
    if C.device.type != "cpu" or C.dtype != torch.float32:
        raise ValueError("C must be a CPU float32 tensor")
    if C.ndim != 2 or C.shape[0] < 1 or C.shape[1] != WIDTH:
        raise ValueError("C must have nonempty shape [vocabulary, 128]")
    if not torch.isfinite(C).all():
        raise ValueError("C must be finite")


def _sign_table(C: Tensor) -> Tensor:
    # Exact V0.1 expression, including sign(+0) = sign(-0) = +1.
    return (torch.where(C >= 0, 1.0, -1.0) / math.sqrt(WIDTH)).to(torch.float32)


def build_tables(C: Tensor) -> tuple[dict[str, Tensor], Tensor]:
    """Return explicit C/U/RS/S FP32 tables and original FP64 L2 row norms.

    C and S retain their defined FP32 values. U and RS are constructed using
    FP64 operations and then cast once. Positive tiny rows are divided by their
    actual norm, without epsilon/clamping; exactly zero U rows use S instead.
    Every returned table has independent storage, detached from caller state.
    """
    _validate_source(C)
    continuous = C.detach().clone()
    sign = _sign_table(continuous)
    source64 = continuous.double()
    r = source64.norm(p=2, dim=1)
    nonzero = r > 0
    unit64 = sign.double()
    unit64[nonzero] = source64[nonzero] / r[nonzero, None]
    restored64 = r[:, None] * sign.double()
    tables = {"C": continuous, "U": unit64.float(), "RS": restored64.float(), "S": sign}
    if not all(torch.isfinite(table).all() for table in tables.values()):
        raise ValueError("A constructed table is nonfinite")
    return tables, r


def _error(actual: Tensor, expected: Tensor) -> dict:
    actual, expected = actual.double(), expected.double()
    delta = (actual - expected).abs()
    nonzero = expected != 0
    return {
        "passed": bool(torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)),
        "max_abs_error": float(delta.max()) if delta.numel() else 0.0,
        "max_relative_error_on_nonzero_reference": (
            float((delta[nonzero] / expected[nonzero].abs()).max()) if nonzero.any() else None
        ),
        "nonzero_actual_at_zero_reference_count": int(((expected == 0) & (actual != 0)).sum()),
    }


def tables_check(
    tables: Mapping[str, Tensor], r: Tensor, *, source_C: Tensor | None = None
) -> dict:
    """Report actual rounding errors at the predeclared common tolerances.

    Invalid interfaces raise ValueError; numerical invariant failures are
    returned as ``passed=False`` for the caller to record before stopping.
    """
    if set(tables) != set(CONDITIONS):
        raise ValueError("tables must contain exactly C, U, RS, S")
    C = tables["C"]
    _validate_source(C)
    for name in CONDITIONS:
        value = tables[name]
        _validate_source(value)
        if value.shape != C.shape:
            raise ValueError(f"{name} has a different shape")
    if r.device.type != "cpu" or r.dtype != torch.float64 or r.shape != (C.shape[0],):
        raise ValueError("r must be CPU float64 with shape [vocabulary]")
    if not torch.isfinite(r).all() or (r < 0).any():
        raise ValueError("r must be finite and nonnegative")
    if source_C is not None:
        _validate_source(source_C)
        if source_C.shape != C.shape:
            raise ValueError("source_C shape mismatch")
    C64, U64, RS64, S64 = (tables[name].double() for name in CONDITIONS)
    norms = {name: tables[name].double().norm(p=2, dim=1) for name in CONDITIONS}
    rms = {name: tables[name].double().square().mean().sqrt() for name in CONDITIONS}
    ones = torch.ones_like(r)
    expected_sign = _sign_table(C)
    nonzero = r > 0
    expected_unit = expected_sign.double()
    expected_unit[nonzero] = C64[nonzero] / r[nonzero, None]
    zero = ~nonzero
    checks = {
        "C_equals_original_FP32": source_C is None or tensor_sha256(C) == tensor_sha256(source_C),
        "S_equals_exact_V01_FP32_expression": tensor_sha256(tables["S"]) == tensor_sha256(expected_sign),
        "U_equals_specified_FP64_then_FP32_construction": torch.equal(tables["U"], expected_unit.float()),
        "RS_equals_specified_FP64_then_FP32_construction": torch.equal(tables["RS"], (r[:, None] * expected_sign.double()).float()),
        "U_zero_rows_equal_S": torch.equal(tables["U"][zero], tables["S"][zero]),
    }
    errors = {
        "r_matches_original_C_norm": _error(r, C64.norm(p=2, dim=1)),
        "C_equals_r_times_U": _error(C64, r[:, None] * U64),
        "U_unit_row_norm": _error(norms["U"], ones),
        "S_unit_row_norm": _error(norms["S"], ones),
        "RS_row_norm_matches_r": _error(norms["RS"], r),
        "U_global_RMS_unit": _error(rms["U"], torch.tensor(1 / math.sqrt(WIDTH), dtype=torch.float64)),
        "S_global_RMS_unit": _error(rms["S"], torch.tensor(1 / math.sqrt(WIDTH), dtype=torch.float64)),
        "RS_global_RMS_matches_C": _error(rms["RS"], rms["C"]),
    }
    return {
        "passed": all(bool(value) for value in checks.values()) and all(item["passed"] for item in errors.values()),
        "rtol": RTOL,
        "atol": ATOL,
        "all_shape_dtype_and_finiteness_valid": True,
        "exact_checks": {key: bool(value) for key, value in checks.items()},
        "actual_errors": errors,
    }


def _quantiles(values: Tensor) -> dict[str, float]:
    probabilities = torch.tensor([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1], dtype=torch.float64)
    return dict(zip(
        ["min", "p1", "p25", "median", "p75", "p99", "max"],
        (float(value) for value in torch.quantile(values.double(), probabilities)),
    ))


def describe_tables(
    tables: Mapping[str, Tensor],
    r: Tensor,
    *,
    source_C: Tensor | None = None,
    pi: Tensor | None = None,
    counts: Tensor | None = None,
    special_token_ids: Mapping[str, int | Sequence[int] | None] | None = None,
    tokenizer_length: int | None = None,
) -> dict:
    """Return JSON-safe descriptive metadata; no fitting or recalibration.

    Optional counts are actual selected-training loss-target counts. Optional
    pi is the already fixed training prior, used only to report energy. This
    function does not estimate either quantity and does not inspect dev data.
    """
    checks = tables_check(tables, r, source_C=source_C)
    vocabulary = tables["C"].shape[0]
    if counts is not None:
        if counts.device.type != "cpu" or counts.shape != (vocabulary,):
            raise ValueError("counts must be CPU with shape [vocabulary]")
        if counts.is_floating_point() or (counts < 0).any():
            raise ValueError("counts must be nonnegative integers")
    if pi is not None:
        if pi.device.type != "cpu" or pi.shape != (vocabulary,):
            raise ValueError("pi must be CPU with shape [vocabulary]")
        if not torch.isfinite(pi).all() or (pi <= 0).any() or abs(float(pi.double().sum()) - 1) > 1e-12:
            raise ValueError("pi must be a finite positive normalized prior")
    if tokenizer_length is not None and not 0 < tokenizer_length <= vocabulary:
        raise ValueError("tokenizer_length must be in [1, vocabulary]")
    special = {}
    for name, token_ids in (special_token_ids or {}).items():
        ids = [] if token_ids is None else ([token_ids] if isinstance(token_ids, int) else list(token_ids))
        if any(not isinstance(i, int) or i < 0 or i >= vocabulary for i in ids):
            raise ValueError(f"Invalid special token IDs for {name}")
        special[str(name)] = ids
    zero_ids = (r == 0).nonzero().flatten().tolist()
    tiny_ids = ((r > 0) & (r < TINY_NORM)).nonzero().flatten().tolist()

    def row_metadata(token_id: int) -> dict:
        return {
            "token_id": token_id,
            "original_L2_norm": float(r[token_id]),
            "training_target_count": int(counts[token_id]) if counts is not None else None,
            "special_token_names": [name for name, ids in special.items() if token_id in ids],
            "tokenizer_unused_output_row": token_id >= tokenizer_length if tokenizer_length is not None else None,
        }

    descriptions = {}
    for name in CONDITIONS:
        value64 = tables[name].double()
        norms = value64.norm(p=2, dim=1)
        item = {
            "shape": list(tables[name].shape),
            "dtype": str(tables[name].dtype),
            "device": str(tables[name].device),
            "tensor_sha256": tensor_sha256(tables[name]),
            "bytes": tables[name].numel() * tables[name].element_size(),
            "finite": bool(torch.isfinite(tables[name]).all()),
            "row_norm_quantiles": _quantiles(norms),
            "global_RMS": float(value64.square().mean().sqrt()),
        }
        if pi is not None:
            item["training_pi_weighted_RMS"] = float((pi.double().dot(norms.square()) / WIDTH).sqrt())
        descriptions[name] = item
    return {
        "formulas": {"C": "original FP32 C", "U": "C.double()/r for r>0; S.double() for r=0, then FP32", "RS": "r*S.double(), then FP32", "S": "where(C>=0,1.,-1.)/sqrt(128), FP32"},
        "sign_zero": 1,
        "zero_row_rule": "U=S, RS=0; token ID retained",
        "positive_tiny_row_rule": "divide by true FP64 norm; no epsilon or clamp",
        "r_source": "CPU FP64 L2 row norm of original FP32 output C",
        "r_dtype": str(r.dtype),
        "r_tensor_sha256": tensor_sha256(r),
        "original_C_tensor_sha256": tensor_sha256(source_C if source_C is not None else tables["C"]),
        "r_quantiles": _quantiles(r),
        "zero_row_count": len(zero_ids),
        "zero_row_ids": zero_ids,
        "zero_row_metadata": [row_metadata(i) for i in zero_ids],
        "positive_tiny_threshold": TINY_NORM,
        "positive_tiny_row_count": len(tiny_ids),
        "positive_tiny_row_ids": tiny_ids,
        "positive_tiny_row_metadata": [row_metadata(i) for i in tiny_ids],
        "vocabulary": vocabulary,
        "all_output_categories_retained": True,
        "tokenizer_length": tokenizer_length,
        "tokenizer_unused_output_row_count": vocabulary - tokenizer_length if tokenizer_length is not None else None,
        "special_token_ids": special,
        "special_and_unused_rows_policy": "all output rows retained, including effective EOS and tokenizer-unused output rows",
        "training_target_count_total": int(counts.sum()) if counts is not None else None,
        "storage_dtype": "float32",
        "bit_packed": False,
        "bias_policy": "fixed log(pi) added after each table score; never multiplied by r",
        "no_further_normalization_or_calibration": True,
        "tables": descriptions,
        "checks": checks,
    }
