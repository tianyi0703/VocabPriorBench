"""CPU-only, train-prior constant-output fits; never a contextual LM bound."""

from __future__ import annotations

import json
import math
import os
import time
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

import torch

from common import (
    CFG, ROOT, RUN, REPORT, PRIVATE, sha, utc, write_json,
    verify_inputs, verify_snapshot,
)


class DeadlineReached(RuntimeError):
    pass


def main() -> None:
    started = time.perf_counter()
    diagnostic = CFG["diagnostic"]
    deadline = float(os.environ.get(
        "V01_CPU_DEADLINE_EPOCH",
        str(time.time() + CFG["budget"]["cpu_numerical_process_limit_seconds"] - 5),
    ))
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    verify_snapshot()
    verify_inputs()
    event_path = RUN / "diagnostic_events.jsonl"
    output_path = REPORT / "diagnostics.json"
    if event_path.exists() or output_path.exists():
        raise RuntimeError("Refuse to overwrite existing diagnostic evidence")

    def check_deadline() -> None:
        if time.time() >= deadline:
            raise DeadlineReached("Shared CPU numerical deadline reached")

    prepared_path = PRIVATE / "prepared.pt"
    code_path = ROOT / "artifacts/codes.pt"
    prepared = torch.load(prepared_path, map_location="cpu", weights_only=True)
    codes = torch.load(code_path, map_location="cpu", weights_only=True)
    pi = prepared["pi"].to(device="cpu", dtype=torch.float64)
    counts = prepared["counts"].to(device="cpu", dtype=torch.int64)
    dev_counts = prepared["dev_counts"].to(device="cpu", dtype=torch.int64)
    cout32 = codes["output"].to(device="cpu", dtype=torch.float32)
    vocabulary, width = cout32.shape
    assert (vocabulary, width) == (CFG["vocabulary"], CFG["code_width"])
    assert pi.shape == counts.shape == dev_counts.shape == (vocabulary,)
    assert torch.isfinite(pi).all() and (pi > 0).all()
    assert (counts >= 0).all() and (dev_counts >= 0).all()
    assert counts.sum() > 0 and dev_counts.sum() > 0
    assert torch.isfinite(cout32).all()
    alpha = CFG["prior"]["alpha"]
    expected_pi = (counts.double() + alpha) / (counts.sum().item() + alpha * vocabulary)
    assert torch.allclose(pi, expected_pi, atol=1e-15, rtol=0), "pi/count mismatch"
    assert abs(pi.sum().item() - 1.0) <= 1e-12
    log_pi = pi.log()
    entropy = float(-(pi * log_pi).sum())
    dev_total = int(dev_counts.sum())
    dev_weights = dev_counts.double() / dev_total
    pi_dev_nll = float(-(dev_weights * log_pi).sum())
    pi_error = float((torch.softmax(log_pi, dim=0) - pi).abs().max())
    assert pi_error <= diagnostic["softmax_prior_atol"]
    # Match the actual float32 sign-table construction, then fit in float64.
    sign32 = torch.where(cout32 >= 0, 1.0, -1.0) / math.sqrt(width)
    tables = {"continuous": cout32.double(), "sign": sign32.double()}
    result = {
        "version": "V0.1",
        "status": "RUNNING",
        "device": "cpu",
        "dtype": "float64",
        "created_utc": utc(),
        "configuration": diagnostic,
        "cpu_deadline_epoch": deadline,
        "deadline_source": "shared_supervisor_environment" if "V01_CPU_DEADLINE_EPOCH" in os.environ else "standalone_fallback",
        "training_valid_targets": int(counts.sum()),
        "dev_valid_targets": dev_total,
        "entropy_pi": entropy,
        "train_pi_dev_nll": pi_dev_nll,
        "softmax_log_pi_max_abs_error": pi_error,
        "softmax_log_pi_atol": diagnostic["softmax_prior_atol"],
        "prepared_artifact_sha256": sha(prepared_path),
        "codes_artifact_sha256": sha(code_path),
        "scope": "one constant, context-free distribution; not a contextual language-model NLL lower bound",
        "fit_uses_dev": False,
        "rank_diagnostic": "Not computed; unnecessary for the specified constant-distribution fit",
        "fits": {},
    }

    def save() -> None:
        result["elapsed_seconds"] = time.perf_counter() - started
        write_json(output_path, result)

    with event_path.open("x", encoding="utf-8", buffering=1) as events:
        def event(kind: str, **values) -> None:
            row = {"utc": utc(), "event": kind, **values}
            events.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

        event("diagnostic_start", device="cpu", dtype="float64", configuration=diagnostic,
              entropy_pi=entropy, train_pi_dev_nll=pi_dev_nll,
              training_valid_targets=int(counts.sum()), dev_valid_targets=dev_total)
        save()
        for family, table in tables.items():
            fit_start = time.perf_counter()
            if time.time() >= deadline:
                result["fits"][family] = {"status": "NOT_RUN", "stop_reason": "SHARED_CPU_DEADLINE"}
                result["status"] = "PARTIAL"
                save()
                continue
            a = torch.zeros(width, dtype=torch.float64, device="cpu", requires_grad=True)
            optimizer = torch.optim.LBFGS(
                [a], lr=1.0,
                max_iter=diagnostic["lbfgs_max_iter"], max_eval=diagnostic["max_eval"],
                history_size=diagnostic["history_size"],
                tolerance_grad=diagnostic["tolerance_grad"],
                tolerance_change=diagnostic["tolerance_change"],
                line_search_fn=diagnostic["line_search"],
            )
            closure_calls = 0

            def objective() -> torch.Tensor:
                logits = table @ a
                return torch.logsumexp(logits, dim=0) - pi.dot(logits)

            def inspect_point() -> dict:
                value = objective()
                gradient, = torch.autograd.grad(value, a)
                with torch.no_grad():
                    logits = table @ a
                    log_q = logits - torch.logsumexp(logits, dim=0)
                    q = log_q.exp()
                    hand_gradient = table.T @ (q - pi)
                    gradient_error = float((gradient - hand_gradient).abs().max())
                    assert gradient_error <= 1e-10, "Analytic/autograd gradient mismatch"
                    assert torch.isfinite(value) and torch.isfinite(gradient).all()
                    gap = float(value.detach()) - entropy
                    assert gap >= -1e-10, "Negative KL residual exceeds floating tolerance"
                    return {
                        "objective": float(value.detach()), "L_minus_Hpi": gap,
                        "gradient_inf": float(gradient.abs().max()),
                        "gradient_l2": float(torch.linalg.vector_norm(gradient)),
                        "analytic_autograd_max_abs_error": gradient_error,
                        "dev_nll": float(-(dev_weights * log_q).sum()),
                        "a_l2": float(torch.linalg.vector_norm(a)),
                    }

            try:
                check_deadline()
                initial = inspect_point()
            except DeadlineReached:
                result["fits"][family] = {"status": "NOT_RUN", "stop_reason": "SHARED_CPU_DEADLINE"}
                result["status"] = "PARTIAL"
                save()
                continue
            assert a.detach().count_nonzero() == 0
            with torch.no_grad():
                zero_with_prior_error = float((torch.softmax(table @ a + log_pi, dim=0) - pi).abs().max())
            assert zero_with_prior_error <= diagnostic["softmax_prior_atol"]
            event("fit_start", family=family, initialization="zero_vector", initial=initial,
                  zero_plus_log_pi_max_abs_error=zero_with_prior_error)

            def closure() -> torch.Tensor:
                nonlocal closure_calls
                check_deadline()
                optimizer.zero_grad(set_to_none=True)
                value = objective()
                assert torch.isfinite(value), "Nonfinite constant-fit objective"
                value.backward()
                assert a.grad is not None and torch.isfinite(a.grad).all()
                closure_calls += 1
                state = optimizer.state.get(a, {})
                event("closure", family=family, closure=closure_calls,
                      optimizer_iteration=int(state.get("n_iter", 0)), objective=float(value.detach()),
                      gradient_inf=float(a.grad.abs().max()),
                      gradient_l2=float(torch.linalg.vector_norm(a.grad)),
                      fit_elapsed_seconds=time.perf_counter() - fit_start)
                return value

            fit_status = "COMPLETED"
            exception = None
            try:
                optimizer.step(closure)
            except DeadlineReached as error:
                fit_status = "PARTIAL"
                exception = str(error)
            except BaseException as error:
                fit_status = "FAILED"
                exception = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}

            # A short final audit/save is allowed within the supervisor cleanup reserve.
            try:
                final = inspect_point()
            except BaseException as error:
                final = {}
                fit_status = "FAILED"
                exception = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            state = optimizer.state.get(a, {})
            iterations = int(state.get("n_iter", 0))
            evaluations = int(state.get("func_evals", 0))
            gradient_converged = bool(final) and final["gradient_inf"] <= diagnostic["convergence_grad_inf"]
            if fit_status == "PARTIAL":
                stop_reason = "SHARED_CPU_DEADLINE"
            elif fit_status == "FAILED":
                stop_reason = "NUMERICAL_OR_RUNTIME_ERROR"
            elif final["gradient_inf"] <= diagnostic["tolerance_grad"]:
                stop_reason = "GRADIENT_TOLERANCE"
            elif iterations >= diagnostic["lbfgs_max_iter"]:
                stop_reason = "MAX_ITERATIONS"
            elif evaluations >= diagnostic["max_eval"]:
                stop_reason = "MAX_EVALUATIONS"
            else:
                stop_reason = "OTHER_LBFGS_TOLERANCE_TERMINATION_EXACT_BRANCH_NOT_EXPOSED"
            a_path = PRIVATE / f"constant_fit_a_{family}.pt"
            torch.save({"a": a.detach(), "family": family, "dtype": "float64"}, a_path)
            fit = {
                "status": fit_status, "stop_reason": stop_reason,
                "iterations": iterations, "optimizer_function_evaluations": evaluations,
                "closure_calls_logged": closure_calls,
                "initial": initial, **final,
                "entropy_pi": entropy, "train_pi_dev_nll": pi_dev_nll,
                "gradient_convergence_threshold": diagnostic["convergence_grad_inf"],
                "gradient_threshold_met": gradient_converged,
                "residual_interpretation": "approximate_fitted_optimal_gap" if fit_status == "COMPLETED" and gradient_converged else "current_optimization_residual",
                "convergence_note": "Convex objective with an independently recomputed first-order gradient; gradient tolerance is not a certified bound on distance to the exact optimum",
                "zero_plus_log_pi_max_abs_error": zero_with_prior_error,
                "elapsed_seconds": time.perf_counter() - fit_start,
                "a_private_path": str(a_path.relative_to(ROOT)), "a_sha256": sha(a_path),
            }
            if exception is not None:
                fit["exception"] = exception
            result["fits"][family] = fit
            result["status"] = "PARTIAL" if fit_status != "COMPLETED" else "RUNNING"
            save()
            event("fit_end", family=family, **fit)
            print(json.dumps({"event": "fit_end", "family": family, **fit}, ensure_ascii=False, allow_nan=False), flush=True)
            if fit_status == "FAILED":
                raise RuntimeError(f"{family} constant fit failed; partial evidence preserved")

        result["status"] = "COMPLETED" if all(f.get("status") == "COMPLETED" for f in result["fits"].values()) and len(result["fits"]) == 2 else "PARTIAL"
        result["end_utc"] = utc()
        save()
        event("diagnostic_end", status=result["status"], elapsed_seconds=result["elapsed_seconds"])
        print(json.dumps({"event": "diagnostic_end", "status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
