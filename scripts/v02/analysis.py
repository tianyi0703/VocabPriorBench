"""Pure arithmetic for the predeclared V0.2 norm/direction screen.

This module imports nothing and performs no file I/O. Evidence validity,
trajectory completeness and anchor-bridge status are gates owned by the
caller; call ``classify`` only after those gates permit a mechanism label.
"""


LABEL_A = "NORM_DOMINANT_IN_THIS_SCREEN"
LABEL_B = "DIRECTION_RESIDUAL_AT_BOTH_NORMS"
LABEL_C = "INTERACTION_OR_COMPENSATION"
LABEL_D = "INCONCLUSIVE_OR_OTHER_PATTERN"
PRIORITY = (LABEL_A, LABEL_B, LABEL_C, LABEL_D)
CONTRAST_NAMES = ("G", "D_r", "D_1", "N_cont", "N_sign", "I")


def _finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + " must be a finite number") from exc
    if not -float("inf") < number < float("inf"):
        raise ValueError(name + " must be a finite number")
    return number


def contrasts(mapping):
    """Return signed contrasts from a mapping of C/U/RS/S dev NLL values.

    Identity errors are signed floating-point arithmetic residuals. They are
    diagnostics of subtraction identities, not a causal mediation analysis.
    No clipping, rounding, absolute-value conversion or epsilon is applied.
    """
    try:
        values = {name: _finite(mapping[name], name) for name in ("C", "U", "RS", "S")}
    except (KeyError, TypeError) as exc:
        raise ValueError("mapping must provide C, U, RS and S") from exc
    C, U, RS, S = (values[name] for name in ("C", "U", "RS", "S"))
    result = {
        "G": S - C,
        "D_r": RS - C,
        "D_1": S - U,
        "N_cont": U - C,
        "N_sign": S - RS,
    }
    result["I"] = result["D_1"] - result["D_r"]
    result["identity_errors"] = {
        "G_minus_sum_N_sign_D_r": result["G"] - (result["N_sign"] + result["D_r"]),
        "G_minus_sum_D_1_N_cont": result["G"] - (result["D_1"] + result["N_cont"]),
        "I_minus_difference_N_sign_N_cont": result["I"] - (result["N_sign"] - result["N_cont"]),
    }
    result["max_abs_identity_error"] = max(abs(x) for x in result["identity_errors"].values())
    return result


def _criterion_record(criteria, explanations):
    unmet = [explanations[name] for name, passed in criteria.items() if not passed]
    return {"criteria": criteria, "passed": all(criteria.values()), "unmet_reasons": unmet}


def classify(final_rows, tau=0.05):
    """Classify exactly two final rows, each carrying seed and six contrasts.

    ``final_rows`` is a sequence of mappings with ``seed``, ``G``, ``D_r``,
    ``D_1``, ``N_cont``, ``N_sign`` and ``I``. Reject missing, duplicate or
    nonfinite inputs instead of treating an incomplete comparison as a result.
    The caller must independently verify that these are the valid step-767
    rows and that the evidence/bridge gates pass.

    Thresholds use the exact prescribed > and <= relations, with no tolerance
    added to tau. B and C deliberately have no additional requirement on G.
    """
    tau = _finite(tau, "tau")
    if tau < 0:
        raise ValueError("tau must be nonnegative")
    rows = list(final_rows)
    if len(rows) != 2:
        raise ValueError("Exactly two independent seed rows are required")
    per_seed = []
    seed_keys = set()
    for row in rows:
        try:
            seed = row["seed"]
            values = {name: _finite(row[name], name) for name in CONTRAST_NAMES}
        except (KeyError, TypeError) as exc:
            raise ValueError("Every final row must provide seed and all six contrasts") from exc
        if not isinstance(seed, (int, str)) or isinstance(seed, bool):
            raise ValueError("seed must be an integer or string identifier")
        seed_key = str(seed)
        if seed_key in seed_keys:
            raise ValueError("Duplicate seed rows are not independent repeats")
        seed_keys.add(seed_key)
        A = _criterion_record(
            {
                "G_gt_tau": values["G"] > tau,
                "abs_D_r_le_tau": abs(values["D_r"]) <= tau,
                "abs_D_1_le_tau": abs(values["D_1"]) <= tau,
                "N_cont_gt_tau": values["N_cont"] > tau,
                "N_sign_gt_tau": values["N_sign"] > tau,
            },
            {
                "G_gt_tau": "A requires G > tau",
                "abs_D_r_le_tau": "A requires abs(D_r) <= tau",
                "abs_D_1_le_tau": "A requires abs(D_1) <= tau",
                "N_cont_gt_tau": "A requires N_cont > tau",
                "N_sign_gt_tau": "A requires N_sign > tau",
            },
        )
        B = _criterion_record(
            {"D_r_gt_tau": values["D_r"] > tau, "D_1_gt_tau": values["D_1"] > tau},
            {"D_r_gt_tau": "B requires D_r > tau", "D_1_gt_tau": "B requires D_1 > tau"},
        )
        per_seed.append({"seed": seed, "values": values, "A": A, "B": B})
    all_A = all(row["A"]["passed"] for row in per_seed)
    all_B = all(row["B"]["passed"] for row in per_seed)
    for row in per_seed:
        values = row["values"]
        row["C"] = _criterion_record(
            {
                "abs_D_r_le_tau": abs(values["D_r"]) <= tau,
                "D_1_gt_tau": values["D_1"] > tau,
                "A_not_satisfied_across_both_seeds": not all_A,
            },
            {
                "abs_D_r_le_tau": "C requires abs(D_r) <= tau",
                "D_1_gt_tau": "C requires D_1 > tau",
                "A_not_satisfied_across_both_seeds": "C requires that the complete A rule did not pass",
            },
        )
    all_C = all(row["C"]["passed"] for row in per_seed)
    label = LABEL_A if all_A else LABEL_B if all_B else LABEL_C if all_C else LABEL_D
    return {
        "label": label,
        "tau_nats_per_token": tau,
        "priority": list(PRIORITY),
        "per_seed": per_seed,
        "all_seeds_predicates": {"A": all_A, "B": all_B, "C": all_C},
        "unmet_reasons_by_rule": {
            rule: [
                {"seed": row["seed"], "reasons": row[rule]["unmet_reasons"]}
                for row in per_seed if not row[rule]["passed"]
            ] for rule in ("A", "B", "C")
        },
        "additional_G_gate_for_B_or_C": False,
        "scope": "two-seed exploratory screen; no significance, equivalence, novelty or causal mediation claim",
    }
