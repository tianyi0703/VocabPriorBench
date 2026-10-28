# V0.2 raw-results audit

Status: **PASS**. Read-only arithmetic audit of the original JSON/JSONL evidence. No GPU, torch, tensor/checkpoint, corpus, or holdout access.

Eight completed trajectories have 767 consecutive optimizer steps each, 508 valid targets per step, and 389,636 training targets each: **6,136 steps / 3,117,088 update targets** total. All **24** evaluations are at steps 0/100/767 and use the full **62,601** dev targets (1,502,424 repeated dev-target evaluations). Raw metrics, evaluation events, trajectory resources, and the profile agree.

All recorded frozen input/output/prior hashes are identical before and after training; each matches its CPU input proof and table metadata. Common-state and order hashes match the original per-seed anchors across all four conditions. These are checks of recorded evidence; this audit does not reopen or independently hash model tensors.

C↔V0.1 Fp and S↔V0.1 Sp: **12/12** raw NLL and NLL-sum pairs are exactly equal at steps 0/100/767. Maximum absolute bridge error: **0.0 nats/token** (predeclared tolerance 0.001).

Final raw dev NLL:

| Seed | C | U | RS | S |
|---|---:|---:|---:|---:|
| 1729 | 6.274890513258699 | 6.273991691340855 | 6.454264401890607 | 6.443508820995271 |
| 2718 | 6.269336835657952 | 6.268255440614217 | 6.45421161512371 | 6.443635412322125 |

Signed endpoint contrasts (nats/token):

| Seed | G | D_r | D_1 | N_cont | N_sign | I |
|---|---:|---:|---:|---:|---:|---:|
| 1729 | 0.1686183077365726 | 0.1793738886319085 | 0.169517129654416 | -0.0008988219178434065 | -0.010755580895335903 | -0.009856758977492497 |
| 2718 | 0.1742985766641736 | 0.18487477946575837 | 0.1753799717079083 | -0.001081395043734723 | -0.010576202801584778 | -0.009494807757850054 |

Classification: **DIRECTION_RESIDUAL_AT_BOTH_NORMS** at tau=0.05, applying A→B→C→D priority. Both seeds have D_r>tau and D_1>tau, so B passes without adding a G requirement. A fails both direction thresholds and both positive norm-effect thresholds; C fails abs(D_r)<=tau. All subtraction identities have exact floating-point residual 0.0. Full criterion booleans and all 12 bridge pairs are in the companion JSON.

One GPU process exits 0; no OOM/error is recorded. External process time is **156.68108119996032 s**, containing runner time **153.37049619999016 s**. Adding previous project **332.12303610004625 s** gives **488.80411730000657 s**. The new-process total is below the 1,800 s cap. Per-trajectory timings are short observations, without sustained thermal or energy evidence.

The evidence supports the prescribed two-seed, one-epoch direction-residual screen. Small norm/interaction contrasts are not equivalence tests; row scaling also changes the score subspace. These logs do not establish convergence, causal mediation, novelty, generalization, low-bit storage savings, or compression speedup.

Checks passed: 198/198. Failures: []. Sources remained byte-identical during this audit; their bytes/SHA-256 values are retained in the JSON. Only the two new RAW_RESULTS_AUDIT files were written.

NEXT: **PAUSE_CURRENT_CLAIM_FOR_RESEARCH_DECISION**.
