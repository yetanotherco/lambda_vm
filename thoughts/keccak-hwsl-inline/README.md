# KECCAK_RND: inline HWSL shifts as linear identities — formal verification + bench

This directory contains the SMT (z3) verification and bench evidence behind the
change in this branch: replacing the 120 HWSL bus sends per KECCAK_RND row
(θ rotate-by-1: 20, ρ shifts: 100) with μ-gated degree-2 linear identities

```
in · 2^rnc = right · 2^16 + left
```

over the same committed cells, keeping all existing IS_BYTE (AreBytes) and
IS_BIT checks. Spec: `spec/src/keccak_round_hwsl_inline.toml` (alternative to
`spec/src/keccak_round.toml`).

## Why it is sound

A shift by a compile-time constant is linear over the field. Given
`left ∈ [0, 2^16)` and `right ∈ [0, 2^16)` (θ: `right ∈ {0,1}`), the pair
`(right, left)` is the unique Euclidean quotient/remainder of
`in · 2^rnc ÷ 2^16`; all involved values are `< 2^32 ≪ p`, so field semantics
coincide with the integers. The range checks are load-bearing: without them
`2^16` is invertible mod p and the decomposition is ambiguous.

## What was machine-checked (z3, QF_BV + Int-mod-p)

Model: every trace column a free variable; each bus interaction under its
chip's contract and each eval constraint becomes an equation; assert
`output ≠ FIPS-202 reference round` and ask for a counterexample.

Baseline gate (`z3_verify.py`, `par.log`):
- Original circuit: **UNSAT for all 24 round indices** (round wiring correct
  given the chip contracts).
- Positive control: constraints uniquely pin the output to the reference
  (the UNSATs are not vacuous).
- Negative controls: 9 injected bugs (dropped θ rotate, swapped ρ offsets,
  mangled χ ×2, dropped ι RC, wrong RC, ρ off-by-one, dropped χ equation,
  dropped HWSL carry pinning) — **all SAT** (each detected).
- Reference independently anchored: FIPS-202-generated RC/RHO equal the
  repo constants; concrete mirror reproduces keccak-f over randomized runs
  (`test_ref.py`, `test_dataflow.py`).

Rewrite gate (`hwsl_inline_test.py`, `hwsl_inline.log`):
- HWSL contracts replaced by the linear identities + only the range checks
  that exist in the circuit: **UNSAT for all 24 rounds** (equivalent).
- Bound necessity, in a genuine Int-mod-p model (bitvectors cannot test this:
  2^16 is a zero divisor mod 2^64 but invertible mod p):
  dropping a range bound → **SAT** (ambiguous decomposition);
  dropping IS_BIT → **SAT** (the θ carry becomes forgeable).
- Coverage audit: every left/right halfword's 16-bit bound comes from an
  existing circuit check (AreBytes on `Cxz_left`/`rot_left`/`rot_right`,
  IS_BIT on `Cxz_right`) — zero new sends needed.

Scope: one round's transition given the precomputed-chip contracts
(BITWISE is a fully enumerated 2^20-row preprocessed table). Multiplicity
gating across padding rows degrades identically for the identity and the
old lookup (both ×μ). The model is hand-transcribed from
`keccak_rnd.rs::bus_interactions`; mitigations above (controls + concrete
mirror), long-term fix is generating the model from the constraint IR.

## Cost accounting

- Sends: 1151 → 1031 per row (HWSL 120 → 0).
- LogUp aux: 2 interactions per committed pair column + 1 accumulator,
  extension degree 3 → −60 extension columns = **−180 committed base
  cells/row** (~4.3k per permutation). Zero new columns.
- Constraints: 20 → 140, all measured degree ≤ 3 (identities are μ×linear =
  degree 2); `max_degree()` unchanged at 3.

## Bench (pure-keccak, single-epoch continuation proof)

Guest: 5000 keccak-f[1600] syscalls (`gen_keccak_bench.sh`). Box: 32 cores /
124 GB. Alternated A/B ×4 after warm-up, prover wall time:

| | runs (s) | median |
|---|---|---|
| main | 13.988 13.948 13.976 14.398 | 13.982 |
| branch | 13.083 12.971 13.334 12.855 | 13.027 |

**−6.8% median (−7.2% mean)**; distributions disjoint (slowest branch run
beats fastest main run; cv ≈ 1.3%). keccak_rnd is 86.6% of committed cells in
this bench; committed-cell delta measured 5.61% (prediction: 5.6%); implied
wall-time table delta ≈ 7.9% — above the cell prediction because the removed
cells are all cubic-extension aux (LDE + Merkle + FRI each). For real
workloads: expected gain ≈ 7.9% × keccak_rnd's share of committed cells.

All proofs verify; a branch proof correctly fails main's verifier (aux layout
shrank — the change is intentionally not wire-identical).

## Files

- `z3_verify.py` — baseline gate: free-var model of the original circuit,
  contracts, 24-round check, positive + negative controls
- `z3_parallel.py`, `par.log` — parallel driver + full baseline run
- `tamper_test.py` — changed-constraint and removed-constraint controls
- `hwsl_inline_test.py`, `hwsl_inline.log` — the rewrite gate (equivalence +
  bound necessity in Int-mod-p)
- `keccak_ref.py`, `test_ref.py` — independent FIPS-202 reference +
  external anchoring (hashlib SHA3, regenerated RC/RHO)
- `model_dataflow.py`, `test_dataflow.py` — concrete byte-level mirror of the
  modeled equations vs the reference
- `gen_keccak_bench.sh` — bench guest generator
