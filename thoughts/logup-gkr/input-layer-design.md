# Linear input layer for LogUp-GKR (the leaf-binding fix, shape "2b")

> Design for closing the multi-interaction leaf-binding gap (port-plan.md §6)
> by extending each table's summation tree down to per-interaction leaves —
> the Papini–Hàbock LogUp-GKR shape. Acceptance test:
> `reconstruct_multi_interaction_rejects_fabricated_leaf_claims` un-ignored
> and passing.

## The change in one paragraph

Today each table is ONE GKR instance over N leaves, where leaf i is the
cross-multiplied fraction `Σ_k ±m_k(i)/fp_k(i)` — a degree-K function of the
trace columns, which is why the verifier cannot check the leaf claims
(fail-open). The fix: extend the tree by `log2(K̂)` layers (K̂ = K padded to a
power of two), so the instance has `K̂·N` leaves indexed `(i, k)` with leaf
value `(±m_k(i), fp_k(i))` for k < K and the fraction identity `(0, 1)` for
padding. Leaves are now LINEAR in the trace columns, so the verifier
reconstructs the final claims EXACTLY from the column claims — no new proof
fields, no rational cross-check special cases.

## Fact 1: the N-sized layer is bit-identical to today's tree

Fraction-pair addition `(a,b)+(c,d) = (ad+cb, bd)` is associative at the PAIR
level (not just as rationals): any association order over
`[f_1..f_K, (0,1)…]` yields the identical `(n, d)` pair, and `(0,1)` is the
identity. Therefore the extended tree's layer at size N — the balanced
pairwise sum over each row's K̂ interaction leaves — equals EXACTLY the output
of today's `compute_logup_leaf_fractions` (a sequential fold). Consequence:

- Every layer from size N up to the root is UNCHANGED — same values, same
  materialized `gen_layers` tree, same batch sumcheck code, same memory.
- The root (bus balance) is unchanged → the cross-mode oracle
  (`gkr_root_matches_standard_table_contribution`) still holds as-is.
- The ONLY new prover work is the `log2(K̂)` deep layers between K̂·N and N.

## Fact 2: variable ordering — interaction bits LOW

Leaf index = `i·K̂ + k` (k in the low bits). Then:

- The deep layers pair adjacent k's, so the tree "absorbs" the interaction
  sum first and reaches today's per-row fractions at size N (Fact 1 needs
  this ordering).
- The final input-layer evaluation point splits as `(κ, ρ)`: κ = the low
  `log2(K̂)` coordinates (interaction bits), ρ = the high `log2(N)`
  coordinates (row bits).
- **The bridge is untouched.** Column claims remain row-MLEs `⟨l, col⟩` at
  the ROW point ρ; the Lagrange-kernel/σ columns, their constraints, and the
  aux layout stay exactly as shipped. κ never touches committed data — it
  only enters the verifier's reconstruction weights (Fact 3).
- Bookkeeping: instance n_vars becomes `log2(N) + log2(K̂_t)` (varies per
  table by K); the batch already handles mixed sizes. `instance_eval_point`
  yields the full (κ, ρ) point; ρ = its last `log2(N)` coordinates feeds the
  kernel/bridge exactly where the whole point used to.

## Fact 3: the verifier check becomes exact linear reconstruction

At the input layer the claims `(n̂, d̂)` are MLE evaluations of the leaf
vectors at `(κ, ρ)`. Both vectors are multilinear in (k-bits, columns):

```
d̂ = Σ_{k<K} eq(κ, bits(k)) · (z − bus_k − Σ_e α^e · ĉ_{col(e)})   +  Σ_{k≥K} eq(κ, bits(k)) · 1
n̂ = Σ_{k<K} eq(κ, bits(k)) · sign_k · m_k(ĉ)                      +  0
```

where `ĉ_j` are the column claims at ρ (bound to the trace by the bridge) and
`m_k(ĉ)` is the multiplicity's linear form evaluated on claims. The verifier
computes both sums (O(K·bus_elements) field ops — trivial, guest-friendly)
and checks EXACT equality with the transcript-derived claims. This replaces
`reconstruct_and_verify_gkr_claims`' three-way branch (K=1 exact / 0-layer
exact / else fail-open) with one uniform exact check. Padding contributes the
explicit `+ eq·1` denominator terms and nothing to the numerator.

Soundness: n̂, d̂ are bound by the GKR sumcheck chain to the root (bus
balance); ĉ are bound to the committed trace by the bridge constraint; the
reconstruction ties the two together with no free variables left. The
fail-open branch — and the entire gap — is gone.

## Prover: the deep layers

Only the `log2(K̂)` layers below size N are new. Two implementation stages:

**Stage 1 (correctness): materialized deep layers.** Build the input layer
(K̂N pairs; numerators base-field, `Multiplicity::One` numerators implicit)
and fold up to N materializing each layer; run the existing batch sumcheck
over all layers. Transient memory for the deep part is O(K̂N) ext elements on
the biggest tables (CPU K=40, N≈2^22 → several GB) — acceptable for
validating the protocol end-to-end and running the soundness/forgery gates,
NOT the shipping shape.

**Stage 2 (optimal): virtual deep layers.** Exploit linearity:
- Round 0 of each deep layer evaluates child entries on the fly from raw
  trace columns (O(size) fingerprint evals; the per-row K̂-range partial sums
  are computed in one pass per row).
- Rounds that bind κ bits (at most `log2(K̂)` ≤ 6 of them): eq-weighted
  combinations over bound interaction bits, evaluated per row from the ≤ K̂
  per-row fractions (recomputed per chunk; O(K̂) per row per round).
- Once all κ bits are bound, the partially-folded tables are multilinear in
  the trace columns, so they collapse to "fingerprints of FOLDED columns":
  maintain the C distinct referenced columns folded by the row challenges
  (fold-in-place, O(C·N) memory total, shared across all K interactions) and
  evaluate table entries from them. From this point the layer folds like any
  materialized layer of size ≤ N.
- Peak memory returns to today's footprint (the ≥N tree + O(C·N) folded
  columns); prover time adds O(K̂·N) fingerprint-level work per deep layer
  round for the κ rounds — same order as today's leaf building, now inside
  the protocol (and the OLD leaf cross-multiplication cost disappears from
  `compute_logup_leaf_fractions`; Fact 1's N-layer is still needed, but it
  is the same balanced sum either way).

Stage 1 lands first with all gates; Stage 2 replaces the deep-layer engine
behind the same interface, gated by the same tests + an ABBA-style A/B on
the box (time and peak heap vs Stage 1 and vs the pre-fix branch).

## What changes where

- `crypto/stark/src/gkr.rs`: instances grow `input_num_vars = log2(K̂)`
  metadata; `gen_layers` gains the deep extension (Stage 1) / the batch loop
  gains the virtual deep-layer path (Stage 2). Layer count per instance
  becomes `log2(N) + log2(K̂)`.
- `crypto/stark/src/logup_gkr.rs`: `compute_logup_layers` builds the extended
  instance; `reconstruct_and_verify_gkr_claims` becomes the exact linear
  reconstruction (needs the interaction list + α powers + κ — all present);
  `instance_eval_point` consumers split (κ, ρ) and pass ρ to the bridge
  parameter derivation. `n_layers_by_instance` on the verifier becomes
  `log2(N) + log2(K̂)` (K̂ derived from the AIR's interactions — NOT from the
  proof).
- Bridge, aux columns, rap-challenge layout, proof wire format: UNCHANGED
  (the random-point section of the challenge vector now carries ρ only —
  same length `log2(N)` as today).
- Tests: un-ignore the forgery test (make it exercise K>1, n_layers>0 and
  expect rejection); cross-mode oracle unchanged; bridge parity unchanged;
  all e2e suites (monolithic, recursion, continuation) re-run — wire format
  changes only in the transcript (more sumcheck layers), so GKR-mode proofs
  from before this change will NOT verify (fine: experimental mode, no
  compatibility promise).

## Costs (estimates to validate on the box)

- Proof size: +`log2(K̂)` layers per table ≈ +6 layers × (rounds × 4 evals +
  4 child claims) for the big tables — hundreds of KB total, ≈ noise
  against the −12 % we currently have.
- Guest cycles: the extra layers add sumcheck replay + the K-term
  reconstruction per table — small vs the −3.3 % headroom; measure.
- Prover: Stage 1 expected somewhat slower + more transient memory than
  today; Stage 2 expected ≈ today or better (leaf cross-multiplication was
  the single largest GKR phase and it dissolves into cheaper on-the-fly
  evals).
