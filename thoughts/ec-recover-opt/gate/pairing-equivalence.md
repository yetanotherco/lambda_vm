# Paired ARE_BYTES rewrite — soundness-preservation argument

The `feat/ec-arebytes-pairing` rewrite repacks the single-byte range-check sends
of ECSM/ECDAS. This note argues the gate's soundness theorem (RESULTS.md) is
preserved verbatim — no lemma re-run is required — and records the layout.

## The change (bus repacking only)

Before: every byte-range check was one `AreBytes` send shaped `[b, 0]`
(196/row in ECDAS, 129/row in ECSM). After: adjacent bytes share one send
`[b_even, b_odd]`:

| Table | Block | Pairing |
|---|---|---|
| ECDAS | LAMBDA, XR, YR, Q0[..32], Q1[..32], Q2[..32] | (2i, 2i+1), i = 0..16 → 96 sends |
| ECDAS | odd bytes ROUND, Q0[32], Q1[32], Q2[32] | (ROUND, Q0[32]), (Q1[32], Q2[32]) → 2 sends |
| ECSM | X2, Q0, YG, Q1[..32] | (2i, 2i+1), i = 0..16 → 64 sends |
| ECSM | odd byte Q1[32] | rides alone as [Q1[32], 0] → 1 send |

Interactions/row: **ECDAS 388 → 290, ECSM 579 → 515** (pinned by
`bus_interaction_counts` in prover/src/tests/ecsm_tests.rs). Constraint counts
(413 / 200), NUM_COLUMNS (667 / 521), and every witness value are UNCHANGED —
the diff touches only `bus_interactions()` in ecsm.rs/ecdas.rs and the
mirrored multiplicity collectors `collect_bitwise_from_{ecsm,ecdas}` in
trace_builder.rs (sends and BITWISE multiplicities move together, in the same
tuple order).

## Why the soundness theorem is unaffected

1. **The contract is unchanged and covers pairs by construction.** Contract C1
   (RESULTS.md) states: each element of an `AreBytes[x, y]` send is in
   [0, 256). Authority: the BITWISE receiver matches BOTH tuple elements
   against the precomputed table's X and Y columns (bitwise.rs:783-796), and
   `generate_bitwise_row` (bitwise.rs:117-156) enumerates the full 2^20 index
   space — every (x, y) with x < 256 ∧ y < 256 exists, and ONLY those. A send
   `[a, b]` therefore matches a table row iff a AND b are bytes. The old
   `[b, 0]` form was already the special case y = 0 of the same contract
   (bitwise.rs:646).
2. **Every gate lemma consumes the contract per column, not per send.** The
   width audit (L2), the value lemmas (L3/L4), and the chain argument (L6)
   assume exactly "each of these named byte columns is in [0, 255]". The set of
   byte columns covered is IDENTICAL before and after (every previously-checked
   byte appears in exactly one paired send). No lemma's hypothesis mentions the
   [b, 0] shape; the only [b, 0] references in the gate were citation comments,
   updated in place (l1_l2_lift.py docstring). Hence no re-run: the models
   never encoded send shapes, only the per-column range hypotheses they induce.
3. **Bus balance is preserved by symmetric edits.** Sender side (chip) and
   multiplicity side (trace_builder collectors) were changed to the SAME pair
   layout and tuple order; the ethrex 5/20-transfer e2e proofs (LogUp balance
   over the whole VM) are the executable check of this.
4. **Cross-block pairs are sound.** (ROUND, Q0[32]) and (Q1[32], Q2[32]) pair
   bytes from different logical operands. The contract constrains each element
   independently — no relation between the two elements is asserted or implied
   by an `AreBytes` row (all 2^16 combinations exist in the table), so pairing
   unrelated bytes adds no coupling.
5. **Multiplicity aggregation is unchanged in kind.** Multiplicities remain
   µ-gated columns; padding rows (µ = 0) still send nothing. The pairs map to
   table row index a + 256·b (+ 0·2^16), all of which exist; the honest
   collector increments exactly those rows.

## Effect on cost (the point of the rewrite)

Aux columns = ⌈interactions/2⌉ ext-3 columns (split_interactions):
- ECDAS: ⌈388/2⌉ = 194 → ⌈290/2⌉ = 145 aux cols ⇒ −49 ext cols = **−147 base
  cells/row** of 1103 committed ⇒ **−13.3% ECDAS committed cells**.
- ECSM: ⌈579/2⌉ = 290 → ⌈515/2⌉ = 258 ⇒ −32 ext cols = −96 base cells/row
  (4 rows/ecrecover — negligible but free).
- Per ecrecover (~1528 ECDAS rows): ≈ **−225k committed base cells (−13%)**,
  plus the matching reduction in LDE/Merkle/FRI work on those aux columns
  (keccak precedent: wall-clock beat the cell prediction because removed cells
  were all cubic-ext aux).
- BITWISE receiver side: unchanged (same static table; multiplicity counts per
  row change, column count doesn't).

## Verification ledger for this rewrite

- `bus_interaction_counts` test pins 515/290.
- ECSM battery 15/15 (incl. `test_prove_elfs_ecsm_forged_result_rejected` and
  `test_prove_elfs_ecsm_forged_ecdas_mu_rejected` — tamper-reject e2e),
  trace_builder 28/28, bitwise 41/41.
- `test_prove_ethrex_5_transfers` (added): real ecrecover workload proves and
  verifies with the pairing (44s). 20-transfer variant likewise (see log).
- Full `-p lambda-vm-prover` suite + `make lint` + `cargo fmt`: see final
  report / CI.
- NOT wire-identical (aux layout changes) ⇒ cross-version verification does
  not apply; the gates above + bench are the evidence, per the keccak
  hwsl-inline precedent.
