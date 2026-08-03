# LogUp closure — handoff

Written 2026-07-31 by deep-join, at the end of its context. The leg is
COMPLETE and green; this file exists so the next agent does not have to
re-derive what took the longest to establish. Everything below is first-hand
unless marked otherwise.

**State: `cargo test -p lambda-vm-prover --lib lfm` green (171+), `make lint`
clean, all committed on `feat/lfm-deep-join`.** Files: `prover/src/lfm/logup.rs`
(emitter), `prover/src/lfm/logup_tests.rs` (8 tests), plus
`constraints::emit_table_offset` / `emit_alpha_powers`.

---

## 1. What the closure IS — the part that took longest to establish

Production's check is one block, `crypto/stark/src/verifier.rs:1303-1334`:

```
if needs_lookup_challenges {
    total = Σ over (air, proof) where air.has_trace_interaction()
                                 && proof.bus_table_contribution().is_some()
    if total != *expected_bus_balance { return false }
}
```

Two things about it are not guessable from the name:

**The target is not zero and not a constant.** It is
`compute_expected_commit_bus_balance_view` → `compute_commit_bus_offset`
(`prover/src/lib.rs:909`):

```
expected = Σ_i  1 / (z − (BusId::Commit + (start+i)·α + byte_i·α²))
```

over the public output BYTES. The COMMIT output bus has a receiver the verifier
computes rather than proves, and this is that missing remainder. So half the leg
is a per-byte inverse gadget whose length scales with public output — an
unbudgeted cost item, now on the ledger's WATCH list.

**`L` is NOT an opened value.** It is a proof-carried scalar, absorbed into the
per-table transcript fork (`verifier.rs:1274`). The original charter assumed the
join was to the authentication leg; it is not. What binds `L` to a trace is the
constraint leg — traced end to end:

- `verifier.rs:306` computes `logup_table_offset = L / N`, feeds it to the
  transition context at `:326`;
- `lookup.rs:2260` (`emit_logup_accumulated`) enforces
  `acc_next − acc_curr − Σterms + L/N = 0`;
- `lookup.rs:1346` pins `acc[0] = 0` — note its `_bus_public_inputs` parameter
  is UNUSED, so the boundary is *not* where `L` enters;
- together the accumulator wraps to zero after `N` rows iff `L` is the table's
  true total.

## 2. The gap this leg closed, and the one the audit found after

**Instance 2 (`L`).** `constraint_tests.rs` computed `table_offset = L/N`
host-side and hinted it; the machine never saw `L`. Adding a closure that hinted
`L` would have given the prover two independent words — truthful `L₁/N` so every
accumulator wraps, arbitrary `L₂` so the sum hits the target. Fixed by
`emit_table_offset` (`L · N⁻¹`, `N⁻¹` a program constant since `N` is shape).

**Instance 3 (`alpha_powers`), worse in degree.** They were hinted one word
each, and `Op::AlphaPow{idx}` read them straight. Every LogUp FINGERPRINT is
`z − Σ vⱼ·αʲ`, so a prover choosing the powers chooses the fingerprints and any
tuple can be made to match any other. Fixed by `emit_alpha_powers`, chaining
from the one α.

Both are recorded in `lfm-assembly-obligations.md`. The generalised rule —
**any value TWO legs consume must be one cell** — came out of this leg.

## 3. What is WITNESSED, and how

- COMMIT-bus target vs production's `compute_commit_bus_offset`, 28
  (length, start) combinations including the empty short-circuit. Lengths
  matter: `start` advances *inside* the gadget, so a reset-or-reversed formula
  agrees only at length 1.
- A deliberate fingerprint COLLISION is unprovable (`1/0`), matching
  production's `.ok()?`; the neighbouring non-colliding `z` still folds, so the
  rejection is the collision and not the shape. **This test is mandatory, not
  optional** — the machine's `0/0 = 1` convention means a term written as a
  direct divide would accept exactly what production rejects. Same trap the DEEP
  denominators hit.
- The closure over a real sender/receiver pair whose bus genuinely closes, with
  `multi_verify` at target zero as the oracle (checked FIRST — a fixture whose
  bus did not close would make agreement meaningless). Every single-lane move of
  either contribution rejected.
- The `L` join: a split-arena control accepts 4 coherent forgeries (truthful
  offset so accumulators wrap, forged `L` + matching target so the closure
  balances); the derived shape rejects all 4.
- An ABSOLUTE structural guard (method rule 7):
  `the_derived_uniforms_are_not_arena_words` asserts which cells are
  `Instr::Hint` outputs and which are computed, with positive controls. Immune
  to variant unification. Both negative branches falsified independently — they
  are ordered, so the first masks the second; break them one at a time.
- **Per-CHUNK accumulation** (the degenerate parameter the team lead
  prioritised). `VmAirs::new` builds one AIR per chunk
  (`lib.rs:702`: `(0..table_counts.cpu).map(|i| … CPU[i])`), so a family of `k`
  chunks is `k` sub-proofs and the closure sums per chunk. Witnessed with a
  3-table fixture — one sender, TWO receiver chunks of one family, one lookup
  each. Both halves: the 3-term sum closes, and all three 2-term readings are
  nonzero, plus a closure compiled for 2 tables rejects both chunk drops. On any
  1-chunk-per-family fixture the two readings agree, which is why this needed
  building.
- `has_trace_interaction()` is shape, and production checks the proof's presence
  against it in BOTH directions (`verifier.rs:1238` and `:1244`), so the two can
  never disagree in a proof that verifies. `num_contributing_tables` is
  therefore a program constant; a short arena is rejected.

## 4. What is NOT witnessed — precise statements

- **Zero-row fixed tables.** `T_epoch` includes fixed tables regardless of
  workload (the `FIXED_TABLE_COUNT = 10` lesson), so an epoch carries sub-proofs
  for tables with no real rows. UNVERIFIED first-hand: whether such a table's
  `L` is zero (expected — all-padding rows have multiplicity zero, so every term
  vanishes) and whether it still carries `bus_table_contribution: Some(zero)`
  rather than `None`. It matters only for the COUNT: if a zero-row table reports
  `None`, `has_trace_interaction()` is still true and `verifier.rs:1238` would
  REJECT, so the answer is probably "Some(zero)" — but that is inference from
  the guard, not a measurement. **Next agent: prove one epoch with an unused
  fixed table and read its contribution.** Cheap.
- **Real epoch table-set length.** Every fixture here is 2 or 3 tables. The SUM
  is exercised; its length (twenty-odd for a real epoch) is not.
- **`start_index`** is unbound to the chain — ledger OPEN entry 2. Do not invent
  a binding; read how production carries it across epochs first.
- **The five remaining two-consumer values** — ledger OPEN entry 3. Deliberately
  not fixed leg-side: unifying them means deciding the assembled program's arena
  layout, which is assembly's call.

## 5. Traps for whoever continues

- `open_sub_proof` (constraint_tests) handles the SINGLE-table case only — it
  transcribes `multi_verify_views` without the per-table domain separator. A
  multi-table fixture cannot go through it. That is why the closure fixtures
  read `bus_table_contribution()` off the proof directly rather than replaying.
- `EmptyConstraints` leaves ONE coefficient in the transition run, and
  `open_sub_proof` recovers `beta` from the second (`constraint_tests.rs:972`).
  Any synthetic AIR you want to push through it needs at least one real
  transition constraint. The preprocessed fixture's `CopiedColumn` exists purely
  for this.
- `test_utils::production_airs` builds BITWISE, DECODE, KECCAK_RC, REGISTER and
  PAGE WITHOUT their preprocessed commitments, so `is_preprocessed()` reads
  false on five tables that are preprocessed in a real epoch. Any census taken
  off those objects silently drops an opening group each.
- A doc comment citing a guard test is not evidence the guard exists —
  `challenges_are_not_an_arena_in_the_assembled_verifier` was cited at
  `constraint_tests.rs:165` and had never been written. Ledger OPEN entry 4.
