# FRI leg — state at reg-tree's retirement (2026-07-31)

Written by team-lead from reg-tree's slice reports. The agent hit its session
limit BEFORE writing its own handoff, so this file substitutes: it records
what the agent reported in messages but never committed as prose. Code state
is fully committed and green; nothing here contradicts the tree.

## Where the leg stands

DONE (all on feat/lfm, slice 1 = 85f99c81):
- `FriShape` mirrors production's `FriFoldLayout` (`crypto/stark/src/fri/
  terminal.rs:45-54`), every parameter taken from `ProofOptions` including
  the coset offset (asserted: `shape.coset_offset == opts.coset_offset` on
  the real proof — no hardcoded 3; the ledger's coset deferral is
  coverage-only, as accepted).
- Host-side unit tests over `k ∈ {0, 6, 7, 63}` and the clamp regime
  (`trace_bits ≤ 7`), expectations hand-derived from the spec, not the
  module. This is where the spec §7 dead-branch requirement is discharged
  (ruling: shape is compile-time in LFM, so those branches are emitter
  arithmetic, not emitted control flow).
- The fixture-blindness result, demonstrated at the worst constant:
  deleting `saturating_sub(1)` from `num_committed` fails both synthetic
  tests and PASSES the real-proof differential (fixture has
  `total_folds = 0`). The only real proof available cannot witness the
  fold mechanism at all.
- The sizing prediction committed AS A TEST (`the_fri_sizing_prediction`)
  before any measurement exists: 174/186/198 perms per query,
  38,106 / 20,460 / 14,454 total at blowup 2/4/8. Blowup-2 row reproduces
  the spec §8 worked example.
- The query-index bits join point: `emit_query_with_bits` /
  `emit_sub_proof_with_bits` return `QueryOutput { deep, bits }`
  (8b8e55bf, fully additive). Guarded by an absolute rule-7 test (every
  returned bit consumed by some `Select`).

NOT STARTED: the emitter itself — per-layer walk + fold + terminal check.

## What the successor needs to know beyond the committed spec

1. **Implementation spec** = `others/lfm-fri-verify-spec.md` INCLUDING its
   addendum (reg-tree's first-hand findings folded in at 85f99c81). The
   emission checklist at the end lists the ten things that silently break
   bit-exactness. Read the whole file before emitting anything.
2. **The parity/Select detail** (verify-side, easy to miss from the
   prover-side reading): leaf ordering is parity-dependent
   (`verifier.rs:637-641` — `if iota % 2 == 1 { [sym, v] } else
   { [v, sym] }`), so the machine must Select on the LOW index bit per
   layer. The FOLD needs no parity branch (spec §3's sign-cancellation
   result); parity matters ONLY for leaf byte order.
3. **⚠ OWED CHECK, never completed:** reg-tree hypothesised that
   `sub_proof::emit_leaf_hash` at `GroupShape { num_columns: 1, is_ext:
   true }` is byte-identical to production's FRI leaf
   (`FieldElementPairBackend::hash_data`, 48 bytes, components 0,1,2, 8B
   big-endian each) and said it would verify byte-for-byte RATHER THAN
   ASSUME. No confirmation ever arrived. The successor must do that check
   before reusing the gadget — treat it as unverified.
4. **Join obligation**: fold values and walk leaves through the SAME cells,
   index bits from `QueryOutput.bits` (never a fresh decomposition), and
   the first fold consumes the DEEP leg's `p₀(υ)/p₀(−υ)` cells — the seam
   the ledger's STANDING clause covers. The zero-layer shape
   (`num_committed = 0`) is a first-class emitted shape, pinned by
   `the_fixture_carries_no_fri_layers_so_it_cannot_witness_the_fold`.
5. **Primary instrument** (approved plan): synthetic codewords driven
   through production's OWN `commit_phase_from_evaluations` + `query_phase`,
   differentialled against the verifier's own check, sweeping
   `num_committed` over 0/1/2/3+. Only the input is synthetic.
6. Measure against the pinned prediction test; a miss means the shape is
   not what we think — investigate, never fudge.
