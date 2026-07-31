# Assembly obligations — debts the epoch-verifier assembly must discharge

Started 2026-07-31. Each entry is a deferral whose safety argument is still
owed (standing-decisions method rule 5). Assembly (RESUME item 5) may not be
called done while any entry is OPEN. Add entries as legs flag them; close an
entry only with the verifying evidence named in it.

## OPEN

1. **`reg_fini` felt-width gap** (flagged by reg-tree, slice 1).
   Production's `reg_fini` is `Vec<u32>` — the TYPE is the entire
   enforcement. An LFM arena is untyped felts, so the machine's accepted set
   is wider than production's. Guard test
   `the_derivation_extends_a_non_u32_register_value_demonstrating_hazard`
   asserts the gap still exists. Assembly owes ONE of:
   - a 67-per-column range check on the register boundary columns, OR
   - the verified argument that no epoch proof can exist over a >u32
     register column (plausible via REG-C2's Memory-bus value word —
     currently UNVERIFIED; verifying it means a coherent-forgery analysis
     per method rule 4, not an assertion).
   Default is the range check: if assembly arrives and the argument is
   still unverified, emit the check.

## STATED DEFERRALS (safety argument given and accepted — not open debts)

- **`coset_offset ≠ 3` is unexercised in the FRI leg** (reg-tree, FRI
  slice 0; accepted 2026-07-31). No production config produces another
  value and the domain constants are baked into the program, so the
  emitter's handling of a different offset has no witness. Safety argument:
  a wrong coset offset moves every domain point and therefore every leaf
  and every fold — it can only REJECT proofs (honest ones included), never
  accept a forgery, in either direction of the error. Residual plumbing
  condition: the emitter must derive the baked constants from
  `ProofOptions`' offset (or assert its literal against that source of
  truth), so the deferral covers test coverage only, not a hardcoded-3
  emitter.

## WATCH (anomalies assembly should confirm or explain, not obligations)

- **HALT's constraint-leg cost line is out of step**: 9,859 instructions
  for 22 columns, inconsistent with its neighbours (deep-join, final
  report — noticed, not chased). Assembly composes per-AIR numbers; an
  unexplained per-AIR outlier is exactly where a composition error would
  hide.

## STANDING (from the RESUME, restated so this file is self-contained)

- Every per-epoch number so far is a COMPOSITION of per-AIR measurements,
  not a run. Assembly is what confirms or falsifies them.
- The arena-value join obligation — WIDENED 2026-07-31 (deep-join, LogUp
  scoping): it is NOT about opened values; it is about **any value two legs
  consume**. Opened values were merely the first instance. Every such value
  must be one arena cell (or derived in-machine from one), never parallel
  copies, one per leg.
  - Instance 1, DISCHARGED: constraint/DEEP values = authenticated values
    (deep-join slice 1, shared cells + bound index).
  - Instance 2, found live in a leg considered DONE: the constraint leg
    hinted `table_offset = L/N` host-side and the machine never saw `L` —
    a prover could satisfy every accumulator with truthful `L₁/N` while the
    closure sums arbitrary `L₂`, making bus balance vacuous. Fix in flight
    (deep-join): machine reads `L`, derives `L/N` in-machine as `L · N⁻¹`
    (N is shape, so `N⁻¹` is a program constant); the closure sums the same
    `L` cell. This entry closes when that fix + its split-L control test
    are merged.
  - Lesson for assembly: a hinted arena word that a differential never
    catches (because the host packs it truthfully) is exactly where this
    class hides. Audit every remaining hinted word against the two-consumer
    rule before assembly is called done.
