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

2. **`start_index` is unbound to the chain** (flagged by deep-join, LogUp
   closure slice 1). The COMMIT-bus target reads `start_index` (the carried
   x254) as arena data; nothing yet binds it to the previous epoch's output
   length. A chaining obligation of the same family as the L2G root binding
   and the REGISTER derivation: assembly (or a dedicated chaining slice)
   owes the binding, and no binding should be invented without reading how
   production carries it across epochs.

3. **Assembly must unify the five remaining two-consumer values** (deep-join
   audit, 5e93fe6d). Each is hinted twice today — not exploitable while the
   legs are separate programs, every one a landmine the moment they share an
   arena. Unification means deciding the assembled program's arena layout,
   which is assembly's call — that is WHY they were not fixed leg-side:
   - the OOD frame values (constraint eval vs DEEP invariants, `ood_steps`);
   - the claimed composition parts at `z` (constraint quotient vs DEEP
     `h_sum_zpow`);
   - `ζ` (constraint zerofier vs DEEP `row_points`/`z_pow`);
   - the main-trace roots (Phase A absorb vs authentication root compare);
   - the public output bytes (attestation `program_id` fold vs COMMIT-bus
     target).

4. **The challenges guard cited in comments does not exist yet.** Write
   `challenges_are_not_an_arena_in_the_assembled_verifier` once the
   assembled verifier exists: raw challenges (z, α, ζ, per-table forks)
   must come from `TranscriptReplay`, never from `Instr::Hint` arena words.
   Until then the per-slice differential programs hint them as a documented
   shortcut (`constraint_tests.rs` `differential_program` doc comment, which
   previously cited this guard as if it existed — corrected 2026-07-31).

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

- **The COMMIT-bus target is an unbudgeted per-byte cost item**: the
  closure's second half is `Σ 1/(z − fingerprint(byte_i))` over public
  output BYTES — one inverse chain per byte, scaling with output length
  (deep-join, LogUp slice 1). Not in the target-shape budget. Assembly
  must price it against the real epoch's public-output length.
  First data point (zerorow, 2026-08-03): the fixture epoch's output is
  **8 bytes**, so the gadget is nonempty in practice and the empty
  short-circuit is not the common case. That is a 16-cycle epoch and says
  nothing about a production epoch's output length — it only rules out
  "usually zero".
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
  - Instance 2, found live in a leg considered DONE — now DISCHARGED
    (deep-join 6712b814, merged 94a55e17): the constraint leg hinted
    `table_offset = L/N` host-side and the machine never saw `L` — a
    prover could satisfy every accumulator with truthful `L₁/N` while the
    closure sums arbitrary `L₂`, making bus balance vacuous. Fixed by
    in-machine derivation (`emit_table_offset`: `L · N⁻¹`, N⁻¹ a program
    constant); the closure sums the same `L` cell. Falsified both ways:
    4 forged contributions rejected by the derivation and accepted by a
    split control, and a pass-through stub fails exactly the composition
    check + join test.
  - Instance 3, found by the audit the L gap triggered — DISCHARGED
    (deep-join 5e93fe6d, merged 1418e0b7): `alpha_powers` were hinted, one
    arena word each, and `Op::AlphaPow{idx}` read them straight. Every
    LogUp fingerprint is built from these powers, so a prover supplying
    them independently of α chooses the fingerprints — any tuple can match
    any other; strictly worse in degree than the L gap. Fixed by chaining
    from the one α the challenges carry (`emit_alpha_powers`, one ExtAlu
    per power, count = `max_bus_elements`, which is shape). Guarded by an
    absolute rule-7-compliant test (`the_derived_uniforms_are_not_arena_words`)
    with positive controls, both negative branches falsified independently.
  - Lesson for assembly: a hinted arena word that a differential never
    catches (because the host packs it truthfully) is exactly where this
    class hides. The full audit is done (5e93fe6d): everything else either
    discharged or in OPEN entry 3.
