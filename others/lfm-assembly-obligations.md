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
   closure slice 1). **HALF DISCHARGED** (assembly, slice 3): the reading is
   settled and the CELL is now the right one; the derivation that closes it is
   not built.
   - Settled by research (`lfm-team-lead-start-index-research.md`): production
     has no arithmetic `start + len` check anywhere. It rebuilds epoch N's
     REGISTER preprocessed commitment from epoch N−1's FINI vector and rejects
     unless the proof's root matches, and `verify_epoch` then simply reads
     `register_init[X254_INDEX]` (`continuation.rs:840-851`). Confirmed
     first-hand in the assembly fixture: `compute_expected_commit_bus_balance_view`
     takes `register_init[register::X254_INDEX] as u64`.
   - Done: the assembled spine declares the register-boundary vector as ONE
     arena at production's width and takes `start` from slot 64 of it, so the
     COMMIT-bus target and the future REGISTER derivation read the same cell
     rather than two words. `the_closure_rejects_a_moved_index_or_output`
     moves it by 1, 2 and 7 and the bus fails to close each time.
   - Left: the derivation itself. `start_index` is bound to the chain only
     once Phase A's REGISTER preprocessed root is COMPUTED from that arena
     (reg-tree's emitter) instead of hinted — which is entry 7's work, and
     the two now close together.

3. ~~**Assembly must unify the five remaining two-consumer values**~~ —
   **DISCHARGED** (assembly-w5, slice 1, a1f32859). The legs now hang off the
   spine, so all four staged values have both consumers inside one program and
   there is finally something that could disagree — and nothing does, by
   construction rather than by agreement:
   - the OOD frame values: `epoch_verify::emit_table_verification` rebuilds ONE
     grid with `epoch::emit_reconstruct_ood` and hands the constraint fold and
     the DEEP fold two VIEWS of it (see the new degenerate-parameter note below
     about why they are different views);
   - the claimed parts: the same `absorbs.parts` slice reaches `emit_quotient`'s
     Horner and `emit_deep_invariants`' `h_sum_zpow`;
   - `ζ`: already discharged — it is the `z` the transcript samples, and the
     zerofier, the row points and `z^P` all take that cell;
   - the trace roots: `GroupCommitment::from_lanes(root.lanes, …)` takes the
     lanes Phase A absorbed, so the Merkle compare and the absorb read one
     unpack;
   - the public output bytes: discharged in wave 4 by `emit_output_bytes`.
   The guard is `epoch_verify_tests::the_assembled_verifier_hints_each_proof_
   value_once` — the same ABSOLUTE count as the spine's, but over the program
   that HAS both consumers, plus a positive control that the assembled program
   declares strictly more arena words than the spine (without it the guard would
   pass just as happily over the spine alone, which is what made the wave-4
   version unable to close this entry). 21 tamper vectors over the assembled
   program are all rejected. (Original text kept below.)
   Original: **PARTIALLY DISCHARGED** (assembly, slice 1+2): the
   assembled spine gives each value ONE cell and hands both views out of one
   struct, so the unification is now a construction rather than a rule —
   `epoch::RootCells` holds a root's two words AND the eight halves the
   transcript absorbs, from a single hint and a single `Unpack`, and
   `epoch::TableAbsorbs` is the surface every later leg reads its cells from.
   `ζ` is fully discharged: it is no longer a value at all, but the `z` the
   transcript samples. The other four are STAGED, not closed — their second
   consumers (constraint evaluation, the DEEP fold, the Merkle root compare,
   the `program_id` fold) are not yet wired onto the spine, so there is
   nothing yet to disagree. They close when those legs hang off
   `TableAbsorbs`, and the entry stays OPEN until they do.
   Original text: each is hinted twice today — not exploitable while the
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

4. ~~**The FRI leg's three per-sub-proof values are arena words and must be
   bound at assembly**~~ — **DISCHARGED** (assembly, slice 1+2).
   `epoch::emit_table_challenges` samples each `ζ_k` from the transcript and
   absorbs layer root `k` immediately after it, draws `ζ_C` only when
   `total_folds > 0`, and absorbs every terminal coefficient after the loop —
   production's order at `verifier.rs:1461-1489`. Falsified four ways, each
   caught by the differential: absorbing the root BEFORE its `ζ`, never
   absorbing the roots, skipping the final-fold draw, and never absorbing the
   coefficients. Witnessed at `num_committed = 0/1/2/3` on single-table
   fixtures and at **12 committed layers** on the real epoch's CPU sub-proof.
   The layer roots remain arena cells, which is correct — they are proof data
   — and they are now the SAME cells the FRI walk compares against.
   (Original text kept below for the record.)
   `declare_fri` hints the
   folding challenges `ζ₀..ζ_C`, the terminal-polynomial coefficients and
   the committed layer roots, exactly as `emit_sub_proof` hints `γ`/`ζ`.
   Two different obligations sit here and they are not interchangeable:
   - **`ζ_k` are CHALLENGES.** They must come from `TranscriptReplay`, and
     production's own order is load-bearing: sample `ζ_k`, THEN absorb root
     `k`, per layer, and only then sample the final-fold `ζ_C` — and that
     last one only when `total_folds > 0` (`verifier.rs:1461-1483`,
     mirroring `fri/mod.rs:86-118`). A prover who chose `ζ` chooses the
     fold, so this is entry 4's family and not a convenience.
   - **The coefficients and the layer roots are proof DATA** that the
     transcript must absorb, because later challenges (the query indices
     among them) depend on them: the roots are appended inside the loop
     above and every coefficient is appended after it. Absorbing them in
     the wrong order, or not at all, does not fail any test in
     `fri_tests` — that suite supplies the real values — so assembly owns
     this and nothing leg-side can catch it.

5. ~~**The standalone FRI driver's hinted index is wider than production's**~~
   — **DISCHARGED** (assembly, slice 1+2). The assembled verifier's query
   index reaches the legs as `TranscriptReplay::sample_u64_pow2`'s BITS and
   never as a felt: `TableChallenges::iota_bits` is the only index the epoch
   spine produces, `log2(lde) − 1` of them, which is production's
   `sample_u64(lde_length >> 1)` (`verifier.rs:138-141`) and exactly the
   Merkle depth the walk consumes. Checked against production's own `iotas`
   for every query of every sub-proof of a real epoch. (Original text below.)
   (fri-emitter, noted not deferred). `fri_tests::fri_only_program` hints
   `iota` as a felt and takes its low `log2(lde) − 1` bits, so `iota` and
   `iota + 2^(n−1)` are the same query to the machine, where production's
   `terminal_codeword.get(iota >> C)` would reject the second as
   out-of-range. This is a property of the ISOLATION driver, not of the
   assembled machine: `SpongeVar::squeeze_bits` produces exactly `nbits`
   bits, so an assembled verifier's index is in range by construction.
   Assembly owes only that the index reaches the query legs as those bits
   and never as a hinted felt.

6. ~~**The challenges guard cited in comments does not exist yet.**~~ —
   **DISCHARGED** (assembly, slice 1+2). The guard is no longer a test to
   write but a construction: `epoch::emit_table_challenges` DERIVES β, z, γ,
   every `ζ_k` and every query index from `TranscriptReplay`, and
   `epoch_tests::the_epoch_challenge_spine_matches_production` checks all 111
   of them against production's own `replay_rounds_after_round_1` over a real
   24-sub-proof epoch. The per-slice differential programs still hint their
   challenges; that is now a property of the ISOLATION drivers, not of the
   assembled verifier, and the assembled path has no `Instr::Hint` for any
   challenge because nothing hints one.

7. **The preprocessed commitments are hinted in the assembled spine.**
   ⚠ **THE ENTRY'S OWN TAXONOMY WAS WRONG AND IS CORRECTED HERE** (assembly-w5,
   slice 1, by reading `lib.rs`). The split is **2 constants + 2 ELF-dependent +
   1 derived**, not 3 + 1 + 1:
   - **BITWISE and KECCAK_RC are genuinely compile-time constants** —
     `bitwise::preprocessed_commitment(proof_options)` and
     `tables::keccak_rc::preprocessed_commitment(proof_options)` take the proof
     options and nothing else (`lib.rs:707-713`, `lib.rs:771-774`). Intern them.
   - **DECODE is ELF-DEPENDENT, not a constant.** `VmAirs::new` builds it as
     `create_decode_air(opts).with_preprocessed(decode::commitment_from_elf(elf,
     opts), …)` (`lib.rs:743-750`). Interning it would make program identity
     ELF-dependent — the same always-stop item the entry raised for PAGE alone.
     DECODE is in PAGE's family.
   - **REGISTER is derived** (reg-tree). `programs::emit_register_commitment`
     now exists, extracted from the isolation program so the spine can call it
     on the register-boundary cells it already declares (a1f32859). Wiring it
     into Phase A is the remaining work and closes entry 2 with it.
   - The corroborating evidence was in plain sight and nobody had connected it:
     `recursion::program_id_from_digest` folds `elf_digest`, `pc_start`,
     `decode_commitment` and every `(page_base, page_commitment)` — precisely
     the ELF-dependent roots and none of the options-only ones.
   **PROPOSED RESOLUTION (needs the team lead's ruling, because the alternative
   touches program identity):** DECODE and PAGE stay ARENA CELLS and are bound
   not by program text but by the attestation — the same cell Phase A absorbs is
   the cell the `program_id` fold consumes, which is the two-consumer join one
   level up and which the machine already has an emitter for
   (`machine_tests::program_id_folds_pages_in_the_production_layout`). That keeps
   one LFM program per epoch SHAPE rather than one per guest ELF. ⚠ The residual
   risk is named honestly: `program_id`'s binding is only as strong as the
   consumer-side `check_attestation` compare, which the RESUME already records as
   having ZERO production call sites. So this proposal makes PAGE/DECODE exactly
   as bound as the existing chain is, and no more. The alternative — deriving
   both in-machine from the ELF bytes, REGISTER-style — costs a full in-machine
   LDE+tree per page and needs the ELF itself bound, which is the full-ELF keccak
   pass sim/8 deliberately removed.
   **MEASURED (`epoch_verify_tests::the_preprocessed_commitments_of_a_real_
   epoch`):** only 4 of this epoch's 24 sub-proofs are preprocessed — index 0
   (11 precomputed columns) BITWISE, 1 (5) DECODE, 5 (9) KECCAK_RC, 8 (3)
   REGISTER, per `VmAirs::air_refs`' fixed order (`lib.rs:610-625`). ★ There is
   **no PAGE sub-proof in this epoch at all** (`num_private_input_pages = 0`), so
   the fixture cannot witness PAGE's half — and per the RESUME's premise rule
   this is a claim about the FIXTURE, not about production, so the witness is a
   differently-configured real epoch (a guest with private input pages), not a
   synthetic AIR.
   (Original text below.) Production
   takes each preprocessed root from the AIR and REJECTS a proof whose copy
   disagrees (`verifier.rs:1184-1209`); the root it absorbs is the verifier's,
   never the prover's. `epoch_tests::epoch_challenge_program` hints all of
   them. Only REGISTER's has a derivation today (reg-tree, from the previous
   epoch's `reg_fini`). BITWISE, DECODE and KECCAK_RC are compile-time
   constants of the AIR set and could simply be interned — but **PAGE's cannot
   be**: it is a function of the inner ELF, which is per-proof arena data, so
   baking it would make program identity proof-dependent (an always-stop
   item). PAGE therefore needs a derivation of the same family as REGISTER's,
   and that derivation does not exist. Assembly owes: intern the three
   constants, wire REGISTER's derivation into Phase A, and either build PAGE's
   or state why the ELF-digest binding already covers it.

8. **The OOD absorb ORDER has no production witness** (assembly, slice 2 —
   measured, not argued). Production absorbs each pruned OOD block
   column-major (`verifier.rs:1425-1429`). Injecting a ROW-major absorb leaves
   BOTH the single-table differential and the 24-sub-proof epoch spine green,
   because every OOD block in either is ONE ROW TALL: the current block's
   height is `step_size` (`ood.rs:110-114`) and the phase already knows
   `step_size = 1` collapses production, while the next block's height is
   `num_eval_points − step_size`, which is 1 for any AIR with two transition
   offsets — all 24 of the epoch's are. Measured dims are printed by the spine
   test. This is a fourth member of the degenerate-parameter family and the
   premise check the RESUME asks for was done: it is a claim about
   PRODUCTION, not about fixtures on hand. Closing it needs a synthetic AIR
   with three transition offsets (or `step_size > 1`), proved by the
   production prover so the oracle stays real.

9. **The constraint leg's FRAME-STEP view of the OOD grid has no production
   witness** (assembly-w5, slice 1 — found while writing the seam, and the sketch
   had it wrong). `Op::Var{offset, row}` indexes the frame's evaluation STEP, and
   production's own interpreter asserts `row == 0`
   (`constraint_ir/interp.rs:240-242`) while taking
   `frame.get_evaluation_step(offset)`. A frame step is `step_size` grid rows, so
   the constraint leg must read every `step_size`-th row of the reconstructed OOD
   grid where DEEP reads all of them. `TableVerifyShape::num_frame_steps` now
   carries that, and the emitter builds the strided view.
   At `step_size = 1` the strided view and the whole grid are the SAME vector, so
   nothing in the suite can tell a correct emitter from one that passes the full
   grid to both legs — which is what the wave-5 sketch did. Same family as entry
   8 and closed by the same witness: an AIR with `step_size > 1` proved by the
   production prover. Recording it separately because it is a different SITE
   (entry 8 is the absorb ORDER, this is the constraint leg's frame indexing) and
   a synthetic AIR built for entry 8 must exercise both or it closes only one.

10. **Every per-epoch cost number must name the epoch SHAPE it describes**
    (assembly-w5, slice 1 — measured, and it retracts nothing but reframes
    everything). The phase's composed predictions were computed at a UNIFORM
    `log2_trace = 20` across all sub-proofs (`join_tests::join_leg_cost`'s stated
    constants). A real INTERMEDIATE epoch is not shaped like that: the fixture
    epoch's measured trace lengths (log2) are `[2 x14, 3, 4 x4, 5 x3, 7, 20]` —
    ONE large table and 23 tiny ones. Openings fall 1.88x against the uniform
    model on this epoch (100,959 against 189,727 at blowup 8 / 73 queries), and
    FRI collapses to a single sub-proof's bill because the other 23 have their LDE
    already terminal at blowup 8, so zero committed layers. This does NOT falsify
    213,744 as a model of a production-sized (2^24-step) epoch, where most tables
    are large — it says the number is a claim about a WORKLOAD, and the two
    workloads must never be compared without saying so. Assembly owes: the wrap
    run's numbers must state their epoch's trace-length profile alongside them.

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

- **PRICED** (assembly, slice 3): the fixture epoch's public output is **8
  bytes**, so the COMMIT-bus gadget is 8 inverse chains — negligible here, and
  still unpriced for a production epoch, whose output length is workload-shaped.
  What assembly added is the JOIN: the bytes are no longer an arena of their
  own but are derived from the halves the statement absorbed
  (`epoch::emit_output_bytes`, one `BitDec` and one `MulAdd` per half, whose
  recomposition assert doubles as the `< 2^32` range check). So the cost line
  is "per output half" for the derivation plus "per output byte" for the fold.
  Original entry:
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
