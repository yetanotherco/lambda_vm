# Assembly obligations — debts the epoch-verifier assembly must discharge

Started 2026-07-31. Each entry is a deferral whose safety argument is still
owed (standing-decisions method rule 5). Assembly (RESUME item 5) may not be
called done while any entry is OPEN. Add entries as legs flag them; close an
entry only with the verifying evidence named in it.

## STATUS AT WAVE 6 (2026-08-04)

Entries 1 through 9 are all DISCHARGED. **Entry 10 is the only one left open**,
and it is not a debt but a standing rule for the wrap run: every per-epoch cost
number must name the epoch shape it describes. Two things wave 6 could not close
and that belong to the USER rather than to a leg:

- **A framework ceiling.** The production prover cannot prove any AIR with
  `step_size > 1` (entry 9's note). Lifting it looks like a one-line relaxation
  in `crypto/**` — an always-stop item. Until then no end-to-end run of the
  assembled verifier at `step_size > 1` is possible.
- **PAGE's preprocessed roots MIGRATED rather than closed** (entry 7's note).
  They are the GLOBAL proof's GlobalMemory AIRs' commitments, not an epoch's; no
  continuation epoch of any guest carries a PAGE sub-proof. The taxonomy is
  worked out and the classifier already handles the zero-init (constant) half;
  the ELF-data half would hit its panic, which is the intended handover.

Also unchanged and now more load-bearing than ever: `check_attestation` has ZERO
production call sites, and DECODE's binding rests on it.

## OPEN

1. ~~**`reg_fini` felt-width gap**~~ — **DISCHARGED** (assembly-w6, slice 3) by
   the entry's own stated default. The entry said "if assembly arrives and the
   `no >u32 register column` argument is still unverified, emit the check"; it is
   still unverified, and slice 1's derivation is what made the boundary vectors
   live arena data in the assembled verifier, so the check is now emitted:
   `epoch::assert_u32` on all 134 cells (one `BitDec` plus one recomposition
   each).
   - Placed at the ASSEMBLY call site, deliberately, NOT inside
     `programs::emit_register_commitment`. The isolated derivation's width gap is
     pinned by a guard test that asserts the hazard still exists
     (`the_derivation_extends_a_non_u32_register_value_demonstrating_hazard`) and
     that test is right: an isolated derivation binds nothing. Assembly is where
     the width becomes enforceable, so assembly is where it is enforced, and the
     guard stays green.
   - ⚠ **THE OBVIOUS TEST FOR THIS IS VACUOUS, and it was written before it was
     caught.** "Set a boundary word to `2^32` and watch the epoch fail" fails with
     the check REMOVED too, because a wide value moves the derived root and
     therefore every challenge after Phase A. What is non-vacuous is the pair
     `the_register_boundary_is_width_checked` runs: (a) `assert_u32` in isolation
     admits the whole `u32` range and rejects everything above it, and (b) a
     STRUCTURAL check that every register-arena `Hint` output is the input of a
     32-bit `BitDec`. FALSIFIED: applying the check to a 3-cell prefix — precisely
     the defect a value tamper cannot see — fails (b) and nothing else
     (16 passed / 1 failed).
   - ★ A subtlety the test found, which sizes the gap exactly: an arena word is a
     FIELD ELEMENT, so `FE::from(u64::MAX − 1)` is the felt `2^32 − 3`, a perfectly
     good `u32`. The widening this entry names is therefore the interval
     `[2^32, p)` and nothing beyond; a first draft of the test used
     `u64::MAX − 1` as an out-of-range value and reported the check broken when it
     was not.
   (Original text kept below.) (flagged by reg-tree, slice 1).
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

2. ~~**`start_index` is unbound to the chain**~~ — **DISCHARGED**
   (assembly-w6, slice 1, 2c810857). Phase A now CALLS
   `programs::emit_register_commitment` on the register-boundary arena the spine
   declares plus a new `reg_fini` arena, so the REGISTER preprocessed root the
   transcript absorbs is COMPUTED from those cells. That computation is the
   binding, and it is the binding production itself uses.
   - `start_index` is no longer even a second READ of slot 64: it IS
     `reg_init[X254_INDEX]`, the cell the derivation consumed. Before this slice
     the epoch declared 67 INIT words and read exactly one; now every word of
     both vectors is read, which is why
     `the_spine_hints_each_proof_value_once`' positive control tightened from
     `declared − 67 + 1` to `declared` exactly.
   - The differential is free and total: a wrong derivation moves the absorbed
     root, which moves all 111 challenges, so
     `the_epoch_challenge_spine_matches_production` covers it.
   - FALSIFIED: `the_derivation_binds_every_register_boundary_word` moves ten
     words — first and last of INIT and of FINI, and slots away from 64 — and
     every one makes the epoch unverifiable.
   (Original text kept below.) **HALF DISCHARGED** (assembly, slice 3): the reading is
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

7. ~~**The preprocessed commitments are hinted in the assembled spine.**~~ —
   **DISCHARGED** (assembly-w6, slice 1, 2c810857), and the RULING'S OWN
   TAXONOMY IS AMENDED for the second time. Each root now comes from the source
   its provenance admits, and which source that is comes from a CLASSIFIER
   (`epoch_tests::prep_source`) that recomputes production's candidate functions
   and matches — never from a sub-proof index. A preprocessed table whose root
   matches nothing known PANICS rather than being hinted unbound, which is the
   failure mode the entry needed most.
   - **options-only ⇒ interned as program text**, absorbed as literal bytes with
     no splice arithmetic (`RootCells::constant`,
     `statement_replay::PhaseAPreprocessed::Constant`).
   - **REGISTER ⇒ derived in-machine**, which closes entry 2 with it.
   - **DECODE ⇒ arena cell + the attestation join**: the same cell Phase A
     absorbs is the cell `programs::emit_program_id` folds, differentialled
     against production's `recursion::program_id_from_digest`.
   - The join is denied STRUCTURALLY by a pair of absolute guards, which are
     complete for the class together and neither of which suffices alone: a
     second READ of a word fails `the_assembled_verifier_hints_each_proof_value_
     once`, and a second WORD fails
     `the_assembled_verifier_declares_exactly_the_shape_words` (a closed form
     over the epoch's shapes, not an emitter pass). A fold reading some OTHER
     existing value publishes an id that is not production's, which the spine
     differential catches.
   - FALSIFIED as a COHERENT FORGERY, not a count:
     `epoch_program_with(split_decode = true)` gives the fold its own arena copy,
     and `a_split_decode_cell_forges_the_attestation` shows that program RUNS the
     substitution and attests to a DIFFERENT program's id — while publishing the
     honest id when the same surplus arena holds the honest root, so the forgery
     is a free choice and not a broken proof. On the joined program the
     substitution is inexpressible.
   - ★ **THE RULING'S CONDITION (b) IS UNSATISFIABLE AND UNNECESSARY, and the
     PAGE obligation MIGRATES rather than closing.** The ruling asks for a real
     epoch from a guest with private input pages, on the premise that
     `num_private_input_pages = 0` is a fixture property. Three readings
     overturn it:
     * private-input pages are built NON-preprocessed (`lib.rs:800-828`), so they
       could never witness a PAGE preprocessed root at all;
     * **no continuation epoch of any guest has a PAGE sub-proof.**
       `prove_epoch` REJECTS one outright — "continuation epoch must have no PAGE
       configs (L2G bookend replaces PAGE)" (`continuation.rs:695-702`) — and
       both `build_epoch_airs` call sites pass `page_configs = &[]`
       (`continuation.rs:711-714`, `815-818`). The fixture matches PRODUCTION
       here; it is not stripped down;
     * the ELF-data page genesis roots the attestation folds are the GLOBAL
       proof's GlobalMemory AIRs' preprocessed commitments
       (`continuation.rs:997-1010`) — a different proof, out of an epoch
       verifier's scope. `recursion::program_id_from_digest`'s own doc says
       exactly this ("the supplied DECODE / ELF-data-page roots").
     So the epoch taxonomy is **2 constants + 1 derived + 1 ELF-dependent**,
     asserted by `the_preprocessed_commitments_of_a_real_epoch` (census 2/1/1,
     plus a guard that no sub-proof carries PAGE's preprocessed width). PAGE's
     half becomes the GLOBAL-proof verifier's obligation, whose taxonomy is
     already worked out and already in the classifier: zero-init pages share
     `page::zero_init_preprocessed_commitment(options)` and are therefore
     CONSTANTS (a third finding — the ruling put all of PAGE in the ELF-dependent
     family), while ELF-data pages are ELF-dependent and would hit the
     classifier's panic, since their `PageConfig`-shaped provenance is not in its
     candidate list. That panic is the correct behaviour and the handover note.
   - The residual risk the ruling named is unchanged and now carries more weight:
     `program_id`'s binding is only as strong as the consumer-side
     `check_attestation` compare, which has ZERO production call sites.
   - MEASURED cost of the whole wiring, min preset: +59,743 instructions
     (2.7% of the assembled verifier), +256 permutations and +63 arena words.
     The permutation figure is exactly the prediction — 255 for the REGISTER tree
     at blowup 2 (`128·blowup − 1`, reg-tree's pinned closed form) plus 1 for the
     `program_id` fold's single rate block. The arena figure is exactly
     +67 (`reg_fini`) + 2 (`pc_start`) − 6 (three roots that stopped being arena
     data).
   (Original text kept below for the record.)
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

8. ~~**The OOD absorb ORDER has no production witness**~~ — **DISCHARGED**
   (assembly-w6, slice 2, 36bfd727), and it needed no synthetic AIR at all.
   `stark::examples::fibonacci_multi_column::FibonacciMultiColumnAIR<Gl, Ext3>`
   already carries `transition_offsets: vec![0, 1, 2]` and is generic over the
   extension field, so at three columns its next-row block is **3 columns × 2
   ROWS** — `num_eval_points − step_size = 2`. That is the phase's first OOD block
   where a column-major and a row-major absorb differ.
   - The proof is production's (`multi_prove_ram`, accepted by
     `multi_verify_views`) and so is the oracle: the machine's
     `emit_table_challenges` replay is differentialled against
     `replay_rounds_after_round_1` on every challenge.
   - FALSIFIED, and the falsification re-proves this entry's own claim as a
     by-product: swapping the absorb loop to ROW-major leaves the 24-sub-proof
     epoch spine differential, the assembled-verifier run and the single-table
     replay ALL GREEN (206 passed) and fails only the new fixture. The failure
     MODE is worth recording — the mutation trips the in-program GRINDING check
     first (a moved transcript state invalidates the nonce), so the clean
     statement of the property is the test's own row-major CONTROL program, which
     stops at `γ` and shows it moves against production's `γ`.
   - Unexercised and named rather than chased: an OOD grid with more than TWO
     blocks. Three offsets at `step_size = 1` still yields two, and nothing in the
     machine is shaped by the block count (`emit_reconstruct_ood` takes two
     because the proof carries two), so that is a framework property.
   (Original text kept below.) Production absorbs each pruned OOD block
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

9. ~~**The constraint leg's FRAME-STEP view of the OOD grid has no production
   witness**~~ — **DISCHARGED** (assembly-w6, slice 2, 36bfd727) by an oracle
   rather than a proof, because a proof turned out to be impossible (see the
   ceiling below). The defect is the machine's grid→frame-step MAPPING, and
   production has a pure function for exactly that mapping:
   `StarkTableView::into_frame(main_cols, step_size)`
   (`proof/view.rs:269-294`), which the real verifier calls at
   `verifier.rs:320-321`. It takes a grid and a `step_size` and needs no prover.
   - The rule is extracted out of the emitter as
     `epoch_verify::frame_step_view` precisely so it can be differentialled, and
     `the_frame_step_view_matches_productions_own_frame_assembly` compares it
     against `into_frame` at `(offsets, step_size)` of `(2,1) (3,1) (2,2) (3,2)
     (2,4)`, over main AND aux columns, on grids of distinct values.
   - FALSIFIED: making `frame_step_view` return the whole grid — wave-5's M2
     defect verbatim — fails this test and NOTHING ELSE (206 passed / 1 failed).
   - ⚠ **A FRAMEWORK CEILING, and it is why this entry has no end-to-end
     witness.** The production prover cannot prove ANY AIR with `step_size > 1`:
     its CPU transition evaluator borrows one row per transition offset
     (`RowFrame::from_lde`, called at `evaluator.rs:72`) and asserts the shape
     outright — `debug_assert_eq!(lde_step_size, blowup_factor, "RowFrame requires
     single-row steps (step_size 1)")` — and `lde_step_size = step_size ·
     blowup_factor`, so the equality IS `step_size == 1`. MEASURED, not read:
     `step_size_tests::the_prover_cannot_prove_a_step_size_two_air` is a
     `#[should_panic]` on that message, so the ceiling is recorded and
     self-updating. Nothing here runs the ASSEMBLED verifier at `step_size > 1`,
     and nothing can until it lifts.
   - What lifting it would take, FROM READING and not verified by running: the
     assert looks over-strict for the access pattern that exists.
     `ConstraintBuilder::main(offset, col)` resolves to row 0 of a step
     (`builder.rs:719-724`), `RowFrame::from_lde`'s index for step `k` is
     `row + offset · lde_step_size` — the same row the general, multi-row-capable
     `Frame::read_from_lde` calls `initial_step_row` — and that general path
     already handles `step_size > 1` correctly. So it is plausibly a one-line
     relaxation in `crypto/**`, which is an always-stop item and therefore the
     USER's call, not a leg's.
   - The brief's single-AIR plan ("three transition offsets AND `step_size > 1`")
     is unbuildable for two independent reasons: `AirWithBuses::new` HARDCODES
     `transition_offsets: vec![0, 1]` (`lookup.rs:922`), so three offsets means an
     `AIR` impl and every one outside `crypto/**`'s example tree is IN that tree;
     and `step_size > 1` is unprovable per the ceiling. Entries 8 and 9 therefore
     get two witnesses, each with a production oracle, and neither is a witness of
     the other's defect.
   (Original text kept below.) `Op::Var{offset, row}` indexes the frame's evaluation STEP, and
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
