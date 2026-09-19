//! Tests for the CROSS-EPOCH program builder, which lives in
//! [`crate::lfm::whir_global`].
//!
//! They sit beside `whir_epoch_program_tests`, which does the same job for the
//! level-0 program, and they are about the three things only an ASSEMBLED
//! program can be asked: that it EXECUTES against the proof the host accepted,
//! that its instruction count matches the form leg for leg, and that a table
//! whose preprocessed columns no route covers fails the BUILD.
//!
//! # ⛔ WHAT THE TWO FIXTURES ARE FOR, AND WHY THERE ARE TWO
//!
//! `test_private_input_xpage`'s cross-epoch proof is three bookends and ONE
//! page, and that page is a PRIVATE-INPUT page — so it carries OFFSET alone and
//! the OFFSET+INIT route the block's pages mostly take is NOT REACHED BY IT AT
//! ALL. A suite built on that fixture would gate half the route table and read
//! as if it gated all of it.
//!
//! `data_page_touch` is the other half: it loads, increments and stores a static
//! `.dword`, so its touched page is genuinely ELF-backed and its INIT column is
//! NONZERO. That is what makes the genesis leg reachable at fixture scale, and
//! every test below that names it asserts the nonzero count it actually saw —
//! because a sparse leg run against an all-zero column emits nothing and would
//! pass for the wrong reason.

#[cfg(test)]
mod tests {
    use crate::lfm::whir_global::{
        GlobalPlan, GlobalRoute, global_cost, whir_global_arena, whir_global_program,
    };
    use crate::lfm::whir_real_epoch::whir_process_posture_note;
    use crate::lfm::whir_real_global::{WhirRealGlobal, real_global_from_whir_continuation};
    use crate::multilinear_continuation;
    use crate::tables::types::FE;
    use crate::test_utils::asm_elf_bytes;
    use stark::proof::options::ProofOptions;

    /// A run whose cross-epoch proof carries ONE PRIVATE page — the OFFSET-only
    /// route, and nothing else.
    fn private_page_bundle() -> (
        Vec<u8>,
        ProofOptions,
        multilinear_continuation::ContinuationProof,
    ) {
        let mut input: Vec<u8> = Vec::with_capacity(16);
        input.extend_from_slice(&16u32.to_le_bytes());
        input.extend_from_slice(&[0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
        input.extend_from_slice(&[0u8; 4]);
        let elf_bytes = asm_elf_bytes("test_private_input_xpage");
        let opts = ProofOptions::default_test_options();
        let bundle = multilinear_continuation::prove_continuation(&elf_bytes, &input, 2, &opts)
            .expect("prove the continuation");
        (elf_bytes, opts, bundle)
    }

    /// ★ A run whose cross-epoch proof carries a NON-PRIVATE, ELF-BACKED page —
    /// the OFFSET+INIT route, with a genuinely nonzero genesis column.
    ///
    /// The guest is `data_page_touch`, which exists for exactly this reason and
    /// is already relied on for it on the univariate side
    /// (`continuation.rs`: "touches a real ELF `.data` page, unlike this file's
    /// stack-only fixtures").
    fn genesis_page_bundle() -> (
        Vec<u8>,
        ProofOptions,
        multilinear_continuation::ContinuationProof,
    ) {
        let elf_bytes = asm_elf_bytes("data_page_touch");
        let opts = ProofOptions::default_test_options();
        let bundle = multilinear_continuation::prove_continuation(&elf_bytes, &[], 3, &opts)
            .expect("prove the continuation");
        (elf_bytes, opts, bundle)
    }

    /// ⛔⛔ THE PROCESS POSTURE, AND WHY EVERY EXECUTING ARM BELOW IS
    /// `#[ignore]`d RATHER THAN SKIPPED OR LEFT TO FAIL.
    ///
    /// The machine's [`crate::lfm::whir_transcript::WhirTranscript`] is the
    /// ALGEBRAIC sponge — its own header says so: "the mirror is
    /// `DefaultTranscript<E, RpxTranscriptHash>`". The cross-epoch prover and
    /// verifier both dispatch on `whir_hash_knob::selected()`, which is a
    /// process-wide `OnceLock` over an environment variable and is KECCAK when
    /// that variable is unset. So a bundle proved in a default process carries a
    /// keccak transcript, the machine replays an RPX one, every challenge
    /// diverges from the first squeeze, and the honest proof stops executing
    /// with a `DivByZero` that reads as a broken assembly when it is a
    /// configuration mismatch. That is measured, not feared: it is exactly what
    /// the keccak arm of the first WHIR tree run produced.
    ///
    /// The level-0 suite solves this by RE-PROVING its epochs under a literal
    /// `RpxWhir` (`whir_epoch_program_tests::driver_bundle`), so its tests mean
    /// the same thing in every process. **That door is closed here**:
    /// `prove_global` and `verify_global` take no hash parameter, they dispatch
    /// on the knob inside themselves, and re-implementing either in a test would
    /// be a second derivation of the cross-epoch prover. Making them generic
    /// with the dispatch at their callers is the same change W1g made to
    /// `verify_global_bookends`, and it is that lane's, not this one's.
    ///
    /// ⇒ So these arms are `#[ignore]`d with the knob named in the reason, and
    /// they REFUSE when run in the wrong posture. Both halves matter: the ignore
    /// keeps a default `cargo test` from going red over a configuration it never
    /// set, and the refusal stops a deliberate run in the wrong posture from
    /// producing a `DivByZero` nobody can place. A silent skip would have been
    /// neither — it would be green in every default run and could not fail.
    fn require_rpx_posture(what: &str) {
        let setting = crate::whir_hash_knob::selected();
        assert_eq!(
            setting,
            crate::whir_hash_knob::Setting::Rpx,
            "{what} executes a machine program whose transcript is the ALGEBRAIC sponge, \
             and this process is set to {}. Re-run with {}=rpx; without it the bundle \
             carries a different challenge stream and the honest proof refuses at its \
             first table, which is a configuration mismatch and not a defect in the \
             emitter",
            setting.name(),
            crate::whir_hash_knob::ENV,
        );
    }

    /// The AIR set and the driver together, since everything borrows from them.
    fn harvest(
        elf_bytes: &[u8],
        opts: &ProofOptions,
        bundle: &multilinear_continuation::ContinuationProof,
    ) -> WhirRealGlobal {
        real_global_from_whir_continuation(opts, elf_bytes, bundle)
            .unwrap_or_else(|e| panic!("the cross-epoch proof does not harvest: {e}"))
    }

    /// ★★ THE ASSEMBLED CROSS-EPOCH PROGRAM EXECUTES AGAINST THE PROOF THE HOST
    /// ACCEPTED — the gate every per-leg form is only correct relative to.
    ///
    /// ⚠ WHAT MAKES THIS A CHECK RATHER THAN A SMOKE TEST: the program derives
    /// every challenge from its own transcript, so a statement absorbed with the
    /// wrong field, a roots block with the wrong number of draws, a layout that
    /// differed from the host's or a leg run at the wrong point in the stream
    /// all produce a `z` the proof was never argued at, and the argument stops
    /// satisfying its own refusals. Executing at all is the strong statement.
    #[test]
    #[ignore = "needs LAMBDA_VM_WHIR_HASH=rpx: the machine's transcript is the algebraic sponge and the cross-epoch prover dispatches on the knob"]
    fn the_cross_epoch_program_executes_on_a_private_page_bundle() {
        require_rpx_posture("the private-page execution arm");
        let (elf_bytes, opts, bundle) = private_page_bundle();
        if let Some(note) = whir_process_posture_note() {
            println!("{note}");
        }
        let global = harvest(&elf_bytes, &opts, &bundle);
        let airs = global.airs().refs();
        let routes = GlobalPlan::build(&global, &airs, &elf_bytes)
            .table_routes()
            .to_vec();
        println!(
            "PRIVATE-PAGE FIXTURE: {} tables = {} bookends + {} pages, routes {:?}",
            routes.len(),
            global.num_epochs,
            routes.len() - global.num_epochs,
            &routes[global.num_epochs..],
        );
        // ⚠ The stated limit of this fixture, asserted rather than described:
        // it reaches the OFFSET-only route and NOT the genesis one.
        assert!(
            routes[global.num_epochs..]
                .iter()
                .all(|r| *r == GlobalRoute::PrivatePage),
            "this fixture's pages are private; the genesis route is gated by the other one"
        );

        let arena = whir_global_arena(&global, &airs, &elf_bytes);
        let program = whir_global_program(&global, &airs, &elf_bytes);
        assert_eq!(
            program.arena_schema.lens,
            vec![arena[0].len() as u32],
            "one arena, of exactly the words the filler writes"
        );
        execute_against(&program, &arena);
    }

    /// ★★ THE SAME, ON THE BUNDLE THAT REACHES THE GENESIS ROUTE — and it
    /// asserts the nonzero count it saw, so it cannot pass by folding nothing.
    #[test]
    #[ignore = "needs LAMBDA_VM_WHIR_HASH=rpx: the machine's transcript is the algebraic sponge and the cross-epoch prover dispatches on the knob"]
    fn the_cross_epoch_program_executes_on_a_genesis_page_bundle() {
        require_rpx_posture("the genesis-page execution arm");
        let (elf_bytes, opts, bundle) = genesis_page_bundle();
        let global = harvest(&elf_bytes, &opts, &bundle);
        let airs = global.airs().refs();
        let plan = GlobalPlan::build(&global, &airs, &elf_bytes);
        let routes = plan.table_routes().to_vec();
        let genesis: Vec<usize> = routes
            .iter()
            .enumerate()
            .filter(|(_, r)| **r == GlobalRoute::GenesisPage)
            .map(|(i, _)| i)
            .collect();
        assert!(
            !genesis.is_empty(),
            "this fixture exists to reach the OFFSET+INIT route; it reached none, so \
             every assertion below would be vacuous"
        );

        // ⛔ THE ANTI-VACUITY COUNT, AND IT NAMES THE PAGE. A sparse leg over an
        // all-zero column emits no operation at all, so a green run against one
        // says nothing about the arithmetic. The surviving-entry count is
        // PRINTED beside the page's base and its `init_values` length — three
        // numbers, because `init_values.len()` and the nonzero count differ and
        // the difference is what the leg is paid for — and asserted nonzero.
        let census = crate::lfm::whir_global::genesis_census(
            &elf_bytes,
            &global.page_bases,
            global.num_private_input_pages,
        )
        .expect("the census reads off the same ELF the harvest verified");
        println!("{}", crate::lfm::whir_global::genesis_census_line(&census));
        for entry in &census {
            println!("{}", crate::lfm::whir_global::genesis_entry_line(entry));
        }
        let entries: usize = census.iter().map(|e| e.entries).sum();
        assert!(
            entries > 0,
            "the genesis fixture's INIT columns are all zero, so the sparse leg emitted \
             nothing and this suite would be green against arithmetic it never ran"
        );
        // The plan and the census must agree about which pages are genesis —
        // two derivations off one `global_memory_configs` call, and a gate that
        // says so rather than leaving them to be assumed equal.
        assert_eq!(
            genesis.len(),
            census.iter().filter(|e| !e.is_private).count(),
            "the plan's routes and the census disagree about the genesis pages"
        );

        let arena = whir_global_arena(&global, &airs, &elf_bytes);
        let program = whir_global_program(&global, &airs, &elf_bytes);
        execute_against(&program, &arena);
    }

    /// ★★ THE F1 OVER THE ASSEMBLED EMISSION — the check the epoch program does
    /// not have, and the reason a deleted leg was invisible to seventeen tests.
    ///
    /// The form predicts the compiled program's instruction count by KIND —
    /// operations, the one constant pool, the hints and the publishes — and each
    /// is asserted separately so a gap lands on the kind it belongs to.
    #[test]
    fn the_cross_epoch_programs_instruction_count_is_the_forms() {
        let (elf_bytes, opts, bundle) = genesis_page_bundle();
        let global = harvest(&elf_bytes, &opts, &bundle);
        let airs = global.airs().refs();
        let program = whir_global_program(&global, &airs, &elf_bytes);
        let cost = global_cost(&global, &airs, &elf_bytes);

        let consts = program
            .instrs
            .iter()
            .filter(|i| matches!(i, crate::lfm::instr::Instr::Const { .. }))
            .count();
        let hints = program
            .instrs
            .iter()
            .filter(|i| matches!(i, crate::lfm::instr::Instr::Hint { .. }))
            .count();
        let publics = program
            .instrs
            .iter()
            .filter(|i| matches!(i, crate::lfm::instr::Instr::Public { .. }))
            .count();
        let ops = program.instrs.len() - consts - hints - publics;
        println!(
            "CROSS-EPOCH F1: instrs {} = ops {ops} + consts {consts} + hints {hints} + \
             publics {publics}; predicted ops {} (spine {} + tables {} + groups {} + \
             closure {} + publish {}), consts {}, hints {}, publics {}",
            program.instrs.len(),
            cost.operations(),
            cost.spine,
            cost.tables,
            cost.groups,
            cost.closure,
            cost.publish_ops,
            cost.constants.len(),
            cost.hints,
            cost.publics,
        );
        println!(
            "CROSS-EPOCH NEWTON: D = {} (from {}), pool {} words, count form says {}",
            cost.newton_degree,
            match cost.newton_degree_from {
                Some(table) => format!("table {table}"),
                None => "a fixed leg (GKR 3 / reduce 2 / chain 2)".to_string(),
            },
            crate::lfm::whir_poly::sumcheck_round_constants(cost.newton_degree).len(),
            crate::lfm::whir_poly::sumcheck_round_consts(cost.newton_degree),
        );

        // ⛔ WHEN THE POOL DISAGREES, SAY WHICH WORDS — never just the two
        // counts. A count tells you the form is wrong; the VALUES tell you which
        // leg forgot to name what it interns, which is how the roots block's
        // form was closed twice after coming up short. Printed before the
        // assert, so one run names the gap instead of one run per guess.
        // ⚠ THE ADDRESS IS CARRIED WITH THE VALUE, and that is what turns an
        // unnamed word from a number nobody can place into a leg with a name.
        let interned_at: Vec<(u64, crate::lfm::LfmWord)> = program
            .instrs
            .iter()
            .filter_map(|i| match i {
                crate::lfm::instr::Instr::Const { out, value, .. } => Some((out.0, *value)),
                _ => None,
            })
            .collect();
        let interned: Vec<crate::lfm::LfmWord> =
            interned_at.iter().map(|(_, word)| *word).collect();
        let unnamed: Vec<&crate::lfm::LfmWord> = interned
            .iter()
            .filter(|w| !cost.constants.contains(w))
            .collect();
        let unemitted: Vec<&crate::lfm::LfmWord> = cost
            .constants
            .iter()
            .filter(|w| !interned.contains(w))
            .collect();
        if !unnamed.is_empty() || !unemitted.is_empty() {
            println!(
                "POOL GAP: {} words the program interns that the form does not name, \
                 {} the form names that the program does not intern",
                unnamed.len(),
                unemitted.len(),
            );
            // ⛔⛔ AN UNNAMED WORD IS NAMED BY ITS NEIGHBOURS, NOT BY ITS VALUE.
            // Three rounds of this gap were closed by reading emitters and
            // matching value SHAPES — the Newton pairs, the coset fold's
            // generator powers — and each round cost a box run because a value
            // alone says nothing about who interned it. `locate_addr` is the
            // tool this codebase already has for exactly that question: it
            // reports the instruction that wrote a cell and its neighbours, so
            // the leg that interned a constant is READ rather than guessed.
            //
            // ⚠ It costs nothing on the success path: this block runs only when
            // a word is already unaccounted for.
            for w in unnamed.iter().take(40) {
                println!("  UNNAMED  {w:?}");
                if let Some((addr, _)) = interned_at.iter().find(|(_, word)| word == *w) {
                    println!("{}", crate::lfm::executor::locate_addr(&program, *addr));
                    // ⛔ `locate_addr`'s window is ±4, which shows the SHAPE of
                    // the leg but not its CALLER. Round one of this narrowed the
                    // three survivors to one `algebraic_leaf_hash` over six
                    // felts — `[A, a full four-lane digest, B]`, capacity
                    // `leaf_capacity(6)` — and then stalled, because the loop
                    // that builds that felt vector is outside ±4. A wider window
                    // is the difference between "which leg" and "which call".
                    if let Some(index) = program.instrs.iter().position(|i| {
                        matches!(i, crate::lfm::instr::Instr::Const { out, .. } if out.0 == *addr)
                    }) {
                        let lo = index.saturating_sub(24);
                        let hi = (index + 25).min(program.instrs.len());
                        println!("    ---- wider window {lo}..{hi} ----");
                        for (k, instr) in program.instrs[lo..hi].iter().enumerate() {
                            let mark = if lo + k == index { "→" } else { " " };
                            println!("    {mark} [{}] {instr:?}", lo + k);
                        }
                    }
                }
            }
            for w in unemitted.iter().take(40) {
                println!("  UNEMITTED {w:?}");
            }
        }
        // ⛔ THE POOL IS ASSERTED BOTH WAYS, and it was not always so. Two
        // emitters intern words no cost form reported: `emit_newton_step`'s
        // interpolation weights, and the chains' own constants, which
        // `StackedCost::own_constants` disclaims in its own doc. The first is
        // now named by `sumcheck_round_constants` at this program's maximum
        // degree — the pairs NEST, so one call at the max is the whole pool —
        // and the second turns out to BE the first, reached through the chains'
        // sumcheck rounds.
        //
        // ⇒ BOTH DIRECTIONS ARE NOW EXACT, and the second one only became
        // assertable when the Newton pairs got a VALUES form. Until then the
        // pool's remainder was pinned at a measured 24 because nothing named it.
        assert!(
            unemitted.is_empty(),
            "the form names {} words the program does not intern; a leg that stopped \
             emitting is exactly this shape",
            unemitted.len(),
        );
        assert!(
            unnamed.is_empty(),
            "{} words are interned that no form names. The Newton pairs are accounted \
             at D = {}, so a survivor belongs to a DIFFERENT emitter and is a finding \
             to read off the UNNAMED lines above — not a remainder to pin",
            unnamed.len(),
            cost.newton_degree,
        );
        // ★ THE DEGREE, FROM TWO SOURCES. `cost.newton_degree` is derived from
        // the SHAPES; this reads it back out of the words the compiled program
        // actually holds. A disagreement means a leg runs at a degree no shape
        // predicts, or a form names one no leg reaches.
        let read_back = crate::lfm::whir_poly::interned_newton_degree(&interned);
        assert_eq!(
            read_back, cost.newton_degree,
            "the program has interned the Newton set through degree {read_back}, and the \
             shapes predict {}",
            cost.newton_degree,
        );
        assert_eq!(
            consts,
            cost.constants.len(),
            "the ONE constant pool, by value"
        );
        assert_eq!(hints, cost.hints, "one hint per word the arena writes");
        assert_eq!(
            publics, cost.publics,
            "the published set is the layout's words"
        );
        assert_eq!(
            ops,
            cost.operations(),
            "the legs, summed at the plan's own shapes"
        );
        // ⚠ NO TOTAL ASSERT HERE, DELIBERATELY. `ops` is DEFINED above as
        // `instrs.len() − consts − hints − publics`, so an assert on the total
        // given the four component asserts has two sides that cannot differ at
        // this call site — documentation, not a check. The four components are
        // the whole claim.
    }

    /// ⛔ A TABLE WHOSE PREPROCESSED COLUMNS NO ROUTE COVERS FAILS THE BUILD —
    /// DIRECTION ONE: the route expects MORE columns than the AIR presents.
    ///
    /// The route is decided by the family and the page's own config; the column
    /// count is the CROSS-CHECK. Here the DECLARED INPUT is restated — a bundle
    /// whose page is private says it has none — so the config build classifies
    /// the page as a genesis page and the route expects OFFSET and INIT, while
    /// the AIR set the proof was verified against presents OFFSET alone.
    ///
    /// ⚠ THIS RESTATES AN INPUT, NOT A DERIVED FLAG, and that is the point: a
    /// wrong `num_private_input_pages` is exactly the lie the cross-check
    /// exists to catch, and it is the lie a column-count key would wave through
    /// by routing the page as private and checking its genesis with nothing.
    #[test]
    #[should_panic(expected = "preprocessed columns")]
    fn a_page_the_routes_expect_more_columns_from_is_refused() {
        let (elf_bytes, opts, bundle) = private_page_bundle();
        let mut global = harvest(&elf_bytes, &opts, &bundle);
        // ⚠ THE PRECONDITION IS READ IN ITS OWN SCOPE, so the borrow it takes
        // ends before the restatement below. `airs()` borrows the whole driver,
        // and the lie this arm tells is a field of it.
        {
            let airs = global.airs().refs();
            assert_eq!(
                airs[global.num_epochs].precomputed_columns().len(),
                1,
                "this fixture's page is private, so its AIR presents OFFSET alone — \
                 without that the refusal below would fire for another reason"
            );
        }
        // ⛔ THE LIE, and it is an INPUT: the bundle's page is a private-input
        // page and this says the run had none. The AIR set is UNCHANGED — it is
        // the one the proof was verified against — so the route now expects
        // OFFSET and INIT from a table presenting OFFSET alone.
        global.num_private_input_pages = 0;
        let airs = global.airs().refs();
        let _ = GlobalPlan::build(&global, &airs, &elf_bytes);
    }

    /// ⛔ AND DIRECTION TWO, because one arm of a two-arm guard is half a guard:
    /// the route expects FEWER columns than the AIR presents.
    ///
    /// Here the AIR SET is the one that disagrees. It is rebuilt through the
    /// verifier's own `global_airs_for` with the private count zeroed, so its
    /// page AIR carries OFFSET and INIT, while the driver's own count still
    /// says private and the route expects OFFSET alone. That is the realistic
    /// failure of this design — the AIR set and the config source disagreeing —
    /// and it is why the two are built from one call with one set of arguments
    /// in production.
    #[test]
    #[should_panic(expected = "preprocessed columns")]
    fn a_page_the_routes_expect_fewer_columns_from_is_refused() {
        let (elf_bytes, opts, bundle) = private_page_bundle();
        let global = harvest(&elf_bytes, &opts, &bundle);
        let elf = executor::elf::Elf::load(&elf_bytes).expect("the inner ELF loads");
        let other = crate::multilinear_continuation::global_airs_for(
            &elf,
            &opts,
            global.num_epochs,
            &global.page_bases,
            0,
        );
        let refs = other.refs();
        assert_eq!(
            refs[global.num_epochs].precomputed_columns().len(),
            2,
            "the rebuilt set must present OFFSET and INIT, or this arm refuses for \
             another reason than the one it is written for"
        );
        let _ = GlobalPlan::build(&global, &refs, &elf_bytes);
    }

    /// ★★ THE PUBLISHED SET, WORD FOR WORD — and every word against a
    /// derivation that does not come from this program.
    ///
    /// The count comes through `GlobalLayout`'s own accessor, never `2 + epochs
    /// × lanes` spelled again; and each epoch's four published lanes are
    /// compared against `WhirRealGlobal::bookend_roots`, which the DRIVER took
    /// from the host verification's own return through `GlobalProof::l2g_roots`.
    /// The emitter reached the same roots by a different route — accumulating
    /// `num_polys()` over the group layouts it built — so the two index
    /// arithmetics have to agree, which is the whole content of the check.
    ///
    /// ⛔ THIS IS THE ARM A WRONG-EPOCH ROOT HAS TO FAIL, and it is why the
    /// sixteen-group split exists: every bookend is committed ALONE precisely so
    /// that root `k` can be tied to the epoch that committed it. A program that
    /// published epoch `k`'s root in epoch `j`'s slot would leave the root node
    /// comparing a fold against a permuted list — and nothing else in this suite
    /// could see it, because the program would execute perfectly.
    #[test]
    #[ignore = "needs LAMBDA_VM_WHIR_HASH=rpx: the machine's transcript is the algebraic sponge and the cross-epoch prover dispatches on the knob"]
    fn the_cross_epoch_wrap_publishes_every_bookend_root_at_its_own_epoch() {
        require_rpx_posture("the published-set arm");
        use crate::lfm::algebraic_commit::commitment_to_digest;
        use math::field::traits::IsPrimeField;

        // ⛔⛔ THE PRIVATE-PAGE BUNDLE, AND THE REASON IS THIS ARM'S WHOLE
        // POINT. `data_page_touch` runs to ONE epoch — its published set is
        // `2 + 1 × 4` words, which the box read as `publics 6`. With one epoch
        // the pairwise-distinctness guard below iterates zero times, AND
        // reversing the published order is the IDENTITY, so the transposition
        // mutation this arm exists for could not fire. A check that cannot fail
        // and a mutation that cannot fire, from the same fixture choice.
        // `test_private_input_xpage` runs to THREE, so both become real.
        let (elf_bytes, opts, bundle) = private_page_bundle();
        let global = harvest(&elf_bytes, &opts, &bundle);
        assert!(
            global.num_epochs >= 2,
            "this arm is about which epoch each root belongs to, and a run of {} epoch(s) \
             cannot tell a permutation from the identity",
            global.num_epochs,
        );
        let airs = global.airs().refs();
        let program = whir_global_program(&global, &airs, &elf_bytes);
        let arena = whir_global_arena(&global, &airs, &elf_bytes);
        let exec = run_or_locate(&program, &arena, "the cross-epoch proof the host accepted");
        let public = &exec.public_words;

        // ---- the COUNT, through the layout's OWN accessor
        let layout = &global.published;
        assert_eq!(
            public.len(),
            layout.total(),
            "the cross-epoch wrap publishes `z`, `alpha` and one root per epoch"
        );
        assert_eq!(
            program.public_len as usize,
            public.len(),
            "the program declares the words the execution produced"
        );

        // ⛔ ANTI-VACUITY, on the ANSWERS: two epochs whose roots were EQUAL
        // would make a transposition invisible, and the arm would be green
        // against a defect it is written for.
        let digests: Vec<[FE; 4]> = (0..layout.num_epochs)
            .map(|k| {
                assert_eq!(
                    global.bookend_roots[k].len(),
                    1,
                    "epoch {k}'s bookend is one stacked polynomial here"
                );
                commitment_to_digest(&global.bookend_roots[k][0])
            })
            .collect();
        for k in 0..digests.len() {
            for j in (k + 1)..digests.len() {
                assert_ne!(
                    digests[k], digests[j],
                    "epochs {k} and {j} committed their bookends under the SAME root, so \
                     transposing the two published runs is invisible and this arm cannot \
                     see the defect it exists for"
                );
            }
        }

        // ---- each epoch's lanes, against the driver's own roots
        for (k, digest) in digests.iter().enumerate() {
            for (w, lane) in digest.iter().enumerate() {
                assert_eq!(
                    published_base(public, layout.l2g_word(k, w), "bookend lane"),
                    crate::tables::types::GoldilocksField::canonical(lane.value()),
                    "epoch {k}, lane {w} — the root the cross-epoch proof committed that \
                     epoch's bookend under, as the root node reads it back"
                );
            }
        }
        println!(
            "CROSS-EPOCH PUBLISHED: {} words = 2 + {} epochs x {} lanes",
            public.len(),
            layout.num_epochs,
            layout.lanes_per_root,
        );
    }

    /// ★★ THE BOX ARM: the cross-epoch program at a REAL BLOCK's shape, with
    /// the census the genesis cap is owed.
    ///
    /// `#[ignore]`d because it proves a whole continuation. Everything it
    /// prints is a MEASUREMENT and nothing it asserts is a count, deliberately:
    /// the shapes are the block's and pinning them would pin the posture, which
    /// moves. What it asserts is structural and holds for any program — that
    /// every table has a route, that the published set is the layout's, and
    /// that the program the emitter built is the one the arena fills.
    ///
    /// ⚠ IT SKIPS RATHER THAN FAILS when the block ELF is absent, and says so
    /// on its own line, so a laptop run cannot be read as a block run. The ELF's
    /// FULL 64-hex sha256 is printed — full, never a prefix: a diagnostic that
    /// can agree while the values differ is not a diagnostic.
    ///
    /// ⛔ THE NUMBER THE CAP IS WAITING FOR is `GENESIS CENSUS`. It is a
    /// function of the ELF and the touched page list alone, so it needs no card
    /// and no second run — but it does need a real block's page set, which is
    /// why it lives here rather than in the fixture arms.
    #[test]
    #[ignore = "the box runs it: a real block bundle under the process hash"]
    fn the_block_bundle_builds_its_cross_epoch_program() {
        let name = std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "ethrex".into());
        let input_name = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
        let epoch_size_log2: u32 = std::env::var("LAMBDA_VM_BENCH_EPOCH_LOG2")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);

        let Some(elf_bytes) = bench_elf_if_present(&name) else {
            println!(
                "GLOBAL-PROGRAM SKIPPED - no ELF named {name} in \
                 executor/program_artifacts/{{rust,asm}}; set LAMBDA_VM_BENCH_ELF to a \
                 program that exists"
            );
            return;
        };
        let input = crate::tests::multilinear_bench_tests::input_bytes(&input_name);
        let opts = ProofOptions::default_test_options();
        println!(
            "GLOBAL-PROGRAM fixture {name} sha {} ({} bytes)  input {} ({} bytes)  epoch 2^{}",
            sha256_hex(&elf_bytes),
            elf_bytes.len(),
            if input_name.is_empty() {
                "<none>"
            } else {
                &input_name
            },
            input.len(),
            epoch_size_log2,
        );
        if let Some(note) = whir_process_posture_note() {
            println!("{note}");
        }

        let bundle = multilinear_continuation::prove_continuation(
            &elf_bytes,
            &input,
            epoch_size_log2,
            &opts,
        )
        .expect("prove the continuation under the process hash");
        let global = harvest(&elf_bytes, &opts, &bundle);
        let airs = global.airs().refs();

        // ★ THE CENSUS FIRST, because if it is over the cap the program build
        // REFUSES and the refusal is the finding — printing the number before
        // the build is what makes that legible instead of a panic with no
        // context.
        let census = crate::lfm::whir_global::genesis_census(
            &elf_bytes,
            &global.page_bases,
            global.num_private_input_pages,
        )
        .expect("the census reads off the same ELF the harvest verified");
        println!("{}", crate::lfm::whir_global::genesis_census_line(&census));
        for entry in &census {
            println!("{}", crate::lfm::whir_global::genesis_entry_line(entry));
        }

        let started = std::time::Instant::now();
        let program = whir_global_program(&global, &airs, &elf_bytes);
        let built = started.elapsed();
        let arena = whir_global_arena(&global, &airs, &elf_bytes);
        let cost = global_cost(&global, &airs, &elf_bytes);
        println!(
            "GLOBAL PROGRAM: {} tables = {} bookends + {} pages; {} instrs \
             (ops {} + consts {} + hints {} + publics {}); {} arena words; \
             {} published; built in {:.2}s",
            global.num_tables(),
            global.num_epochs,
            global.num_tables() - global.num_epochs,
            program.instrs.len(),
            cost.operations(),
            cost.constants.len(),
            cost.hints,
            cost.publics,
            arena[0].len(),
            program.public_len,
            built.as_secs_f64(),
        );

        // Structural, and true of any program: the layout's words, and one hint
        // per word the filler writes.
        assert_eq!(program.public_len as usize, global.published.total());
        assert_eq!(program.arena_schema.lens, vec![arena[0].len() as u32]);
        assert_eq!(
            program.instrs.len(),
            cost.instructions(),
            "the assembled emission against the form, at the block's own shape"
        );
    }

    /// ★★★ THE CENSUS INSTRUMENT — PROVE-FREE, CARD-FREE, AND THE NUMBER THE
    /// CAP IS SET FROM.
    ///
    /// It runs the guest ONCE with every prove and trace build omitted
    /// (`continuation::block_page_census`), takes the touched page list and the
    /// private-page count from that execution, rebuilds the page configs through
    /// the verifier's own `global_memory_configs`, and counts the nonzero genesis
    /// bytes per page. No proof, no AIR, no card, no bundle.
    ///
    /// ⚠ THE TOUCHED PAGE LIST IS AN EXECUTION FACT — which cells cross an epoch
    /// boundary — so it is a function of the ELF, the INPUT and the epoch size
    /// together. All three are named on the output line, and the first two are
    /// REFUSED unless the caller states their full sha256: a census quoted
    /// against the wrong input is the campaign's own recurring defect, and a
    /// guard that accepts an unnamed input cannot tell the two apart.
    ///
    /// ⛔ IT REFUSES RATHER THAN SKIPS when the shas are unstated, and SKIPS
    /// with its own line when the ELF is simply absent. Two distinct outcomes,
    /// because "could not run the probe" and "ran it and the input is wrong" are
    /// different findings and a single refusal would report only its own
    /// hypothesis.
    #[test]
    #[ignore = "the box runs it: one execution of the block guest, no proving"]
    fn the_block_genesis_census() {
        let name = std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "ethrex".into());
        let input_name = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
        let epoch_size_log2: u32 = std::env::var("LAMBDA_VM_BENCH_EPOCH_LOG2")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(21);

        let Some(elf_bytes) = bench_elf_if_present(&name) else {
            println!(
                "GENESIS-CENSUS SKIPPED - no ELF named {name} in \
                 executor/program_artifacts/{{rust,asm}}; set LAMBDA_VM_BENCH_ELF"
            );
            return;
        };
        let input = crate::tests::multilinear_bench_tests::input_bytes(&input_name);

        // ⛔ THE GUARD, AND IT MUST BE ABLE TO REFUSE THE RUN IT IS HANDED.
        // Both shas at FULL width: a diagnostic that can agree while the values
        // differ is not a diagnostic.
        let elf_sha = sha256_hex(&elf_bytes);
        let input_sha = sha256_hex(&input);
        let want_elf = std::env::var("LAMBDA_VM_CENSUS_ELF_SHA256").unwrap_or_else(|_| {
            panic!(
                "this census is quoted as a fact about ONE program and ONE input, so it \
                 refuses to run unnamed. Set LAMBDA_VM_CENSUS_ELF_SHA256={elf_sha} and \
                 LAMBDA_VM_CENSUS_INPUT_SHA256={input_sha}"
            )
        });
        let want_input = std::env::var("LAMBDA_VM_CENSUS_INPUT_SHA256").unwrap_or_else(|_| {
            panic!("LAMBDA_VM_CENSUS_INPUT_SHA256 is unset; the input here is {input_sha}")
        });
        assert_eq!(
            elf_sha, want_elf,
            "the ELF is not the one this census was asked for"
        );
        assert_eq!(
            input_sha, want_input,
            "the INPUT is not the one this census was asked for, and the touched page \
             list is a function of it"
        );
        println!(
            "GENESIS-CENSUS elf {name} sha {elf_sha} ({} bytes)  input {} sha {input_sha} \
             ({} bytes)  epoch 2^{epoch_size_log2}",
            elf_bytes.len(),
            if input_name.is_empty() {
                "<none>"
            } else {
                &input_name
            },
            input.len(),
        );

        let started = std::time::Instant::now();
        let pages = crate::continuation::block_page_census(&elf_bytes, &input, epoch_size_log2)
            .expect("the guest runs to completion");
        println!(
            "EXECUTION: {} epochs, {} touched pages, {} private-input pages, in {:.1}s",
            pages.num_epochs,
            pages.touched_page_bases.len(),
            pages.num_private_input_pages,
            started.elapsed().as_secs_f64(),
        );

        let census = crate::lfm::whir_global::genesis_census(
            &elf_bytes,
            &pages.touched_page_bases,
            pages.num_private_input_pages,
        )
        .expect("the page configs rebuild from the ELF");
        for entry in &census {
            println!("{}", crate::lfm::whir_global::genesis_entry_line(entry));
        }
        println!("{}", crate::lfm::whir_global::genesis_census_line(&census));

        // ⛔ NOT AN ASSERT ON THE TOTAL. That number is what this arm exists to
        // READ, and asserting it here would pin the posture and make the cap a
        // thing the test agrees with rather than a thing the reading decides.
        //
        // ⚠ AND NOT ON THE PAGE COUNT EITHER, THOUGH IT LOOKS LIKE THE OBVIOUS
        // ONE. `genesis_census` maps one entry per config and
        // `global_memory_configs` maps one config per base, so
        // `census.len() == touched_page_bases.len()` holds by construction and
        // no input reaches this arm that could make it false. It is ARGUED, not
        // checked — an earlier draft of this comment claimed it was "a count
        // that can disagree", which was wrong in exactly the way this campaign
        // keeps cataloguing.
        //
        // What IS asserted is a relation between two fields the census reads
        // SEPARATELY, which is where a wrong column would show:
        for entry in &census {
            if entry.init_len == 0 {
                assert_eq!(
                    entry.entries, 0,
                    "page {:#018x} loads no genesis bytes yet the census found {} nonzero \
                     entries in its INIT column — the census is reading a column that is \
                     not this page's genesis",
                    entry.page_base, entry.entries,
                );
            }
            assert!(
                entry.entries <= entry.init_len,
                "page {:#018x} loads {} genesis bytes and the census found {} nonzero — a \
                 page cannot hold more nonzero genesis than it loads",
                entry.page_base,
                entry.init_len,
                entry.entries,
            );
            assert!(
                !entry.is_private || entry.entries == 0,
                "page {:#018x} is a private-input page, whose genesis the verifier never \
                 recomputes, yet the census charged it {} entries",
                entry.page_base,
                entry.entries,
            );
        }
    }

    /// The block ELF, if the artifacts hold one — SKIP, never panic, because
    /// this suite's contract on a laptop is to say it did not run.
    fn bench_elf_if_present(name: &str) -> Option<Vec<u8>> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .join("executor/program_artifacts");
        for dir in ["rust", "asm"] {
            if let Ok(bytes) = std::fs::read(root.join(dir).join(format!("{name}.elf"))) {
                return Some(bytes);
            }
        }
        None
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::Digest;
        let mut h = sha2::Sha256::new();
        h.update(bytes);
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Executes, and on a refusal SAYS WHICH ASSERT FAILED.
    ///
    /// ⛔ A `DivByZero` is never an inversion gone wrong. `assert_eq` lowers to
    /// `diff = a − b; _ = diff / ZERO` and the executor reports the NUMERATOR's
    /// address, so the address always names the `diff` cell
    /// (`executor.rs:1389-1398`). Bare, that reads as a machine fault at an
    /// address nobody can place; `locate_addr` turns it into the instruction
    /// that wrote the cell and its neighbours, which identifies the leg without
    /// bisecting the emitter. It costs nothing on the success path.
    fn run_or_locate(
        program: &crate::lfm::LfmProgram,
        arena: &[Vec<crate::lfm::LfmWord>],
        what: &str,
    ) -> crate::lfm::LfmExecution {
        match crate::lfm::execute(program, arena, &crate::hash_pin::BLOCK_HASHER) {
            Ok(exec) => exec,
            Err(crate::lfm::LfmExecError::DivByZero { addr }) => panic!(
                "{what} REFUSED — a failing equality assert, not a machine fault.\n{}",
                crate::lfm::executor::locate_addr(program, addr)
            ),
            Err(why) => panic!("{what} must execute: {why:?}"),
        }
    }

    /// One published word's base value, with its upper lanes asserted zero.
    ///
    /// ⚠ A base publish that carried anything in lanes 1..4 would be read by an
    /// aggregation node as the low lane alone, silently. The assert is what
    /// makes that a failure instead of a truncation.
    fn published_base(public: &[(u32, crate::lfm::LfmWord)], at: usize, what: &str) -> u64 {
        use math::field::traits::IsPrimeField;
        let word = &public[at].1;
        for (lane, value) in word.iter().enumerate().skip(1) {
            assert_eq!(
                crate::tables::types::GoldilocksField::canonical(value.value()),
                0,
                "{what}: lane {lane} of a base publish must be zero"
            );
        }
        crate::tables::types::GoldilocksField::canonical(word[0].value())
    }

    /// Runs the program against the arena the builder wrote, which is the whole
    /// execution gate: a misaligned arena hands the machine somebody else's
    /// field element and the argument stops satisfying its own refusals.
    fn execute_against(program: &crate::lfm::LfmProgram, arena: &[Vec<crate::lfm::LfmWord>]) {
        let exec = run_or_locate(program, arena, "the cross-epoch program");
        println!(
            "CROSS-EPOCH PROGRAM: {} instrs / {} arena words / {} published",
            program.instrs.len(),
            arena[0].len(),
            program.public_len,
        );
        drop(exec);
    }
}
