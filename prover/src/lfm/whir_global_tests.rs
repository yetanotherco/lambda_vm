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
    fn the_cross_epoch_program_executes_on_a_private_page_bundle() {
        let (elf_bytes, opts, bundle) = private_page_bundle();
        if let Some(note) = whir_process_posture_note() {
            println!("{note}");
        }
        let global = harvest(&elf_bytes, &opts, &bundle);
        let airs = global.airs().refs();
        let routes = GlobalPlan::build(&global, &airs).table_routes().to_vec();
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

        let arena = whir_global_arena(&global, &airs);
        let program = whir_global_program(&global, &airs);
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
    fn the_cross_epoch_program_executes_on_a_genesis_page_bundle() {
        let (elf_bytes, opts, bundle) = genesis_page_bundle();
        let global = harvest(&elf_bytes, &opts, &bundle);
        let airs = global.airs().refs();
        let plan = GlobalPlan::build(&global, &airs);
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

        // ⛔ THE ANTI-VACUITY COUNT. A sparse leg over an all-zero column emits
        // no operation at all, so a green run against one says nothing about the
        // arithmetic. The surviving-entry count is PRINTED and asserted nonzero.
        let mut entries = 0usize;
        for &table in &genesis {
            let columns = airs[table].precomputed_columns();
            let init: &[FE] = &columns[1];
            let here = crate::lfm::preprocessed::sparse_entries(&[init]);
            println!(
                "GENESIS PAGE at table {table}: INIT {} rows, {here} nonzero",
                init.len(),
            );
            entries += here;
        }
        assert!(
            entries > 0,
            "the genesis fixture's INIT columns are all zero, so the sparse leg emitted \
             nothing and this suite would be green against arithmetic it never ran"
        );
        println!("GENESIS FIXTURE: {} genesis pages, {entries} nonzero entries", genesis.len());
        // The census the block's cap is owed, printed at the shape that exists.
        println!(
            "{}",
            crate::lfm::whir_global::genesis_census_line(
                &crate::lfm::whir_global::genesis_census(&global, &airs)
            )
        );

        let arena = whir_global_arena(&global, &airs);
        let program = whir_global_program(&global, &airs);
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
        let program = whir_global_program(&global, &airs);
        let cost = global_cost(&global, &airs);

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

        assert_eq!(consts, cost.constants.len(), "the ONE constant pool, by value");
        assert_eq!(hints, cost.hints, "one hint per word the arena writes");
        assert_eq!(publics, cost.publics, "the published set is the layout's words");
        assert_eq!(ops, cost.operations(), "the legs, summed at the plan's own shapes");
        assert_eq!(
            program.instrs.len(),
            cost.instructions(),
            "the assembled emission against the form — the check a per-leg F1 cannot make"
        );
    }

    /// ⛔ A TABLE WHOSE PREPROCESSED COLUMNS NO ROUTE COVERS FAILS THE BUILD.
    ///
    /// The route is decided by the family and the page's own config, and the
    /// column count is the CROSS-CHECK. Here the config is restated — a page the
    /// AIR set built with INIT is told it is private — and the build must refuse
    /// rather than route it as OFFSET-only and leave the genesis column checked
    /// by nothing.
    ///
    /// ⚠ This is the mutation's shape, made reachable as a test: it is the exact
    /// defect a column-count key would wave through.
    #[test]
    #[should_panic(expected = "preprocessed columns")]
    fn a_page_routed_against_its_own_config_is_refused() {
        let (elf_bytes, opts, bundle) = genesis_page_bundle();
        let mut global = harvest(&elf_bytes, &opts, &bundle);
        let genesis = global
            .page_is_private
            .iter()
            .position(|private| !*private)
            .expect("the genesis fixture has a non-private page");
        global.page_is_private[genesis] = true;
        let airs = global.airs().refs();
        let _ = GlobalPlan::build(&global, &airs);
    }

    /// ⛔ AND THE OTHER DIRECTION, because one arm of a two-arm guard is half a
    /// guard: a private page told it carries a genesis column must also be
    /// refused, and for the same reason — it would index a column the AIR never
    /// presented.
    #[test]
    #[should_panic(expected = "preprocessed columns")]
    fn a_private_page_routed_as_genesis_is_refused() {
        let (elf_bytes, opts, bundle) = private_page_bundle();
        let mut global = harvest(&elf_bytes, &opts, &bundle);
        let private = global
            .page_is_private
            .iter()
            .position(|private| *private)
            .expect("this fixture's page is private");
        global.page_is_private[private] = false;
        let airs = global.airs().refs();
        let _ = GlobalPlan::build(&global, &airs);
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
    fn the_cross_epoch_wrap_publishes_every_bookend_root_at_its_own_epoch() {
        use crate::lfm::algebraic_commit::commitment_to_digest;
        use math::field::traits::IsPrimeField;

        let (elf_bytes, opts, bundle) = genesis_page_bundle();
        let global = harvest(&elf_bytes, &opts, &bundle);
        let airs = global.airs().refs();
        let program = whir_global_program(&global, &airs);
        let arena = whir_global_arena(&global, &airs);
        let exec = crate::lfm::execute(&program, &arena, &crate::hash_pin::BLOCK_HASHER)
            .expect("the machine must execute the cross-epoch proof the host accepted");
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
            if input_name.is_empty() { "<none>" } else { &input_name },
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
        let census = crate::lfm::whir_global::genesis_census(&global, &airs);
        println!("{}", crate::lfm::whir_global::genesis_census_line(&census));
        for entry in &census {
            if entry.entries > 0 {
                println!(
                    "  page table {} ({:?}): {} rows, {} nonzero",
                    entry.table, entry.route, entry.rows, entry.entries
                );
            }
        }

        let started = std::time::Instant::now();
        let program = whir_global_program(&global, &airs);
        let built = started.elapsed();
        let arena = whir_global_arena(&global, &airs);
        let cost = global_cost(&global, &airs);
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

    /// One published word's base value, with its upper lanes asserted zero.
    ///
    /// ⚠ A base publish that carried anything in lanes 1..4 would be read by an
    /// aggregation node as the low lane alone, silently. The assert is what
    /// makes that a failure instead of a truncation.
    fn published_base(
        public: &[(u32, crate::lfm::LfmWord)],
        at: usize,
        what: &str,
    ) -> u64 {
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
        let exec = crate::lfm::execute(program, arena, &crate::hash_pin::BLOCK_HASHER)
            .unwrap_or_else(|e| panic!("the cross-epoch program must execute: {e:?}"));
        println!(
            "CROSS-EPOCH PROGRAM: {} instrs / {} arena words / {} published",
            program.instrs.len(),
            arena[0].len(),
            program.public_len,
        );
        drop(exec);
    }
}
