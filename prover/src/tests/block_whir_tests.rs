//! The block proved in one multilinear proof (`crate::block_whir`): round
//! trips at one group and at many, KECCAK_RND split into chunks, the hold
//! between the phases moving no byte, and one refusal per check the block adds.

use crate::block_whir::{
    self, BlockFormat, BlockOptions, BlockWhirProof, Deviations, prove_block_whir_with,
    verify_block_whir,
};
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::{CutPadding, RowFamilies};
use crate::test_utils::{E, asm_elf_bytes};
use crate::zf_format::ZfFormat;
use math::field::element::FieldElement;
use multilinear::whir_chain::{ArgueFormat, StackVars};
use stark::multilinear_table::batched::VerifierChecks;
use stark::proof::options::ProofOptions;

/// The production format; one group holds a small program whole.
fn one_group() -> BlockFormat {
    BlockFormat {
        zf: ZfFormat::DEFAULT,
        group_polys: block_whir::BLOCK_GROUP_POLYS,
        max_groups: block_whir::BLOCK_MAX_GROUPS,
        prepared: true,
        argue: ArgueFormat::PerTable,
        group: multilinear::whir_chain::GroupFold::PerPoly,
        leaf_scale: crate::lfm::whir_block::LeafScale::Fixed,
        leaf_query: crate::lfm::whir_chain::LeafQuery::V1,
    }
}

/// A stack of 2^10 and two polynomials a group: a small program spans several
/// groups, which is the block's shape at a size a test can prove.
fn many_groups() -> BlockFormat {
    let mut zf = ZfFormat::DEFAULT;
    zf.whir_stack = StackVars::new(10).expect("a valid stack");
    BlockFormat {
        zf,
        group_polys: 2,
        max_groups: block_whir::BLOCK_MAX_GROUPS,
        prepared: true,
        argue: ArgueFormat::PerTable,
        group: multilinear::whir_chain::GroupFold::PerPoly,
        leaf_scale: crate::lfm::whir_block::LeafScale::Fixed,
        leaf_query: crate::lfm::whir_chain::LeafQuery::V1,
    }
}

fn options(max_rows: MaxRowsConfig, keccak_rnd_rows_log2: usize) -> BlockOptions {
    BlockOptions {
        max_rows,
        keccak_rnd_rows_log2,
        ecdas_rows_log2: block_whir::BLOCK_ECDAS_ROWS_LOG2,
        keccak_rows_log2: block_whir::BLOCK_KECCAK_ROWS_LOG2,
        ecsm_rows_log2: block_whir::BLOCK_ECSM_ROWS_LOG2,
        copy_rows_log2: block_whir::BLOCK_COPY_ROWS_LOG2,
        drop_levels: multilinear::whir_commit::TreeDrop::uniform(3),
        window_log2: None,
        stream_keccak_rnd: false,
        stream_memw_lt: false,
        drop_streamed_ops: false,
        layout_workers: 0,
        layout_ahead: Some(2),
        pack_rest_as_laid_out: false,
        // Every table packed on a card, so a box run walks the narrow path at a
        // test's size.
        narrow: stark::multilinear_block::Narrowing::Card { min_cells: 0 },
        upload_ahead: true,
        memlog: false,
        // The rest laid out in waves of 1 MiB: several at a test's size.
        finish_keccak_rnd_chunks: true,
        finish_cuts: true,
        finish_stream: true,
        rest_layout_bytes: Some(1 << 20),
        rest_build_bytes: None,
        pack_finished: true,
        gpack: true,
        spill: crate::block_whir::BlockSpillPolicy::Off,
        regen: None,
        phase_b_overlap: stark::multilinear_block::PhaseBOverlap::Off,
    }
}

fn prove_with(
    elf: &[u8],
    format: &BlockFormat,
    options: &BlockOptions,
    deviations: &Deviations,
) -> BlockWhirProof {
    prove_block_whir_with(
        elf,
        &[],
        &ProofOptions::default_test_options(),
        format,
        options,
        deviations,
    )
    .expect("prove")
    .0
}

fn prove(elf: &[u8], format: &BlockFormat, options: &BlockOptions) -> BlockWhirProof {
    prove_with(elf, format, options, &Deviations::default())
}

fn verify(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> bool {
    verify_block_whir(proof, elf, &ProofOptions::default_test_options(), format).unwrap_or(false)
}

fn groups_of(proof: &BlockWhirProof) -> usize {
    proof.proof.columns.len()
}

#[test]
fn a_program_proves_and_verifies_as_one_block() {
    let elf = asm_elf_bytes("sub");
    let proof = prove(&elf, &one_group(), &options(MaxRowsConfig::default(), 16));
    assert_eq!(groups_of(&proof), 1);
    assert!(verify(&proof, &elf, &one_group()));
}

/// ★ `LAMBDA_VM_ZF_WHIR_GRIND_BITS` on the block: the grind bits, and the Q
/// they buy back, are the verifier's constants, never the proof's. A block
/// ground at 18 bits verifies under 18 and is refused by a verifier at 20; a
/// block ground at 20 is refused by a verifier at 18.
#[test]
fn a_block_ground_at_18_bits_is_refused_by_a_verifier_at_20() {
    let at = |bits: u8| BlockFormat {
        zf: ZfFormat {
            whir_grind_bits: bits,
            ..many_groups().zf
        },
        ..many_groups()
    };
    let (c18, c20) = (
        at(18).chain_config(&[(1, 27)]),
        at(20).chain_config(&[(1, 27)]),
    );
    assert_eq!(
        (
            c18.num_queries,
            c18.grind.query,
            c20.num_queries,
            c20.grind.query
        ),
        (229, 18, 225, 20)
    );
    let elf = asm_elf_bytes("sub");
    let options = options(MaxRowsConfig::small(), 3);
    let ground18 = prove(&elf, &at(18), &options);
    assert!(verify(&ground18, &elf, &at(18)), "the control at 18 bits");
    assert!(
        !verify(&ground18, &elf, &at(20)),
        "a verifier at 20 bits must refuse a block ground at 18"
    );
    let ground20 = prove(&elf, &at(20), &options);
    assert!(verify(&ground20, &elf, &at(20)), "the control at 20 bits");
    assert!(
        !verify(&ground20, &elf, &at(18)),
        "a verifier at 18 bits must refuse a block ground at 20"
    );
}

/// ★ `LAMBDA_VM_ZF_WHIR_BLOWUP` on the block: the base code's rate, and the Q
/// the rule charges for it, are the verifier's constants, never the proof's. A
/// block at blowup 2 verifies under blowup 2 and is refused by a verifier at
/// blowup 4; a block at blowup 4 is refused by a verifier at blowup 2.
#[test]
fn a_block_at_blowup_2_is_refused_by_a_verifier_at_blowup_4() {
    let at = |blowup: u8| BlockFormat {
        zf: ZfFormat {
            whir_blowup: blowup,
            ..many_groups().zf
        },
        ..many_groups()
    };
    let (c4, c2) = (
        at(4).chain_config(&[(1, 27)]),
        at(2).chain_config(&[(1, 27)]),
    );
    assert_eq!(
        (c4.log_blowup, c4.num_queries, c2.log_blowup, c2.num_queries),
        (2, 114, 1, 229)
    );
    let elf = asm_elf_bytes("sub");
    let options = options(MaxRowsConfig::small(), 3);
    let half = prove(&elf, &at(2), &options);
    assert!(verify(&half, &elf, &at(2)), "the control at blowup 2");
    assert!(
        !verify(&half, &elf, &at(4)),
        "a verifier at blowup 4 must refuse a block at blowup 2"
    );
    let today = prove(&elf, &at(4), &options);
    assert!(verify(&today, &elf, &at(4)), "the control at blowup 4");
    assert!(
        !verify(&today, &elf, &at(2)),
        "a verifier at blowup 2 must refuse a block at blowup 4"
    );
}

/// The bytes of a small block at each blowup, for a comparison across builds:
/// a card build and a host build of one sha, each run as its own process under
/// `LAMBDA_VM_FIXED_TRACE_HASH=1` and `LAMBDA_VM_DETERMINISTIC_GRIND=1`, must
/// print the same `RATE BLOCK DIGEST` lines (the card's commit, folds,
/// successors and openings prove the host's bytes at blowup 2 as at 4). Each
/// proof also verifies here.
#[test]
#[ignore = "prints proof digests for a comparison across builds; box tier"]
fn a_block_at_each_blowup_prints_its_proof_bytes() {
    let elf = asm_elf_bytes("all_instructions_64");
    let options = options(MaxRowsConfig::small(), 16);
    for blowup in crate::zf_format::WHIR_BLOWUPS {
        let format = BlockFormat {
            zf: ZfFormat {
                whir_blowup: blowup,
                ..many_groups().zf
            },
            ..many_groups()
        };
        let proof = prove(&elf, &format, &options);
        assert!(verify(&proof, &elf, &format), "blowup {blowup}");
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("serialize");
        println!(
            "RATE BLOCK DIGEST blowup={blowup} groups={} bytes={} blake3={}",
            groups_of(&proof),
            bytes.len(),
            &blake3::hash(&bytes).to_hex()[..32]
        );
    }
}

/// The block's shape: many chunks of every table, spread over several groups,
/// each argued and opened on its own fork.
#[test]
fn a_program_spread_over_many_groups_proves_and_verifies() {
    let elf = asm_elf_bytes("all_instructions_64");
    let proof = prove(&elf, &many_groups(), &options(MaxRowsConfig::small(), 16));
    assert!(groups_of(&proof) >= 3, "{} groups", groups_of(&proof));
    assert!(verify(&proof, &elf, &many_groups()));
    // The groups are the verifier's: the same proof under another grouping is
    // refused.
    assert!(!verify(&proof, &elf, &one_group()));
}

/// KECCAK_RND cut into tables of 2^3 rows: several chunks of one AIR, whose
/// bus sums add up to the whole table's.
#[test]
fn keccak_rnd_split_into_chunks_proves_and_verifies() {
    let elf = asm_elf_bytes("test_keccak");
    let proof = prove(&elf, &many_groups(), &options(MaxRowsConfig::small(), 3));
    assert!(
        proof.table_counts.keccak_rnd >= 2,
        "{} KECCAK_RND tables",
        proof.table_counts.keccak_rnd
    );
    assert!(verify(&proof, &elf, &many_groups()));
}

/// ★ The hold between the phases moves no byte: the same traces proved with
/// every tree level kept and with most of them dropped (re-hashed from the
/// rebuilt codewords at each query) give the same proof, byte for byte — when
/// the grind is deterministic (`LAMBDA_VM_DETERMINISTIC_GRIND`, read once per
/// process, so run this test alone to see the bytes). Otherwise the nonces a
/// parallel grind finds differ between any two proves and only the verdicts
/// are compared; the byte identity of a revived opening is also pinned without
/// grinding by `stacked_eval`'s `a_retired_and_revived_stack_opens_to_the_same_bytes`.
#[test]
fn the_kept_tree_depth_moves_no_byte_of_the_proof() {
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let program = Elf::load(&elf).expect("the ELF loads");
    let logs = Executor::new(&program, Vec::new())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs;
    let mut traces = crate::tables::trace_builder::Traces::from_elf_and_logs(
        &program,
        &logs,
        &MaxRowsConfig::small(),
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("the traces build");
    block_whir::split_keccak_rnd(&mut traces, 16, CutPadding::default()).expect("the split");
    let mut proved = |drop_levels: usize| {
        let mut o = options(MaxRowsConfig::small(), 16);
        o.drop_levels = multilinear::whir_commit::TreeDrop::uniform(drop_levels);
        let proof = block_whir::prove_traces(
            &program,
            &elf,
            &mut traces,
            &ProofOptions::default_test_options(),
            &format,
            &o,
            &Deviations::default(),
            false,
            &|_, _| {},
            &mut Default::default(),
        )
        .expect("prove");
        assert!(verify(&proof, &elf, &format), "drop {drop_levels}");
        rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .expect("serialize")
            .to_vec()
    };
    let whole = proved(0);
    let (three, eight, all) = (proved(3), proved(8), proved(64));
    if crypto::grinding::deterministic() {
        assert_eq!(whole, three);
        assert_eq!(whole, eight);
        assert_eq!(whole, all);
    }
}

/// ★ Narrow storage moves no byte of the proof: the same traces proved with
/// every group held wide between the phases and held narrow (packed on the
/// host here, into the bytes the card packs) give the same proof — byte for
/// byte when the grind is deterministic (`LAMBDA_VM_DETERMINISTIC_GRIND`, read
/// once per process, so run this test alone to see the bytes; otherwise only
/// the verdicts are compared) — under the per-table and the batched argue.
/// With no card, every reader of a narrow table widens it on the host.
#[test]
fn narrow_storage_moves_no_byte_of_the_proof() {
    use executor::elf::Elf;
    use executor::vm::execution::Executor;
    use stark::multilinear_block::Narrowing;

    let elf = asm_elf_bytes("all_instructions_64");
    let program = Elf::load(&elf).expect("the ELF loads");
    let logs = Executor::new(&program, Vec::new())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs;
    let mut traces = crate::tables::trace_builder::Traces::from_elf_and_logs(
        &program,
        &logs,
        &MaxRowsConfig::small(),
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("the traces build");
    block_whir::split_keccak_rnd(&mut traces, 16, CutPadding::default()).expect("the split");
    for argue in [ArgueFormat::PerTable, ArgueFormat::BATCHED] {
        let format = BlockFormat {
            argue,
            ..many_groups()
        };
        let mut proved = |narrow: Narrowing| {
            let mut o = options(MaxRowsConfig::small(), 16);
            o.narrow = narrow;
            let mut stamps = Default::default();
            let proof = block_whir::prove_traces(
                &program,
                &elf,
                &mut traces,
                &ProofOptions::default_test_options(),
                &format,
                &o,
                &Deviations::default(),
                false,
                &|_, _| {},
                &mut stamps,
            )
            .expect("prove");
            assert!(verify(&proof, &elf, &format), "{narrow:?} {argue:?}");
            let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
                .expect("serialize")
                .to_vec();
            (bytes, stamps)
        };
        let (wide, wide_stamps) = proved(Narrowing::Wide);
        let (narrow, narrow_stamps) = proved(Narrowing::Host);
        let packed = |s: &block_whir::BlockStamps| -> usize {
            s.groups.iter().map(|g| g.packed_tables).sum()
        };
        assert_eq!(packed(&wide_stamps), 0);
        assert_eq!(
            packed(&narrow_stamps),
            narrow_stamps.tables,
            "every table packed"
        );
        // A process-wide count: other tests add to it, never take from it.
        assert!(narrow_stamps.host_widens.0 > 0, "no card: the host widens");
        if crypto::grinding::deterministic() {
            assert_eq!(wide, narrow, "{argue:?}");
        }
    }
}

/// ★ Uploading ahead moves no byte of the proof: the same traces proved with
/// each group's columns put on the card beside the previous group's commit and
/// put there after it give the same proof — byte for byte when the grind is
/// deterministic (`LAMBDA_VM_DETERMINISTIC_GRIND`, read once per process, so
/// run this test alone to see the bytes; otherwise only the verdicts are
/// compared) — under the per-table and the batched argue. Each group's paid
/// upload is at most its upload, and without the overlap all of it.
#[test]
fn uploading_ahead_moves_no_byte_of_the_proof() {
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    let elf = asm_elf_bytes("all_instructions_64");
    let program = Elf::load(&elf).expect("the ELF loads");
    let logs = Executor::new(&program, Vec::new())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs;
    let mut traces = crate::tables::trace_builder::Traces::from_elf_and_logs(
        &program,
        &logs,
        &MaxRowsConfig::small(),
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("the traces build");
    block_whir::split_keccak_rnd(&mut traces, 16, CutPadding::default()).expect("the split");
    for argue in [ArgueFormat::PerTable, ArgueFormat::BATCHED] {
        let format = BlockFormat {
            argue,
            ..many_groups()
        };
        let mut proved = |upload_ahead: bool| {
            let mut o = options(MaxRowsConfig::small(), 16);
            o.upload_ahead = upload_ahead;
            let mut stamps = block_whir::BlockStamps::default();
            let proof = block_whir::prove_traces(
                &program,
                &elf,
                &mut traces,
                &ProofOptions::default_test_options(),
                &format,
                &o,
                &Deviations::default(),
                false,
                &|_, _| {},
                &mut stamps,
            )
            .expect("prove");
            assert!(
                verify(&proof, &elf, &format),
                "ahead {upload_ahead} {argue:?}"
            );
            assert!(stamps.groups.len() >= 3, "{} groups", stamps.groups.len());
            for g in &stamps.groups {
                assert!(
                    g.upload_paid <= g.upload_a + 1e-9,
                    "paid {} of {}",
                    g.upload_paid,
                    g.upload_a
                );
                if !upload_ahead {
                    assert_eq!(
                        g.upload_paid, g.upload_a,
                        "nothing hidden without the overlap"
                    );
                }
            }
            rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
                .expect("serialize")
                .to_vec()
        };
        let after = proved(false);
        let ahead = proved(true);
        if crypto::grinding::deterministic() {
            assert_eq!(after, ahead, "{argue:?}");
        }
    }
}

/// The streamed build uploading ahead: each group's columns go up beside the
/// previous group's commit as the groups arrive, and the proof verifies.
#[test]
fn a_streamed_block_uploading_ahead_proves_and_verifies() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    for upload_ahead in [false, true] {
        let mut o = streamed(MaxRowsConfig::small(), 16, 3);
        o.upload_ahead = upload_ahead;
        let proof = prove(&elf, &format, &o);
        assert!(groups_of(&proof) >= 3, "{} groups", groups_of(&proof));
        assert!(verify(&proof, &elf, &format), "ahead {upload_ahead}");
    }
}

/// The streamed build held narrow: every table packed exactly once — by its
/// group as the group is committed during the collect, by the finish (the
/// rest's tables, held narrow from the start), or, under G-pack, by its
/// generator (each streamed chunk written packed and laid out narrow, so its
/// group's commit finds it packed) — with G-pack off and on, and the proof
/// verifies.
#[test]
fn a_streamed_block_held_narrow_proves_and_verifies() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    for gpack in [false, true] {
        let mut o = streamed(MaxRowsConfig::small(), 16, 3);
        o.narrow = stark::multilinear_block::Narrowing::Host;
        o.gpack = gpack;
        let (proof, stamps) = prove_block_whir_with(
            &elf,
            &[],
            &ProofOptions::default_test_options(),
            &format,
            &o,
            &Deviations::default(),
        )
        .expect("prove");
        assert!(groups_of(&proof) >= 3, "{} groups", groups_of(&proof));
        assert!(o.pack_finished && stamps.layout.packed_rest.0 > 0);
        // The chunks G-pack wrote packed: every streamed chunk, counted by this
        // block (`BlockStamps::streamed`). G-pack's own counts, the BLOCK GPACK
        // line's, are the process's and split the chunks and the finish's
        // tables by whether a kind's width hint was there yet, so they bound
        // the chunks from above and no more. A `debug-checks` build writes
        // every trace wide (`gpack::Plan`).
        let chunks = stamps.streamed.1;
        assert!(chunks > 0, "the build streamed");
        let written = if gpack && !cfg!(feature = "debug-checks") {
            assert!(
                stamps.layout.gpack.0 && stamps.layout.gpack.1.iter().sum::<usize>() >= chunks,
                "G-pack built every streamed chunk: {:?}",
                stamps.layout.gpack
            );
            chunks
        } else {
            0
        };
        assert_eq!(
            stamps.groups.iter().map(|g| g.packed_tables).sum::<usize>()
                + stamps.layout.packed_rest.0
                + written,
            stamps.tables,
            "gpack {gpack}: every table packed exactly once"
        );
        assert!(stamps.report().contains("BLOCK NARROW: Host"));
        assert!(verify(&proof, &elf, &format), "gpack {gpack}");
    }
}

/// The memory log (`BlockOptions::memlog`) moves no byte of the proof — the
/// bytes are compared when the grind is deterministic
/// (`LAMBDA_VM_DETERMINISTIC_GRIND`, read once per process), the verdicts
/// always — and its accounting closes: when the proof is done every term in
/// flight and the committed tables are back at zero, and what is left is the
/// prepared columns.
#[test]
fn a_memory_log_moves_no_byte_and_its_terms_close() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let proved = |memlog: bool, workers: usize| {
        let mut o = streamed(MaxRowsConfig::small(), 16, 3);
        o.memlog = memlog;
        // Laid out on worker threads, the rest packed as it is laid out.
        o.layout_workers = workers;
        o.pack_rest_as_laid_out = workers > 0;
        let (proof, stamps) = prove_block_whir_with(
            &elf,
            &[],
            &ProofOptions::default_test_options(),
            &format,
            &o,
            &Deviations::default(),
        )
        .expect("prove");
        assert!(verify(&proof, &elf, &format), "memlog {memlog}");
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .expect("serialize")
            .to_vec();
        (bytes, stamps)
    };
    let (off, off_stamps) = proved(false, 0);
    let (on, on_stamps) = proved(true, 0);
    let (_, workers_stamps) = proved(true, 3);
    assert!(off_stamps.mem_terms.is_empty());
    for stamps in [&on_stamps, &workers_stamps] {
        terms_close(stamps);
    }
    if crypto::grinding::deterministic() {
        assert_eq!(off, on, "the memory log moved a proof byte");
    }
}

/// Every memory term in flight, and the committed tables as held, are zero once
/// the proof is done; the prepared columns are left.
fn terms_close(stamps: &block_whir::BlockStamps) {
    let term = |name: &str| -> usize {
        stamps
            .mem_terms
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("no term {name}"))
            .1
    };
    for name in [
        "exec",
        "logs",
        "walk",
        "walked",
        "builder",
        "image",
        "jobs",
        "laying",
        "open",
        "sent",
        "rest",
        "rest_laid",
        "committing",
        "ahead",
        "pack_wide",
        "pack_ready",
        "tops",
        // Phase B lets the tables go; a release that took more than was held
        // would wrap past zero.
        "held_narrow",
        "held_wide",
    ] {
        assert_eq!(
            term(name),
            0,
            "{name} is still counted when the proof is done"
        );
    }
    assert!(term("prepared") > 0, "the prepared columns are held");
}

/// The rest's waves: consecutive, in order, each within the budget unless one
/// item alone is larger, and every item in exactly one.
#[test]
fn the_rest_is_cut_into_waves_within_the_budget() {
    let sizes = [3usize, 4, 2, 9, 1, 1, 5, 0, 6];
    for budget in [1usize, 5, 7, 10, 100] {
        let waves = block_whir::waves(sizes.to_vec(), budget, |&b| b);
        assert_eq!(
            waves.iter().flatten().copied().collect::<Vec<_>>(),
            sizes,
            "budget {budget}: the order or an item changed"
        );
        for wave in &waves {
            let sum: usize = wave.iter().sum();
            assert!(!wave.is_empty());
            assert!(
                sum <= budget || wave.len() == 1,
                "budget {budget}: {wave:?}"
            );
        }
        // A wave closes only when the next item would take it past the budget.
        for pair in waves.windows(2) {
            let sum: usize = pair[0].iter().sum();
            assert!(sum + pair[1][0] > budget, "budget {budget}: {pair:?}");
        }
    }
    assert!(block_whir::waves(Vec::<usize>::new(), 4, |&b| b).is_empty());
}

/// The rest laid out in waves, KECCAK_RND built as its tables at the finish
/// (b1: BlockOptions::rest_layout_bytes, finish_keccak_rnd_chunks) and the
/// finish's tables packed as they are built, laid out narrow and committed from
/// the card (b2: BlockOptions::pack_finished) move no byte: the same tables,
/// partition and statement as with all of them off, inline and on the worker
/// layout, and the proof bytes equal under the deterministic grind.
#[test]
fn the_rest_in_waves_and_the_finish_packed_move_no_byte() {
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    for workers in [0usize, 3] {
        // (b1, b2)
        let proved = |b1: bool, b2: bool| {
            // Rows of 2^5, the least a KECCAK_RND table takes: it spans several.
            let mut o = streamed(MaxRowsConfig::small(), 5, 3);
            o.layout_workers = workers;
            o.pack_rest_as_laid_out = workers > 0;
            o.finish_keccak_rnd_chunks = b1;
            o.rest_layout_bytes = b1.then_some(1 << 16);
            o.pack_finished = b2;
            let (proof, stamps) = prove_block_whir_with(
                &elf,
                &[],
                &ProofOptions::default_test_options(),
                &format,
                &o,
                &Deviations::default(),
            )
            .expect("prove");
            assert!(
                verify(&proof, &elf, &format),
                "b1 {b1}, b2 {b2}, {workers} workers"
            );
            (proof, stamps)
        };
        let (before, before_stamps) = proved(false, false);
        assert!(
            before.table_counts.keccak_rnd > 1,
            "KECCAK_RND is one table: nothing was split"
        );
        assert_eq!(before_stamps.layout.packed_rest.0, 0);
        for (b1, b2) in [(true, false), (false, true), (true, true)] {
            let (after, stamps) = proved(b1, b2);
            let what = format!("b1 {b1}, b2 {b2}, {workers} workers");
            assert_eq!(before.groups, after.groups, "{what}: the partition");
            assert_eq!(before.table_num_vars, after.table_num_vars, "{what}");
            assert_eq!(
                format!("{:?}", before.table_counts),
                format!("{:?}", after.table_counts),
                "{what}"
            );
            if b2 {
                assert!(
                    stamps.layout.packed_rest.0 > 0,
                    "{what}: no table was laid out narrow"
                );
            }
            if crypto::grinding::deterministic() {
                let bytes = |p: &BlockWhirProof| {
                    rkyv::to_bytes::<rkyv::rancor::Error>(p)
                        .expect("serialize")
                        .to_vec()
                };
                assert_eq!(bytes(&before), bytes(&after), "{what}: the proof");
            }
        }
    }
}

/// ★ G-pack (BlockOptions::gpack: the streamed chunks written packed and laid
/// out narrow, the finish's tables written packed) moves no byte: the same
/// partition, tables and statement as without it, inline and on the worker
/// layout, every streamed kind past its first chunk written packed, and the
/// proof bytes equal under the deterministic grind.
#[test]
fn gpack_moves_no_byte() {
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    for workers in [0usize, 3] {
        let proved = |gpack: bool| {
            let mut o = streamed(MaxRowsConfig::small(), 5, 3);
            o.layout_workers = workers;
            o.gpack = gpack;
            let (proof, stamps) = prove_block_whir_with(
                &elf,
                &[],
                &ProofOptions::default_test_options(),
                &format,
                &o,
                &Deviations::default(),
            )
            .expect("prove");
            assert!(
                verify(&proof, &elf, &format),
                "gpack {gpack}, {workers} workers"
            );
            (proof, stamps)
        };
        let (before, before_stamps) = proved(false);
        let (after, stamps) = proved(true);
        let what = format!("{workers} workers");
        assert!(!before_stamps.layout.gpack.0, "{what}");
        let [direct, narrowed, again, _] = stamps.layout.gpack.1;
        assert!(
            stamps.layout.gpack.0 && direct + narrowed + again > 0,
            "{what}: {:?}",
            stamps.layout.gpack
        );
        assert_eq!(before.groups, after.groups, "{what}: the partition");
        assert_eq!(before.table_num_vars, after.table_num_vars, "{what}");
        assert_eq!(
            format!("{:?}", before.table_counts),
            format!("{:?}", after.table_counts),
            "{what}"
        );
        if crypto::grinding::deterministic() {
            let bytes = |p: &BlockWhirProof| {
                rkyv::to_bytes::<rkyv::rancor::Error>(p)
                    .expect("serialize")
                    .to_vec()
            };
            assert_eq!(bytes(&before), bytes(&after), "{what}: the proof");
        }
    }
}

/// ★ Spilling the held tables moves no byte: with the spill on, phase A hands
/// every group past the first two to the store, phase B reads each back once
/// in group order, and the proof
/// verifies with the same partition and statement as with every table held,
/// its bytes equal under the deterministic grind.
#[test]
fn a_block_spilled_to_disk_proves_the_same_bytes() {
    use crate::block_whir::BlockSpillPolicy;

    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let proved = |spill: BlockSpillPolicy| {
        let mut o = streamed(MaxRowsConfig::small(), 5, 3);
        o.spill = spill;
        let (proof, stamps) = prove_block_whir_with(
            &elf,
            &[],
            &ProofOptions::default_test_options(),
            &format,
            &o,
            &Deviations::default(),
        )
        .expect("prove");
        assert!(verify(&proof, &elf, &format), "spill {spill:?}");
        (proof, stamps)
    };
    let (held, held_stamps) = proved(BlockSpillPolicy::Off);
    assert!(held_stamps.spill.is_none() && held_stamps.spill_stats.is_none());
    assert!(
        held.groups.len() > 2,
        "{} groups: none past the first two",
        held.groups.len()
    );
    let (spilled, stamps) = proved(BlockSpillPolicy::Always);
    let stats = stamps.spill_stats.expect("the store's counters");
    assert!(stats.slots > 0, "nothing spilled: {stats}");
    assert_eq!(stats.mismatches, 0, "{stats}");
    assert_eq!(stats.failure, None, "{stats}");
    assert_eq!(
        stats.reads + stats.memory_reads,
        stats.slots,
        "each slot read back once: {stats}"
    );
    assert_eq!(held.groups, spilled.groups, "the partition");
    assert_eq!(held.table_num_vars, spilled.table_num_vars);
    if crypto::grinding::deterministic() {
        let bytes = |p: &BlockWhirProof| {
            rkyv::to_bytes::<rkyv::rancor::Error>(p)
                .expect("serialize")
                .to_vec()
        };
        assert_eq!(bytes(&held), bytes(&spilled), "the proof");
    }
}

/// The policy decides per table: a resident budget the block fits in spills
/// nothing, one of zero spills every table past the first two groups, and
/// both verify.
#[test]
fn a_spill_budget_decides_what_goes() {
    use crate::block_whir::BlockSpillPolicy;

    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    // The walk's end hands the parked tables to the store once the peak
    // passes the pressure share of the host's target: out of reach, so the
    // budget alone decides on any host.
    let spilled = |policy: BlockSpillPolicy| {
        let mut o = streamed(MaxRowsConfig::small(), 5, 3);
        o.spill = policy;
        let (proof, stamps) = prove_block_whir_with(
            &elf,
            &[],
            &ProofOptions::default_test_options(),
            &format,
            &o,
            &Deviations {
                spill_target: Some(u64::MAX),
                ..Deviations::default()
            },
        )
        .expect("prove");
        assert!(verify(&proof, &elf, &format), "{policy:?}");
        stamps.spill_stats.expect("the store's counters").slots
    };
    assert_eq!(
        spilled(BlockSpillPolicy::Budget(1 << 40)),
        0,
        "a budget it fits in"
    );
    assert!(spilled(BlockSpillPolicy::Budget(0)) > 0, "a budget of zero");
}

/// A byte flipped in a spilled table on disk is caught by the store's digest
/// when phase B reads it back: the prover refuses, naming the table, instead
/// of proving over other columns.
#[test]
fn a_byte_flipped_in_a_spilled_table_is_refused() {
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let mut o = streamed(MaxRowsConfig::small(), 5, 3);
    o.spill = crate::block_whir::BlockSpillPolicy::Always;
    let refused = prove_block_whir_with(
        &elf,
        &[],
        &ProofOptions::default_test_options(),
        &format,
        &o,
        &Deviations {
            spilled_byte: true,
            ..Deviations::default()
        },
    );
    let Err(crate::Error::Prover(why)) = refused else {
        panic!("a flipped spilled byte was proved over");
    };
    assert!(
        why.contains("SpillFailed") && why.contains("not the ones written"),
        "{why}"
    );
}

/// A spilled table whose columns never come back is refused when phase B
/// reaches its group, before any reader meets it: a prover error naming the
/// table, not a proof over empty columns.
#[test]
fn a_spilled_table_that_does_not_come_back_is_refused() {
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let mut o = streamed(MaxRowsConfig::small(), 5, 3);
    o.spill = crate::block_whir::BlockSpillPolicy::Always;
    let refused = prove_block_whir_with(
        &elf,
        &[],
        &ProofOptions::default_test_options(),
        &format,
        &o,
        &Deviations {
            spilled_slot_lost: true,
            ..Deviations::default()
        },
    );
    let Err(crate::Error::Prover(why)) = refused else {
        panic!("a table that never came back was proved over");
    };
    assert!(
        why.contains("SpillFailed") && why.contains("still spilled"),
        "{why}"
    );
}

/// ★ Parked tables handed to the store at the walk's end move no byte: with a
/// budget the block fits in, every table past the first two groups is parked
/// on the host; the hand-off (here every parked one, whatever the forecast)
/// moves them to the store before the finish, phase B reads each back once,
/// and the proof verifies with the same partition, its bytes equal to the
/// held run's under the deterministic grind.
#[test]
fn parked_tables_handed_off_at_the_walk_prove_the_same_bytes() {
    use crate::block_whir::BlockSpillPolicy;

    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    // Without the deviation the hand-off is the forecast's, past the pressure
    // share of the host's target: out of reach, so no host's memory hands
    // off the run that must not.
    let proved = |spill: BlockSpillPolicy, hand_off_all: bool| {
        let mut o = streamed(MaxRowsConfig::small(), 5, 3);
        o.spill = spill;
        let (proof, stamps) = prove_block_whir_with(
            &elf,
            &[],
            &ProofOptions::default_test_options(),
            &format,
            &o,
            &Deviations {
                hand_off_all,
                spill_target: Some(u64::MAX),
                ..Deviations::default()
            },
        )
        .expect("prove");
        assert!(
            verify(&proof, &elf, &format),
            "{spill:?} hand-off {hand_off_all}"
        );
        (proof, stamps)
    };
    let (held, _) = proved(BlockSpillPolicy::Off, false);
    let (parked, parked_stamps) = proved(BlockSpillPolicy::Budget(1 << 40), false);
    assert_eq!(
        parked_stamps
            .spill_stats
            .expect("the store's counters")
            .slots,
        0,
        "parked and never handed off"
    );
    let (handed, stamps) = proved(BlockSpillPolicy::Budget(1 << 40), true);
    let stats = stamps.spill_stats.expect("the store's counters");
    assert!(stats.slots > 0, "nothing handed off: {stats}");
    assert_eq!(stats.mismatches, 0, "{stats}");
    assert_eq!(
        stats.reads + stats.memory_reads,
        stats.slots,
        "each slot read back once: {stats}"
    );
    let line = stamps.spill.expect("the spill's line");
    assert!(line.contains("hand-off at the walk's end"), "{line}");
    assert_eq!(held.groups, parked.groups, "the partition");
    assert_eq!(held.groups, handed.groups, "the partition");
    if crypto::grinding::deterministic() {
        let bytes = |p: &BlockWhirProof| {
            rkyv::to_bytes::<rkyv::rancor::Error>(p)
                .expect("serialize")
                .to_vec()
        };
        assert_eq!(bytes(&held), bytes(&parked), "the proof, parked");
        assert_eq!(bytes(&held), bytes(&handed), "the proof, handed off");
    }
}

/// A table the hand-off moved to the store whose columns never come back is
/// refused when phase B reaches its group: a prover error naming the table.
#[test]
fn a_handed_off_table_that_does_not_come_back_is_refused() {
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let mut o = streamed(MaxRowsConfig::small(), 5, 3);
    o.spill = crate::block_whir::BlockSpillPolicy::Budget(1 << 40);
    let refused = prove_block_whir_with(
        &elf,
        &[],
        &ProofOptions::default_test_options(),
        &format,
        &o,
        &Deviations {
            hand_off_all: true,
            spilled_slot_lost: true,
            ..Deviations::default()
        },
    );
    let Err(crate::Error::Prover(why)) = refused else {
        panic!("a handed-off table that never came back was proved over");
    };
    assert!(
        why.contains("SpillFailed") && why.contains("still spilled"),
        "{why}"
    );
}

/// A table the finish packed is laid out narrow only if its preprocessed
/// columns are the program's, word for word: one word changed in a packed
/// table's first preprocessed column and the layout refuses it.
#[test]
fn a_packed_table_with_a_wrong_preprocessed_column_is_refused() {
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    let elf = asm_elf_bytes("all_instructions_64");
    let program = Elf::load(&elf).expect("the ELF loads");
    let logs = Executor::new(&program, Vec::new())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs;
    let mut traces = crate::tables::trace_builder::Traces::from_elf_and_logs(
        &program,
        &logs,
        &MaxRowsConfig::small(),
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("the traces build");
    let opts = ProofOptions::default_test_options();
    let counts = traces.table_counts();
    let airs = crate::VmAirs::new(
        &program,
        &opts,
        false,
        &traces.page_configs,
        &counts,
        &crate::cpu_heights_from_traces(&traces),
        None,
        true,
        None,
        crate::test_utils::halted_register_commitment(&opts, &program, &traces.public_output_bytes)
            .preprocessed(),
    );
    let refs = airs.air_refs();
    let pairs = airs.air_trace_pairs(&mut traces);
    let (air, trace) = pairs
        .into_iter()
        .zip(refs.iter().copied())
        .map(|((_, trace, _), air)| (air, trace))
        .find(|(air, trace)| !air.precomputed_columns().is_empty() && trace.main_table.width != 0)
        .expect("a table with preprocessed columns");
    let shape = (
        trace.main_table.width,
        trace.main_table.height.trailing_zeros() as usize,
    );
    let mut honest = trace.clone();
    assert!(honest.pack_main_narrow(), "{}: packs", air.name());
    block_whir::table_of_narrow(air, &mut honest, shape).expect("the honest table lays out");
    let mut wrong = trace.clone();
    let word = *wrong.main_table.get(0, 0);
    wrong
        .main_table
        .set(0, 0, word + FieldElement::<crate::test_utils::F>::one());
    assert!(wrong.pack_main_narrow());
    let refused = block_whir::table_of_narrow(air, &mut wrong, shape)
        .err()
        .expect("a wrong preprocessed column is refused");
    assert!(
        format!("{refused:?}").contains("preprocessed column 0"),
        "{refused:?}"
    );
}

/// A narrow table whose width map is wrong widens to other words than were
/// committed, and the prover refuses at the opening: the recomputed codeword
/// does not hash to the kept tree top.
#[test]
fn a_wrong_narrow_width_map_is_refused_by_the_kept_tree() {
    let elf = asm_elf_bytes("all_instructions_64");
    let mut o = options(MaxRowsConfig::small(), 16);
    o.narrow = stark::multilinear_block::Narrowing::Host;
    let refused = prove_block_whir_with(
        &elf,
        &[],
        &ProofOptions::default_test_options(),
        &many_groups(),
        &o,
        &Deviations {
            narrow_width_map: true,
            ..Default::default()
        },
    );
    match refused {
        Err(e) => assert!(
            format!("{e:?}").contains("RecomputedCodewordMismatch"),
            "refused for another reason: {e:?}"
        ),
        Ok(_) => panic!("a wrong width map proved"),
    }
}

/// The same on a card: the group's tables packed there after the commit and
/// widened there again for phase B, one width map broken in between.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "box: needs a card"]
fn a_wrong_narrow_width_map_is_refused_on_the_card() {
    let elf = asm_elf_bytes("all_instructions_64");
    let o = options(MaxRowsConfig::small(), 16);
    let (honest, stamps) = prove_block_whir_with(
        &elf,
        &[],
        &ProofOptions::default_test_options(),
        &many_groups(),
        &o,
        &Deviations::default(),
    )
    .expect("prove");
    assert_eq!(
        stamps.groups.iter().map(|g| g.packed_tables).sum::<usize>(),
        stamps.tables,
        "every table packed on the card"
    );
    // The tables too small for the card's argue are read on the host.
    println!("NARROW CARD: host widens {:?}", stamps.host_widens);
    assert!(verify(&honest, &elf, &many_groups()));
    let refused = prove_block_whir_with(
        &elf,
        &[],
        &ProofOptions::default_test_options(),
        &many_groups(),
        &o,
        &Deviations {
            narrow_width_map: true,
            ..Default::default()
        },
    );
    match refused {
        Err(e) => assert!(
            format!("{e:?}").contains("RecomputedCodewordMismatch"),
            "refused for another reason: {e:?}"
        ),
        Ok(_) => panic!("a wrong width map proved"),
    }
}

#[test]
fn a_block_proof_survives_serialization() {
    let elf = asm_elf_bytes("sub");
    let proof = prove(&elf, &many_groups(), &options(MaxRowsConfig::small(), 16));
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("serialize");
    let back: BlockWhirProof =
        rkyv::from_bytes::<_, rkyv::rancor::Error>(&bytes).expect("deserialize");
    assert!(verify(&back, &elf, &many_groups()));
}

/// A group proved on another group's fork is refused: the index each fork
/// absorbs after `S_post` is what ties a group's messages to it.
#[test]
fn a_group_proved_on_another_fork_is_refused() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let swapped = prove_with(
        &elf,
        &format,
        &options(MaxRowsConfig::small(), 16),
        &Deviations {
            fork_of: Some(|g| g ^ 1),
            ..Default::default()
        },
    );
    assert!(groups_of(&swapped) >= 2);
    assert!(!verify(&swapped, &elf, &format));
}

/// A prover that leaves one KECCAK_RND chunk out — its rows, its argument and
/// its count, so every table it does prove is consistent — is refused on the
/// bus balance, which is checked once over every table of the block. The
/// chunk left out holds real rounds (the first): the last one of a small
/// program can be padding alone, which the bus does not see.
#[test]
fn a_block_missing_an_instance_is_refused_on_the_bus() {
    let elf = asm_elf_bytes("test_keccak");
    let format = many_groups();
    let o = options(MaxRowsConfig::small(), 3);
    let honest = prove(&elf, &format, &o);
    let short = prove_with(
        &elf,
        &format,
        &o,
        &Deviations {
            omit_first_keccak_rnd: true,
            ..Default::default()
        },
    );
    assert_eq!(
        short.table_counts.keccak_rnd + 1,
        honest.table_counts.keccak_rnd
    );
    assert!(!verify(&short, &elf, &format));
}

type Felt = FieldElement<crate::test_utils::F>;

/// The deviations alone: a block proved without the prover's own check of its
/// preprocessed columns verifies, so each refusal below is its tamper's.
#[test]
fn a_block_proved_without_the_provers_column_check_verifies() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let o = options(MaxRowsConfig::small(), 16);
    let proof = prove_with(
        &elf,
        &format,
        &o,
        &Deviations {
            unchecked_preprocessed: true,
            ..Default::default()
        },
    );
    assert!(
        proof.table_counts.cpu >= 2,
        "{} CPU tables",
        proof.table_counts.cpu
    );
    assert!(verify(&proof, &elf, &format));
}

/// ⛔ A CPU table's TIMESTAMP is the verifier's column (main's preprocessed
/// clock): a table with one timestamp moved, the clock a prover could
/// otherwise set row by row, is refused.
#[test]
fn a_block_with_a_moved_cpu_timestamp_is_refused() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let o = options(MaxRowsConfig::small(), 16);
    let moved = prove_with(
        &elf,
        &format,
        &o,
        &Deviations {
            unchecked_preprocessed: true,
            tamper_traces: Some(|traces| {
                let ts = crate::tables::cpu::cols::TIMESTAMP;
                let table = &mut traces.cpus[1].main_table;
                let value = *table.get(1, ts);
                table.set(1, ts, value + Felt::from(4u64));
            }),
            ..Default::default()
        },
    );
    assert!(!verify(&moved, &elf, &format));
}

/// ⛔ REGISTER's FINI is the halted file (main's final-register pin): a
/// block whose final register file is another one is refused.
#[test]
fn a_block_with_another_final_register_file_is_refused() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let o = options(MaxRowsConfig::small(), 16);
    let other = prove_with(
        &elf,
        &format,
        &o,
        &Deviations {
            unchecked_preprocessed: true,
            tamper_traces: Some(|traces| {
                let fini = crate::tables::register::cols::FINI;
                let table = &mut traces.register.main_table;
                let value = *table.get(2, fini);
                table.set(2, fini, value + Felt::one());
            }),
            ..Default::default()
        },
    );
    assert!(!verify(&other, &elf, &format));
}

/// ⛔ KECCAK's IS_WH messages need the table that receives them: a block
/// proved without it (its statement then counts none) is refused on the bus.
#[test]
fn a_block_without_its_is_wh_table_is_refused() {
    let elf = asm_elf_bytes("test_keccak");
    let format = many_groups();
    let o = options(MaxRowsConfig::small(), 16);
    let honest = prove(&elf, &format, &o);
    assert_eq!(honest.table_counts.is_wh, 1);
    assert!(verify(&honest, &elf, &format));
    let short = prove_with(
        &elf,
        &format,
        &o,
        &Deviations {
            tamper_traces: Some(|traces| traces.is_whs.clear()),
            ..Default::default()
        },
    );
    assert_eq!(short.table_counts.is_wh, 0);
    assert!(!verify(&short, &elf, &format));
}

/// ⛔ KECCAK's μ is a bit (main's multiplicity fix): a row at μ = −1 would
/// receive its IS_WH and IS_HALF messages instead of sending them, and could
/// cancel forged sends. A block with one is refused.
#[test]
fn a_block_with_a_negative_keccak_multiplicity_is_refused() {
    let elf = asm_elf_bytes("test_keccak");
    let format = many_groups();
    let o = options(MaxRowsConfig::small(), 16);
    let negative = prove_with(
        &elf,
        &format,
        &o,
        &Deviations {
            tamper_traces: Some(|traces| {
                let mu = crate::tables::keccak::cols::MU;
                traces.keccaks[0].main_table.set(0, mu, -Felt::one());
            }),
            ..Default::default()
        },
    );
    assert!(!verify(&negative, &elf, &format));
}

fn streamed(
    max_rows: MaxRowsConfig,
    keccak_rnd_rows_log2: usize,
    window_log2: usize,
) -> BlockOptions {
    let mut o = options(max_rows, keccak_rnd_rows_log2);
    o.window_log2 = Some(window_log2);
    o
}

/// ★ The build streamed into phase A: tables are packed into groups as their
/// chunks complete, the partition rides the statement, and the proof verifies.
#[test]
fn a_streamed_block_proves_and_verifies() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let proof = prove(&elf, &format, &streamed(MaxRowsConfig::small(), 16, 3));
    assert!(groups_of(&proof) >= 3, "{} groups", groups_of(&proof));
    // Streaming packs in arrival order, not table order.
    let order: Vec<u32> = proof.groups.iter().flatten().copied().collect();
    assert!(
        order.windows(2).any(|w| w[0] > w[1]),
        "the groups are table order"
    );
    assert!(verify(&proof, &elf, &format));
}

/// A host-only build (a box harness's measure of the host's pace) drains the
/// groups and tables the streamed prove commits, and commits none.
#[test]
fn a_host_only_build_drains_the_groups_the_prove_commits() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let options = streamed(MaxRowsConfig::small(), 16, 3);
    let proof = prove(&elf, &format, &options);
    let (stamps, drained) = block_whir::trace_block_host_only(
        &elf,
        &[],
        &ProofOptions::default_test_options(),
        &format,
        &options,
    )
    .expect("the host-only build");
    let tables: usize = proof.groups.iter().map(Vec::len).sum();
    assert!(
        drained.contains(&format!(
            "· {} groups · {tables} tables ·",
            groups_of(&proof)
        )),
        "{drained}"
    );
    assert!(stamps.phase_a > 0.0 && stamps.build > 0.0);
}

#[test]
fn a_streamed_block_with_keccak_chunks_proves_and_verifies() {
    let elf = asm_elf_bytes("test_keccak");
    let format = many_groups();
    let proof = prove(&elf, &format, &streamed(MaxRowsConfig::small(), 3, 4));
    assert!(proof.table_counts.keccak_rnd >= 2);
    assert!(verify(&proof, &elf, &format));
}

/// The streamed build cuts ECDAS at the run's end like the whole-run build:
/// test_ecsm_multi's 42 ECDAS steps (64 rows padded) in 16-row tables,
/// windows of 2^4 cycles; at the default the fourth table, padding rows alone,
/// is not built.
#[test]
fn a_streamed_block_with_ecdas_chunks_proves_and_verifies() {
    let elf = asm_elf_bytes("test_ecsm_multi");
    let format = many_groups();
    let mut o = streamed(MaxRowsConfig::small(), 16, 4);
    o.ecdas_rows_log2 = 4;
    let proof = prove(&elf, &format, &o);
    assert_eq!(proof.table_counts.ecdas, 3);
    assert!(verify(&proof, &elf, &format));
}

/// The same with KECCAK_RND's chunks streamed as their ops arrive (chunks of
/// 2^5 rows, the builder's smallest): the same table count as built at the
/// end, and the proof verifies.
#[test]
fn a_streamed_block_with_keccak_rnd_streamed_proves_and_verifies() {
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let at_end = prove(&elf, &format, &streamed(MaxRowsConfig::small(), 5, 3));
    let mut o = streamed(MaxRowsConfig::small(), 5, 3);
    o.stream_keccak_rnd = true;
    let proof = prove(&elf, &format, &o);
    assert!(proof.table_counts.keccak_rnd >= 2);
    assert_eq!(
        proof.table_counts.keccak_rnd,
        at_end.table_counts.keccak_rnd
    );
    assert!(verify(&proof, &elf, &format));
}

/// The same with each window's MEMW-derived LT ops streamed (the block path's
/// other LT chunking): the same LT chunk count as without, and the proof
/// verifies — every LT constraint and the bus hold over the chunks.
#[test]
fn a_streamed_block_with_memw_lt_streamed_proves_and_verifies() {
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let plain = prove(&elf, &format, &streamed(MaxRowsConfig::small(), 16, 3));
    let mut o = streamed(MaxRowsConfig::small(), 16, 3);
    o.stream_memw_lt = true;
    let proof = prove(&elf, &format, &o);
    assert_eq!(proof.table_counts.lt, plain.table_counts.lt);
    assert!(verify(&proof, &elf, &format));
}

/// ★ With the builder dropping each streamed chunk's ops as it leaves (D-MEMORY
/// M1), alone and with the MEMW-derived LT ops streamed: the same groups and
/// table counts as keeping them, and the proof verifies.
#[test]
fn a_streamed_block_with_dropped_ops_proves_and_verifies() {
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    for stream_memw_lt in [false, true] {
        let mut keep = streamed(MaxRowsConfig::small(), 5, 3);
        keep.stream_memw_lt = stream_memw_lt;
        let kept = prove(&elf, &format, &keep);
        let mut o = keep.clone();
        o.drop_streamed_ops = true;
        let proof = prove(&elf, &format, &o);
        assert!(groups_of(&proof) >= 3, "{} groups", groups_of(&proof));
        assert_eq!(proof.groups, kept.groups, "LT streamed {stream_memw_lt}");
        assert_eq!(
            format!("{:?}", proof.table_counts),
            format!("{:?}", kept.table_counts)
        );
        assert!(verify(&proof, &elf, &format));
    }
}

/// A streamed prove with the shadow regenerator: its proof, its stamps.
fn shadowed(
    elf: &[u8],
    format: &BlockFormat,
    o: &BlockOptions,
    deviations: &Deviations,
) -> (BlockWhirProof, block_whir::BlockStamps) {
    let mut o = o.clone();
    o.regen = Some(block_whir::regen::RegenMode::Shadow);
    prove_block_whir_with(
        elf,
        &[],
        &ProofOptions::default_test_options(),
        format,
        &o,
        deviations,
    )
    .expect("prove")
}

/// ★ The shadow regenerator (D-WHIR-NODISK N1) rebuilds every streamed chunk
/// from a fresh execution of the run and moves no byte: phase A records a
/// recipe per streamed chunk, each digested as it was committed; phase B
/// regenerates every one with no mismatch, failure or miss, and plans them in
/// group order and rest first; the proof has the partition of the prove
/// without the shadow, verifies, and its bytes are equal under the
/// deterministic grind. Inline and on three layout threads, the streamed ops
/// kept and dropped.
#[test]
fn the_shadow_regenerates_every_streamed_chunk_and_moves_no_byte() {
    let format = many_groups();
    for name in ["test_keccak_multi"] {
        let elf = asm_elf_bytes(name);
        for (workers, drop_ops) in [(0, false), (3, true)] {
            let mut o = streamed(MaxRowsConfig::small(), 5, 3);
            o.layout_workers = workers;
            o.drop_streamed_ops = drop_ops;
            o.regen = Some(block_whir::regen::RegenMode::Off);
            let (plain, plain_stamps) = prove_block_whir_with(
                &elf,
                &[],
                &ProofOptions::default_test_options(),
                &format,
                &o,
                &Deviations::default(),
            )
            .expect("prove");
            assert!(plain_stamps.regen.is_none(), "off records nothing");
            let (proof, stamps) = shadowed(&elf, &format, &o, &Deviations::default());
            let at = format!("{name}, {workers} layout threads, ops dropped {drop_ops}");
            let r = stamps.regen.as_ref().expect("the shadow's readout");
            let lines = r.lines.join("\n");
            let streamed = stamps.streamed.1;
            assert!(streamed > 0, "{at}: nothing streamed");
            assert_eq!(r.recipes, streamed, "{at}: a recipe a chunk\n{lines}");
            assert_eq!(r.digested, streamed, "{at}: every recipe digested\n{lines}");
            assert_eq!(r.stray, 0, "{at}\n{lines}");
            assert_eq!(r.regenerated, streamed, "{at}: every chunk\n{lines}");
            assert_eq!(
                (r.mismatches, r.failed, r.missing),
                (0, 0, 0),
                "{at}\n{lines}"
            );
            assert_eq!(r.error, None, "{at}\n{lines}");
            assert!(r.would_wait_rest_first.is_some(), "{at}\n{lines}");
            assert!(
                lines.contains("BLOCK REGEN plan (rest first)")
                    && lines.contains("BLOCK REGEN plan (group order)"),
                "{lines}"
            );
            assert_eq!(proof.groups, plain.groups, "{at}: the partition");
            assert!(verify(&proof, &elf, &format), "{at}");
            if crypto::grinding::deterministic() {
                let bytes = |p: &BlockWhirProof| {
                    rkyv::to_bytes::<rkyv::rancor::Error>(p)
                        .expect("serialize")
                        .to_vec()
                };
                assert_eq!(bytes(&plain), bytes(&proof), "{at}: the proof");
            }
        }
    }
}

/// ★ The shadow's check is load-bearing: a regenerator whose slicer drops the
/// first CPU op (or MEMW_R row) rebuilds a chunk that is not the one
/// committed, which the shadow reports as a mismatch — never a panic, and
/// never a proof over it: the proof still verifies, and the rest-first plan
/// has no would-wait to give.
#[test]
fn a_wrong_regeneration_is_reported_never_proved() {
    use crate::tables::trace_builder::StreamTable;
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let o = streamed(MaxRowsConfig::small(), 5, 3);
    for table in [StreamTable::Cpu, StreamTable::MemwRegister] {
        let (proof, stamps) = shadowed(
            &elf,
            &format,
            &o,
            &Deviations {
                regen_drop_one_op: Some(table),
                ..Deviations::default()
            },
        );
        let r = stamps.regen.expect("the shadow's readout");
        let lines = r.lines.join("\n");
        assert!(r.mismatches > 0, "{table:?}: no mismatch\n{lines}");
        assert!(
            lines.contains(&format!("BLOCK REGEN MISMATCH {table:?}[0]")),
            "{lines}"
        );
        assert_eq!(r.would_wait_rest_first, None, "{lines}");
        assert!(verify(&proof, &elf, &format), "{table:?}");
    }
}

/// The shadow runs only where the regenerator rebuilds every streamed chunk:
/// with KECCAK_RND's chunks streamed, or the MEMW-derived LT ops, or the
/// chunks generated 64-bit, nothing is recorded and the prove is as without
/// it.
#[test]
fn the_shadow_stays_off_where_it_cannot_rebuild_the_chunks() {
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let base = streamed(MaxRowsConfig::small(), 5, 3);
    let mut keccak_rnd = base.clone();
    keccak_rnd.stream_keccak_rnd = true;
    let mut memw_lt = base.clone();
    memw_lt.stream_memw_lt = true;
    let mut wide = base.clone();
    wide.gpack = false;
    for (what, o) in [
        ("KECCAK_RND", keccak_rnd),
        ("MEMW-derived LT", memw_lt),
        ("64-bit", wide),
    ] {
        let (proof, stamps) = shadowed(&elf, &format, &o, &Deviations::default());
        assert!(stamps.regen.is_none(), "{what}: recorded");
        assert!(verify(&proof, &elf, &format), "{what}");
    }
}

/// ★ Phase B's order moves no byte (D-WHIR-NODISK §2.3): each group proves on
/// its own fork `S_post ‖ g`, so taking the groups reversed, odd ones first, or
/// a group's spilled columns read back out of group order, proves the bytes of
/// group order under the deterministic grind, with the same partition, and
/// verifies; an order that is not a permutation of the groups is refused. So
/// does each group's openings run beside the next group's argue (phase B's
/// overlap), in group order, reversed, and reversed with the columns spilled.
#[test]
fn phase_b_in_any_order_proves_the_same_bytes() {
    use crate::block_whir::BlockSpillPolicy;
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    use stark::multilinear_block::PhaseBOverlap::{Off, On, Pinned};
    let proved = |order: Option<fn(usize) -> Vec<usize>>,
                  spill: BlockSpillPolicy,
                  overlap: stark::multilinear_block::PhaseBOverlap| {
        let mut o = streamed(MaxRowsConfig::small(), 5, 3);
        o.spill = spill;
        o.phase_b_overlap = overlap;
        prove_block_whir_with(
            &elf,
            &[],
            &ProofOptions::default_test_options(),
            &format,
            &o,
            &Deviations {
                phase_b_order: order,
                ..Deviations::default()
            },
        )
        .map(|(proof, _)| proof)
    };
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    let in_order = proved(None, BlockSpillPolicy::Off, Off).expect("prove");
    assert!(
        in_order.groups.len() > 3,
        "{} groups",
        in_order.groups.len()
    );
    let reversed: fn(usize) -> Vec<usize> = |n| (0..n).rev().collect();
    let odd_first: fn(usize) -> Vec<usize> =
        |n| (1..n).step_by(2).chain((0..n).step_by(2)).collect();
    for (what, order, spill, overlap) in [
        ("reversed", Some(reversed), BlockSpillPolicy::Off, Off),
        ("odd first", Some(odd_first), BlockSpillPolicy::Off, Off),
        (
            "reversed, spilled",
            Some(reversed),
            BlockSpillPolicy::Always,
            Off,
        ),
        ("in order, overlapped", None, BlockSpillPolicy::Off, On),
        (
            "reversed, overlapped",
            Some(reversed),
            BlockSpillPolicy::Off,
            On,
        ),
        (
            "reversed, spilled, overlapped",
            Some(reversed),
            BlockSpillPolicy::Always,
            On,
        ),
        ("in order, pinned", None, BlockSpillPolicy::Off, Pinned),
        (
            "reversed, spilled, pinned",
            Some(reversed),
            BlockSpillPolicy::Always,
            Pinned,
        ),
    ] {
        let proof = proved(order, spill, overlap).expect("prove");
        assert_eq!(proof.groups, in_order.groups, "{what}: the partition");
        assert!(verify(&proof, &elf, &format), "{what}");
        if crypto::grinding::deterministic() {
            assert_eq!(bytes(&proof), bytes(&in_order), "{what}: the proof");
        }
    }
    let twice: fn(usize) -> Vec<usize> = |n| std::iter::repeat_n(0, n).collect();
    let short: fn(usize) -> Vec<usize> = |n| (1..n).collect();
    for (what, order) in [("a group twice", twice), ("a group left out", short)] {
        for overlap in [Off, On, Pinned] {
            assert!(
                proved(Some(order), BlockSpillPolicy::Off, overlap).is_err(),
                "{what}: proved"
            );
        }
    }
}

/// Phase B's overlap on a streamed block: an opening or an argue that fails
/// beside the other, or a panic on the opening thread, is the prove's error,
/// never a panic; a refusal keeps its transition's overlap and turns the rest
/// serial, with the proof's bytes unchanged, and the readout counts each.
#[test]
fn phase_b_overlap_failures_are_the_proves_error_and_a_refusal_turns_it_serial() {
    use stark::multilinear_block::OverlapFaults;
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let mut o = streamed(MaxRowsConfig::small(), 5, 3);
    o.phase_b_overlap = stark::multilinear_block::PhaseBOverlap::On;
    let run = |faults: OverlapFaults| {
        prove_block_whir_with(
            &elf,
            &[],
            &ProofOptions::default_test_options(),
            &format,
            &o,
            &Deviations {
                overlap_faults: faults,
                ..Deviations::default()
            },
        )
    };
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    // First the overlap with no fault: every group but the last opened beside
    // an argue, so the faults below meet the opening thread where they aim.
    let (plain, stamps) = run(OverlapFaults::default()).expect("prove");
    let n = stamps.groups.len();
    assert!(n > 3, "{n} groups");
    assert_eq!(
        stamps.groups.iter().filter(|g| g.overlapped).count(),
        n - 1,
        "every group but the last opened beside an argue\n{}",
        stamps.report()
    );
    // The panic at place 0: the opening thread's first openings, in the first
    // transition, before any refusal could turn the rest serial.
    for (what, faults, stage) in [
        (
            "an opening fails",
            OverlapFaults {
                open_fails: Some(1),
                ..OverlapFaults::default()
            },
            "a test's failed opening",
        ),
        (
            "an argue fails",
            OverlapFaults {
                argue_fails: Some(2),
                ..OverlapFaults::default()
            },
            "a test's failed argue",
        ),
        (
            "the opening thread panics",
            OverlapFaults {
                open_panics: Some(0),
                ..OverlapFaults::default()
            },
            "the previous group's openings",
        ),
    ] {
        match run(faults) {
            Err(e) => assert!(format!("{e:?}").contains(stage), "{what}: {e:?}"),
            Ok(_) => panic!("{what}: proved"),
        }
    }
    let (latched, stamps) = run(OverlapFaults {
        refusal_at: Some(2),
        ..OverlapFaults::default()
    })
    .expect("prove");
    assert!(verify(&latched, &elf, &format));
    if crypto::grinding::deterministic() {
        assert_eq!(bytes(&latched), bytes(&plain), "a refusal: the proof");
    }
    let report = stamps.report();
    let counted = format!(
        "BLOCK PHASE-B OVERLAP: on · transitions overlapped 2 of {} (serial after a refusal {})",
        n - 1,
        n - 3
    );
    assert!(report.contains(&counted), "{counted}\n{report}");
}

/// `LAMBDA_VM_PHASE_B_OVERLAP`: unset `pinned` on a card build and off on a
/// host-only one, `1` on, `pinned` on with the openings' read-backs through
/// pinned staging; `0` or any other word off.
#[test]
fn the_phase_b_overlap_knob_reads_one_as_on() {
    use crate::block_whir::{PHASE_B_OVERLAP_DEFAULT, phase_b_overlap_from};
    use stark::multilinear_block::PhaseBOverlap::{Off, On, Pinned};
    let default = if cfg!(feature = "cuda") { Pinned } else { Off };
    assert_eq!(PHASE_B_OVERLAP_DEFAULT, default);
    assert_eq!(phase_b_overlap_from(None), default);
    assert_eq!(phase_b_overlap_from(Some("0")), Off);
    assert_eq!(phase_b_overlap_from(Some("1")), On);
    assert_eq!(phase_b_overlap_from(Some(" 1 ")), On);
    assert_eq!(phase_b_overlap_from(Some("pinned")), Pinned);
    assert_eq!(phase_b_overlap_from(Some(" pinned ")), Pinned);
    assert_eq!(phase_b_overlap_from(Some("on")), Off);
    assert_eq!(phase_b_overlap_from(Some("Pinned")), Off);
    assert_eq!(phase_b_overlap_from(Some("blocking")), Off);
    assert_eq!(phase_b_overlap_from(Some("2")), Off);
}

/// `LAMBDA_VM_BLOCK_SCHEDSTAT`: `1` on, anything else off.
#[test]
fn the_schedstat_knob_reads_one_as_on() {
    use crate::block_whir::phase_b_sched_from;
    assert!(!phase_b_sched_from(None));
    assert!(!phase_b_sched_from(Some("0")));
    assert!(phase_b_sched_from(Some("1")));
    assert!(phase_b_sched_from(Some(" 1 ")));
    assert!(!phase_b_sched_from(Some("on")));
}

/// The driver thread's schedstat readout, from stamps alone: each turn's CPU
/// and run-queue delay in the tenth of phase B it began in, the first three
/// tenths summed; unreadable or all zeros is said as unavailable, never as 0;
/// with the knob off, no line.
#[test]
fn the_schedstat_readout_counts_turns_by_tenth_and_says_unavailable() {
    use stark::multilinear_block::GroupStamps;
    let turn = |start_b: f64, sched_b: Option<(u64, u64)>| GroupStamps {
        start_b,
        sched_b,
        ..GroupStamps::default()
    };
    let ms = 1_000_000u64;
    let stamps = |groups: Vec<GroupStamps>, on: bool| {
        block_whir::BlockStamps {
            groups,
            phase_b: 10.0,
            phase_b_sched: on,
            ..Default::default()
        }
        .report()
    };
    let read = stamps(
        vec![
            turn(0.5, Some((400 * ms, 300 * ms))),
            turn(2.5, Some((500 * ms, 100 * ms))),
            turn(9.9, Some((600 * ms, 20 * ms))),
        ],
        true,
    );
    assert!(
        read.contains(
            "BLOCK PHASE-B SCHED: the driver thread over 3 group turns · CPU 1.500 s · run-queue \
             delay 0.420 s · first three tenths of phase B: CPU 0.900 s, run-queue delay 0.400 s · \
             by tenth (a turn in the tenth it began) CPU [0.40, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, \
             0.00, 0.00, 0.60] · run-queue delay [0.30, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, \
             0.00, 0.02]\n"
        ),
        "{read}"
    );
    let none = stamps(vec![turn(0.5, None), turn(2.5, None)], true);
    assert!(
        none.contains(
            "BLOCK PHASE-B SCHED: unavailable (/proc/thread-self/schedstat did not read)\n"
        ),
        "{none}"
    );
    let zeros = stamps(vec![turn(0.5, Some((0, 0)))], true);
    assert!(
        zeros.contains("BLOCK PHASE-B SCHED: unavailable (schedstat reads zeros"),
        "{zeros}"
    );
    let off = stamps(vec![turn(0.5, Some((400 * ms, 300 * ms)))], false);
    assert!(!off.contains("BLOCK PHASE-B SCHED"), "{off}");
}

/// The overlap's readout, on and off, from stamps alone: the per-group `paid`,
/// the opening helper, the measured transitions on the phase-B ledger and the
/// overlap's own line; off, every line as it was and the overlap line `off`.
#[test]
fn the_phase_b_overlap_readout_names_its_transitions() {
    use stark::multilinear_block::GroupStamps;
    let gib = 1u64 << 30;
    let groups = vec![
        GroupStamps {
            argue: 0.5,
            encode: 0.01,
            open: 0.09,
            open_paid: 0.02,
            overlapped: true,
            open_draws: (40, 7),
            ..GroupStamps::default()
        },
        GroupStamps {
            argue: 0.4,
            encode: 0.02,
            open: 0.08,
            open_paid: 0.0,
            overlapped: true,
            transition_reserved: 20 * gib,
            open_draws: (30, 5),
            ..GroupStamps::default()
        },
        GroupStamps {
            argue: 0.3,
            encode: 0.01,
            open: 0.05,
            open_paid: 0.06,
            transition_reserved: 21 * gib,
            open_draws: (6, 0),
            ..GroupStamps::default()
        },
    ];
    let on = block_whir::BlockStamps {
        groups: groups.clone(),
        phase_b_overlap: true,
        ..Default::default()
    }
    .report();
    assert!(on.contains("encode 0.010 open 0.090 paid 0.020\n"), "{on}");
    assert!(
        on.contains("+ one upload helper + one opening helper\n"),
        "{on}"
    );
    assert!(
        on.contains("open(g) beside argue(g+1) [20480, 21504] MiB measured ·"),
        "{on}"
    );
    assert!(
        on.contains(
            "BLOCK PHASE-B OVERLAP: on · transitions overlapped 2 of 2 (serial after a refusal 0) · \
             openings beside an argue 0.200 s, hidden 0.180 s, paid 0.020 s + alone 0.060 s · \
             argue 1.200 s · transition ledger peak max 21504 MiB"
        ),
        "{on}"
    );
    assert!(
        on.contains(
            "openings' stream draws beside an argue background 12 / foreground 70 (the argue's \
             too), alone background 0 / foreground 6 · trims 0 · "
        ),
        "{on}"
    );
    assert!(
        on.contains(
            "· trims 0 · the openings' read-backs pinned 0 / pageable 0 (0.0 MiB pinned)\n"
        ),
        "{on}"
    );
    let pinned = block_whir::BlockStamps {
        groups: groups.clone(),
        phase_b_overlap: true,
        phase_b_pinned: true,
        background_readbacks: (5887, 0, 3 << 20),
        ..Default::default()
    }
    .report();
    assert!(
        pinned.contains("BLOCK PHASE-B OVERLAP: pinned · transitions overlapped 2 of 2"),
        "{pinned}"
    );
    assert!(
        pinned.contains(
            "· trims 0 · the openings' read-backs pinned 5887 / pageable 0 (3.0 MiB pinned)\n"
        ),
        "{pinned}"
    );
    let off = block_whir::BlockStamps {
        groups,
        ..Default::default()
    }
    .report();
    assert!(off.contains("encode 0.010 open 0.090\n"), "{off}");
    assert!(off.contains("+ one upload helper\n"), "{off}");
    assert!(!off.contains(" measured"), "{off}");
    assert!(
        off.contains("BLOCK PHASE-B OVERLAP: off · transitions overlapped 2 of 2"),
        "{off}"
    );
}

/// ★ On a card with a background pool (`LAMBDA_VM_STREAM_CLASSES=equal` or
/// `priority`), every stream the openings beside an argue draw is a
/// background one: as many as the same group's openings drew alone, all of
/// them foreground, with the overlap off. A device call they made from
/// another thread would draw a foreground stream and show here as fewer.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "box: needs a card and LAMBDA_VM_STREAM_CLASSES=equal"]
fn phase_b_overlap_openings_draw_background_streams() {
    assert!(
        math_cuda::stream_class::setting().split(),
        "run with LAMBDA_VM_STREAM_CLASSES=equal or priority"
    );
    let elf = asm_elf_bytes("test_keccak_multi");
    let mut format = many_groups();
    // Codewords of 2^16 and more: the card's, not the host's.
    format.zf.whir_stack = StackVars::new(14).expect("a valid stack");
    let run = |overlap: bool| {
        let mut o = streamed(MaxRowsConfig::small(), 5, 3);
        o.phase_b_overlap = if overlap {
            stark::multilinear_block::PhaseBOverlap::On
        } else {
            stark::multilinear_block::PhaseBOverlap::Off
        };
        prove_block_whir_with(
            &elf,
            &[],
            &ProofOptions::default_test_options(),
            &format,
            &o,
            &Deviations::default(),
        )
        .expect("prove")
        .1
    };
    let (alone, beside) = (run(false), run(true));
    assert!(alone.groups.len() > 2, "{} groups", alone.groups.len());
    let drawn: u64 = alone.groups.iter().map(|g| g.open_draws.0).sum();
    println!(
        "PHASE-B DRAWS: alone {:?} · beside {:?}",
        alone
            .groups
            .iter()
            .map(|g| g.open_draws)
            .collect::<Vec<_>>(),
        beside
            .groups
            .iter()
            .map(|g| g.open_draws)
            .collect::<Vec<_>>()
    );
    assert!(drawn > 0, "no opening drew a stream: nothing to compare");
    let mut compared = 0;
    for (g, (a, b)) in alone.groups.iter().zip(&beside.groups).enumerate() {
        assert_eq!(a.open_draws.1, 0, "group {g}: a background draw alone");
        if b.overlapped {
            assert_eq!(
                b.open_draws.1, a.open_draws.0,
                "group {g}: beside an argue, the openings' draws"
            );
            compared += usize::from(a.open_draws.0 > 0);
        }
    }
    assert!(compared > 0, "no overlapped group drew a stream");
}

/// ★ On a card, `pinned` sends every read-back of the openings' threads
/// through pinned staging, none the stream's own way, and the proof is the
/// overlap's own bytes and verifies. With the overlap on but not pinned, the
/// openings read back the stream's own way and none takes the pinned path.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "box: needs a card"]
fn pinned_openings_read_back_through_pinned_staging() {
    use stark::multilinear_block::PhaseBOverlap::{On, Pinned};
    let elf = asm_elf_bytes("test_keccak_multi");
    let mut format = many_groups();
    // Codewords of 2^16 and more: the card's, not the host's.
    format.zf.whir_stack = StackVars::new(14).expect("a valid stack");
    let run = |overlap| {
        let mut o = streamed(MaxRowsConfig::small(), 5, 3);
        o.phase_b_overlap = overlap;
        prove_block_whir_with(
            &elf,
            &[],
            &ProofOptions::default_test_options(),
            &format,
            &o,
            &Deviations::default(),
        )
        .expect("prove")
    };
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    let (on_proof, on) = run(On);
    let (pinned_proof, pinned) = run(Pinned);
    assert!(
        pinned.groups.iter().any(|g| g.overlapped),
        "no transition overlapped"
    );
    let (on_pinned, on_pageable, _) = on.background_readbacks;
    assert!(
        on_pinned == 0 && on_pageable > 0,
        "on: the openings' read-backs {:?}",
        on.background_readbacks
    );
    let (pinned_pinned, pinned_pageable, _) = pinned.background_readbacks;
    assert!(
        pinned_pinned > 0 && pinned_pageable == 0,
        "pinned: the openings' read-backs {:?}",
        pinned.background_readbacks
    );
    let report = pinned.report();
    assert!(
        report.contains("BLOCK PHASE-B OVERLAP: pinned ·"),
        "{report}"
    );
    assert!(verify(&pinned_proof, &elf, &format), "pinned");
    if crypto::grinding::deterministic() {
        assert_eq!(bytes(&pinned_proof), bytes(&on_proof), "the proof");
    }
}

/// A streamed prove under live regeneration `mode` and the spill `policy`,
/// with `deviations`: its proof or why it was refused, and its stamps.
fn regenerated(
    elf: &[u8],
    format: &BlockFormat,
    mode: block_whir::regen::RegenMode,
    policy: crate::block_whir::BlockSpillPolicy,
    deviations: Deviations,
) -> Result<(BlockWhirProof, block_whir::BlockStamps), crate::Error> {
    regenerated_at(
        elf,
        format,
        mode,
        policy,
        deviations,
        MaxRowsConfig::small(),
    )
}

/// [`regenerated`] with chunks of `max_rows`.
fn regenerated_at(
    elf: &[u8],
    format: &BlockFormat,
    mode: block_whir::regen::RegenMode,
    policy: crate::block_whir::BlockSpillPolicy,
    deviations: Deviations,
    max_rows: MaxRowsConfig,
) -> Result<(BlockWhirProof, block_whir::BlockStamps), crate::Error> {
    let mut o = streamed(max_rows, 5, 3);
    o.layout_workers = 3;
    o.drop_streamed_ops = true;
    o.spill = policy;
    o.regen = Some(mode);
    // KECCAK_RND built on the host unless a test asks otherwise: its tables
    // are then droppable, and these tests rebuild them with LT's. CPU and
    // MEMW_R likewise (a card build's default fills them from their records):
    // these tests drop and rebuild them as packed chunks, and the records
    // tests ask for theirs.
    let deviations = Deviations {
        card_fill: deviations.card_fill.or(Some(false)),
        card_fill_streamed: deviations.card_fill_streamed.or(Some([false, false])),
        ..deviations
    };
    prove_block_whir_with(
        elf,
        &[],
        &ProofOptions::default_test_options(),
        format,
        &o,
        &deviations,
    )
}

/// How long a hang guard waits for a regenerating prove: ten times the plain
/// prove the test timed before it, and at least ten minutes. A hang never
/// returns, so any bound finds one; a bound scaled to the plain prove also
/// holds where every prove is slower (a host without a card, a costlier base
/// hash), where a fixed one fails a prove that completes: one-byte windows
/// ran past 600 s on the laptop under the RPX base, phase B never waiting.
fn hang_bound(plain: std::time::Duration) -> std::time::Duration {
    std::time::Duration::from_secs(600).max(plain.saturating_mul(10))
}

/// The hang guard's bound: ten minutes under a fast plain prove, ten plain
/// proves past one minute.
#[test]
fn a_hang_bound_is_ten_plain_proves_and_at_least_ten_minutes() {
    use std::time::Duration;
    assert_eq!(hang_bound(Duration::ZERO), Duration::from_secs(600));
    assert_eq!(
        hang_bound(Duration::from_secs(60)),
        Duration::from_secs(600)
    );
    assert_eq!(
        hang_bound(Duration::from_secs(61)),
        Duration::from_secs(610)
    );
    assert_eq!(
        hang_bound(Duration::from_secs(550)),
        Duration::from_secs(5500)
    );
    assert_eq!(hang_bound(Duration::MAX), Duration::MAX);
}

/// ★ Live regeneration moves no byte (D-WHIR-NODISK N2): with `always` every
/// streamed chunk is dropped after its group's commit and rebuilt in phase B —
/// with no store (no disk), and with the rest spilled — and `auto` armed from
/// the start (a target of 0, no store: no disk) drops them all too; the proof
/// is the one regeneration off proves, with the same partition, under the
/// deterministic grind, and verifies. `auto` that never arms (a budget the
/// block fits in, and a target out of reach) drops nothing and runs no
/// regenerator.
#[test]
fn live_regeneration_proves_the_same_bytes() {
    use crate::block_whir::BlockSpillPolicy;
    use crate::block_whir::regen::RegenMode;
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    let (plain, plain_stamps) = regenerated(
        &elf,
        &format,
        RegenMode::Off,
        BlockSpillPolicy::Off,
        Deviations::default(),
    )
    .expect("prove");
    assert!(plain_stamps.regen.is_none());
    let streamed = plain_stamps.streamed.1;
    assert!(streamed > 0);
    for (what, mode, policy, target) in [
        (
            "always, no disk",
            RegenMode::Always,
            BlockSpillPolicy::Off,
            None,
        ),
        (
            "always, the rest spilled",
            RegenMode::Always,
            BlockSpillPolicy::Always,
            None,
        ),
        (
            "auto armed, no disk",
            RegenMode::Auto,
            BlockSpillPolicy::Off,
            Some(0),
        ),
    ] {
        let (proof, stamps) = regenerated(
            &elf,
            &format,
            mode,
            policy,
            Deviations {
                spill_target: target,
                ..Deviations::default()
            },
        )
        .unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let r = stamps.regen.expect("live regeneration's readout");
        let lines = r.lines.join("\n");
        // Every streamed chunk, and KECCAK_RND's and LT's tables of the rest
        // (N3, one class a group), dropped and back.
        assert!(r.rest_dropped > 0, "{what}: no rest table dropped\n{lines}");
        assert_eq!(
            r.dropped,
            streamed + r.rest_dropped,
            "{what}: every chunk dropped\n{lines}"
        );
        assert_eq!(
            r.regenerated, r.dropped,
            "{what}: every table back\n{lines}"
        );
        assert_eq!(
            r.rest_regenerated, r.rest_dropped,
            "{what}: the rest back\n{lines}"
        );
        assert_eq!((r.mismatches, r.failed), (0, 0), "{what}\n{lines}");
        assert_eq!(r.error, None, "{what}\n{lines}");
        assert!(lines.contains("groups rebuilt ("), "{lines}");
        assert!(lines.contains("BLOCK REGEN rest: "), "{lines}");
        assert!(
            lines.contains("BLOCK REGEN phase B order taken: "),
            "{lines}"
        );
        // The interleaved order is not group order: the bytes below hold only
        // because each group proves on its own fork (`S_post ‖ g`).
        assert_ne!(
            r.phase_b_order,
            (0..proof.groups.len()).collect::<Vec<_>>(),
            "{what}: group order"
        );
        if policy == BlockSpillPolicy::Off {
            assert!(stamps.spill_stats.is_none(), "{what}: a store opened");
            assert!(
                stamps
                    .spill
                    .as_deref()
                    .is_some_and(|l| l.contains("no store")),
                "{what}: {:?}",
                stamps.spill
            );
        } else {
            let stats = stamps.spill_stats.expect("the store's counters");
            assert!(stats.slots > 0, "{what}: the rest spilled nothing: {stats}");
        }
        assert_eq!(proof.groups, plain.groups, "{what}: the partition");
        assert!(verify(&proof, &elf, &format), "{what}");
        if crypto::grinding::deterministic() {
            assert_eq!(bytes(&proof), bytes(&plain), "{what}: the proof");
        }
    }
    // The budget decides each table; the walk's end still hands off (and
    // arms) once the peak passes the pressure share of the host's target, so
    // the target is out of reach and no host's memory decides it.
    let (proof, stamps) = regenerated(
        &elf,
        &format,
        RegenMode::Auto,
        BlockSpillPolicy::Budget(1 << 40),
        Deviations {
            spill_target: Some(u64::MAX),
            ..Deviations::default()
        },
    )
    .expect("prove");
    let r = stamps.regen.expect("live regeneration's readout");
    let lines = r.lines.join("\n");
    assert_eq!(r.dropped, 0, "{lines}");
    assert!(
        lines.contains("never armed") && !lines.contains("BLOCK REGEN live:"),
        "{lines}"
    );
    assert!(lines.contains("0 of"), "{lines}");
    if crypto::grinding::deterministic() {
        assert_eq!(bytes(&proof), bytes(&plain), "auto unarmed: the proof");
    }
}

/// The production default's spill, no disk (N4b), is `off` beside live
/// regeneration (no store, the dropped line says so) and `auto` without it (a
/// target of 0 spills every table it can): either way the same bytes.
#[test]
fn no_disk_is_off_beside_live_regeneration_and_auto_without_it() {
    use crate::block_whir::BlockSpillPolicy;
    use crate::block_whir::regen::RegenMode;
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    let (plain, _) = regenerated(
        &elf,
        &format,
        RegenMode::Off,
        BlockSpillPolicy::Off,
        Deviations::default(),
    )
    .expect("prove");
    let (live, stamps) = regenerated(
        &elf,
        &format,
        RegenMode::Always,
        BlockSpillPolicy::NoDisk,
        Deviations::default(),
    )
    .expect("prove");
    let r = stamps.regen.expect("live regeneration's readout");
    let lines = r.lines.join("\n");
    assert!(r.dropped > 0 && r.regenerated == r.dropped, "{lines}");
    assert!(lines.contains("(no disk)"), "{lines}");
    assert!(stamps.spill_stats.is_none(), "a store opened");
    assert!(
        stamps
            .spill
            .as_deref()
            .is_some_and(|l| l.starts_with("off") && l.contains("no store")),
        "{:?}",
        stamps.spill
    );
    let (spilled, stamps) = regenerated(
        &elf,
        &format,
        RegenMode::Off,
        BlockSpillPolicy::NoDisk,
        Deviations {
            spill_target: Some(0),
            ..Deviations::default()
        },
    )
    .expect("prove");
    assert!(stamps.regen.is_none());
    let stats = stamps.spill_stats.expect("auto's store");
    assert!(stats.slots > 0, "auto spilled nothing: {stats}");
    assert!(
        stamps
            .spill
            .as_deref()
            .is_some_and(|l| l.starts_with("auto")),
        "{:?}",
        stamps.spill
    );
    for (what, proof) in [("live", &live), ("spilled", &spilled)] {
        assert_eq!(proof.groups, plain.groups, "{what}: the partition");
        assert!(verify(proof, &elf, &format), "{what}");
        if crypto::grinding::deterministic() {
            assert_eq!(bytes(proof), bytes(&plain), "{what}: the proof");
        }
    }
}

/// ★ The streamed chunks' regenerator starts before phase B, from each
/// chunk's fate as phase A decided it: its thread up where the streamed chunks
/// end and started once their fates are known (the default), or started at
/// the finish's end; the rest's starts with phase B. Every dropped table is
/// rebuilt and deposited, and the proof is regeneration off's bytes; started
/// with phase B instead (the knob off), the same. (A fate decided after the
/// regenerator asks is `stark::multilinear_block`'s
/// `a_regenerator_started_early_waits_for_each_tables_fate`.)
#[test]
fn regenerators_started_at_the_finishs_end_prove_the_same_bytes() {
    use crate::block_whir::BlockSpillPolicy;
    use crate::block_whir::regen::{EarlySpawn, RegenMode};
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    let (plain, _) = regenerated(
        &elf,
        &format,
        RegenMode::Off,
        BlockSpillPolicy::Off,
        Deviations::default(),
    )
    .expect("prove");
    for early in [EarlySpawn::Fates, EarlySpawn::Finish, EarlySpawn::Off] {
        let (proof, stamps) = regenerated(
            &elf,
            &format,
            RegenMode::Always,
            BlockSpillPolicy::Off,
            Deviations {
                regen_early: Some(early),
                ..Deviations::default()
            },
        )
        .unwrap_or_else(|e| panic!("early {early:?}: {e:?}"));
        let r = stamps.regen.expect("live regeneration's readout");
        let lines = r.lines.join("\n");
        assert_eq!(r.head_start.is_some(), early != EarlySpawn::Off, "{lines}");
        assert!(r.rest_dropped > 0 && r.dropped > r.rest_dropped, "{lines}");
        assert_eq!(r.regenerated, r.dropped, "{lines}");
        assert_eq!((r.mismatches, r.failed), (0, 0), "{lines}");
        assert_eq!(proof.groups, plain.groups, "early {early:?}: the partition");
        assert!(verify(&proof, &elf, &format), "early {early:?}");
        if crypto::grinding::deterministic() {
            assert!(
                bytes(&proof) == bytes(&plain),
                "early {early:?}: the proof differs"
            );
        }
    }
}

/// ★ The regenerator's thread is up before the drops: with each streamed
/// chunk's drop held back, it asks for fates phase A has not decided yet, waits
/// for every one, and rebuilds exactly what was dropped; the proof is
/// regeneration off's bytes. A fate answered "kept" before it is decided would
/// leave ranks out, and phase A's end refuses that ("the early regenerator
/// took … streamed chunks, phase A dropped …").
#[test]
fn a_regenerator_up_before_the_drops_proves_the_same_bytes() {
    use crate::block_whir::BlockSpillPolicy;
    use crate::block_whir::regen::{EarlySpawn, RegenMode};
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    let (plain, _) = regenerated(
        &elf,
        &format,
        RegenMode::Off,
        BlockSpillPolicy::Off,
        Deviations::default(),
    )
    .expect("prove");
    let (proof, stamps) = regenerated(
        &elf,
        &format,
        RegenMode::Always,
        BlockSpillPolicy::Off,
        Deviations {
            regen_early: Some(EarlySpawn::Fates),
            streamed_drop_delay: Some(std::time::Duration::from_millis(100)),
            ..Deviations::default()
        },
    )
    .unwrap_or_else(|e| panic!("the drops held back: {e:?}"));
    let r = stamps.regen.expect("live regeneration's readout");
    let lines = r.lines.join("\n");
    assert!(r.head_start.is_some(), "{lines}");
    assert!(r.dropped > r.rest_dropped, "{lines}");
    assert_eq!(r.regenerated, r.dropped, "{lines}");
    assert_eq!((r.mismatches, r.failed), (0, 0), "{lines}");
    assert_eq!(proof.groups, plain.groups, "the partition");
    assert!(verify(&proof, &elf, &format), "the drops held back");
    if crypto::grinding::deterministic() {
        assert!(bytes(&proof) == bytes(&plain), "the proof differs");
    }
}

/// N4c's liveness: phase A ends in an error right after the finish, with the
/// streamed chunks' regenerator started and waiting on chunks not yet
/// decided. The prove returns the error (its window closed, its thread
/// joined) and never hangs.
#[test]
fn an_early_regenerator_never_hangs_a_failed_phase_a() {
    use crate::block_whir::BlockSpillPolicy;
    use crate::block_whir::regen::{EarlySpawn, RegenMode};
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let out = regenerated(
            &elf,
            &format,
            RegenMode::Always,
            BlockSpillPolicy::Off,
            Deviations {
                regen_early: Some(EarlySpawn::Fates),
                streamed_drop_delay: Some(std::time::Duration::from_millis(400)),
                abort_after_finish: true,
                ..Deviations::default()
            },
        );
        let _ = tx.send(out.map(|_| ()).map_err(|e| format!("{e:?}")));
    });
    let out = rx
        .recv_timeout(std::time::Duration::from_secs(600))
        .expect("the failed prove returned");
    let why = out.expect_err("phase A failed");
    assert!(
        why.contains("a test's abort right after the finish"),
        "{why}"
    );
}

/// ★ Live regeneration never proves over columns it did not rebuild, and never
/// hangs: a deposit one bit off is refused at its slot before any device work
/// ("not the ones dropped"); with the digest off, the kept tree top refuses it
/// in the opening; a walk that fails, a generator that dies outside its catch,
/// or a chunk the slicer never cuts leaves its table refused ("did not bring
/// its columns back"), the prove returning; and phase B told to take the
/// rebuilt groups out of rank order is refused before it starts.
#[test]
fn live_regeneration_refuses_what_it_did_not_rebuild() {
    use crate::block_whir::BlockSpillPolicy;
    use crate::block_whir::regen::{LiveFaults, RegenMode};
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let reversed: fn(usize) -> Vec<usize> = |n| (0..n).rev().collect();
    type Order = Option<fn(usize) -> Vec<usize>>;
    let cases: [(&str, LiveFaults, Order, &str); 6] = [
        (
            "a bent deposit",
            LiveFaults {
                bend_rank: Some(1),
                ..LiveFaults::default()
            },
            None,
            "not the ones dropped",
        ),
        (
            "a bent deposit, the digest off",
            LiveFaults {
                bend_rank: Some(1),
                digest_off: true,
                ..LiveFaults::default()
            },
            None,
            "RecomputedCodewordMismatch",
        ),
        (
            "a walk that fails",
            LiveFaults {
                walk_error_at: Some(0),
                ..LiveFaults::default()
            },
            None,
            "did not bring its columns back",
        ),
        (
            "a generator that dies",
            LiveFaults {
                generator_panics_at_rank: Some(0),
                ..LiveFaults::default()
            },
            None,
            "did not bring its columns back",
        ),
        (
            "a chunk never cut",
            LiveFaults {
                skip_rank: Some(0),
                ..LiveFaults::default()
            },
            None,
            "did not bring its columns back",
        ),
        (
            "rebuilt groups out of order",
            LiveFaults::default(),
            Some(reversed),
            "rebuilt groups in group order",
        ),
    ];
    for (what, faults, order, why) in cases {
        let (tx, rx) = std::sync::mpsc::channel();
        let elf = elf.clone();
        std::thread::spawn(move || {
            let out = regenerated(
                &elf,
                &format,
                RegenMode::Always,
                BlockSpillPolicy::Off,
                Deviations {
                    regen_faults: faults,
                    phase_b_order: order,
                    ..Deviations::default()
                },
            );
            let _ = tx.send(out.map(|_| ()).map_err(|e| format!("{e:?}")));
        });
        let got = rx
            .recv_timeout(std::time::Duration::from_secs(600))
            .unwrap_or_else(|_| panic!("{what}: the prove hung"));
        match got {
            Err(e) => assert!(e.contains(why), "{what}: refused for another reason: {e}"),
            Ok(()) => panic!("{what}: proved"),
        }
    }
}

/// ★ Windows of one byte (N3): each admits only its frontier, so a phase-B
/// order or a group that takes a class out of its rank order cannot complete.
/// With every streamed chunk and every KECCAK_RND and LT table dropped, the
/// interleaved order proves the same bytes, within a bound on the time
/// ([`hang_bound`]).
#[test]
fn one_byte_windows_take_each_class_in_its_order() {
    use crate::block_whir::BlockSpillPolicy;
    use crate::block_whir::regen::RegenMode;
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let started = std::time::Instant::now();
    let (plain, _) = regenerated(
        &elf,
        &format,
        RegenMode::Off,
        BlockSpillPolicy::Off,
        Deviations::default(),
    )
    .expect("prove");
    let bound = hang_bound(started.elapsed());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let out = regenerated(
            &elf,
            &format,
            RegenMode::Always,
            BlockSpillPolicy::Off,
            Deviations {
                regen_windows: Some((1, 1)),
                ..Deviations::default()
            },
        );
        let _ = tx.send(out.map_err(|e| format!("{e:?}")));
    });
    let (proof, stamps) = rx
        .recv_timeout(bound)
        .expect("the prove did not hang")
        .expect("prove");
    let r = stamps.regen.expect("live regeneration's readout");
    let lines = r.lines.join("\n");
    assert!(r.rest_dropped > 0 && r.dropped > r.rest_dropped, "{lines}");
    assert_eq!(r.regenerated, r.dropped, "{lines}");
    assert_ne!(
        r.phase_b_order,
        (0..proof.groups.len()).collect::<Vec<_>>(),
        "the interleaved order is group order"
    );
    assert_eq!(proof.groups, plain.groups);
    if crypto::grinding::deterministic() {
        let bytes = |p: &BlockWhirProof| {
            rkyv::to_bytes::<rkyv::rancor::Error>(p)
                .expect("serialize")
                .to_vec()
        };
        assert!(bytes(&proof) == bytes(&plain), "the proof differs");
    }
}

/// ★ One class a group (N3): with groups of 2^24-cell polynomials the block
/// is one group, holding every streamed chunk and every KECCAK_RND and LT table
/// of the rest. Under `always` it drops the streamed class only, the class of
/// its first droppable table, and holds the rest's tables in place (phase B
/// could not take one group in both classes' orders); the proof is the bytes
/// regeneration off proves, and verifies. A group that dropped both classes is
/// refused before phase B, never proved and never left to hang.
#[test]
fn a_mixed_boundary_group_drops_one_class_only() {
    use crate::block_whir::BlockSpillPolicy;
    use crate::block_whir::regen::RegenMode;
    let elf = asm_elf_bytes("test_keccak_multi");
    let mut format = many_groups();
    format.zf.whir_stack = StackVars::new(24).expect("a valid stack");
    let started = std::time::Instant::now();
    let (plain, plain_stamps) = regenerated(
        &elf,
        &format,
        RegenMode::Off,
        BlockSpillPolicy::Off,
        Deviations::default(),
    )
    .expect("prove");
    let bound = hang_bound(started.elapsed());
    let streamed = plain_stamps.streamed.1;
    let (tx, rx) = std::sync::mpsc::channel();
    {
        let elf = elf.clone();
        std::thread::spawn(move || {
            let out = regenerated(
                &elf,
                &format,
                RegenMode::Always,
                BlockSpillPolicy::Off,
                Deviations::default(),
            );
            let _ = tx.send(out.map_err(|e| format!("{e:?}")));
        });
    }
    let (proof, stamps) = rx
        .recv_timeout(bound)
        .expect("the prove did not hang")
        .expect("a mixed group is proved, one class dropped");
    let r = stamps.regen.expect("live regeneration's readout");
    let lines = r.lines.join("\n");
    assert_eq!(proof.groups.len(), 1, "one group: {:?}", proof.groups);
    assert_eq!(r.dropped, streamed, "the streamed class dropped\n{lines}");
    assert_eq!(r.rest_dropped, 0, "no rest table dropped\n{lines}");
    assert!(r.held > 0, "the rest's tables held\n{lines}");
    assert_eq!(r.regenerated, r.dropped, "{lines}");
    assert!(lines.contains(&format!("{} held", r.held)), "{lines}");
    assert_eq!(proof.groups, plain.groups);
    assert!(verify(&proof, &elf, &format));
    if crypto::grinding::deterministic() {
        let bytes = |p: &BlockWhirProof| {
            rkyv::to_bytes::<rkyv::rancor::Error>(p)
                .expect("serialize")
                .to_vec()
        };
        assert!(bytes(&proof) == bytes(&plain), "the proof differs");
    }
}

/// ★ The rest's regenerator refuses what it did not build again (N3): a
/// deposit of other columns, a generator that dies, a table never built — each
/// at the rest's first dropped table — refuse the prove before any of that
/// table's device work, never prove over it, and never hang.
#[test]
fn the_rest_regenerator_refuses_what_it_did_not_rebuild() {
    use crate::block_whir::BlockSpillPolicy;
    use crate::block_whir::regen::{LiveFaults, RegenMode};
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let cases: [(&str, LiveFaults, &str); 3] = [
        (
            "a bent rest deposit",
            LiveFaults {
                bend_rank: Some(0),
                ..LiveFaults::default()
            },
            "not the ones dropped",
        ),
        (
            "a rest generator that dies",
            LiveFaults {
                generator_panics_at_rank: Some(0),
                ..LiveFaults::default()
            },
            "did not bring its columns back",
        ),
        (
            "a rest table never built",
            LiveFaults {
                skip_rank: Some(0),
                ..LiveFaults::default()
            },
            "did not bring its columns back",
        ),
    ];
    for (what, faults, why) in cases {
        let (tx, rx) = std::sync::mpsc::channel();
        let elf = elf.clone();
        std::thread::spawn(move || {
            let out = regenerated(
                &elf,
                &format,
                RegenMode::Always,
                BlockSpillPolicy::Off,
                Deviations {
                    rest_faults: faults,
                    ..Deviations::default()
                },
            );
            let _ = tx.send(out.map(|_| ()).map_err(|e| format!("{e:?}")));
        });
        let got = rx
            .recv_timeout(std::time::Duration::from_secs(600))
            .unwrap_or_else(|_| panic!("{what}: the prove hung"));
        match got {
            Err(e) => assert!(e.contains(why), "{what}: refused for another reason: {e}"),
            Ok(()) => panic!("{what}: proved"),
        }
    }
}

/// The streamed chunks laid out on three threads, and the rest of the run
/// packed as it is laid out, are packed in the inline order all the same: the
/// same groups and table counts as laid out on one thread, and the proofs
/// verify.
#[test]
fn a_streamed_block_laid_out_on_three_threads_packs_the_same_groups() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let inline = prove(&elf, &format, &streamed(MaxRowsConfig::small(), 16, 3));
    for pack_rest in [false, true] {
        let mut o = streamed(MaxRowsConfig::small(), 16, 3);
        o.layout_workers = 3;
        o.pack_rest_as_laid_out = pack_rest;
        let proof = prove(&elf, &format, &o);
        assert!(groups_of(&proof) >= 3, "{} groups", groups_of(&proof));
        assert_eq!(proof.groups, inline.groups, "pack_rest {pack_rest}");
        assert_eq!(
            format!("{:?}", proof.table_counts),
            format!("{:?}", inline.table_counts)
        );
        assert!(verify(&proof, &elf, &format));
    }
}

/// ★ With BITWISE handed on last ([`Deviations::rest_bitwise_last`], as
/// `LAMBDA_VM_BLOCK_REST_BITWISE_LAST=1`), the streamed rest is packed in AIR
/// order but BITWISE (AIR index 0), which comes after every other table: the
/// partition moves, the tables and their counts do not, and the proof verifies
/// under the unchanged verifier. Without it BITWISE leads the rest, as today.
#[test]
fn bitwise_handed_on_last_moves_the_partition_and_verifies() {
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let mut o = streamed(MaxRowsConfig::small(), 5, 3);
    o.layout_workers = 3;
    assert!(o.rest_streams(), "the rest streams");
    let proved = |bitwise_last: bool| {
        let proof = prove_with(
            &elf,
            &format,
            &o,
            &Deviations {
                rest_bitwise_last: bitwise_last,
                ..Deviations::default()
            },
        );
        assert!(verify(&proof, &elf, &format), "bitwise_last {bitwise_last}");
        proof
    };
    let air = proved(false);
    let last = proved(true);
    assert_eq!(
        format!("{:?}", last.table_counts),
        format!("{:?}", air.table_counts)
    );
    let flat = |p: &BlockWhirProof| p.groups.iter().flatten().copied().collect::<Vec<u32>>();
    let (air_flat, last_flat) = (flat(&air), flat(&last));
    let sorted = |mut v: Vec<u32>| {
        v.sort_unstable();
        v
    };
    assert_eq!(
        sorted(air_flat.clone()),
        sorted(last_flat.clone()),
        "the same tables"
    );
    assert_eq!(last_flat.last(), Some(&0), "BITWISE packed last");
    assert_ne!(
        air_flat.last(),
        Some(&0),
        "BITWISE packed last without the knob"
    );
    assert_ne!(last.groups, air.groups, "the partition did not move");
}

/// ★ BITWISE's lookups from LT's, BRANCH's and BYTEWISE's tables
/// ([`Deviations::bitwise_from_tables`], as
/// `LAMBDA_VM_BLOCK_BITWISE_FROM_TABLES=1`), counted from each table's rows as
/// it is built, commit what BITWISE handed on last commits with phase 4's count
/// from the ops: the same partition and the same group roots, and the proof
/// verifies. The finish keeps only the streamed chunks' tails, as the block
/// does.
#[test]
fn bitwise_from_tables_commits_bitwise_lasts_tables_and_verifies() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let mut o = streamed(MaxRowsConfig::small(), 5, 3);
    o.layout_workers = 3;
    o.drop_streamed_ops = true;
    assert!(o.rest_streams(), "the rest streams");
    let proved = |what: &str, deviations: Deviations| {
        let proof = prove_with(&elf, &format, &o, &deviations);
        assert!(verify(&proof, &elf, &format), "{what}");
        proof
    };
    let from_ops = proved(
        "from the ops",
        Deviations {
            rest_bitwise_last: true,
            ..Deviations::default()
        },
    );
    let from_tables = proved(
        "from the tables",
        Deviations {
            bitwise_from_tables: true,
            ..Deviations::default()
        },
    );
    assert_eq!(from_tables.groups, from_ops.groups, "the partition");
    assert_eq!(
        from_tables.proof.roots, from_ops.proof.roots,
        "the committed tables"
    );
}

/// The bounded layout (D-EXEC E3): with three workers and `layout_ahead =
/// Some(k)`, at most `k + 1` streamed chunks are laid out (or in the making)
/// and not yet packed at once, and the groups are the inline layout's.
#[test]
fn the_bounded_layout_holds_at_most_k_plus_one_chunks_unpacked() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let inline = prove(&elf, &format, &streamed(MaxRowsConfig::small(), 16, 3));
    for k in [0, 2] {
        let mut o = streamed(MaxRowsConfig::small(), 16, 3);
        o.layout_workers = 3;
        o.layout_ahead = Some(k);
        let (proof, stamps) = prove_block_whir_with(
            &elf,
            &[],
            &ProofOptions::default_test_options(),
            &format,
            &o,
            &Deviations::default(),
        )
        .expect("prove");
        let most = stamps.layout.ahead_most;
        assert!(
            (1..=k + 1).contains(&most),
            "k {k}: {most} chunks unpacked at once"
        );
        assert_eq!(proof.groups, inline.groups, "k {k}");
        assert!(verify(&proof, &elf, &format));
    }
}

/// Closed groups waiting for phase A behind one another move no byte of the
/// proof: a no-disk block proved with no bytes for them to wait in, one byte,
/// every byte, and under the room rule packs the groups of the channel of one
/// and, under the deterministic grind, proves its bytes; with every byte,
/// groups do wait behind one another, and with none they never do. A block
/// with a store keeps the channel of one whatever the knob says.
#[test]
fn a_deeper_group_queue_moves_no_byte_of_the_proof() {
    use crate::block_whir::BlockSpillPolicy;
    use crate::block_whir::group_queue::Setting;
    use crate::block_whir::regen::RegenMode;
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    // Phase A takes a group every ten milliseconds, so the packer is ahead of
    // it in every arm.
    let queued = |setting: Setting, policy: BlockSpillPolicy| {
        let (proof, stamps) = regenerated(
            &elf,
            &format,
            RegenMode::Always,
            policy,
            Deviations {
                group_queue: Some(setting),
                group_take_pause: Some(std::time::Duration::from_millis(10)),
                ..Deviations::default()
            },
        )
        .unwrap_or_else(|e| panic!("{setting:?}, {policy:?}: {e:?}"));
        let line = stamps.group_queue.expect("the queue's line");
        (proof, line)
    };
    let (plain, line) = queued(Setting::None, BlockSpillPolicy::Off);
    assert!(groups_of(&plain) >= 3, "{} groups", groups_of(&plain));
    assert!(verify(&plain, &elf, &format));
    assert_eq!(
        line,
        "BLOCK GROUP QUEUE: none · one closed group waits (the default)"
    );
    let same = |what: &str, proof: &BlockWhirProof| {
        assert_eq!(proof.groups, plain.groups, "{what}: the partition");
        assert_eq!(
            format!("{:?}", proof.table_counts),
            format!("{:?}", plain.table_counts),
            "{what}"
        );
        assert!(verify(proof, &elf, &format), "{what}");
        if crypto::grinding::deterministic() {
            assert_eq!(bytes(proof), bytes(&plain), "{what}: the proof");
        }
    };
    for (what, setting) in [
        ("no byte", Setting::Bytes(0)),
        ("one byte", Setting::Bytes(1)),
        ("every byte", Setting::Bytes(u64::MAX)),
    ] {
        let (proof, line) = queued(setting, BlockSpillPolicy::Off);
        same(what, &proof);
        assert!(line.contains("fixed, the host not read"), "{what}: {line}");
        let waited = !line.contains("none waited behind another");
        assert_eq!(
            waited,
            setting == Setting::Bytes(u64::MAX),
            "{what}: {line}"
        );
    }
    let (proof, line) = queued(Setting::Room, BlockSpillPolicy::Off);
    same("room", &proof);
    assert!(
        line.contains("GiB waiting while VmRSS + the group + 16.00 stays under the line"),
        "{line}"
    );
    let (proof, line) = queued(Setting::Bytes(u64::MAX), BlockSpillPolicy::Always);
    same("a store", &proof);
    assert!(
        line.contains("the block does not prove with no disk: one closed group waits"),
        "{line}"
    );
}

/// The partition is part of the statement: a table named twice, two tables
/// swapped inside a group, or a table moved to another group is refused.
#[test]
fn a_tampered_partition_is_refused() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let proof = prove(&elf, &format, &streamed(MaxRowsConfig::small(), 16, 3));
    assert!(verify(&proof, &elf, &format));

    let mut twice = proof.clone();
    let first = twice.groups[0][0];
    twice.groups[1][0] = first;
    assert!(!verify(&twice, &elf, &format));

    let mut swapped = proof.clone();
    let g = swapped
        .groups
        .iter()
        .position(|g| g.len() >= 2)
        .expect("a group of two");
    swapped.groups[g].swap(0, 1);
    assert!(!verify(&swapped, &elf, &format));

    let mut moved = proof.clone();
    let table = moved.groups[0].pop().expect("a table");
    moved.groups[1].insert(0, table);
    assert!(!verify(&moved, &elf, &format));
}

/// A table missing from the partition is refused.
#[test]
fn a_partition_missing_a_table_is_refused() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let mut proof = prove(&elf, &format, &streamed(MaxRowsConfig::small(), 16, 3));
    let g = proof
        .groups
        .iter()
        .position(|g| g.len() >= 2)
        .expect("a group of two");
    proof.groups[g].pop();
    assert!(!verify(&proof, &elf, &format));
}

/// ★ A group over the verifier's polynomial budget is refused — by that check
/// alone: the same honest proof verifies under a budget that admits it.
#[test]
fn a_group_over_the_stack_budget_is_refused() {
    let elf = asm_elf_bytes("all_instructions_64");
    let mut wide = many_groups();
    wide.group_polys = 4;
    let proof = prove(&elf, &wide, &streamed(MaxRowsConfig::small(), 16, 3));
    assert!(
        verify(&proof, &elf, &wide),
        "the honest proof under its own budget"
    );
    let narrow = BlockFormat {
        group_polys: 2,
        ..wide
    };
    assert!(!verify(&proof, &elf, &narrow));
}

/// ★ More groups than the verifier's maximum is refused — by that check alone:
/// the same honest proof verifies under a maximum that admits it.
#[test]
fn a_partition_over_the_group_maximum_is_refused() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let proof = prove(&elf, &format, &streamed(MaxRowsConfig::small(), 16, 3));
    let n = proof.groups.len();
    assert!(n >= 3 && verify(&proof, &elf, &format));
    let tight = BlockFormat {
        max_groups: n - 1,
        ..format
    };
    assert!(!verify(&proof, &elf, &tight));
    let exact = BlockFormat {
        max_groups: n,
        ..format
    };
    assert!(verify(&proof, &elf, &exact));
}

/// ★ The prover refuses a partition over the group maximum as its groups
/// close, with the verifier's own error, on the streamed path and the
/// whole-run one, instead of proving a block the verifier refuses.
#[test]
fn the_prover_refuses_a_partition_over_the_group_maximum() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    for opts in [
        streamed(MaxRowsConfig::small(), 16, 3),
        options(MaxRowsConfig::small(), 16),
    ] {
        let proof = prove(&elf, &format, &opts);
        let n = proof.groups.len();
        assert!(n >= 2, "{n} groups");
        let tight = BlockFormat {
            max_groups: n - 1,
            ..format
        };
        let refused = prove_block_whir_with(
            &elf,
            &[],
            &ProofOptions::default_test_options(),
            &tight,
            &opts,
            &Deviations::default(),
        );
        let Err(crate::Error::InvalidTableCounts(prover)) = refused else {
            panic!("the prover proved {n} groups over a maximum of {}", n - 1);
        };
        let Err(crate::Error::InvalidTableCounts(verifier)) =
            verify_block_whir(&proof, &elf, &ProofOptions::default_test_options(), &tight)
        else {
            panic!("the verifier did not refuse {n} groups");
        };
        assert_eq!(prover, verifier);
    }
}

/// One table's claimed column value moved: its group's opening refuses it.
#[test]
fn a_tampered_column_claim_is_refused() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let mut proof = prove(&elf, &format, &options(MaxRowsConfig::small(), 16));
    let last = proof.proof.tables.len() - 1;
    proof.proof.tables[last].constraint.reduce.column_values[0] += FieldElement::<E>::one();
    assert!(!verify(&proof, &elf, &format));
}

/// One table's bus output moved: its GKR no longer lands on its trace.
#[test]
fn a_tampered_bus_output_is_refused() {
    let elf = asm_elf_bytes("sub");
    let format = many_groups();
    let mut proof = prove(&elf, &format, &options(MaxRowsConfig::small(), 16));
    proof.proof.tables[0].bus_output.0 += FieldElement::<E>::one();
    assert!(!verify(&proof, &elf, &format));
}

#[test]
fn a_tampered_public_output_is_refused() {
    let elf = asm_elf_bytes("sub");
    let mut proof = prove(&elf, &one_group(), &options(MaxRowsConfig::default(), 16));
    proof.public_output.push(0xff);
    assert!(!verify(&proof, &elf, &one_group()));
}

#[test]
fn a_restated_table_height_is_refused() {
    let elf = asm_elf_bytes("sub");
    let mut proof = prove(&elf, &one_group(), &options(MaxRowsConfig::default(), 16));
    proof.table_num_vars[0] += 1;
    assert!(!verify(&proof, &elf, &one_group()));
}

/// ⛔ A CPU table's stated height is what its timestamp base and the final PC
/// timestamp are built from: restated (and so absorbed again by the verifier,
/// the statement re-bound), within every bound, a full CPU table halved or the
/// last one doubled, the proof is refused; past the block's 2^21 cap it is
/// refused before anything is built.
#[test]
fn a_restated_cpu_table_height_is_refused() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let honest = prove(&elf, &format, &options(MaxRowsConfig::small(), 16));
    assert!(verify(&honest, &elf, &format));
    let start = crate::cpu_proof_start(true, &honest.table_counts);
    let cpus = honest.table_counts.cpu;
    assert!(cpus >= 2, "{cpus} CPU tables");
    let last = start + cpus - 1;
    for (what, at, vars) in [
        (
            "a full table halved",
            start,
            honest.table_num_vars[start] - 1,
        ),
        (
            "the last table doubled",
            last,
            honest.table_num_vars[last] + 1,
        ),
        ("a table past the block's cap", start, 22),
    ] {
        let mut restated = honest.clone();
        restated.table_num_vars[at] = vars;
        assert!(!verify(&restated, &elf, &format), "{what}: accepted");
    }
}

/// A root the layouts do not claim is refused before anything is drawn.
#[test]
fn an_extra_root_is_refused() {
    let elf = asm_elf_bytes("sub");
    let mut proof = prove(&elf, &one_group(), &options(MaxRowsConfig::default(), 16));
    let root = proof.proof.roots[0];
    proof.proof.roots.push(root);
    assert!(!verify(&proof, &elf, &one_group()));
}

/// DECODE's columns are settled by a prepared opening: one per prepared table,
/// and a small program's only prepared table is DECODE.
#[test]
fn the_block_carries_decodes_prepared_opening() {
    let elf = asm_elf_bytes("sub");
    let proof = prove(&elf, &one_group(), &options(MaxRowsConfig::default(), 16));
    assert_eq!(proof.prepared.len(), 1);
    assert!(verify(&proof, &elf, &one_group()));
}

/// The prepared set is DECODE and every genesis page the cross-epoch rule
/// stacks — a page with thousands of nonzero genesis bytes — and never a zero
/// page or a private one, whose columns cost the guest nothing or are committed.
#[test]
fn the_prepared_set_is_decode_and_the_dense_pages() {
    use crate::tables::page::PageConfig;
    let elf = asm_elf_bytes("sub");
    let counts = prove(&elf, &one_group(), &options(MaxRowsConfig::default(), 16)).table_counts;
    let program = executor::elf::Elf::load(&elf).expect("the ELF loads");
    let configs = vec![
        PageConfig::zero_init(0x1000_0000),
        PageConfig {
            page_base: 0x2000_0000,
            init_values: Some(vec![7u8; 20_000]),
            is_private_input: false,
        },
        PageConfig {
            page_base: 0x3000_0000,
            init_values: Some(vec![7u8; 20_000]),
            is_private_input: true,
        },
    ];
    let opts = ProofOptions::default_test_options();
    let airs = crate::VmAirs::new(
        &program,
        &opts,
        false,
        &configs,
        &counts,
        &vec![4; counts.cpu],
        None,
        true,
        None,
        crate::test_utils::halted_register_commitment(&opts, &program, &[]).preprocessed(),
    );
    let prepared =
        block_whir::prepared_tables(&airs, &configs, &one_group()).expect("the prepared set");
    let refs = airs.air_refs();
    let decode = refs
        .iter()
        .position(|air| air.name() == "DECODE")
        .expect("a DECODE");
    let first_page = refs.len() - airs.pages.len() - {
        // The tables after the pages, in `air_refs` order.
        airs.memw_registers.len()
            + airs.eqs.len()
            + airs.bytewises.len()
            + airs.stores.len()
            + airs.cpu32s.len()
    };
    let tables: Vec<usize> = prepared.iter().map(|(t, _)| *t).collect();
    assert_eq!(tables, vec![decode, first_page + 1]);
    assert_eq!(
        prepared[1].1,
        crate::tables::page::preprocessed_columns(&configs[1])
    );
}

/// A prepared opening altered in one value is refused. The openings sit last on
/// their group's fork, so nothing after them reads the tamper: only the
/// opening's own check can refuse it.
#[test]
fn a_tampered_prepared_opening_is_refused() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let mut proof = prove(&elf, &format, &options(MaxRowsConfig::small(), 16));
    assert!(verify(&proof, &elf, &format));
    proof.prepared[0].polys[0].final_value += FieldElement::<E>::one();
    assert!(!verify(&proof, &elf, &format));
}

/// A proof that leaves a prepared opening out is refused: the verifier derives
/// the prepared set from the program, not from the proof.
#[test]
fn a_missing_prepared_opening_is_refused() {
    let elf = asm_elf_bytes("sub");
    let mut proof = prove(&elf, &one_group(), &options(MaxRowsConfig::default(), 16));
    proof.prepared.pop();
    assert!(!verify(&proof, &elf, &one_group()));
}

/// A prover that commits DECODE's prepared columns for another program (one
/// entry differs) and opens that commitment is refused: the verifier absorbs
/// the root it derived itself, so the transcripts part at the roots block.
#[test]
fn a_prepared_opening_of_another_program_is_refused() {
    let elf = asm_elf_bytes("sub");
    let other = prove_with(
        &elf,
        &one_group(),
        &options(MaxRowsConfig::default(), 16),
        &Deviations {
            other_prepared: true,
            ..Default::default()
        },
    );
    assert!(!verify(&other, &elf, &one_group()));
}

/// The statement as one byte run is the stream `absorb_block` appends: two
/// transcripts fed each way draw the same challenge after a root.
#[test]
fn the_statement_bytes_are_what_absorb_block_appends() {
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let proof = prove(&elf, &format, &options(MaxRowsConfig::small(), 16));
    let opts = ProofOptions::default_test_options();
    let frame = block_whir::block_frame(proof.statement(), &elf, &opts, &format).expect("frame");
    let digest = crate::statement::elf_digest(&elf);
    let tag = block_whir::block_statement_tag(&format.zf.base);
    let bytes = block_whir::block_statement_bytes(proof.statement(), &tag, &digest, &frame.config);
    let (a, b) = crate::with_whir_hash!(|H| {
        type T = DefaultTranscript<E, <H as multilinear::whir_hash::WhirHash>::Transcript>;
        let mut host = T::new(&[]);
        block_whir::absorb_block(
            &mut host,
            &tag,
            &elf,
            &proof.public_output,
            &proof.table_counts,
            proof.num_private_input_pages,
            &proof.runtime_page_ranges,
            &proof.table_num_vars,
            &frame.config,
            &proof.groups,
        );
        let mut run = T::new(&[]);
        run.append_bytes(&bytes);
        for t in [&mut host, &mut run] {
            t.append_bytes(&proof.proof.roots[0]);
        }
        let a: FieldElement<E> = host.sample_field_element();
        let b: FieldElement<E> = run.sample_field_element();
        (a, b)
    });
    assert_eq!(a, b);
    assert_eq!(bytes.len() % 8, 0, "the run ends on a felt boundary");
}

// ======== prepared openings on two dense pages (test_dense_pages) ========

/// The prepared tables of `proof`'s statement, in AIR order, as the verifier
/// derives them: `(AIR index, position in the proof's table order)`.
fn prepared_of(elf: &[u8], proof: &BlockWhirProof, format: &BlockFormat) -> Vec<(usize, usize)> {
    let opts = ProofOptions::default_test_options();
    let frame = block_whir::block_frame(proof.statement(), elf, &opts, format).expect("frame");
    let order: Vec<usize> = proof.groups.iter().flatten().map(|&t| t as usize).collect();
    block_whir::prepared_tables(&frame.airs, &frame.page_configs, format)
        .expect("the prepared set")
        .into_iter()
        .map(|(table, _)| {
            let position = order.iter().position(|&t| t == table).expect("in a group");
            (table, position)
        })
        .collect()
}

fn verify_skipping_prepared(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> bool {
    block_whir::verify_block_whir_with(
        proof,
        elf,
        &ProofOptions::default_test_options(),
        format,
        true,
        VerifierChecks::ALL,
    )
    .unwrap_or(false)
}

fn dense_block_in(format: &BlockFormat, deviations: &Deviations) -> (Vec<u8>, BlockWhirProof) {
    let elf = asm_elf_bytes("test_dense_pages");
    let proof = prove_with(
        &elf,
        format,
        &options(MaxRowsConfig::default(), 16),
        deviations,
    );
    (elf, proof)
}

fn dense_block(deviations: &Deviations) -> (Vec<u8>, BlockWhirProof) {
    dense_block_in(&one_group(), deviations)
}

/// The group holding AIR table `table` in `proof`'s partition.
fn group_of(proof: &BlockWhirProof, table: usize) -> usize {
    proof
        .groups
        .iter()
        .position(|g| g.contains(&(table as u32)))
        .expect("in a group")
}

/// The fixture is what the negatives below need: DECODE and at least two dense
/// genesis pages prepared, one stack a group that holds any, and an honest
/// proof of it verifies.
#[test]
fn the_dense_fixture_prepares_decode_and_two_pages() {
    let (elf, proof) = dense_block(&Deviations::default());
    let prepared = prepared_of(&elf, &proof, &one_group());
    assert!(prepared.len() >= 3, "DECODE and two pages: {prepared:?}");
    let mut groups: Vec<usize> = prepared.iter().map(|&(t, _)| group_of(&proof, t)).collect();
    groups.dedup();
    assert_eq!(proof.prepared.len(), groups.len(), "one opening a group");
    assert!(verify(&proof, &elf, &one_group()));
}

/// Two groups' prepared openings swapped in the proof: each opens against the
/// other group's derived roots, and is refused. With the openings left
/// unchecked (the mutation) the swap verifies: the openings are what refuse it.
#[test]
fn two_groups_prepared_openings_swapped_are_refused() {
    let format = many_groups();
    let (elf, mut proof) = dense_block_in(&format, &Deviations::default());
    assert!(proof.prepared.len() >= 2, "{} stacks", proof.prepared.len());
    assert!(verify(&proof, &elf, &format));
    proof.prepared.swap(0, 1);
    assert!(!verify(&proof, &elf, &format));
    assert!(verify_skipping_prepared(&proof, &elf, &format));
}

/// A page's block opened at another table's point (a table of its group and its
/// height) — a valid opening of its own stack, of the wrong claim — is refused;
/// the mutation admits it.
#[test]
fn a_page_opened_at_another_tables_point_is_refused() {
    let (elf, honest) = dense_block(&Deviations::default());
    let prepared = prepared_of(&elf, &honest, &one_group());
    let page = prepared[1].0;
    let height = honest.table_num_vars[page];
    let other = honest.groups[group_of(&honest, page)]
        .iter()
        .map(|&t| t as usize)
        .find(|&t| t != page && honest.table_num_vars[t] == height)
        .expect("the page shares its group with a table of its height");
    let (_, wrong) = dense_block(&Deviations {
        prepared_points: vec![(page, other)],
        ..Default::default()
    });
    assert!(!verify(&wrong, &elf, &one_group()));
    assert!(verify_skipping_prepared(&wrong, &elf, &one_group()));
}

/// Stacks committed wrongly — two pages' blocks swapped in their stack, a page's
/// block holding the other page's columns, DECODE's block another program's —
/// each opened consistently: refused at the roots block, since the verifier
/// absorbs the stacks it derives from the program and the partition, so even
/// with the openings unchecked. (The leaf's form, whose mutation trusts the
/// prover's roots, is in `lfm::whir_block_tests`.)
#[test]
fn a_wrongly_committed_prepared_stack_is_refused() {
    use crate::block_whir::StackTamper;
    let (elf, honest) = dense_block(&Deviations::default());
    let prepared = prepared_of(&elf, &honest, &one_group());
    let (a, b) = (prepared[1].0, prepared[2].0);
    let mut tampers = vec![Deviations {
        prepared_stack: vec![StackTamper::ColumnsOf { table: a, from: b }],
        ..Default::default()
    }];
    if group_of(&honest, a) == group_of(&honest, b) {
        tampers.push(Deviations {
            prepared_stack: vec![StackTamper::Swap(a, b)],
            ..Default::default()
        });
    }
    tampers.push(Deviations {
        other_prepared: true,
        ..Default::default()
    });
    for deviations in &tampers {
        let (_, wrong) = dense_block(deviations);
        assert!(!verify(&wrong, &elf, &one_group()));
        assert!(!verify_skipping_prepared(&wrong, &elf, &one_group()));
    }
}

/// With the openings off (`BlockFormat::prepared`, the measurement arm) the
/// block carries none and the host verifier evaluates the columns itself.
#[test]
fn a_block_without_prepared_openings_verifies_on_the_host() {
    let format = BlockFormat {
        prepared: false,
        ..one_group()
    };
    let elf = asm_elf_bytes("test_dense_pages");
    let proof = prove(&elf, &format, &options(MaxRowsConfig::default(), 16));
    assert!(proof.prepared.is_empty());
    assert!(verify(&proof, &elf, &format));
    assert!(
        !verify(&proof, &elf, &one_group()),
        "the formats are not interchangeable"
    );
}

/// The KECCAK_RND count is bounded before the verifier builds an AIR off it,
/// and a table the block does not chunk (HINT) stays one table.
#[test]
fn an_inflated_keccak_rnd_count_is_refused() {
    let mut counts = prove(
        &asm_elf_bytes("sub"),
        &one_group(),
        &options(MaxRowsConfig::default(), 16),
    )
    .table_counts;
    counts.keccak_rnd = block_whir::BLOCK_MAX_KECCAK_RND + 1;
    assert!(block_whir::validate_block_counts(&counts).is_err());
    counts.keccak_rnd = 3;
    assert!(block_whir::validate_block_counts(&counts).is_ok());
    counts.hint = 2;
    assert!(block_whir::validate_block_counts(&counts).is_err());
}

/// Every accelerator present, KECCAK, KECCAK_RND and ECSM in two tables each
/// and ECDAS in three: in proof order (`VmAirs::air_refs`) the five fixed
/// tables, COMMIT 5, KECCAK 6–7, KECCAK_RND 8–9, ECSM 10–11, ECDAS 12–14,
/// HINT 15, CPU 16–17, MEMW_R 18.
fn chunked_counts() -> crate::TableCounts {
    crate::TableCounts {
        cpu: 2,
        lt: 0,
        memw: 0,
        memw_aligned: 0,
        load: 0,
        mul: 0,
        dvrm: 0,
        shift: 0,
        branch: 0,
        memw_register: 1,
        eq: 0,
        bytewise: 0,
        store: 0,
        cpu32: 0,
        keccak: 2,
        keccak_rnd: 2,
        ecsm: 2,
        ecdas: 3,
        hint: 1,
        copy: 2,
        blake3: 0,
        is_wh: 0,
    }
}

/// The ECDAS count is bounded before the verifier builds an AIR off it, and
/// HINT (a table the block does not chunk) stays one table.
#[test]
fn an_inflated_ecdas_count_is_refused() {
    let mut counts = chunked_counts();
    assert!(block_whir::validate_block_counts(&counts).is_ok());
    counts.ecdas = block_whir::BLOCK_MAX_ECDAS + 1;
    assert!(block_whir::validate_block_counts(&counts).is_err());
    counts.ecdas = block_whir::BLOCK_MAX_ECDAS;
    assert!(block_whir::validate_block_counts(&counts).is_ok());
    counts.hint = 2;
    assert!(block_whir::validate_block_counts(&counts).is_err());
}

/// ★ COPY, KECCAK and ECSM are chunked in the block: any count up to
/// [`block_whir::BLOCK_MAX_COPY`] / [`block_whir::BLOCK_MAX_KECCAK`] /
/// [`block_whir::BLOCK_MAX_ECSM`] passes, one more is refused before the
/// verifier builds an AIR off it. Every other verifier keeps them to one table
/// ([`crate::TableCounts::validate`]).
///
/// Mutations (laptop): the bound dropped and `KECCAK: MAX + 1` passes; the
/// one-table bound left on KECCAK or ECSM and `2` is refused.
#[test]
fn keccak_and_ecsm_counts_are_bounded_by_the_block() {
    type Field = fn(&mut crate::TableCounts) -> &mut usize;
    let tables: [(&str, usize, Field); 3] = [
        ("COPY", block_whir::BLOCK_MAX_COPY, |c| &mut c.copy),
        ("KECCAK", block_whir::BLOCK_MAX_KECCAK, |c| &mut c.keccak),
        ("ECSM", block_whir::BLOCK_MAX_ECSM, |c| &mut c.ecsm),
    ];
    for (name, max, field) in tables {
        let mut single = chunked_counts();
        (single.copy, single.keccak, single.keccak_rnd) = (1, 1, 1);
        (single.ecsm, single.ecdas) = (1, 1);
        assert!(single.validate().is_ok());
        *field(&mut single) = 2;
        assert!(
            single.validate().is_err(),
            "every other verifier keeps {name} to one table"
        );
        for (count, ok) in [(1, true), (2, true), (max, true), (max + 1, false)] {
            let mut counts = chunked_counts();
            *field(&mut counts) = count;
            let checked = block_whir::validate_block_counts(&counts);
            assert_eq!(
                checked.is_ok(),
                ok,
                "{name} count {count}: {checked:?} (the block takes at most {max})"
            );
        }
    }
}

/// ★ The statement's heights are capped for the chunked tables, and only
/// theirs: a COPY table over [`block_whir::BLOCK_COPY_MAX_VARS`], a KECCAK
/// table over [`block_whir::BLOCK_KECCAK_MAX_VARS`], a KECCAK_RND table over
/// [`block_whir::BLOCK_KECCAK_RND_MAX_VARS`], an ECSM table over
/// [`block_whir::BLOCK_ECSM_MAX_VARS`] and an ECDAS table over
/// [`block_whir::BLOCK_ECDAS_MAX_VARS`] are refused; every other table far over
/// every cap passes. The controls next to the ranges (HALT before COPY, HINT
/// after ECDAS) are what an off-by-one in the ranges would cap instead; each
/// table at its own cap passing pins which cap each range carries.
#[test]
fn the_block_caps_the_chunked_tables_heights() {
    let counts = chunked_counts();
    let n = counts.total().expect("fits") + crate::FIXED_TABLE_COUNT;
    assert_eq!(n, 20);
    let check = |vars: &[u8]| block_whir::check_chunked_heights(&counts, vars);
    let honest = vec![10u8; n];
    assert!(check(&honest).is_ok());
    let copy_cap = block_whir::BLOCK_COPY_MAX_VARS as u8;
    let (keccak_cap, keccak_rnd_cap, ecsm_cap, ecdas_cap) = (
        block_whir::BLOCK_KECCAK_MAX_VARS as u8,
        block_whir::BLOCK_KECCAK_RND_MAX_VARS as u8,
        block_whir::BLOCK_ECSM_MAX_VARS as u8,
        block_whir::BLOCK_ECDAS_MAX_VARS as u8,
    );
    let mut at_caps = honest.clone();
    at_caps[5..7].fill(copy_cap);
    at_caps[7..9].fill(keccak_cap);
    at_caps[9..11].fill(keccak_rnd_cap);
    at_caps[11..13].fill(ecsm_cap);
    at_caps[13..16].fill(ecdas_cap);
    assert!(check(&at_caps).is_ok(), "a table at its cap passes");
    for (what, idx, vars) in [
        ("ECDAS[2] over its cap", 15, ecdas_cap + 1),
        ("ECDAS[0] over its cap", 13, ecdas_cap + 1),
        ("ECSM[1] over its cap", 12, ecsm_cap + 1),
        ("ECSM[0] over its cap", 11, ecsm_cap + 1),
        ("KECCAK_RND[1] over its cap", 10, keccak_rnd_cap + 1),
        ("KECCAK_RND[0] over its cap", 9, keccak_rnd_cap + 1),
        ("KECCAK[1] over its cap", 8, keccak_cap + 1),
        ("KECCAK[0] over its cap", 7, keccak_cap + 1),
        ("COPY[1] over its cap", 6, copy_cap + 1),
        ("COPY[0] over its cap", 5, copy_cap + 1),
    ] {
        let mut stated = honest.clone();
        stated[idx] = vars;
        let refused = check(&stated);
        assert!(refused.is_err(), "the block must refuse {what}");
        println!("BLOCK WHIR refuses {what}: {:?}", refused.err());
    }
    for (what, idx) in [("HALT", 4), ("HINT", 16), ("CPU[0]", 17), ("MEMW_R[0]", 19)] {
        let mut stated = honest.clone();
        stated[idx] = 22;
        assert!(
            check(&stated).is_ok(),
            "{what} is not a chunked table: its height is not capped"
        );
    }
}

/// ★ Every chunked table's cap is the tallest power of two at which its
/// columns fill at most one polynomial of the format's 2^27 stack: COPY
/// 41 × 2^21, KECCAK 511 × 2^18, KECCAK_RND 1,480 × 2^16, ECSM 667 × 2^17,
/// ECDAS 521 × 2^17. One doubling past a cap would stack the table into two
/// polynomials.
///
/// Mutation (laptop): the KECCAK or the ECSM cap lifted, and this test fails
/// (the caps test reads the constants, so it cannot see a lifted cap).
#[test]
fn every_chunked_cap_is_the_tallest_table_one_stacked_polynomial_holds() {
    use crate::tables::{copy, ecdas, ecsm, keccak, keccak_rnd};
    let stack = ZfFormat::DEFAULT.whir_stack.get();
    let widths = [
        ("COPY", copy::cols::NUM_COLUMNS),
        ("KECCAK", keccak::cols::NUM_COLUMNS),
        ("KECCAK_RND", keccak_rnd::cols::NUM_COLUMNS),
        ("ECSM", ecsm::cols::NUM_COLUMNS),
        ("ECDAS", ecdas::cols::NUM_COLUMNS),
    ];
    for ((name, _, cap), (table, width)) in block_whir::chunked_table_ranges(&chunked_counts())
        .into_iter()
        .zip(widths)
    {
        assert_eq!(name, table);
        let cells = |vars: usize| width << vars;
        assert!(
            cells(cap) <= 1 << stack,
            "{name}: {width} columns at 2^{cap} rows must fit one polynomial of 2^{stack}"
        );
        assert!(
            cells(cap + 1) > 1 << stack,
            "{name}: 2^{cap} rows must be the tallest that fits one polynomial of 2^{stack}"
        );
        println!(
            "BLOCK WHIR CAP {name}: {width} columns × 2^{cap} = {} ≤ 2^{stack} < {width} × 2^{}",
            cells(cap),
            cap + 1
        );
    }
}

/// ★ G2: no table may be stated taller than [`block_whir::BLOCK_MAX_TABLE_VARS`]
/// (2^27), so no chain is taller than the 2^27 stack; at the cap it passes.
#[test]
fn no_table_may_be_stated_over_the_table_cap() {
    let cap = block_whir::BLOCK_MAX_TABLE_VARS as u8;
    let mut stated = vec![10u8; 17];
    assert!(block_whir::check_table_heights(&stated).is_ok());
    stated[3] = cap;
    assert!(
        block_whir::check_table_heights(&stated).is_ok(),
        "at the cap"
    );
    stated[3] = cap + 1;
    let refused = block_whir::check_table_heights(&stated);
    assert!(
        format!("{refused:?}").contains("table 3"),
        "a table over the cap must be refused, got {refused:?}"
    );
}

/// ★ G2 end to end: a statement giving table 0 (BITWISE) 2^28 rows is refused
/// at the frame with the cap's own error, before any AIR is built.
///
/// Mutation (box script): `check_table_heights` returning `Ok(())` and the error
/// is no longer the cap's.
#[test]
fn a_table_stated_over_the_table_cap_is_refused() {
    let elf = asm_elf_bytes("sub");
    let format = many_groups();
    let mut proof = prove(&elf, &format, &options(MaxRowsConfig::default(), 16));
    assert!(verify(&proof, &elf, &format));
    proof.table_num_vars[0] = (block_whir::BLOCK_MAX_TABLE_VARS + 1) as u8;
    let refused =
        block_whir::verify_block_whir(&proof, &elf, &ProofOptions::default_test_options(), &format);
    let msg = format!("{refused:?}");
    assert!(
        msg.contains("table 0 states 2^28 rows") && msg.contains("at most 2^27"),
        "the frame must refuse a table over the table cap with the cap's error, got {msg}"
    );
    println!("BLOCK WHIR TABLE OVER CAP: refused at the frame: {msg}");
}

/// The ranges the caps read are the chunked AIRs' positions in
/// [`crate::VmAirs::air_refs`] — the statement's table order — and nothing
/// next to them.
#[test]
fn chunked_table_ranges_name_the_chunked_airs() {
    use executor::elf::Elf;
    let elf = asm_elf_bytes("poc_rodata_commit");
    let program = Elf::load(&elf).expect("load the ELF");
    let page_configs = crate::tables::trace_builder::Traces::page_configs_from_elf(&program);
    let counts = chunked_counts();
    let airs = crate::VmAirs::new(
        &program,
        &ProofOptions::default_test_options(),
        false,
        &page_configs,
        &counts,
        &vec![4; counts.cpu],
        None,
        true,
        None,
        crate::test_utils::halted_register_commitment(
            &ProofOptions::default_test_options(),
            &program,
            &[],
        )
        .preprocessed(),
    );
    let names: Vec<String> = airs
        .air_refs()
        .iter()
        .map(|a| a.name().to_string())
        .collect();
    let ranges = block_whir::chunked_table_ranges(&counts);
    assert_eq!(
        ranges.each_ref().map(|(table, ..)| *table),
        ["COPY", "KECCAK", "KECCAK_RND", "ECSM", "ECDAS"],
        "the chunked tables in AIR order"
    );
    for (table, range, _) in ranges {
        assert!(
            range.len() >= 2,
            "{table}: every chunked table in two tables or more"
        );
        for (k, idx) in range.clone().enumerate() {
            assert_eq!(names[idx], format!("{table}[{k}]"));
        }
        let prefix = format!("{table}[");
        assert!(!names[range.start - 1].starts_with(&prefix));
        assert!(!names[range.end].starts_with(&prefix));
    }
}

/// ECDAS cut into tables of 2^4 rows: test_ecsm_multi's 42 double/add steps
/// (calls k = 1, 5, 0xABCDEF) over three tables at the default (the fourth,
/// padding rows alone, is not built), the 0xABCDEF call through all three;
/// their bus sums add up to the whole table's.
#[test]
fn ecdas_split_into_chunks_proves_and_verifies() {
    let elf = asm_elf_bytes("test_ecsm_multi");
    let mut o = options(MaxRowsConfig::default(), 16);
    o.ecdas_rows_log2 = 4;
    let proof = prove(&elf, &many_groups(), &o);
    assert_eq!(proof.table_counts.ecdas, 3, "42 rows in 16-row tables");
    assert!(verify(&proof, &elf, &many_groups()));
    // The production cut leaves the same program one table.
    let one = prove(&elf, &many_groups(), &options(MaxRowsConfig::default(), 16));
    assert_eq!(one.table_counts.ecdas, 1);
    assert!(verify(&one, &elf, &many_groups()));
}

/// `program`'s traces built once, KECCAK_RND at its production cut, then
/// `cut`.
fn split_traces(
    program: &str,
    cut: impl FnOnce(&mut crate::tables::trace_builder::Traces),
) -> (
    executor::elf::Elf,
    Vec<u8>,
    crate::tables::trace_builder::Traces,
) {
    use executor::elf::Elf;
    use executor::vm::execution::Executor;
    let elf = asm_elf_bytes(program);
    let program = Elf::load(&elf).expect("the ELF loads");
    let logs = Executor::new(&program, Vec::new())
        .expect("the executor starts")
        .run()
        .expect("the program runs")
        .logs;
    let mut traces = crate::tables::trace_builder::Traces::from_elf_and_logs(
        &program,
        &logs,
        &MaxRowsConfig::default(),
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("the traces build");
    block_whir::split_keccak_rnd(&mut traces, 16, CutPadding::default()).expect("the split");
    cut(&mut traces);
    (program, elf, traces)
}

/// test_ecsm_multi's traces with ECDAS cut at 16 rows, built once.
fn ecsm_multi_split() -> (
    executor::elf::Elf,
    Vec<u8>,
    crate::tables::trace_builder::Traces,
) {
    split_traces("test_ecsm_multi", |traces| {
        block_whir::split_ecdas(traces, 4, CutPadding::WholeRun);
        assert_eq!(traces.ecdases.len(), 4);
    })
}

/// Whether the traces `build` makes, after `tamper`, prove into a block the
/// verifier accepts (a prover refusal counts as not accepted). The traces are
/// built afresh for each prove.
fn traces_block_accepts(
    build: impl Fn() -> (
        executor::elf::Elf,
        Vec<u8>,
        crate::tables::trace_builder::Traces,
    ),
    tamper: impl Fn(&mut crate::tables::trace_builder::Traces),
) -> bool {
    let format = many_groups();
    let (program, elf, mut traces) = build();
    tamper(&mut traces);
    match block_whir::prove_traces(
        &program,
        &elf,
        &mut traces,
        &ProofOptions::default_test_options(),
        &format,
        &options(MaxRowsConfig::default(), 16),
        &Deviations::default(),
        false,
        &|_, _| {},
        &mut Default::default(),
    ) {
        Ok(proof) => verify(&proof, &elf, &format),
        Err(e) => {
            println!("    (the prover refused: {e:?})");
            false
        }
    }
}

/// Whether test_ecsm_multi's split traces, after `tamper`, prove into a block
/// the verifier accepts.
fn split_block_accepts(tamper: impl Fn(&mut crate::tables::trace_builder::Traces)) -> bool {
    traces_block_accepts(ecsm_multi_split, tamper)
}

/// ★ A call split across ECDAS tables is held together by the Ecdas bus alone,
/// keyed by the call's timestamp and the step's `(round, op)`.
///
/// - Control: two whole rows swapped between table 0 and table 1 (a step of
///   the 5·G call and a step of the 0xABCDEF call) is the same multiset of
///   rows, and verifies.
/// - Negative: the same two rows swap only their timestamps, so each claims
///   the other's call. No constraint and no range check reads the timestamp,
///   and both rows have `NEXT_OP = 0` (their `Bit` sends are off): only the
///   Ecdas tuples moved, and the block is refused.
///
/// Mutation (box script): drop the timestamp from `ecsm::ecdas_tuple` and the
/// negative verifies — the call key is what refuses it.
#[test]
fn a_split_ecdas_call_is_keyed_on_the_bus() {
    use crate::tables::ecdas::cols;
    use crate::tables::types::FE;
    let (a, b) = {
        let (_, _, traces) = ecsm_multi_split();
        let next_op_off =
            |t: usize, r: usize| *traces.ecdases[t].main_table.get(r, cols::NEXT_OP) == FE::zero();
        let ts = |t: usize, r: usize| {
            (
                *traces.ecdases[t].main_table.get(r, cols::TIMESTAMP_0),
                *traces.ecdases[t].main_table.get(r, cols::TIMESTAMP_1),
            )
        };
        let a = (0..3).find(|&r| next_op_off(0, r)).expect("a 5·G step");
        let b = (0..16)
            .find(|&r| next_op_off(1, r))
            .expect("a 0xABCDEF step");
        assert_ne!(ts(0, a), ts(1, b), "two different calls");
        assert_eq!(*traces.ecdases[0].main_table.get(a, cols::MU), FE::one());
        assert_eq!(*traces.ecdases[1].main_table.get(b, cols::MU), FE::one());
        (a, b)
    };
    let swap = |cols_to_swap: Vec<usize>| {
        move |t: &mut crate::tables::trace_builder::Traces| {
            for &col in &cols_to_swap {
                let x = *t.ecdases[0].main_table.get(a, col);
                let y = *t.ecdases[1].main_table.get(b, col);
                t.ecdases[0].main_table.set(a, col, y);
                t.ecdases[1].main_table.set(b, col, x);
            }
        }
    };
    assert!(split_block_accepts(|_| {}), "the honest split verifies");
    assert!(
        split_block_accepts(swap((0..cols::NUM_COLUMNS).collect())),
        "whole rows swapped across tables are the same steps: accepted"
    );
    println!("BLOCK WHIR ECDAS SPLIT CONTROL: whole rows swapped across tables, accepted");
    assert!(
        !split_block_accepts(swap(vec![cols::TIMESTAMP_0, cols::TIMESTAMP_1])),
        "a continuation row reattached to another call must be refused"
    );
    println!(
        "BLOCK WHIR ECDAS SPLIT NEGATIVE: table 0 row {a} and table 1 row {b} swapped calls, refused"
    );
}

/// ★ COPY cut inside a call: test_copy_ecalls' 17 COPY rows (a 21-byte
/// memcpy's seven, a memset's four, a zero-length call, an overlapping
/// two-byte copy and a three-byte `write`) at one, two and four rows a table:
/// 17, 9 and 5 tables at the default (the tables of padding rows alone are not
/// built), and every proof verifies; the production cut leaves the guest one
/// table, and so does the streamed build at two rows a table.
#[test]
fn copy_split_into_chunks_proves_and_verifies() {
    let format = many_groups();
    let elf = asm_elf_bytes("test_copy_ecalls");
    for (rows_log2, tables) in [(0, 17), (1, 9), (2, 5)] {
        let mut o = options(MaxRowsConfig::default(), 16);
        o.copy_rows_log2 = rows_log2;
        let proof = prove(&elf, &format, &o);
        assert_eq!(
            proof.table_counts.copy, tables,
            "17 rows in 2^{rows_log2}-row tables"
        );
        assert!(
            verify(&proof, &elf, &format),
            "COPY: {tables} tables verify"
        );
        println!("BLOCK WHIR COPY CHUNKED: {tables} tables of 2^{rows_log2} rows, accepted");
    }
    let mut o = streamed(MaxRowsConfig::small(), 16, 4);
    o.copy_rows_log2 = 1;
    let proof = prove(&elf, &format, &o);
    assert_eq!(proof.table_counts.copy, 9, "COPY streamed");
    assert!(verify(&proof, &elf, &format), "COPY streamed");
    let (_, _, traces) = split_traces("test_copy_ecalls", |traces| {
        block_whir::split_copy(
            traces,
            block_whir::BLOCK_COPY_ROWS_LOG2,
            CutPadding::default(),
        );
    });
    assert_eq!(traces.copies.len(), 1, "COPY: production cut");
}

/// ★ A COPY cut may fall inside one call, whose rows reach each other only
/// through the CopyNext bus, keyed by the call's timestamp and its running
/// `(src, dst, count)`. At one row a table, the 21-byte memcpy's first two
/// rows are tables 0 and 1.
///
/// - Control: the two rows swapped as whole rows across the two tables are
///   the same multiset of rows, and verify.
/// - Negative: the second row's source moved by eight, its successor alike so
///   the row's own `ADD` still holds: the first row's CopyNext send no longer
///   meets it, and the block is refused.
#[test]
fn split_copy_chains_are_keyed_on_the_bus() {
    use crate::tables::copy::cols;
    use crate::tables::types::FE;
    let build = || {
        split_traces("test_copy_ecalls", |traces| {
            block_whir::split_copy(traces, 0, CutPadding::default());
            assert_eq!(traces.copies.len(), 17, "COPY: one row a table");
        })
    };
    {
        let (_, _, traces) = build();
        let get = |t: usize, col: usize| *traces.copies[t].main_table.get(0, col);
        assert_eq!(get(0, cols::FIRST), FE::one(), "table 0 starts the memcpy");
        assert_eq!(get(1, cols::FIRST), FE::zero(), "table 1 continues it");
        assert_eq!(get(0, cols::COUNT_0), FE::from(21u64));
        assert_eq!(get(1, cols::COUNT_0), FE::from(13u64));
    }
    assert!(
        traces_block_accepts(build, |_| {}),
        "COPY: the honest split verifies"
    );
    assert!(
        traces_block_accepts(build, |traces| {
            for col in 0..cols::NUM_COLUMNS {
                let x = *traces.copies[0].main_table.get(0, col);
                let y = *traces.copies[1].main_table.get(0, col);
                traces.copies[0].main_table.set(0, col, y);
                traces.copies[1].main_table.set(0, col, x);
            }
        }),
        "COPY: rows of one call swapped whole across tables are the same rows: accepted"
    );
    println!("BLOCK WHIR COPY SPLIT CONTROL: whole rows swapped across tables, accepted");
    assert!(
        !traces_block_accepts(build, |traces| {
            for col in [cols::SRC_0, cols::SRC_INCR_0] {
                let v = *traces.copies[1].main_table.get(0, col);
                traces.copies[1].main_table.set(0, col, v + FE::from(8u64));
            }
        }),
        "COPY: a continuation row's source moved across a cut must be refused"
    );
    println!("BLOCK WHIR COPY SPLIT NEGATIVE: a moved continuation source, refused");
}

/// The two call tables the block cuts between calls, with their guest (three
/// calls each: one table of four rows, the last row padding).
#[derive(Clone, Copy)]
enum CallTable {
    Keccak,
    Ecsm,
}

impl CallTable {
    const ALL: [CallTable; 2] = [CallTable::Keccak, CallTable::Ecsm];

    fn name(self) -> &'static str {
        match self {
            CallTable::Keccak => "KECCAK",
            CallTable::Ecsm => "ECSM",
        }
    }

    fn guest(self) -> &'static str {
        match self {
            CallTable::Keccak => "test_keccak_multi",
            CallTable::Ecsm => "test_ecsm_multi",
        }
    }

    /// `o` with this table cut at 2^`rows_log2` rows.
    fn cut(self, mut o: BlockOptions, rows_log2: usize) -> BlockOptions {
        match self {
            CallTable::Keccak => o.keccak_rows_log2 = rows_log2,
            CallTable::Ecsm => o.ecsm_rows_log2 = rows_log2,
        }
        o
    }

    /// The whole run padded: the padding row keeps its own table.
    fn split(self, traces: &mut crate::tables::trace_builder::Traces, rows_log2: usize) {
        match self {
            CallTable::Keccak => block_whir::split_keccak(traces, rows_log2, CutPadding::WholeRun),
            CallTable::Ecsm => block_whir::split_ecsm(traces, rows_log2, CutPadding::WholeRun),
        }
    }

    fn count(self, counts: &crate::TableCounts) -> usize {
        match self {
            CallTable::Keccak => counts.keccak,
            CallTable::Ecsm => counts.ecsm,
        }
    }

    fn tables(
        self,
        traces: &mut crate::tables::trace_builder::Traces,
    ) -> &mut Vec<stark::trace::TraceTable<crate::test_utils::F, E>> {
        match self {
            CallTable::Keccak => &mut traces.keccaks,
            CallTable::Ecsm => &mut traces.ecsms,
        }
    }

    /// Width, `MU` and the two timestamp columns.
    fn columns(self) -> (usize, usize, [usize; 2]) {
        use crate::tables::{ecsm, keccak};
        match self {
            CallTable::Keccak => (
                keccak::cols::NUM_COLUMNS,
                keccak::cols::MU,
                [keccak::cols::TIMESTAMP_0, keccak::cols::TIMESTAMP_1],
            ),
            CallTable::Ecsm => (
                ecsm::cols::NUM_COLUMNS,
                ecsm::cols::MU,
                [ecsm::cols::TIMESTAMP_0, ecsm::cols::TIMESTAMP_1],
            ),
        }
    }

    /// The guest's traces with this table cut at one row a table: its three
    /// calls and the padding row, four tables.
    fn one_call_a_table(
        self,
    ) -> (
        executor::elf::Elf,
        Vec<u8>,
        crate::tables::trace_builder::Traces,
    ) {
        split_traces(self.guest(), |traces| {
            self.split(traces, 0);
            assert_eq!(
                self.tables(traces).len(),
                4,
                "{}: one row a table",
                self.name()
            );
        })
    }
}

/// ★ KECCAK and ECSM cut between calls: test_keccak_multi's three
/// permutations and test_ecsm_multi's three scalar multiplications (one table
/// of four rows each, the last row padding) at one row a table (three tables
/// at the default: one call each, the padding row's table not built) and at
/// two rows a table (two); every proof verifies, the bus sums over the tables
/// adding up to the one table's. The production cuts leave each guest one
/// table.
#[test]
fn keccak_and_ecsm_split_into_chunks_prove_and_verify() {
    let format = many_groups();
    for table in CallTable::ALL {
        let (name, elf) = (table.name(), asm_elf_bytes(table.guest()));
        for (rows_log2, tables) in [(0, 3), (1, 2)] {
            let o = table.cut(options(MaxRowsConfig::default(), 16), rows_log2);
            let proof = prove(&elf, &format, &o);
            assert_eq!(
                table.count(&proof.table_counts),
                tables,
                "{name}: 4 rows in 2^{rows_log2}-row tables"
            );
            assert!(
                verify(&proof, &elf, &format),
                "{name}: {tables} tables verify"
            );
            println!("BLOCK WHIR {name} CHUNKED: {tables} tables of 2^{rows_log2} rows, accepted");
        }
        let (_, _, mut traces) = split_traces(table.guest(), |traces| {
            block_whir::split_keccak(
                traces,
                block_whir::BLOCK_KECCAK_ROWS_LOG2,
                CutPadding::default(),
            );
            block_whir::split_ecsm(
                traces,
                block_whir::BLOCK_ECSM_ROWS_LOG2,
                CutPadding::default(),
            );
        });
        assert_eq!(table.tables(&mut traces).len(), 1, "{name}: production cut");
    }
}

/// The streamed build cuts KECCAK and ECSM at the run's end like the whole-run
/// build: two rows a table, windows of 2^4 cycles.
#[test]
fn a_streamed_block_with_keccak_and_ecsm_chunks_proves_and_verifies() {
    let format = many_groups();
    for table in CallTable::ALL {
        let elf = asm_elf_bytes(table.guest());
        let o = table.cut(streamed(MaxRowsConfig::small(), 16, 4), 1);
        let proof = prove(&elf, &format, &o);
        assert_eq!(table.count(&proof.table_counts), 2, "{}", table.name());
        assert!(verify(&proof, &elf, &format), "{} streamed", table.name());
    }
}

/// ★ A KECCAK or ECSM row is one whole call, so a cut falls between calls; a
/// call reaches its rounds or steps, its scalar bits and its memory only
/// through buses keyed by its timestamp.
///
/// - Control: call 0 (table 0) and call 1 (table 1) swapped as whole rows is
///   the same multiset of rows, and verifies: the cut carries nothing a bus
///   does not.
/// - Negative: the same two rows swap only their timestamps, so each claims
///   the other's call (the Ecall receives still balance: both timestamps are
///   still received once); its memory, rounds or steps no longer meet it on
///   the buses, and the block is refused.
///
/// No mutation of its own: the timestamp key is each table's bus design,
/// which the cut leaves as it is (one table of these rows is refused alike).
#[test]
fn split_keccak_and_ecsm_calls_are_keyed_on_the_bus() {
    use crate::tables::types::FE;
    for table in CallTable::ALL {
        let name = table.name();
        let (width, mu, ts_cols) = table.columns();
        {
            let (_, _, mut traces) = table.one_call_a_table();
            let tables = table.tables(&mut traces);
            let ts = |t: usize| ts_cols.map(|c| *tables[t].main_table.get(0, c));
            assert_eq!(*tables[0].main_table.get(0, mu), FE::one());
            assert_eq!(*tables[1].main_table.get(0, mu), FE::one());
            assert_ne!(ts(0), ts(1), "{name}: two different calls");
        }
        let swap = |cols: Vec<usize>| {
            move |traces: &mut crate::tables::trace_builder::Traces| {
                let tables = table.tables(traces);
                for &col in &cols {
                    let x = *tables[0].main_table.get(0, col);
                    let y = *tables[1].main_table.get(0, col);
                    tables[0].main_table.set(0, col, y);
                    tables[1].main_table.set(0, col, x);
                }
            }
        };
        let build = || table.one_call_a_table();
        assert!(
            traces_block_accepts(build, |_| {}),
            "{name}: the honest split verifies"
        );
        assert!(
            traces_block_accepts(build, swap((0..width).collect())),
            "{name}: whole calls swapped across tables are the same calls: accepted"
        );
        println!("BLOCK WHIR {name} SPLIT CONTROL: whole rows swapped across tables, accepted");
        assert!(
            !traces_block_accepts(build, swap(ts_cols.to_vec())),
            "{name}: a call reattached to another call's timestamp must be refused"
        );
        println!(
            "BLOCK WHIR {name} SPLIT NEGATIVE: table 0 and table 1 swapped timestamps, refused"
        );
    }
}

/// ★ The verifier refuses a statement that gives a chunked table (KECCAK,
/// KECCAK_RND, ECSM or ECDAS) a height over its cap, at the frame (before any
/// AIR is built), with the cap's own error; the honest proofs verify.
///
/// Mutation (box script): `check_chunked_heights` returning `Ok(())`, or the
/// KECCAK and ECSM caps lifted, and the errors are no longer the cap's.
#[test]
fn a_chunked_table_stated_over_its_cap_is_refused() {
    let format = many_groups();
    for (program, tables) in [
        (
            "test_ecsm_multi",
            [
                ("ECDAS", block_whir::BLOCK_ECDAS_MAX_VARS),
                ("ECSM", block_whir::BLOCK_ECSM_MAX_VARS),
            ],
        ),
        (
            "test_keccak",
            [
                ("KECCAK_RND", block_whir::BLOCK_KECCAK_RND_MAX_VARS),
                ("KECCAK", block_whir::BLOCK_KECCAK_MAX_VARS),
            ],
        ),
    ] {
        let elf = asm_elf_bytes(program);
        let proof = prove(&elf, &format, &options(MaxRowsConfig::default(), 16));
        assert!(verify(&proof, &elf, &format), "{program} verifies");
        for (name, cap) in tables {
            let mut proof = proof.clone();
            let (_, range, _) = block_whir::chunked_table_ranges(&proof.table_counts)
                .into_iter()
                .find(|(n, ..)| *n == name)
                .expect("the table's range");
            assert!(!range.is_empty(), "{program} has a {name} table");
            proof.table_num_vars[range.start] = (cap + 1) as u8;
            let refused = block_whir::verify_block_whir(
                &proof,
                &elf,
                &ProofOptions::default_test_options(),
                &format,
            );
            let msg = format!("{refused:?}");
            assert!(
                msg.contains(&format!("{name}[0]")) && msg.contains(&format!("at most 2^{cap}")),
                "the frame must refuse {name} over its cap with the cap's error, got {msg}"
            );
            println!("BLOCK WHIR {name} OVER CAP: refused at the frame: {msg}");
        }
    }
}

/// ★ W1's readout on a real block (box only, `--ignored`): the block proved
/// in one proof at the production format, its phases stamped, then verified on
/// the host. Reads `BLOCK_WHIR_ELF` and `BLOCK_WHIR_INPUT`; prints the
/// `BLOCK …` lines of [`block_whir::BlockStamps::report`], the host peak and
/// `BLOCK VERIFY`.
#[test]
#[ignore = "a real block: box only"]
fn block_whir_on_a_real_block() {
    let elf_path = std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF");
    let input_path = std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT");
    let elf = std::fs::read(&elf_path).expect("read the ELF");
    let input = std::fs::read(&input_path).expect("read the input");
    // `BLOCK_WHIR_PREPARED=0`: the openings off, for the arm that prices them.
    let prepared = std::env::var("BLOCK_WHIR_PREPARED").map_or(true, |v| v.trim() != "0");
    // `BLOCK_WHIR_ARGUE=batched|per-table` (production: batched): each group's
    // tables argued together, at the format's bin cap (`BLOCK_WHIR_ARGUE_CAP=k`
    // for 2^k), or each on its own.
    let argue = match std::env::var("BLOCK_WHIR_ARGUE").as_deref().map(str::trim) {
        Ok("batched") => ArgueFormat::Batched {
            bin_log_cells: std::env::var("BLOCK_WHIR_ARGUE_CAP")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(match ArgueFormat::BATCHED {
                    ArgueFormat::Batched { bin_log_cells } => bin_log_cells,
                    ArgueFormat::PerTable => unreachable!("the batched format"),
                }),
        },
        Ok("per-table") => ArgueFormat::PerTable,
        Err(_) => BlockFormat::production().argue,
        Ok(other) => panic!("BLOCK_WHIR_ARGUE={other}: per-table or batched"),
    };
    // The block's base is its proof family's (D-P3B P3). `BLOCK_WHIR_BASE`
    // was this harness's word for it: `p1`/`p1w` are still accepted and change
    // nothing (an A/B against an older binary keeps equal environment bytes);
    // any other value — `rpx` — is refused rather than ignored.
    if let Ok(word) = std::env::var("BLOCK_WHIR_BASE") {
        assert!(
            matches!(word.trim(), "p1" | "p1w"),
            "BLOCK_WHIR_BASE={word}: the block's base is its proof family's (p1)"
        );
    }
    let production = BlockFormat::production();
    let base = production.zf.base;
    let format = BlockFormat {
        prepared,
        argue,
        ..production
    };
    println!(
        "BLOCK BASE: {:?} · arity-4 cap {} · statement tag {}",
        base.hash,
        base.arity4_cap,
        String::from_utf8_lossy(&block_whir::block_statement_tag(&base)),
    );
    let mut options = BlockOptions::production();
    // `BLOCK_WHIR_LAYOUT_WORKERS=n` (production 6; 0 is the inline layout) and
    // `BLOCK_WHIR_PACK_REST=0|1`, as the tree's harness takes them;
    // `BLOCK_WHIR_DROP_OPS=0`: the builder keeps the streamed chunks' ops
    // (production drops them).
    if let Some(n) = std::env::var("BLOCK_WHIR_LAYOUT_WORKERS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
    {
        options.layout_workers = n;
    }
    // `BLOCK_WHIR_LAYOUT_AHEAD=k` bounds the chunks unpacked at k + 1;
    // `none` lifts the bound (the reverted version, BIG 390).
    match std::env::var("BLOCK_WHIR_LAYOUT_AHEAD")
        .as_deref()
        .map(str::trim)
    {
        Ok("none") => options.layout_ahead = None,
        Ok(k) => {
            if let Ok(k) = k.parse() {
                options.layout_ahead = Some(k);
            }
        }
        Err(_) => {}
    }
    // `BLOCK_WHIR_PACK_REST=0|1` (production 0): the rest packed as it is laid
    // out.
    match std::env::var("BLOCK_WHIR_PACK_REST")
        .as_deref()
        .map(str::trim)
    {
        Ok("0") => options.pack_rest_as_laid_out = false,
        Ok("1") => options.pack_rest_as_laid_out = true,
        Ok(other) => panic!("BLOCK_WHIR_PACK_REST={other}: 0 or 1"),
        Err(_) => {}
    }
    match std::env::var("BLOCK_WHIR_DROP_OPS")
        .as_deref()
        .map(str::trim)
    {
        Ok("0") => options.drop_streamed_ops = false,
        Ok("1") => options.drop_streamed_ops = true,
        _ => {}
    }
    // `BLOCK_WHIR_NARROW=wide|card|host` (production card): how phase A holds
    // each group's columns once committed.
    if let Some(narrow) = crate::block_whir::narrow_from_env() {
        options.narrow = narrow;
    }
    // `BLOCK_WHIR_KECCAK_LOG2=k`, `BLOCK_WHIR_ECSM_LOG2=k`: force a KECCAK or
    // ECSM split (production 2^18 / 2^17).
    crate::block_whir::chunk_cuts_from_env(&mut options);
    // `BLOCK_WHIR_UPLOAD_AHEAD=0|1` (production 1): phase A puts each group's
    // columns on the card beside the previous group's commit.
    if let Some(ahead) = crate::block_whir::upload_ahead_from_env() {
        options.upload_ahead = ahead;
    }
    println!("BLOCK UPLOAD AHEAD: {}", options.upload_ahead);
    // `BLOCK_WHIR_REST_LAYOUT=all|<MiB>` (production 4096),
    // `BLOCK_WHIR_KR_FINISH_CHUNKS=0|1` and `BLOCK_WHIR_PACK_FINISHED=0|1`
    // (production 1): the rest's layout in waves, KECCAK_RND built as its
    // tables, the finish's tables packed as they are built.
    crate::block_whir::rest_layout_from_env(&mut options);
    println!(
        "BLOCK REST LAYOUT CONFIG: {} · KECCAK_RND built as its tables {} · finish packed {}",
        crate::block_whir::rest_layout_label(&options, crate::block_whir::rest_slide()),
        options.finish_keccak_rnd_chunks,
        options.pack_finished,
    );
    println!(
        "BLOCK CONFIG: group_polys {} · stack {} · keccak_rnd 2^{} · drop {} · prepared {} · layout workers {} (ahead {:?}) · rest packed as laid out {} · streamed ops dropped {} · argue {:?} · {}",
        format.group_polys,
        format.zf.whir_stack.get(),
        options.keccak_rnd_rows_log2,
        options.drop_levels,
        if prepared { "on" } else { "off" },
        options.layout_workers,
        options.layout_ahead,
        options.pack_rest_as_laid_out,
        if options.drop_streamed_ops {
            "on"
        } else {
            "off"
        },
        format.argue,
        format.zf.banner(),
    );
    // The options #1010's base proves its epochs under.
    let opts = crate::lfm::proof::block_base_options();
    // The wall clock at the prove's start, to place a box's memory samples.
    println!(
        "W3 PROVE START: unix {:.3}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64())
    );
    let wall = std::time::Instant::now();
    let (proof, stamps) = block_whir::prove_block_whir(&elf, &input, &opts, &format, &options)
        .expect("the block proves");
    let prove_wall = wall.elapsed().as_secs_f64();
    print!("{}", stamps.report());
    // A device that declined any of it would make the readout a host number.
    println!(
        "BLOCK FALLBACKS: commit/encode to the host {} · device commit errors {} · openings on the host {}",
        multilinear::gpu::host_fallbacks(),
        multilinear::gpu::commit_errors(),
        multilinear::gpu::open_host_fallbacks(),
    );
    println!(
        "BLOCK ARGUE DEVICE: GKR tree refusals {} · fused sessions {} · fused declines {}",
        multilinear::gpu::gkr_tree_refusals(),
        multilinear::gpu_fused::fused_sessions(),
        multilinear::gpu_fused::fused_declines(),
    );
    println!(
        "BLOCK PROVE WALL: {prove_wall:.2}s · host peak {:.2} GiB",
        host_peak_gib()
    );
    let t = std::time::Instant::now();
    let ok = verify_block_whir(&proof, &elf, &opts, &format).expect("the verifier runs");
    println!(
        "BLOCK VERIFY: {} in {:.2}s · proof {} tables, {} batched argues, {} groups, {} roots",
        if ok { "ACCEPTED" } else { "REJECTED" },
        t.elapsed().as_secs_f64(),
        proof.proof.tables.len(),
        proof.argues.len(),
        proof.proof.columns.len(),
        proof.proof.roots.len(),
    );
    // The statement's shape, beside the roots: two proves of one block give
    // the same digest when they built the same tables and partition (row
    // order inside the HashMap-ordered tables aside), whatever their proof
    // bytes.
    let shape = {
        use std::hash::{Hash, Hasher};
        let mut h = std::hash::DefaultHasher::new();
        proof.table_num_vars.hash(&mut h);
        proof.groups.hash(&mut h);
        format!("{:?}", proof.table_counts).hash(&mut h);
        format!("{:?}", proof.runtime_page_ranges).hash(&mut h);
        proof.public_output.hash(&mut h);
        proof.num_private_input_pages.hash(&mut h);
        h.finish()
    };
    println!("BLOCK SHAPE DIGEST: {shape:016x}");
    println!(
        "BLOCK CHUNKED: {} (cuts KECCAK 2^{} · ECSM 2^{} · COPY 2^{})",
        block_whir::chunked_census(&proof.table_counts, &proof.table_num_vars),
        options.keccak_rows_log2,
        options.ecsm_rows_log2,
        options.copy_rows_log2,
    );
    // The proof's bytes: two processes prove the same ones under
    // LAMBDA_VM_FIXED_TRACE_HASH=1 and LAMBDA_VM_DETERMINISTIC_GRIND=1.
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("serialize");
    println!(
        "BLOCK PROOF DIGEST: {} (blake3 of the {} proof bytes)",
        &blake3::hash(&bytes).to_hex()[..32],
        bytes.len()
    );
    assert!(ok, "the block proof must verify");
}

/// The control for W1's readout (box only, `--ignored`): #1010's epoch base on
/// the same binary and block — `prove_continuation` at 2^21 — with
/// `LAMBDA_VM_BASE_SPLIT=1`, summing every epoch's and the cross-epoch
/// proof's argue, openings and commit. Prints `BLOCK REFERENCE`.
#[test]
#[ignore = "a real block: box only"]
fn block_whir_epoch_reference_on_a_real_block() {
    let elf_path = std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF");
    let input_path = std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT");
    let elf = std::fs::read(&elf_path).expect("read the ELF");
    let input = std::fs::read(&input_path).expect("read the input");
    assert!(
        multilinear::whir_split::enabled(),
        "the reference reads the split: export LAMBDA_VM_BASE_SPLIT=1"
    );
    let opts = crate::lfm::proof::block_base_options();
    let _ = multilinear::whir_split::drain();
    let wall = std::time::Instant::now();
    let bundle = crate::multilinear_continuation::prove_continuation(&elf, &input, 21, &opts)
        .expect("the epochs prove");
    let base = wall.elapsed().as_secs_f64();
    let (_, prover) = multilinear::whir_split::drain();
    let sum =
        |f: fn(&multilinear::whir_split::ProverSplit) -> f64| prover.iter().map(f).sum::<f64>();
    let (argue, groups, prepared) = (
        sum(|r| r.argue),
        sum(|r| r.open_groups),
        sum(|r| r.open_prepared),
    );
    println!(
        "BLOCK REFERENCE: base {base:.2}s · epochs {} · records {} · Σargue {argue:.2} · Σopen {:.2} (groups {groups:.2} + prepared {prepared:.2}) · Σcommit {:.2} · argue+open {:.2} · host peak {:.2} GiB",
        bundle.num_epochs(),
        prover.len(),
        groups + prepared,
        sum(|r| r.commit),
        argue + groups + prepared,
        host_peak_gib(),
    );
}

/// The process's peak resident set (`VmHWM`), GiB; 0 where `/proc` is absent.
fn host_peak_gib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map(|kb| kb / (1u64 << 20) as f64)
        .unwrap_or(0.0)
}

/// KECCAK_RND's split takes whole rows: each piece is rows `[k·r, (k+1)·r)` of
/// the table, column for column.
#[test]
fn a_split_table_is_its_rows_in_order() {
    use crate::test_utils::{E, F};
    use math::field::element::FieldElement as FE;
    let width = 5usize;
    let height = 32usize;
    let columns: Vec<Vec<FE<F>>> = (0..width)
        .map(|c| {
            (0..height)
                .map(|r| FE::<F>::from((r * 100 + c) as u64))
                .collect()
        })
        .collect();
    let table = stark::trace::TraceTable::<F, E>::from_columns_main(columns.clone(), 1);
    let pieces = block_whir::split_rows(table, 8);
    assert_eq!(pieces.len(), 4);
    for (k, piece) in pieces.iter().enumerate() {
        let got = piece.columns_main();
        for (c, column) in columns.iter().enumerate() {
            assert_eq!(
                got[c],
                column[k * 8..(k + 1) * 8].to_vec(),
                "piece {k} column {c}"
            );
        }
    }
}

/// ★ KECCAK_RND filled from its records ([`Deviations::card_fill`], the
/// `LAMBDA_VM_BLOCK_CARD_FILL` knob) moves no byte: the streamed finish holds
/// every KECCAK_RND table as its keccak operations, the card fills each from
/// them at its phase-A and its phase-B upload (on a host without one, a host
/// reader fills it there; at this test's size the commit's own encode is the
/// host's, under the card's floor), and the proof verifies with the same partition,
/// tables and statement, its bytes equal under the deterministic grind. The
/// inline layout's finish is not streamed and holds none.
#[test]
fn keccak_rnd_filled_from_records_moves_no_byte() {
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    for workers in [0usize, 3] {
        let proved = |card_fill: bool| {
            let mut o = streamed(MaxRowsConfig::small(), 5, 3);
            o.layout_workers = workers;
            let (proof, stamps) = prove_block_whir_with(
                &elf,
                &[],
                &ProofOptions::default_test_options(),
                &format,
                &o,
                &Deviations {
                    card_fill: Some(card_fill),
                    ..Deviations::default()
                },
            )
            .expect("prove");
            assert!(
                verify(&proof, &elf, &format),
                "card fill {card_fill}, {workers} workers"
            );
            (proof, stamps)
        };
        let (host, host_stamps) = proved(false);
        let (card, stamps) = proved(true);
        let what = format!("{workers} workers");
        assert_eq!(host_stamps.layout.records_rest.0, 0, "{what}: knob off");
        assert_eq!(host_stamps.records_fills, (0, 0), "{what}: knob off");
        let tables = card.table_counts.keccak_rnd;
        assert!(tables > 1, "{what}: {tables} KECCAK_RND tables");
        let held = stamps.layout.records_rest.0;
        if workers == 0 {
            assert_eq!(held, 0, "{what}: the inline finish holds none");
        } else {
            assert_eq!(held, tables, "{what}: every KECCAK_RND table held");
            assert_eq!(stamps.layout.records_rest.1 % 208, 0, "{what}: 208 B an op");
            let (on_card, on_host) = stamps.records_fills;
            if cfg!(feature = "cuda") {
                // Each table's group goes up at its commit and at its argument,
                // filled on the card each time. At this size a commit is under
                // the card's floor (a codeword of 2^12, `gpu::COMMIT_THRESHOLD`)
                // and encodes on the host from the host's fill, once a table.
                let held = held as u64;
                assert!(
                    (held..=2 * held).contains(&on_card) && on_host <= held,
                    "{what}: {on_card} on the card, {on_host} on the host, not each on the \
                     card at both its uploads"
                );
            } else {
                assert_eq!(
                    (on_card, on_host),
                    (0, held as u64),
                    "{what}: each filled on the host once"
                );
            }
        }
        assert_eq!(host.groups, card.groups, "{what}: the partition");
        assert_eq!(host.table_num_vars, card.table_num_vars, "{what}");
        assert_eq!(
            format!("{:?}", host.table_counts),
            format!("{:?}", card.table_counts),
            "{what}"
        );
        if crypto::grinding::deterministic() {
            assert_eq!(bytes(&host), bytes(&card), "{what}: the proof");
        }
    }
}

/// ★ A KECCAK_RND table held as its records is never dropped: with live
/// regeneration dropping every table it can, the rest's regenerator builds
/// LT's tables again and no KECCAK_RND table (it builds both without the
/// records), and the proof is the one without regeneration.
#[test]
fn keccak_rnd_held_as_records_is_never_dropped() {
    use crate::block_whir::BlockSpillPolicy;
    use crate::block_whir::regen::RegenMode;

    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    let card = |card_fill: bool| Deviations {
        card_fill: Some(card_fill),
        ..Deviations::default()
    };
    let (plain, _) = regenerated(
        &elf,
        &format,
        RegenMode::Off,
        BlockSpillPolicy::Off,
        card(true),
    )
    .expect("prove");
    let rest_line = |stamps: &crate::block_whir::BlockStamps| {
        let r = stamps.regen.as_ref().expect("live regeneration's readout");
        assert_eq!((r.mismatches, r.failed), (0, 0), "{:?}", r.lines);
        assert_eq!(r.rest_regenerated, r.rest_dropped, "{:?}", r.lines);
        assert!(r.rest_dropped > 0, "no rest table dropped: {:?}", r.lines);
        r.lines
            .iter()
            .find(|l| l.starts_with("BLOCK REGEN rest: "))
            .cloned()
            .expect("the rest's line")
    };
    let (built, stamps) = regenerated(
        &elf,
        &format,
        RegenMode::Always,
        BlockSpillPolicy::Off,
        card(false),
    )
    .expect("prove");
    let line = rest_line(&stamps);
    assert!(
        !line.contains("KECCAK_RND 0 ·"),
        "built, so dropped: {line}"
    );
    assert!(verify(&built, &elf, &format));
    let (held, stamps) = regenerated(
        &elf,
        &format,
        RegenMode::Always,
        BlockSpillPolicy::Off,
        card(true),
    )
    .expect("prove");
    assert!(stamps.layout.records_rest.0 > 0, "no table held as records");
    let line = rest_line(&stamps);
    assert!(
        line.contains("KECCAK_RND 0 · LT "),
        "held, never dropped: {line}"
    );
    assert!(!line.contains("LT 0 built"), "LT dropped and back: {line}");
    assert_eq!(held.groups, plain.groups, "the partition");
    assert!(verify(&held, &elf, &format));
    if crypto::grinding::deterministic() {
        assert_eq!(bytes(&held), bytes(&plain), "the proof");
        assert_eq!(bytes(&built), bytes(&plain), "the proof");
    }
}

/// ★ A wrong KECCAK_RND table held as records is refused by the verifier: with
/// its records starting a row late ([`Deviations::card_fill_records_fault`]),
/// every fill of them (the card's at both uploads, or a host reader's) writes
/// the same wrong table, so the prover's own checks of what it committed pass
/// and it makes a proof, and the unchanged verifier refuses it. (A fill that
/// differs between the phases is the prover's to refuse first: its opening
/// re-hashes what phase A committed.)
#[test]
fn a_wrong_records_table_is_refused_by_the_verifier() {
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let mut o = streamed(MaxRowsConfig::small(), 5, 3);
    o.layout_workers = 3;
    let (proof, stamps) = prove_block_whir_with(
        &elf,
        &[],
        &ProofOptions::default_test_options(),
        &format,
        &o,
        &Deviations {
            card_fill: Some(true),
            card_fill_records_fault: 1,
            ..Deviations::default()
        },
    )
    .unwrap_or_else(|e| panic!("the prover refused ({e:?}): the verifier was never asked"));
    assert!(
        stamps.layout.records_rest.0 > 0,
        "no KECCAK_RND table held as records"
    );
    assert!(
        !verify(&proof, &elf, &format),
        "a proof over a wrong KECCAK_RND table verified"
    );
}

/// Every streamed CPU and MEMW_R chunk held as records
/// ([`Deviations::card_fill_streamed`]), the room rule admitting every one.
fn streamed_records(cpu: bool, memw_r: bool) -> Deviations {
    Deviations {
        card_fill_streamed: Some([cpu, memw_r]),
        records_room: Some(std::sync::Arc::new(|_| true)),
        ..Deviations::default()
    }
}

/// ★ CPU and MEMW_R chunks filled from their records (the
/// `LAMBDA_VM_BLOCK_CARD_FILL` knob's `cpu` and `memw_r`) move no byte: every
/// full streamed chunk of those families is handed out as its logs or rows,
/// the card fills each from them at its phase-A and its phase-B upload (on a
/// host without one, a host reader fills it there), and the proof verifies
/// with the same partition, tables and statement, its bytes equal under the
/// deterministic grind; each family alone, both, and both beside KECCAK_RND's;
/// laid out inline and on workers; for programs whose chunks lie in several
/// windows.
#[test]
fn cpu_and_memw_r_filled_from_records_move_no_byte() {
    let format = many_groups();
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    for name in ["all_instructions_64", "test_keccak_multi"] {
        let elf = asm_elf_bytes(name);
        for workers in [0usize, 3] {
            let proved = |deviations: Deviations| {
                let mut o = streamed(MaxRowsConfig::small(), 5, 3);
                o.layout_workers = workers;
                o.drop_streamed_ops = true;
                let (proof, stamps) = prove_block_whir_with(
                    &elf,
                    &[],
                    &ProofOptions::default_test_options(),
                    &format,
                    &o,
                    &deviations,
                )
                .expect("prove");
                assert!(verify(&proof, &elf, &format), "{name}, {workers} workers");
                (proof, stamps)
            };
            let (host, host_stamps) = proved(Deviations {
                card_fill: Some(false),
                card_fill_streamed: Some([false, false]),
                ..Deviations::default()
            });
            let off = host_stamps.windows.card_fill;
            assert_eq!(
                (off.cpu, off.memw_r, off.records_bytes),
                ([0, 0], [0, 0], 0),
                "knob off"
            );
            assert_eq!(host_stamps.records_fills, (0, 0), "knob off");
            for (cpu, memw_r, keccak_rnd) in [
                (true, false, false),
                (false, true, false),
                (true, true, false),
                (true, true, true),
            ] {
                let what = format!(
                    "{name}, {workers} workers, cpu {cpu} memw_r {memw_r} keccak_rnd {keccak_rnd}"
                );
                // Regeneration is off here, and parked records need it: kept.
                let (card, stamps) = proved(Deviations {
                    card_fill: Some(keccak_rnd),
                    records_keep: Some(true),
                    ..streamed_records(cpu, memw_r)
                });
                let c = stamps.windows.card_fill;
                assert_eq!(c.out_of_step, 0, "{what}: {c:?}");
                assert_eq!(c.room_out, None, "{what}: {c:?}");
                assert_eq!((c.cpu[1], c.memw_r[1]), (0, 0), "{what}: {c:?}");
                assert_eq!(c.cpu[0] > 0, cpu, "{what}: {c:?}");
                assert_eq!(c.memw_r[0] > 0, memw_r, "{what}: {c:?}");
                let held = (c.cpu[0] + c.memw_r[0] + stamps.layout.records_rest.0) as u64;
                let (on_card, on_host) = stamps.records_fills;
                if cfg!(feature = "cuda") {
                    // At its commit and at its argument on the card; at this
                    // size a commit under the card's floor encodes on the host
                    // from the host's fill, once a table.
                    assert!(
                        (held..=2 * held).contains(&on_card) && on_host <= held,
                        "{what}: {on_card} on the card, {on_host} on the host, of {held} held"
                    );
                } else {
                    assert_eq!(
                        (on_card, on_host),
                        (0, held),
                        "{what}: each on the host once"
                    );
                }
                assert_eq!(host.groups, card.groups, "{what}: the partition");
                assert_eq!(host.table_num_vars, card.table_num_vars, "{what}");
                assert_eq!(
                    format!("{:?}", host.table_counts),
                    format!("{:?}", card.table_counts),
                    "{what}"
                );
                if crypto::grinding::deterministic() {
                    assert_eq!(bytes(&host), bytes(&card), "{what}: the proof");
                }
            }
        }
    }
}

/// ★ With records kept (`LAMBDA_VM_BLOCK_RECORDS_DROP=keep`), a streamed chunk
/// held as its records is never dropped and no regenerator waits on it: with
/// live regeneration dropping every table it can (the streamed chunks'
/// regenerator started as early as their fates are known), every dropped
/// chunk comes back with its digest, none is missing, fewer streamed chunks
/// drop than without the records, and the proof is the one without
/// regeneration.
#[test]
fn streamed_records_chunks_kept_are_never_dropped() {
    use crate::block_whir::BlockSpillPolicy;
    use crate::block_whir::regen::RegenMode;

    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    // Records without regeneration: kept (parked records need live
    // regeneration).
    let (plain, _) = regenerated(
        &elf,
        &format,
        RegenMode::Off,
        BlockSpillPolicy::Off,
        Deviations {
            records_keep: Some(true),
            ..streamed_records(true, true)
        },
    )
    .expect("prove");
    let readout = |stamps: &crate::block_whir::BlockStamps| {
        let r = stamps.regen.clone().expect("live regeneration's readout");
        assert_eq!(
            (r.mismatches, r.failed, r.missing),
            (0, 0, 0),
            "{:?}",
            r.lines
        );
        assert_eq!(r.regenerated, r.dropped, "{:?}", r.lines);
        assert!(r.error.is_none(), "{:?}", r.lines);
        r
    };
    let (built, stamps) = regenerated(
        &elf,
        &format,
        RegenMode::Always,
        BlockSpillPolicy::Off,
        Deviations::default(),
    )
    .expect("prove");
    let all = readout(&stamps);
    assert!(verify(&built, &elf, &format));
    let (held, stamps) = regenerated(
        &elf,
        &format,
        RegenMode::Always,
        BlockSpillPolicy::Off,
        Deviations {
            records_keep: Some(true),
            ..streamed_records(true, true)
        },
    )
    .expect("prove");
    let c = stamps.windows.card_fill;
    assert!(c.cpu[0] > 0 && c.memw_r[0] > 0, "{c:?}");
    let some = readout(&stamps);
    let streamed = |r: &crate::block_whir::regen::RegenStamps| r.dropped - r.rest_dropped;
    assert!(
        streamed(&some) < streamed(&all),
        "records chunks dropped: {} of {} without records",
        streamed(&some),
        streamed(&all)
    );
    assert_eq!(
        some.recipes + c.cpu[0] + c.memw_r[0],
        all.recipes,
        "no recipe for a records chunk"
    );
    assert_eq!(held.groups, plain.groups, "the partition");
    assert!(verify(&held, &elf, &format));
    if crypto::grinding::deterministic() {
        assert_eq!(bytes(&held), bytes(&plain), "the proof");
        assert_eq!(bytes(&built), bytes(&plain), "the proof");
    }
}

/// ★ A wrong CPU table held as records is refused by the verifier: with each
/// log's register values changed in the records
/// ([`Deviations::card_fill_streamed_fault`]), every fill of them (the card's
/// at both uploads, or a host reader's) writes the same wrong table, so the
/// prover's own checks of what it committed pass and it makes a proof, and the
/// unchanged verifier refuses it. The fault is shown to bite first: against
/// the same prove without it, the partition is the same and the committed
/// group roots move (a fault the tables do not see would leave them equal, and
/// the refusal below would be vacuous).
#[test]
fn a_wrong_cpu_records_table_is_refused_by_the_verifier() {
    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let mut o = streamed(MaxRowsConfig::small(), 5, 3);
    o.layout_workers = 3;
    o.drop_streamed_ops = true;
    let proved = |fault: u64| {
        prove_block_whir_with(
            &elf,
            &[],
            &ProofOptions::default_test_options(),
            &format,
            &o,
            &Deviations {
                card_fill_streamed_fault: fault,
                // Regeneration is off here, and parked records need it: kept.
                records_keep: Some(true),
                ..streamed_records(true, false)
            },
        )
        .unwrap_or_else(|e| {
            panic!("fault {fault}: the prover refused ({e:?}): the verifier was never asked")
        })
    };
    let (honest, _) = proved(0);
    assert!(verify(&honest, &elf, &format), "the honest proof");
    let (proof, stamps) = proved(1);
    assert!(
        stamps.windows.card_fill.cpu[0] > 0,
        "no CPU chunk held as records"
    );
    assert_eq!(proof.groups, honest.groups, "the partition");
    assert_ne!(
        proof.proof.roots, honest.proof.roots,
        "the fault left every committed table as it was: the refusal below would be vacuous"
    );
    assert!(
        !verify(&proof, &elf, &format),
        "a proof over a wrong CPU table verified"
    );
}

/// ★ Parked records need live regeneration: with a store and regeneration off,
/// or regeneration off and no store, not one streamed chunk is held as its
/// records, even with the room rule admitting every one: the streamed card
/// fill is never set up, so no chunk is even handed to it, and each is built
/// packed on the host, droppable as today. Records kept on purpose
/// (`LAMBDA_VM_BLOCK_RECORDS_DROP=keep`) are still held there: the A/B
/// control. The proofs are one.
#[test]
fn parked_records_need_live_regeneration() {
    use crate::block_whir::BlockSpillPolicy;
    use crate::block_whir::regen::RegenMode;

    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    let none_held = |policy: BlockSpillPolicy| {
        let (proof, stamps) = regenerated(
            &elf,
            &format,
            RegenMode::Off,
            policy,
            streamed_records(true, true),
        )
        .expect("prove");
        let c = stamps.windows.card_fill;
        assert_eq!(
            (c.cpu, c.memw_r, c.records_bytes),
            ([0, 0], [0, 0], 0),
            "{policy:?}: chunks handed to the streamed card fill with no live regeneration: {c:?}"
        );
        assert!(verify(&proof, &elf, &format), "{policy:?}");
        proof
    };
    let stored = none_held(BlockSpillPolicy::Always);
    let held = none_held(BlockSpillPolicy::Off);
    let (kept, stamps) = regenerated(
        &elf,
        &format,
        RegenMode::Off,
        BlockSpillPolicy::Off,
        Deviations {
            records_keep: Some(true),
            ..streamed_records(true, true)
        },
    )
    .expect("prove");
    let c = stamps.windows.card_fill;
    assert!(c.cpu[0] > 0 && c.memw_r[0] > 0, "kept: {c:?}");
    assert!(verify(&kept, &elf, &format));
    assert_eq!(stored.groups, kept.groups, "the partition");
    if crypto::grinding::deterministic() {
        assert_eq!(bytes(&stored), bytes(&kept), "the proof");
        assert_eq!(bytes(&held), bytes(&kept), "the proof");
    }
}

/// ★ A streamed chunk held as its records parks and drops like any chunk (the
/// default, `LAMBDA_VM_BLOCK_RECORDS_DROP=park`): with live regeneration
/// dropping every table it can, as many streamed chunks drop as without the
/// records, those held as records dropped as them; each comes back as packed
/// columns checked by its rebuilt records' digest, none is missing or
/// mismatched, and the proof is the one without regeneration and the one
/// without the records. Without regeneration's drops, the parked records come
/// back as they were, and phase B lets go of every records table after its
/// group's openings; with no spill (no regeneration) none is let go.
#[test]
fn streamed_records_chunks_drop_as_records_and_come_back() {
    use crate::block_whir::BlockSpillPolicy;
    use crate::block_whir::regen::RegenMode;

    let elf = asm_elf_bytes("test_keccak_multi");
    let format = many_groups();
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    let readout = |stamps: &crate::block_whir::BlockStamps| {
        let r = stamps.regen.clone().expect("live regeneration's readout");
        assert_eq!(
            (r.mismatches, r.failed, r.missing),
            (0, 0, 0),
            "{:?}",
            r.lines
        );
        assert_eq!(r.regenerated, r.dropped, "{:?}", r.lines);
        assert!(r.error.is_none(), "{:?}", r.lines);
        r
    };
    let as_records = |r: &crate::block_whir::regen::RegenStamps| -> usize {
        r.lines
            .iter()
            .find_map(|l| {
                let at = l.find(" dropped as their records")?;
                l[..at].rsplit(' ').next()?.parse().ok()
            })
            .expect("the dropped line counts the records")
    };
    let streamed = |r: &crate::block_whir::regen::RegenStamps| r.dropped - r.rest_dropped;
    let released = |stamps: &crate::block_whir::BlockStamps| -> usize {
        stamps.groups.iter().map(|g| g.records_released).sum()
    };
    // Records without regeneration: kept (parked records need live
    // regeneration).
    let (plain, stamps) = regenerated(
        &elf,
        &format,
        RegenMode::Off,
        BlockSpillPolicy::Off,
        Deviations {
            records_keep: Some(true),
            ..streamed_records(true, true)
        },
    )
    .expect("prove");
    assert!(stamps.windows.card_fill.records_bytes > 0);
    assert_eq!(released(&stamps), 0, "no spill: no records let go");
    let (built, stamps) = regenerated(
        &elf,
        &format,
        RegenMode::Always,
        BlockSpillPolicy::Off,
        Deviations::default(),
    )
    .expect("prove");
    let all = readout(&stamps);
    assert_eq!(as_records(&all), 0, "{:?}", all.lines);
    let (parked, stamps) = regenerated(
        &elf,
        &format,
        RegenMode::Always,
        BlockSpillPolicy::Off,
        streamed_records(true, true),
    )
    .expect("prove");
    let c = stamps.windows.card_fill;
    assert!(c.cpu[0] > 0 && c.memw_r[0] > 0, "{c:?}");
    let some = readout(&stamps);
    assert_eq!(
        streamed(&some),
        streamed(&all),
        "every streamed chunk drops"
    );
    assert_eq!(some.recipes, all.recipes, "a recipe for every chunk");
    assert_eq!(
        as_records(&some),
        c.cpu[0] + c.memw_r[0],
        "every records chunk dropped as its records: {:?}",
        some.lines
    );
    // Only the rest's records tables (KECCAK_RND's, where the card fills
    // them) can be let go: every streamed one dropped before phase B.
    assert!(
        released(&stamps) <= stamps.layout.records_rest.1,
        "a streamed records chunk let go in phase B, yet all were dropped"
    );
    assert!(verify(&parked, &elf, &format));
    // Live regeneration that never arms: every records chunk parks and comes
    // back as it was. `auto` arms once the host's bytes pass its target, and
    // the walk's end hands off (and arms) once the peak passes its pressure
    // share: a host short of room (a CI runner's) arms at once. The target is
    // out of reach, so no host's memory decides it.
    let (never, stamps) = regenerated(
        &elf,
        &format,
        RegenMode::Auto,
        BlockSpillPolicy::Off,
        Deviations {
            spill_target: Some(u64::MAX),
            ..streamed_records(true, true)
        },
    )
    .expect("prove");
    let unarmed = readout(&stamps);
    assert!(
        unarmed.dropped == 0
            && unarmed
                .lines
                .iter()
                .any(|l| l.starts_with("BLOCK REGEN dropped: Auto") && l.contains(" never armed ")),
        "{:?}",
        unarmed.lines
    );
    assert!(stamps.windows.card_fill.cpu[0] > 0);
    assert_eq!(
        released(&stamps),
        stamps.windows.card_fill.records_bytes + stamps.layout.records_rest.1,
        "every records table let go after its group's openings"
    );
    // Their lists' pages handed back on a thread of their own: some list let
    // go by its last holder, none refused. (These windows are a few cycles,
    // so a list holds no whole page: the bytes are a real block's to show.)
    let mut pages = multilinear::records::PagesReturned::default();
    for g in &stamps.groups {
        pages += g.records_pages;
    }
    assert!(pages.buffers > 0 && pages.failed == 0, "{pages:?}");
    assert!(verify(&never, &elf, &format));
    assert_eq!(parked.groups, plain.groups, "the partition");
    if crypto::grinding::deterministic() {
        assert_eq!(bytes(&parked), bytes(&plain), "the proof");
        assert_eq!(bytes(&built), bytes(&plain), "the proof");
        assert_eq!(bytes(&never), bytes(&plain), "the proof");
    }
}

/// The chunk heights the row families' records tests run at: chunks of four
/// rows, so that the small programs fill several.
fn rows_max() -> MaxRowsConfig {
    MaxRowsConfig {
        memw_aligned: 1 << 2,
        load: 1 << 2,
        store: 1 << 2,
        memw: 1 << 2,
        shift: 1 << 2,
        ..MaxRowsConfig::small()
    }
}

/// Every streamed chunk of the row families in `rows` held as its rows
/// ([`Deviations::card_fill_rows`]), no CPU or MEMW_R chunk, the room rule
/// admitting every one.
fn row_records(rows: RowFamilies) -> Deviations {
    Deviations {
        card_fill_rows: Some(rows),
        ..streamed_records(false, false)
    }
}

/// The row family alone whose name ([`ROW_NAMES`]) is `name`.
fn row_family(name: &str) -> RowFamilies {
    let mut rows = RowFamilies::NONE;
    let flag = match name {
        "memw_a" => &mut rows.memw_a,
        "load" => &mut rows.load,
        "store" => &mut rows.store,
        "memw" => &mut rows.memw,
        "shift" => &mut rows.shift,
        _ => panic!("no row family {name}"),
    };
    *flag = true;
    rows
}

/// The row families' names, in [`row_counts`]' order.
const ROW_NAMES: [&str; 5] = ["memw_a", "load", "store", "memw", "shift"];

/// Each row family's `[as records, on the host]`, in [`ROW_NAMES`]' order.
fn row_counts(c: &crate::tables::trace_builder::CardFillStamps) -> [[usize; 2]; 5] {
    [c.memw_a, c.load, c.store, c.memw, c.shift]
}

/// ★ The row families' chunks filled from their rows (the
/// `LAMBDA_VM_BLOCK_CARD_FILL` knob's `memw_a`, `load`, `store`, `memw` and
/// `shift`) move no byte: every full streamed chunk of those families is
/// handed out as its rows, the card fills each from them at its phase-A and
/// its phase-B upload (on a host without one, a host reader fills it there),
/// and the proof verifies with the same partition, tables and statement, its
/// bytes equal under the deterministic grind; the row families alone and
/// beside every other family; chunks of four rows, from a program that loads
/// and stores at every width (laid out inline and on workers) and one that
/// stores and shifts. (No small program makes four MEMW ops; a real block's
/// MEMW chunks are the card test's and the gate's.)
#[test]
fn row_families_filled_from_their_rows_move_no_byte() {
    let format = many_groups();
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    // Each program, its layouts, and the families it fills a chunk of.
    for (name, layouts, fills) in [
        (
            "all_loadstore_32",
            &[0usize, 3][..],
            &["memw_a", "load", "store"][..],
        ),
        (
            "test_ecsm_multi",
            &[3][..],
            &["memw_a", "store", "shift"][..],
        ),
    ] {
        let elf = asm_elf_bytes(name);
        for &workers in layouts {
            let proved = |deviations: Deviations| {
                let mut o = streamed(rows_max(), 5, 3);
                o.layout_workers = workers;
                o.drop_streamed_ops = true;
                let (proof, stamps) = prove_block_whir_with(
                    &elf,
                    &[],
                    &ProofOptions::default_test_options(),
                    &format,
                    &o,
                    &deviations,
                )
                .expect("prove");
                assert!(verify(&proof, &elf, &format), "{name}, {workers} workers");
                (proof, stamps)
            };
            let (host, host_stamps) = proved(Deviations {
                card_fill: Some(false),
                card_fill_streamed: Some([false, false]),
                card_fill_rows: Some(RowFamilies::NONE),
                ..Deviations::default()
            });
            let off = host_stamps.windows.card_fill;
            assert!(
                row_counts(&off).iter().all(|f| *f == [0, 0]) && off.records_bytes == 0,
                "knob off: {off:?}"
            );
            assert_eq!(host_stamps.records_fills, (0, 0), "knob off");
            for rest in [false, true] {
                let what = format!("{name}, {workers} workers, the rest {rest}");
                // Regeneration is off here, and parked records need it: kept.
                let (card, stamps) = proved(Deviations {
                    card_fill: Some(rest),
                    card_fill_streamed: Some([rest, rest]),
                    records_keep: Some(true),
                    ..row_records(RowFamilies::ALL)
                });
                let c = stamps.windows.card_fill;
                assert_eq!(c.room_out, None, "{what}: {c:?}");
                for (family, counts) in ROW_NAMES.iter().zip(row_counts(&c)) {
                    assert_eq!(counts[1], 0, "{what}: {family} on the host: {c:?}");
                    assert_eq!(
                        counts[0] > 0,
                        fills.contains(family),
                        "{what}: {family} held: {c:?}"
                    );
                }
                let rows: usize = row_counts(&c).iter().map(|f| f[0]).sum();
                let held = (c.cpu[0] + c.memw_r[0] + rows + stamps.layout.records_rest.0) as u64;
                let (on_card, on_host) = stamps.records_fills;
                if cfg!(feature = "cuda") {
                    // At its commit and at its argument on the card; at this
                    // size a commit under the card's floor encodes on the host
                    // from the host's fill, once a table.
                    assert!(
                        (held..=2 * held).contains(&on_card) && on_host <= held,
                        "{what}: {on_card} on the card, {on_host} on the host, of {held} held"
                    );
                } else {
                    assert_eq!(
                        (on_card, on_host),
                        (0, held),
                        "{what}: each on the host once"
                    );
                }
                assert_eq!(host.groups, card.groups, "{what}: the partition");
                assert_eq!(host.table_num_vars, card.table_num_vars, "{what}");
                assert_eq!(
                    format!("{:?}", host.table_counts),
                    format!("{:?}", card.table_counts),
                    "{what}"
                );
                if crypto::grinding::deterministic() {
                    assert_eq!(bytes(&host), bytes(&card), "{what}: the proof");
                }
            }
        }
    }
}

/// ★ The row families' chunks held as their rows park and drop like any
/// chunk: with live regeneration dropping every table it can, as many
/// streamed chunks drop as without the records, those held as rows dropped
/// as them, each come back as packed columns checked by its rebuilt rows'
/// digest, none missing or mismatched. Kept
/// (`LAMBDA_VM_BLOCK_RECORDS_DROP=keep`), none drops and none has a recipe;
/// with no live regeneration and not kept, none is held. The proofs are the
/// one without regeneration. (MEMW_A, LOAD and STORE chunks: the rebuild is
/// the families' one path.) With no live regeneration and not kept, the
/// streamed card fill is never set up: no chunk is handed to it.
#[test]
fn row_records_chunks_drop_as_records_and_come_back() {
    use crate::block_whir::BlockSpillPolicy;
    use crate::block_whir::regen::{RegenMode, RegenStamps};

    let elf = asm_elf_bytes("all_loadstore_32");
    let format = many_groups();
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    let readout = |stamps: &crate::block_whir::BlockStamps| {
        let r = stamps.regen.clone().expect("live regeneration's readout");
        assert_eq!(
            (r.mismatches, r.failed, r.missing),
            (0, 0, 0),
            "{:?}",
            r.lines
        );
        assert_eq!(r.regenerated, r.dropped, "{:?}", r.lines);
        assert!(r.error.is_none(), "{:?}", r.lines);
        r
    };
    let as_records = |r: &RegenStamps| -> usize {
        r.lines
            .iter()
            .find_map(|l| {
                let at = l.find(" dropped as their records")?;
                l[..at].rsplit(' ').next()?.parse().ok()
            })
            .expect("the dropped line counts the records")
    };
    let streamed = |r: &RegenStamps| r.dropped - r.rest_dropped;
    let held_rows = |c: &crate::tables::trace_builder::CardFillStamps| -> [usize; 5] {
        row_counts(c).map(|f| f[0])
    };
    let proved = |mode, deviations| {
        regenerated_at(
            &elf,
            &format,
            mode,
            BlockSpillPolicy::Off,
            deviations,
            rows_max(),
        )
        .expect("prove")
    };
    let kept = |deviations: Deviations| Deviations {
        records_keep: Some(true),
        ..deviations
    };
    let all_rows = || row_records(RowFamilies::ALL);
    let (plain, _) = proved(RegenMode::Off, kept(all_rows()));
    let (built, stamps) = proved(RegenMode::Always, Deviations::default());
    let all = readout(&stamps);
    assert_eq!(as_records(&all), 0, "{:?}", all.lines);
    // Parked, the default.
    let (parked, stamps) = proved(RegenMode::Always, all_rows());
    let c = stamps.windows.card_fill;
    let rows = held_rows(&c);
    assert!(rows[0] > 0 && rows[1] > 0 && rows[2] > 0, "{c:?}");
    let some = readout(&stamps);
    assert_eq!(
        streamed(&some),
        streamed(&all),
        "every streamed chunk drops"
    );
    assert_eq!(some.recipes, all.recipes, "a recipe for every chunk");
    assert_eq!(
        as_records(&some),
        rows.iter().sum::<usize>(),
        "every records chunk dropped as its records: {:?}",
        some.lines
    );
    assert!(verify(&parked, &elf, &format));
    // Kept: never dropped, no recipe.
    let (held, stamps) = proved(RegenMode::Always, kept(all_rows()));
    let c = stamps.windows.card_fill;
    let rows = held_rows(&c);
    assert!(rows[0] > 0 && rows[1] > 0 && rows[2] > 0, "kept: {c:?}");
    let none = readout(&stamps);
    assert!(
        streamed(&none) < streamed(&all),
        "kept records chunks dropped: {} of {} without records",
        streamed(&none),
        streamed(&all)
    );
    assert_eq!(
        none.recipes + rows.iter().sum::<usize>(),
        all.recipes,
        "no recipe for a kept records chunk"
    );
    assert!(verify(&held, &elf, &format));
    // No live regeneration, not kept: none held.
    let (unheld, stamps) = proved(RegenMode::Off, all_rows());
    // The streamed card fill is never set up, so no chunk is even handed to
    // it: every count stays zero.
    let c = stamps.windows.card_fill;
    assert!(
        row_counts(&c).iter().all(|f| *f == [0, 0]) && c.records_bytes == 0,
        "chunks handed to the streamed card fill with no live regeneration: {c:?}"
    );
    assert!(verify(&unheld, &elf, &format));
    assert_eq!(parked.groups, plain.groups, "the partition");
    if crypto::grinding::deterministic() {
        for (what, proof) in [
            ("parked", &parked),
            ("kept", &held),
            ("not held", &unheld),
            ("no records", &built),
        ] {
            assert_eq!(bytes(proof), bytes(&plain), "{what}: the proof");
        }
    }
}

/// ★ A wrong row family's table held as rows is refused by the verifier: with
/// each row a cycle later (a SHIFT op's input moved by one;
/// [`Deviations::card_fill_streamed_fault`]), every fill of them (the card's
/// at both uploads, or a host reader's) writes the same wrong table, so the
/// prover makes a proof, and the unchanged verifier refuses it. Each family
/// alone that a small program fills.
#[test]
fn a_wrong_row_records_table_is_refused_by_the_verifier() {
    let format = many_groups();
    for (name, family) in [
        ("all_loadstore_32", "memw_a"),
        ("all_loadstore_32", "load"),
        ("all_loadstore_32", "store"),
        ("test_ecsm_multi", "shift"),
    ] {
        let elf = asm_elf_bytes(name);
        let what = format!("{name}, {family}");
        let mut o = streamed(rows_max(), 5, 3);
        o.layout_workers = 3;
        o.drop_streamed_ops = true;
        let (proof, stamps) = prove_block_whir_with(
            &elf,
            &[],
            &ProofOptions::default_test_options(),
            &format,
            &o,
            &Deviations {
                card_fill_streamed_fault: 1,
                // Regeneration is off here, and parked records need it: kept.
                records_keep: Some(true),
                ..row_records(row_family(family))
            },
        )
        .unwrap_or_else(|e| {
            panic!("{what}: the prover refused ({e:?}): the verifier was never asked")
        });
        let c = stamps.windows.card_fill;
        let at = ROW_NAMES
            .iter()
            .position(|n| *n == family)
            .expect("a family");
        assert!(
            row_counts(&c)[at][0] > 0,
            "{what}: no chunk held as rows: {c:?}"
        );
        assert!(
            !verify(&proof, &elf, &format),
            "{what}: a proof over a wrong table verified"
        );
    }
}
