//! The block proved in one multilinear proof (`crate::block_whir`): round
//! trips at one group and at many, KECCAK_RND split into chunks, the hold
//! between the phases moving no byte, and one refusal per check the block adds.

use crate::block_whir::{
    self, BlockFormat, BlockOptions, BlockWhirProof, Deviations, prove_block_whir_with,
    verify_block_whir,
};
use crate::tables::MaxRowsConfig;
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
    }
}

fn options(max_rows: MaxRowsConfig, keccak_rnd_rows_log2: usize) -> BlockOptions {
    BlockOptions {
        max_rows,
        keccak_rnd_rows_log2,
        ecdas_rows_log2: block_whir::BLOCK_ECDAS_ROWS_LOG2,
        keccak_rows_log2: block_whir::BLOCK_KECCAK_ROWS_LOG2,
        ecsm_rows_log2: block_whir::BLOCK_ECSM_ROWS_LOG2,
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
        rest_layout_bytes: Some(1 << 20),
        pack_finished: true,
        gpack: true,
        spill: crate::block_whir::BlockSpillPolicy::Off,
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
        (114, 18, 112, 20)
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
    block_whir::split_keccak_rnd(&mut traces, 16);
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
    block_whir::split_keccak_rnd(&mut traces, 16);
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
    block_whir::split_keccak_rnd(&mut traces, 16);
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

/// The streamed build held narrow: each group packed as it is committed during
/// the collect — every table but the ones the finish built packed, which are
/// held narrow from the start — and the proof verifies.
#[test]
fn a_streamed_block_held_narrow_proves_and_verifies() {
    let elf = asm_elf_bytes("all_instructions_64");
    let format = many_groups();
    let mut o = streamed(MaxRowsConfig::small(), 16, 3);
    o.narrow = stark::multilinear_block::Narrowing::Host;
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
    assert_eq!(
        stamps.groups.iter().map(|g| g.packed_tables).sum::<usize>() + stamps.layout.packed_rest.0,
        stamps.tables
    );
    assert!(stamps.report().contains("BLOCK NARROW: Host"));
    assert!(verify(&proof, &elf, &format));
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
    let spilled = |policy: BlockSpillPolicy| {
        let mut o = streamed(MaxRowsConfig::small(), 5, 3);
        o.spill = policy;
        let (proof, stamps) = prove_block_whir_with(
            &elf,
            &[],
            &ProofOptions::default_test_options(),
            &format,
            &o,
            &Deviations::default(),
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
        None,
        true,
        None,
        None,
        None,
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

#[test]
fn a_streamed_block_with_keccak_chunks_proves_and_verifies() {
    let elf = asm_elf_bytes("test_keccak");
    let format = many_groups();
    let proof = prove(&elf, &format, &streamed(MaxRowsConfig::small(), 3, 4));
    assert!(proof.table_counts.keccak_rnd >= 2);
    assert!(verify(&proof, &elf, &format));
}

/// The streamed build cuts ECDAS at the run's end like the whole-run build:
/// test_ecsm_multi's 64 ECDAS rows in 16-row tables, windows of 2^4 cycles.
#[test]
fn a_streamed_block_with_ecdas_chunks_proves_and_verifies() {
    let elf = asm_elf_bytes("test_ecsm_multi");
    let format = many_groups();
    let mut o = streamed(MaxRowsConfig::small(), 16, 4);
    o.ecdas_rows_log2 = 4;
    let proof = prove(&elf, &format, &o);
    assert_eq!(proof.table_counts.ecdas, 4);
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
        &program, &opts, false, &configs, &counts, None, true, None, None, None,
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
    let bytes = block_whir::block_statement_bytes(proof.statement(), &digest, &frame.config);
    let (a, b) = crate::with_whir_hash!(|H| {
        type T = DefaultTranscript<E, <H as multilinear::whir_hash::WhirHash>::Transcript>;
        let mut host = T::new(&[]);
        block_whir::absorb_block(
            &mut host,
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
        commit: 1,
        blake3: 0,
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

/// ★ KECCAK and ECSM are chunked in the block: any count up to
/// [`block_whir::BLOCK_MAX_KECCAK`] / [`block_whir::BLOCK_MAX_ECSM`] passes,
/// one more is refused before the verifier builds an AIR off it. Every other
/// verifier keeps both to one table ([`crate::TableCounts::validate`]).
///
/// Mutations (laptop): the bound dropped and `KECCAK: MAX + 1` passes; the
/// one-table bound left on KECCAK or ECSM and `2` is refused.
#[test]
fn keccak_and_ecsm_counts_are_bounded_by_the_block() {
    type Field = fn(&mut crate::TableCounts) -> &mut usize;
    let tables: [(&str, usize, Field); 2] = [
        ("KECCAK", block_whir::BLOCK_MAX_KECCAK, |c| &mut c.keccak),
        ("ECSM", block_whir::BLOCK_MAX_ECSM, |c| &mut c.ecsm),
    ];
    for (name, max, field) in tables {
        let mut single = chunked_counts();
        (single.keccak, single.keccak_rnd, single.ecsm, single.ecdas) = (1, 1, 1, 1);
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
/// theirs: a KECCAK table over [`block_whir::BLOCK_KECCAK_MAX_VARS`], a
/// KECCAK_RND table over [`block_whir::BLOCK_KECCAK_RND_MAX_VARS`], an ECSM
/// table over [`block_whir::BLOCK_ECSM_MAX_VARS`] and an ECDAS table over
/// [`block_whir::BLOCK_ECDAS_MAX_VARS`] are refused; every other table far over
/// every cap passes. The controls next to the ranges (COMMIT before KECCAK,
/// HINT after ECDAS) are what an off-by-one in the ranges would cap instead;
/// each table at its own cap passing pins which cap each range carries.
#[test]
fn the_block_caps_the_chunked_tables_heights() {
    let counts = chunked_counts();
    let n = counts.total().expect("fits") + crate::FIXED_TABLE_COUNT;
    assert_eq!(n, 19);
    let check = |vars: &[u8]| block_whir::check_chunked_heights(&counts, vars);
    let honest = vec![10u8; n];
    assert!(check(&honest).is_ok());
    let (keccak_cap, keccak_rnd_cap, ecsm_cap, ecdas_cap) = (
        block_whir::BLOCK_KECCAK_MAX_VARS as u8,
        block_whir::BLOCK_KECCAK_RND_MAX_VARS as u8,
        block_whir::BLOCK_ECSM_MAX_VARS as u8,
        block_whir::BLOCK_ECDAS_MAX_VARS as u8,
    );
    let mut at_caps = honest.clone();
    at_caps[6..8].fill(keccak_cap);
    at_caps[8..10].fill(keccak_rnd_cap);
    at_caps[10..12].fill(ecsm_cap);
    at_caps[12..15].fill(ecdas_cap);
    assert!(check(&at_caps).is_ok(), "a table at its cap passes");
    for (what, idx, vars) in [
        ("ECDAS[2] over its cap", 14, ecdas_cap + 1),
        ("ECDAS[0] over its cap", 12, ecdas_cap + 1),
        ("ECSM[1] over its cap", 11, ecsm_cap + 1),
        ("ECSM[0] over its cap", 10, ecsm_cap + 1),
        ("KECCAK_RND[1] over its cap", 9, keccak_rnd_cap + 1),
        ("KECCAK_RND[0] over its cap", 8, keccak_rnd_cap + 1),
        ("KECCAK[1] over its cap", 7, keccak_cap + 1),
        ("KECCAK[0] over its cap", 6, keccak_cap + 1),
    ] {
        let mut stated = honest.clone();
        stated[idx] = vars;
        let refused = check(&stated);
        assert!(refused.is_err(), "the block must refuse {what}");
        println!("BLOCK WHIR refuses {what}: {:?}", refused.err());
    }
    for (what, idx) in [
        ("COMMIT", 5),
        ("HINT", 15),
        ("CPU[0]", 16),
        ("MEMW_R[0]", 18),
    ] {
        let mut stated = honest.clone();
        stated[idx] = 22;
        assert!(
            check(&stated).is_ok(),
            "{what} is not a chunked table: its height is not capped"
        );
    }
}

/// ★ Every chunked table's cap is the tallest power of two at which its
/// columns fill at most one polynomial of the format's 2^27 stack: KECCAK
/// 511 × 2^18, KECCAK_RND 1,480 × 2^16, ECSM 667 × 2^17, ECDAS 521 × 2^17. One
/// doubling past a cap would stack the table into two polynomials.
///
/// Mutation (laptop): the KECCAK or the ECSM cap lifted, and this test fails
/// (the caps test reads the constants, so it cannot see a lifted cap).
#[test]
fn every_chunked_cap_is_the_tallest_table_one_stacked_polynomial_holds() {
    use crate::tables::{ecdas, ecsm, keccak, keccak_rnd};
    let stack = ZfFormat::DEFAULT.whir_stack.get();
    let widths = [
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
        None,
        true,
        None,
        None,
        None,
    );
    let names: Vec<String> = airs
        .air_refs()
        .iter()
        .map(|a| a.name().to_string())
        .collect();
    let ranges = block_whir::chunked_table_ranges(&counts);
    assert_eq!(
        ranges.each_ref().map(|(table, ..)| *table),
        ["KECCAK", "KECCAK_RND", "ECSM", "ECDAS"],
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
/// (calls k = 1, 5, 0xABCDEF) over four tables, the 0xABCDEF call through
/// three of them; their bus sums add up to the whole table's.
#[test]
fn ecdas_split_into_chunks_proves_and_verifies() {
    let elf = asm_elf_bytes("test_ecsm_multi");
    let mut o = options(MaxRowsConfig::default(), 16);
    o.ecdas_rows_log2 = 4;
    let proof = prove(&elf, &many_groups(), &o);
    assert_eq!(proof.table_counts.ecdas, 4, "64 rows in 16-row tables");
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
    block_whir::split_keccak_rnd(&mut traces, 16);
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
        block_whir::split_ecdas(traces, 4);
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

    fn split(self, traces: &mut crate::tables::trace_builder::Traces, rows_log2: usize) {
        match self {
            CallTable::Keccak => block_whir::split_keccak(traces, rows_log2),
            CallTable::Ecsm => block_whir::split_ecsm(traces, rows_log2),
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
/// of four rows each, the last row padding) at one row a table (four tables:
/// one call each and the padding row) and at two rows a table (two); every
/// proof verifies, the bus sums over the tables adding up to the one table's.
/// The production cuts leave each guest one table.
#[test]
fn keccak_and_ecsm_split_into_chunks_prove_and_verify() {
    let format = many_groups();
    for table in CallTable::ALL {
        let (name, elf) = (table.name(), asm_elf_bytes(table.guest()));
        for (rows_log2, tables) in [(0, 4), (1, 2)] {
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
            block_whir::split_keccak(traces, block_whir::BLOCK_KECCAK_ROWS_LOG2);
            block_whir::split_ecsm(traces, block_whir::BLOCK_ECSM_ROWS_LOG2);
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
    let format = BlockFormat {
        prepared,
        argue,
        ..BlockFormat::production()
    };
    let mut options = BlockOptions::production();
    // `BLOCK_WHIR_LAYOUT_WORKERS=n` (production 3; 0 is the inline layout) and
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
    // `BLOCK_WHIR_REST_LAYOUT=all|<MiB>` (production 2048),
    // `BLOCK_WHIR_KR_FINISH_CHUNKS=0|1` and `BLOCK_WHIR_PACK_FINISHED=0|1`
    // (production 1): the rest's layout in waves, KECCAK_RND built as its
    // tables, the finish's tables packed as they are built.
    crate::block_whir::rest_layout_from_env(&mut options);
    println!(
        "BLOCK REST LAYOUT CONFIG: waves of {} · KECCAK_RND built as its tables {} · finish packed {}",
        options
            .rest_layout_bytes
            .map_or("all at once".to_string(), |b| format!("{} MiB", b >> 20)),
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
        "BLOCK CHUNKED: {} (cuts KECCAK 2^{} · ECSM 2^{})",
        block_whir::chunked_census(&proof.table_counts, &proof.table_num_vars),
        options.keccak_rows_log2,
        options.ecsm_rows_log2,
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
