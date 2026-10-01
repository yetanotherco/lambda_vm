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
use multilinear::whir_chain::StackVars;
use stark::proof::options::ProofOptions;

/// The production format; one group holds a small program whole.
fn one_group() -> BlockFormat {
    BlockFormat {
        zf: ZfFormat::DEFAULT,
        group_polys: block_whir::BLOCK_GROUP_POLYS,
        max_groups: block_whir::BLOCK_MAX_GROUPS,
        prepared: true,
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
    }
}

fn options(max_rows: MaxRowsConfig, keccak_rnd_rows_log2: usize) -> BlockOptions {
    BlockOptions {
        max_rows,
        keccak_rnd_rows_log2,
        drop_levels: 3,
        window_log2: None,
        stream_keccak_rnd: false,
        stream_memw_lt: false,
        layout_workers: 0,
        pack_rest_as_laid_out: false,
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
        o.drop_levels = drop_levels;
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
    let (three, all) = (proved(3), proved(64));
    if crypto::grinding::deterministic() {
        assert_eq!(whole, three);
        assert_eq!(whole, all);
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

/// The KECCAK_RND count is bounded before the verifier builds an AIR off it.
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
    counts.keccak = 2;
    assert!(block_whir::validate_block_counts(&counts).is_err());
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
    let format = BlockFormat {
        prepared,
        ..BlockFormat::production()
    };
    let mut options = BlockOptions::production();
    // `BLOCK_WHIR_LAYOUT_WORKERS=n` (production 3; 0 is the inline layout) and
    // `BLOCK_WHIR_PACK_REST=1`, as the tree's harness takes them.
    if let Some(n) = std::env::var("BLOCK_WHIR_LAYOUT_WORKERS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
    {
        options.layout_workers = n;
    }
    options.pack_rest_as_laid_out =
        std::env::var("BLOCK_WHIR_PACK_REST").is_ok_and(|v| v.trim() == "1");
    println!(
        "BLOCK CONFIG: group_polys {} · stack {} · keccak_rnd 2^{} · drop {} · prepared {} · layout workers {} · rest packed as laid out {} · {}",
        format.group_polys,
        format.zf.whir_stack.get(),
        options.keccak_rnd_rows_log2,
        options.drop_levels,
        if prepared { "on" } else { "off" },
        options.layout_workers,
        options.pack_rest_as_laid_out,
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
        "BLOCK PROVE WALL: {prove_wall:.2}s · host peak {:.2} GiB",
        host_peak_gib()
    );
    let t = std::time::Instant::now();
    let ok = verify_block_whir(&proof, &elf, &opts, &format).expect("the verifier runs");
    println!(
        "BLOCK VERIFY: {} in {:.2}s · proof {} tables, {} groups, {} roots",
        if ok { "ACCEPTED" } else { "REJECTED" },
        t.elapsed().as_secs_f64(),
        proof.proof.tables.len(),
        proof.proof.columns.len(),
        proof.proof.roots.len(),
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
