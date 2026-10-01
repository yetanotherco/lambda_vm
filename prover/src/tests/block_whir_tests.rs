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
    }
}

/// A stack of 2^10 and two polynomials a group: a small program spans several
/// groups, which is the block's shape at a size a test can prove.
fn many_groups() -> BlockFormat {
    let mut zf = ZfFormat::DEFAULT;
    zf.whir_stack = StackVars::new(10).expect("a valid stack");
    BlockFormat { zf, group_polys: 2 }
}

fn options(max_rows: MaxRowsConfig, keccak_rnd_rows_log2: usize) -> BlockOptions {
    BlockOptions {
        max_rows,
        keccak_rnd_rows_log2,
        drop_levels: 3,
        window_log2: None,
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

/// One table's claimed column value moved: its group's opening refuses it./// One table's claimed column value moved: its group's opening refuses it.
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
    let format = BlockFormat::production();
    let options = BlockOptions::production();
    println!(
        "BLOCK CONFIG: group_polys {} · stack {} · keccak_rnd 2^{} · drop {} · {}",
        format.group_polys,
        format.zf.whir_stack.get(),
        options.keccak_rnd_rows_log2,
        options.drop_levels,
        format.zf.banner(),
    );
    // The options #1010's base proves its epochs under.
    let opts = crate::lfm::proof::block_base_options();
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
