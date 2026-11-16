//! The word memory format on blocks. Its proof field
//! (`BlockWhirProof::word_memory`): a split and a fused proof are the bytes
//! before the field then its seven zero counts; any nonzero count on a split
//! or fused proof is refused (and verifies with the refusal taken out, so the
//! rule is what refuses it); a split or fused proof does not verify under the
//! word format. Its proofs: chip-free runs prove and verify under it (and not
//! under the split or fused format), their word tables the ones the walk
//! counts; a word row moved later is refused by the bus balance alone; a
//! hand-built proof stating a chip's table is refused by the chip rule alone.
//! Runs that call the COPY chip (memcpy and memmove, memset, the commit
//! syscall) prove and verify under it, COPY's tables COPY_W's and MEMW_W
//! taking their word accesses; a COPY_W row moved later is refused by the
//! bus balance alone.
//! The proving tests run where a block can be proved (the box's lib run and
//! CI). The word format's statement checks on hand-built statements are
//! `tables::word_memory_tests`', its walk's on the builders' traces
//! `tables::trace_builder::word_walk_tests`'.

use crate::block_whir::{
    self, BlockFormat, BlockWhirProof, CHIP_COUNTS_UNCHECKED, Deviations, WORD_COUNTS_UNCHECKED,
    prove_block_whir_with, verify_block_whir,
};
use crate::tables::MaxRowsConfig;
use crate::tables::fused_memory::MemoryFormat;
use crate::test_utils::asm_elf_bytes;
use stark::multilinear_table::batched::VerifierChecks;
use stark::proof::options::ProofOptions;

fn format(memory: MemoryFormat) -> BlockFormat {
    BlockFormat::production().with_memory(memory)
}

/// `elf` proved as a streamed block under `format`.
fn prove(elf: &[u8], format: &BlockFormat) -> BlockWhirProof {
    let options = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
    prove_block_whir_with(
        elf,
        &[],
        &ProofOptions::default_test_options(),
        format,
        &options,
        &Deviations::default(),
    )
    .expect("prove")
    .0
}

fn verifies(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> bool {
    verify_block_whir(proof, elf, &ProofOptions::default_test_options(), format).unwrap_or(false)
}

/// [`BlockWhirProof`] as it was before the word memory format: every field
/// but the last, in its order.
#[derive(rkyv::Archive, rkyv::Serialize)]
struct BeforeWords {
    proof: stark::multilinear_table::MultiProof<crate::test_utils::F, crate::test_utils::E>,
    table_num_vars: Vec<u8>,
    runtime_page_ranges: Vec<crate::RuntimePageRange>,
    table_counts: crate::TableCounts,
    public_output: Vec<u8>,
    num_private_input_pages: usize,
    groups: Vec<Vec<u32>>,
    prepared:
        Vec<multilinear::stacked_eval::StackedProof<crate::test_utils::F, crate::test_utils::E>>,
    argues: Vec<stark::multilinear_table::BatchedArgue<crate::test_utils::E>>,
    commit_ood: Vec<math::field::element::FieldElement<crate::test_utils::E>>,
    fused_memory: [u64; 2],
}

impl BeforeWords {
    fn of(proof: &BlockWhirProof) -> Self {
        let p = proof.clone();
        Self {
            proof: p.proof,
            table_num_vars: p.table_num_vars,
            runtime_page_ranges: p.runtime_page_ranges,
            table_counts: p.table_counts,
            public_output: p.public_output,
            num_private_input_pages: p.num_private_input_pages,
            groups: p.groups,
            prepared: p.prepared,
            argues: p.argues,
            commit_ood: p.commit_ood,
            fused_memory: p.fused_memory,
        }
    }
}

/// ★ A split and a fused proof's bytes are the format's before the word
/// counts, then the counts: the struct before them archives to exactly the
/// proof's bytes but its last 56, which are zero. So the bytes a binary
/// before the word format proves are this prefix, and a byte moved in any
/// earlier field moves the prefix (the check's mutant: one flipped byte of a
/// group index).
#[test]
fn a_split_and_a_fused_proof_are_the_bytes_before_the_word_counts_then_zeros() {
    let elf = asm_elf_bytes("fused_mem_cases");
    for memory in [MemoryFormat::Split, MemoryFormat::Fused] {
        let proof = prove(&elf, &format(memory));
        assert_eq!(proof.word_memory, [0; 7], "{memory:?}");
        let now = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("serialize");
        let before =
            rkyv::to_bytes::<rkyv::rancor::Error>(&BeforeWords::of(&proof)).expect("serialize");
        assert_eq!(now.len(), before.len() + 56, "{memory:?}: 56 bytes after");
        assert_eq!(
            &now[..before.len()],
            &before[..],
            "{memory:?}: the bytes before"
        );
        assert_eq!(
            &now[before.len()..],
            &[0u8; 56],
            "{memory:?}: the counts, zero"
        );
        // The comparison's mutant: a proof with one byte moved before the
        // counts is not the struct before them, the counts still zero.
        let mut moved = proof.clone();
        moved.groups[0][0] ^= 1;
        let moved_now = rkyv::to_bytes::<rkyv::rancor::Error>(&moved).expect("serialize");
        assert_eq!(
            moved_now.len(),
            now.len(),
            "{memory:?}: the moved proof's length"
        );
        assert_ne!(
            &moved_now[..before.len()],
            &before[..],
            "{memory:?}: a moved byte parts the prefix"
        );
        assert_eq!(&moved_now[before.len()..], &[0u8; 56], "{memory:?}");
    }
}

/// ★ A split or fused proof stating word tables is refused: the counts are
/// absorbed nowhere under those formats, so no value may ride in a proof. One
/// family at 1, and one at `2^32 − 1`, on an honest proof are each refused;
/// with the refusal taken out (a control mutant) each verifies, the counts
/// unbound.
#[test]
fn a_split_or_fused_proof_stating_word_tables_is_refused() {
    let elf = asm_elf_bytes("fused_mem_cases");
    for memory in [MemoryFormat::Split, MemoryFormat::Fused] {
        let f = format(memory);
        let honest = prove(&elf, &f);
        assert!(verifies(&honest, &elf, &f), "{memory:?}");
        for stated in [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, u64::from(u32::MAX)],
        ] {
            let mut forged = honest.clone();
            forged.word_memory = stated;
            assert!(
                !verifies(&forged, &elf, &f),
                "{memory:?} {stated:?} refused"
            );
            WORD_COUNTS_UNCHECKED.with(|c| c.set(true));
            let control = verifies(&forged, &elf, &f);
            WORD_COUNTS_UNCHECKED.with(|c| c.set(false));
            assert!(
                control,
                "{memory:?} {stated:?}: without the refusal it verifies"
            );
        }
    }
}

/// ★ A split or a fused proof does not verify under the word format: the
/// format is the verifier's constant. Each is refused by the word format's
/// own rule on what it retires: the split proof states LOAD tables, the fused
/// one fused tables (both of `fused_mem_cases`).
#[test]
fn a_split_or_fused_proof_is_refused_under_the_word_format() {
    let elf = asm_elf_bytes("fused_mem_cases");
    for (memory, why) in [
        (MemoryFormat::Split, "states no LOAD table"),
        (MemoryFormat::Fused, "states no fused table"),
    ] {
        let proof = prove(&elf, &format(memory));
        assert!(verifies(&proof, &elf, &format(memory)));
        let refused = verify_block_whir(
            &proof,
            &elf,
            &ProofOptions::default_test_options(),
            &format(MemoryFormat::Words),
        );
        assert!(
            matches!(&refused, Err(crate::Error::InvalidTableCounts(w)) if w.contains(why)),
            "{memory:?}: {refused:?}"
        );
    }
}

/// The chip-free programs a word block is proved of: every width, sign and
/// alignment of the CPU's loads and stores, every instruction, the misaligned
/// accesses one by one.
const WORD_PROGRAMS: [&str; 11] = [
    "fused_mem_cases",
    "all_loadstore_32",
    "all_instructions_64",
    "misalign_ld",
    "misalign_lh",
    "misalign_lhu",
    "misalign_lw",
    "misalign_lwu",
    "misalign_sd",
    "misalign_sh",
    "misalign_sw",
];

/// `elf`'s run built by the windowed builder under `format` (windows of 1000
/// cycles, the small chunks), REGISTER's files the verifier's: what
/// [`prove_built`] proves.
fn word_build(elf: &[u8], format: &BlockFormat) -> crate::tables::trace_builder::Traces {
    use crate::tables::trace_builder::WindowedTraceBuilder;
    use crate::tables::trace_builder::in_row_forgery_tests::with_verifier_files;
    let program = executor::elf::Elf::load(elf).expect("the ELF loads");
    let logs = executor::vm::execution::Executor::new(&program, Vec::new())
        .expect("the executor")
        .run()
        .expect("the run")
        .logs;
    let mut builder = WindowedTraceBuilder::new(&program, &[], &MaxRowsConfig::small())
        .and_then(|b| b.register_format(format.registers))
        .and_then(|b| b.memw_aligned_order(format.memw_aligned))
        .map(|b| b.keccak_rnd_form(format.keccak_rnd))
        .and_then(|b| b.memory_format(format.memory))
        .expect("the builder");
    let window = 1000;
    let body = logs.len() - 1;
    let cut = body - body % window;
    let mut chunks = Vec::new();
    for w in logs[..cut].chunks(window) {
        chunks.extend(builder.push(w).expect("a window"));
    }
    let mut traces = builder.finish(&logs[cut..]).expect("the last window");
    traces
        .insert_streamed(chunks)
        .expect("every chunk has a placeholder");
    with_verifier_files(&program, &mut traces);
    traces
}

/// `traces` proved as a block under `format`: the prover takes any trace.
fn prove_built(
    elf: &[u8],
    traces: &mut crate::tables::trace_builder::Traces,
    format: &BlockFormat,
) -> Result<BlockWhirProof, crate::Error> {
    let program = executor::elf::Elf::load(elf).expect("the ELF loads");
    let options = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
    let mut stamps = block_whir::BlockStamps::default();
    block_whir::prove_traces(
        &program,
        elf,
        traces,
        &ProofOptions::default_test_options(),
        format,
        &options,
        &Deviations::default(),
        true,
        &|_, _| {},
        &mut stamps,
    )
}

/// The verifier's verdict on `proof` under `format` and `checks`.
fn verdict(
    proof: &BlockWhirProof,
    elf: &[u8],
    format: &BlockFormat,
    checks: VerifierChecks,
) -> Result<Result<(), block_whir::BlockRefusal>, String> {
    block_whir::verify_block_whir_traced(
        proof,
        elf,
        &ProofOptions::default_test_options(),
        format,
        false,
        checks,
    )
    .map_err(|e| format!("{e:?}"))
}

/// ★ Word blocks prove and verify: every chip-free program, proved as a
/// streamed block under the word memory format (the production register
/// format and MEMW_A order; the prove's own walk, record mode and walkers),
/// verifies under it and under neither the split nor the fused format. No
/// retired table is stated and no fused or MEMW_W table; the word tables are
/// the ones the walk makes of the run (each family's count the windowed
/// build's), so every load and store is one row of one of them; on
/// `fused_mem_cases` every CPU family holds a table.
#[test]
fn a_word_block_proves_and_verifies() {
    let words = format(MemoryFormat::Words);
    for name in WORD_PROGRAMS {
        let elf = asm_elf_bytes(name);
        let proof = prove(&elf, &words);
        assert!(verifies(&proof, &elf, &words), "{name}");
        // Under the split and fused formats the word counts are absorbed
        // nowhere: refused by that rule (every program here holds a word
        // table but `all_instructions_64`, which loads and stores nothing).
        for other in [MemoryFormat::Split, MemoryFormat::Fused] {
            let refused = verify_block_whir(
                &proof,
                &elf,
                &ProofOptions::default_test_options(),
                &format(other),
            );
            let by_rule = matches!(
                &refused,
                Err(crate::Error::InvalidTableCounts(w)) if w.contains("states no word table")
            );
            assert!(
                by_rule || (proof.word_memory == [0; 7] && !matches!(refused, Ok(true))),
                "{name}: under {other:?}: {refused:?}"
            );
        }
        let c = &proof.table_counts;
        assert_eq!(
            (c.load, c.store, c.memw, c.memw_aligned),
            (0, 0, 0, 0),
            "{name}: no retired table"
        );
        assert_eq!(proof.fused_memory, [0, 0], "{name}");
        assert_eq!(proof.word_memory[6], 0, "{name}: no MEMW_W");
        let built = word_build(&elf, &words).word_counts().map(|n| n as u64);
        assert_eq!(proof.word_memory, built, "{name}: the walk's word tables");
        if name == "fused_mem_cases" {
            assert!(
                proof.word_memory[..6].iter().all(|&n| n > 0),
                "{name}: {:?}",
                proof.word_memory
            );
        }
        println!("WORD BLOCK {name}: word tables {:?}", proof.word_memory);
    }
}

/// ★ A word block whose first row of each CPU word family is moved four
/// cycles later (its dispatch then names no CPU row, its tokens no word's
/// timestamp), proved from the built traces, is refused by the bus balance,
/// and verifies with that one check off: nothing but the balance reads the
/// move. The honest build proves and verifies.
#[test]
fn a_word_block_with_a_moved_word_row_is_refused_by_the_balance_alone() {
    use crate::tables::{mem_load2w, mem_load8, mem_loadw, mem_store2w, mem_store8, mem_storew};
    let words = format(MemoryFormat::Words);
    let elf = asm_elf_bytes("fused_mem_cases");
    let mut honest = word_build(&elf, &words);
    let proof = prove_built(&elf, &mut honest, &words).expect("the honest build proves");
    assert!(verifies(&proof, &elf, &words), "the honest build verifies");
    let timestamps = [
        mem_load8::cols::TIMESTAMP,
        mem_store8::cols::TIMESTAMP,
        mem_loadw::cols::TIMESTAMP,
        mem_storew::cols::TIMESTAMP,
        mem_load2w::cols::TIMESTAMP,
        mem_store2w::cols::TIMESTAMP,
    ];
    let mut wrong = Vec::new();
    for (family, &ts) in timestamps.iter().enumerate() {
        let mut forged = word_build(&elf, &words);
        let table = &mut forged.word_tables[family][0];
        let moved = table.get_main(0, ts).canonical_u64() + 16;
        table
            .main_table
            .set(0, ts, crate::tables::types::FE::from(moved));
        let proof = prove_built(&elf, &mut forged, &words).expect("the forged build proves");
        let all = verdict(&proof, &elf, &words, VerifierChecks::ALL);
        let off = verdict(
            &proof,
            &elf,
            &words,
            VerifierChecks {
                balance: false,
                ..VerifierChecks::ALL
            },
        );
        let named = matches!(
            &all,
            Ok(Err(block_whir::BlockRefusal::Proof(e))) if *e == multilinear::Error::BusImbalance
        );
        if !(named && matches!(&off, Ok(Ok(())))) {
            wrong.push(format!(
                "{}: every check {all:?} · without the balance {off:?}",
                block_whir::WORD_TABLES[family]
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// ★ A hand-built word block stating the table of a chip the format takes no
/// call of yet is refused by the chip rule alone: an honest chip-free build
/// with one HINT table of padding rows added (every interaction at
/// multiplicity zero, so every bus still balances) proves, and the verifier
/// refuses it for stating HINT; with the chip rule taken out (a control
/// mutant) it verifies. The verifier does not leave such a chip's table under
/// the word format to the bus.
#[test]
fn a_word_block_stating_a_chip_table_is_refused_by_the_chip_rule_alone() {
    let words = format(MemoryFormat::Words);
    let elf = asm_elf_bytes("fused_mem_cases");
    let mut forged = word_build(&elf, &words);
    forged.hints = vec![crate::tables::hint::generate_hint_trace(&[])];
    let proof = prove_built(&elf, &mut forged, &words).expect("the forged build proves");
    assert_eq!(
        proof.table_counts.hint, 1,
        "the proof states one HINT table"
    );
    let refused = verify_block_whir(&proof, &elf, &ProofOptions::default_test_options(), &words);
    assert!(
        matches!(&refused, Err(crate::Error::InvalidTableCounts(why)) if why.contains("no HINT table")),
        "refused by the chip rule: {refused:?}"
    );
    CHIP_COUNTS_UNCHECKED.with(|c| c.set(true));
    let control = verifies(&proof, &elf, &words);
    CHIP_COUNTS_UNCHECKED.with(|c| c.set(false));
    assert!(control, "without the chip rule the forged block verifies");
}

/// The runs that call the COPY chip: asm programs of every row shape,
/// overlapping copies and commits split across calls; Rust guests' memcpy,
/// memmove, memset and commit cases.
const COPY_ASM: [&str; 3] = [
    "test_copy_ecalls",
    "test_copy_overlaps",
    "test_commit_split",
];
const COPY_RUST: [&str; 4] = [
    "dma_memcpy_cases",
    "dma_memmove_cases",
    "dma_memset_cases",
    "commit",
];

fn rust_elf_bytes(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../executor/program_artifacts/rust")
        .join(format!("{name}.elf"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// ★ Word blocks that call the COPY chip prove and verify: each run, proved
/// as a streamed block under the word memory format, verifies under it and
/// under neither the split nor the fused format; it states COPY_W tables in
/// COPY's place and MEMW_W tables (the walk's count), no retired or fused
/// table; and its public output is the run's.
#[test]
fn a_word_block_calling_the_copy_chip_proves_and_verifies() {
    let words = format(MemoryFormat::Words);
    let runs = COPY_ASM
        .iter()
        .map(|&name| (name, asm_elf_bytes(name)))
        .chain(COPY_RUST.iter().map(|&name| (name, rust_elf_bytes(name))));
    for (name, elf) in runs {
        let proof = prove(&elf, &words);
        assert!(verifies(&proof, &elf, &words), "{name}");
        for other in [MemoryFormat::Split, MemoryFormat::Fused] {
            assert!(
                !verifies(&proof, &elf, &format(other)),
                "{name}: verifies under {other:?}"
            );
        }
        let c = &proof.table_counts;
        assert_eq!(
            (c.load, c.store, c.memw, c.memw_aligned),
            (0, 0, 0, 0),
            "{name}: no retired table"
        );
        assert_eq!(proof.fused_memory, [0, 0], "{name}");
        assert!(
            c.copy > 0 && proof.word_memory[6] > 0,
            "{name}: COPY_W and MEMW_W"
        );
        let build = word_build(&elf, &words);
        assert_eq!(
            proof.word_memory,
            build.word_counts().map(|n| n as u64),
            "{name}: the walk's word tables"
        );
        assert_eq!(
            proof.public_output, build.public_output_bytes,
            "{name}: the run's public output"
        );
        println!(
            "WORD COPY BLOCK {name}: COPY_W tables {}, word tables {:?}, output {} bytes",
            c.copy,
            proof.word_memory,
            proof.public_output.len()
        );
    }
}

/// ★ A word block whose first COPY_W row is moved four cycles later (its
/// call, its chain and its word messages then name no one's), proved from the
/// built traces, is refused by the bus balance, and verifies with that one
/// check off: no constraint reads the call's timestamp. The honest build
/// proves and verifies.
#[test]
fn a_word_block_with_a_moved_copy_row_is_refused_by_the_balance_alone() {
    let words = format(MemoryFormat::Words);
    let elf = asm_elf_bytes("test_copy_ecalls");
    let mut honest = word_build(&elf, &words);
    let proof = prove_built(&elf, &mut honest, &words).expect("the honest build proves");
    assert!(verifies(&proof, &elf, &words), "the honest build verifies");
    let mut forged = word_build(&elf, &words);
    let ts = crate::tables::copy_w::cols::TIMESTAMP;
    let table = &mut forged.copies[0];
    let moved = table.get_main(0, ts).canonical_u64() + 16;
    table
        .main_table
        .set(0, ts, crate::tables::types::FE::from(moved));
    let proof = prove_built(&elf, &mut forged, &words).expect("the forged build proves");
    let all = verdict(&proof, &elf, &words, VerifierChecks::ALL);
    let off = verdict(
        &proof,
        &elf,
        &words,
        VerifierChecks {
            balance: false,
            ..VerifierChecks::ALL
        },
    );
    assert!(
        matches!(
            &all,
            Ok(Err(block_whir::BlockRefusal::Proof(e))) if *e == multilinear::Error::BusImbalance
        ),
        "every check: {all:?}"
    );
    assert!(matches!(&off, Ok(Ok(()))), "without the balance: {off:?}");
}

/// ★ Word blocks that call keccak256 and ECREC prove and verify: each run (an
/// asm program of every keccak256 shape, one of ECREC calls with aligned
/// operands), proved as a streamed block under the word memory format,
/// verifies under it and under neither the split nor the fused format; it
/// states KECCAK_W or ECREC_W tables in their chips' places and MEMW_W tables
/// (the walk's count), no retired or fused table; and its public output is
/// the run's.
#[test]
fn a_word_block_calling_keccak256_and_ecrec_proves_and_verifies() {
    let words = format(MemoryFormat::Words);
    for name in ["test_keccak256", "test_ecrec"] {
        let elf = asm_elf_bytes(name);
        let proof = prove(&elf, &words);
        assert!(verifies(&proof, &elf, &words), "{name}");
        for other in [MemoryFormat::Split, MemoryFormat::Fused] {
            assert!(
                !verifies(&proof, &elf, &format(other)),
                "{name}: verifies under {other:?}"
            );
        }
        let c = &proof.table_counts;
        assert_eq!(
            (c.load, c.store, c.memw, c.memw_aligned),
            (0, 0, 0, 0),
            "{name}: no retired table"
        );
        assert_eq!(proof.fused_memory, [0, 0], "{name}");
        assert!(
            (c.keccak > 0 || c.ecrec > 0) && proof.word_memory[6] > 0,
            "{name}: KECCAK_W or ECREC_W, and MEMW_W"
        );
        let build = word_build(&elf, &words);
        assert_eq!(
            proof.word_memory,
            build.word_counts().map(|n| n as u64),
            "{name}: the walk's word tables"
        );
        assert_eq!(
            proof.public_output, build.public_output_bytes,
            "{name}: the run's public output"
        );
        println!(
            "WORD CHIP BLOCK {name}: KECCAK_W {} ECREC_W {} ECDAS {}, word tables {:?}, output {} bytes",
            c.keccak,
            c.ecrec,
            c.ecdas,
            proof.word_memory,
            proof.public_output.len()
        );
    }
}
