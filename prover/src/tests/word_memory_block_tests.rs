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
//! bus balance alone. Runs that call keccak256 and ECREC prove and verify
//! under it. A forger's block under the 32-bit register tokens is refused
//! by the retired-table rule alone; PAGE_W's preprocessed columns, private
//! pages and the prepared tables' dense arm hold on real proofs; word
//! families cut into several tables prove.
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
/// proof's bytes but its last 56, which are zero. On a chip-free run and on
/// runs that call COPY (commits), keccak256 and ECREC, whose tables the word
/// format replaces. So the bytes a binary
/// before the word format proves are this prefix, and a byte moved in any
/// earlier field moves the prefix (the check's mutant: one flipped byte of a
/// group index).
#[test]
fn a_split_and_a_fused_proof_are_the_bytes_before_the_word_counts_then_zeros() {
    // Runs `test_ecrec`: the ECREC forgery switch stays off meanwhile.
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    for (name, memory) in [
        "fused_mem_cases",
        "test_commit_split",
        "test_keccak256",
        "test_ecrec",
    ]
    .into_iter()
    .flat_map(|name| [(name, MemoryFormat::Split), (name, MemoryFormat::Fused)])
    {
        let elf = asm_elf_bytes(name);
        let proof = prove(&elf, &format(memory));
        if name != "fused_mem_cases" {
            let c = &proof.table_counts;
            assert!(
                c.copy + c.keccak + c.ecrec > 0,
                "{name}: COPY, KECCAK or ECREC tables, the chips' own"
            );
        }
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
    // Runs `test_ecrec`: the ECREC forgery switch stays off meanwhile.
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
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
    // The chips' runs: a split or fused proof stating COPY, KECCAK or ECREC
    // tables (the byte-message chips) is refused under the word format by
    // what it retires.
    for (name, memory) in ["test_commit_split", "test_keccak256", "test_ecrec"]
        .into_iter()
        .flat_map(|name| [(name, MemoryFormat::Split), (name, MemoryFormat::Fused)])
    {
        let elf = asm_elf_bytes(name);
        let proof = prove(&elf, &format(memory));
        assert!(verifies(&proof, &elf, &format(memory)));
        let refused = verify_block_whir(
            &proof,
            &elf,
            &ProofOptions::default_test_options(),
            &format(MemoryFormat::Words),
        );
        assert!(
            matches!(&refused, Err(crate::Error::InvalidTableCounts(w)) if w.contains("the word memory format states no")),
            "{name} {memory:?}: {refused:?}"
        );
    }
}

/// The chip-free programs a word block is proved of: every width, sign and
/// alignment of the CPU's loads and stores, every instruction, the misaligned
/// accesses one by one, and loads and stores across `2^32` (their second
/// word's key carried into the high word).
const WORD_PROGRAMS: [&str; 12] = [
    "fused_mem_cases",
    "cross_4gib",
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
    word_build_with(elf, format, &[])
}

/// [`word_build`] of a run given `private` input.
fn word_build_with(
    elf: &[u8],
    format: &BlockFormat,
    private: &[u8],
) -> crate::tables::trace_builder::Traces {
    use crate::tables::trace_builder::WindowedTraceBuilder;
    use crate::tables::trace_builder::in_row_forgery_tests::with_verifier_files;
    let program = executor::elf::Elf::load(elf).expect("the ELF loads");
    let logs = executor::vm::execution::Executor::new(&program, private.to_vec())
        .expect("the executor")
        .run()
        .expect("the run")
        .logs;
    let mut builder = WindowedTraceBuilder::new(&program, private, &MaxRowsConfig::small())
        .and_then(|b| b.register_format(format.registers))
        .and_then(|b| b.memw_aligned_order(format.memw_aligned))
        .map(|b| b.keccak_rnd_form(format.keccak_rnd))
        .map(|b| b.ecdas_form(format.ecdas))
        .and_then(|b| b.memory_format(format.memory))
        .and_then(|b| b.page_run_log2(format.page_run_log2))
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
    if format.registers == crate::tables::register::RegisterFormat::InRow {
        with_verifier_files(&program, &mut traces);
    }
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
/// overlapping copies, commits split across calls and calls across `2^32`;
/// Rust guests' memcpy, memmove, memset and commit cases.
const COPY_ASM: [&str; 4] = [
    "test_copy_ecalls",
    "copy_4gib",
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
    // Runs `test_ecrec`: the ECREC forgery switch stays off meanwhile.
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
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

/// ★ The word memory format under the 32-bit register tokens, which the
/// prover refuses (the walk and the prove's gate): a forger's walk skips the
/// refusal (on this thread), and its block proves. The build states a MEMW_A
/// table (the register accesses MEMW_R cannot take), and the verifier
/// refuses the proof by the retired-table rule, under either register
/// format; with that rule taken out (a control mutant) it verifies under
/// the 32-bit tokens, so the rule is the one check.
#[test]
fn a_word_block_under_the_32_bit_registers_is_refused_by_the_retired_table_rule_alone() {
    use crate::tables::register::RegisterFormat;
    use crate::tables::trace_builder::WORDS_UNDER_TOKENS32;
    let tokens32 = BlockFormat {
        registers: RegisterFormat::Tokens32,
        ..format(MemoryFormat::Words)
    };
    for name in ["all_instructions_64", "fused_mem_cases"] {
        let elf = asm_elf_bytes(name);
        WORDS_UNDER_TOKENS32.with(|c| c.set(true));
        let mut forged = word_build(&elf, &tokens32);
        WORDS_UNDER_TOKENS32.with(|c| c.set(false));
        let proof = prove_built(&elf, &mut forged, &tokens32).expect("the forged build proves");
        assert_eq!(
            proof.table_counts.memw_aligned, 1,
            "{name}: one MEMW_A table"
        );
        for registers in [RegisterFormat::Tokens32, RegisterFormat::InRow] {
            let f = BlockFormat {
                registers,
                ..format(MemoryFormat::Words)
            };
            let refused =
                verify_block_whir(&proof, &elf, &ProofOptions::default_test_options(), &f);
            assert!(
                matches!(&refused, Err(crate::Error::InvalidTableCounts(why)) if why.contains("states no MEMW_A table")),
                "{name} {registers:?}: {refused:?}"
            );
        }
        block_whir::RETIRED_COUNTS_UNCHECKED.with(|c| c.set(true));
        let control = verifies(&proof, &elf, &tokens32);
        block_whir::RETIRED_COUNTS_UNCHECKED.with(|c| c.set(false));
        assert!(
            control,
            "{name}: without the retired-table rule it verifies"
        );
    }
}

/// [`prove_built`] with the prover's own check of the preprocessed columns
/// off: a forger's prover, so the verifier is what answers for them.
fn prove_forged(
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
        &Deviations {
            unchecked_preprocessed: true,
            ..Deviations::default()
        },
        true,
        &|_, _| {},
        &mut stamps,
    )
}

/// The last row of PAGE_W table `page` no access touched (final timestamp
/// 0), not the page's last row.
fn untouched(traces: &crate::tables::trace_builder::Traces, page: usize) -> usize {
    use crate::tables::page_w::cols as pw;
    let t = &traces.pages[page];
    (1..t.num_rows() - 1)
        .rev()
        .find(|&r| t.get_main(r, pw::TIMESTAMP).canonical_u64() == 0)
        .expect("an untouched word")
}

/// One edit of a PAGE_W row no access touched (its first and last tokens are
/// one token, made and taken by the row, so every bus still balances
/// whatever the row's key or word): `"offset"`, its `OFFSET` one more (the
/// next word's key); `"init"`, its initial word one more (its final with
/// it); `"half"`, its initial low half `2^16` more (a private half past 16
/// bits).
fn edit_page(
    traces: &mut crate::tables::trace_builder::Traces,
    page: usize,
    row: usize,
    edit: &str,
) {
    use crate::tables::page_w::cols as pw;
    use crate::tables::types::FE;
    let t = &mut traces.pages[page];
    let mut bump = |col: usize, by: u64| {
        let v = *t.get_main(row, col);
        t.main_table.set(row, col, v + FE::from(by));
    };
    match edit {
        "offset" => bump(pw::OFFSET, 1),
        "init" => {
            bump(pw::INIT[0], 1);
            bump(pw::FINI_LO, 1);
        }
        _ => {
            bump(pw::INIT[0], 1 << 16);
            bump(pw::FINI_LO, 1 << 16);
        }
    }
}

/// Whether the verifier refuses: a refusal or an error.
fn refused(said: &Result<Result<(), block_whir::BlockRefusal>, String>) -> bool {
    matches!(said, Ok(Err(_)) | Err(_))
}

/// ★ PAGE_W's preprocessed columns are the verifier's on a real proof. On
/// `fused_mem_cases`, a word block whose zero page or ELF page has one
/// untouched row's `OFFSET` moved (a key minted twice) or its initial word
/// changed, proved from the edited traces by a prover that skips its own
/// check: refused by the verifier; with the preprocessed check off (a
/// control) each verifies, so that check is the one that answers.
#[test]
fn page_w_columns_are_the_verifiers_at_proof_level() {
    let words = format(MemoryFormat::Words);
    let elf = asm_elf_bytes("fused_mem_cases");
    let honest = word_build(&elf, &words);
    let zero = honest
        .page_configs
        .iter()
        .position(|c| !c.is_private_input && c.init_values.is_none())
        .expect("a zero page");
    let elf_page = honest
        .page_configs
        .iter()
        .position(|c| !c.is_private_input && c.init_values.is_some())
        .expect("an ELF page");
    for page in [zero, elf_page] {
        for edit in ["offset", "init"] {
            let mut forged = word_build(&elf, &words);
            let row = untouched(&forged, page);
            edit_page(&mut forged, page, row, edit);
            assert!(
                prove_built(&elf, &mut word_edit(&elf, &words, page, row, edit), &words).is_err(),
                "page {page} {edit}: the honest prover refuses the edited traces"
            );
            let proof = prove_forged(&elf, &mut forged, &words).expect("the forger proves");
            let all = verdict(&proof, &elf, &words, VerifierChecks::ALL);
            assert!(refused(&all), "page {page} {edit}: {all:?}");
            let without = verdict(
                &proof,
                &elf,
                &words,
                VerifierChecks {
                    preprocessed: false,
                    ..VerifierChecks::ALL
                },
            );
            assert!(
                matches!(without, Ok(Ok(()))),
                "page {page} {edit}: control: {without:?}"
            );
        }
    }
}

/// `elf`'s word build with one page edit (see [`edit_page`]).
fn word_edit(
    elf: &[u8],
    words: &BlockFormat,
    page: usize,
    row: usize,
    edit: &str,
) -> crate::tables::trace_builder::Traces {
    let mut t = word_build(elf, words);
    edit_page(&mut t, page, row, edit);
    t
}

/// ★ Completeness and the private input: a word block given private input
/// proves and verifies (its private pages are PAGE_W tables whose initial
/// halves are main columns under `IS_HALF`), by the prove's own walk and
/// from the built traces. A private row's initial half `2^16` more on an
/// untouched row is refused by the balance (the `IS_HALF` lookup) and
/// verifies with the balance off; a private row's `OFFSET` moved is refused
/// by the preprocessed check and verifies with it off.
#[test]
fn a_word_block_with_private_input_proves_and_its_private_rows_are_held() {
    let words = format(MemoryFormat::Words);
    let elf = asm_elf_bytes("fused_mem_cases");
    let private: Vec<u8> = (0..300u32).map(|i| (i * 7 + 3) as u8).collect();
    let options = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
    let opts = ProofOptions::default_test_options();
    let (proof, _) = prove_block_whir_with(
        &elf,
        &private,
        &opts,
        &words,
        &options,
        &Deviations::default(),
    )
    .expect("the prove's own walk proves");
    assert_eq!(proof.num_private_input_pages, 1);
    assert!(matches!(
        verify_block_whir(&proof, &elf, &opts, &words),
        Ok(true)
    ));
    let mut honest = word_build_with(&elf, &words, &private);
    let page = honest
        .page_configs
        .iter()
        .position(|c| c.is_private_input)
        .expect("a private page");
    let proof = prove_built(&elf, &mut honest, &words).expect("the built traces prove");
    assert!(matches!(
        verify_block_whir(&proof, &elf, &opts, &words),
        Ok(true)
    ));
    for (edit, off) in [
        (
            "half",
            VerifierChecks {
                balance: false,
                ..VerifierChecks::ALL
            },
        ),
        (
            "offset",
            VerifierChecks {
                preprocessed: false,
                ..VerifierChecks::ALL
            },
        ),
    ] {
        let mut forged = word_build_with(&elf, &words, &private);
        let row = untouched(&forged, page);
        edit_page(&mut forged, page, row, edit);
        let proof = prove_forged(&elf, &mut forged, &words).expect("the forger proves");
        let all = verdict(&proof, &elf, &words, VerifierChecks::ALL);
        assert!(refused(&all), "{edit}: {all:?}");
        let control = verdict(&proof, &elf, &words, off);
        assert!(
            matches!(control, Ok(Ok(()))),
            "{edit}: control: {control:?}"
        );
    }
}

/// ★ The dense arm of the prepared tables under the word format, which no
/// test program reaches on its own (a page is dense only past about 11,700
/// nonzero bytes): with every page that has initial data settled by the
/// prepared opening on both sides (on this thread), a word block proves and
/// verifies with a prepared opening, and an ELF page's `OFFSET` moved or
/// initial word changed on an untouched row is refused by that opening, with
/// or without the preprocessed check.
#[test]
fn a_word_block_with_dense_pages_proves_and_holds_its_pages() {
    let words = format(MemoryFormat::Words);
    let elf = asm_elf_bytes("fused_mem_cases");
    block_whir::DENSE_PAGES_FORCED.with(|c| c.set(true));
    let checked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut honest = word_build(&elf, &words);
        let proof = prove_built(&elf, &mut honest, &words).expect("the honest build proves");
        assert!(!proof.prepared.is_empty(), "a prepared opening");
        let said = verdict(&proof, &elf, &words, VerifierChecks::ALL);
        assert!(matches!(said, Ok(Ok(()))), "the honest block: {said:?}");
        let page = honest
            .page_configs
            .iter()
            .position(|c| !c.is_private_input && c.init_values.is_some())
            .expect("an ELF page");
        for edit in ["offset", "init"] {
            let mut forged = word_build(&elf, &words);
            let row = untouched(&forged, page);
            edit_page(&mut forged, page, row, edit);
            let proof = prove_forged(&elf, &mut forged, &words).expect("the forger proves");
            for checks in [
                VerifierChecks::ALL,
                VerifierChecks {
                    preprocessed: false,
                    ..VerifierChecks::ALL
                },
            ] {
                let said = verdict(&proof, &elf, &words, checks);
                assert!(refused(&said), "{edit}: {said:?}");
            }
        }
    }));
    block_whir::DENSE_PAGES_FORCED.with(|c| c.set(false));
    if let Err(panic) = checked {
        std::panic::resume_unwind(panic);
    }
}

/// ★ Completeness: a word block whose word families are cut into several
/// tables (at most 4 rows a table) proves and verifies.
#[test]
fn a_word_block_with_cut_word_families_proves_and_verifies() {
    let words = format(MemoryFormat::Words);
    let elf = asm_elf_bytes("fused_mem_cases");
    let tiny = MaxRowsConfig {
        mem_load: 4,
        mem_store: 4,
        ..MaxRowsConfig::small()
    };
    let options = super::block_whir_tests::streamed(tiny, 3, 9);
    let opts = ProofOptions::default_test_options();
    let (proof, _) =
        prove_block_whir_with(&elf, &[], &opts, &words, &options, &Deviations::default())
            .expect("the cut block proves");
    assert!(
        proof.word_memory.iter().any(|&n| n > 1),
        "{:?}",
        proof.word_memory
    );
    assert!(matches!(
        verify_block_whir(&proof, &elf, &opts, &words),
        Ok(true)
    ));
}

/// ★ A word block whose word families stream in chunks as the walk fills
/// them (`Deviations::word_stream`, the knob `LAMBDA_VM_WORD_STREAM=1`)
/// proves and verifies, and states what the finish-built block states: the
/// same table counts, word tables, heights and public output. At word caps of
/// 4 rows, so every family with rows cuts into chunks; on a CPU run and on
/// runs that call COPY, keccak256 and ECREC.
#[test]
fn a_word_block_with_streamed_word_families_states_the_finish_built_block() {
    // Runs `test_ecrec`: the ECREC forgery switch stays off meanwhile.
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    let words = format(MemoryFormat::Words);
    let tiny = MaxRowsConfig {
        mem_load: 4,
        mem_store: 4,
        ..MaxRowsConfig::small()
    };
    let opts = ProofOptions::default_test_options();
    for name in [
        "fused_mem_cases",
        "test_copy_ecalls",
        "test_keccak256",
        "test_ecrec",
    ] {
        let elf = asm_elf_bytes(name);
        let options = super::block_whir_tests::streamed(tiny.clone(), 3, 9);
        let prove_with = |stream: bool| {
            prove_block_whir_with(
                &elf,
                &[],
                &opts,
                &words,
                &options,
                &Deviations {
                    word_stream: Some(stream),
                    ..Deviations::default()
                },
            )
            .unwrap_or_else(|e| panic!("{name} stream {stream}: {e:?}"))
            .0
        };
        let (built, streamed) = (prove_with(false), prove_with(true));
        assert!(
            matches!(verify_block_whir(&streamed, &elf, &opts, &words), Ok(true)),
            "{name}"
        );
        assert_eq!(
            format!("{:?}", streamed.table_counts),
            format!("{:?}", built.table_counts),
            "{name}: the counts"
        );
        assert_eq!(
            streamed.word_memory, built.word_memory,
            "{name}: the word tables"
        );
        assert_eq!(
            streamed.table_num_vars, built.table_num_vars,
            "{name}: the heights"
        );
        assert_eq!(
            streamed.public_output, built.public_output,
            "{name}: the output"
        );
        assert!(
            streamed.word_memory.iter().any(|&n| n > 1),
            "{name}: a family cut into several tables"
        );
    }
}

/// The private input of [`page_runs_block`]: two pages' worth.
fn two_pages_of_input() -> Vec<u8> {
    (0..(crate::tables::page::DEFAULT_PAGE_SIZE as u32 + 300))
        .map(|i| (i * 7 + 3) as u8)
        .collect()
}

/// `page_runs` (runs of zero pages from pages 64 and 81, two private-input
/// pages) proved by the prove's own walk under `format`.
fn page_runs_block(format: &BlockFormat) -> (Vec<u8>, BlockWhirProof) {
    let elf = asm_elf_bytes("page_runs");
    let options = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
    let (proof, _) = prove_block_whir_with(
        &elf,
        &two_pages_of_input(),
        &ProofOptions::default_test_options(),
        format,
        &options,
        &Deviations::default(),
    )
    .expect("the prove's own walk proves");
    (elf, proof)
}

/// ★ Completeness under page runs: `page_runs` proves and verifies with runs
/// off and at runs of 2 and 16 pages, by the prove's own walk and from the
/// built traces; its page tables are the rule's cut of the verifier's pages,
/// zero and private blocks among them under runs; each proof is refused at
/// the other runs, whose cuts differ.
#[test]
fn a_word_block_with_page_runs_proves_and_verifies() {
    use crate::tables::page_w::{self, PageTableKind};
    let opts = ProofOptions::default_test_options();
    let runs = [0u8, 1, 4];
    let mut proofs = Vec::new();
    for run_log2 in runs {
        let words = format(MemoryFormat::Words).with_page_run(run_log2);
        let (elf, proof) = page_runs_block(&words);
        assert!(
            matches!(verify_block_whir(&proof, &elf, &opts, &words), Ok(true)),
            "at 2^{run_log2}"
        );
        assert_eq!(proof.num_private_input_pages, 2);
        let frame =
            block_whir::block_frame(proof.statement(), &elf, &opts, &words).expect("frames");
        assert_eq!(
            frame.page_tables,
            page_w::page_tables(&frame.page_configs, run_log2).expect("cut")
        );
        let blocks = |kind| {
            frame
                .page_tables
                .iter()
                .filter(|t| t.kind == kind && t.pages() > 1)
                .count()
        };
        if run_log2 == 0 {
            assert_eq!(frame.page_tables.len(), frame.page_configs.len());
        } else {
            assert!(
                blocks(PageTableKind::Zero) > 0,
                "at 2^{run_log2}: a zero block"
            );
            assert_eq!(blocks(PageTableKind::Private), 1, "at 2^{run_log2}");
        }
        // From the built traces: the finish cut the pages as the format.
        let mut built = word_build_with(&elf, &words, &two_pages_of_input());
        assert_eq!(
            built.pages.len(),
            frame.page_tables.len(),
            "at 2^{run_log2}"
        );
        let again = prove_built(&elf, &mut built, &words).expect("the built traces prove");
        assert!(matches!(
            verify_block_whir(&again, &elf, &opts, &words),
            Ok(true)
        ));
        proofs.push((run_log2, words, elf, proof, frame.page_tables.clone()));
    }
    for (r, _, elf, proof, cut) in &proofs {
        for (s, theirs, _, _, their_cut) in &proofs {
            if r != s {
                assert_ne!(cut, their_cut, "the cuts at 2^{r} and 2^{s} differ");
                assert!(
                    !matches!(verify_block_whir(proof, elf, &opts, theirs), Ok(true)),
                    "proved at 2^{r}, refused at 2^{s}"
                );
            }
        }
    }
}

/// The last row of page table `page` no access touched (final timestamp 0),
/// not its last row: a zero block's timestamp is its fourth column.
fn untouched_row(traces: &crate::tables::trace_builder::Traces, page: usize) -> usize {
    use crate::tables::page_w::{cols as pw, zero_cols as zw};
    let t = &traces.pages[page];
    let ts = if t.main_table.width == zw::NUM_COLUMNS {
        zw::TIMESTAMP
    } else {
        pw::TIMESTAMP
    };
    (1..t.num_rows() - 1)
        .rev()
        .find(|&r| t.get_main(r, ts).canonical_u64() == 0)
        .expect("an untouched word")
}

/// ★ A page block's columns are the verifier's on a real proof, at runs of 16
/// pages: a zero block's and a private block's untouched row's `OFFSET` moved
/// (a key minted twice), proved by a forger: refused, and verifying with the
/// preprocessed check off (the control); a private block's untouched row's
/// initial half `2^16` more: refused, and verifying with the balance off.
#[test]
fn page_block_columns_are_the_verifiers_at_proof_level() {
    use crate::tables::page_w::{self, PageTableKind};
    let words = format(MemoryFormat::Words).with_page_run(4);
    let elf = asm_elf_bytes("page_runs");
    let input = two_pages_of_input();
    let honest = word_build_with(&elf, &words, &input);
    let tables = page_w::page_tables(&honest.page_configs, 4).expect("cut");
    let block = |kind| {
        tables
            .iter()
            .position(|t| t.kind == kind && t.pages() > 1)
            .expect("a block of the class")
    };
    let off = |preprocessed: bool| VerifierChecks {
        preprocessed: !preprocessed,
        balance: preprocessed,
        ..VerifierChecks::ALL
    };
    for (page, edit) in [
        (block(PageTableKind::Zero), "offset"),
        (block(PageTableKind::Private), "offset"),
        (block(PageTableKind::Private), "half"),
    ] {
        let mut forged = word_build_with(&elf, &words, &input);
        let row = untouched_row(&forged, page);
        edit_page(&mut forged, page, row, edit);
        let proof = prove_forged(&elf, &mut forged, &words).expect("the forger proves");
        let all = verdict(&proof, &elf, &words, VerifierChecks::ALL);
        assert!(refused(&all), "{page} {edit}: {all:?}");
        let control = verdict(&proof, &elf, &words, off(edit == "offset"));
        assert!(
            matches!(control, Ok(Ok(()))),
            "{page} {edit}: control: {control:?}"
        );
    }
}

/// ★ The dense arm under page runs: with every page that has initial data
/// settled by the prepared opening on both sides (on this thread, so the
/// prover is the built traces' prove, which derives its prepared tables on the
/// calling thread: the streamed prove derives them on its layout thread, where
/// this thread's switch is off), `page_runs` proves and verifies with runs off
/// and at runs of 16 pages, and its opening settles the same tables (by name,
/// a page of data among them) with the same columns at both.
#[test]
fn a_word_block_with_page_runs_holds_its_dense_pages() {
    block_whir::DENSE_PAGES_FORCED.with(|c| c.set(true));
    let checked = std::panic::catch_unwind(|| {
        let opts = ProofOptions::default_test_options();
        let elf = asm_elf_bytes("page_runs");
        let mut settled = Vec::new();
        for run_log2 in [0u8, 4] {
            let words = format(MemoryFormat::Words).with_page_run(run_log2);
            let mut built = word_build_with(&elf, &words, &two_pages_of_input());
            let proof = prove_built(&elf, &mut built, &words).expect("the built traces prove");
            assert!(!proof.prepared.is_empty(), "a prepared opening");
            let said = verify_block_whir(&proof, &elf, &opts, &words);
            assert!(matches!(said, Ok(true)), "at 2^{run_log2}: {said:?}");
            let frame =
                block_whir::block_frame(proof.statement(), &elf, &opts, &words).expect("frames");
            let refs = frame.airs.air_refs();
            let names: Vec<(String, Vec<Vec<crate::tables::types::FE>>)> =
                block_whir::prepared_tables(&frame.airs, &frame.page_configs, &words)
                    .expect("prepared")
                    .into_iter()
                    .map(|(t, columns)| (refs[t].name().to_string(), columns))
                    .collect();
            assert!(
                names
                    .iter()
                    .any(|(n, _)| n.starts_with(crate::tables::page_w::NAME_PREFIX)),
                "at 2^{run_log2}: a page of data settled by the opening"
            );
            settled.push(names);
        }
        assert!(settled[0] == settled[1], "the same tables settled");
    });
    block_whir::DENSE_PAGES_FORCED.with(|c| c.set(false));
    if let Err(panic) = checked {
        std::panic::resume_unwind(panic);
    }
}
