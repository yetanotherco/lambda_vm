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

/// The production format with `memory`'s loads and stores, its pages one
/// table a page (the page-run tests below name their runs).
fn format(memory: MemoryFormat) -> BlockFormat {
    BlockFormat::production()
        .with_memory(memory)
        .with_page_run(0)
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

/// Each proof covers the word-count wire layout and its cross-format refusal.
mod split_and_fused_block_proofs {
    use super::MemoryFormat;

    macro_rules! cases {
        ($name:ident) => {
            mod $name {
                #[test]
                fn split() {
                    super::super::assert_split_or_fused_block(
                        stringify!($name),
                        super::MemoryFormat::Split,
                    );
                }

                #[test]
                fn fused() {
                    super::super::assert_split_or_fused_block(
                        stringify!($name),
                        super::MemoryFormat::Fused,
                    );
                }
            }
        };
    }

    cases!(fused_mem_cases);
    cases!(test_commit_split);
    cases!(test_keccak256);
    cases!(test_ecrec);
}

fn assert_split_or_fused_block(name: &str, memory: MemoryFormat) {
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    let elf = asm_elf_bytes(name);
    let f = format(memory);
    let proof = prove(&elf, &f);
    assert_split_or_fused_wire_bytes(name, memory, &proof);
    assert!(verifies(&proof, &elf, &f), "{name} {memory:?}");
    if name == "fused_mem_cases" {
        assert_unbound_word_counts_refused(&proof, &elf, &f);
    }
    let refused = verify_block_whir(
        &proof,
        &elf,
        &ProofOptions::default_test_options(),
        &format(MemoryFormat::Words),
    );
    let why = match (name, memory) {
        ("fused_mem_cases", MemoryFormat::Split) => "states no LOAD table",
        ("fused_mem_cases", MemoryFormat::Fused) => "states no fused table",
        _ => "the word memory format states no",
    };
    assert!(
        matches!(&refused, Err(crate::Error::InvalidTableCounts(w)) if w.contains(why)),
        "{name} {memory:?}: {refused:?}"
    );
}

/// The split or fused proof appends 56 zero bytes of word counts to the
/// previous wire format. Moving a group index changes the prefix.
fn assert_split_or_fused_wire_bytes(name: &str, memory: MemoryFormat, proof: &BlockWhirProof) {
    if name != "fused_mem_cases" {
        let c = &proof.table_counts;
        assert!(
            c.copy + c.keccak + c.ecrec > 0,
            "{name}: COPY, KECCAK or ECREC tables, the chips' own"
        );
    }
    assert_eq!(proof.word_memory, [0; 7], "{memory:?}");
    let now = rkyv::to_bytes::<rkyv::rancor::Error>(proof).expect("serialize");
    let before = rkyv::to_bytes::<rkyv::rancor::Error>(&BeforeWords::of(proof)).expect("serialize");
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

/// ★ A split or fused proof stating word tables is refused: the counts are
/// absorbed nowhere under those formats, so no value may ride in a proof. One
/// family at 1, and one at `2^32 − 1`, on an honest proof are each refused;
/// with the refusal taken out (a control mutant) each verifies, the counts
/// unbound.
fn assert_unbound_word_counts_refused(honest: &BlockWhirProof, elf: &[u8], format: &BlockFormat) {
    for stated in [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, u64::from(u32::MAX)],
    ] {
        let mut forged = honest.clone();
        forged.word_memory = stated;
        assert!(
            !verifies(&forged, elf, format),
            "{:?} {stated:?} refused",
            format.memory
        );
        WORD_COUNTS_UNCHECKED.with(|c| c.set(true));
        let control = verifies(&forged, elf, format);
        WORD_COUNTS_UNCHECKED.with(|c| c.set(false));
        assert!(
            control,
            "{:?} {stated:?}: without the refusal it verifies",
            format.memory
        );
    }
}

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
fn assert_word_block_verifies(name: &str) {
    let words = format(MemoryFormat::Words);
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

mod a_word_block_proves_and_verifies {
    macro_rules! cases {
        ($($name:ident),+ $(,)?) => {
            $(
                #[test]
                fn $name() {
                    super::assert_word_block_verifies(stringify!($name));
                }
            )+
        };
    }

    cases!(
        fused_mem_cases,
        cross_4gib,
        all_loadstore_32,
        all_instructions_64,
        misalign_ld,
        misalign_lh,
        misalign_lhu,
        misalign_lw,
        misalign_lwu,
        misalign_sd,
        misalign_sh,
        misalign_sw,
    );
}

/// The honest build verifies. Each edited word family still produces a proof
/// that the verifier refuses by the bus balance alone.
mod a_word_block_with_a_moved_word_row_is_refused_by_the_balance_alone {
    use super::*;
    use crate::tables::{mem_load2w, mem_load8, mem_loadw, mem_store2w, mem_store8, mem_storew};

    #[test]
    fn honest() {
        let words = format(MemoryFormat::Words);
        let elf = asm_elf_bytes("fused_mem_cases");
        let mut honest = word_build(&elf, &words);
        let proof = prove_built(&elf, &mut honest, &words).expect("the honest build proves");
        assert!(verifies(&proof, &elf, &words), "the honest build verifies");
    }

    macro_rules! cases {
        ($($family:ident = $index:literal),+ $(,)?) => {
            $(
                #[test]
                fn $family() {
                    super::assert_moved_word_row_refused($index, $family::cols::TIMESTAMP);
                }
            )+
        };
    }

    cases!(
        mem_load8 = 0,
        mem_store8 = 1,
        mem_loadw = 2,
        mem_storew = 3,
        mem_load2w = 4,
        mem_store2w = 5,
    );
}

fn assert_moved_word_row_refused(family: usize, ts: usize) {
    let words = format(MemoryFormat::Words);
    let elf = asm_elf_bytes("fused_mem_cases");
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
    assert!(
        named && matches!(&off, Ok(Ok(()))),
        "{}: every check {all:?} · without the balance {off:?}",
        block_whir::WORD_TABLES[family]
    );
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
mod a_word_block_calling_the_copy_chip_proves_and_verifies {
    macro_rules! cases {
        ($load:ident: $($name:ident),+ $(,)?) => {
            $(
                #[test]
                fn $name() {
                    let name = stringify!($name);
                    super::assert_copy_block_verifies(name, &super::$load(name));
                }
            )+
        };
    }

    cases!(asm_elf_bytes: test_copy_ecalls, copy_4gib, test_copy_overlaps, test_commit_split);
    cases!(rust_elf_bytes: dma_memcpy_cases, dma_memmove_cases, dma_memset_cases, commit);
}

fn assert_copy_block_verifies(name: &str, elf: &[u8]) {
    let words = format(MemoryFormat::Words);
    let proof = prove(elf, &words);
    assert!(verifies(&proof, elf, &words), "{name}");
    for other in [MemoryFormat::Split, MemoryFormat::Fused] {
        assert!(
            !verifies(&proof, elf, &format(other)),
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
    let build = word_build(elf, &words);
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
mod a_word_block_calling_keccak256_and_ecrec_proves_and_verifies {
    #[test]
    fn keccak256() {
        super::assert_word_chip_block_verifies("test_keccak256");
    }

    #[test]
    fn ecrec() {
        let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
        super::assert_word_chip_block_verifies("test_ecrec");
    }
}

fn assert_word_chip_block_verifies(name: &str) {
    let words = format(MemoryFormat::Words);
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
mod page_w_columns_are_the_verifiers_at_proof_level {
    #[test]
    fn zero_page_offset() {
        super::assert_page_w_columns_are_bound(false, "offset");
    }

    #[test]
    fn zero_page_init() {
        super::assert_page_w_columns_are_bound(false, "init");
    }

    #[test]
    fn elf_page_offset() {
        super::assert_page_w_columns_are_bound(true, "offset");
    }

    #[test]
    fn elf_page_init() {
        super::assert_page_w_columns_are_bound(true, "init");
    }
}

fn assert_page_w_columns_are_bound(elf_page: bool, edit: &str) {
    let words = format(MemoryFormat::Words);
    let elf = asm_elf_bytes("fused_mem_cases");
    let mut forged = word_build(&elf, &words);
    let page = forged
        .page_configs
        .iter()
        .position(|c| !c.is_private_input && c.init_values.is_some() == elf_page)
        .expect("the requested zero or ELF page");
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
/// same table counts, word tables, heights and public output; so does one
/// whose streamed ops are dropped as handed out and whose word chunks leave
/// as the windows' parts their rows lie in (`Deviations::word_chunks`, the
/// knob `LAMBDA_VM_WORD_CHUNKS=part`). At word caps of 4 rows, so every
/// family with rows cuts into chunks; on a CPU run and on runs that call
/// COPY, keccak256 and ECREC.
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
        let prove_with = |stream: bool, in_place: bool| {
            let mut options = super::block_whir_tests::streamed(tiny.clone(), 3, 9);
            options.drop_streamed_ops = in_place;
            prove_block_whir_with(
                &elf,
                &[],
                &opts,
                &words,
                &options,
                &Deviations {
                    word_stream: Some(stream),
                    word_chunks: Some(if in_place {
                        crate::block_whir::WordChunks::Parts
                    } else {
                        crate::block_whir::WordChunks::Copy
                    }),
                    ..Deviations::default()
                },
            )
            .unwrap_or_else(|e| panic!("{name} stream {stream} in place {in_place}: {e:?}"))
            .0
        };
        let built = prove_with(false, false);
        for in_place in [false, true] {
            let streamed = prove_with(true, in_place);
            assert!(
                matches!(verify_block_whir(&streamed, &elf, &opts, &words), Ok(true)),
                "{name} {in_place}"
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
}

/// Two pages of private input.
pub(super) fn two_pages_of_input() -> Vec<u8> {
    (0..(crate::tables::page::DEFAULT_PAGE_SIZE as u32 + 300))
        .map(|i| (i * 7 + 3) as u8)
        .collect()
}

/// Each page grouping proves by the streamed walk and from the built traces.
/// The streamed proof is also checked against the other groupings.
mod a_word_block_with_page_runs_proves_and_verifies {
    macro_rules! cases {
        ($($streamed:ident, $built:ident = $run_log2:literal),+ $(,)?) => {
            $(
                #[test]
                fn $streamed() {
                    super::assert_streamed_page_run($run_log2);
                }

                #[test]
                fn $built() {
                    super::assert_built_page_run($run_log2);
                }
            )+
        };
    }

    cases!(
        streamed_off,
        built_off = 0,
        streamed_2,
        built_2 = 1,
        streamed_16,
        built_16 = 4
    );
}

fn assert_streamed_page_run(run_log2: u8) {
    let elf = asm_elf_bytes("page_runs");
    let opts = ProofOptions::default_test_options();
    let words = format(MemoryFormat::Words).with_page_run(run_log2);
    let options = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
    let (proof, _) = prove_block_whir_with(
        &elf,
        &two_pages_of_input(),
        &opts,
        &words,
        &options,
        &Deviations::default(),
    )
    .expect("the prove's own walk proves");
    assert!(matches!(
        verify_block_whir(&proof, &elf, &opts, &words),
        Ok(true)
    ));
    assert_eq!(proof.num_private_input_pages, 2);
    let frame = block_whir::block_frame(proof.statement(), &elf, &opts, &words).expect("frames");
    assert_eq!(
        frame.page_tables,
        crate::tables::page_w::page_tables(&frame.page_configs, run_log2).expect("cut")
    );
    use crate::tables::page_w::PageTableKind;
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
    let built = word_build_with(&elf, &words, &two_pages_of_input());
    assert_eq!(
        built.pages.len(),
        frame.page_tables.len(),
        "at 2^{run_log2}"
    );
    for other in [0, 1, 4].into_iter().filter(|&other| other != run_log2) {
        let their_cut =
            crate::tables::page_w::page_tables(&frame.page_configs, other).expect("other cut");
        assert_ne!(
            frame.page_tables, their_cut,
            "the cuts at 2^{run_log2} and 2^{other} differ"
        );
        assert!(
            !matches!(
                verify_block_whir(&proof, &elf, &opts, &words.with_page_run(other)),
                Ok(true)
            ),
            "proved at 2^{run_log2}, refused at 2^{other}"
        );
    }
}

fn assert_built_page_run(run_log2: u8) {
    let elf = asm_elf_bytes("page_runs");
    let opts = ProofOptions::default_test_options();
    let words = format(MemoryFormat::Words).with_page_run(run_log2);
    let mut built = word_build_with(&elf, &words, &two_pages_of_input());
    let proof = prove_built(&elf, &mut built, &words).expect("the built traces prove");
    assert!(matches!(
        verify_block_whir(&proof, &elf, &opts, &words),
        Ok(true)
    ));
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
