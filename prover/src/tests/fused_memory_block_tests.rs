//! The fused memory format (`tables::fused_memory::MemoryFormat::Fused`) on a
//! block: blocks proved and verified under it, the format a verifier's
//! constant (each format refuses the other's proof), the fused pair of a split
//! proof refused unless `[0, 0]`, and a split proof's bytes those of the
//! format before it with the pair after. The proving tests run where a block
//! can be proved (the box's lib run and CI); the statement's checks below them
//! run anywhere. The inner buses' balance and the tables' obligations, on the
//! builders' traces, are `tables::trace_builder::fused_memory_tests`'.

use crate::block_whir::{
    self, BlockFormat, BlockWhirProof, Deviations, FUSED_COUNTS_UNCHECKED, fused_tables,
    prove_block_whir_with, verify_block_whir,
};
use crate::tables::MaxRowsConfig;
use crate::tables::fused_memory::MemoryFormat;
use crate::test_utils::asm_elf_bytes;
use stark::proof::options::ProofOptions;

/// The production format under the fused memory format.
fn fused() -> BlockFormat {
    BlockFormat::production().with_memory(MemoryFormat::Fused)
}

/// The production format under the split memory format.
fn split() -> BlockFormat {
    BlockFormat::production().with_memory(MemoryFormat::Split)
}

/// `elf` proved as a streamed block (the windowed build, small chunks, so
/// every table streams some) under `format`.
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

/// ★ The fused pair is a verifier constant's reading: under the split format
/// only `[0, 0]` is taken, under the fused format any pair a `usize` holds; with
/// the split format's refusal taken out (a control mutant) any pair reads
/// `[0, 0]`, the field absorbed nowhere. A fused block needs MEMW_A's limbs.
#[test]
fn the_fused_pair_is_read_under_the_verifiers_format() {
    assert_eq!(fused_tables(&split(), [0, 0]).ok(), Some([0, 0]));
    for stated in [[1, 0], [0, 1], [u64::from(u32::MAX), 0]] {
        assert!(fused_tables(&split(), stated).is_err(), "{stated:?}");
        assert_eq!(
            fused_tables(&fused(), stated).ok(),
            Some(stated.map(|n| n as usize)),
            "{stated:?}"
        );
        FUSED_COUNTS_UNCHECKED.with(|c| c.set(true));
        let unchecked = fused_tables(&split(), stated);
        FUSED_COUNTS_UNCHECKED.with(|c| c.set(false));
        assert_eq!(unchecked.ok(), Some([0, 0]), "the control reads none");
    }
    assert!(block_whir::check_memory_format(&fused()).is_ok());
    assert!(block_whir::check_memory_format(&split()).is_ok());
    let lt = fused().with_memw_aligned(crate::tables::memw_aligned::MemwAlignedOrder::Lt);
    assert!(block_whir::check_memory_format(&lt).is_err());
    // The production format (the base's) is fused; the verifier's pinned
    // preset, the only one the tree takes, is split.
    assert_eq!(BlockFormat::pinned().memory, MemoryFormat::Split);
    assert_eq!(BlockFormat::production().memory, MemoryFormat::Fused);
}

/// ★ The split format's statement is the one before the fused format: it
/// absorbs no fused word; the fused format's absorbs `MEMFUSED` and its two
/// counts right after the table counts.
#[test]
fn only_the_fused_format_absorbs_its_pair() {
    assert!(block_whir::fused_statement_words(MemoryFormat::Split, [3, 4]).is_empty());
    assert_eq!(
        block_whir::fused_statement_words(MemoryFormat::Fused, [3, 4]),
        vec![block_whir::FUSED_MEMORY_WORD, 3, 4]
    );
    assert_eq!(
        block_whir::FUSED_MEMORY_WORD.to_le_bytes(),
        *b"MEMFUSED",
        "the word reads as its name"
    );
}

/// ★ Fused blocks prove and verify: every width, sign and alignment of the
/// CPU's loads and stores with bytes of mixed timestamps (`fused_mem_cases`),
/// every instruction, KECCAK's and the commit syscall's memory rows (MEMW_A's
/// ECALL rows beside the fused ones), loads and stores of every kind. Each
/// holds fused tables of both kinds, and LOAD or STORE tables for the
/// accesses MEMW_A would not take.
#[test]
fn a_fused_block_proves_and_verifies() {
    for name in [
        "fused_mem_cases",
        "all_instructions_64",
        "all_loadstore_32",
        "test_keccak_multi",
        "test_commit_split",
    ] {
        let elf = asm_elf_bytes(name);
        let proof = prove(&elf, &fused());
        assert!(verifies(&proof, &elf, &fused()), "{name}");
        if name == "fused_mem_cases" {
            assert!(
                proof.fused_memory[0] > 0 && proof.fused_memory[1] > 0,
                "{name}: {:?} fused tables",
                proof.fused_memory
            );
            assert!(
                proof.table_counts.load > 0 && proof.table_counts.store > 0,
                "the unaligned and mixed accesses stay LOAD's and STORE's"
            );
        }
    }
}

/// ★ A fused block whose streamed chunks are filled from their records moves
/// no byte: every full MEM_LOAD and MEM_STORE chunk handed out as its rows,
/// beside the CPU chunks as their logs and every other row family as its rows
/// (chunks of four rows), each filled from them at its uploads (by the card in
/// a build with one, by a host reader without), proves with the partition,
/// tables and statement of the block built on the host, its bytes equal under
/// the deterministic grind, and verifies.
#[test]
fn a_fused_block_filled_from_its_records_moves_no_byte() {
    use crate::tables::trace_builder::RowFamilies;
    let _fills = super::block_whir_tests::records_fill_lock();
    let max_rows = MaxRowsConfig {
        memw_aligned: 1 << 2,
        load: 1 << 2,
        store: 1 << 2,
        memw: 1 << 2,
        shift: 1 << 2,
        mem_load: 1 << 2,
        mem_store: 1 << 2,
        ..MaxRowsConfig::small()
    };
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    for name in ["fused_mem_cases", "all_loadstore_32"] {
        let elf = asm_elf_bytes(name);
        let proved = |deviations: Deviations| {
            let mut o = super::block_whir_tests::streamed(max_rows.clone(), 5, 3);
            o.drop_streamed_ops = true;
            let (proof, stamps) = prove_block_whir_with(
                &elf,
                &[],
                &ProofOptions::default_test_options(),
                &fused(),
                &o,
                &deviations,
            )
            .expect("prove");
            assert!(verifies(&proof, &elf, &fused()), "{name}");
            (proof, stamps)
        };
        let (host, stamps) = proved(Deviations {
            card_fill: Some(false),
            card_fill_streamed: Some([false, false]),
            card_fill_rows: Some(RowFamilies::NONE),
            ..Deviations::default()
        });
        let off = stamps.windows.card_fill;
        assert_eq!(
            (off.mem_load, off.mem_store, off.records_bytes),
            ([0, 0], [0, 0], 0),
            "{name}: knob off"
        );
        // Regeneration is off here, and parked records need it: kept.
        let (card, stamps) = proved(Deviations {
            card_fill: Some(true),
            card_fill_rows: Some(RowFamilies::ALL),
            records_keep: Some(true),
            ..super::block_whir_tests::streamed_records(true, true)
        });
        let c = stamps.windows.card_fill;
        assert_eq!(c.room_out, None, "{name}: {c:?}");
        assert!(
            c.mem_load[0] > 0 && c.mem_store[0] > 0,
            "{name}: fused chunks held: {c:?}"
        );
        assert_eq!(
            (c.mem_load[1], c.mem_store[1]),
            (0, 0),
            "{name}: on the host: {c:?}"
        );
        let (on_card, on_host) = stamps.records_fills;
        if cfg!(feature = "cuda") {
            assert!(
                on_card > 0,
                "{name}: {on_card} on the card, {on_host} on the host"
            );
        } else {
            assert_eq!(on_card, 0, "{name}: no card");
        }
        assert_eq!(host.groups, card.groups, "{name}: the partition");
        assert_eq!(host.table_num_vars, card.table_num_vars, "{name}");
        assert_eq!(host.fused_memory, card.fused_memory, "{name}");
        assert_eq!(
            format!("{:?}", host.table_counts),
            format!("{:?}", card.table_counts),
            "{name}"
        );
        if crypto::grinding::deterministic() {
            assert_eq!(bytes(&host), bytes(&card), "{name}: the proof");
        }
    }
}

/// ★ A wrong MEM_LOAD or MEM_STORE table is refused by the verifier: with the
/// family's streamed chunks held as their rows and each row moved four cycles
/// later, its step kept ([`Deviations::card_fill_streamed_fault`]), every fill
/// of them (the card's at both uploads, or a host reader's) writes the same
/// wrong table, so the prover makes a proof of a run that did not happen. The
/// unchanged verifier refuses it on the bus balance (the CPU's request and the
/// moved row's memory tokens no longer cancel), and with the balance check
/// alone switched off it would accept it: the balance is the check that
/// refuses. The control, the same block with no fault, verifies.
#[test]
fn a_wrong_fused_table_is_refused_by_the_verifier() {
    use crate::block_whir::{BlockRefusal, verify_block_whir_traced};
    use crate::tables::trace_builder::RowFamilies;
    use stark::multilinear_table::batched::VerifierChecks;
    let _fills = super::block_whir_tests::records_fill_lock();
    let opts = ProofOptions::default_test_options();
    let max_rows = MaxRowsConfig {
        mem_load: 1 << 2,
        mem_store: 1 << 2,
        ..MaxRowsConfig::small()
    };
    let elf = asm_elf_bytes("fused_mem_cases");
    for family in ["MEM_LOAD", "MEM_STORE"] {
        let mut rows = RowFamilies::NONE;
        match family {
            "MEM_LOAD" => rows.mem_load = true,
            _ => rows.mem_store = true,
        }
        let proved = |fault: u64| {
            let mut o = super::block_whir_tests::streamed(max_rows.clone(), 5, 3);
            o.drop_streamed_ops = true;
            let (proof, stamps) = prove_block_whir_with(
                &elf,
                &[],
                &opts,
                &fused(),
                &o,
                &Deviations {
                    card_fill_streamed_fault: fault,
                    card_fill_rows: Some(rows),
                    // Regeneration is off here, and parked records need it: kept.
                    records_keep: Some(true),
                    ..super::block_whir_tests::streamed_records(false, false)
                },
            )
            .unwrap_or_else(|e| {
                panic!("{family}: the prover refused ({e:?}): the verifier was never asked")
            });
            let c = stamps.windows.card_fill;
            let held = match family {
                "MEM_LOAD" => c.mem_load[0],
                _ => c.mem_store[0],
            };
            assert!(held > 0, "{family}: no chunk held as rows: {c:?}");
            proof
        };
        let honest = proved(0);
        assert!(verifies(&honest, &elf, &fused()), "{family}: the control");
        let forged = proved(4);
        let traced = |checks| {
            verify_block_whir_traced(&forged, &elf, &opts, &fused(), false, checks)
                .unwrap_or_else(|e| panic!("{family}: the statement did not frame: {e:?}"))
        };
        let refusal = traced(VerifierChecks::ALL);
        assert!(
            matches!(
                refusal,
                Err(BlockRefusal::Proof(multilinear::Error::BusImbalance))
            ),
            "{family}: refused for another reason, or accepted: {refusal:?}"
        );
        let unbalanced = traced(VerifierChecks {
            balance: false,
            ..VerifierChecks::ALL
        });
        assert!(
            unbalanced.is_ok(),
            "{family}: with the balance check off another check refused: {unbalanced:?}"
        );
        println!(
            "FUSED FORGERY REFUSED: {family} rows four cycles later: refused as BusImbalance; \
             accepted with the balance check off; the honest block verifies"
        );
    }
}

/// ★ Each memory format refuses the other's proof: the format is the
/// verifier's constant, never the proof's.
#[test]
fn each_memory_format_refuses_the_others_proof() {
    let elf = asm_elf_bytes("fused_mem_cases");
    let fused_proof = prove(&elf, &fused());
    let split_proof = prove(&elf, &split());
    assert_eq!(split_proof.fused_memory, [0, 0]);
    assert!(verifies(&fused_proof, &elf, &fused()));
    assert!(verifies(&split_proof, &elf, &split()));
    assert!(!verifies(&fused_proof, &elf, &split()));
    assert!(!verifies(&split_proof, &elf, &fused()));
}

/// ★ A split proof stating fused tables is refused: the pair is absorbed
/// nowhere under the split format, so no value of it may ride in a proof.
/// `(1, 0)`, `(0, 1)` and `(2^32 − 1, 0)` on an honest split proof are each
/// refused; with the refusal taken out (a control mutant) each verifies, the
/// pair unbound.
#[test]
fn a_split_proof_stating_fused_tables_is_refused() {
    let elf = asm_elf_bytes("fused_mem_cases");
    let honest = prove(&elf, &split());
    assert!(verifies(&honest, &elf, &split()));
    for stated in [[1, 0], [0, 1], [u64::from(u32::MAX), 0]] {
        let mut forged = honest.clone();
        forged.fused_memory = stated;
        assert!(!verifies(&forged, &elf, &split()), "{stated:?} refused");
        FUSED_COUNTS_UNCHECKED.with(|c| c.set(true));
        let control = verifies(&forged, &elf, &split());
        FUSED_COUNTS_UNCHECKED.with(|c| c.set(false));
        assert!(control, "{stated:?}: without the refusal it verifies");
    }
}

/// [`BlockWhirProof`] as it was before the fused memory format: every field
/// but the last, in its order.
#[derive(rkyv::Archive, rkyv::Serialize)]
struct BeforeFused {
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
}

impl BeforeFused {
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
        }
    }
}

/// ★ A split proof's bytes are the format's before the fused pair, then the
/// pair: the struct before it archives to exactly the split proof's bytes but
/// its last sixteen, which are `[0, 0]`. So the bytes a pre-format binary
/// proves are compared as this prefix, and a byte moved in any earlier field
/// moves the prefix (the check's mutant: one flipped byte of a group index).
#[test]
fn a_split_proof_is_the_bytes_before_the_fused_pair_then_the_pair() {
    let elf = asm_elf_bytes("all_instructions_64");
    let proof = prove(&elf, &split());
    let now = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("serialize");
    let before =
        rkyv::to_bytes::<rkyv::rancor::Error>(&BeforeFused::of(&proof)).expect("serialize");
    assert_eq!(
        now.len(),
        before.len() + 16,
        "the pair, sixteen bytes after"
    );
    assert_eq!(
        &now[..before.len()],
        &before[..],
        "the bytes before the pair"
    );
    assert_eq!(&now[before.len()..], &[0u8; 16], "the pair, [0, 0]");
    let mut moved = proof.clone();
    moved.groups[0][0] ^= 1;
    let moved_before =
        rkyv::to_bytes::<rkyv::rancor::Error>(&BeforeFused::of(&moved)).expect("serialize");
    assert_ne!(
        &moved_before[..],
        &before[..],
        "a moved byte parts the prefix"
    );
}
