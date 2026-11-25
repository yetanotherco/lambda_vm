//! ★ `LAMBDA_VM_EXEC_STREAM=lean` on real block proves: a block whose
//! executor wrote only its access records, ECALLs and window starting cells,
//! each window re-run by its walker, is the block proved from the executor's
//! logs, under the word memory format: on a CPU run, on runs that call COPY,
//! the commit, Keccak-256 and ECREC, and on one with a private input
//! across two pages; with live regeneration rebuilding every streamed chunk
//! (`RegenMode::Always`, whose digests cover the rebuilt logs); the same
//! partition, heights, counts, output and group roots, both verify, and
//! under the deterministic grind (`LAMBDA_VM_DETERMINISTIC_GRIND`, read once
//! a process: run alone) the same bytes. The proving tests run where a block
//! can be proved (the box's lib run and CI).

use crate::block_whir::regen::RegenMode;
use crate::block_whir::{
    BlockFormat, BlockWhirProof, Deviations, prove_block_whir_with, verify_block_whir,
};
use crate::tables::MaxRowsConfig;
use crate::tables::fused_memory::MemoryFormat;
use crate::test_utils::asm_elf_bytes;
use stark::proof::options::ProofOptions;

/// Two pages' worth of private input, ending inside the second.
fn two_pages_of_input() -> Vec<u8> {
    (0..(crate::tables::page::DEFAULT_PAGE_SIZE as u32 + 300))
        .map(|i| (i * 7 + 3) as u8)
        .collect()
}

fn bytes(proof: &BlockWhirProof) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .expect("serialize")
        .to_vec()
}

/// Whether the process runs record mode (the lean stream's ground): unless
/// the walkers or the history are turned off.
fn record_mode() -> bool {
    std::env::var("LAMBDA_VM_BLOCK_WALKERS").map_or(true, |v| v.trim() != "0")
        && std::env::var("LAMBDA_VM_EXEC_HISTORY").map_or(true, |v| v.trim() != "0")
}

#[test]
fn the_lean_stream_proves_the_same_block() {
    // Runs `test_ecrec`: the ECREC forgery switch stays off meanwhile.
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    let programs = [
        ("fused_mem_cases", Vec::new()),
        ("test_copy_ecalls", Vec::new()),
        ("test_commit_split", Vec::new()),
        ("test_keccak256", Vec::new()),
        ("test_ecrec", Vec::new()),
        ("page_runs", two_pages_of_input()),
    ];
    let regens = [None, Some(RegenMode::Always)];
    // Every block this test compares, named before any is proved: a reader
    // of the output counts the `SAME BYTES` lines against these, so a block
    // skipped (record mode off, the grind not deterministic) never reads as
    // compared.
    for (name, _) in &programs {
        for regen in regens {
            eprintln!("LEAN STREAM WANTS: {name}, regeneration {regen:?}");
        }
    }
    if !record_mode() {
        eprintln!("the lean stream needs record mode: skipped (walkers or history off)");
        return;
    }
    let opts = ProofOptions::default_test_options();
    let format = BlockFormat::production()
        .with_memory(MemoryFormat::Words)
        .with_page_run(0);
    for (name, input) in programs {
        let elf = asm_elf_bytes(name);
        for regen in regens {
            let what = format!("{name}, regeneration {regen:?}");
            let prove = |lean: bool| {
                let mut options = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
                if let Some(mode) = regen {
                    options.drop_streamed_ops = true;
                    options.regen = Some(mode);
                }
                prove_block_whir_with(
                    &elf,
                    &input,
                    &opts,
                    &format,
                    &options,
                    &Deviations {
                        exec_stream_lean: Some(lean),
                        ..Deviations::default()
                    },
                )
                .unwrap_or_else(|e| panic!("{what}, lean {lean}: {e:?}"))
                .0
            };
            let off = prove(false);
            let on = prove(true);
            assert!(
                matches!(verify_block_whir(&on, &elf, &opts, &format), Ok(true)),
                "{what}: lean verifies"
            );
            assert!(
                matches!(verify_block_whir(&off, &elf, &opts, &format), Ok(true)),
                "{what}: full verifies"
            );
            assert_eq!(on.groups, off.groups, "{what}: the partition");
            assert_eq!(on.table_num_vars, off.table_num_vars, "{what}: the heights");
            assert_eq!(
                format!("{:?}", on.table_counts),
                format!("{:?}", off.table_counts),
                "{what}: the counts"
            );
            assert_eq!(on.word_memory, off.word_memory, "{what}: the word tables");
            assert_eq!(on.public_output, off.public_output, "{what}: the output");
            assert_eq!(on.proof.roots, off.proof.roots, "{what}: the group roots");
            if crypto::grinding::deterministic() {
                assert_eq!(bytes(&on), bytes(&off), "{what}: the proof's bytes");
                eprintln!("LEAN STREAM SAME BYTES: {what}");
            }
        }
    }
}

/// The lean stream is refused, never ignored, where it does not run: the
/// whole-run build (no windows), and the split memory format.
#[test]
fn the_lean_stream_is_refused_where_it_does_not_run() {
    let elf = asm_elf_bytes("fused_mem_cases");
    let opts = ProofOptions::default_test_options();
    let lean = Deviations {
        exec_stream_lean: Some(true),
        ..Deviations::default()
    };
    let words = BlockFormat::production()
        .with_memory(MemoryFormat::Words)
        .with_page_run(0);
    let whole = super::block_whir_tests::options(MaxRowsConfig::small(), 3);
    let err = prove_block_whir_with(&elf, &[], &opts, &words, &whole, &lean).map(|_| ());
    assert!(
        matches!(&err, Err(crate::Error::Prover(m)) if m.contains("windowed build")),
        "{err:?}"
    );
    let split = BlockFormat::production()
        .with_memory(MemoryFormat::Split)
        .with_page_run(0);
    let streamed = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
    let err = prove_block_whir_with(&elf, &[], &opts, &split, &streamed, &lean).map(|_| ());
    assert!(
        matches!(&err, Err(crate::Error::Prover(m)) if m.contains("LAMBDA_VM_EXEC_STREAM=lean needs")),
        "{err:?}"
    );
}
