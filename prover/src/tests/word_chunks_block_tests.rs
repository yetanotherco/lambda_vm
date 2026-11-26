//! ★ The default word chunks on real block proves: a word block whose
//! streamed word families' chunks are handed out as the windows' parts their
//! rows lie in is the block whose chunks are copied out of one list a family
//! (`LAMBDA_VM_WORD_CHUNKS=copy`), alone and beside the host build's other
//! levers (the serial host work hoisted, every walk cut, both finish steps, the executor's lean
//! stream), and with live regeneration rebuilding every streamed chunk
//! (`RegenMode::Always`): the same partition, heights, counts, word tables,
//! output and group roots, every proof verifies, and under the deterministic
//! grind (`LAMBDA_VM_DETERMINISTIC_GRIND`, read once a process: run alone) the
//! same bytes. At word caps of 4 rows, so every family with rows hands out
//! chunks. The proving tests run where a block can be proved (the box's lib
//! run and CI).

use crate::block_whir::regen::RegenMode;
use crate::block_whir::{
    BlockFormat, BlockWhirProof, Deviations, WordChunks, prove_block_whir_with, verify_block_whir,
};
use crate::tables::MaxRowsConfig;
use crate::tables::fused_memory::MemoryFormat;
use crate::tables::trace_builder::{FinishEarly, HostHoist, WalkCuts};
use crate::test_utils::asm_elf_bytes;
use stark::proof::options::ProofOptions;

fn bytes(proof: &BlockWhirProof) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .expect("serialize")
        .to_vec()
}

/// Two pages' worth of private input, ending inside the second.
fn two_pages_of_input() -> Vec<u8> {
    (0..(crate::tables::page::DEFAULT_PAGE_SIZE as u32 + 300))
        .map(|i| (i * 7 + 3) as u8)
        .collect()
}

/// Whether the process runs record mode (the lean stream's ground): unless
/// the walkers or the history are turned off.
fn record_mode() -> bool {
    std::env::var("LAMBDA_VM_BLOCK_WALKERS").map_or(true, |v| v.trim() != "0")
        && std::env::var("LAMBDA_VM_EXEC_HISTORY").map_or(true, |v| v.trim() != "0")
}

#[test]
fn the_word_chunks_in_parts_prove_the_same_block() {
    // Runs `test_ecrec`: the ECREC forgery switch stays off meanwhile.
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    let programs = [
        ("fused_mem_cases", Vec::new()),
        ("test_copy_ecalls", Vec::new()),
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
            eprintln!("WORD CHUNKS WANTS: {name}, regeneration {regen:?}");
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
    let tiny = MaxRowsConfig {
        mem_load: 4,
        mem_store: 4,
        ..MaxRowsConfig::small()
    };
    for (name, input) in programs {
        let elf = asm_elf_bytes(name);
        for regen in regens {
            let what = format!("{name}, regeneration {regen:?}");
            // `levers`: the hoist, every walk cut, both finish steps and the
            // lean stream on, on three walkers.
            let prove = |chunks: WordChunks, levers: bool| {
                let mut options = super::block_whir_tests::streamed(tiny.clone(), 3, 9);
                options.drop_streamed_ops = true;
                options.regen = regen;
                let (hoist, cuts, early) = if levers {
                    (HostHoist::all(), WalkCuts::all(), FinishEarly::all())
                } else {
                    (
                        HostHoist::default(),
                        WalkCuts::default(),
                        FinishEarly::default(),
                    )
                };
                prove_block_whir_with(
                    &elf,
                    &input,
                    &opts,
                    &format,
                    &options,
                    &Deviations {
                        word_stream: Some(true),
                        word_chunks: Some(chunks),
                        walkers: Some(3),
                        host_hoist: Some(hoist),
                        walk_cuts: Some(cuts),
                        finish_early: Some(early),
                        exec_stream_lean: Some(levers),
                        ..Deviations::default()
                    },
                )
                .unwrap_or_else(|e| panic!("{what}, {chunks:?}, levers {levers}: {e:?}"))
                .0
            };
            let off = prove(WordChunks::Copy, false);
            assert!(
                matches!(verify_block_whir(&off, &elf, &opts, &format), Ok(true)),
                "{what}: copy verifies"
            );
            assert!(
                off.word_memory.iter().any(|&n| n > 1),
                "{what}: a word family cut into several tables {:?}",
                off.word_memory
            );
            for levers in [false, true] {
                let arm = format!("{what}, part, levers {levers}");
                let on = prove(WordChunks::Parts, levers);
                assert!(
                    matches!(verify_block_whir(&on, &elf, &opts, &format), Ok(true)),
                    "{arm}: verifies"
                );
                assert_eq!(on.groups, off.groups, "{arm}: the partition");
                assert_eq!(on.table_num_vars, off.table_num_vars, "{arm}: the heights");
                assert_eq!(
                    format!("{:?}", on.table_counts),
                    format!("{:?}", off.table_counts),
                    "{arm}: the counts"
                );
                assert_eq!(on.word_memory, off.word_memory, "{arm}: the word tables");
                assert_eq!(on.public_output, off.public_output, "{arm}: the output");
                assert_eq!(on.proof.roots, off.proof.roots, "{arm}: the group roots");
                if crypto::grinding::deterministic() {
                    assert_eq!(bytes(&on), bytes(&off), "{arm}: the proof's bytes");
                }
            }
            if crypto::grinding::deterministic() {
                eprintln!("WORD CHUNKS SAME BYTES: {what}");
            }
        }
    }
}

/// The production defaults preserve the bytes of both explicit opt-outs.
#[test]
#[ignore = "box tier, run alone with deterministic grind and fixed trace hash"]
fn the_default_word_chunks_and_walkers_preserve_the_opt_out_proof_bytes() {
    assert!(crypto::grinding::deterministic());
    assert_eq!(
        std::env::var("LAMBDA_VM_FIXED_TRACE_HASH").as_deref(),
        Ok("1")
    );
    for knob in ["LAMBDA_VM_WORD_CHUNKS", "LAMBDA_VM_BLOCK_WALKERS"] {
        assert!(std::env::var_os(knob).is_none(), "run with {knob} unset");
    }
    let elf = asm_elf_bytes("fused_mem_cases");
    let opts = ProofOptions::default_test_options();
    let format = BlockFormat::production()
        .with_memory(MemoryFormat::Words)
        .with_page_run(0);
    let tiny = MaxRowsConfig {
        mem_load: 4,
        mem_store: 4,
        ..MaxRowsConfig::small()
    };
    let mut options = super::block_whir_tests::streamed(tiny, 3, 4);
    options.drop_streamed_ops = true;
    let prove = |word_chunks, walkers| {
        let (proof, _) = prove_block_whir_with(
            &elf,
            &[],
            &opts,
            &format,
            &options,
            &Deviations {
                word_chunks,
                walkers,
                ..Deviations::default()
            },
        )
        .expect("the word block proves");
        assert!(
            proof.word_memory.iter().any(|&n| n > 1),
            "the walk hands out word chunks"
        );
        assert!(verify_block_whir(&proof, &elf, &opts, &format).expect("verify runs"));
        bytes(&proof)
    };
    let default = prove(None, None);
    assert_eq!(prove(Some(WordChunks::Copy), None), default, "copy");
    assert_eq!(prove(None, Some(4)), default, "four walkers");
}

/// Explicit `part` requires streamed and dropped word rows. The default
/// also handles split memory, the word stream off, and kept ops.
#[test]
fn the_word_chunks_in_parts_are_refused_where_the_word_families_do_not_stream() {
    let elf = asm_elf_bytes("fused_mem_cases");
    let opts = ProofOptions::default_test_options();
    let words = BlockFormat::production()
        .with_memory(MemoryFormat::Words)
        .with_page_run(0);
    let dropped = || {
        let mut o = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
        o.drop_streamed_ops = true;
        o
    };
    let kept = || {
        let mut o = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
        o.drop_streamed_ops = false;
        o
    };
    let cases = [
        (
            "the split format",
            BlockFormat::production(),
            dropped(),
            Some(true),
        ),
        ("the word stream off", words, dropped(), Some(false)),
        ("the ops kept", words, kept(), Some(true)),
    ];
    for (what, format, options, stream) in cases {
        let with = |chunks: Option<WordChunks>| {
            prove_block_whir_with(
                &elf,
                &[],
                &opts,
                &format,
                &options,
                &Deviations {
                    word_stream: stream,
                    word_chunks: chunks,
                    ..Deviations::default()
                },
            )
            .map(|_| ())
        };
        let err = with(Some(WordChunks::Parts));
        assert!(
            matches!(&err, Err(crate::Error::Prover(m)) if m.contains("LAMBDA_VM_WORD_CHUNKS=part needs")),
            "{what}: {err:?}"
        );
        assert!(with(None).is_ok(), "{what}: the default proves");
    }
}
