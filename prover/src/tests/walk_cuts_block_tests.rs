//! ★ `LAMBDA_VM_WALK_CUTS` on real block proves: a block proved with every
//! cut of the walk on is the block proved with none, on three walkers in
//! record mode (the default) and on one, under the split memory format with
//! many windows and under the word memory format with a private input across
//! two pages: the same partition, heights, counts, output and group roots,
//! both proofs verify, and under the deterministic grind
//! (`LAMBDA_VM_DETERMINISTIC_GRIND`, read once a process: run alone) the
//! same bytes. The proving tests run where a block can be proved (the box's
//! lib run and CI).

use crate::block_whir::{
    BlockFormat, BlockWhirProof, Deviations, prove_block_whir_with, verify_block_whir,
};
use crate::tables::MaxRowsConfig;
use crate::tables::fused_memory::MemoryFormat;
use crate::tables::trace_builder::WalkCuts;
use crate::test_utils::asm_elf_bytes;
use stark::proof::options::ProofOptions;

fn bytes(proof: &BlockWhirProof) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .expect("serialize")
        .to_vec()
}

#[test]
fn the_walk_cuts_prove_the_same_block() {
    let opts = ProofOptions::default_test_options();
    let two_pages: Vec<u8> = (0..(crate::tables::page::DEFAULT_PAGE_SIZE as u32 + 300))
        .map(|i| (i * 7 + 3) as u8)
        .collect();
    for (what, program, input, format, window_log2) in [
        (
            "split, windows of 8 cycles",
            "all_instructions_64",
            Vec::new(),
            super::block_whir_tests::many_groups(),
            3,
        ),
        (
            "words, two private pages",
            "page_runs",
            two_pages,
            BlockFormat::production()
                .with_memory(MemoryFormat::Words)
                .with_page_run(0),
            9,
        ),
    ] {
        let elf = asm_elf_bytes(program);
        for walkers in [3, 1] {
            let proved = |cuts: WalkCuts| {
                let options =
                    super::block_whir_tests::streamed(MaxRowsConfig::small(), 16, window_log2);
                prove_block_whir_with(
                    &elf,
                    &input,
                    &opts,
                    &format,
                    &options,
                    &Deviations {
                        walkers: Some(walkers),
                        walk_cuts: Some(cuts),
                        ..Deviations::default()
                    },
                )
                .expect("the block proves")
                .0
            };
            let what = format!("{what}, {walkers} walkers");
            let off = proved(WalkCuts::default());
            let on = proved(WalkCuts::all());
            assert!(
                matches!(verify_block_whir(&on, &elf, &opts, &format), Ok(true)),
                "{what}: with the cuts verifies"
            );
            assert!(
                matches!(verify_block_whir(&off, &elf, &opts, &format), Ok(true)),
                "{what}: without verifies"
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
            }
        }
    }
}
