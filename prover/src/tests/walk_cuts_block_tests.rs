//! ★ `LAMBDA_VM_WALK_CUTS` and `LAMBDA_VM_FINISH_EARLY` on real block
//! proves: a block proved with every cut of the walk on, with every finish
//! part on, and with both, is the block proved with none, on three walkers in
//! record mode (the default) and on one, under the split memory format with
//! many windows, under the word memory format with a private input across
//! two pages, and on ECREC calls whose witnesses the walk hands off (the
//! finish taking ECDAS's lookups counted beside them): the same partition,
//! heights, counts, output and group roots,
//! every proof verifies, and under the deterministic grind
//! (`LAMBDA_VM_DETERMINISTIC_GRIND`, read once a process: run alone) the
//! same bytes. The proving tests run where a block can be proved (the box's
//! lib run and CI).

use crate::block_whir::{
    BlockFormat, BlockWhirProof, Deviations, prove_block_whir_with, verify_block_whir,
};
use crate::tables::MaxRowsConfig;
use crate::tables::fused_memory::MemoryFormat;
use crate::tables::trace_builder::{FinishEarly, WalkCuts};
use crate::test_utils::asm_elf_bytes;
use stark::proof::options::ProofOptions;

fn bytes(proof: &BlockWhirProof) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .expect("serialize")
        .to_vec()
}

mod the_walk_cuts_prove_the_same_block {
    use super::*;

    #[test]
    fn split_windows() {
        let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
        assert_walk_cuts_identity(
            "test_commit_4",
            &[],
            super::super::block_whir_tests::many_groups(),
            3,
        );
    }

    #[test]
    fn words_with_private_pages() {
        let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
        assert_walk_cuts_identity(
            "test_private_input_multipage",
            &super::super::word_memory_block_tests::two_pages_of_input(),
            words(),
            2,
        );
    }

    #[test]
    fn words_with_ecrec() {
        let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
        assert_walk_cuts_identity("test_ecrec", &[], words(), 4);
    }

    fn words() -> BlockFormat {
        BlockFormat::production()
            .with_memory(MemoryFormat::Words)
            .with_page_run(0)
    }
}

fn assert_walk_cuts_identity(program: &str, input: &[u8], format: BlockFormat, window_log2: usize) {
    let opts = ProofOptions::default_test_options();
    let elf = asm_elf_bytes(program);
    // Count only these proofs, including when other tests run every finish part.
    let none = (WalkCuts::none(), FinishEarly::none().counted());
    for (walkers, arms) in [
        (
            3,
            vec![
                (WalkCuts::all(), FinishEarly::none().counted()),
                (WalkCuts::none(), FinishEarly::all().counted()),
                (WalkCuts::all(), FinishEarly::all().counted()),
            ],
        ),
        (1, vec![(WalkCuts::all(), FinishEarly::all().counted())]),
    ] {
        let proved = |(cuts, early): (WalkCuts, FinishEarly)| {
            // The rest streamed under its gate (a layout thread), as the
            // prove streams it: BITWISE's job is the sliding finish's.
            let mut options =
                super::block_whir_tests::streamed(MaxRowsConfig::small(), 16, window_log2);
            options.layout_workers = 3;
            let (proof, stamps) = prove_block_whir_with(
                &elf,
                input,
                &opts,
                &format,
                &options,
                &Deviations {
                    walkers: Some(walkers),
                    walk_cuts: Some(cuts),
                    finish_early: Some(early),
                    ..Deviations::default()
                },
            )
            .expect("the block proves");
            assert!(
                stamps.executor.windows >= walkers,
                "{program}: {} windows for {walkers} walkers",
                stamps.executor.windows
            );
            if !input.is_empty() {
                assert_eq!(proof.num_private_input_pages, 2);
            }
            proof
        };
        let off = proved(none);
        assert!(
            matches!(verify_block_whir(&off, &elf, &opts, &format), Ok(true)),
            "{program}, {walkers} walkers: without verifies"
        );
        for arm in arms {
            let what = format!(
                "{program}, {walkers} walkers, cuts {}, finish {}",
                arm.0.describe(),
                arm.1.describe()
            );
            let tails = crate::tables::trace_builder::finish_early::bitwise_tails();
            let ecdas = crate::tables::trace_builder::finish_early::ecdas_counts();
            let on = proved(arm);
            assert_eq!(
                crate::tables::trace_builder::finish_early::bitwise_tails() > tails,
                arm.1.bitwise_tail,
                "{what}: BITWISE's job split"
            );
            if program == "test_ecrec" && arm.1.ecdas_count {
                assert!(
                    crate::tables::trace_builder::finish_early::ecdas_counts() > ecdas,
                    "{what}: ECDAS's lookups counted beside the witnesses"
                );
            }
            assert!(
                matches!(verify_block_whir(&on, &elf, &opts, &format), Ok(true)),
                "{what}: verifies"
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
