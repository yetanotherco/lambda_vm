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

    fn words() -> BlockFormat {
        BlockFormat::production()
            .with_memory(MemoryFormat::Words)
            .with_page_run(0)
    }

    macro_rules! schedules {
        ($name:ident, $program:literal, $input:expr, $format:expr, $window_log2:literal) => {
            mod $name {
                use super::*;

                fn compare(walkers: usize) {
                    let _ecrec = ($program == "test_ecrec_stream")
                        .then(crate::tables::trace_builder::ecrec_forge_free);
                    assert_walk_cuts_identity($program, $input, $format, $window_log2, walkers);
                }

                #[test]
                fn one_walker() {
                    compare(1);
                }

                #[test]
                fn three_walkers() {
                    compare(3);
                }
            }
        };
    }

    schedules!(
        split_windows,
        "test_commit_4",
        &[],
        crate::tests::block_whir_tests::many_groups(),
        3
    );
    schedules!(
        words_with_private_pages,
        "test_private_input_multipage",
        &crate::tests::word_memory_block_tests::two_pages_of_input(),
        words(),
        2
    );
    // A single valid recovery still hands off ECDAS witnesses. Four-cycle
    // windows spread the smaller guest across all three walkers.
    schedules!(words_with_ecrec, "test_ecrec_stream", &[], words(), 2);
}

fn assert_walk_cuts_identity(
    program: &str,
    input: &[u8],
    format: BlockFormat,
    window_log2: usize,
    walkers: usize,
) {
    // Nextest gives each case its own process. The ordinary libtest runner
    // shares the finish counters, so take one counted turn there.
    static COUNTED_PROOFS: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _turn = COUNTED_PROOFS.lock().unwrap_or_else(|err| err.into_inner());
    let opts = ProofOptions::default_test_options();
    let elf = asm_elf_bytes(program);
    // Count only these proofs, including when other tests run every finish part.
    let none = (WalkCuts::none(), FinishEarly::none().counted());
    let both = (WalkCuts::all(), FinishEarly::all().counted());
    let arms: &[_] = if walkers == 1 {
        &[both]
    } else {
        &[(WalkCuts::all(), none.1), (none.0, both.1), both]
    };
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
        if program == "test_ecrec_stream" {
            assert!(proof.table_counts.ecrec > 0 && proof.table_counts.ecdas > 0);
        }
        proof
    };
    let off = proved(none);
    assert!(
        matches!(verify_block_whir(&off, &elf, &opts, &format), Ok(true)),
        "{program}, {walkers} walkers: without verifies"
    );
    for &arm in arms {
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
        if program == "test_ecrec_stream" && arm.1.ecdas_count {
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
