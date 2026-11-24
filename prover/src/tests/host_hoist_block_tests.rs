//! ★ `LAMBDA_VM_HOST_HOIST` on real block proves: a block proved with every
//! hoisted part on is the block proved with none, under the word memory
//! format (pages one table a page and in runs) and the split format, with a
//! private input across two pages: the same partition, heights, counts,
//! output and group roots, the proof verifies (in record mode, the default,
//! the hoisted final state built once), and under the deterministic
//! grind (`LAMBDA_VM_DETERMINISTIC_GRIND`, read once a process: run alone)
//! the same bytes. The proving tests run where a block can be proved (the
//! box's lib run and CI).

use crate::block_whir::{
    BlockFormat, BlockWhirProof, Deviations, prove_block_whir_with, verify_block_whir,
};
use crate::tables::MaxRowsConfig;
use crate::tables::fused_memory::MemoryFormat;
use crate::tables::trace_builder::HostHoist;
use crate::test_utils::asm_elf_bytes;
use stark::proof::options::ProofOptions;

/// Two pages' worth of private input, ending inside the second.
fn two_pages_of_input() -> Vec<u8> {
    (0..(crate::tables::page::DEFAULT_PAGE_SIZE as u32 + 300))
        .map(|i| (i * 7 + 3) as u8)
        .collect()
}

/// `page_runs` (runs of zero pages, two private-input pages) proved by the
/// prove's own walk under `format` with `hoist`.
fn proved(format: &BlockFormat, hoist: HostHoist) -> (Vec<u8>, BlockWhirProof) {
    let elf = asm_elf_bytes("page_runs");
    let options = super::block_whir_tests::streamed(MaxRowsConfig::small(), 3, 9);
    let (proof, _) = prove_block_whir_with(
        &elf,
        &two_pages_of_input(),
        &ProofOptions::default_test_options(),
        format,
        &options,
        &Deviations {
            host_hoist: Some(hoist),
            ..Deviations::default()
        },
    )
    .expect("the block proves");
    (elf, proof)
}

fn bytes(proof: &BlockWhirProof) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .expect("serialize")
        .to_vec()
}

#[test]
fn the_host_hoist_proves_the_same_block() {
    use crate::tables::trace_builder::host_hoist::calls;
    let _turn = calls::turn();
    // Record mode unless the walkers or the history are turned off.
    let record = std::env::var("LAMBDA_VM_BLOCK_WALKERS").map_or(true, |v| v.trim() != "0")
        && std::env::var("LAMBDA_VM_EXEC_HISTORY").map_or(true, |v| v.trim() != "0");
    let opts = ProofOptions::default_test_options();
    for (what, format) in [
        (
            "words",
            BlockFormat::production()
                .with_memory(MemoryFormat::Words)
                .with_page_run(0),
        ),
        (
            "words, runs of 16 pages",
            BlockFormat::production()
                .with_memory(MemoryFormat::Words)
                .with_page_run(4),
        ),
        (
            "split",
            BlockFormat::production()
                .with_memory(MemoryFormat::Split)
                .with_page_run(0),
        ),
    ] {
        let _ = calls::take_states();
        let (elf, off) = proved(&format, HostHoist::default());
        assert_eq!(
            calls::take_states(),
            0,
            "{what}: off never hoists the state"
        );
        let (_, on) = proved(&format, HostHoist::all());
        assert_eq!(
            calls::take_states(),
            usize::from(record),
            "{what}: the state hoisted once in record mode"
        );
        assert!(
            matches!(verify_block_whir(&on, &elf, &opts, &format), Ok(true)),
            "{what}: hoisted verifies"
        );
        assert!(
            matches!(verify_block_whir(&off, &elf, &opts, &format), Ok(true)),
            "{what}: not hoisted verifies"
        );
        assert_eq!(on.num_private_input_pages, 2, "{what}: private pages");
        assert_eq!(on.groups, off.groups, "{what}: the partition");
        assert_eq!(on.table_num_vars, off.table_num_vars, "{what}: the heights");
        assert_eq!(
            format!("{:?}", on.table_counts),
            format!("{:?}", off.table_counts),
            "{what}: the counts"
        );
        assert_eq!(on.word_memory, off.word_memory, "{what}: the word tables");
        assert_eq!(
            format!("{:?}", on.runtime_page_ranges),
            format!("{:?}", off.runtime_page_ranges),
            "{what}: the runtime pages"
        );
        assert_eq!(on.public_output, off.public_output, "{what}: the output");
        assert_eq!(on.proof.roots, off.proof.roots, "{what}: the group roots");
        if crypto::grinding::deterministic() {
            assert_eq!(bytes(&on), bytes(&off), "{what}: the proof's bytes");
        }
    }
}
