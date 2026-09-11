//! The accelerator chips' `max_rows` limits and the chunking that enforces them.

use crate::tables::trace_builder::Traces;
use crate::tables::{MaxRowsConfig, commit, ecdas, ecsm, hint, keccak, keccak_rnd, max_rows};
use crate::{Elf, Executor};

/// `main_cols + 3 × buses`, the cost model `max_rows` is derived from (see the
/// table on [`max_rows`]). Pinning it here makes a column or bus added to one of
/// these chips a failing test rather than a silently stale row limit.
#[test]
fn accelerator_max_rows_track_effective_width() {
    fn effective_width(main_cols: usize, buses: usize) -> usize {
        main_cols + 3 * buses
    }

    let widths = [
        (
            "KECCAK",
            effective_width(keccak::cols::NUM_COLUMNS, keccak::bus_interactions().len()),
            913,
        ),
        (
            "COMMIT",
            effective_width(commit::cols::NUM_COLUMNS, commit::bus_interactions().len()),
            73,
        ),
        (
            "KECCAK_RND",
            effective_width(
                keccak_rnd::cols::NUM_COLUMNS,
                keccak_rnd::bus_interactions().len(),
            ),
            4573,
        ),
        (
            "ECSM",
            effective_width(ecsm::cols::NUM_COLUMNS, ecsm::bus_interactions().len()),
            2404,
        ),
        (
            "ECDAS",
            effective_width(ecdas::cols::NUM_COLUMNS, ecdas::bus_interactions().len()),
            1685,
        ),
        (
            "HINT",
            effective_width(hint::cols::NUM_COLUMNS, hint::bus_interactions().len()),
            122,
        ),
    ];
    let drifted: Vec<String> = widths
        .iter()
        .filter(|(_, actual, documented)| actual != documented)
        .map(|(name, actual, documented)| format!("{name}: {documented} -> {actual}"))
        .collect();
    assert!(
        drifted.is_empty(),
        "effective width changed ({}) — revisit those max_rows and the table in tables::max_rows",
        drifted.join(", ")
    );
}

/// A keccak-heavy run splits KECCAK and KECCAK_RND into several chunks, each
/// within its limit, and KECCAK_RND splits on whole permutations.
#[test]
fn keccak_traces_split_into_bounded_chunks() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_keccak_multi");
    let elf = Elf::load(&elf_bytes).expect("load ELF");
    let executor = Executor::new(&elf, vec![]).expect("create executor");
    let logs = executor.run().expect("run program").logs;

    // One permutation per chunk.
    let limits = MaxRowsConfig {
        keccak: 1,
        keccak_rnd: keccak_rnd::ROUNDS_PER_OP,
        ..Default::default()
    };
    let mut traces = Traces::from_elf_and_logs_minimal(&elf, &logs, &limits, &[]).unwrap();

    assert!(
        traces.keccaks.len() > 1,
        "expected the keccak calls to span several chunks, got {}",
        traces.keccaks.len()
    );
    assert_eq!(
        traces.keccak_rnds.len(),
        traces.keccaks.len(),
        "one KECCAK_RND chunk per KECCAK chunk at one call per chunk"
    );
    for t in &traces.keccak_rnds {
        assert_eq!(
            t.num_rows(),
            keccak_rnd::ROUNDS_PER_OP.next_power_of_two(),
            "a chunk holds exactly one permutation's rounds, padded"
        );
    }

    // The split has to survive the buses too: KECCAK↔KECCAK_RND↔KECCAK_RC and the
    // memory argument are multiset arguments, so they balance across chunks or not
    // at all.
    assert!(
        crate::tests::prove_elfs_tests::prove_and_verify_vm_minimal(&elf, &mut traces),
        "a multi-chunk keccak trace must prove and verify"
    );
}

/// The default limits leave a small program in one chunk per accelerator, so the
/// chunking adds no sub-proofs to the common case.
#[test]
fn default_limits_keep_small_programs_single_chunk() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_keccak");
    let elf = Elf::load(&elf_bytes).expect("load ELF");
    let executor = Executor::new(&elf, vec![]).expect("create executor");
    let logs = executor.run().expect("run program").logs;

    let traces = Traces::from_elf_and_logs_minimal(&elf, &logs, &Default::default(), &[]).unwrap();
    let counts = traces.table_counts();
    assert_eq!(
        (
            counts.keccak,
            counts.keccak_rnd,
            counts.ecsm,
            counts.ecdas,
            counts.hint,
            counts.commit
        ),
        (1, 1, 1, 1, 1, 1)
    );
    assert!(traces.keccaks[0].num_rows() <= max_rows::KECCAK);
    assert!(traces.keccak_rnds[0].num_rows() <= max_rows::KECCAK_RND);
}
