//! Tests for storage-mode auto-selection.

use crate::auto_storage::{
    CUBIC_EXT_BYTES, GOLDILOCKS_BYTES, KECCAK_NODE_BYTES, peak_bytes, select_storage_mode,
};
use crate::tables::cpu::bus_interactions as cpu_buses;
use crate::tables::cpu::cols::NUM_COLUMNS as CPU_COLS;
use crate::tables::trace_builder::TableLengths;
use stark::storage_mode::StorageMode;

const GB: u64 = 1_000_000_000;
/// Larger than the table count, so every table lands in the top-k and the
/// per-table delta in `peak_bytes_per_table_increment_is_exact` is purely
/// additive.
const ALL_TABLES: usize = 1_000;

fn empty_lengths() -> TableLengths {
    TableLengths::default()
}

/// Adding rows to a single chunked table must increase `peak_bytes` by
/// exactly the per-row contribution from the formula in the module doc.
/// Verifies the per-table breakdown is exact rather than averaged.
#[test]
fn peak_bytes_per_table_increment_is_exact() {
    let blowup = 2u8;
    let b = blowup as u64;

    let baseline = peak_bytes(&empty_lengths(), blowup, ALL_TABLES);

    let mut lengths = empty_lengths();
    lengths.cpu_padded_rows = 4;
    let bumped = peak_bytes(&lengths, blowup, ALL_TABLES);

    let cpu_main = CPU_COLS as u64;
    let cpu_aux = cpu_buses().len().div_ceil(2) as u64;
    let per_row_persistent = cpu_main * GOLDILOCKS_BYTES * (1 + b)
        + cpu_aux * CUBIC_EXT_BYTES * (1 + b)
        + 2 * b * KECCAK_NODE_BYTES   // main Merkle (1 tree)
        + 2 * b * KECCAK_NODE_BYTES; // aux Merkle
    let per_row_transient = b * CUBIC_EXT_BYTES   // constraint_evaluations
        + 2 * b * CUBIC_EXT_BYTES                  // composition LDE (2 parts, d=2)
        + b * KECCAK_NODE_BYTES                    // composition Merkle (PairKeccak)
        + b * CUBIC_EXT_BYTES                      // FRI evals (geometric ≈ 1)
        + b * KECCAK_NODE_BYTES; // FRI Merkle (geometric ≈ 1)
    let per_row_domain = (3 + 2 * b) * GOLDILOCKS_BYTES;

    // CPU adds 4 rows of persistent + transient (top-k by ALL_TABLES) +
    // its 4-row Domain entry (a fresh unique key not previously present).
    assert_eq!(
        bumped - baseline,
        4 * (per_row_persistent + per_row_transient + per_row_domain)
    );
}

/// Higher blowup_factor should produce a strictly larger estimate.
#[test]
fn peak_bytes_scales_with_blowup() {
    let lengths = empty_lengths();
    let two = peak_bytes(&lengths, 2, ALL_TABLES);
    let four = peak_bytes(&lengths, 4, ALL_TABLES);
    let eight = peak_bytes(&lengths, 8, ALL_TABLES);
    assert!(two < four);
    assert!(four < eight);
}

/// Lower table_parallelism caps the transient sum to fewer tables, so the
/// estimate must be monotone in `k`.
#[test]
fn peak_bytes_monotone_in_table_parallelism() {
    let lengths = empty_lengths();
    let k1 = peak_bytes(&lengths, 2, 1);
    let k4 = peak_bytes(&lengths, 2, 4);
    let k_all = peak_bytes(&lengths, 2, ALL_TABLES);
    assert!(k1 < k4);
    assert!(k4 <= k_all);
}

#[test]
fn select_ram_when_estimate_below_threshold() {
    // 10 GB estimated, 32 GB available → threshold 28.8 GB → Ram.
    let mode = select_storage_mode(10 * GB, Some(32 * GB));
    assert_eq!(mode, StorageMode::Ram);
}

#[test]
fn select_disk_when_estimate_exceeds_threshold() {
    // 30 GB estimated, 32 GB available → threshold 28.8 GB → Disk.
    let mode = select_storage_mode(30 * GB, Some(32 * GB));
    assert_eq!(mode, StorageMode::Disk);
}

#[test]
fn unknown_available_defaults_to_disk() {
    let mode = select_storage_mode(peak_bytes(&empty_lengths(), 2, ALL_TABLES), None);
    assert_eq!(mode, StorageMode::Disk);
}

/// A shape with one PAGE table — everything the monolithic path proves today.
/// The top-k sum has saturated well before the table count, so the estimate is
/// insensitive to `k` in that range: this is why raising the *scheduler's* `k`
/// to `num_airs` does not move the storage decision on a normal workload.
#[test]
fn peak_bytes_is_k_saturated_on_single_page_shapes() {
    let mut lengths = empty_lengths();
    lengths.cpu_padded_rows = 1 << 20;
    lengths.memw_padded_rows = 1 << 20;
    lengths.decode_rows = 1 << 16;
    lengths.unique_page_count = 1;

    let bounded = peak_bytes(&lengths, 2, 12);
    let unbounded = peak_bytes(&lengths, 2, ALL_TABLES);
    assert!(
        unbounded * 100 <= bounded * 101,
        "estimate moved {bounded} -> {unbounded} on a one-page shape"
    );
}

/// …and why `decide` must not simply be handed the scheduler's `k`. PAGE tables
/// are all the same size, so once there are many of them the top-k truncation
/// is doing real work: summing every table's transients inflates the estimate
/// by >20 % here, which spills proofs to disk that fit in RAM.
#[test]
fn unbounded_k_inflates_peak_bytes_on_many_page_shapes() {
    let mut lengths = empty_lengths();
    lengths.cpu_padded_rows = 1 << 20;
    lengths.memw_padded_rows = 1 << 20;
    lengths.decode_rows = 1 << 16;
    lengths.unique_page_count = 128;

    let bounded = peak_bytes(&lengths, 2, 21);
    let unbounded = peak_bytes(&lengths, 2, ALL_TABLES);
    assert!(
        unbounded * 10 > bounded * 12,
        "expected >20 % inflation, got {bounded} -> {unbounded}"
    );
}
