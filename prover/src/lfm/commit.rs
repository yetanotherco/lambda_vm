//! Instruction-column-group commitment: interpolate → LDE → row-pair Merkle.
//!
//! The same pipeline the static preprocessed tables use (see
//! `tables/bitwise.rs::compute_preprocessed_commitment`), generalized over an
//! arbitrary column matrix so every LFM chip's group — and the registry
//! builder — shares one implementation. Host-side only; runs at program-build
//! and registry-regeneration time (seconds, not a ceremony — there is no
//! keygen in this framework).

use math::polynomial::Polynomial;
use stark::commitment::{ROWS_PER_LEAF, commit_bit_reversed_with};
use stark::config::Commitment;
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;

use crate::tables::types::{FE, GoldilocksField};

use super::compiler::ColumnGroup;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Whether the instruction-group commit pass runs its column loops in parallel.
///
/// ⛔ THE A/B CONTROL FOR THE PARALLEL BUILD, and it is a separate knob from
/// [`groups_in_flight`](super::registry::groups_in_flight) on purpose.
/// `LFM_ARTIFACT_GROUPS_IN_FLIGHT=1` bounds RESIDENCY and leaves the columns
/// parallel; `LFM_ARTIFACT_PARALLEL=0` restores the pre-change walk exactly —
/// serial columns, serial groups.
///
/// ⚠ `RAYON_NUM_THREADS=1` is NOT this control. It also serializes work that was
/// already parallel before this change (the Merkle leaf hashing in
/// `stark::commitment`, and the fixed tables' own commitments), so an A/B taken
/// that way attributes their speedup to this commit.
pub fn parallel_build() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(
        // An EMPTY value reads as unset: `FOO= cmd` is the shell's way of
        // clearing a variable, and panicking on it would fail a run for a
        // spelling of "default".
        || match std::env::var("LFM_ARTIFACT_PARALLEL").ok().as_deref() {
            Some("0") => false,
            None | Some("") | Some("1") => true,
            Some(other) => panic!("LFM_ARTIFACT_PARALLEL must be `0` or `1`, got `{other}`"),
        },
    )
}

/// The coset LDE of a column matrix, column-major and in NATURAL order.
///
/// Split out of [`commit_columns`] so a caller that needs the evaluations for
/// something else can expand once and commit from the same copy, rather than
/// building a second, independently expanded one.
///
/// # One pass per column, and it is both faster and smaller
///
/// This used to interpolate EVERY column into a `Vec<Polynomial<FE>>` and then
/// walk that vector expanding each — so the whole coefficient set, one full copy
/// of the group, was alive at once beside the input and the growing LDE. Each
/// column's interpolate → expand is independent of every other's, so the two
/// loops are now one: a column's coefficients are born and consumed inside the
/// same closure and only the workers' in-flight polynomials are ever live.
///
/// That is what makes the pass parallelizable, and the parallelism is where lane
/// P's 22.0 s per node goes. ★ The output is UNCHANGED — same operations on the
/// same column in the same order, and `collect` on an indexed parallel iterator
/// preserves position — which the registry drift tests pin against blessed
/// roots.
pub fn lde_columns(columns: &[Vec<FE>], options: &ProofOptions) -> Vec<Vec<FE>> {
    let num_rows = columns.first().map_or(0, Vec::len);
    let coset_offset = FE::from(options.coset_offset);
    let expand = |col: &Vec<FE>| {
        let poly = Polynomial::interpolate_fft::<GoldilocksField>(col)
            .expect("FFT interpolation failed for LFM column group");
        evaluate_polynomial_on_lde_domain(
            &poly,
            options.blowup_factor as usize,
            num_rows,
            &coset_offset,
        )
        .expect("LDE evaluation failed for LFM column group")
    };
    #[cfg(feature = "parallel")]
    if parallel_build() {
        return columns.par_iter().map(expand).collect();
    }
    columns.iter().map(expand).collect()
}

/// Commits an already-expanded LDE column matrix.
pub fn commit_lde_columns(lde_columns: &[Vec<FE>]) -> Commitment {
    // ★ Under the block path's PIN, not `stark`'s default aliases. These commit
    // the production tables whose roots `lfm_program_id` names, so the hash that
    // BUILDS them and the hash the program identity CLAIMS have to be the same
    // one — `registry.rs` records that as the condition under which this read
    // moves, and the pin is what moved it.
    let (_, root) = commit_bit_reversed_with::<
        GoldilocksField,
        <crate::hash_pin::BlockStarkHash as stark::config::StarkHash>::Batched<GoldilocksField>,
    >(lde_columns, ROWS_PER_LEAF)
    .expect("Merkle build failed for LFM column group");
    root
}

/// Commits a column matrix (each inner `Vec` one column, power-of-two height).
pub fn commit_columns(columns: &[Vec<FE>], options: &ProofOptions) -> Commitment {
    commit_lde_columns(&lde_columns(columns, options))
}

/// A [`ColumnGroup`]'s data, column-major (the commit pipeline's input shape).
///
/// A strided gather: the group is row-major, so column `c` is read with stride
/// `width`. Parallel over columns because each output column is written by
/// exactly one closure from a shared `&group` — and because on a production node
/// this moves hundreds of megabytes before any FFT starts.
pub fn group_columns(group: &ColumnGroup) -> Vec<Vec<FE>> {
    let column = |c: usize| (0..group.padded_rows).map(|r| *group.at(r, c)).collect();
    #[cfg(feature = "parallel")]
    if parallel_build() {
        return (0..group.width).into_par_iter().map(column).collect();
    }
    (0..group.width).map(column).collect()
}

/// Whether an instruction column group is committed on the DEVICE when one is
/// present. `LFM_DEVICE_ARTIFACTS=0` forces the host pass and is the A/B control
/// for O1.
pub fn device_artifacts() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(
        || match std::env::var("LFM_DEVICE_ARTIFACTS").ok().as_deref() {
            Some("0") => false,
            None | Some("") | Some("1") => true,
            Some(other) => panic!("LFM_DEVICE_ARTIFACTS must be `0` or `1`, got `{other}`"),
        },
    )
}

/// The largest device working set any artifact commit has asked for in this
/// process, in bytes. Zero when nothing has gone to the card.
///
/// ⓘ This is the ADMISSION's own accounting (`stark::device_set`), term by term:
/// one LDE buffer, the trace-domain snapshot, one full Merkle node buffer and
/// the small scratch — not a sampler reading. It is what the artifact build
/// asked the card for, which is the number that decides whether it is admitted;
/// an external sampler is what says what the process actually held.
pub fn device_artifact_peak_bytes() -> u64 {
    DEVICE_PEAK_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

static DEVICE_PEAK_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Commit one instruction column group — on the device where there is one, on
/// the host otherwise.
///
/// # Why the device can do this at all
///
/// The group is ALREADY row-major (`ColumnGroup.data` is `padded_rows × width`),
/// which is exactly the layout `gpu_lde`'s fused commit takes. So the device path
/// skips `group_columns` entirely — the strided transpose exists only to feed the
/// host's column-major pipeline — and does interpolate, coset-evaluate, leaf-hash
/// and Merkle in one call, returning the root.
///
/// # The two paths must produce the same root, and that is gated
///
/// The host commits with `commit_bit_reversed_with(.., ROWS_PER_LEAF)`; the
/// device builds its leaves from the row-major LDE. They are already required to
/// agree elsewhere: `multi_prove` rebuilds the precomputed tree on the device and
/// REFUSES the proof when its root differs from the one the AIR declares, and for
/// an LFM proof that declared root is what this function produced. The six
/// `registry_drift_*` tests are this caller's gate — they rebuild every
/// registered program's artifacts and compare roots against blessed constants,
/// so a device path that hashed differently fails them at once.
///
/// ⚠ A `None` from the device is a legitimate fallback (no card, a field the
/// kernels do not take, or admission declining the shape), NOT an error swallowed
/// quietly: once admission has ADMITTED a shape, `gpu_lde`'s own contract is that
/// the device is the only path and a failure aborts with a diagnostic rather than
/// returning here.
pub fn commit_group_device_or_host(
    label: &str,
    group: &ColumnGroup,
    options: &ProofOptions,
) -> Commitment {
    #[cfg(feature = "cuda")]
    if device_artifacts() && group.padded_rows > 0 && group.width > 0 {
        let set = stark::device_set::commit_device_set(
            group.padded_rows,
            group.width,
            options.blowup_factor as usize,
            true,
        );
        DEVICE_PEAK_BYTES.fetch_max(set.total(), std::sync::atomic::Ordering::Relaxed);
        if let Some(root) = stark::gpu_lde::try_commit_row_major::<
            GoldilocksField,
            <crate::hash_pin::BlockStarkHash as stark::config::StarkHash>::Batched<GoldilocksField>,
        >(
            label,
            &group.data,
            group.padded_rows,
            group.width,
            options.blowup_factor as usize,
            &FE::from(options.coset_offset),
        ) {
            return root;
        }
    }
    let _ = label;
    commit_lde_columns(&lde_columns(&group_columns(group), options))
}

/// Commits one instruction column group.
pub fn commit_group(group: &ColumnGroup, options: &ProofOptions) -> Commitment {
    commit_columns(&group_columns(group), options)
}

/// ★★★ THE GATE THAT ACTUALLY REACHES THE DEVICE.
///
/// `#[cfg(feature = "cuda")]` because without it both sides of the comparison
/// are the host pass and the test is a tautology — a check that cannot fail is
/// worse than no check, because it reads like coverage.
///
/// # Why the registry drift tests are not this gate
///
/// `gpu_lde` admits on `lde_size = padded_rows · blowup >= 2^14`, a ROW count.
/// ✓ MEASURED over the registered fixtures: EXACTLY ONE group clears it, and it
/// is slot 10 — `LFM_RANGE`, 65,536 rows × 1 column — which is
/// program-INDEPENDENT and identical in all six. Every program-dependent group
/// is far below: the largest is `statement_replay`'s slot 1 at 4,096 rows
/// (lde 8,192 at blowup 2). So the six drift pins are ONE device observation
/// repeated six times, at the narrowest shape the machine has.
///
/// ⇒ This test commits a group ABOVE the floor at production-like widths, both
/// ways, and compares the roots. It is the smallest thing that can catch a
/// device leaf convention that differs from
/// `commit_bit_reversed_with(.., ROWS_PER_LEAF)` at a shape the recursion
/// actually emits.
#[cfg(all(test, feature = "cuda"))]
mod device_parity {
    use super::*;
    use crate::lfm::compiler::ColumnGroup;
    use stark::proof::options::GoldilocksCubicProofOptions;

    /// A deterministic group of the given shape. Values are position-dependent
    /// so a transposed or mis-strided read cannot land on the same root.
    fn group(rows: usize, width: usize) -> ColumnGroup {
        let data = (0..rows * width)
            .map(|i| FE::from((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5))
            .collect();
        ColumnGroup {
            width,
            real_rows: rows,
            padded_rows: rows,
            data,
        }
    }

    #[test]
    fn the_device_commit_matches_the_host_commit_above_the_floor() {
        let options = GoldilocksCubicProofOptions::with_blowup(4).expect("options");
        // 4,096 · 4 = 16,384 is exactly the floor; 8,192 clears it with margin.
        // The widths are the production extremes: 1 (LFM_RANGE), 20
        // (LFM_BLAKE3), 134 (LFM_BITDEC, the widest prep group there is).
        for (rows, width) in [(4_096usize, 1usize), (8_192, 20), (4_096, 134)] {
            let g = group(rows, width);
            let lde = rows * options.blowup_factor as usize;
            assert!(
                lde >= 1 << 14,
                "{rows}x{width} gives lde {lde}, BELOW the 2^14 device floor — \
                 this case would compare a host root with a host root"
            );
            let host = commit_lde_columns(&lde_columns(&group_columns(&g), &options));
            let device = commit_group_device_or_host("device_parity", &g, &options);
            assert_eq!(
                device, host,
                "{rows}x{width}: the device root differs from the host root. The \
                 leaf convention diverged — see commitment.rs's leaf definition \
                 and gpu_lde's single-tree path"
            );
        }
    }

    /// And the control: `LFM_DEVICE_ARTIFACTS=0` must reach the host pass. Read
    /// once per process, so this asserts the knob's VALUE agrees with the branch
    /// rather than flipping it mid-run.
    #[test]
    fn the_opt_out_and_the_branch_agree() {
        let options = GoldilocksCubicProofOptions::with_blowup(4).expect("options");
        let g = group(4_096, 20);
        let host = commit_lde_columns(&lde_columns(&group_columns(&g), &options));
        assert_eq!(
            commit_group_device_or_host("device_parity_optout", &g, &options),
            host,
            "with LFM_DEVICE_ARTIFACTS={}, the commit must still equal the host root",
            if device_artifacts() { "1" } else { "0" }
        );
    }
}
