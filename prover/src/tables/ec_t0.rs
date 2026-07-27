//! EC_T0: preprocessed lookup table of the lincomb2 NUMS correction constants.
//!
//! Exactly 256 rows, every one of them real — there is no padding. Row `j`
//! carries `LEN_M1 = j` and the blinding point for schedule length
//! `len = j + 1`:
//!
//! ```text
//!   row j  =  ( j, x(2^(j+1)·T₀), y(−2^(j+1)·T₀) )     j ∈ [0, 255]
//! ```
//!
//! The joint chain seeds its accumulator with the nothing-up-my-sleeve point
//! `T₀` (`ecsm::witness::t0`), so after `len` doublings the accumulator carries
//! a blind of `2^len·T₀`. The chain's last row strips it by adding
//! `−2^len·T₀`; it gets that addend by looking up this table on the `EcT0` bus.
//!
//! # Sign convention: the y column holds the NEGATED coordinate
//!
//! `Y` is `y(−2^len·T₀) = p − y(2^len·T₀)`, **not** `y(2^len·T₀)`. This is not
//! a preference — it is what `ecsm::witness::lincomb2_witness` emits. Its
//! `JointSel::Correction` row builds `neg_tpow` by negating the blind's `y` and
//! passes it as the row's *addend*, which `build_step` writes into the step's
//! `(x_g, y_g)`. Storing the negation therefore lets the consumer chip wire a
//! table lookup straight into its addend columns with no in-circuit modular
//! negation. `ecsm::tests::lincomb2_table_tests` asserts the match against real
//! witnesses; `ec_t0_tests::table_matches_lincomb2_witness_correction_row`
//! re-asserts it against the committed columns of this table.
//!
//! ## ⚠️ The witness carries the OPPOSITE convention in a second place
//!
//! `Lincomb2Witness::x_t0_pow` / `y_t0_pow` (`witness.rs:936-937` at the time of
//! writing) hold the **positive** `2^len·T₀` — not the negation this table
//! stores. Only `y` differs, since `x(−P) = x(P)`:
//!
//! ```text
//!   x_t0_pow == this table's X          (identical, no conversion)
//!   y_t0_pow == p − this table's Y      (a modular negation apart)
//! ```
//!
//! A consumer that binds its correction addend from this table does **not**
//! need `y_t0_pow` at all. Reading `y_t0_pow` where you meant `Y` is a silent
//! sign flip that still type-checks — do not mix them.
//!
//! # Why no consumer range check is needed
//!
//! The key column stores `len − 1`, and the bus receive re-adds the 1
//! (`LinearTerm::Constant(1)` in [`bus_interactions`]). So the tuple the
//! receiver publishes spans exactly `len ∈ [1, 256]`, with one row per value
//! and nothing else in the table. A send at `len = 0` or `len > 256` matches no
//! row, so the LogUp argument cannot balance and the proof is rejected — the
//! bound holds **by construction**.
//!
//! That range is also exactly the reachable one:
//! `len = max(u1.bits(), u2.bits())` (`witness.rs:830`) with both scalars
//! non-zero (`witness.rs:801-802` rejects `ScalarIsZero`) and `< N < 2^256`,
//! so `len ∈ [1, 256]` and `LEN_M1` fills a byte with no unreachable row.
//!
//! **Do not add a `len ≤ 256` check to the consumer chip** — it would be
//! redundant with the lookup itself. An earlier draft of this table keyed by
//! `len` directly over 257 rows padded to 512, which *did* need such a check,
//! because a send at `len > 256` would have resolved to a zeroed padding row
//! holding the off-curve point `(0, 0)`. The `LEN_M1` encoding removes both the
//! padding and the obligation.
//!
//! Follows the KECCAK_RC preprocessed-table pattern: precomputed columns are
//! committed via a static lookup table (with recompute as fallback for
//! `ProofOptions` not covered by the static table).

use std::sync::LazyLock;

use math::polynomial::Polynomial;
use stark::commitment::{ROWS_PER_LEAF, commit_bit_reversed};
use stark::config::Commitment;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::TraceTable;

use ecsm::lincomb2_table::neg_t0_pow2_points;
use ecsm::to_le_32;

use super::ecsm::point_coord_busvalues;
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};

// =========================================================================
// Column indices
// =========================================================================

pub mod cols {
    /// `len − 1`, the row index. A byte: `len ∈ [1, 256] ⇒ LEN_M1 ∈ [0, 255]`.
    /// The bus receive re-adds the 1 (see [`super::bus_interactions`]).
    pub const LEN_M1: usize = 0;
    /// `x(2^(LEN_M1+1)·T₀)`, 32 little-endian bytes.
    pub const X: usize = 1;
    pub const X_END: usize = X + 32; // = 33
    /// `y(−2^(LEN_M1+1)·T₀) = p − y(2^(LEN_M1+1)·T₀)`, 32 little-endian bytes.
    pub const Y: usize = X_END;
    pub const Y_END: usize = Y + 32; // = 65
    /// Multiplicity (how many times this row is looked up).
    pub const MU: usize = Y_END;

    pub const NUM_COLUMNS: usize = 66;
}

/// Number of precomputed columns (everything except MU).
pub const NUM_PRECOMPUTED_COLS: usize = 65;

/// Smallest schedule length the table defines.
pub const MIN_LEN: usize = 1;

/// Largest schedule length the table defines.
pub const MAX_LEN: usize = 256;

/// Rows in the trace. Every row is real — `MAX_LEN − MIN_LEN + 1` is already a
/// power of two, so the table needs no padding and a lookup can only ever
/// resolve to a genuine curve point.
pub const NUM_ROWS: usize = MAX_LEN - MIN_LEN + 1; // 256

/// Whether this table is preprocessed.
pub const fn is_preprocessed() -> bool {
    true
}

/// All `NUM_ROWS` precomputed rows, built once.
///
/// The constants come from [`neg_t0_pow2_points`] — a deterministic doubling
/// chain off the pinned `T₀`, so this is a constant table in every sense that
/// matters; it is merely computed rather than transcribed. That helper is
/// indexed by the exponent `i` (entry `i` = `−2^i·T₀`), so row `j` takes entry
/// `j + MIN_LEN`; entry 0 (`−T₀`, the chain's anchor) is unreachable as a
/// schedule length and has no row.
static ROWS: LazyLock<Vec<[u64; NUM_PRECOMPUTED_COLS]>> = LazyLock::new(|| {
    let points = neg_t0_pow2_points();
    assert!(
        points.len() > MAX_LEN,
        "EC_T0: constant chain too short ({} entries, need {} to reach len={MAX_LEN})",
        points.len(),
        MAX_LEN + 1,
    );
    (0..NUM_ROWS)
        .map(|idx| {
            let mut row = [0u64; NUM_PRECOMPUTED_COLS];
            row[cols::LEN_M1] = idx as u64;
            let pt = &points[idx + MIN_LEN];
            for (i, b) in to_le_32(&pt.x).iter().enumerate() {
                row[cols::X + i] = *b as u64;
            }
            for (i, b) in to_le_32(&pt.y).iter().enumerate() {
                row[cols::Y + i] = *b as u64;
            }
            row
        })
        .collect()
});

/// Generate one precomputed row: `[len − 1, x[0..32], y_neg[0..32]]`.
pub fn generate_row(idx: usize) -> [u64; NUM_PRECOMPUTED_COLS] {
    ROWS[idx]
}

// =========================================================================
// Preprocessed commitment
// =========================================================================

/// Returns the static EC_T0 preprocessed commitment for `blowup_factor`, or
/// `None` if no value is shipped for it. Values were generated by the
/// `compute_static_commitments` binary at the project's standard
/// `coset_offset = 3` (the value every in-tree `ProofOptions` constructor
/// pins) and pinned by the `ec_t0_static_matches_recompute_*` test so any
/// drift in the AIR, in `T₀`, or in the FFT pipeline is caught at test time.
/// The verifier reads these from its compiled binary — no input data is
/// trusted.
///
/// # Regenerating
///
/// Only regenerate these match arms after a *deliberate, reviewed* change to
/// the EC_T0 table layout, the AIR's preprocessed column count, the pinned
/// `T₀`, or the FFT / LDE / Merkle pipeline. Run:
///
/// ```text
/// cargo run --bin compute_static_commitments --release
/// ```
///
/// and paste the printed match arms over the ones below.
///
/// **If a drift test failed, do not regenerate first.** The drift tests exist
/// to force a human to ask "why did this change?" before the new bytes get
/// blessed. Re-pasting on a drift failure silently launders an unintended
/// change to the NUMS blinding constants into the verifier's compiled-in trust
/// anchor.
fn static_commitment(blowup_factor: u8) -> Option<Commitment> {
    match blowup_factor {
        2 => Some([
            0x8e, 0x6a, 0xa6, 0x11, 0x05, 0x57, 0x36, 0x2e, 0x32, 0xc8, 0x2f, 0xc4, 0x25, 0x1d,
            0xfd, 0x33, 0x5a, 0xa4, 0x93, 0x3a, 0x84, 0xde, 0xc5, 0x95, 0x23, 0xae, 0x7c, 0x66,
            0x3f, 0xd5, 0xb6, 0x4d,
        ]),
        4 => Some([
            0xe7, 0xdf, 0x42, 0x22, 0x59, 0xfa, 0xef, 0x01, 0x80, 0x34, 0xaa, 0x04, 0xfa, 0xcf,
            0x27, 0x64, 0x21, 0x4b, 0x0a, 0x2f, 0xd4, 0x74, 0x94, 0x65, 0x80, 0x6b, 0x16, 0xee,
            0x74, 0x47, 0x78, 0x1e,
        ]),
        8 => Some([
            0x4b, 0xc4, 0x57, 0x2d, 0xb0, 0xad, 0x11, 0x7f, 0xe8, 0xdd, 0x49, 0x39, 0xbc, 0x01,
            0x0e, 0x12, 0x5e, 0x53, 0xc7, 0x71, 0xb5, 0xdc, 0x99, 0x41, 0x0a, 0xf3, 0xec, 0x16,
            0x14, 0xd6, 0x2d, 0x1f,
        ]),
        _ => None,
    }
}

/// Exposed for the `compute_static_commitments` binary and the
/// drift-detection tests in `static_commitments_tests`. Production callers
/// should go through [`preprocessed_commitment`] so the static const-table
/// shortcut is used when applicable.
#[doc(hidden)]
pub fn compute_preprocessed_commitment(options: &ProofOptions) -> Commitment {
    // Generate precomputed columns
    let mut columns: Vec<Vec<FE>> = (0..NUM_PRECOMPUTED_COLS)
        .map(|_| Vec::with_capacity(NUM_ROWS))
        .collect();
    for idx in 0..NUM_ROWS {
        let row = generate_row(idx);
        for (col_idx, &value) in row.iter().enumerate() {
            columns[col_idx].push(FE::from(value));
        }
    }

    // Interpolate each column to a polynomial
    let polys: Vec<Polynomial<FE>> = columns
        .iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for ec_t0 column")
        })
        .collect();

    // Evaluate on LDE domain
    let blowup_factor = options.blowup_factor as usize;
    let coset_offset = FE::from(options.coset_offset);
    let lde_columns: Vec<Vec<FE>> = polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, NUM_ROWS, &coset_offset)
                .expect("LDE evaluation failed for ec_t0 polynomial")
        })
        .collect();

    let (_, root) = commit_bit_reversed(&lde_columns, ROWS_PER_LEAF)
        .expect("Failed to build Merkle tree for ec_t0 LDE");
    root
}

/// Returns the preprocessed commitment for the EC_T0 table.
///
/// Looks up `blowup_factor` via [`static_commitment`] when `coset_offset == 3`
/// (the value every in-tree `ProofOptions` constructor pins, and the offset
/// the static bytes were generated for); on miss — either a non-3 coset or a
/// `blowup_factor` outside `STATIC_BLOWUP_FACTORS` — recomputes from scratch.
#[inline]
pub fn preprocessed_commitment(options: &ProofOptions) -> Commitment {
    if options.coset_offset == 3
        && let Some(commitment) = static_commitment(options.blowup_factor)
    {
        return commitment;
    }
    log::warn!(
        "ec_t0 preprocessed commitment not static for (blowup={}, coset={}); \
         falling back to recompute. Add a match arm to `static_commitment` by running \
         `cargo run --bin compute_static_commitments --release`.",
        options.blowup_factor,
        options.coset_offset,
    );
    compute_preprocessed_commitment(options)
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generate the EC_T0 trace table.
///
/// All precomputed columns are filled; MU is initialized to zero and must be
/// updated via [`update_multiplicities`] once every correction-row lookup is
/// known.
pub fn generate_ec_t0_trace() -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(NUM_ROWS * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for idx in 0..NUM_ROWS {
        let row = generate_row(idx);
        for (col_idx, &value) in row.iter().enumerate() {
            table.set_u64(idx, col_idx, value);
        }
        // MU = 0 (will be updated later)
    }

    trace
}

/// Set MU from the `len` of every lincomb2 correction row in the proof.
///
/// Takes schedule lengths (`len`, not `len − 1`) and writes row `len − 1`, so
/// callers work in the same units as `Lincomb2Witness::len`. Each lincomb2
/// evaluation performs exactly one lookup, so `MU[len − 1]` is the number of
/// evaluations that used that schedule length.
///
/// Panics outside `[MIN_LEN, MAX_LEN]`. The lookup already enforces that range
/// on the proof side ("Why no consumer range check is needed" in the module
/// header), so this is a witness-side backstop: it turns a malformed schedule
/// into a loud failure here rather than an unbalanced bus much later.
pub fn update_multiplicities(
    trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    lens: impl IntoIterator<Item = u16>,
) {
    let mut counts = vec![0u64; NUM_ROWS];
    for len in lens {
        let len = len as usize;
        assert!(
            (MIN_LEN..=MAX_LEN).contains(&len),
            "EC_T0: correction lookup at len={len} outside [{MIN_LEN}, {MAX_LEN}]",
        );
        counts[len - MIN_LEN] += 1;
    }
    for (idx, count) in counts.iter().enumerate() {
        if *count != 0 {
            trace.main_table.set_u64(idx, cols::MU, *count);
        }
    }
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Single receiver on the EcT0 bus.
///
/// Format: `[len, x[0..32](Direct), y_neg[0..32](Direct)]` — 65 elements.
///
/// The key element is `LEN_M1 + 1`, so the consumer sends a plain `len` and
/// never has to know about the `−1` storage encoding. Because `LEN_M1` spans
/// `[0, 255]` across the 256 rows and there are no other rows, the published
/// keys are exactly `[1, 256]` — which is what makes the range bound hold by
/// construction rather than by a consumer-side check.
///
/// Coordinates ride the bus as 32 individual byte elements, the same shape
/// ECSM/ECDAS use for every point tuple ([`point_coord_busvalues`]), so the
/// consumer's addend columns inherit byte-ness from these committed constants
/// through plain tuple equality.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut values = vec![BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::LEN_M1,
        },
        LinearTerm::Constant(1),
    ])];
    values.extend(point_coord_busvalues(cols::X));
    values.extend(point_coord_busvalues(cols::Y));

    vec![BusInteraction::receiver(
        BusId::EcT0,
        Multiplicity::Column(cols::MU),
        values,
    )]
}
