//! Tests for the EC_T0 preprocessed table (`−2^len·T₀`, keyed by `len − 1`).
//!
//! The table's constants come from `ecsm::lincomb2_table`, which doubles the
//! pinned `T₀` in k256 projective coordinates. Everything checked here is
//! recomputed *independently* — an affine `Fp` doubling written in this file,
//! or the byte output of `lincomb2_witness` — so a bug in the k256 path cannot
//! validate itself.

use num_bigint::BigUint;

use ecsm::curve::AffinePoint;
use ecsm::witness::{JointSel, lincomb2_witness, t0};
use ecsm::{B, p, to_le_32};

use stark::proof::options::GoldilocksCubicProofOptions;

use stark::lookup::{BusValue, LinearTerm};

use crate::tables::BusId;
use crate::tables::ec_t0::{
    self, MAX_LEN, MIN_LEN, NUM_PRECOMPUTED_COLS, NUM_ROWS, cols, generate_row,
};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use stark::trace::TraceTable;

// ---- an independent affine Fp group law (no k256, no BigInt witness code) ----

fn inv(a: &BigUint, modulus: &BigUint) -> BigUint {
    a.modpow(&(modulus - 2u32), modulus)
}

/// `2·pt` in affine coordinates: `λ = 3x²/(2y)`, `x₃ = λ² − 2x`,
/// `y₃ = λ(x − x₃) − y`.
fn double(pt: &AffinePoint, modulus: &BigUint) -> AffinePoint {
    let lam = (3u32 * &pt.x * &pt.x % modulus) * inv(&(2u32 * &pt.y % modulus), modulus) % modulus;
    let x3 = (&lam * &lam + modulus * 2u32 - 2u32 * &pt.x) % modulus;
    let y3 = (&lam * ((&pt.x + modulus - &x3) % modulus) + modulus - &pt.y) % modulus;
    AffinePoint { x: x3, y: y3 }
}

/// Reads a 32-byte little-endian coordinate out of one trace row.
fn coord(
    trace: &TraceTable<GoldilocksField, GoldilocksExtension>,
    row: usize,
    col: usize,
) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        let v = trace.main_table.get(row, col + i).canonical();
        assert!(v < 256, "row {row} col {}: not a byte ({v})", col + i);
        *byte = v as u8;
    }
    out
}

fn cell(trace: &TraceTable<GoldilocksField, GoldilocksExtension>, row: usize, col: usize) -> u64 {
    trace.main_table.get(row, col).canonical()
}

// ---- layout ----

#[test]
fn layout_is_as_documented() {
    assert_eq!(cols::LEN_M1, 0);
    assert_eq!(cols::X, 1);
    assert_eq!(cols::Y, 33);
    assert_eq!(cols::MU, 65);
    assert_eq!(cols::NUM_COLUMNS, 66);
    assert_eq!(
        NUM_PRECOMPUTED_COLS, 65,
        "everything but MU is preprocessed"
    );
    assert_eq!(NUM_PRECOMPUTED_COLS, cols::MU, "MU is the only main column");
    assert_eq!(
        MIN_LEN, 1,
        "len = 0 is unreachable (both scalars are non-zero)"
    );
    assert_eq!(MAX_LEN, 256, "len <= 256 (both scalars are < N < 2^256)");
    assert_eq!(NUM_ROWS, 256, "one row per reachable len, no padding");
    assert_eq!(
        NUM_ROWS,
        MAX_LEN - MIN_LEN + 1,
        "every row is real: LEN_M1 fills a byte exactly",
    );
    assert!(ec_t0::is_preprocessed());
}

// ---- the acceptance gate: recompute the whole table from t0() ----

/// Recomputes every row from `ecsm::witness::t0()` by repeated affine doubling
/// (this file's `double`, not the generator's k256 chain), negates each `y`, and
/// compares byte-for-byte against the committed trace columns. Row `j` holds the
/// blind for `len = j + MIN_LEN`, so the chain is advanced `MIN_LEN` times
/// before the first row.
#[test]
fn trace_rows_recompute_from_t0_by_doubling() {
    let trace = ec_t0::generate_ec_t0_trace();
    let modulus = p();
    let mut pt = t0();
    for _ in 0..MIN_LEN {
        pt = double(&pt, &modulus);
    }

    for row in 0..NUM_ROWS {
        let len = row + MIN_LEN;
        assert_eq!(
            cell(&trace, row, cols::LEN_M1),
            row as u64,
            "row {row}: LEN_M1 key must equal len - 1",
        );
        assert_eq!(
            coord(&trace, row, cols::X),
            to_le_32(&pt.x),
            "row {row}: X != x(2^{len}·T0)",
        );
        assert_eq!(
            coord(&trace, row, cols::Y),
            to_le_32(&(&modulus - &pt.y)),
            "row {row}: Y != y(-2^{len}·T0); the table must store the NEGATED point",
        );
        pt = double(&pt, &modulus);
    }
}

/// Every entry is a canonical on-curve point: `y² ≡ x³ + b (mod p)` with both
/// coordinates in `[0, p)` and `y ≠ 0`. There are no padding rows to skip —
/// this covers the whole table.
#[test]
fn every_entry_is_on_curve_and_canonical() {
    let trace = ec_t0::generate_ec_t0_trace();
    let modulus = p();

    for row in 0..NUM_ROWS {
        let x = BigUint::from_bytes_le(&coord(&trace, row, cols::X));
        let y = BigUint::from_bytes_le(&coord(&trace, row, cols::Y));
        assert!(x < modulus, "row {row}: x >= p");
        assert!(y < modulus, "row {row}: y >= p");
        assert!(y > BigUint::ZERO, "row {row}: y = 0");
        let lhs = &y * &y % &modulus;
        let rhs = (&x * &x % &modulus * &x + B) % &modulus;
        assert_eq!(lhs, rhs, "row {row} is not on the secp256k1 curve");
    }
}

/// Row 0 holds `−2·T₀`, i.e. the blind for the smallest reachable `len = 1` —
/// NOT `−T₀`. The `len = 0` anchor of the doubling chain is deliberately absent:
/// both scalars are non-zero, so `len = 0` cannot occur.
#[test]
fn row_zero_is_the_blind_for_min_len_not_t0_itself() {
    let trace = ec_t0::generate_ec_t0_trace();
    let modulus = p();
    let t = t0();

    let expected = double(&t, &modulus);
    assert_eq!(coord(&trace, 0, cols::X), to_le_32(&expected.x));
    assert_eq!(
        coord(&trace, 0, cols::Y),
        to_le_32(&(&modulus - &expected.y))
    );

    // And T0 itself is not in the table under any key.
    let t0_x = to_le_32(&t.x);
    for row in 0..NUM_ROWS {
        assert_ne!(
            coord(&trace, row, cols::X),
            t0_x,
            "row {row} holds T0 itself; len = 0 must have no row",
        );
    }
}

/// No two rows share a lookup key, and the published keys are exactly the
/// reachable `len` range. This is what makes the range bound hold by
/// construction: a send at `len = 0` or `len > MAX_LEN` matches no row.
#[test]
fn keys_cover_the_reachable_len_range_exactly_and_uniquely() {
    let trace = ec_t0::generate_ec_t0_trace();
    let mut keys: Vec<u64> = (0..NUM_ROWS)
        .map(|row| cell(&trace, row, cols::LEN_M1) + 1)
        .collect();
    keys.sort_unstable();
    keys.dedup();
    assert_eq!(keys.len(), NUM_ROWS, "duplicate lookup keys");
    assert_eq!(*keys.first().unwrap(), MIN_LEN as u64);
    assert_eq!(*keys.last().unwrap(), MAX_LEN as u64);
    assert!(
        (0..NUM_ROWS).all(|row| cell(&trace, row, cols::LEN_M1) < 256),
        "LEN_M1 must fit in a byte",
    );
}

// ---- multiplicities ----

#[test]
fn mu_starts_at_zero_everywhere() {
    let trace = ec_t0::generate_ec_t0_trace();
    for row in 0..NUM_ROWS {
        assert_eq!(cell(&trace, row, cols::MU), 0, "row {row}: MU");
    }
}

#[test]
fn update_multiplicities_counts_lookups_per_len() {
    let mut trace = ec_t0::generate_ec_t0_trace();
    ec_t0::update_multiplicities(&mut trace, [1u16, 256, 7, 256, 256]);

    // Callers pass `len`; row `len - 1` is written.
    assert_eq!(cell(&trace, 0, cols::MU), 1, "len = 1 -> row 0");
    assert_eq!(cell(&trace, 6, cols::MU), 1, "len = 7 -> row 6");
    assert_eq!(cell(&trace, 255, cols::MU), 3, "len = 256 -> row 255");
    for row in [1usize, 2, 5, 7, 254] {
        assert_eq!(
            cell(&trace, row, cols::MU),
            0,
            "row {row} was not looked up"
        );
    }
}

#[test]
#[should_panic(expected = "outside [1, 256]")]
fn update_multiplicities_rejects_len_above_max() {
    let mut trace = ec_t0::generate_ec_t0_trace();
    ec_t0::update_multiplicities(&mut trace, [257u16]);
}

#[test]
#[should_panic(expected = "outside [1, 256]")]
fn update_multiplicities_rejects_len_zero() {
    let mut trace = ec_t0::generate_ec_t0_trace();
    ec_t0::update_multiplicities(&mut trace, [0u16]);
}

// ---- the sign convention, at trace level ----

/// The load-bearing convention test: `lincomb2_witness`'s correction row takes
/// its addend from `(x_g, y_g)`, and those bytes must be exactly the committed
/// EC_T0 row `len - MIN_LEN`. If the table ever flips to storing `+2^len·T₀`,
/// or the `LEN_M1` offset goes out of step with the constants, this is what
/// fails.
#[test]
fn table_matches_lincomb2_witness_correction_row() {
    let trace = ec_t0::generate_ec_t0_trace();

    let gx = BigUint::parse_bytes(
        b"79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
        16,
    )
    .expect("generator x");
    let g = AffinePoint {
        y: ecsm::curve::recover_y_canonical(&gx).expect("generator y"),
        x: gx,
    };
    // 2·G — a distinct x, so the `P1 + P2` precompute is a genuine chord.
    let p2 = double(&g, &p());

    // Scalars spanning a range of `len` values, including the 256 top row.
    let scalars = [
        BigUint::from(1u8),
        BigUint::from(0xFFFFu32),
        BigUint::from(1u8) << 200,
        ecsm::n() - 1u32,
    ];

    let mut saw_max_len = false;
    for u1 in &scalars {
        for u2 in &scalars {
            let w = lincomb2_witness(&to_le_32(u1), &to_le_32(u2), &g, &p2).expect("witness");
            let len = w.len as usize;
            assert!((1..=MAX_LEN).contains(&len), "len {len} out of range");
            saw_max_len |= len == MAX_LEN;

            let corr = w
                .steps
                .iter()
                .find(|s| matches!(s.sel, JointSel::Correction))
                .expect("correction row");

            let row = len - MIN_LEN;
            assert_eq!(
                cell(&trace, row, cols::LEN_M1) + 1,
                len as u64,
                "row {row}: bus key must reconstruct len {len}",
            );
            assert_eq!(
                corr.step.x_g,
                coord(&trace, row, cols::X),
                "len {len}: correction addend x != EC_T0 row x",
            );
            assert_eq!(
                corr.step.y_g,
                coord(&trace, row, cols::Y),
                "len {len}: correction addend y != EC_T0 row y (sign convention broken)",
            );

            // The witness's `*_t0_pow` fields carry the OPPOSITE convention:
            // the positive blind. x matches the table directly; y is its
            // negation. Pinned here so a future reader cannot confuse them.
            assert_eq!(
                w.x_t0_pow,
                coord(&trace, row, cols::X),
                "len {len}: x_t0_pow != EC_T0 row x",
            );
            assert_eq!(
                BigUint::from_bytes_le(&w.y_t0_pow)
                    + BigUint::from_bytes_le(&coord(&trace, row, cols::Y)),
                p(),
                "len {len}: y_t0_pow must be the negation of EC_T0 row y",
            );
        }
    }
    assert!(saw_max_len, "len = 256 was never exercised");
}

// ---- commitment ----

/// The same constants every time: `generate_row` and the generated trace agree,
/// and repeated calls are byte-identical (the `LazyLock` cache cannot drift).
#[test]
fn generated_trace_matches_generate_row_and_is_reproducible() {
    let a = ec_t0::generate_ec_t0_trace();
    let b = ec_t0::generate_ec_t0_trace();
    for row in 0..NUM_ROWS {
        for (col, value) in generate_row(row).iter().enumerate() {
            assert_eq!(cell(&a, row, col), *value, "row {row} col {col}");
            assert_eq!(
                a.main_table.get(row, col),
                b.main_table.get(row, col),
                "row {row} col {col} differs between instantiations",
            );
        }
    }
}

/// The commitment is stable across runs and instantiations, and equals the
/// static bytes compiled into the verifier for every shipped blowup.
#[test]
fn commitment_is_stable_and_matches_the_shipped_static_bytes() {
    for &blowup in crate::tables::STATIC_BLOWUP_FACTORS {
        let options = GoldilocksCubicProofOptions::with_blowup(blowup).expect("valid blowup");
        let first = ec_t0::compute_preprocessed_commitment(&options);
        let second = ec_t0::compute_preprocessed_commitment(&options);
        assert_eq!(first, second, "blowup={blowup}: commitment is not stable");
        assert_eq!(
            first,
            ec_t0::preprocessed_commitment(&options),
            "blowup={blowup}: shipped static commitment != recompute",
        );
    }
}

/// Different blowups must not collide (a smoke check that the commitment
/// actually depends on the LDE parameters).
#[test]
fn commitment_differs_across_blowups() {
    let opts_2 = GoldilocksCubicProofOptions::with_blowup(2).expect("valid blowup");
    let opts_4 = GoldilocksCubicProofOptions::with_blowup(4).expect("valid blowup");
    assert_ne!(
        ec_t0::preprocessed_commitment(&opts_2),
        ec_t0::preprocessed_commitment(&opts_4),
    );
}

// ---- bus ----

/// One receiver on `EcT0`, keyed by `len`, carrying both coordinates as 32
/// individual byte elements (the shape ECSM/ECDAS use for every point tuple).
#[test]
fn bus_interaction_shape() {
    let buses = ec_t0::bus_interactions();
    assert_eq!(buses.len(), 1, "EC_T0: exactly one receiver");
    let bus = &buses[0];
    assert_eq!(bus.bus_id, BusId::EcT0 as u64);
    assert_eq!(bus.values.len(), 1 + 32 + 32, "len + x[32] + y[32]");

    // The key element must be `LEN_M1 + 1`, not a bare column read — that
    // `+1` is what confines the published keys to [1, 256].
    match &bus.values[0] {
        BusValue::Linear(terms) => {
            assert!(
                terms.iter().any(|t| matches!(
                    t,
                    LinearTerm::Column {
                        coefficient: 1,
                        column
                    } if *column == cols::LEN_M1
                )),
                "key must read LEN_M1 with coefficient 1",
            );
            assert!(
                terms.iter().any(|t| matches!(t, LinearTerm::Constant(1))),
                "key must add the +1 that turns LEN_M1 back into len",
            );
        }
        other => panic!("EC_T0 key must be a linear term, got {other:?}"),
    }
}
