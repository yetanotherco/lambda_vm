//! Validation for the `−2^i·T₀` correction constants.
//!
//! The constants are produced by `k256` projective doubling + `batch_normalize`;
//! every check below uses a *different* implementation (the `Fp` BigUint
//! reference, or `lincomb2_witness`'s own BigInt group law) so a bug in one
//! cannot hide behind the other.

use num_bigint::BigUint;

use crate::curve::AffinePoint;
use crate::lincomb2_table::{NEG_T0_POW2_ROWS, neg_t0_pow2_points};
use crate::tests::reference::point_double;
use crate::witness::{JointSel, lincomb2_witness, t0};
use crate::{B, n, p};

fn le32(v: &BigUint) -> [u8; 32] {
    crate::to_le_32(v)
}

/// `−pt`, computed here rather than read from the table under test.
fn negate(pt: &AffinePoint) -> AffinePoint {
    AffinePoint {
        x: pt.x.clone(),
        y: &p() - &pt.y,
    }
}

/// Every entry is a canonical on-curve point (`y² ≡ x³ + b mod p`, both
/// coordinates `< p`). Independent of how the table was built.
#[test]
fn neg_t0_pow2_entries_are_on_curve_and_canonical() {
    let table = neg_t0_pow2_points();
    assert_eq!(table.len(), NEG_T0_POW2_ROWS);
    let modulus = p();
    for (i, pt) in table.iter().enumerate() {
        assert!(pt.x < modulus, "entry {i}: x >= p");
        assert!(pt.y < modulus, "entry {i}: y >= p");
        assert!(pt.y > BigUint::ZERO, "entry {i}: y = 0 (2-torsion)");
        let lhs = (&pt.y * &pt.y) % &modulus;
        let rhs = (&pt.x * &pt.x % &modulus * &pt.x + B) % &modulus;
        assert_eq!(lhs, rhs, "entry {i} is not on the curve");
    }
}

/// The table is exactly the doubling chain of `T₀`, negated: entry `0` is
/// `−T₀` and entry `i+1` is `−2·(−entry i)`. Recomputed with the `Fp`
/// reference doubling, not the `k256` path the table itself uses.
#[test]
fn neg_t0_pow2_matches_reference_doubling_chain() {
    let table = neg_t0_pow2_points();
    let mut expected = t0();
    for (i, entry) in table.iter().enumerate() {
        assert_eq!(
            *entry,
            negate(&expected),
            "entry {i} != -(2^{i}·T0) under the Fp reference doubling",
        );
        expected = point_double(&expected);
    }
}

/// The stored `y` is the negation of the positive multiple's `y`, and the
/// stored `x` is unchanged: `x(−P) = x(P)`, `y(−P) + y(P) ≡ 0 mod p`. This is
/// the convention assertion — if the table ever flips to storing `+2^i·T₀`,
/// this test is what fails.
#[test]
fn neg_t0_pow2_stores_the_negation_not_the_positive_multiple() {
    let table = neg_t0_pow2_points();
    let modulus = p();
    let mut positive = t0();
    for (i, entry) in table.iter().enumerate() {
        assert_eq!(entry.x, positive.x, "entry {i}: x must equal x(2^i·T0)");
        assert_eq!(
            (&entry.y + &positive.y) % &modulus,
            BigUint::ZERO,
            "entry {i}: y must be the additive inverse of y(2^i·T0)",
        );
        assert_ne!(
            entry.y, positive.y,
            "entry {i}: negation must be a real change (y != p - y for y != 0)",
        );
        positive = point_double(&positive);
    }
}

/// The load-bearing test: `lincomb2_witness`'s correction row consumes its
/// addend from `(x_g, y_g)`, and that addend must be table entry `len`
/// verbatim. Covers the whole sign convention end to end, on real witnesses
/// spanning several `len` values (including the `len = 256` top row).
#[test]
fn correction_row_addend_equals_table_entry_at_len() {
    let table = neg_t0_pow2_points();
    let g = {
        // The generator, via the crate's own recovery path.
        let gx = BigUint::parse_bytes(
            b"79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
            16,
        )
        .unwrap();
        let gy = crate::curve::recover_y_canonical(&gx).unwrap();
        AffinePoint { x: gx, y: gy }
    };
    // 2·G — a second base with an x distinct from G's, so the `P1 + P2`
    // precompute is a genuine chord.
    let p2 = point_double(&g);

    // Scalars chosen to exercise a spread of `len` values, including the max.
    let cases = [
        BigUint::from(1u8),
        BigUint::from(3u8),
        BigUint::from(0xFFFFu32),
        BigUint::from(1u8) << 200,
        n() - 1u32, // len = 256
    ];

    let mut seen_lens = Vec::new();
    for u1 in &cases {
        for u2 in &cases {
            let w = lincomb2_witness(&le32(u1), &le32(u2), &g, &p2).expect("witness");
            let len = w.len as usize;
            assert!(
                (1..=256).contains(&len),
                "len {len} outside the reachable range",
            );
            seen_lens.push(len);

            let corr = w
                .steps
                .iter()
                .find(|s| matches!(s.sel, JointSel::Correction))
                .expect("correction row");
            let entry = &table[len];
            assert_eq!(
                corr.step.x_g,
                le32(&entry.x),
                "len {len}: correction addend x != table entry x",
            );
            assert_eq!(
                corr.step.y_g,
                le32(&entry.y),
                "len {len}: correction addend y != table entry y \
                 (sign convention broken: the table must store -2^len·T0)",
            );

            // The witness also records the *positive* blind separately; the
            // table's x half serves it directly, its y half only after negation.
            assert_eq!(
                w.x_t0_pow,
                le32(&entry.x),
                "len {len}: x_t0_pow != table entry x",
            );
            assert_eq!(
                BigUint::from_bytes_le(&w.y_t0_pow) + &entry.y,
                p(),
                "len {len}: y_t0_pow is not the negation of the table entry y",
            );
        }
    }
    assert!(
        seen_lens.contains(&256),
        "the len = 256 top row was never exercised",
    );
}
