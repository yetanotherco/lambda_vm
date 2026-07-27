//! The lincomb2 NUMS correction constants: `−2^i·T₀` for `i ∈ [0, 256]`.
//!
//! The lincomb2 joint chain seeds its accumulator with the nothing-up-my-sleeve
//! point `T₀` (see [`crate::witness::t0`] and
//! `thoughts/ec-recover-opt/lincomb2/T0.md`), so after `len` doublings the
//! accumulator carries a blind of `2^len·T₀`. The chain's final row —
//! [`crate::witness::JointSel::Correction`] — strips it by *adding*
//! `−2^len·T₀`.
//!
//! # Sign convention: these are the NEGATED points
//!
//! Entry `i` is `−2^i·T₀`, i.e. `(x(2^i·T₀), p − y(2^i·T₀))`, **not** `2^i·T₀`.
//! That is exactly the addend `lincomb2_witness` writes into the correction
//! row's `(x_g, y_g)` columns (`witness.rs`: `neg_tpow` is built by negating
//! `tpow`'s `y`, and is passed as `joint_row`'s `addend`), so a chip that wires
//! a lookup of this table straight into its addend columns needs no modular
//! negation at all. `crate::tests::lincomb2_table_tests` asserts the match
//! against a real witness rather than restating it.
//!
//! Note that only `y` differs between the two conventions — `x(−P) = x(P)` —
//! so the `x` half of an entry serves either reading.
//!
//! # Why 257 entries
//!
//! `len = max(bits(u1), bits(u2))` for scalars in `[1, N)`, so the reachable
//! range is `[1, 256]`. Entry `0` (`−T₀` itself) is defined anyway: it is the
//! anchor of the doubling recurrence and costs nothing.

use num_bigint::BigUint;

use k256::elliptic_curve::group::Curve as _;
use k256::{AffinePoint as K256Affine, ProjectivePoint};

use crate::curve::{AffinePoint, from_k256_affine, to_k256_affine};
use crate::p;
use crate::witness::t0;

/// Number of defined entries: `i ∈ [0, 256]`.
pub const NEG_T0_POW2_ROWS: usize = 257;

/// `−2^i·T₀` for every `i ∈ [0, 256]`, indexed by `i`.
///
/// Derived from [`crate::witness::t0`] by repeated doubling in `k256`
/// projective coordinates (one `batch_normalize` for the whole chain), then
/// negating each `y`. Deterministic and reproducible: the only input is the
/// pinned `T₀`.
///
/// See the module header for the sign convention — these are the *negated*
/// points, matching the correction row's addend.
pub fn neg_t0_pow2_points() -> Vec<AffinePoint> {
    let mut proj = ProjectivePoint::from(to_k256_affine(&t0()));
    let mut chain = Vec::with_capacity(NEG_T0_POW2_ROWS);
    for _ in 0..NEG_T0_POW2_ROWS {
        chain.push(proj);
        proj = proj.double();
    }

    let mut affine = vec![K256Affine::IDENTITY; NEG_T0_POW2_ROWS];
    ProjectivePoint::batch_normalize(&chain, &mut affine);

    let modulus = p();
    affine
        .iter()
        .map(|a| {
            let pt = from_k256_affine(a);
            // `T₀` generates the prime-order group, so no multiple of it is the
            // identity or a 2-torsion point: `y` is never 0 and `p - y` is a
            // canonical nonzero field element.
            debug_assert!(pt.y != BigUint::from(0u8), "2^i·T0 has y = 0");
            AffinePoint {
                x: pt.x,
                y: &modulus - &pt.y,
            }
        })
        .collect()
}
