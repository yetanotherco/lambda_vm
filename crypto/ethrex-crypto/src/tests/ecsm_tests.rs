//! Tests for the `ecsm_lincomb2` ABI marshalling.
//!
//! # What these cover, and what they cannot
//!
//! The accelerated path is now three steps: pack the operands, make one ecall,
//! parse the result. The ecall exists only on riscv64, so **step 2 is not
//! reachable from a host test** — `ecsm_lincomb2` returns `None` on host before
//! it is ever issued.
//!
//! Steps 1 and 3 are ordinary code, and they are where a bug would actually
//! live: a byte-order slip in either direction silently produces a wrong point.
//! So these tests pin the ABI from both ends, and [`soft_lincomb2`] closes the
//! loop by standing in for the ecall — decoding the operands exactly as the
//! executor does, computing `Q` in pure Rust, and re-encoding it in the layout
//! the chip writes. Chained between the two real functions that is a genuine
//! differential against `ProjectivePoint::lincomb` over the whole ABI.
//!
//! What remains uncovered here is the ecall and the chip behind it. That is
//! carried by a proven guest — the ethrex block tests in the prover crate run
//! real ecrecovers through this path and verify — and by the executor's own
//! `lincomb2` suite for the status contract. See the phase-G report.

use crate::*;

/// secp256k1 `Gx‖Gy` as the ABI carries it: two 32-byte little-endian values.
/// Written out independently of anything in this crate (the constant an
/// ecrecover caller's `P1` operand must equal), so it cross-checks
/// [`lincomb2_operands`]'s byte order against a source outside k256.
const GENERATOR_LE: [u8; 64] = {
    let mut out = [0u8; 64];
    // Gx, big-endian 79BE667E…16F81798, reversed below.
    let gx_be: [u8; 32] = [
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
        0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8,
        0x17, 0x98,
    ];
    // Gy, big-endian 483ADA77…FB10D4B8.
    let gy_be: [u8; 32] = [
        0x48, 0x3A, 0xDA, 0x77, 0x26, 0xA3, 0xC4, 0x65, 0x5D, 0xA4, 0xFB, 0xFC, 0x0E, 0x11, 0x08,
        0xA8, 0xFD, 0x17, 0xB4, 0x48, 0xA6, 0x85, 0x54, 0x19, 0x9C, 0x47, 0xD0, 0x8F, 0xFB, 0x10,
        0xD4, 0xB8,
    ];
    let mut i = 0;
    while i < 32 {
        out[i] = gx_be[31 - i];
        out[32 + i] = gy_be[31 - i];
        i += 1;
    }
    out
};

fn g_times(n: u64) -> ProjectivePoint {
    ProjectivePoint::GENERATOR * Scalar::from(n)
}

/// Decodes one 32-byte little-endian half of an operand into big-endian bytes.
fn le_half(op: &Operand, half: usize) -> [u8; 32] {
    let mut be = [0u8; 32];
    for (i, b) in be.iter_mut().enumerate() {
        *b = op.0[half * 32 + 31 - i];
    }
    be
}

/// Software stand-in for the `ecsm_lincomb2` ecall.
///
/// Decodes the three operands the way the executor does, computes
/// `Q = u1·P1 + u2·P2` with the pure-Rust lincomb, and writes `xQ‖yQ` back in
/// the chip's little-endian layout. Returns `None` where the accelerator would
/// return a non-zero status.
fn soft_lincomb2(p1: &Operand, p2: &Operand, u: &Operand) -> Option<Operand> {
    let point = |op: &Operand| -> Option<ProjectivePoint> {
        let x = Option::<FieldElement>::from(FieldElement::from_bytes(&le_half(op, 0).into()))?;
        let y = Option::<FieldElement>::from(FieldElement::from_bytes(&le_half(op, 1).into()))?;
        point_from_xy(&x, &y)
    };
    let scalar = |op: &Operand, half: usize| -> Option<Scalar> {
        Option::from(Scalar::from_repr(le_half(op, half).into()))
    };

    let q = ProjectivePoint::lincomb(&point(p1)?, &scalar(u, 0)?, &point(p2)?, &scalar(u, 1)?);
    let affine = q.to_affine();
    if bool::from(affine.is_identity()) {
        return None; // the accelerator reports status 6 (ResultInfinity)
    }
    let (x, y) = affine_xy(&affine)?;
    Some(operand(&x.to_bytes(), &y.to_bytes()))
}

// ── The ABI, both directions ────────────────────────────────────────────────

/// The full round trip: pack real operands, run the software stand-in for the
/// ecall, parse the result — and land on exactly what `ProjectivePoint::lincomb`
/// computes. A byte-order error anywhere in the ABI breaks this.
#[test]
fn abi_round_trip_matches_software_lincomb() {
    let cases = [
        (g_times(3), 123_456_789u64, g_times(7), 987_654_321u64),
        (g_times(11), 2u64.pow(20) + 5, g_times(2), 42u64),
        (ProjectivePoint::GENERATOR, 7u64, g_times(5), 9u64),
        // The ecrecover shape: generator first, R second.
        (
            ProjectivePoint::GENERATOR,
            0xdead_beefu64,
            g_times(0x1234),
            0x0bad_f00du64,
        ),
    ];
    for (p1, k1, p2, k2) in cases {
        let (k1, k2) = (Scalar::from(k1), Scalar::from(k2));
        let expected = ProjectivePoint::lincomb(&p1, &k1, &p2, &k2);

        let (p1_op, p2_op, u_op) =
            lincomb2_operands(&p1, &k1, &p2, &k2).expect("non-identity points must marshal");
        let q = soft_lincomb2(&p1_op, &p2_op, &u_op).expect("non-degenerate case");
        let got = point_from_le_q(&q.0).expect("the result must parse back");

        assert_eq!(got.to_affine(), expected.to_affine());
    }
}

/// A base point with odd `y` marshals correctly — the parity byte moves from
/// index 31 (big-endian) to index 0 (little-endian), so a half-reversed
/// conversion would survive the even-`y` cases above and fail here.
#[test]
fn odd_y_base_point_round_trips() {
    let p1 = (2u64..200)
        .find_map(|n| {
            let p = g_times(n);
            let (_, y) = affine_xy(&p.to_affine())?;
            (y.normalize().to_bytes()[31] & 1 == 1).then_some(p)
        })
        .expect("one of the first 200 multiples of G has odd y");

    let (k1, k2) = (Scalar::from(54321u64), Scalar::from(11111u64));
    let p2 = g_times(13);
    let expected = ProjectivePoint::lincomb(&p1, &k1, &p2, &k2);

    let (p1_op, p2_op, u_op) = lincomb2_operands(&p1, &k1, &p2, &k2).unwrap();
    let q = soft_lincomb2(&p1_op, &p2_op, &u_op).unwrap();
    assert_eq!(
        point_from_le_q(&q.0).unwrap().to_affine(),
        expected.to_affine()
    );
}

/// `P1 = G` must marshal to the exact 64 bytes the chip pins as the generator.
/// If this drifts, every ecrecover silently takes the software fallback with
/// status 7 (`P1 != G`) and the accelerator does nothing.
#[test]
fn generator_operand_matches_the_pinned_constant() {
    let (k1, k2) = (Scalar::from(5u64), Scalar::from(9u64));
    let (p1_op, _, _) =
        lincomb2_operands(&ProjectivePoint::GENERATOR, &k1, &g_times(3), &k2).unwrap();
    assert_eq!(p1_op.0, GENERATOR_LE, "P1 operand must be G in ABI order");
}

/// The scalar operand is `u1‖u2`, in that order, each little-endian. Swapping
/// the halves would compute `u2·G + u1·R` and still land on the curve, so pin it.
#[test]
fn scalar_operand_packs_u1_then_u2() {
    let k1 = Scalar::from(0x0102_0304_0506_0708u64);
    let k2 = Scalar::from(0x1112_1314_1516_1718u64);
    let (_, _, u_op) =
        lincomb2_operands(&ProjectivePoint::GENERATOR, &k1, &g_times(3), &k2).unwrap();

    assert_eq!(
        &le_half(&u_op, 0)[..],
        &k1.to_bytes()[..],
        "first half is u1"
    );
    assert_eq!(
        &le_half(&u_op, 1)[..],
        &k2.to_bytes()[..],
        "second half is u2"
    );
    // Little-endian: the least-significant byte leads.
    assert_eq!(u_op.0[0], 0x08);
    assert_eq!(u_op.0[32], 0x18);
}

/// The executor rejects a misaligned operand as a hard `ExecutionError`, not a
/// status word — so a 1-byte-aligned buffer would abort the proof rather than
/// fall back. `Operand` carries `repr(align(8))` to prevent that; pin it here,
/// since nothing else in the crate would notice if it were removed.
#[test]
fn operands_are_eight_byte_aligned() {
    assert_eq!(core::mem::align_of::<Operand>(), 8);
    let (p1_op, p2_op, u_op) = lincomb2_operands(
        &ProjectivePoint::GENERATOR,
        &Scalar::from(7u64),
        &g_times(3),
        &Scalar::from(9u64),
    )
    .unwrap();
    for op in [&p1_op, &p2_op, &u_op] {
        assert_eq!(
            op.0.as_ptr() as usize % 8,
            0,
            "operand bytes must start 8-byte aligned"
        );
    }
}

// ── Guards ─────────────────────────────────────────────────────────────────

/// An identity input has no affine `(x, y)` to marshal, so the call is never
/// formed. This is the only degeneracy the guest still screens: every other one
/// comes back as a status word.
#[test]
fn identity_points_do_not_marshal() {
    let p = g_times(3);
    let k = Scalar::from(7u64);
    let id = ProjectivePoint::IDENTITY;
    assert!(lincomb2_operands(&id, &k, &p, &k).is_none());
    assert!(lincomb2_operands(&p, &k, &id, &k).is_none());
}

/// The result parser rejects a point that is not on the curve — the backstop
/// against a marshalling bug on our side.
#[test]
fn off_curve_result_is_rejected() {
    let (x, y) = affine_xy(&g_times(3).to_affine()).unwrap();
    let mut q = operand(&x.to_bytes(), &y.to_bytes());
    assert!(point_from_le_q(&q.0).is_some(), "the real point must parse");
    q.0[32] ^= 1; // perturb yQ: (x, y+1) is not on the curve
    assert!(
        point_from_le_q(&q.0).is_none(),
        "an off-curve result must be rejected"
    );
}

/// Degenerate configurations the accelerator reports as a status: `P1 = ±P2`
/// (status 5) and a cancelling combination (`Q = ∞`, status 6). The guest
/// marshals them happily — that is correct, the chip is the one that declines —
/// so this pins that the *stand-in* declines where the chip would, keeping the
/// round-trip test honest about what it is and is not exercising.
#[test]
fn cancelling_terms_are_declined_by_the_accelerator_not_the_guest() {
    let p = g_times(3);
    let k = Scalar::from(7u64);

    // P1 = P2: marshals fine; the chip answers SumDegenerate.
    let (p1_op, p2_op, _) = lincomb2_operands(&p, &k, &p, &k).expect("marshals");
    assert_eq!(p1_op.0, p2_op.0);

    // k1·P + k2·(−P) = O with k1 = k2: the stand-in returns None, as the chip
    // would with ResultInfinity.
    let (p1_op, p2_op, u_op) = lincomb2_operands(&p, &k, &(-p), &k).expect("marshals");
    assert!(soft_lincomb2(&p1_op, &p2_op, &u_op).is_none());
}
