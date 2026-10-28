//! Known-answer tests for the full `ecsm_ecrecover` path (r/s parse,
//! decompress + parity, z-reduction, u1/u2, final keccak(X‖Y) address).
//!
//! On host, `ecsm_lincomb2` returns `None`, so these exercise the recovery
//! wiring through the pure-Rust `ProjectivePoint::lincomb` fallback.

use crate::*;

/// Build a valid ECDSA/secp256k1 signature from (d, kk, msg) using only the
/// k256 primitives already imported and return `(sig, recid, expected_addr)`.
///
/// `expected_addr` = keccak(X‖Y) of the uncompressed public key, exactly as
/// `ecsm_ecrecover` computes it.
fn make_ecdsa_fixture(d: Scalar, kk: Scalar, msg: [u8; 32]) -> ([u8; 64], u8, [u8; 32]) {
    assert!(!bool::from(d.is_zero()), "private key must be nonzero");
    assert!(!bool::from(kk.is_zero()), "nonce must be nonzero");

    // Public key Q = d·G.
    let q = (ProjectivePoint::GENERATOR * d).to_affine();
    let q_uncompressed = q.to_encoded_point(false);
    let expected = keccak_hash(&q_uncompressed.as_bytes()[1..65]);

    // R = kk·G; r = reduce(Rx); assert r ≠ 0.
    let r_point = (ProjectivePoint::GENERATOR * kk).to_affine();
    let (rx, ry) = affine_xy(&r_point).expect("R is not identity");
    let r = <Scalar as Reduce<U256>>::reduce_bytes(&rx.to_bytes());
    assert!(!bool::from(r.is_zero()), "r must be nonzero");
    // rx is in Fp; since n < p, rx >= n with probability ~2^{-128}. When that
    // happens r = rx-n and the signature requires the high-x recovery bit
    // (recid >= 2, meaning R.x = r+n) which ecsm_ecrecover does not handle.
    // Assert no reduction occurred so the low-x path is valid.
    assert_eq!(
        r.to_bytes(),
        rx.to_bytes(),
        "rx >= n: this kk needs high-x recovery (recid >= 2) — pick a different nonce"
    );

    // recid parity: low bit of Ry (big-endian, byte 31).
    let recid = ry.normalize().to_bytes()[31] & 1;

    // z = reduce(msg).
    let z = <Scalar as Reduce<U256>>::reduce_bytes(&FieldBytes::from(msg));

    // s = kk⁻¹ · (z + r·d).
    let s = kk.invert_vartime().expect("kk is nonzero") * (z + r * d);
    assert!(!bool::from(s.is_zero()), "s must be nonzero");

    // sig = r (BE, 32 bytes) ‖ s (BE, 32 bytes).
    let mut sig = [0u8; 64];
    sig[..32].copy_from_slice(&r.to_bytes());
    sig[32..].copy_from_slice(&s.to_bytes());

    (sig, recid, expected)
}

#[test]
fn ecrecover_known_answer_three_tuples() {
    // Three distinct (d, kk, msg) tuples — deterministic, no RNG.
    let tuples: &[(u64, u64, [u8; 32])] = &[
        (0x0000_0000_0000_0001u64, 0x0000_0000_dead_beefu64, {
            let mut m = [0u8; 32];
            m[31] = 0x42;
            m
        }),
        (0x00c0_ffee_dead_beef_u64, 0x0123_4567_89ab_cdef_u64, {
            let mut m = [0u8; 32];
            m[0] = 0xff;
            m[31] = 0x01;
            m
        }),
        (0x0bad_f00d_1337_cafe, 0xfeed_face_0000_0001, {
            let mut m = [0u8; 32];
            for (i, b) in m.iter_mut().enumerate() {
                *b = i as u8;
            }
            m
        }),
    ];

    for &(d_u64, kk_u64, msg) in tuples {
        let d = Scalar::from(d_u64);
        let kk = Scalar::from(kk_u64);
        let (sig, recid, expected) = make_ecdsa_fixture(d, kk, msg);
        let crypto = LambdaVmEcsmCrypto;
        match crypto.secp256k1_ecrecover(&sig, recid, &msg) {
            Ok(got) => assert_eq!(
                got, expected,
                "ecrecover returned wrong address for d={d_u64:#x} kk={kk_u64:#x}"
            ),
            Err(e) => panic!("ecrecover failed for d={d_u64:#x} kk={kk_u64:#x}: {e:?}"),
        }
    }
}

#[test]
fn ecrecover_rejects_zero_s() {
    // sig = valid r ‖ 0x00..00 (s = 0) must return InvalidSignature.
    let mut sig = [0u8; 64];
    // r = 1 (nonzero, but s = 0 in the second half).
    sig[31] = 0x01;
    let msg = [0u8; 32];
    let crypto = LambdaVmEcsmCrypto;
    assert!(
        matches!(
            crypto.secp256k1_ecrecover(&sig, 0, &msg),
            Err(CryptoError::InvalidSignature)
        ),
        "expected InvalidSignature for zero s"
    );
}

#[test]
fn ecrecover_rejects_zero_r() {
    // sig = 0x00..00 ‖ valid s must return InvalidSignature.
    let mut sig = [0u8; 64];
    sig[63] = 0x01; // s = 1, r = 0
    let msg = [0u8; 32];
    let crypto = LambdaVmEcsmCrypto;
    assert!(
        matches!(
            crypto.secp256k1_ecrecover(&sig, 0, &msg),
            Err(CryptoError::InvalidSignature)
        ),
        "expected InvalidSignature for zero r"
    );
}

/// Software stand-in for the ECSM precompile: `x(k·P)` for the point with
/// x-coordinate `x` (parity-invariant, like the real ecall).
fn soft_ecsm(x: &FieldElement, k: &Scalar) -> Option<FieldElement> {
    let xn = x.normalize();
    let y2 = (xn.square() * xn + FieldElement::from_u64(7)).normalize();
    let y = Option::<FieldElement>::from(y2.sqrt())?;
    let p = point_from_xy(&xn, &y.normalize())?;
    Some(affine_xy(&(p * k).to_affine())?.0)
}

/// Software stand-in for the field square-root hint.
fn soft_sqrt(rhs_be: &[u8; 32]) -> [u8; 32] {
    let rhs = Option::<FieldElement>::from(FieldElement::from_bytes(&(*rhs_be).into()))
        .expect("canonical");
    match Option::<FieldElement>::from(rhs.sqrt()) {
        Some(y) => y.to_bytes().into(),
        None => [0u8; 32],
    }
}

/// The accelerated recovery shape (R decompressed through the hint, the linear
/// combination through x-only queries, all on affine coordinates) with software
/// oracles, against the expected public key and against the software path
/// `ecsm_ecrecover` takes on the host. Honest and lying hints, both parities.
#[test]
fn accelerated_shape_recovers_the_signing_key() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let mut checked = 0;
    while checked < 300 {
        let d = Scalar::from(next()) * Scalar::from(next()) + Scalar::from(next());
        let kk = Scalar::from(next()) * Scalar::from(next()) + Scalar::from(next());
        let mut msg = [0u8; 32];
        for chunk in msg.chunks_mut(8) {
            chunk.copy_from_slice(&next().to_le_bytes());
        }
        if bool::from(d.is_zero()) || bool::from(kk.is_zero()) {
            continue;
        }
        // Skip the ~2^-128 nonces whose R.x needs the high-x recovery id.
        let r_point = (ProjectivePoint::GENERATOR * kk).to_affine();
        let rx = affine_xy(&r_point).unwrap().0;
        if <Scalar as Reduce<U256>>::reduce_bytes(&rx.to_bytes()).to_bytes() != rx.to_bytes() {
            continue;
        }
        let (sig, recid, _) = make_ecdsa_fixture(d, kk, msg);
        let q = (ProjectivePoint::GENERATOR * d).to_affine();
        let mut want = [0u8; 64];
        want.copy_from_slice(&q.to_encoded_point(false).as_bytes()[1..65]);

        for (lie, parity_flip) in [(false, false), (true, false), (false, true)] {
            let recid = recid ^ u8::from(parity_flip);
            let got = ecrecover_with(
                &sig,
                recid,
                &msg,
                |r, odd| {
                    if lie {
                        decompress_r_xy_with_oracle(r, odd, |_| [0x11; 32])
                    } else {
                        decompress_r_xy_with_oracle(r, odd, soft_sqrt)
                    }
                },
                |x1, y1, k1, x2, y2, k2| lincomb2_xy_with_oracle(x1, y1, k1, x2, y2, k2, soft_ecsm),
            );
            let got = got.ok();
            let software = ecsm_ecrecover(&sig, recid, &msg).ok();
            assert_eq!(
                got, software,
                "round {checked} lie {lie} flip {parity_flip}"
            );
            if !parity_flip {
                assert_eq!(got, Some(want), "round {checked} lie {lie}");
            }
        }
        checked += 1;
    }
}
