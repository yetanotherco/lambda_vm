//! Tests for the ECDSA-recovery call's reference (`ecrecover`) and witness
//! (`compute_ecrec_witness`): k256-signed signatures recover their key on both
//! parities, every input class the call bails on bails, and the witness agrees with
//! the reference and satisfies its relations (the witness builder asserts every
//! limb identity it writes).

use k256::elliptic_curve::PrimeField;
use k256::elliptic_curve::ops::Reduce;
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::{ProjectivePoint, Scalar, U256};
use num_bigint::BigUint;

use crate::ecrec::{ECREC_STATUS_BAIL, ECREC_STATUS_KEY, compute_ecrec_witness, ecrecover};
use crate::{n, p, to_le_32};

/// A deterministic xorshift stream.
fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed;
    move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn scalar(next: &mut impl FnMut() -> u64) -> Scalar {
    Scalar::from(next()) * Scalar::from(next()) * Scalar::from(next()) + Scalar::from(next())
}

/// `(x, y)` of a point, big-endian.
fn xy(q: &ProjectivePoint) -> ([u8; 32], [u8; 32]) {
    let e = q.to_affine().to_encoded_point(false);
    let (mut x, mut y) = ([0u8; 32], [0u8; 32]);
    x.copy_from_slice(e.x().unwrap());
    y.copy_from_slice(e.y().unwrap());
    (x, y)
}

/// A signature by key `d` with nonce `k` on hash `msg`: `(r ‖ s, parity of R's y)`,
/// or `None` for the ~2^-128 nonces whose R.x exceeds N.
fn sign(d: Scalar, k: Scalar, msg: &[u8; 32]) -> Option<([u8; 64], u8)> {
    let (rx, ry) = xy(&(ProjectivePoint::GENERATOR * k));
    let r = Option::<Scalar>::from(Scalar::from_repr(rx.into()))?;
    let z = <Scalar as Reduce<U256>>::reduce_bytes(&(*msg).into());
    let s = Option::<Scalar>::from(k.invert())? * (z + r * d);
    let mut sig = [0u8; 64];
    sig[..32].copy_from_slice(&r.to_bytes());
    sig[32..].copy_from_slice(&s.to_bytes());
    Some((sig, ry[31] & 1))
}

/// Valid signatures recover their signer, and the witness says the same key.
#[test]
fn signatures_recover_their_key() {
    let mut next = rng(0x9E37_79B9_7F4A_7C15);
    let mut checked = 0;
    while checked < 60 {
        let (d, k) = (scalar(&mut next), scalar(&mut next));
        let mut msg = [0u8; 32];
        for c in msg.chunks_mut(8) {
            c.copy_from_slice(&next().to_le_bytes());
        }
        let Some((sig, parity)) = sign(d, k, &msg) else {
            continue;
        };
        let want = xy(&(ProjectivePoint::GENERATOR * d));
        assert_eq!(ecrecover(&sig, &msg, parity), Some(want), "round {checked}");
        let w = compute_ecrec_witness(&sig, &msg, parity);
        assert!(w.ok);
        assert_eq!(w.status(), ECREC_STATUS_KEY);
        let (mut qx, mut qy) = (w.x_q, w.y_q);
        qx.reverse();
        qy.reverse();
        assert_eq!((qx, qy), want, "round {checked}: the witness's key");
        // The other parity is the other R: a valid recovery of a different key.
        let other = ecrecover(&sig, &msg, parity ^ 1);
        assert!(other.is_some() && other != Some(want));
        checked += 1;
    }
}

/// Every input class the call refuses to compute bails, in the reference and the
/// witness alike.
#[test]
fn invalid_inputs_bail() {
    let mut next = rng(7);
    let (d, k) = (scalar(&mut next), scalar(&mut next));
    let msg = [0x5Au8; 32];
    let (sig, _) = sign(d, k, &msg).expect("a valid signature");
    let n_be = {
        let mut b = to_le_32(&n());
        b.reverse();
        b
    };
    let with = |r: Option<[u8; 32]>, s: Option<[u8; 32]>| {
        let mut out = sig;
        if let Some(r) = r {
            out[..32].copy_from_slice(&r);
        }
        if let Some(s) = s {
            out[32..].copy_from_slice(&s);
        }
        out
    };
    let mut cases: Vec<(&str, [u8; 64], [u8; 32])> = vec![
        ("r = 0", with(Some([0; 32]), None), msg),
        ("r = N", with(Some(n_be), None), msg),
        ("r = 2^256 - 1", with(Some([0xFF; 32]), None), msg),
        ("s = 0", with(None, Some([0; 32])), msg),
        ("s = N", with(None, Some(n_be)), msg),
        ("s = 2^256 - 1", with(None, Some([0xFF; 32])), msg),
        ("z = 0", sig, [0; 32]),
        ("z = N", sig, n_be),
    ];
    // An r with no curve point: scan small values for a non-residue r^3 + 7.
    let pp = p();
    let nonres = (2u64..200)
        .find(|&x| {
            let rhs = (BigUint::from(x).pow(3) + 7u8) % &pp;
            rhs.modpow(&((&pp - 1u8) / 2u8), &pp) != BigUint::from(1u8)
        })
        .expect("a non-residue below 200");
    let mut r_nonres = [0u8; 32];
    r_nonres[24..].copy_from_slice(&nonres.to_be_bytes());
    cases.push(("r not an x-coordinate", with(Some(r_nonres), None), msg));
    for (name, sig, msg) in cases {
        for parity in [0, 1] {
            assert_eq!(
                ecrecover(&sig, &msg, parity),
                None,
                "{name}, parity {parity}"
            );
            let w = compute_ecrec_witness(&sig, &msg, parity);
            assert!(!w.ok, "{name}");
            assert_eq!(w.status(), ECREC_STATUS_BAIL);
            assert!(w.chain1.is_empty() && w.chain2.is_empty() && w.add.is_none());
            assert_eq!((w.x_q, w.y_q), ([0; 32], [0; 32]));
        }
    }
}

/// `u1·G = ±u2·R`: the chord is undefined (`A = B`) or the key is the point at
/// infinity (`A = −B`); both bail. With `R = t·G`, `A = −B` exactly when
/// `z = s·t` and `A = B` when `z = −s·t` (mod N).
#[test]
fn a_degenerate_final_addition_bails() {
    let mut next = rng(11);
    let mut done = 0;
    while done < 4 {
        let (t, s) = (scalar(&mut next), scalar(&mut next));
        let (rx, ry) = xy(&(ProjectivePoint::GENERATOR * t));
        if Option::<Scalar>::from(Scalar::from_repr(rx.into())).is_none() {
            continue;
        }
        let mut sig = [0u8; 64];
        sig[..32].copy_from_slice(&rx);
        sig[32..].copy_from_slice(&s.to_bytes());
        for z in [s * t, -(s * t)] {
            let msg: [u8; 32] = z.to_bytes().into();
            assert_eq!(ecrecover(&sig, &msg, ry[31] & 1), None);
            assert!(!compute_ecrec_witness(&sig, &msg, ry[31] & 1).ok);
        }
        done += 1;
    }
}

/// A message hash at or above N is reduced as the recovery reduces it: `z` and
/// `z − N` recover the same key (the witness's `u1` relation takes `z` unreduced).
#[test]
fn a_hash_above_n_is_reduced() {
    let mut next = rng(13);
    let mut z_small = [0u8; 32];
    z_small[31] = 5; // z = 5; z + N < 2^256
    let z_big = {
        let mut b = to_le_32(&(n() + 5u8));
        b.reverse();
        b
    };
    let (d, k) = (scalar(&mut next), scalar(&mut next));
    let (sig, parity) = sign(d, k, &z_small).expect("valid");
    let want = xy(&(ProjectivePoint::GENERATOR * d));
    assert_eq!(ecrecover(&sig, &z_small, parity), Some(want));
    assert_eq!(ecrecover(&sig, &z_big, parity), Some(want));
    let w = compute_ecrec_witness(&sig, &z_big, parity);
    assert!(w.ok);
    assert_eq!(
        w.z,
        to_le_32(&(n() + 5u8)),
        "z is the hash as read, unreduced"
    );
}

/// The witness's chains are the double-and-add of u1 and u2: one ECDAS row per
/// double and per add below the top bit.
#[test]
fn the_chains_have_one_row_per_step() {
    let mut next = rng(17);
    let (d, k) = (scalar(&mut next), scalar(&mut next));
    let msg = [0x33u8; 32];
    let (sig, parity) = sign(d, k, &msg).expect("valid");
    let w = compute_ecrec_witness(&sig, &msg, parity);
    for (u, len, chain) in [(&w.u1, w.len1, &w.chain1), (&w.u2, w.len2, &w.chain2)] {
        let ones: u32 = u.iter().map(|b| b.count_ones()).sum();
        assert_eq!(chain.len(), len as usize + ones as usize - 1);
        assert_eq!(BigUint::from_bytes_le(u).bits() as u8 - 1, len);
    }
    let add = w.add.as_ref().expect("the final addition");
    assert_eq!((add.op, add.next_op, add.round), (1, 0, 0));
    assert_eq!(
        (add.x_a, add.y_a, add.x_g, add.y_g),
        (w.x_a, w.y_a, w.x_b, w.y_b)
    );
    assert_eq!((add.x_r, add.y_r), (w.x_q, w.y_q));
    assert_eq!(w.y_r[0] & 1, parity);
    assert_eq!(w.y_r[0], 2 * w.h + parity);
}
