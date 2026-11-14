//! The four-limb base-field type (`fe.rs`) against k256's `FieldElement`: every
//! operation on edge values and seeded random ones, inputs in the weak form included
//! (limb patterns from p up to 2^256 − 1, which stand for value − p).

use crate::fe::Fe;
use crate::*;
use k256::elliptic_curve::bigint::ArrayEncoding;

/// p, big-endian.
const P_BE: [u8; 32] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFC, 0x2F,
];

fn p() -> U256 {
    U256::from_be_slice(&P_BE)
}

/// The k256 element a limb pattern stands for, reduced independently of `fe.rs`.
fn reference(limbs: [u64; 4]) -> FieldElement {
    let mut v = U256::from_words(limbs);
    if v >= p() {
        v = v.wrapping_sub(&p());
    }
    let be: [u8; 32] = v.to_be_byte_array().into();
    Option::<FieldElement>::from(FieldElement::from_bytes(&be.into())).expect("below p")
}

fn canonical_be(x: FieldElement) -> [u8; 32] {
    x.normalize().to_bytes().into()
}

/// Edge limb patterns: zero, one, around p, around 2^256, around C = 2^256 − p, and
/// patterns that make carries and borrows run through every limb.
fn edges() -> Vec<[u64; 4]> {
    let p = p().to_words();
    let mut out = vec![
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [7, 0, 0, 0],
        [0x1_0000_03D0, 0, 0, 0],
        [0x1_0000_03D1, 0, 0, 0],
        [0x1_0000_03D2, 0, 0, 0],
        [u64::MAX, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1 << 63],
        [u64::MAX; 4],
        [u64::MAX - 1, u64::MAX, u64::MAX, u64::MAX],
        [0, u64::MAX, u64::MAX, u64::MAX],
        [u64::MAX, u64::MAX, u64::MAX, 0],
        [0, 0, 0, u64::MAX],
    ];
    for d in [0u64, 1, 2, 977, 0x1_0000_03D0] {
        // p + d and p − 1 − d.
        let (lo, c) = p[0].overflowing_add(d);
        if !c {
            out.push([lo, p[1], p[2], p[3]]);
        }
        out.push([p[0] - 1 - d, p[1], p[2], p[3]]);
    }
    out
}

fn random_limbs(next: &mut impl FnMut() -> u64, round: usize) -> [u64; 4] {
    let mut l = [next(), next(), next(), next()];
    match round % 5 {
        // Near 2^256: the weak range [p, 2^256) and just below it.
        0 => {
            l[1] = u64::MAX;
            l[2] = u64::MAX;
            l[3] = u64::MAX;
        }
        // Small values.
        1 => {
            l[1] = 0;
            l[2] = 0;
            l[3] = 0;
        }
        _ => {}
    }
    l
}

fn rng() -> impl FnMut() -> u64 {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn check_pair(a: [u64; 4], b: [u64; 4]) {
    let (fa, fb) = (Fe::from_limbs(a), Fe::from_limbs(b));
    let (ka, kb) = (reference(a), reference(b));
    let ctx = || format!("a = {a:x?}, b = {b:x?}");
    assert_eq!(
        (fa + fb).to_be_bytes(),
        canonical_be(ka + kb),
        "add {}",
        ctx()
    );
    assert_eq!(
        (fa - fb).to_be_bytes(),
        canonical_be(ka - kb),
        "sub {}",
        ctx()
    );
    assert_eq!(
        (fa * fb).to_be_bytes(),
        canonical_be(ka * kb),
        "mul {}",
        ctx()
    );
    assert_eq!(
        fa.same(fb),
        canonical_be(ka) == canonical_be(kb),
        "same {}",
        ctx()
    );
}

fn check_one(a: [u64; 4]) {
    let fa = Fe::from_limbs(a);
    let ka = reference(a);
    let ctx = || format!("a = {a:x?}");
    let want = canonical_be(ka);
    assert_eq!(fa.to_be_bytes(), want, "to_be_bytes {}", ctx());
    let mut want_le = want;
    want_le.reverse();
    assert_eq!(fa.to_le_bytes(), want_le, "to_le_bytes {}", ctx());
    assert_eq!(
        fa.square().to_be_bytes(),
        canonical_be(ka.square()),
        "square {}",
        ctx()
    );
    assert_eq!(
        fa.double().to_be_bytes(),
        canonical_be(ka.double()),
        "double {}",
        ctx()
    );
    assert_eq!((-fa).to_be_bytes(), canonical_be(-ka), "neg {}", ctx());
    assert_eq!(
        fa.is_zero(),
        bool::from(ka.normalizes_to_zero()),
        "is_zero {}",
        ctx()
    );
    assert_eq!(fa.is_odd(), want[31] & 1 == 1, "is_odd {}", ctx());
    assert_eq!(
        Fe::from_k256(&ka).to_be_bytes(),
        want,
        "from_k256 {}",
        ctx()
    );
    assert_eq!(
        fa.to_k256().map(canonical_be),
        Some(want),
        "to_k256 {}",
        ctx()
    );
    // Decoding accepts exactly the canonical encodings, as `FieldElement::from_bytes`.
    let mut raw_be = [0u8; 32];
    for (chunk, limb) in raw_be.rchunks_exact_mut(8).zip(a) {
        chunk.copy_from_slice(&limb.to_be_bytes());
    }
    let mut raw_le = raw_be;
    raw_le.reverse();
    let k256_accepts = bool::from(FieldElement::from_bytes(&raw_be.into()).is_some());
    assert_eq!(
        Fe::from_be_bytes(&raw_be).is_some(),
        k256_accepts,
        "from_be_bytes {}",
        ctx()
    );
    assert_eq!(
        Fe::from_le_bytes(&raw_le).is_some(),
        k256_accepts,
        "from_le_bytes {}",
        ctx()
    );
    if k256_accepts {
        assert_eq!(
            Fe::from_be_bytes(&raw_be).map(Fe::to_be_bytes),
            Some(raw_be)
        );
        assert_eq!(Fe::from_le_bytes(&raw_le).map(Fe::limbs), Some(a));
    }
}

#[test]
fn edge_values_match_k256() {
    let edges = edges();
    for &a in &edges {
        check_one(a);
        for &b in &edges {
            check_pair(a, b);
        }
    }
}

#[test]
fn random_values_match_k256() {
    let mut next = rng();
    let edges = edges();
    for round in 0..20_000 {
        let a = random_limbs(&mut next, round);
        let b = random_limbs(&mut next, round / 5);
        check_one(a);
        check_pair(a, b);
        check_pair(a, edges[round % edges.len()]);
    }
}

/// Long chains in the weak form stay correct: 2,000 rounds of products, squares,
/// sums and differences fed back into each other, against the same chain in k256.
#[test]
fn chained_operations_match_k256() {
    let mut next = rng();
    let mut f = [Fe::from_limbs([u64::MAX; 4]), Fe::from_limbs([3, 0, 0, 0])];
    let mut k = [reference([u64::MAX; 4]), reference([3, 0, 0, 0])];
    for round in 0..2_000 {
        let i = round % 2;
        let x = random_limbs(&mut next, round);
        let (fx, kx) = (Fe::from_limbs(x), reference(x));
        match next() % 5 {
            0 => {
                f[i] = f[i] * f[1 - i];
                k[i] = (k[i] * k[1 - i]).normalize();
            }
            1 => {
                f[i] = f[i].square() + fx;
                k[i] = (k[i].square() + kx).normalize();
            }
            2 => {
                f[i] = f[i] - fx;
                k[i] = (k[i] - kx).normalize();
            }
            3 => {
                f[i] = -(f[i] + f[1 - i]);
                k[i] = (-(k[i] + k[1 - i]).normalize()).normalize();
            }
            _ => {
                f[i] = f[i].double() * fx;
                k[i] = (k[i].double() * kx).normalize();
            }
        }
        assert_eq!(f[i].to_be_bytes(), canonical_be(k[i]), "round {round}");
    }
}

/// The generator constants are k256's generator.
#[test]
fn generator_constants_are_k256s() {
    let (x, y) = affine_xy(&AffinePoint::GENERATOR).unwrap();
    assert_eq!(GENERATOR_X.to_be_bytes(), canonical_be(x));
    assert_eq!(GENERATOR_Y.to_be_bytes(), canonical_be(y));
}
