//! Rescue-Prime eXtended (RPX256 / XHash12) over Goldilocks at width 12 — the
//! `LFM_HASH` socket's second production-candidate tenant, and the fast one.
//!
//! # What it is, and what it shares with RPO
//!
//! RPX is a **round-function swap on RPO's geometry**, not a redesign
//! ([eprint 2023/1045](https://eprint.iacr.org/2023/1045); family paper in
//! Designs, Codes and Cryptography, 2026). ✓ VERIFIED against miden-crypto's
//! `Rpx256`: it takes **the same state width 12, the same rate 8 / capacity 4,
//! the same 4-felt digest, the same MDS and — literally — the same `ARK1`/`ARK2`
//! constant tables** as [`super::rpo`]. This module imports them rather than
//! re-deriving them, so the two tenants cannot drift apart.
//!
//! What changes is the seven-round schedule:
//!
//! | round | kind | content |
//! |---|---|---|
//! | 0, 2, 4 | **FB** | MDS → +ARK1 → `x^7` → MDS → +ARK2 → `x^{1/7}` — RPO's round exactly |
//! | 1, 3, 5 | **E** | +ARK1 → `x^7` in the **degree-3 EXTENSION** field, on four lane-triples. **No MDS.** |
//! | 6 | **M** | MDS → +ARK1. A linear finish; no S-box at all. |
//!
//! # Why this lane cares
//!
//! The inverse S-box is ~2^63-dense and this codebase measured it at **60% of
//! an RPO permutation on the box** (`rpo::throughput`). RPX runs **three**
//! inverse layers where RPO runs seven, and replaces the rest with an extension
//! `x^7` that is four multiplications on a triple. That is a direct attack on
//! the one term that dominates, which is why the design is reported ~2× faster.
//!
//! It is also narrower in the AIR: **316 value columns against RPO's 436**,
//! because an E round commits two extension intermediates per triple where an FB
//! round commits two ladders per lane, and the M round commits nothing at all.
//!
//! # ⚠ PROVENANCE — WEAKER THAN RPO'S, AND THAT MUST BE SAID
//!
//! [`super::rpo`] rests on nineteen EXTERNAL known-answer vectors published by
//! miden-crypto. **miden publishes no RPX known-answer table** — ✓ VERIFIED, its
//! `rpx/tests.rs` carries only structural tests (consistency, determinism,
//! padding, no-panic), no oracle. So RPX cannot be anchored the way RPO is, and
//! this module does not pretend otherwise. What it anchors instead:
//!
//! 1. **The shared half is externally anchored through RPO.** The constants, the
//!    MDS and the whole FB round are RPO's, pinned by RPO's nineteen vectors.
//! 2. **The new half is pinned to an INDEPENDENT algorithm.** The cubic
//!    extension's product is checked against naive polynomial multiplication
//!    reduced mod `x³ − x − 1`, and `power7` against generic square-and-multiply
//!    exponentiation in that extension — different algorithms for the same
//!    functions, not a second transcription of the same one.
//! 3. **The schedule** is the one miden's `Rpx256::apply_permutation` runs.
//!
//! ⚖ Net: strong on arithmetic, weaker on end-to-end identity than RPO. A
//! deployment decision should treat "no published KAT" as a real cost.
//!
//! # ⚠ NOT XHash8
//!
//! XHash8 is the faster sibling and it is **deliberately not built here**. Its
//! extra speed comes from a PARTIAL S-box layer (8 lanes of 12), and a partial
//! layer is one of the three structural footholds this project's own break
//! analysis identified in the 2026 Poseidon collapse — eprint 2026/1692's
//! S-box-skipping gadget restricts into the affine complement of the unS-boxed
//! lanes and works independent of round constants and MDS choice. XHash8's
//! S-boxes are not Poseidon's and eprint 2024/605 analyses XHASH8/12 directly,
//! so this is a flag rather than a verdict — but it is not a thing to adopt
//! quietly for the speed.

use crate::tables::types::FE;

use super::hash::{HASH_STATE_FELTS, LfmHasher};
use super::rpo::{ARK1, ARK2, DOMAIN_COMPRESS, DOMAIN_LEAF, DOMAIN_TRANSCRIPT, Rpo256, domain_iv};
use super::word::LfmWord;

/// Rounds — the same seven RPO has, differently shaped.
pub const NUM_ROUNDS: usize = 7;

/// Lanes per extension element: the extension is degree 3, so an E round reads
/// the twelve-lane state as FOUR triples.
pub const EXT_DEGREE: usize = 3;

/// Extension elements per E round.
pub const EXT_ELEMENTS: usize = HASH_STATE_FELTS / EXT_DEGREE;

/// Is round `r` an **FB** round — MDS, forward S-box, MDS, inverse S-box?
pub const fn is_fb_round(r: usize) -> bool {
    r.is_multiple_of(2) && r + 1 < NUM_ROUNDS
}

/// Is round `r` an **E** round — constants then `x^7` in the cubic extension,
/// with NO linear layer?
pub const fn is_ext_round(r: usize) -> bool {
    !r.is_multiple_of(2)
}

/// Is round `r` the **M** round — MDS then constants, and nothing else?
pub const fn is_final_round(r: usize) -> bool {
    r + 1 == NUM_ROUNDS
}

/// Arithmetic in `GF(p³) = GF(p)[φ] / (φ³ − φ − 1)`.
///
/// ⚠ **This is NOT the VM's own extension.** `crate::tables::types::FEE` is
/// built on `w³ = 2` (see `layout::xalu`); RPX's is `φ³ = φ + 1`. Mixing them
/// would be a wrong hash that still type-checks, so this module carries its own
/// arithmetic explicitly and never reaches for `FEE`.
pub mod cubic_ext {
    use super::FE;

    /// An extension element `a0 + a1·φ + a2·φ²`.
    pub type Ext = [FE; super::EXT_DEGREE];

    /// The product, reduced by `φ³ = φ + 1` and `φ⁴ = φ² + φ`.
    ///
    /// Written as the closed form rather than miden's Karatsuba arrangement:
    /// the AIR has to state these three coefficients as constraints, so the
    /// host computing them the same way is what makes the chip a transcription
    /// of this function instead of a second derivation.
    /// [`super::tests::the_extension_product_matches_naive_polynomial_arithmetic`]
    /// pins it against an independent algorithm.
    pub fn mul(a: &Ext, b: &Ext) -> Ext {
        [
            &(&a[0] * &b[0]) + &(&(&a[1] * &b[2]) + &(&a[2] * &b[1])),
            &(&(&a[0] * &b[1]) + &(&a[1] * &b[0]))
                + &(&(&(&a[1] * &b[2]) + &(&a[2] * &b[1])) + &(&a[2] * &b[2])),
            &(&(&a[0] * &b[2]) + &(&a[1] * &b[1])) + &(&(&a[2] * &b[0]) + &(&a[2] * &b[2])),
        ]
    }

    /// The square. One function, so a squaring and a product can never disagree.
    pub fn square(a: &Ext) -> Ext {
        mul(a, a)
    }

    /// `a^7` by the chain `a² → a³ → a⁶ → a⁷` — two squarings and two products,
    /// in exactly the association the degree-3 AIR lowering uses
    /// (`a⁷ = (a³)²·a` over the witnessed `a²`/`a³`).
    pub fn power7(a: &Ext) -> Ext {
        let a2 = square(a);
        let a3 = mul(&a2, a);
        let a6 = square(&a3);
        mul(&a6, a)
    }
}

/// One **FB** round's recorded intermediates — identical in shape to
/// [`super::rpo::RpoRound`], because it is the same round.
#[derive(Clone, Copy, Debug)]
pub struct FbRound {
    /// `u = MDS(state) + ARK1[r]` — the forward S-box input.
    pub u: [FE; HASH_STATE_FELTS],
    /// `u²`, committed.
    pub u2: [FE; HASH_STATE_FELTS],
    /// `u³`, committed. `u^7 = (u³)²·u` is then degree 3.
    pub u3: [FE; HASH_STATE_FELTS],
    /// `v = MDS(u^7) + ARK2[r]` — the inverse S-box input.
    pub v: [FE; HASH_STATE_FELTS],
    /// `y = v^{1/7}` — the round output, verified as the FORWARD power.
    pub y: [FE; HASH_STATE_FELTS],
    /// `y²`, committed.
    pub y2: [FE; HASH_STATE_FELTS],
    /// `y³`, committed.
    pub y3: [FE; HASH_STATE_FELTS],
}

/// One **E** round's recorded intermediates, per lane-triple.
#[derive(Clone, Copy, Debug)]
pub struct ExtRound {
    /// `x = state + ARK1[r]` — the extension input. Recorded for cross-checking;
    /// the AIR recomputes it as a degree-1 expression.
    pub x: [FE; HASH_STATE_FELTS],
    /// `x²` in the extension, committed — four triples laid out flat.
    pub t2: [FE; HASH_STATE_FELTS],
    /// `x³` in the extension, committed.
    pub t3: [FE; HASH_STATE_FELTS],
    /// `x⁷ = (x³)²·x` — the round output.
    pub out: [FE; HASH_STATE_FELTS],
}

/// Every intermediate the AIR witnesses. Indexed by round; the entry a round
/// does not use stays zero, the same convention `poseidon::PoseidonWitness`
/// follows for its partial rounds.
#[derive(Clone, Copy, Debug)]
pub struct RpxWitness {
    /// Rounds 0, 2, 4.
    pub fb: [FbRound; 3],
    /// Rounds 1, 3, 5.
    pub ext: [ExtRound; 3],
    /// The M round's output — `MDS(state) + ARK1[6]`, which IS `OUT`.
    pub final_out: [FE; HASH_STATE_FELTS],
}

/// The index of round `r` within its own kind's array.
pub const fn kind_index(r: usize) -> usize {
    r / 2
}

/// Records the permutation's intermediates for the trace generator.
///
/// ⚠ Written independently of [`Rpx256::permute`] rather than factored out of
/// it, the same discipline `rpo::permutation_witness` follows (standing-decisions
/// rule 7): a recording wrapper would make
/// [`tests::the_witness_agrees_with_the_permutation`] a tautology.
pub fn permutation_witness(state: [FE; HASH_STATE_FELTS]) -> RpxWitness {
    let zero = [FE::zero(); HASH_STATE_FELTS];
    let mut w = RpxWitness {
        fb: [FbRound {
            u: zero,
            u2: zero,
            u3: zero,
            v: zero,
            y: zero,
            y2: zero,
            y3: zero,
        }; 3],
        ext: [ExtRound {
            x: zero,
            t2: zero,
            t3: zero,
            out: zero,
        }; 3],
        final_out: zero,
    };
    let mut s = state;
    for r in 0..NUM_ROUNDS {
        if is_fb_round(r) {
            let round = &mut w.fb[kind_index(r)];
            let mixed = Rpo256::mds(&s);
            round.u = core::array::from_fn(|i| &mixed[i] + FE::from(ARK1[r][i]));
            let mut x = [FE::zero(); HASH_STATE_FELTS];
            for (lane, x_lane) in x.iter_mut().enumerate() {
                let u = &round.u[lane];
                round.u2[lane] = u * u;
                round.u3[lane] = &round.u2[lane] * u;
                *x_lane = &(&round.u3[lane] * &round.u3[lane]) * u;
            }
            let mixed = Rpo256::mds(&x);
            round.v = core::array::from_fn(|i| &mixed[i] + FE::from(ARK2[r][i]));
            round.y = round.v;
            Rpo256::inv_sbox_layer(&mut round.y);
            for lane in 0..HASH_STATE_FELTS {
                let y = &round.y[lane];
                round.y2[lane] = y * y;
                round.y3[lane] = &round.y2[lane] * y;
            }
            s = round.y;
        } else if is_ext_round(r) {
            let round = &mut w.ext[kind_index(r)];
            round.x = core::array::from_fn(|i| &s[i] + FE::from(ARK1[r][i]));
            for e in 0..EXT_ELEMENTS {
                let base = e * EXT_DEGREE;
                let x: cubic_ext::Ext = core::array::from_fn(|k| round.x[base + k]);
                let t2 = cubic_ext::square(&x);
                let t3 = cubic_ext::mul(&t2, &x);
                let t6 = cubic_ext::square(&t3);
                let out = cubic_ext::mul(&t6, &x);
                round.t2[base..base + EXT_DEGREE].copy_from_slice(&t2);
                round.t3[base..base + EXT_DEGREE].copy_from_slice(&t3);
                round.out[base..base + EXT_DEGREE].copy_from_slice(&out);
            }
            s = round.out;
        } else {
            debug_assert!(is_final_round(r));
            let mixed = Rpo256::mds(&s);
            w.final_out = core::array::from_fn(|i| &mixed[i] + FE::from(ARK1[r][i]));
            s = w.final_out;
        }
    }
    w
}

/// Rescue-Prime eXtended at width 12, rate 8, capacity 4, 7 rounds — RPX256.
pub struct Rpx256;

impl LfmHasher for Rpx256 {
    /// The schedule: `FB E FB E FB E M`.
    ///
    /// Note the E round has **no linear layer** — its only mixing is the
    /// extension multiplication inside each triple, and diffusion across triples
    /// is the FB rounds' job. That is the design, not an omission
    /// (✓ miden `Rpx256::apply_ext_round_ref`), and it is why an E round costs
    /// the AIR 36 columns where an FB round costs 60.
    fn permute(&self, state: [FE; HASH_STATE_FELTS]) -> [FE; HASH_STATE_FELTS] {
        let mut s = state;
        for r in 0..NUM_ROUNDS {
            if is_fb_round(r) {
                s = Rpo256::mds(&s);
                for (lane, v) in s.iter_mut().enumerate() {
                    *v += FE::from(ARK1[r][lane]);
                }
                for v in s.iter_mut() {
                    *v = Rpo256::sbox(v);
                }
                s = Rpo256::mds(&s);
                for (lane, v) in s.iter_mut().enumerate() {
                    *v += FE::from(ARK2[r][lane]);
                }
                Rpo256::inv_sbox_layer(&mut s);
            } else if is_ext_round(r) {
                for (lane, v) in s.iter_mut().enumerate() {
                    *v += FE::from(ARK1[r][lane]);
                }
                let mut next = [FE::zero(); HASH_STATE_FELTS];
                for e in 0..EXT_ELEMENTS {
                    let base = e * EXT_DEGREE;
                    let x: cubic_ext::Ext = core::array::from_fn(|k| s[base + k]);
                    let p = cubic_ext::power7(&x);
                    next[base..base + EXT_DEGREE].copy_from_slice(&p);
                }
                s = next;
            } else {
                s = Rpo256::mds(&s);
                for (lane, v) in s.iter_mut().enumerate() {
                    *v += FE::from(ARK1[r][lane]);
                }
            }
        }
        s
    }

    /// The same capacity-domain separation RPO uses, over the same lanes — the
    /// geometry is identical, so the convention carries across unchanged.
    fn compress_iv(&self) -> LfmWord {
        domain_iv(DOMAIN_COMPRESS).map(FE::from)
    }

    fn transcript_iv(&self) -> LfmWord {
        domain_iv(DOMAIN_TRANSCRIPT).map(FE::from)
    }

    fn leaf_iv(&self) -> LfmWord {
        domain_iv(DOMAIN_LEAF).map(FE::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive polynomial multiplication reduced mod `φ³ − φ − 1` — an
    /// INDEPENDENT algorithm for [`cubic_ext::mul`]'s function, written the
    /// obvious slow way precisely so it shares no structure with the closed
    /// form under test.
    fn naive_ext_mul(a: &cubic_ext::Ext, b: &cubic_ext::Ext) -> cubic_ext::Ext {
        let mut c = [FE::zero(); 5];
        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                c[i + j] = &c[i + j] + &(ai * bj);
            }
        }
        // φ³ = φ + 1, φ⁴ = φ² + φ.
        let c3 = c[3];
        let c4 = c[4];
        [&c[0] + &c3, &(&c[1] + &c3) + &c4, &c[2] + &c4]
    }

    /// Generic square-and-multiply in the extension — an INDEPENDENT algorithm
    /// for [`cubic_ext::power7`]'s function.
    fn naive_ext_pow(a: &cubic_ext::Ext, mut e: u32) -> cubic_ext::Ext {
        let mut result: cubic_ext::Ext = [FE::one(), FE::zero(), FE::zero()];
        let mut base = *a;
        while e > 0 {
            if e & 1 == 1 {
                result = naive_ext_mul(&result, &base);
            }
            base = naive_ext_mul(&base, &base);
            e >>= 1;
        }
        result
    }

    fn sample_ext(seed: u64) -> cubic_ext::Ext {
        core::array::from_fn(|k| {
            FE::from(
                seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(k as u64 + 1),
            )
        })
    }

    /// ★ The pin that carries RPX's NEW arithmetic, since miden publishes no
    /// known-answer table for it. Two different algorithms, same answer.
    #[test]
    fn the_extension_product_matches_naive_polynomial_arithmetic() {
        for seed in 0..64u64 {
            let a = sample_ext(seed);
            let b = sample_ext(seed.wrapping_add(1_000));
            assert_eq!(
                cubic_ext::mul(&a, &b),
                naive_ext_mul(&a, &b),
                "mul at {seed}"
            );
            assert_eq!(
                cubic_ext::square(&a),
                naive_ext_mul(&a, &a),
                "square at {seed}"
            );
        }
    }

    /// `power7` is the seventh power, checked against generic exponentiation.
    #[test]
    fn the_extension_power7_matches_generic_exponentiation() {
        for seed in 0..32u64 {
            let a = sample_ext(seed);
            assert_eq!(
                cubic_ext::power7(&a),
                naive_ext_pow(&a, 7),
                "power7 at {seed}"
            );
        }
    }

    /// ★ The AIR's degree-3 lowering, checked in the extension: committing `a²`
    /// and `a³` and writing the output as `(a³)²·a` must be the seventh power.
    /// If this ever fails, `eval_rpx`'s E-round arm is proving a different map.
    #[test]
    fn the_air_lowering_computes_the_seventh_power_in_the_extension() {
        for seed in 0..32u64 {
            let a = sample_ext(seed);
            let a2 = cubic_ext::square(&a);
            let a3 = cubic_ext::mul(&a2, &a);
            let lowered = cubic_ext::mul(&cubic_ext::square(&a3), &a);
            assert_eq!(lowered, naive_ext_pow(&a, 7), "lowering at {seed}");
        }
    }

    /// The extension must be a FIELD over the polynomial claimed, or `x ↦ x^7`
    /// is not a permutation of it. `φ³ − φ − 1` is irreducible over Goldilocks
    /// iff it has no root; checked by exhaustion over the only cheap witness we
    /// have — that `power7` is injective on a sample — plus the algebraic
    /// identity that makes the S-box invertible: `gcd(7, p³ − 1) = 1`.
    #[test]
    fn the_extension_sbox_is_a_permutation() {
        const P: u128 = (1u128 << 64) - (1u128 << 32) + 1;
        // `x ↦ x^7` permutes GF(p³) iff gcd(7, p³ − 1) = 1, i.e. iff 7 ∤ p³ − 1.
        // Computed mod 7 rather than over p³, which does not fit.
        let p_mod_7 = (P % 7) as u64;
        // p³ − 1 ≡ p_mod_7³ − 1 (mod 7)
        let cube_minus_one = (p_mod_7 * p_mod_7 % 7 * p_mod_7 % 7 + 7 - 1) % 7;
        assert_ne!(
            cube_minus_one, 0,
            "7 must not divide p³ − 1, or x^7 is not a permutation of GF(p³)"
        );
        // And injectivity on a sample, as the concrete counterpart.
        let mut seen = std::collections::BTreeSet::new();
        for seed in 0..128u64 {
            let a = sample_ext(seed);
            assert!(
                seen.insert(cubic_ext::power7(&a).map(|f| *f.value())),
                "power7 collided at seed {seed}"
            );
        }
    }

    /// The schedule is the one miden runs: FB E FB E FB E M.
    #[test]
    fn the_schedule_is_three_fb_three_ext_and_one_final() {
        let kinds: Vec<&str> = (0..NUM_ROUNDS)
            .map(|r| {
                if is_fb_round(r) {
                    "FB"
                } else if is_ext_round(r) {
                    "E"
                } else {
                    "M"
                }
            })
            .collect();
        assert_eq!(kinds, vec!["FB", "E", "FB", "E", "FB", "E", "M"]);
        // Exactly one kind per round — no round is two things, none is nothing.
        for r in 0..NUM_ROUNDS {
            let n = usize::from(is_fb_round(r))
                + usize::from(is_ext_round(r))
                + usize::from(is_final_round(r));
            assert_eq!(n, 1, "round {r} must have exactly one kind");
        }
        assert_eq!(EXT_ELEMENTS, 4);
        assert_eq!(EXT_DEGREE * EXT_ELEMENTS, HASH_STATE_FELTS);
    }

    /// ★ RPX shares RPO's constants LITERALLY, and this asserts the import
    /// rather than trusting it: a future edit that gave RPX its own tables would
    /// silently make it a different hash from the one miden ships.
    #[test]
    fn rpx_uses_rpos_constant_tables() {
        assert_eq!(ARK1.len(), NUM_ROUNDS);
        assert_eq!(ARK2.len(), NUM_ROUNDS);
        assert_eq!(ARK1[0][0], super::super::rpo::ARK1[0][0]);
        assert_eq!(
            super::super::rpo::MDS_CIRC_ROW,
            [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8]
        );
        // Only the FB rounds consume ARK2; the E and M rounds use ARK1 alone.
        // Asserted so the schedule and the constant usage cannot drift apart.
        assert_eq!((0..NUM_ROUNDS).filter(|r| is_fb_round(*r)).count(), 3);
    }

    /// A genuine differential: two independently written round loops.
    #[test]
    fn the_witness_agrees_with_the_permutation() {
        for seed in 0..8u64 {
            let input: [FE; HASH_STATE_FELTS] =
                core::array::from_fn(|i| FE::from(seed.wrapping_mul(0x9E37_79B9) + i as u64));
            let w = permutation_witness(input);
            assert_eq!(
                w.final_out,
                Rpx256.permute(input),
                "witness and permute must agree at seed {seed}"
            );
        }
    }

    /// The witness records the association the AIR constrains, in both round
    /// kinds.
    #[test]
    fn the_witness_records_the_degree_three_associations() {
        let input: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| FE::from(5 * i as u64 + 3));
        let w = permutation_witness(input);
        for round in w.fb.iter() {
            for lane in 0..HASH_STATE_FELTS {
                assert_eq!(round.u2[lane], &round.u[lane] * &round.u[lane]);
                assert_eq!(round.u3[lane], &round.u2[lane] * &round.u[lane]);
                assert_eq!(round.y2[lane], &round.y[lane] * &round.y[lane]);
                assert_eq!(round.y3[lane], &round.y2[lane] * &round.y[lane]);
                assert_eq!(
                    &(&round.y3[lane] * &round.y3[lane]) * &round.y[lane],
                    round.v[lane],
                    "the FB fold must hold"
                );
            }
        }
        for round in w.ext.iter() {
            for e in 0..EXT_ELEMENTS {
                let base = e * EXT_DEGREE;
                let x: cubic_ext::Ext = core::array::from_fn(|k| round.x[base + k]);
                let t2: cubic_ext::Ext = core::array::from_fn(|k| round.t2[base + k]);
                let t3: cubic_ext::Ext = core::array::from_fn(|k| round.t3[base + k]);
                let out: cubic_ext::Ext = core::array::from_fn(|k| round.out[base + k]);
                assert_eq!(t2, cubic_ext::square(&x));
                assert_eq!(t3, cubic_ext::mul(&t2, &x));
                assert_eq!(out, cubic_ext::mul(&cubic_ext::square(&t3), &x));
            }
        }
    }

    /// ★ The padding row, in all three round kinds. With the mode sum zero the
    /// chip scales every round constant away, and the all-zero state must be a
    /// fixed point of what remains — MDS, both S-boxes, and the extension power.
    #[test]
    fn an_all_zero_state_is_a_fixed_point_without_constants() {
        let mut s = [FE::zero(); HASH_STATE_FELTS];
        for r in 0..NUM_ROUNDS {
            if is_fb_round(r) {
                s = Rpo256::mds(&s);
                for v in s.iter_mut() {
                    *v = Rpo256::sbox(v);
                }
                s = Rpo256::mds(&s);
                Rpo256::inv_sbox_layer(&mut s);
            } else if is_ext_round(r) {
                let mut next = [FE::zero(); HASH_STATE_FELTS];
                for e in 0..EXT_ELEMENTS {
                    let base = e * EXT_DEGREE;
                    let x: cubic_ext::Ext = core::array::from_fn(|k| s[base + k]);
                    next[base..base + EXT_DEGREE].copy_from_slice(&cubic_ext::power7(&x));
                }
                s = next;
            } else {
                s = Rpo256::mds(&s);
            }
        }
        assert_eq!(s, [FE::zero(); HASH_STATE_FELTS]);
    }

    /// ⚠ RPX must not accidentally BE RPO. The two share constants, an MDS and
    /// three of seven rounds, so a schedule bug could plausibly collapse one
    /// into the other; this is the check that says it did not.
    #[test]
    fn rpx_and_rpo_are_different_functions() {
        let input: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| FE::from(i as u64));
        assert_ne!(Rpx256.permute(input), Rpo256.permute(input));
    }

    /// The three socket domains carry across from RPO unchanged, and remain
    /// three different functions.
    #[test]
    fn the_three_socket_domains_are_distinct() {
        let a: LfmWord = core::array::from_fn(|i| FE::from(11 * i as u64 + 1));
        let b: LfmWord = core::array::from_fn(|i| FE::from(7 * i as u64 + 2));
        let parent = Rpx256.compress(&a, &b);
        let step = Rpx256.transcript(&a, &b);
        let leaf = Rpx256.leaf(&a, &b);
        assert_ne!(parent, step);
        assert_ne!(parent, leaf);
        assert_ne!(step, leaf);
        assert_eq!(Rpx256.compress_iv(), Rpo256.compress_iv());
    }
}

/// ★ The three algebraic candidates and the incumbent, timed on ONE machine in
/// ONE run — the host half of the comparison the campaign asked for.
///
/// `#[ignore]`d because it is a timing measurement, not a property. Run with
/// `cargo test --release -p lambda-vm-prover --lib hash_ladder_throughput -- --ignored --nocapture`.
///
/// **Why one test rather than three.** Per-permutation cost is the only input to
/// the host-commit column that is architecture-dependent, and quoting three
/// numbers measured in three runs on two machines is exactly the mistake this
/// campaign keeps having to correct. Measuring them together means the RATIOS
/// are machine-independent even when the absolutes are not.
#[cfg(test)]
mod ladder {
    use super::*;
    use crate::lfm::hash::HasherKind;
    use crate::lfm::poseidon::PoseidonGoldilocks;
    use std::time::Instant;

    const PERMUTATIONS: usize = 20_000;

    fn time_permutation(label: &str, kind: HasherKind, baseline: Option<f64>) -> f64 {
        let mut state: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| FE::from(i as u64 + 1));
        let start = Instant::now();
        for _ in 0..PERMUTATIONS {
            state = kind.permute(state);
        }
        let ns = start.elapsed().as_nanos() as f64 / PERMUTATIONS as f64;
        assert_ne!(state[0], FE::zero(), "the chain must not collapse to zero");
        match baseline {
            None => println!("  {label:<22} {ns:>9.0} ns/perm"),
            Some(b) => println!("  {label:<22} {ns:>9.0} ns/perm   {:.2}× RPO", ns / b),
        }
        ns
    }

    #[test]
    #[ignore]
    fn hash_ladder_throughput() {
        println!("algebraic permutations, single thread, this machine:");
        let rpo = time_permutation("RPO256", HasherKind::Rpo, None);
        let rpx = time_permutation("RPX256 (XHash12)", HasherKind::Rpx, Some(rpo));
        let pos = time_permutation("Poseidon (UNSHIPPABLE)", HasherKind::Poseidon, Some(rpo));

        // The incumbent, at the shape a Merkle parent takes, on the same machine.
        let left = [0x5Au8; 32];
        let mut right = [0xA5u8; 32];
        const PARENTS: usize = 2_000_000;
        let start = Instant::now();
        for _ in 0..PARENTS {
            right = crypto::hash::blake3::chain::blake3_parent(&left, &right);
        }
        let b3 = start.elapsed().as_nanos() as f64 / PARENTS as f64;
        assert_ne!(right, [0u8; 32]);
        println!("  {:<22} {b3:>9.0} ns/parent", "BLAKE3 64-byte");
        println!();
        println!("  vs BLAKE3 per 2-to-1 compression:");
        println!(
            "    RPO {:.0}×   RPX {:.0}×   Poseidon {:.0}×",
            rpo / b3,
            rpx / b3,
            pos / b3
        );

        // ★ The claim RPX exists to make: fewer inverse S-box layers is faster.
        // Three of seven against seven of seven, so a real multiple is expected;
        // 1.2× is a floor loose enough that no honest machine trips it while
        // still failing if the schedule ever collapsed back to RPO's.
        assert!(
            rpo / rpx > 1.2,
            "RPX must be materially faster than RPO: RPO {rpo:.0} ns, RPX {rpx:.0} ns"
        );
        // And the sanity check that we are timing three different functions.
        let probe: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| FE::from(i as u64));
        assert_ne!(Rpx256.permute(probe), Rpo256.permute(probe));
        assert_ne!(Rpx256.permute(probe), PoseidonGoldilocks.permute(probe));
    }
}
