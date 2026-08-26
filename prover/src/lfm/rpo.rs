//! Rescue-Prime Optimized (RPO256) over Goldilocks at width 12 — the `LFM_HASH`
//! socket's production-candidate tenant.
//!
//! # Why this hash
//!
//! The Poseidon family is off the table (see [`super::poseidon`]'s header and
//! the break record: eprint 2026/306 for Poseidon2's linear layers, eprint
//! 2026/1692 for Poseidon-original's partial layer). RPO is the strongest
//! algebraic candidate that the break class does not structurally reach: it has
//! **no partial rounds** — all twelve lanes pass a nonlinear map in every
//! half-round, so the subspace-restriction gadget has no affine complement to
//! live in — and **no cheap algebraic direction**, because composing forward
//! costs `x^{1/7}` layers and composing backward costs `x^7` layers. That is
//! Rescue's entire design thesis and it is why the Poseidon literature keeps not
//! porting to it.
//!
//! ⚠ This is a risk decision, not an arithmetic one: RPO is a 2022 design and
//! carries a "young design" discount BLAKE3 does not. It is unbroken since the
//! Marvellous line began in 2019, and the spec defends its N = 7 (12.5% under
//! its own formula's 8) with a 1.5× margin argument, but the Gröbner frontier
//! moves every year.
//!
//! # Parameter provenance — READ THIS BEFORE TRUSTING A MEASUREMENT
//!
//! Every constant below has **two independent sources that were checked against
//! each other**, which is the whole reason to trust them:
//!
//! 1. The spec's own generator ([eprint 2022/1577](https://eprint.iacr.org/2022/1577),
//!    reference implementation at `github.com/ASDiscreteMathematics/rpo`,
//!    `rescue_prime_optimized.sage::get_round_constants`): round constants are
//!    `SHAKE256("RPO(18446744069414584321,12,4,128)", 9·2·12·7)` cut into
//!    nine-byte little-endian chunks reduced mod `p`.
//! 2. `miden-crypto`'s shipped `ARK1`/`ARK2` tables
//!    (`src/hash/algebraic_sponge/rescue/mod.rs`), production code since 2022.
//!
//! The SHAKE256 derivation was re-run outside this repository and reproduces
//! miden's 168 constants exactly. The MDS row is likewise the spec's
//! `get_mds(12)` and miden's `MDS` first row, identically.
//!
//! # Lane convention — a gift, not a choice
//!
//! The RPO **permutation** is lane-agnostic; a deployment picks which lanes are
//! rate, which are capacity and where the digest is read. Two conventions exist:
//! the 2022 paper's (capacity `0..4`, rate `4..12`, digest `4..8`) and miden's
//! (rate `0..8`, capacity `8..12`, digest `0..4`).
//!
//! **This module follows miden's, because it is exactly the `LFM_HASH` socket's
//! own layout**: the socket materializes the capacity cell at lanes 8–11
//! (`S8..S11`) and reads the digest from `OUT0..3`. No lane permutation is
//! needed anywhere — the chip's frozen prefix already sits where RPO256 wants
//! it, and a `Compress` row with the zero IV is *literally* `Rpo256::merge`.
//!
//! # Domain separation — the capacity slot, per miden's blessed construction
//!
//! [`LfmHasher`]'s trait defaults let a single-domain hasher ship a transcript
//! that is indistinguishable from a Merkle parent; `hash.rs` records that
//! weakening deliberately and says a production candidate must override it.
//! RPO does, by the mechanism miden ships as `merge_in_domain`: **capacity lane
//! 9 carries a domain identifier**, while capacity lane 8 stays reserved for the
//! sponge's padding flag. The security argument is the RPX spec's Appendix C
//! ([eprint 2023/1045](https://eprint.iacr.org/2023/1045)): setting a capacity
//! element to a domain tag degrades only pre-image resistance, by at most the
//! log₂ of the domain space, and pre-image is not the sponge's binding term
//! until it falls under 2^128 — which three one-word tags do not approach.
//!
//! The tags reuse the BLAKE3 socket's ASCII names ([`super::blake3_socket`]'s
//! `TAG_LFMT` / `TAG_LFML`) so one domain has one name across both tenants.
//! `Compress` takes domain ZERO on purpose: that makes a Merkle parent under
//! this machine bit-identical to a standard `Rpo256::merge`, externally
//! checkable against miden.
//!
//! # Oracle
//!
//! [`tests::the_sponge_matches_the_miden_known_answer_vectors`] replays all
//! nineteen of miden-crypto's `hash_elements` test vectors through this
//! permutation. Nothing in this repository produced those numbers, and they
//! pin the constants, the round order, the MDS orientation, both S-box
//! exponents and the lane convention at once.

use crate::tables::types::FE;

use super::hash::{HASH_STATE_FELTS, LfmHasher};
use super::word::LfmWord;

/// The forward S-box exponent. Like Poseidon's, 7 is forced by Goldilocks:
/// `p - 1 = 2^32 · 3 · 5 · 17 · 257 · 65537`, so neither 3 nor 5 is coprime to
/// it and neither `x³` nor `x⁵` is a permutation.
pub const ALPHA: u32 = 7;

/// The inverse S-box exponent, `ALPHA⁻¹ mod (p − 1)`.
///
/// ≈ 2^63, and that is the point: the map is cheap in one direction and
/// astronomically dense in the other, in BOTH directions of the round function.
/// [`tests::the_inverse_exponent_inverts_alpha`] re-derives it rather than
/// trusting the literal.
pub const INV_ALPHA: u64 = 10540996611094048183;

/// Rounds. The spec's own formula gives 8; RPO ships 7 and defends the 12.5%
/// shave in §4.2 with a 1.5× margin argument and Gröbner estimates above twice
/// the security level. This is the number the AIR's column count is linear in —
/// raising it to 8 costs 48 value columns.
pub const NUM_ROUNDS: usize = 7;

/// Half-rounds per round: forward `x^7`, then inverse `x^{1/7}`.
pub const HALVES_PER_ROUND: usize = 2;

/// First ROW of the circulant MDS matrix, so `M[i][j] = MDS_CIRC_ROW[(j - i) mod 12]`.
///
/// The spec's `get_mds(12)` and miden's `MDS[0]`, identically. RPO's security
/// argument is MDS-AGNOSTIC (spec §4.1: "Rescue-Prime is secure when
/// instantiated with any MDS matrix"), so this row is a speed choice — it is
/// NTT-friendly — and not a security parameter. That matters: it is exactly the
/// property Poseidon2 lacked.
pub const MDS_CIRC_ROW: [u64; HASH_STATE_FELTS] = [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8];

/// Round constants for the FIRST half of each round — added after the first MDS
/// and before the forward `x^7` layer. See the module header for the two-source
/// provenance.
pub const ARK1: [[u64; HASH_STATE_FELTS]; NUM_ROUNDS] = [
    [
        5789762306288267392,
        6522564764413701783,
        17809893479458208203,
        107145243989736508,
        6388978042437517382,
        15844067734406016715,
        9975000513555218239,
        3344984123768313364,
        9959189626657347191,
        12960773468763563665,
        9602914297752488475,
        16657542370200465908,
    ],
    [
        12987190162843096997,
        653957632802705281,
        4441654670647621225,
        4038207883745915761,
        5613464648874830118,
        13222989726778338773,
        3037761201230264149,
        16683759727265180203,
        8337364536491240715,
        3227397518293416448,
        8110510111539674682,
        2872078294163232137,
    ],
    [
        18072785500942327487,
        6200974112677013481,
        17682092219085884187,
        10599526828986756440,
        975003873302957338,
        8264241093196931281,
        10065763900435475170,
        2181131744534710197,
        6317303992309418647,
        1401440938888741532,
        8884468225181997494,
        13066900325715521532,
    ],
    [
        5674685213610121970,
        5759084860419474071,
        13943282657648897737,
        1352748651966375394,
        17110913224029905221,
        1003883795902368422,
        4141870621881018291,
        8121410972417424656,
        14300518605864919529,
        13712227150607670181,
        17021852944633065291,
        6252096473787587650,
    ],
    [
        4887609836208846458,
        3027115137917284492,
        9595098600469470675,
        10528569829048484079,
        7864689113198939815,
        17533723827845969040,
        5781638039037710951,
        17024078752430719006,
        109659393484013511,
        7158933660534805869,
        2955076958026921730,
        7433723648458773977,
    ],
    [
        16308865189192447297,
        11977192855656444890,
        12532242556065780287,
        14594890931430968898,
        7291784239689209784,
        5514718540551361949,
        10025733853830934803,
        7293794580341021693,
        6728552937464861756,
        6332385040983343262,
        13277683694236792804,
        2600778905124452676,
    ],
    [
        7123075680859040534,
        1034205548717903090,
        7717824418247931797,
        3019070937878604058,
        11403792746066867460,
        10280580802233112374,
        337153209462421218,
        13333398568519923717,
        3596153696935337464,
        8104208463525993784,
        14345062289456085693,
        17036731477169661256,
    ],
];

/// Round constants for the SECOND half of each round — added after the second
/// MDS and before the inverse `x^{1/7}` layer.
pub const ARK2: [[u64; HASH_STATE_FELTS]; NUM_ROUNDS] = [
    [
        6077062762357204287,
        15277620170502011191,
        5358738125714196705,
        14233283787297595718,
        13792579614346651365,
        11614812331536767105,
        14871063686742261166,
        10148237148793043499,
        4457428952329675767,
        15590786458219172475,
        10063319113072092615,
        14200078843431360086,
    ],
    [
        6202948458916099932,
        17690140365333231091,
        3595001575307484651,
        373995945117666487,
        1235734395091296013,
        14172757457833931602,
        707573103686350224,
        15453217512188187135,
        219777875004506018,
        17876696346199469008,
        17731621626449383378,
        2897136237748376248,
    ],
    [
        8023374565629191455,
        15013690343205953430,
        4485500052507912973,
        12489737547229155153,
        9500452585969030576,
        2054001340201038870,
        12420704059284934186,
        355990932618543755,
        9071225051243523860,
        12766199826003448536,
        9045979173463556963,
        12934431667190679898,
    ],
    [
        18389244934624494276,
        16731736864863925227,
        4440209734760478192,
        17208448209698888938,
        8739495587021565984,
        17000774922218161967,
        13533282547195532087,
        525402848358706231,
        16987541523062161972,
        5466806524462797102,
        14512769585918244983,
        10973956031244051118,
    ],
    [
        6982293561042362913,
        14065426295947720331,
        16451845770444974180,
        7139138592091306727,
        9012006439959783127,
        14619614108529063361,
        1394813199588124371,
        4635111139507788575,
        16217473952264203365,
        10782018226466330683,
        6844229992533662050,
        7446486531695178711,
    ],
    [
        3736792340494631448,
        577852220195055341,
        6689998335515779805,
        13886063479078013492,
        14358505101923202168,
        7744142531772274164,
        16135070735728404443,
        12290902521256031137,
        12059913662657709804,
        16456018495793751911,
        4571485474751953524,
        17200392109565783176,
    ],
    [
        17130398059294018733,
        519782857322261988,
        9625384390925085478,
        1664893052631119222,
        7629576092524553570,
        3485239601103661425,
        9755891797164033838,
        15218148195153269027,
        16460604813734957368,
        9643968136937729763,
        3611348709641382851,
        18256379591337759196,
    ],
];

/// Capacity lane carrying the sponge padding flag — reserved, never a domain.
///
/// Miden's `hash_elements` writes `total_len % RATE` here; the socket's modes
/// are all exactly one full rate block, for which that flag is zero, so the
/// socket leaves it zero. Naming it stops the domain from being put here.
pub const CAPACITY_PAD_LANE: usize = 0;

/// Capacity lane carrying the DOMAIN identifier — miden's `merge_in_domain`
/// slot. See the module header for why this is sound.
pub const CAPACITY_DOMAIN_LANE: usize = 1;

/// The `Compress` (Merkle parent) domain: ZERO, deliberately.
///
/// A compress row is then bit-identical to `Rpo256::merge`, so a parent this
/// machine proves is checkable against miden's shipped implementation without
/// knowing anything about this codebase.
pub const DOMAIN_COMPRESS: u64 = 0;

/// The Fiat–Shamir transcript domain — `"LFMT"` as a little-endian `u32`, the
/// same name [`super::blake3_socket::TAG_LFMT`] carries.
pub const DOMAIN_TRANSCRIPT: u64 = u32::from_le_bytes(*b"LFMT") as u64;

/// The Merkle LEAF domain — `"LFML"`, matching
/// [`super::blake3_socket::TAG_LFML`].
pub const DOMAIN_LEAF: u64 = u32::from_le_bytes(*b"LFML") as u64;

/// The capacity cell for a domain: `[0, domain, 0, 0]`.
///
/// One rule, so the chip's constraints, the executor's state and the trace
/// filler cannot disagree about which lane the tag lives in.
pub const fn domain_iv(domain: u64) -> [u64; 4] {
    let mut iv = [0u64; 4];
    iv[CAPACITY_DOMAIN_LANE] = domain;
    iv
}

/// One round's recorded intermediates, in the association the degree-3 AIR
/// lowering needs.
///
/// The round is `u = MDS(s) + ark1` → `x = u^7` → `v = MDS(x) + ark2` →
/// `y = v^{1/7}`. The AIR recomputes `u`, `x` and `v` as expressions and
/// commits only what it cannot: the two S-box ladders and the round output.
#[derive(Clone, Copy, Debug)]
pub struct RpoRound {
    /// `u = MDS(state) + ARK1[r]` — the forward S-box input. Recorded for
    /// cross-checking only; the AIR recomputes it as a degree-1 expression.
    pub u: [FE; HASH_STATE_FELTS],
    /// `u²` — committed.
    pub u2: [FE; HASH_STATE_FELTS],
    /// `u³` — committed. `u^7 = (u³)²·u` is then a degree-3 expression.
    pub u3: [FE; HASH_STATE_FELTS],
    /// `v = MDS(u^7) + ARK2[r]` — the inverse S-box input. Recorded for
    /// cross-checking; the AIR recomputes it as a degree-3 expression.
    pub v: [FE; HASH_STATE_FELTS],
    /// `y = v^{1/7}` — this round's output, next round's input. **Committed,
    /// and verified as the FORWARD power**: `(y³)²·y = v` is the spec's own
    /// §4.3 folding trick, degree 3 on both sides.
    pub y: [FE; HASH_STATE_FELTS],
    /// `y²` — committed.
    pub y2: [FE; HASH_STATE_FELTS],
    /// `y³` — committed.
    pub y3: [FE; HASH_STATE_FELTS],
}

/// Every intermediate the AIR witnesses, one entry per round.
pub type RpoWitness = [RpoRound; NUM_ROUNDS];

/// Records the permutation's intermediates for the trace generator.
///
/// ⚠ **Written independently of [`Rpo256::permute`] rather than factored out of
/// it, deliberately** — the same discipline `poseidon::permutation_witness`
/// follows (standing-decisions rule 7). A recording wrapper that `permute`
/// delegated to would make
/// [`tests::the_witness_agrees_with_the_permutation`] a tautology at the moment
/// of the refactor. Both paths are pinned to the SAME external KAT instead.
pub fn permutation_witness(state: [FE; HASH_STATE_FELTS]) -> RpoWitness {
    let zero = [FE::zero(); HASH_STATE_FELTS];
    let mut rounds = [RpoRound {
        u: zero,
        u2: zero,
        u3: zero,
        v: zero,
        y: zero,
        y2: zero,
        y3: zero,
    }; NUM_ROUNDS];
    let mut s = state;
    for (r, round) in rounds.iter_mut().enumerate() {
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
    }
    rounds
}

/// Rescue-Prime Optimized at width 12, rate 8, capacity 4, 7 rounds — RPO256.
pub struct Rpo256;

impl Rpo256 {
    /// `x^7` by square-and-multiply, in exactly the association the AIR's
    /// degree-3 lowering uses (`x²`, `x³ = x²·x`, `x^7 = (x³)²·x`), so the
    /// executor and the chip agree by construction rather than by luck.
    pub(crate) fn sbox(x: &FE) -> FE {
        let x2 = x * x;
        let x3 = &x2 * x;
        let x6 = &x3 * &x3;
        &x6 * x
    }

    /// `x^{1/7}` over the WHOLE STATE — the inverse S-box layer, by
    /// miden-crypto's documented addition chain (72 multiplications for a
    /// ~2^63 exponent, against ~93 for naive square-and-multiply).
    ///
    /// ★ **Whole-state rather than per-element, and that is a measurement, not
    /// a style.** The chain is 72 multiplications each depending on the last,
    /// so a single lane is LATENCY-bound: the Goldilocks multiply's ~2.6 ns of
    /// latency times 72 is the entire cost, and the multiplier pipeline sits
    /// idle between them. The twelve lanes are independent, so running them in
    /// lockstep interleaves twelve chains and fills that pipeline — the same
    /// shape miden's scalar fallback has, and the shape its SVE/AVX2/AVX512
    /// kernels vectorize from.
    ///
    /// This layer is 87% of the permutation and the permutation is the host's
    /// commitment cost, so this loop is the hot one in the whole hash.
    /// [`tests::the_inverse_sbox_chain_agrees_with_the_exponent`] pins it
    /// against `pow(INV_ALPHA)` — a different algorithm for the same number —
    /// and [`tests::the_inverse_sbox_inverts_the_forward_sbox`] pins both
    /// against the property that actually matters.
    pub fn inv_sbox_layer(state: &mut [FE; HASH_STATE_FELTS]) {
        // `base^(2^m) · tail`, lane-wise — the chain's one building block.
        fn exp_acc(
            base: &[FE; HASH_STATE_FELTS],
            tail: &[FE; HASH_STATE_FELTS],
            m: usize,
        ) -> [FE; HASH_STATE_FELTS] {
            let mut acc = *base;
            for _ in 0..m {
                for a in acc.iter_mut() {
                    *a = a.square();
                }
            }
            core::array::from_fn(|i| &acc[i] * &tail[i])
        }

        let t1: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| state[i].square());
        let t2: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| t1[i].square());
        let t3 = exp_acc(&t2, &t2, 3);
        let t4 = exp_acc(&t3, &t3, 6);
        let t5 = exp_acc(&t4, &t4, 12);
        let t6 = exp_acc(&t5, &t3, 6);
        let t7 = exp_acc(&t6, &t6, 31);
        for (i, s) in state.iter_mut().enumerate() {
            let a = (&t7[i].square() * &t6[i]).square().square();
            let b = &(&t1[i] * &t2[i]) * &*s;
            *s = &a * &b;
        }
    }

    /// [`Rpo256::inv_sbox_layer`] for a single element.
    ///
    /// The trace filler and the permutation both go through the layer; this
    /// exists for the tests and for the witness recorder's per-lane reading, and
    /// it is deliberately the SAME chain rather than a second transcription.
    pub fn inv_sbox(x: &FE) -> FE {
        let mut state = [*x; HASH_STATE_FELTS];
        Self::inv_sbox_layer(&mut state);
        state[0]
    }

    /// The circulant MDS product, `out_i = Σ_j MDS_CIRC_ROW[(j − i) mod 12]·s_j`
    /// — the orientation the external KAT pins, and the same one
    /// `poseidon::PoseidonGoldilocks::mds` uses.
    ///
    /// ★ **One `u128` accumulation and one reduction per lane, not twelve field
    /// multiplications.** The MDS constants are all ≤ 26, so every term
    /// `c·s_j` fits in 70 bits and the whole twelve-term row sum fits in 73 —
    /// comfortably inside a `u128` (the bound is asserted in
    /// [`tests::the_mds_row_sum_cannot_overflow_a_u128`]). So the row is
    /// accumulated with no reduction at all and reduced once at the end, using
    /// `2^64 ≡ EPSILON (mod p)`: `hi·2^64 + lo ≡ lo + hi·EPSILON`, and with
    /// `hi < 2^9` the correction term `hi·EPSILON < 2^41` needs no reduction of
    /// its own.
    ///
    /// This matters because the MDS runs FOURTEEN times per permutation and,
    /// once the inverse S-box layer stops dominating, it is the largest
    /// remaining share of the host's commitment cost.
    pub(crate) fn mds(state: &[FE; HASH_STATE_FELTS]) -> [FE; HASH_STATE_FELTS] {
        /// `2^32 − 1`, and `2^64 ≡ EPSILON (mod p)` for the Goldilocks prime.
        /// Written here rather than imported because the field crate keeps its
        /// own copy private; [`tests::the_epsilon_identity_holds`] re-derives it.
        const EPSILON: u64 = 0xFFFF_FFFF;

        let raw: [u64; HASH_STATE_FELTS] = core::array::from_fn(|j| *state[j].value());
        core::array::from_fn(|i| {
            let mut acc: u128 = 0;
            for (j, s) in raw.iter().enumerate() {
                let c = MDS_CIRC_ROW[(j + HASH_STATE_FELTS - i) % HASH_STATE_FELTS];
                acc += (*s as u128) * (c as u128);
            }
            let lo = acc as u64;
            let hi = (acc >> 64) as u64;
            // hi < 2^9, so hi·EPSILON < 2^41 and neither `from` reduces twice.
            FE::from(lo) + FE::from(hi * EPSILON)
        })
    }
}

impl LfmHasher for Rpo256 {
    /// Seven rounds of `MDS → +ARK1 → x^7 → MDS → +ARK2 → x^{1/7}`.
    ///
    /// Note the round STARTS with the linear layer, which is RPO's reordering
    /// of Rescue-Prime (spec §2.4) and is what lets an AIR fold a round into
    /// one row: nothing precedes the first MDS, so the row's input columns feed
    /// it directly.
    fn permute(&self, state: [FE; HASH_STATE_FELTS]) -> [FE; HASH_STATE_FELTS] {
        let mut s = state;
        for r in 0..NUM_ROUNDS {
            s = Self::mds(&s);
            for (lane, v) in s.iter_mut().enumerate() {
                *v += FE::from(ARK1[r][lane]);
            }
            for v in s.iter_mut() {
                *v = Self::sbox(v);
            }
            s = Self::mds(&s);
            for (lane, v) in s.iter_mut().enumerate() {
                *v += FE::from(ARK2[r][lane]);
            }
            Self::inv_sbox_layer(&mut s);
        }
        s
    }

    /// The Merkle-parent capacity: all zeros, so a compress row IS
    /// `Rpo256::merge`.
    fn compress_iv(&self) -> LfmWord {
        domain_iv(DOMAIN_COMPRESS).map(FE::from)
    }

    /// The transcript capacity — `merge_in_domain(·, "LFMT")`.
    fn transcript_iv(&self) -> LfmWord {
        domain_iv(DOMAIN_TRANSCRIPT).map(FE::from)
    }

    /// The leaf capacity — `merge_in_domain(·, "LFML")`.
    ///
    /// This is what retires the weakening `LfmHasher::leaf_out` records for the
    /// single-domain hashers: under RPO a leaf and a parent over the same two
    /// cells are different functions, so the O5 second-preimage split no longer
    /// rests on fixed tree depth alone.
    fn leaf_iv(&self) -> LfmWord {
        domain_iv(DOMAIN_LEAF).map(FE::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// miden-crypto's own `hash_elements` known-answer table — an EXTERNAL
    /// oracle.
    ///
    /// Source: `miden-crypto/src/hash/algebraic_sponge/rescue/rpo/tests.rs`,
    /// `EXPECTED` / `hash_test_vectors`. Entry `n` is the digest of the field
    /// elements `[0, 1, …, n]`. Nothing in this repository produced these
    /// seventy-six numbers.
    ///
    /// Entries 1–7 and 9–19 exercise the padding path (`total_len % 8 ≠ 0`),
    /// entries 8 and 16 the exact-block path, and everything above 8 chains two
    /// permutations through the capacity — so the table pins the sponge's
    /// carry, not only one permutation.
    const MIDEN_HASH_ELEMENTS: [[u64; 4]; 19] = [
        [
            8563248028282119176,
            14757918088501470722,
            14042820149444308297,
            7607140247535155355,
        ],
        [
            8762449007102993687,
            4386081033660325954,
            5000814629424193749,
            8171580292230495897,
        ],
        [
            16710087681096729759,
            10808706421914121430,
            14661356949236585983,
            5683478730832134441,
        ],
        [
            5309818427047650994,
            17172251659920546244,
            8288476618870804357,
            18080473279382182941,
        ],
        [
            3647545403045515695,
            3358383208908083302,
            8797161010298072910,
            2412100201132087248,
        ],
        [
            8409780526028662686,
            214479528340808320,
            13626616722984122219,
            13991752159726061594,
        ],
        [
            4800410126693035096,
            8293686005479024958,
            16849389505608627981,
            12129312715917897796,
        ],
        [
            5421234586123900205,
            9738602082989433872,
            7017816005734536787,
            8635896173743411073,
        ],
        [
            11707446879505873182,
            7588005580730590001,
            4664404372972250366,
            17613162115550587316,
        ],
        [
            6991094187713033844,
            10140064581418506488,
            1235093741254112241,
            16755357411831959519,
        ],
        [
            18007834547781860956,
            5262789089508245576,
            4752286606024269423,
            15626544383301396533,
        ],
        [
            5419895278045886802,
            10747737918518643252,
            14861255521757514163,
            3291029997369465426,
        ],
        [
            16916426112258580265,
            8714377345140065340,
            14207246102129706649,
            6226142825442954311,
        ],
        [
            7320977330193495928,
            15630435616748408136,
            10194509925259146809,
            15938750299626487367,
        ],
        [
            9872217233988117092,
            5336302253150565952,
            9650742686075483437,
            8725445618118634861,
        ],
        [
            12539853708112793207,
            10831674032088582545,
            11090804155187202889,
            105068293543772992,
        ],
        [
            7287113073032114129,
            6373434548664566745,
            8097061424355177769,
            14780666619112596652,
        ],
        [
            17147873541222871127,
            17350918081193545524,
            5785390176806607444,
            12480094913955467088,
        ],
        [
            17273934282489765074,
            8007352780590012415,
            16690624932024962846,
            8137543572359747206,
        ],
    ];

    /// miden's `hash_elements`, in this module's lane convention: capacity lane
    /// 8 takes `len % 8`, the rate is OVERWRITTEN (spec §2.6 — absorption costs
    /// no field operations), the tail is zero-padded, the digest is lanes 0–3.
    ///
    /// Test-only: the machine's sponge lives in the eDSL, not here. This exists
    /// to drive the external vectors through [`Rpo256::permute`].
    fn hash_elements(elements: &[u64]) -> [FE; 4] {
        let mut state = [FE::zero(); HASH_STATE_FELTS];
        state[8] = FE::from((elements.len() % 8) as u64);
        let mut i = 0;
        for e in elements {
            state[i] = FE::from(*e);
            i += 1;
            if i == 8 {
                state = Rpo256.permute(state);
                i = 0;
            }
        }
        if i > 0 {
            while i < 8 {
                state[i] = FE::zero();
                i += 1;
            }
            state = Rpo256.permute(state);
        }
        [state[0], state[1], state[2], state[3]]
    }

    /// ★ The differential this whole module rests on.
    #[test]
    fn the_sponge_matches_the_miden_known_answer_vectors() {
        for (n, want) in MIDEN_HASH_ELEMENTS.iter().enumerate() {
            let input: Vec<u64> = (0..=n as u64).collect();
            let got = hash_elements(&input);
            let want: [FE; 4] = core::array::from_fn(|i| FE::from(want[i]));
            assert_eq!(got, want, "hash_elements of 0..={n} must match miden");
        }
    }

    /// A `Compress` row is a standard `Rpo256::merge`, pinned to the external
    /// table rather than to our own permutation: merging two digest cells is
    /// the same thing as hashing the eight felts they hold, and the vector for
    /// eight elements is `MIDEN_HASH_ELEMENTS[7]`.
    ///
    /// This is what makes the zero compress IV a checkable claim instead of a
    /// convention: any implementation of RPO256 anywhere computes this digest
    /// for this Merkle parent.
    #[test]
    fn a_compress_row_is_a_standard_rpo256_merge() {
        let a: LfmWord = core::array::from_fn(|i| FE::from(i as u64));
        let b: LfmWord = core::array::from_fn(|i| FE::from(i as u64 + 4));
        let got = Rpo256.compress(&a, &b);
        let want: [FE; 4] = core::array::from_fn(|i| FE::from(MIDEN_HASH_ELEMENTS[7][i]));
        assert_eq!(got, want, "compress(a, b) must be Rpo256::merge([a, b])");
    }

    /// `ALPHA · INV_ALPHA ≡ 1 (mod p − 1)`, re-derived rather than trusted.
    #[test]
    fn the_inverse_exponent_inverts_alpha() {
        const P_MINUS_ONE: u128 = (1u128 << 64) - (1u128 << 32);
        assert_eq!(
            (ALPHA as u128 * INV_ALPHA as u128) % P_MINUS_ONE,
            1,
            "INV_ALPHA must be ALPHA's inverse in the exponent group"
        );
    }

    /// The S-box exponent must be coprime to `p − 1`, or it is not a
    /// permutation — the same trap `poseidon` asserts against, checked here so
    /// this module stands on its own.
    #[test]
    fn the_sbox_exponent_is_coprime_to_the_group_order() {
        const P_MINUS_ONE: u128 = (1u128 << 64) - (1u128 << 32);
        fn gcd(a: u128, b: u128) -> u128 {
            if b == 0 { a } else { gcd(b, a % b) }
        }
        assert_eq!(gcd(ALPHA as u128, P_MINUS_ONE), 1);
    }

    /// The addition chain must compute the exponent it claims. Checked against
    /// `pow`, which is a different algorithm for the same number.
    #[test]
    fn the_inverse_sbox_chain_agrees_with_the_exponent() {
        for seed in 0..16u64 {
            let x = FE::from(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(7));
            assert_eq!(
                Rpo256::inv_sbox(&x),
                x.pow(INV_ALPHA),
                "the chain must equal x^INV_ALPHA at seed {seed}"
            );
        }
    }

    /// The property that actually matters, and the one the AIR verifies in the
    /// forward direction: `(x^{1/7})^7 = x`.
    #[test]
    fn the_inverse_sbox_inverts_the_forward_sbox() {
        for seed in 0..16u64 {
            let x = FE::from(seed.wrapping_mul(0xD1B5_4A32_D192_ED03).wrapping_add(3));
            let y = Rpo256::inv_sbox(&x);
            assert_eq!(Rpo256::sbox(&y), x, "sbox(inv_sbox(x)) must be x at {seed}");
            assert_eq!(Rpo256::inv_sbox(&Rpo256::sbox(&x)), x);
        }
        // Zero is the fixed point the all-zero padding row rides on.
        assert_eq!(Rpo256::inv_sbox(&FE::zero()), FE::zero());
    }

    /// The `u128` accumulation in [`Rpo256::mds`] must not be able to overflow,
    /// and the margin must be large rather than lucky.
    #[test]
    fn the_mds_row_sum_cannot_overflow_a_u128() {
        let max_c = MDS_CIRC_ROW.iter().copied().max().expect("twelve entries");
        // Every stored value is < 2^64 (the field allows non-canonical storage).
        let max_row_sum = (HASH_STATE_FELTS as u128) * (max_c as u128) * ((1u128 << 64) - 1);
        assert!(
            max_row_sum.checked_add(1).is_some(),
            "the twelve-term row sum must fit a u128"
        );
        // The carry the reduction multiplies must stay small enough that
        // `hi * EPSILON` cannot itself overflow a u64.
        let max_hi = max_row_sum >> 64;
        assert!(
            max_hi * 0xFFFF_FFFF < (1u128 << 64),
            "hi·EPSILON must fit a u64 without its own reduction"
        );
    }

    /// `2^64 ≡ EPSILON (mod p)` — the identity [`Rpo256::mds`]'s single
    /// reduction rests on, re-derived rather than trusted to a comment.
    #[test]
    fn the_epsilon_identity_holds() {
        const P: u128 = (1u128 << 64) - (1u128 << 32) + 1;
        assert_eq!((1u128 << 64) % P, 0xFFFF_FFFF);
    }

    #[test]
    fn the_round_constant_tables_have_one_row_per_round() {
        assert_eq!(ARK1.len(), NUM_ROUNDS);
        assert_eq!(ARK2.len(), NUM_ROUNDS);
        assert_eq!(NUM_ROUNDS, 7);
        assert_eq!(HALVES_PER_ROUND, 2);
    }

    /// The three socket domains must be three different capacities, or the
    /// separation the module header claims does not exist.
    #[test]
    fn the_three_socket_domains_are_distinct() {
        let c = Rpo256.compress_iv();
        let t = Rpo256.transcript_iv();
        let l = Rpo256.leaf_iv();
        assert_ne!(c, t);
        assert_ne!(c, l);
        assert_ne!(t, l);
        // The padding lane stays zero in every domain — the tag goes in lane 9.
        for iv in [&c, &t, &l] {
            assert_eq!(iv[CAPACITY_PAD_LANE], FE::zero());
            assert_eq!(iv[2], FE::zero());
            assert_eq!(iv[3], FE::zero());
        }
        assert_eq!(c[CAPACITY_DOMAIN_LANE], FE::zero());
    }

    /// A transcript step and a Merkle parent over the SAME two cells must be
    /// different digests. This is the weakening `LfmHasher::transcript_out`
    /// records for the single-domain hashers, asserted as retired here.
    #[test]
    fn a_transcript_step_is_not_a_merkle_parent() {
        let a: LfmWord = core::array::from_fn(|i| FE::from(11 * i as u64 + 1));
        let b: LfmWord = core::array::from_fn(|i| FE::from(7 * i as u64 + 2));
        let parent = Rpo256.compress(&a, &b);
        let step = Rpo256.transcript(&a, &b);
        let leaf = Rpo256.leaf(&a, &b);
        assert_ne!(parent, step);
        assert_ne!(parent, leaf);
        assert_ne!(step, leaf);
    }

    /// The witness's last round must reproduce the SAME external vector
    /// `permute` is pinned to — not `permute`'s output, which would only say
    /// the two agree. This is the absolute pin on the recording path.
    #[test]
    fn the_witness_final_round_matches_the_miden_known_answer_vector() {
        // The eight-element vector is one permutation of `[0..8 ‖ 0,0,0,0]`.
        let input: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| {
            if i < 8 {
                FE::from(i as u64)
            } else {
                FE::zero()
            }
        });
        let w = permutation_witness(input);
        for (i, want) in MIDEN_HASH_ELEMENTS[7].iter().enumerate() {
            assert_eq!(
                w[NUM_ROUNDS - 1].y[i],
                FE::from(*want),
                "witness digest lane {i} must match miden"
            );
        }
    }

    /// A genuine differential: two independently written round loops, neither
    /// delegating to the other (rule 7), on inputs the KAT does not cover.
    #[test]
    fn the_witness_agrees_with_the_permutation() {
        for seed in 0..8u64 {
            let input: [FE; HASH_STATE_FELTS] =
                core::array::from_fn(|i| FE::from(seed.wrapping_mul(0x9E37_79B9) + i as u64));
            let w = permutation_witness(input);
            assert_eq!(
                w[NUM_ROUNDS - 1].y,
                Rpo256.permute(input),
                "witness and permute must agree at seed {seed}"
            );
        }
    }

    /// The intermediates must be the ones the AIR constrains: `u2 = u²`,
    /// `u3 = u²·u`, `y2 = y²`, `y3 = y²·y`, and the fold `(y3)²·y = v`.
    #[test]
    fn the_witness_records_the_degree_three_association() {
        let input: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| FE::from(3 * i as u64 + 1));
        let w = permutation_witness(input);
        let mut s = input;
        for (r, round) in w.iter().enumerate() {
            let mixed = Rpo256::mds(&s);
            for lane in 0..HASH_STATE_FELTS {
                assert_eq!(
                    round.u[lane],
                    &mixed[lane] + FE::from(ARK1[r][lane]),
                    "round {r} lane {lane} u"
                );
                assert_eq!(round.u2[lane], &round.u[lane] * &round.u[lane]);
                assert_eq!(round.u3[lane], &round.u2[lane] * &round.u[lane]);
                assert_eq!(round.y2[lane], &round.y[lane] * &round.y[lane]);
                assert_eq!(round.y3[lane], &round.y2[lane] * &round.y[lane]);
                // ★ the fold: the inverse S-box verified as the FORWARD power.
                assert_eq!(
                    &(&round.y3[lane] * &round.y3[lane]) * &round.y[lane],
                    round.v[lane],
                    "round {r} lane {lane} must satisfy (y³)²·y = v"
                );
            }
            // `v` is the MDS of the forward S-box outputs, plus ARK2.
            let x: [FE; HASH_STATE_FELTS] =
                core::array::from_fn(|i| &(&round.u3[i] * &round.u3[i]) * &round.u[i]);
            let mixed = Rpo256::mds(&x);
            for lane in 0..HASH_STATE_FELTS {
                assert_eq!(round.v[lane], &mixed[lane] + FE::from(ARK2[r][lane]));
            }
            s = round.y;
        }
    }

    /// ★ The padding row. An all-zero row must satisfy every constraint the
    /// chip emits, which is what lets the AIR carry no `IS_REAL` gate — and
    /// the property rests on `0^7 = 0` and `0^{1/7} = 0` at every lane of every
    /// round.
    #[test]
    fn an_all_zero_state_permutes_to_zero_when_the_constants_are_gated_off() {
        // With the mode sum `m = 0` the chip scales every round constant to
        // zero, so the host analogue is the constant-free permutation.
        let mut s = [FE::zero(); HASH_STATE_FELTS];
        for _ in 0..NUM_ROUNDS {
            s = Rpo256::mds(&s);
            for v in s.iter_mut() {
                *v = Rpo256::sbox(v);
            }
            s = Rpo256::mds(&s);
            for v in s.iter_mut() {
                *v = Rpo256::inv_sbox(v);
            }
        }
        assert_eq!(s, [FE::zero(); HASH_STATE_FELTS]);
    }
}

/// Host throughput of the RPO permutation — the blueprint's second-riskiest
/// unknown, measured.
///
/// `#[ignore]`d because it is a timing measurement, not a property: it prints
/// numbers and asserts only a floor loose enough that no honest machine trips
/// it. Run it with
/// `cargo test --release -p lambda-vm-prover --lib rpo_throughput -- --ignored --nocapture`.
///
/// **Why this number matters.** Under BLAKE3 the host's commitment hashing is a
/// rounding error next to the LDE. RPO is native field arithmetic — roughly
/// 8.4k Goldilocks multiplications per permutation, most of it the two inverse
/// S-box layers — so commitment hashing becomes a phase with a name. What this
/// measures is how big a phase.
#[cfg(test)]
mod throughput {
    use super::*;
    use std::time::Instant;

    /// Permutations timed. Small enough to run in seconds, large enough that
    /// the timer's resolution is not the measurement.
    const PERMUTATIONS: usize = 20_000;

    /// Felts one permutation absorbs at rate 8 — the sponge's throughput unit.
    const RATE: usize = 8;

    #[test]
    #[ignore]
    fn rpo_throughput() {
        // A chained input so the optimizer cannot hoist the permutation out of
        // the loop: each iteration's input depends on the last one's output.
        let mut state: [FE; HASH_STATE_FELTS] = core::array::from_fn(|i| FE::from(i as u64 + 1));
        let start = Instant::now();
        for _ in 0..PERMUTATIONS {
            state = Rpo256.permute(state);
        }
        let elapsed = start.elapsed();
        // Consume the result so the loop is not dead code.
        assert_ne!(state[0], FE::zero(), "the chain must not collapse to zero");

        let per_perm_ns = elapsed.as_nanos() as f64 / PERMUTATIONS as f64;
        let perms_per_sec = 1e9 / per_perm_ns;
        let felts_per_sec = perms_per_sec * RATE as f64;

        println!("RPO256 host permutation, single thread:");
        println!("  {per_perm_ns:.0} ns / permutation");
        println!("  {:.2} M permutations / s", perms_per_sec / 1e6);
        println!(
            "  {:.1} M felts / s absorbed at rate {RATE} ({:.0} MB/s of field data)",
            felts_per_sec / 1e6,
            felts_per_sec * 8.0 / 1e6
        );

        // The inverse S-box is the cost centre; price it alone so a regression
        // in the addition chain is attributable rather than diffuse.
        let mut layer: [FE; HASH_STATE_FELTS] =
            core::array::from_fn(|i| FE::from(0x9E37_79B9_7F4A_7C15u64 + i as u64));
        let layer_iters = PERMUTATIONS * NUM_ROUNDS;
        let start = Instant::now();
        for _ in 0..layer_iters {
            Rpo256::inv_sbox_layer(&mut layer);
        }
        let inv_elapsed = start.elapsed();
        assert_ne!(layer[0], FE::zero());
        let layer_ns = inv_elapsed.as_nanos() as f64 / layer_iters as f64;
        println!(
            "  inverse S-box LAYER (12 lanes): {layer_ns:.0} ns; {NUM_ROUNDS} per permutation \
             ⇒ {:.0} ns, {:.0}% of the permutation ({:.1} ns per lane)",
            layer_ns * NUM_ROUNDS as f64,
            100.0 * layer_ns * NUM_ROUNDS as f64 / per_perm_ns,
            layer_ns / HASH_STATE_FELTS as f64
        );

        // The other half. Two MDS products and one forward S-box layer per round
        // is what is left once the inverse layer stops dominating, and knowing
        // which of them to attack next is the point of measuring both.
        let mut mds_state: [FE; HASH_STATE_FELTS] =
            core::array::from_fn(|i| FE::from(0xD1B5_4A32_D192_ED03u64 + i as u64));
        let mds_iters = PERMUTATIONS * NUM_ROUNDS * 2;
        let start = Instant::now();
        for _ in 0..mds_iters {
            mds_state = Rpo256::mds(&mds_state);
        }
        let mds_elapsed = start.elapsed();
        assert_ne!(mds_state[0], FE::zero());
        let mds_ns = mds_elapsed.as_nanos() as f64 / mds_iters as f64;
        println!(
            "  MDS product: {mds_ns:.0} ns; {} per permutation ⇒ {:.0} ns, {:.0}% of the permutation",
            NUM_ROUNDS * 2,
            mds_ns * (NUM_ROUNDS * 2) as f64,
            100.0 * mds_ns * (NUM_ROUNDS * 2) as f64 / per_perm_ns
        );

        let mut fwd: [FE; HASH_STATE_FELTS] =
            core::array::from_fn(|i| FE::from(0x1234_5678_9ABC_DEF0u64 + i as u64));
        let start = Instant::now();
        for _ in 0..(PERMUTATIONS * NUM_ROUNDS) {
            for v in fwd.iter_mut() {
                *v = Rpo256::sbox(v);
            }
        }
        let fwd_elapsed = start.elapsed();
        assert_ne!(fwd[0], FE::zero());
        let fwd_ns = fwd_elapsed.as_nanos() as f64 / (PERMUTATIONS * NUM_ROUNDS) as f64;
        println!(
            "  forward S-box layer: {fwd_ns:.0} ns; {NUM_ROUNDS} per permutation ⇒ {:.0} ns, {:.0}% of the permutation",
            fwd_ns * NUM_ROUNDS as f64,
            100.0 * fwd_ns * NUM_ROUNDS as f64 / per_perm_ns
        );

        // Attribution, so a bad number is a bad number SOMEWHERE rather than a
        // verdict on RPO. A permutation is ~8.4k Goldilocks multiplications, so
        // if the field multiply is slow the permutation is slow and the hash
        // has nothing to do with it.
        let mut a = FE::from(0x9E37_79B9_7F4A_7C15u64);
        let b = FE::from(0xD1B5_4A32_D192_ED03u64);
        const MULS: usize = 20_000_000;
        let start = Instant::now();
        for _ in 0..MULS {
            a = &a * &b;
        }
        let mul_elapsed = start.elapsed();
        assert_ne!(a, FE::zero());
        let mul_ns = mul_elapsed.as_nanos() as f64 / MULS as f64;
        println!("  Goldilocks FieldElement multiply: {mul_ns:.2} ns");
        println!(
            "  ⇒ 72 SERIAL multiplies would be {:.0} ns of dependent field work; the layer \
             does 12 such chains in {layer_ns:.0} ns, i.e. {:.1}× the serial rate",
            72.0 * mul_ns,
            12.0 * 72.0 * mul_ns / layer_ns
        );

        // BLAKE3 on the SAME machine, at the shape a Merkle parent takes: two
        // 32-byte nodes. This is the ratio that matters for the host commit
        // phase, and measuring both here means it is one machine's number and
        // not two quoted from different pages.
        let left = [0x5Au8; 32];
        let mut right = [0xA5u8; 32];
        const PARENTS: usize = 2_000_000;
        let start = Instant::now();
        for _ in 0..PARENTS {
            right = crypto::hash::blake3::chain::blake3_parent(&left, &right);
        }
        let b3_elapsed = start.elapsed();
        assert_ne!(right, [0u8; 32]);
        let b3_ns = b3_elapsed.as_nanos() as f64 / PARENTS as f64;
        println!("  BLAKE3 64-byte parent on this machine: {b3_ns:.0} ns");
        println!(
            "  ⇒ RPO / BLAKE3 per 2-to-1 compression: {:.0}×",
            per_perm_ns / b3_ns
        );

        // A floor no honest machine trips. It exists so the test is a test and
        // not only a print; the numbers above are the point.
        assert!(
            perms_per_sec > 5_000.0,
            "under 5k permutations/s means something is very wrong, not slow"
        );
    }
}
