//! `fold_coset` as a machine leg — the arithmetic half of the query phase.
//!
//! `whir_commit::fold_coset` (`crypto/multilinear/src/whir_commit.rs:475-511`)
//! folds one opened block down to the single value it contributes, and the
//! verifier runs it once per query per round: at the round's own successor
//! block in `whir_round::verify` (`:159`) and against the final constant in
//! `verify_final` (`:1138`). The hashing half is [`super::whir_open`] beside
//! it.
//!
//! Each level halves the block by `fold_block_level` (`:434-465`):
//!
//! ```text
//!     out[t] = ½·(a + b) + α·(½·x⁻¹)·(a − b),   a = v[t], b = v[t + half]
//! ```
//!
//! with `x` running over the coset: slot `t` sits at `g^p·η^t`, where `p` is
//! the block's position in the level's domain and `η` is that level's stride.
//!
//! # ★ Why only ONE exponentiation, and the identity that buys it
//!
//! `x` starts at `generator^position`, which is RUNTIME — `position` is the
//! query index the transcript drew — so it is built from the index's BITS by
//! [`super::edsl::pow_bits`], whose factors `g^(2^i)` are program constants.
//!
//! Every LATER level is free of a second exponentiation. The host squares the
//! domain and reduces the position into it (`:501-507`), so its point is
//! `(g²)^(p mod n/2)`; and
//!
//! ```text
//!     (g²)^(p mod n/2) = g^(2p − k·n) = g^(2p) = (g^p)²
//! ```
//!
//! because `g` has order `n` (`Domain::squared` squares the generator and drops
//! one from the log size, `whir.rs:50-58`). So the emitter squares the point it
//! already has, and a squaring is one row where a second `pow_bits` would be
//! `2·index_bits`. `η` is a program constant at every level, because the domain
//! and the block width are both emit-time.
//!
//! ⚠ The reduction reads as the hard part of that identity and is not part of
//! it at all: `Domain::new` takes a PRIMITIVE `2^log_size`-th root
//! (`whir.rs:36-47`), so each level's generator has order exactly that level's
//! size and `g_l^p = g_l^(p mod n_l)` for every `p`. The host's `position %=`
//! (`whir_commit.rs:502`, `:507`) keeps the exponent small for `pow` and
//! changes no value. Recorded because an earlier draft of this doc — and a test
//! written against it — treated the reduction as something the emitter had to
//! survive.
//!
//! # The base round
//!
//! Round 0's current codeword is base-field, and the host folds its first level
//! in the base field before the multiply by `α` lifts it. This emits that level
//! in the extension instead: the values agree exactly — a base element embedded
//! in the cubic extension is `(v, 0, 0)`, and every operation here is the
//! embedding of the host's — so it is a COST difference and not a value one. It
//! is worth naming rather than hiding: an `LFM_XALU` row is 18 base-equivalent
//! cells against `LFM_BALU`'s 10, so folding round 0's first level in the base
//! field would save `8 · (18 − 10)` cells a query and is a lever for the
//! optimisation rounds, not a correctness matter.
//!
//! ⚠ Lifting a HINTED base value is only sound because the block's leaf hash
//! pins its upper lanes to zero — see [`super::whir_open`]'s module doc, which
//! carries the obligation.
//!
//! # Provenance
//!
//! The shape of this emitter follows an uncommitted draft left in lane V1b's
//! worktree. The closed form below, its terms, its gates and its mutations are
//! this lane's own: the form was re-derived from `fold_block_level` before the
//! draft's was read back, and the two agree.

use multilinear::whir::Domain;

use crate::tables::types::{FE, GoldilocksField};

use super::builder::{Bit, Ext, LfmBuilder};
use super::edsl::pow_bits;
use super::word::{LfmWord, base_word};

/// INSTRUCTIONS [`emit_fold_coset`] emits for a block of `block` values over
/// `log2(block)` levels, with a query index of `index_bits` bits.
///
/// Every term by the shape it comes from:
///
/// - `2·index_bits` for the ONE `pow_bits` — a `Select` and a `Mul` a bit
///   (`edsl.rs:856-880`);
/// - SIX rows an output slot: `a + b`, its halving, `½·x⁻¹` (ONE `Div`, because
///   `Div` is a reversed-multiply constraint so the halving rides along with
///   the reciprocal — `chips.rs:175`), `a − b`, its scaling, and the `MulAdd`
///   that joins the halves at `α`. A block of `2^levels` has `block − 1` output
///   slots across all its levels, so that is `6·(block − 1)`;
/// - ONE step of `x` along the coset per slot except the last of its level,
///   which is `(block − 1) − levels`;
/// - ONE squaring per level after the first, which is `levels − 1`.
///
/// The last two cancel to a constant: `−levels + levels − 1 = −1`. So
///
/// ```text
///     2·index_bits + 7·(block − 1) − 1
/// ```
///
/// and the cancellation is why the form has no `levels` in it even though two
/// of its terms do.
pub const fn fold_coset_rows(block: usize, index_bits: usize) -> usize {
    if block <= 1 {
        // `fold_coset` returns `values[0]` lifted when there is nothing to fold
        // (`whir_commit.rs:492-494`) and the emitter returns the wire. No chain
        // reaches it — a round folds at least one variable — and it is stated
        // rather than left to underflow `block − 1`.
        return 0;
    }
    2 * index_bits + 7 * (block - 1) - 1
}

/// `LFM_CONST` rows a fold interns.
///
/// The constants a fold NAMES are `pow_bits`' scale of one, one factor
/// `g^(2^i)` per index bit, the half, and one coset stride per level that has
/// two slots to step between. That count is an upper bound and not the answer,
/// because **the strides and the factors are powers of the same generator and
/// DO collide** — found by this form's own assertion failing, not predicted.
///
/// Every constant but the half is `g^e` for an exponent the shape fixes:
///
/// ```text
///     scale    e = 0
///     factor i e = 2^i                        for i < index_bits
///     stride l e = (N / block) · 2^l          for l < levels − 1
/// ```
///
/// with `N` the domain's size — level `l` has `N/2^l` points and `block/2^l`
/// values, so its stride exponent is `(N/block)·2^l` measured in the ORIGINAL
/// generator. Two constants are the same felt exactly when their exponents
/// agree modulo `N`, so the count is the size of that exponent set plus one for
/// the half. At a 2^8 domain with a block of sixteen and six index bits the
/// strides are `g^16, g^32, g^64` and the factors already include `g^16` and
/// `g^32`: eleven names, nine constants.
///
/// ⚠ This is reached in production, not only in a test. At `S = 25` a round's
/// domain is `2^(27 − 4r)` and its index is `D_r − 4` bits wide, so the first
/// stride `g^16` is the index factor `2^4` in every round whose index is more
/// than four bits — which is most of them.
///
/// The arithmetic here is over integer exponents modulo `N`. Nothing evaluates
/// the field, so this is a derivation from the shape rather than a second copy
/// of what the emitter interns.
///
/// ★ The half is counted as one more and assumed distinct from every `g^e`.
/// That is an assumption about a discrete log, not a proof; it is what the
/// tests would catch if it ever failed, in the same way the stride collision
/// was caught.
pub fn fold_coset_consts(log_domain: usize, levels: usize, index_bits: usize) -> usize {
    if levels == 0 {
        return 0;
    }
    fold_coset_exponents(log_domain, levels, index_bits).len() + 1
}

/// ⛔⛔ **NEVER UNION THIS ACROSS DOMAINS**, which is the warning the Newton
/// count needed in the other direction and this one needs in its own.
///
/// `sumcheck_round_consts`' pairs NEST in the degree, so one call at a
/// program's maximum is its whole pool. These do NOT nest: the exponent set is
/// the same integers under a DIFFERENT generator, so two chains over different
/// domains intern DIFFERENT field elements for the same exponents. A program's
/// fold pool is a genuine UNION over its distinct domains — a maximum of
/// anything is wrong here — and a count cannot express a union any more than it
/// could express the other.
///
/// ⇒ [`fold_coset_constants`] is the program-level form; this one is correct
/// for exactly one thing, which is what ONE fold over ONE domain interns.
///
/// The exponents `g` is raised to, deduplicated — the shape half of what
/// [`emit_fold_coset`] interns, over integer exponents modulo `N`.
///
/// - `0`, the base [`pow_bits`] starts its accumulator at;
/// - `2^i` for each index bit, the factors `pow_bits` multiplies in;
/// - `(N / block) << l` at each level that has two slots to step between —
///   level `l`'s stride, which is `g^{2^{l + log_domain − levels}}` once the
///   squared domain's generator is unfolded.
fn fold_coset_exponents(log_domain: usize, levels: usize, index_bits: usize) -> Vec<u128> {
    if levels == 0 {
        return Vec::new();
    }
    let n = 1u128 << log_domain;
    let block = 1u128 << levels;
    let mut exponents = vec![0u128];
    for i in 0..index_bits {
        exponents.push((1u128 << i) % n);
    }
    for l in 0..levels - 1 {
        exponents.push(((n / block) << l) % n);
    }
    exponents.sort_unstable();
    exponents.dedup();
    exponents
}

/// ★★ THE VALUES ONE FOLD INTERNS — the form a program-level pool can consume.
///
/// [`fold_coset_consts`] counts these and a count is exactly what a pool cannot
/// take: the builder interns on the canonical word, so two folds sharing a
/// value pay for it once and adding their counts charges it twice. Same defect
/// as the sumcheck round's, found the same way — as an unnamed remainder in an
/// assembled program's pool.
///
/// ★ AND IT RETIRES AN ASSUMPTION. The count's own doc says its last term —
/// `two_inv` — is "counted as one more and assumed distinct from every `g^e`",
/// and calls that "an assumption about a discrete log, not a proof". This form
/// does not need it: the words are deduplicated BY VALUE, so a collision makes
/// the pool one word smaller and the count identity is where it shows.
pub fn fold_coset_constants(
    domain: &Domain<GoldilocksField>,
    levels: usize,
    index_bits: usize,
) -> Vec<LfmWord> {
    if levels == 0 {
        return Vec::new();
    }
    let log_domain = domain.size().trailing_zeros() as usize;
    let generator = domain.generator();
    let two_inv = (FE::one() + FE::one())
        .inv()
        .expect("2 is invertible in Goldilocks");

    let mut words: Vec<LfmWord> = Vec::new();
    for exponent in fold_coset_exponents(log_domain, levels, index_bits) {
        let word = base_word(generator.pow(exponent as u64));
        if !words.contains(&word) {
            words.push(word);
        }
    }
    let half = base_word(two_inv);
    if !words.contains(&half) {
        words.push(half);
    }
    words
}

/// ★ `whir_commit::fold_coset`, emitted.
///
/// `values` are the block's openings in the order the host holds them —
/// `coset_of`'s order, which is what `CosetOpening::values` carries and what the
/// leaf hashed. A round-0 block is base-field on the host and is lifted by the
/// caller; see the module doc for why that lift is sound and what it costs.
///
/// Panics when the block is not `2^alphas.len()`, which the host returns an
/// error for. The block's width is an emit-time constant here, so a mismatch is
/// a bug in the emitter rather than a condition to carry at runtime — the
/// `epoch_verify.rs:171-179` idiom, where the absence of a second value
/// replaces a check somebody could forget.
pub fn emit_fold_coset(
    b: &mut LfmBuilder,
    values: &[Ext],
    domain: &Domain<GoldilocksField>,
    index_bits: &[Bit],
    alphas: &[Ext],
) -> Ext {
    assert_eq!(
        values.len(),
        1usize << alphas.len(),
        "a block holds one value per folded variable"
    );
    if alphas.is_empty() {
        return values[0];
    }

    let two_inv = (FE::one() + FE::one())
        .inv()
        .expect("2 is invertible in Goldilocks");
    let half_const = b.felt_const(two_inv);

    // `x` at the first level: `generator^position`, from the index's bits. The
    // factors are program constants; the bits are the transcript's.
    let generator = domain.generator();
    let factors: Vec<FE> = (0..index_bits.len())
        .map(|i| generator.pow(1u64 << i))
        .collect();
    let mut x = pow_bits(b, index_bits, &factors, FE::one());

    let mut current: Vec<Ext> = values.to_vec();
    let mut current_domain = domain.clone();

    for (level, alpha) in alphas.iter().enumerate() {
        if level > 0 {
            // The squared domain's point is the previous point squared — the
            // identity in the module doc, and the whole reason there is one
            // `pow_bits` and not `levels` of them.
            x = b.mul(x, x);
        }
        let half = current.len() / 2;
        // This level's coset stride, a program constant; only needed when the
        // level has two slots to step between.
        let stride = (half > 1).then(|| {
            let eta = current_domain
                .generator()
                .pow((current_domain.size() / current.len()) as u64);
            b.felt_const(eta)
        });

        let mut next = Vec::with_capacity(half);
        let mut point = x;
        for t in 0..half {
            let (a, c) = (current[t], current[t + half]);
            let sum = b.eadd(a, c);
            let even = b.emul_base(sum, half_const);
            // `½·x⁻¹` in ONE row. A domain element is never zero, so `Div`'s
            // `0/0 = 1` convention is unreachable and the reciprocal is the
            // only satisfying assignment.
            let point_inv = b.div(half_const, point);
            let difference = b.esub(a, c);
            let odd = b.emul_base(difference, point_inv);
            next.push(b.emul_add(odd, *alpha, even));
            if t + 1 < half {
                point = b.mul(point, stride.expect("a level with two slots has a stride"));
            }
        }
        current = next;
        current_domain = current_domain
            .squared()
            .expect("a domain of at least two points squares");
    }

    current[0]
}
