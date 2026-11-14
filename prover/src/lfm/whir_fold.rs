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

use multilinear::whir::Domain;

use crate::tables::types::{FE, GoldilocksField};

use super::builder::{Bit, Ext, LfmBuilder};
use super::edsl::{pow_bits, pow_bits_alu};
use super::word::{LfmWord, base_word};

// ============================================================================
// The lean fold
// ============================================================================

/// A round's folding challenges as the lean fold consumes them: `½·α` per level,
/// computed ONCE per round rather than once per query, plus the half itself.
pub struct LeanAlphas {
    half_alphas: Vec<Ext>,
    half: super::builder::Felt,
}

/// [`LeanAlphas`] for one round: one `MulBase` a level.
pub fn prepare_lean_alphas(b: &mut LfmBuilder, alphas: &[Ext]) -> LeanAlphas {
    let half = b.felt_const(two_inv());
    LeanAlphas {
        half_alphas: alphas.iter().map(|a| b.emul_base(*a, half)).collect(),
        half,
    }
}

fn two_inv() -> FE {
    (FE::one() + FE::one())
        .inv()
        .expect("2 is invertible in Goldilocks")
}

/// ★ `whir_commit::fold_coset`, emitted lean — the same value, in three
/// extension rows a folded value and none in the base field.
///
/// Each level's output is rewritten around the SECOND value of its pair:
///
/// ```text
///     ½·(a + c) + ½·α·x⁻¹·(a − c)  =  (a − c)·w + c,    w = ½ + ½·α·x⁻¹
/// ```
///
/// and along a level `x_t = x_0·η^t`, so `½·α·x_t⁻¹ = u·η^(−t)` with
/// `u = ½·α·x_0⁻¹` — ONE extension product a level. Per slot that leaves
/// `w = u·η^(−t) + ½` (a `MulAdd` against the program constant `η^(−t)`),
/// `a − c`, and the `MulAdd` that joins them: three `LFM_XALU` rows, where the
/// form above as written pays five plus a base `Div` for `½·x⁻¹` and a base
/// `Mul` to step `x`. No reciprocal is taken at all: `x_0⁻¹ = g^(−p)` comes
/// straight out of [`pow_bits`] over the INVERSE factors `g^(−2^i)`, and each
/// level after the first squares it (`(x²)⁻¹ = (x⁻¹)²`, the module doc's
/// identity inverted).
///
/// Rows: `2·index_bits` (the point) + `levels − 1` (squarings) + `levels` (the
/// `u` of each level) + `3·(block − 1)`; see [`fold_coset_rows_lean`]. The
/// `½·α` of [`prepare_lean_alphas`] are per ROUND and not counted here.
pub fn emit_fold_coset_lean(
    b: &mut LfmBuilder,
    values: &[Ext],
    domain: &Domain<GoldilocksField>,
    index_bits: &[Bit],
    prepared: &LeanAlphas,
) -> Ext {
    let levels = prepared.half_alphas.len();
    assert_eq!(
        values.len(),
        1usize << levels,
        "a block holds one value per folded variable"
    );
    if levels == 0 {
        return values[0];
    }

    // `x_0⁻¹ = g^(−p)`, from the index's bits against the inverse factors.
    let generator_inv = domain
        .generator()
        .inv()
        .expect("a domain generator is nonzero");
    let factors: Vec<FE> = (0..index_bits.len())
        .map(|i| generator_inv.pow(1u64 << i))
        .collect();
    // A Poseidon1 leaf's `Select` table binds its leaves (I-WHIR-P1 §S5.4): its
    // point reads the bits through the base ALU instead.
    let mut x_inv = match b.wrap_hash() {
        super::edsl::WrapHash::Poseidon1 => pow_bits_alu(b, index_bits, &factors),
        _ => pow_bits(b, index_bits, &factors, FE::one()),
    };
    let half = prepared.half.as_ext();

    let mut current: Vec<Ext> = values.to_vec();
    let mut current_domain = domain.clone();
    for (level, half_alpha) in prepared.half_alphas.iter().enumerate() {
        if level > 0 {
            x_inv = b.mul(x_inv, x_inv);
        }
        let pairs = current.len() / 2;
        // This level's stride `η`, inverted: `η^(−t)` is a program constant per
        // slot, so the point never steps at run time.
        let eta_inv = current_domain
            .generator()
            .pow((current_domain.size() / current.len()) as u64)
            .inv()
            .expect("a coset stride is nonzero");
        let u = b.emul_base(*half_alpha, x_inv);

        let mut next = Vec::with_capacity(pairs);
        for t in 0..pairs {
            let (a, c) = (current[t], current[t + pairs]);
            let w = if t == 0 {
                b.eadd(u, half)
            } else {
                let step = b.felt_const(eta_inv.pow(t as u64));
                b.emul_add(u, step.as_ext(), half)
            };
            let difference = b.esub(a, c);
            next.push(b.emul_add(difference, w, c));
        }
        current = next;
        current_domain = current_domain
            .squared()
            .expect("a domain of at least two points squares");
    }

    current[0]
}

/// INSTRUCTIONS [`emit_fold_coset_lean`] emits for a block of `block` values
/// with a query index of `index_bits` bits:
///
/// ```text
///     2·index_bits + 3·(block − 1) + 2·levels − 1
/// ```
///
/// `2·index_bits` for the point, three rows an output slot (`block − 1` of them
/// across the levels), one `u` a level and one squaring a level after the first.
pub const fn fold_coset_rows_lean(block: usize, index_bits: usize) -> usize {
    if block <= 1 {
        // `fold_coset` returns `values[0]` lifted when there is nothing to fold
        // and the emitter returns the wire. No chain reaches it — a round folds
        // at least one variable — and it is stated rather than left to
        // underflow `block − 1`.
        return 0;
    }
    let levels = block.trailing_zeros() as usize;
    2 * index_bits + 3 * (block - 1) + 2 * levels - 1
}

/// Rows [`prepare_lean_alphas`] emits for a round of `levels` levels: one
/// `MulBase` each.
pub const fn lean_prepare_rows(levels: usize) -> usize {
    levels
}

/// ★ THE VALUES ONE LEAN FOLD INTERNS, deduplicated by value: `g^(−e)` for
/// `e` in `{0} ∪ {2^i : i < index_bits}` (the scale and the inverse factors of
/// the point) and `{s·N/block : 1 ≤ s < block/2}` (every `η_l^(−t)` of every
/// level is one of level 0's `η_0^(−s)`, since `η_l = η_0^(2^l)`), plus the
/// half.
///
/// ⛔ A program's fold pool is a UNION over its distinct domains, never a
/// maximum: the same exponents under a different generator are different field
/// elements. The builder interns on the canonical word, so two folds sharing a
/// value pay for it once.
pub fn fold_coset_constants_lean(
    domain: &Domain<GoldilocksField>,
    levels: usize,
    index_bits: usize,
) -> Vec<LfmWord> {
    if levels == 0 {
        return Vec::new();
    }
    let n = domain.size() as u128;
    let block = 1u128 << levels;
    let generator_inv = domain
        .generator()
        .inv()
        .expect("a domain generator is nonzero");
    let mut exponents = vec![0u128];
    for i in 0..index_bits {
        exponents.push((1u128 << i) % n);
    }
    for s in 1..block / 2 {
        exponents.push((s * (n / block)) % n);
    }
    let mut words: Vec<LfmWord> = Vec::new();
    for e in exponents {
        let word = base_word(generator_inv.pow(e as u64));
        if !words.contains(&word) {
            words.push(word);
        }
    }
    let half = base_word(two_inv());
    if !words.contains(&half) {
        words.push(half);
    }
    words
}

/// A round's fold, prepared once per round and applied once per query: the
/// round's `½·α` ([`prepare_lean_alphas`]), then [`emit_fold_coset_lean`] a
/// query.
pub struct RoundFold(LeanAlphas);

impl RoundFold {
    /// Prepare the round's challenges: one `MulBase` a level.
    pub fn prepare(b: &mut LfmBuilder, alphas: &[Ext]) -> Self {
        RoundFold(prepare_lean_alphas(b, alphas))
    }

    /// Fold one opened block at one query.
    pub fn fold(
        &self,
        b: &mut LfmBuilder,
        values: &[Ext],
        domain: &Domain<GoldilocksField>,
        index_bits: &[Bit],
    ) -> Ext {
        emit_fold_coset_lean(b, values, domain, index_bits, &self.0)
    }
}
