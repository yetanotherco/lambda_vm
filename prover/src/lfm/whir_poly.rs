//! Polynomial primitives the WHIR verifier's legs share.
//!
//! `eq` is the one everything reaches for: the GKR layer relation multiplies by
//! it once per layer, the stacked-evaluation weight is a product of two of
//! them per column, and every sumcheck closes against one. So it is worth
//! emitting once, in the cheapest shape, with its cost written down.

use crate::tables::types::{FE, FEE};

use super::builder::{Ext, LfmBuilder};

/// INSTRUCTIONS [`emit_eq_eval`] emits over `n` variables, as the FIRST leg in
/// its program.
///
/// Four rows per variable and one product to fold each into the running
/// accumulator, except the first, which *is* the accumulator. Plus the interned
/// `1`, counted the way [`super::preprocessed::bitwise_preprocessed_rows`]
/// counts it — an `LFM_CONST` row, paid once per program rather than once per
/// leg (`LfmBuilder::ext_const` interns by canonical word), so a second leg in
/// the same program costs [`eq_eval_rows_again`] and this number is an upper
/// bound there.
///
/// ⚠ The three terms sum to exactly `5n`, so a measurement of ONE leg cannot
/// tell them apart: it forces `per_variable + fold_per_variable = 5` and
/// `interned_one = fold_per_variable`, and leaves the split free. What pins the
/// split is a second leg in the same program, which pays `5n − interned_one`.
/// That is why the cost has two pins rather than one.
///
/// Pinned against the emitter by
/// `whir_poly_tests::the_eq_leg_emits_its_closed_form` (this form) and
/// `whir_poly_tests::the_interned_one_is_paid_once_per_program` (the split).
pub const fn eq_eval_rows(n: usize) -> usize {
    if n == 0 {
        // `eq` over no variables is the empty product. Nothing is emitted but
        // the constant it returns.
        return 1;
    }
    INTERNED_ONE + eq_eval_rows_again(n)
}

/// INSTRUCTIONS a FURTHER [`emit_eq_eval`] emits in a program that already has
/// one: the same rows less the constant, which is already interned.
///
/// This is the form the assembled verifier's census adds up — it emits `eq`
/// once per GKR layer, twice per stacked column and once per sumcheck close,
/// and only the first of those pays for the `1`.
pub const fn eq_eval_rows_again(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let per_variable = 4 * n; // r·x, 1−r, 1−x, and the MulAdd that joins them
    let fold = n - 1; // one product per variable past the first
    per_variable + fold
}

/// The `LFM_CONST` row holding `1`, paid once per program.
const INTERNED_ONE: usize = 1;

/// ★ `eq(r, x) = Π_i [ r_i·x_i + (1−r_i)(1−x_i) ]`, emitted.
///
/// The factor is written as `(1−r)(1−x) + r·x` rather than as the sum of two
/// products, so the join is one `MulAdd` and a variable costs four rows instead
/// of five. `MulAdd` is the same one row as `Mul` on `LFM_XALU`, which is what
/// makes the shape free to choose.
///
/// Panics if the two points differ in length — the host returns an error there,
/// but a straight-line program's lengths are emit-time constants, so a mismatch
/// is a bug in the emitter rather than a condition to carry at runtime.
pub fn emit_eq_eval(b: &mut LfmBuilder, r: &[Ext], x: &[Ext]) -> Ext {
    assert_eq!(
        r.len(),
        x.len(),
        "eq's two points must have the same number of variables"
    );
    let one = b.ext_const(&FEE::one());
    let mut acc = one;
    for (i, (&ri, &xi)) in r.iter().zip(x).enumerate() {
        let rx = b.emul(ri, xi);
        let not_r = b.esub(one, ri);
        let not_x = b.esub(one, xi);
        let term = b.emul_add(not_r, not_x, rx);
        acc = if i == 0 { term } else { b.emul(acc, term) };
    }
    acc
}

/// INSTRUCTIONS one [`emit_sumcheck_round`] emits at degree `d`, once the
/// program has interned its constants.
///
/// One subtract to recover `g(0)`; `d(d+1)/2` subtracts for the forward
/// difference triangle (`d` at the first level down to one at the last, and
/// every entry of it is used); `d − 1` `MulAdd`s for the Newton steps `u_1 ..
/// u_{d−1}` (`u_0` is the challenge itself and costs nothing); and `d` more for
/// the nest that folds them. So **7 rows at degree 2, 12 at degree 3, and 42 at
/// degree 7** — the degrees the verifier reaches are 3 in the GKR ladder
/// (`gkr.rs:522`), 2 in `claim_reduce` (`claim_reduce.rs:291`) and in the WHIR
/// chain (`whir_chain.rs:991`), and the worst rule's degree in the main batched
/// sumcheck (`batch.rs:431`, `degree_of`), which is table-dependent.
///
/// Pinned as a SLOPE — one more round in the same program — by
/// `whir_poly_tests::a_sumcheck_round_costs_its_closed_form`, so the constants
/// below cancel out of it and are pinned separately.
pub const fn sumcheck_round_rows(degree: usize) -> usize {
    let d = clamp_degree(degree);
    let recover_g0 = 1;
    let triangle = d * (d + 1) / 2;
    let newton_steps = d - 1; // u_0 is the challenge; only u_1 .. u_{d−1} cost
    let nest = d; // one MulAdd per level of the nest
    recover_g0 + triangle + newton_steps + nest
}

/// `LFM_CONST` rows a degree-`d` round interns, paid once per program however
/// many rounds share the degree: the pair `(1/(j+1), −j/(j+1))` for each Newton
/// step `u_1 .. u_{d−1}`.
pub const fn sumcheck_round_consts(degree: usize) -> usize {
    2 * (clamp_degree(degree) - 1)
}

/// `verify_rounds` clamps the degree to at least one (`sumcheck.rs:366`), so a
/// leg emitted for degree 0 must cost what degree 1 costs rather than underflow
/// the forms above.
const fn clamp_degree(degree: usize) -> usize {
    if degree == 0 { 1 } else { degree }
}

/// ★ One sumcheck round's claim recursion, as `sumcheck::verify_rounds`
/// computes it (`crypto/multilinear/src/sumcheck.rs:357-397`).
///
/// The round polynomial travels as its evaluations at `1 .. d`; `g(0)` is not
/// sent, because `g(0) + g(1)` is the claim carried in. So the nodes are
/// `0 .. d` with values `[claim − e_0, e_0, …, e_{d−1}]`, and the new claim is
/// that polynomial at the round's challenge.
///
/// ★ **Why Newton rather than Lagrange.** The nodes are `0 .. d`, equally
/// spaced and known at emit time, so
///
/// ```text
///     g(r) = y_0 + u_0·(Δ¹ + u_1·(Δ² + u_2·(Δ³ + …))),   u_j = (r − j)/(j + 1)
/// ```
///
/// with `Δ^k` the forward differences of the values. `u_0 = r`, and every other
/// `u_j` is `r·(1/(j+1)) + (−j/(j+1))` — one `MulAdd` against two interned
/// constants, not a subtract and a scale. The host's `interpolate`
/// (`sumcheck.rs:71-92`) rebuilds Lagrange denominators and batch-inverts them
/// every round; that is a host inefficiency this does not copy. Newton and
/// Lagrange are the same polynomial through the same `d + 1` points, so the two
/// agree identically — and unlike a barycentric form this never divides by
/// `r − node`, so there is no special case when a challenge lands on one.
///
/// ⚠ **The challenge is an INPUT here: this leg neither absorbs nor draws.**
/// What the transcript replay owes it, stated so the two cannot drift: absorb
/// `e_0 … e_{d−1}` in that order, as field elements, then draw exactly one
/// challenge. Until the replay supplies it, a caller's challenge is hinted, and
/// a hinted challenge is a forgery by the arena rule (`builder.rs:617-620`:
/// never derive challenges from arenas) — test-only, never a verifier.
///
/// The host's length check (`evaluations.len() == degree`) has no counterpart
/// here and needs none: the emitter reads exactly `d` words at fixed offsets,
/// so a proof carrying a different count has no way to be supplied. That is the
/// `epoch_verify.rs:171-179` idiom — the absence of a second value, not an
/// assert somebody could forget.
pub fn emit_sumcheck_round(
    b: &mut LfmBuilder,
    claim: Ext,
    evaluations: &[Ext],
    challenge: Ext,
) -> Ext {
    let d = evaluations.len();
    assert!(
        d >= 1,
        "a sumcheck round sends at least one evaluation; the host clamps the degree to 1"
    );

    // `g(0)` is recovered, not sent.
    let mut level = Vec::with_capacity(d + 1);
    level.push(b.esub(claim, evaluations[0]));
    level.extend_from_slice(evaluations);

    // The difference triangle, keeping `Δ^k y_0` — the head of every level.
    let mut deltas = Vec::with_capacity(d + 1);
    deltas.push(level[0]);
    while level.len() > 1 {
        let next: Vec<Ext> = level
            .windows(2)
            .map(|pair| b.esub(pair[1], pair[0]))
            .collect();
        deltas.push(next[0]);
        level = next;
    }

    // The nest, from the innermost level out.
    let mut acc = deltas[d];
    for k in (0..d).rev() {
        let u = if k == 0 {
            challenge
        } else {
            emit_newton_step(b, challenge, k)
        };
        acc = b.emul_add(u, acc, deltas[k]);
    }
    acc
}

/// A group of rounds against a running claim, which is what every caller of
/// `sumcheck::verify_rounds` actually asks for. Returns the claim the group
/// leaves; the point is the challenges it was handed, so there is nothing to
/// return for it.
pub fn emit_sumcheck_rounds(
    b: &mut LfmBuilder,
    claim: Ext,
    rounds: &[Vec<Ext>],
    challenges: &[Ext],
) -> Ext {
    assert_eq!(
        rounds.len(),
        challenges.len(),
        "a sumcheck group draws exactly one challenge per round"
    );
    let mut current = claim;
    for (evaluations, &r) in rounds.iter().zip(challenges) {
        current = emit_sumcheck_round(b, current, evaluations, r);
    }
    current
}

/// `u_j = (r − j)/(j + 1)`, one `MulAdd` against two interned constants.
fn emit_newton_step(b: &mut LfmBuilder, r: Ext, j: usize) -> Ext {
    let inv = FE::from((j + 1) as u64)
        .inv()
        .expect("j + 1 is a small nonzero Goldilocks element");
    let scale = b.ext_const(&FEE::new([inv, FE::zero(), FE::zero()]));
    let shift = b.ext_const(&FEE::new([
        FE::zero() - FE::from(j as u64) * inv,
        FE::zero(),
        FE::zero(),
    ]));
    b.emul_add(r, scale, shift)
}
