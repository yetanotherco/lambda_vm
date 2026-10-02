//! The LogUp-GKR closure as a machine leg: `gkr::verify`, layer by layer.
//!
//! This is the WHIR verifier's largest fixed term per epoch — one tree per
//! table, `m = n + ceil_log2(I)` layers each — and it is where the degree-3
//! sumcheck round and `eq` earn their keep, so it is built on top of them
//! rather than beside them.

use super::builder::{Ext, LfmBuilder};
use super::whir_poly::{
    emit_eq_eval, emit_sumcheck_rounds, eq_eval_rows_again, sumcheck_round_consts,
    sumcheck_round_rows,
};

/// The degree the GKR ladder's sumchecks run at (`gkr.rs:522`).
pub const GKR_SUMCHECK_DEGREE: usize = 3;

/// One layer's proof values as wires: the sumcheck's round evaluations and the
/// four bound values of the layer relation.
pub struct GkrLayerWires {
    /// `i` rounds of `GKR_SUMCHECK_DEGREE` evaluations at layer `i`.
    pub sumcheck: Vec<Vec<Ext>>,
    pub p_lo: Ext,
    pub p_hi: Ext,
    pub q_lo: Ext,
    pub q_hi: Ext,
}

/// The challenges one layer consumes, in the order `gkr::verify` draws them:
/// the batching λ, one per sumcheck round, then the halves' folding point.
pub struct GkrLayerChallenges {
    pub lambda: Ext,
    pub rounds: Vec<Ext>,
    pub c: Ext,
}

/// What the ladder leaves: the input layer's claim.
pub struct GkrClaimWires {
    pub point: Vec<Ext>,
    pub p: Ext,
    pub q: Ext,
}

/// INSTRUCTIONS layer `i` emits — the layer whose point carries `i` variables,
/// which is the `i`th from the output.
///
/// The `12` that does not depend on `i`, each term by its shape: one `MulAdd`
/// for `claimed_sum = p + λ·q`; two for the numerator `p_lo·q_hi + p_hi·q_lo`
/// (a `Mul` then a `MulAdd`); one for the denominator `q_lo·q_hi`; one `MulAdd`
/// for `numerator + λ·denominator`; one `Mul` by `eq_at`; **two for the
/// layer-relation assert**, which lowers to `diff = a − b; _ = diff / 0` and is
/// the machine's `LayerRelationMismatch`; and four for the two
/// `combine_halves`, a `Sub` and a `MulAdd` each.
///
/// The rest is `i` degree-3 sumcheck rounds and one `eq` over `i` variables.
///
/// ⚠ At layer 0 the `eq` is the empty product and the `Mul` by it is a wasted
/// row — about 34 rows an epoch across the tables. It is emitted anyway, so the
/// shape stays uniform with the host's and the form has no special case.
pub const fn gkr_layer_rows(i: usize) -> usize {
    let fixed = 12;
    fixed + i * sumcheck_round_rows(GKR_SUMCHECK_DEGREE) + eq_eval_rows_again(i)
}

/// INSTRUCTIONS a whole ladder of `layers` layers emits, constants excluded.
///
/// Layer `i` carries a point of `i` variables and a sumcheck of `i` rounds, so
/// the ladder's shape follows from its length alone.
pub fn gkr_verify_rows(layers: usize) -> usize {
    (0..layers).map(gkr_layer_rows).sum()
}

/// `LFM_CONST` rows the ladder interns, once per program: `eq`'s `1`, the
/// assert's `0`, and — only once a layer actually runs a sumcheck round, which
/// takes two layers — the degree-3 round's Newton constants.
pub const fn gkr_verify_consts(layers: usize) -> usize {
    let eq_one_and_assert_zero = 2;
    let newton = if layers >= 2 {
        sumcheck_round_consts(GKR_SUMCHECK_DEGREE)
    } else {
        0
    };
    eq_one_and_assert_zero + newton
}

/// ★ `gkr::verify` (`crypto/multilinear/src/gkr.rs:505-546`), emitted.
///
/// Per layer: draw λ and form the batched claim `p + λ·q`; run the layer's
/// sumcheck; check the residual against the layer relation
///
/// ```text
///     eq(point, r) · (p_lo·q_hi + p_hi·q_lo + λ·q_lo·q_hi)
/// ```
///
/// absorb the four values and draw `c`; and fold each half at `c`. The point
/// the next layer carries is `c` followed by this layer's sumcheck challenges,
/// which is why layer `i`'s sumcheck has exactly `i` rounds.
///
/// ⚠ The challenges are INPUTS, as in [`super::whir_poly::emit_sumcheck_round`]
/// — this leg neither absorbs nor draws. What the replay owes it, in order:
/// draw λ; per round absorb the three evaluations then draw; absorb `p_lo`,
/// `p_hi`, `q_lo`, `q_hi` in that order; draw `c`.
///
/// ★ The layer-relation check is a REAL REJECT, and the only one in the legs so
/// far: `assert_eq_ext` has no satisfying assignment when the two differ
/// (`builder.rs:289-292`), so a proof whose relation fails cannot be executed,
/// let alone proven.
pub fn emit_gkr_verify(
    b: &mut LfmBuilder,
    output: (Ext, Ext),
    layers: &[GkrLayerWires],
    challenges: &[GkrLayerChallenges],
) -> GkrClaimWires {
    assert_eq!(
        layers.len(),
        challenges.len(),
        "every GKR layer draws its own challenges"
    );
    let (mut p_claim, mut q_claim) = output;
    let mut point: Vec<Ext> = Vec::new();

    for (layer, drawn) in layers.iter().zip(challenges) {
        assert_eq!(
            layer.sumcheck.len(),
            point.len(),
            "layer {}'s sumcheck runs over the point it carries",
            point.len()
        );
        assert_eq!(
            drawn.rounds.len(),
            point.len(),
            "one challenge per sumcheck round"
        );

        let claimed_sum = b.emul_add(drawn.lambda, q_claim, p_claim);
        let residual = emit_sumcheck_rounds(b, claimed_sum, &layer.sumcheck, &drawn.rounds);

        // The sumcheck's point IS the challenges it drew.
        let eq_at = emit_eq_eval(b, &point, &drawn.rounds);
        let cross = b.emul(layer.p_lo, layer.q_hi);
        let numerator = b.emul_add(layer.p_hi, layer.q_lo, cross);
        let denominator = b.emul(layer.q_lo, layer.q_hi);
        let inner = b.emul_add(drawn.lambda, denominator, numerator);
        let expected = b.emul(eq_at, inner);
        b.assert_eq_ext(expected, residual);

        p_claim = emit_combine_halves(b, layer.p_lo, layer.p_hi, drawn.c);
        q_claim = emit_combine_halves(b, layer.q_lo, layer.q_hi, drawn.c);
        point = std::iter::once(drawn.c)
            .chain(drawn.rounds.iter().copied())
            .collect();
    }

    GkrClaimWires {
        point,
        p: p_claim,
        q: q_claim,
    }
}

/// `combine_halves(lo, hi, c) = lo + c·(hi − lo)` (`gkr.rs:384-390`).
fn emit_combine_halves(b: &mut LfmBuilder, lo: Ext, hi: Ext, c: Ext) -> Ext {
    let spread = b.esub(hi, lo);
    b.emul_add(c, spread, lo)
}
