//! The bus statements as a machine leg: `logup::claim_statements`, emitted.
//!
//! The mirrors are `multilinear::logup::claim_statements`
//! (`logup.rs:220-275`) and, behind it, `stark::multilinear_logup::interactions`
//! (`multilinear_logup.rs:84-127`). Together they turn a table's GKR
//! input-layer claim into the two rules the main sumcheck batches beside the
//! zerocheck:
//!
//! ```text
//!     numerator   = weight_row · Σ_i w_i · numerator_i(f)
//!     denominator = weight_row · ( Σ_i w_i · denominator_i(f) + padding )
//!     w           = eq_evals(interaction_point)          — 2^ceil_log2(I) of them
//!     padding     = Σ of the weights past the live interactions
//! ```
//!
//! # ⛔ Only ONE of the two affine forms has runtime coefficients
//!
//! `interactions` builds each interaction as a signed multiplicity over
//! `z − fingerprint`. The NUMERATOR is `multiplicity.evaluate_with`, negated
//! for a receiver (`multilinear_logup.rs:117-120`) — it reads neither `z` nor
//! `alpha`, so its coefficients and its constant are emit-time values and this
//! leg interns them. Only the DENOMINATOR's coefficients are wires, and they
//! are wires because the fingerprint rides the powers of `alpha`, a challenge
//! drawn after the commitment.
//!
//! # ★ The regrouping, with its cost, so it is not mistaken for a difference
//!
//! The host FUSES the fingerprint into one affine per interaction whose
//! coefficient for column `c` is `−Σ_p alpha^p·c_{p,c}` — `probe` recovers that
//! sum by evaluating at basis rows. Emitting THAT form means building every
//! column's alpha-polynomial as a wire first: `Σ_c (one MulAdd per power that
//! touches c) + |terms|`, which is `|columns|·(W+1)` when the interaction's
//! columns are spread over its bus elements.
//!
//! This emits the UNFUSED form, straight off `fingerprint_at`'s own loop
//! (`multilinear_logup.rs:24-38`):
//!
//! ```text
//!     denominator_i = z − bus_id − Σ_p alpha^{p+1} · elements[p](f)
//! ```
//!
//! which costs `Σ_p |terms_p| + W`. Identical in value; strictly fewer rows
//! whenever a bus element reads fewer columns than its interaction does, which
//! is every packed value. The structure comes from
//! `multilinear_logup::interaction_shapes`, whose own gate is that recombining
//! it at any `(z, alpha)` lands on what `interactions` fuses — so both
//! groupings are pinned and neither is re-derived here.
//!
//! # ⚠ What `interaction_shapes` already did, and this leg must not undo
//!
//! A column past the main width is PINNED TO ZERO in the recovered affine
//! rather than carried, because `interactions` filters its probe candidates to
//! `column < num_main_columns`. That happens in the accessor; this leg only
//! reads the affines it hands back.
//!
//! # The weights, and the one term that is exponential in anything
//!
//! `eq_evals(interaction_point)` is `2^ceil_log2(I)` values for a table with
//! `I` interactions (`logup.rs:235`). It is the only place a table's verify
//! cost grows as a power of anything, and it is small: over epoch 0's 34 tables
//! (ts1, 2026-09-18) `Σ 2^ceil_log2(I)` is 4,444 against `Σ I` of 2,531, so the
//! whole epoch's weights are ≈9,006 rows.

use multilinear::logup::Affine;
use stark::multilinear_logup::InteractionShape;

use crate::tables::types::{FEE, GoldilocksExtension};

use super::builder::{Ext, LfmBuilder};

type Aff = Affine<GoldilocksExtension>;
type Shape = InteractionShape<GoldilocksExtension>;

/// What a leg costs: its operation rows, and the DISTINCT constants it interns.
///
/// Constants are pooled per PROGRAM (`builder.rs:180-190`), so counting them
/// per sub-form and adding would charge a shared value twice. This carries the
/// values and deduplicates once, the way `whir_program::steps_rows` counts a
/// DAG's `Fixed` steps.
#[derive(Default, Debug, Clone)]
pub struct Cost {
    operations: usize,
    constants: Vec<FEE>,
}

impl Cost {
    /// One `LFM_XALU` row.
    fn op(&mut self) {
        self.operations += 1;
    }

    /// `count` of them.
    fn ops(&mut self, count: usize) {
        self.operations += count;
    }

    /// One `LFM_CONST` row, unless this value is already interned.
    fn constant(&mut self, value: FEE) {
        if !self.constants.contains(&value) {
            self.constants.push(value);
        }
    }

    /// The instructions the leg emits.
    pub fn rows(&self) -> usize {
        self.operations + self.constants.len()
    }

    pub fn operations(&self) -> usize {
        self.operations
    }

    pub fn constants(&self) -> usize {
        self.constants.len()
    }
}

/// The wire a factor slot reads.
fn read(values: &[Ext], slot: usize) -> Ext {
    *values.get(slot).unwrap_or_else(|| {
        panic!(
            "the bus reads factor {slot}, but only {} were supplied",
            values.len()
        )
    })
}

/// `Σ c_j·v_{s_j} + k` over the factor wires, with the coefficients interned.
///
/// A coefficient of ONE is folded away exactly as the host's
/// `Builder::weighted_sum` folds it (`program.rs:304-316`): it needs no
/// constant, and when it is the term that opens the fold it needs no row
/// either — the wire is already the running value.
pub fn emit_affine(b: &mut LfmBuilder, affine: &Aff, values: &[Ext]) -> Ext {
    let mut accumulated: Option<Ext> = None;
    for (slot, coefficient) in affine.terms() {
        let value = read(values, *slot);
        let unit = *coefficient == FEE::one();
        accumulated = Some(match (accumulated, unit) {
            (None, true) => value,
            (None, false) => {
                let c = b.ext_const(coefficient);
                b.emul(c, value)
            }
            (Some(acc), true) => b.eadd(value, acc),
            (Some(acc), false) => {
                let c = b.ext_const(coefficient);
                b.emul_add(c, value, acc)
            }
        });
    }

    let constant = affine.constant_term();
    match accumulated {
        None => b.ext_const(constant),
        Some(acc) if *constant == FEE::zero() => acc,
        Some(acc) => {
            let k = b.ext_const(constant);
            b.eadd(acc, k)
        }
    }
}

/// INSTRUCTIONS [`emit_affine`] emits, added to `cost`.
///
/// One row per term, less the one a leading coefficient of one saves; one more
/// for a nonzero constant. The interned values are every coefficient that is
/// not one, and the constant when it is carried at all.
pub fn affine_cost(affine: &Aff, cost: &mut Cost) {
    for (index, (_, coefficient)) in affine.terms().iter().enumerate() {
        if *coefficient == FEE::one() {
            // The leading term needs no row: the factor's wire IS the running
            // value. Every later one is an add.
            if index > 0 {
                cost.op();
            }
        } else {
            cost.constant(*coefficient);
            cost.op();
        }
    }

    let constant = affine.constant_term();
    if affine.terms().is_empty() {
        // A constant expression is one `LFM_CONST` and no arithmetic — and a
        // zero constant still costs that row, because the wire has to exist.
        cost.constant(*constant);
    } else if *constant != FEE::zero() {
        cost.constant(*constant);
        cost.op();
    }
}

/// ★ One interaction's two values: the signed multiplicity, and
/// `z − fingerprint`.
///
/// `alpha_powers[p]` must be `alpha^p`; the element at index `p` rides
/// `alpha^{p+1}`, which is `fingerprint_at`'s `power` counter starting at one
/// with the bus id at `alpha^0`.
///
/// Panics if an element has no power to ride — an emit-time length, so a
/// mismatch is a bug in the caller rather than a condition to carry at
/// runtime.
pub fn emit_interaction(
    b: &mut LfmBuilder,
    shape: &Shape,
    z: Ext,
    alpha_powers: &[Ext],
    values: &[Ext],
) -> (Ext, Ext) {
    let numerator = emit_affine(b, &shape.numerator, values);

    let mut fingerprint = b.ext_const(&shape.bus_id);
    for (index, element) in shape.elements.iter().enumerate() {
        let value = emit_affine(b, element, values);
        let power = *alpha_powers.get(index + 1).unwrap_or_else(|| {
            panic!(
                "bus element {index} rides alpha^{}, but only {} powers were supplied",
                index + 1,
                alpha_powers.len()
            )
        });
        fingerprint = b.emul_add(power, value, fingerprint);
    }
    let denominator = b.esub(z, fingerprint);

    (numerator, denominator)
}

/// INSTRUCTIONS [`emit_interaction`] emits, added to `cost`.
pub fn interaction_cost(shape: &Shape, cost: &mut Cost) {
    affine_cost(&shape.numerator, cost);
    // The fingerprint opens at the bus id, one `LFM_CONST`.
    cost.constant(shape.bus_id);
    for element in &shape.elements {
        affine_cost(element, cost);
        // One `MulAdd` folding this element's alpha power in.
        cost.op();
    }
    // `z − fingerprint`.
    cost.op();
}

/// How many powers of alpha [`emit_interaction`] READS over a whole bus:
/// `1 + max |elements|`.
///
/// ⚠ ONE FEWER than the host builds. `interactions` sizes its ladder at
/// `width + 1` where `width` is the largest `num_bus_elements`, and
/// `num_bus_elements` COUNTS THE BUS ID (`lookup.rs:1779-1787`) — so the host
/// holds `max |elements| + 2` powers and its last is never read. The values
/// are identical; only the dead tail is missing, which is the same
/// relationship `challenge_powers_rows` already records.
///
/// ⚠ `alpha^0` is in both ladders and read by neither: `fingerprint_at` seeds
/// its sum with the bus id directly rather than with `alpha_powers[0]·bus_id`.
/// It is kept so the index arithmetic stays the host's.
pub fn alpha_powers_read(shapes: &[Shape]) -> usize {
    1 + shapes
        .iter()
        .map(|shape| shape.elements.len())
        .max()
        .unwrap_or(0)
}

/// ★ `multilinear::eq::eq_evals`, emitted: the weight of every interaction
/// slot, live and padding alike.
///
/// The host's own doubling (`eq.rs:49-61`): seed the table with one, then per
/// variable taken in REVERSE scale the live half by `r_i` into the upper half
/// and by `1 − r_i` in place.
pub fn emit_eq_evals(b: &mut LfmBuilder, point: &[Ext]) -> Vec<Ext> {
    let one = b.ext_const(&FEE::one());
    let mut table = vec![one; 1usize << point.len()];
    for (level, r) in point.iter().rev().enumerate() {
        let one_minus = b.esub(one, *r);
        let half = 1usize << level;
        for index in 0..half {
            let live = table[index];
            table[half + index] = b.emul(live, *r);
            table[index] = b.emul(live, one_minus);
        }
    }
    table
}

/// INSTRUCTIONS [`emit_eq_evals`] emits over `n` variables, added to `cost`.
///
/// One `LFM_CONST` for the seed `1`; one `esub` per variable for `1 − r_i`;
/// two multiplies per live entry per level, `Σ_{level<n} 2·2^level`. Standing
/// alone that is
///
/// ```text
///     eq_evals rows = 1 + n + 2·(2^n − 1)
/// ```
///
/// and it is stated as an identity the gate asserts rather than as a second
/// implementation — the seed's row is shared with any other user of `1` in the
/// same program, which is why this adds to a [`Cost`] instead of returning a
/// number.
pub fn eq_evals_cost(n: usize, cost: &mut Cost) {
    cost.constant(FEE::one());
    cost.ops(n);
    cost.ops(2 * ((1usize << n) - 1));
}

/// The two rule values a table's bus leaves for the main sumcheck.
#[derive(Debug, Clone, Copy)]
pub struct BusValues {
    pub numerator: Ext,
    pub denominator: Ext,
}

/// The variables the claim point spans — the host's own
/// `multilinear::logup::input_layer_vars`, called rather than restated.
pub fn claim_point_vars(interactions: usize, num_row_vars: usize) -> usize {
    multilinear::logup::input_layer_vars(interactions, num_row_vars)
}

/// ★ `logup::claim_statements`' two rules, emitted as their VALUES at the
/// factor wires.
///
/// The host returns two `Program`s and `constraint_argument::verify_core` runs
/// them through `Rule::apply` (`batch.rs:79-96`); a straight-line emitter has
/// no reason to build the program and then walk it, so this is the walk.
///
/// `claim_point` is the GKR claim's point as wires, interaction half first;
/// `weight` is the factor slot holding `eq(row_point, ·)`, which the caller
/// registers as a public factor. The row half of the claim point is the
/// caller's to turn into that factor, exactly as the host hands `row_point`
/// back.
///
/// Panics on an empty bus, and on a claim point of the wrong arity. Both are
/// emit-time lengths. The empty case is where this REFUSES what the host
/// merely computes: `claim_statements` would return `row · 0` and `row · 1`,
/// but `TableLayout::new` rejects a table with no interactions (`EmptyPolynomial`,
/// measured by V1d), so there is no such table to emit and a zero-length bus
/// here is a caller's bug.
pub fn emit_claim_statements(
    b: &mut LfmBuilder,
    shapes: &[Shape],
    claim_point: &[Ext],
    num_row_vars: usize,
    z: Ext,
    alpha_powers: &[Ext],
    values: &[Ext],
    weight: usize,
) -> BusValues {
    assert!(
        !shapes.is_empty(),
        "a table with no bus interactions does not lay out"
    );
    let expected = claim_point_vars(shapes.len(), num_row_vars);
    assert_eq!(
        claim_point.len(),
        expected,
        "the claim point spans the interaction bits and the rows"
    );

    let interaction_point = &claim_point[..claim_point.len() - num_row_vars];
    let weights = emit_eq_evals(b, interaction_point);

    // The tail belongs to the 0/1 padding slots, whose denominators are one and
    // whose numerators vanish — so it reaches the denominator as a sum and the
    // numerator not at all.
    let padding = weights[shapes.len()..]
        .iter()
        .copied()
        .reduce(|acc, w| b.eadd(acc, w));

    let mut numerator: Option<Ext> = None;
    let mut denominator: Option<Ext> = None;
    for (shape, w) in shapes.iter().zip(&weights) {
        let (num, den) = emit_interaction(b, shape, z, alpha_powers, values);
        numerator = Some(match numerator {
            None => b.emul(*w, num),
            Some(acc) => b.emul_add(*w, num, acc),
        });
        denominator = Some(match denominator {
            None => b.emul(*w, den),
            Some(acc) => b.emul_add(*w, den, acc),
        });
    }

    let mut numerator = numerator.expect("a non-empty bus batches at least one interaction");
    let mut denominator = denominator.expect("a non-empty bus batches at least one interaction");
    if let Some(padding) = padding {
        denominator = b.eadd(denominator, padding);
    }

    let row = read(values, weight);
    numerator = b.emul(row, numerator);
    denominator = b.emul(row, denominator);

    BusValues {
        numerator,
        denominator,
    }
}

/// INSTRUCTIONS [`emit_claim_statements`] emits.
///
/// Every term by the shape it comes from, with `I = shapes.len()` and
/// `n = ceil_log2(I)`:
///
/// - the weights, `eq_evals_cost(n)`;
/// - the padding fold, `2^n − I − 1` adds when there are padding slots at all;
/// - each interaction's two values, `interaction_cost`;
/// - `2I` rows batching them, one `Mul` opening each of the two folds and a
///   `MulAdd` for every later interaction;
/// - one add folding the padding into the denominator, when there is one;
/// - two multiplies by the row weight.
pub fn claim_statements_cost(shapes: &[Shape], num_row_vars: usize) -> Cost {
    let mut cost = Cost::default();
    let interactions = shapes.len();
    let n = claim_point_vars(interactions, num_row_vars) - num_row_vars;
    eq_evals_cost(n, &mut cost);

    let padding = (1usize << n) - interactions;
    if padding > 0 {
        cost.ops(padding - 1);
    }

    for shape in shapes {
        interaction_cost(shape, &mut cost);
    }
    cost.ops(2 * interactions);
    if padding > 0 {
        cost.op();
    }
    cost.ops(2);

    cost
}
