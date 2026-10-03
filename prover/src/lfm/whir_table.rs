//! The per-table verify as a machine leg.
//!
//! The mirror is `stark::multilinear_table::verify`
//! (`multilinear_table.rs:721-785`): the bus output absorbed, the GKR ladder,
//! the row challenges, the bus statements, the zerocheck rule, one batched
//! sumcheck over all three, and `claim_reduce` to leave every COLUMN claimed at
//! one point.
//!
//! This module starts with the one kernel none of the earlier legs has: the row
//! SELECTOR, which is what a public factor's value is.

use multilinear::claim_reduce::FactorSource;
use multilinear::constraint_argument::FactorKind;
use multilinear::logup::input_layer_vars;
use multilinear::selector::Selector;
use stark::multilinear_air::IrShape;
use stark::multilinear_logup::InteractionShape;
use stark::multilinear_table::weight_slots;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::algebraic_commit::leaf_capacity;
use super::builder::{Ext, LfmBuilder};
use super::whir_air::{combine_rows, emit_combine};
use super::whir_bus::{BusInputs, Cost, claim_statements_cost, emit_claim_statements};
use super::whir_gkr::{
    GKR_SUMCHECK_DEGREE, GkrLayerChallenges, GkrLayerWires, emit_gkr_verify, gkr_verify_rows,
};
use super::whir_poly::{
    challenge_powers_rows, emit_challenge_powers, emit_eq_eval, emit_sumcheck_rounds,
    eq_eval_rows_again, sumcheck_round_rows,
};
use super::whir_reduce::{REDUCE_DEGREE, ReduceWires, claim_reduce_rows, emit_claim_reduce_verify};
use super::whir_transcript::{
    COORDINATES_PER_EXT, SpongeEntry, SpongeSchedule, WhirTranscript, absorb_unpack_rows,
    sample_ext_rows,
};

/// ★ `Selector::evaluate` (`crypto/multilinear/src/selector.rs:61-100`),
/// emitted.
///
/// A constraint that reads the next row cannot apply on the last one, so it
/// carries `Selector { end_exemptions }` and the verifier needs that
/// indicator's value at the point its sumcheck settled on. The host computes it
/// in `O(num_vars)` rather than over the cube:
///
/// ```text
///     s(x) = 1 − geq(x, cutoff),      cutoff = 2^n − end_exemptions
/// ```
///
/// where `geq` is the multilinear indicator of `index(x) >= cutoff`: either `x`
/// matches `cutoff` bit for bit, or the two first differ at a position where
/// `cutoff` has a zero and `x` a one. `prefix` is the running "everything above
/// this bit matched"; VARIABLE 0 IS THE MOST SIGNIFICANT BIT.
///
/// # ⚠ Why the literal values are tracked rather than emitted
///
/// `prefix` starts at ONE and `geq` at ZERO. A row that multiplies a wire by a
/// literal one, or adds a literal zero to it, computes nothing — so this
/// carries each as `Option<Ext>`, where `None` IS the literal the host starts
/// from, and spends a row only once the value is a wire. It is not an
/// optimisation bolted onto the host's expression; it is the host's expression
/// with the identity steps left out, and the cost form below counts the same
/// cases.
///
/// Panics when there are more exemptions than rows — the host returns
/// `TooManyExemptions`, and here the two are emit-time numbers, so a mismatch
/// is a bug in the caller.
pub fn emit_selector(b: &mut LfmBuilder, selector: Selector, point: &[Ext]) -> Ext {
    let num_vars = point.len();
    let size = 1usize << num_vars;
    assert!(
        selector.end_exemptions <= size,
        "a selector cannot exempt more rows than the table has: {} of {size}",
        selector.end_exemptions
    );

    let one = b.ext_const(&FEE::one());
    if selector.is_trivial() {
        return one;
    }
    let cutoff = size - selector.end_exemptions;
    if cutoff == 0 {
        // Every row is exempt: the constraint applies nowhere.
        return b.ext_const(&FEE::zero());
    }

    // `None` is the literal the host starts from: `prefix = 1`, `geq = 0`.
    let mut prefix: Option<Ext> = None;
    let mut geq: Option<Ext> = None;
    for (i, x) in point.iter().enumerate() {
        let bit = (cutoff >> (num_vars - 1 - i)) & 1;
        if bit == 0 {
            // `x` exceeds `cutoff` here when everything above matched and
            // `x_i = 1`.
            geq = Some(match (prefix, geq) {
                (None, None) => *x,
                (None, Some(running)) => b.eadd(*x, running),
                (Some(above), None) => b.emul(above, *x),
                (Some(above), Some(running)) => b.emul_add(above, *x, running),
            });
            let one_minus = b.esub(one, *x);
            prefix = Some(match prefix {
                None => one_minus,
                Some(above) => b.emul(above, one_minus),
            });
        } else {
            prefix = Some(match prefix {
                None => *x,
                Some(above) => b.emul(above, *x),
            });
        }
    }

    // Both branches write `prefix`, and a selector that is neither trivial nor
    // all-exempt has at least one variable — `num_vars = 0` leaves `cutoff = 0`
    // for every non-trivial exemption count, which returned above.
    let prefix = prefix.expect("a live selector reads at least one variable");
    // The remaining prefix is the `x == cutoff` case.
    let total = match geq {
        None => prefix,
        Some(running) => b.eadd(running, prefix),
    };
    b.esub(one, total)
}

/// INSTRUCTIONS [`emit_selector`] emits over `num_vars` variables, added to
/// `cost`.
///
/// The same case analysis, counted rather than emitted:
///
/// - a trivial selector is the interned `1` and nothing else;
/// - an all-exempt one is the interned `0` and nothing else;
/// - otherwise, per variable, a `1 − x_i` subtract at every zero bit of the
///   cutoff, plus one row for each of `geq` and `prefix` once that value has
///   stopped being a literal;
/// - then the `geq + prefix` add, which the all-ones cutoff does not need, and
///   the final subtract from one.
pub fn selector_cost(selector: Selector, num_vars: usize, cost: &mut Cost) {
    let size = 1usize << num_vars;
    assert!(selector.end_exemptions <= size, "more exemptions than rows");

    cost.constant(FEE::one());
    if selector.is_trivial() {
        return;
    }
    let cutoff = size - selector.end_exemptions;
    if cutoff == 0 {
        cost.constant(FEE::zero());
        return;
    }

    let mut prefix_live = false;
    let mut geq_live = false;
    for i in 0..num_vars {
        let bit = (cutoff >> (num_vars - 1 - i)) & 1;
        if bit == 0 {
            if prefix_live || geq_live {
                cost.op();
            }
            geq_live = true;
            // `1 − x_i` is a row whatever the running product holds.
            cost.op();
            if prefix_live {
                cost.op();
            }
            prefix_live = true;
        } else {
            if prefix_live {
                cost.op();
            }
            prefix_live = true;
        }
    }

    if geq_live {
        cost.op();
    }
    cost.op();
}

// ---------------------------------------------------------------------------
// The assembly: `multilinear_table::verify`, emitted.
// ---------------------------------------------------------------------------

/// The `TableProof` as wires.
///
/// Every field is proof data — a value the prover chose — which is what makes
/// the challenges below DERIVED rather than supplied: nothing the transcript
/// absorbs is computed by this leg.
pub struct TableProofWires<'a> {
    /// The table's share of the bus, `(p, q)`.
    pub bus_output: (Ext, Ext),
    /// One per GKR ladder layer, output first; layer `i` carries `i` sumcheck
    /// rounds.
    pub gkr: &'a [GkrLayerWires],
    /// The main batched sumcheck's rounds: `num_vars` of them, each carrying
    /// `degree_of(rules)` evaluations.
    pub sumcheck: &'a [Vec<Ext>],
    /// Each COMMITTED factor's value at the sumcheck point. The public factors
    /// are absent — this leg computes those.
    pub factor_values: &'a [Ext],
    /// `claim_reduce`'s own proof.
    pub reduce: ReduceWires<'a>,
}

/// The table's emit-time structure: what `TableStatement` carries that is not
/// proof data.
pub struct TableShape<'a> {
    /// The constraint DAG, its roots and their selectors.
    pub ir: &'a IrShape<GoldilocksField, GoldilocksExtension>,
    /// The bus, with the challenges factored out (`interaction_shapes`).
    pub bus: &'a [InteractionShape<GoldilocksExtension>],
    /// The factor list, committed and public interleaved as the rules index it.
    pub kinds: &'a [FactorKind],
    /// The table's committed columns.
    pub num_columns: usize,
    pub num_vars: usize,
}

impl TableShape<'_> {
    /// The GKR ladder's length — `logup::input_layer_vars`, the host's own.
    pub fn gkr_layers(&self) -> usize {
        input_layer_vars(self.bus.len(), self.num_vars)
    }

    /// The main batched sumcheck's degree: `degree_of([zerocheck, numerator,
    /// denominator])` (`batch.rs:264`), where the zerocheck rule is compiled at
    /// `shape.degree() + 1` and both bus rules at 2.
    pub fn sumcheck_degree(&self) -> usize {
        (self.ir.degree() + 1).max(2)
    }

    /// The committed factors' sources — `constraint_argument::sources_of`.
    pub fn sources(&self) -> Vec<FactorSource> {
        self.kinds.iter().filter_map(FactorKind::source).collect()
    }
}

/// What a table's verify leaves for the caller.
pub struct TableVerdictWires {
    /// The table's share of the bus, for the caller to sum.
    pub bus_output: (Ext, Ext),
    /// The point every COLUMN is now claimed at.
    pub point: Vec<Ext>,
    pub column_values: Vec<Ext>,
}

/// `constraint_argument::weave` (`:103-136`), at emit time.
///
/// The committed factors' values come out of the proof and the public ones are
/// computed; this puts them back in factor order, with any weight tables a
/// statement appended landing at the end. Costs NO rows: it is a reordering of
/// wires that already exist.
fn weave(kinds: &[FactorKind], committed: &[Ext], public: &[Ext]) -> Vec<Ext> {
    let want_committed = kinds.iter().filter(|k| k.source().is_some()).count();
    assert_eq!(
        committed.len(),
        want_committed,
        "one proof value per committed factor"
    );
    let want_public = kinds.len() - want_committed;
    assert!(
        public.len() >= want_public,
        "one computed value per public factor, plus the weight tables"
    );

    let (public, weights) = public.split_at(want_public);
    let mut committed = committed.iter();
    let mut public = public.iter();
    let mut values: Vec<Ext> = kinds
        .iter()
        .map(|kind| match kind {
            FactorKind::Committed(_) => committed.next(),
            FactorKind::Public => public.next(),
        })
        .map(|v| *v.expect("the counts were checked"))
        .collect();
    values.extend(weights);
    values
}

/// ★ `stark::multilinear_table::verify` (`multilinear_table.rs:721-785`),
/// emitted.
///
/// `z`, `alpha_powers` and `beta` are the LogUp and batching challenges, drawn
/// ONCE for the whole epoch and shared by every table, so they arrive as wires.
/// Everything else this leg needs it draws itself.
///
/// # ★ The challenges are DERIVED, which is what makes an honest proof the gate
///
/// The leg drives the transcript, so a wrong challenge anywhere leaves a
/// refusal the machine cannot satisfy: the GKR layer relation, the batch's
/// residual, and `claim_reduce`'s shifted read are three independent
/// divisions-by-zero. Executing a real proof is therefore the challenge-stream
/// comparison, and the verdict compared against the host's is the value gate.
///
/// # ⚠ One reordering, stated so it is not mistaken for a difference
///
/// The host interleaves the GKR ladder's transcript operations with its
/// arithmetic; this draws a layer's challenges and then runs
/// [`emit_gkr_verify`] over all the layers. The ORDER OF TRANSCRIPT OPERATIONS
/// is unchanged, which is the only order that is a property of the protocol:
/// `gkr::verify` absorbs nothing it computed — every absorb is proof data — so
/// where the arithmetic sits between the absorbs is not observable. A
/// straight-line program is a DAG, and this is the same non-difference V1d
/// recorded when moving two independent emissions past each other.
///
/// # ⛔ `check_preprocessed` IS NOT EMITTED, and this is the seam
///
/// The host evaluates each preprocessed column's MLE at the reduced point
/// (`check_preprocessed`), a full pass per column. A verifier that is itself
/// proven cannot pay it. The caller supplies the replacement and the two are
/// named here rather than anywhere else: BITWISE by
/// [`super::preprocessed::emit_bitwise_preprocessed`], DECODE by its own pinned
/// commitment group. A table with no preprocessed columns owes nothing, and the
/// gate below is on such a table — said so rather than left to read as
/// coverage.
///
/// ★ The host now names half of that itself: `verify`'s `settled_out_of_band`
/// is how many LEADING preprocessed columns a prepared opening already settled
/// at this very point, and those are skipped. So DECODE's replacement is not a
/// thing this emitter invents — it is the argument the host takes, and the two
/// sides have to agree on the same count. What stays owed here is the
/// REMAINDER: every preprocessed column past that prefix is still an MLE pass
/// the host makes and this leg does not.
pub fn emit_table_verify(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    proof: &TableProofWires<'_>,
    shape: &TableShape<'_>,
    z: Ext,
    alpha_powers: &[Ext],
    beta: Ext,
) -> TableVerdictWires {
    let num_vars = shape.num_vars;
    let layers = shape.gkr_layers();
    assert_eq!(
        proof.gkr.len(),
        layers,
        "one GKR layer per variable of the input layer"
    );
    assert_eq!(
        proof.sumcheck.len(),
        num_vars,
        "the main sumcheck runs over the table's rows"
    );
    let degree = shape.sumcheck_degree();

    // The table's share of the bus, before anything is drawn from it.
    transcript.absorb_ext(b, proof.bus_output.0);
    transcript.absorb_ext(b, proof.bus_output.1);

    // `gkr::verify`'s draws, layer by layer: the batching lambda, one challenge
    // per sumcheck round, then the folding point behind the four halves.
    let mut drawn: Vec<GkrLayerChallenges> = Vec::with_capacity(layers);
    for layer in proof.gkr {
        let lambda = transcript.sample_ext(b);
        let mut rounds = Vec::with_capacity(layer.sumcheck.len());
        for evaluations in &layer.sumcheck {
            for evaluation in evaluations {
                transcript.absorb_ext(b, *evaluation);
            }
            rounds.push(transcript.sample_ext(b));
        }
        for value in [layer.p_lo, layer.p_hi, layer.q_lo, layer.q_hi] {
            transcript.absorb_ext(b, value);
        }
        let c = transcript.sample_ext(b);
        drawn.push(GkrLayerChallenges { lambda, rounds, c });
    }
    let gkr_claim = emit_gkr_verify(b, proof.bus_output, proof.gkr, &drawn);

    // The zerocheck's own weight point.
    let r: Vec<Ext> = (0..num_vars).map(|_| transcript.sample_ext(b)).collect();
    let (weight_r, weight_z) = weight_slots(shape.kinds.len());

    // The row half of the GKR claim's point is where the bus's weight table
    // belongs (`logup.rs:234`).
    let row_point = &gkr_claim.point[gkr_claim.point.len() - num_vars..];
    let betas = emit_challenge_powers(b, beta, shape.ir.num_roots());

    // `batch::verify`: the three claims, then the batching challenge. The
    // zerocheck's claim is the LITERAL zero — the constraints must vanish — so
    // it is a constant on both sides of the transcript.
    let zero = b.ext_const(&FEE::zero());
    transcript.absorb_ext(b, zero);
    transcript.absorb_ext(b, gkr_claim.p);
    transcript.absorb_ext(b, gkr_claim.q);
    let batching = transcript.sample_ext(b);
    let lambdas = emit_challenge_powers(b, batching, RULES);
    // `Σ lambda^i · claim_i`, and the first claim contributes nothing.
    let claimed = b.emul(lambdas[1], gkr_claim.p);
    let claimed = b.emul_add(lambdas[2], gkr_claim.q, claimed);

    let mut point: Vec<Ext> = Vec::with_capacity(num_vars);
    for evaluations in proof.sumcheck {
        assert_eq!(
            evaluations.len(),
            degree,
            "a round carries the batch's degree in evaluations"
        );
        for evaluation in evaluations {
            transcript.absorb_ext(b, *evaluation);
        }
        point.push(transcript.sample_ext(b));
    }
    let residual = emit_sumcheck_rounds(b, claimed, proof.sumcheck, &point);

    // `values_at`: the selectors the AIR made public, then the two weight
    // tables, woven back together with the committed values.
    let mut public: Vec<Ext> = shape
        .ir
        .public_selectors()
        .iter()
        .map(|selector| emit_selector(b, *selector, &point))
        .collect();
    public.push(emit_eq_eval(b, &r, &point));
    public.push(emit_eq_eval(b, row_point, &point));
    let values = weave(shape.kinds, proof.factor_values, &public);

    // The three rules at that point. `IrShape::program` applies the zerocheck's
    // own weight, which `combine` does not (`multilinear_air.rs:889-916`).
    let combined = emit_combine(b, shape.ir, &betas, &values);
    let zerocheck = b.emul(values[weight_r], combined);
    let bus = emit_claim_statements(
        b,
        shape.bus,
        &BusInputs {
            claim_point: &gkr_claim.point,
            num_row_vars: num_vars,
            z,
            alpha_powers,
            values: &values,
            weight: weight_z,
        },
    );

    // `Σ lambda^i · rule_i`, and the first weight is the literal one.
    let rebuilt = b.emul_add(lambdas[1], bus.numerator, zerocheck);
    let rebuilt = b.emul_add(lambdas[2], bus.denominator, rebuilt);
    // The host's `BatchMismatch`: a division by zero has no satisfying
    // assignment, so a proof that fails it cannot be executed.
    b.assert_eq_ext(rebuilt, residual);

    let reduced = emit_claim_reduce_verify(
        b,
        transcript,
        &proof.reduce,
        &shape.sources(),
        proof.factor_values,
        &point,
        shape.num_columns,
    );

    TableVerdictWires {
        bus_output: proof.bus_output,
        point: reduced.point,
        column_values: reduced.column_values,
    }
}

/// The three statements one table batches: the zerocheck and the bus's two.
const RULES: usize = 3;

/// Rows an `assert_eq_ext` lowers to: the difference and the division by zero
/// (`builder.rs:289-292`).
const ASSERT_ROWS: usize = 2;

/// What a table's verify costs: the straight-line rows, and the sponge.
///
/// Two halves because they are two different measurements — the leg's rows are
/// a function of the shape alone, the sponge's of what the transcript was
/// holding when the table started, which is why [`table_verify_cost`] takes an
/// entry and hands one back.
pub struct TableCost {
    pub leg: Cost,
    pub schedule: SpongeSchedule,
}

impl TableCost {
    pub fn rows(&self) -> usize {
        self.leg.rows() + self.schedule.rows()
    }

    pub fn perms(&self) -> usize {
        self.schedule.perms()
    }

    /// Where the sponge is left, so the next table in the same program
    /// continues from it.
    pub fn entry(&self) -> SpongeEntry {
        self.schedule.entry()
    }
}

/// INSTRUCTIONS and permutations [`emit_table_verify`] costs.
///
/// Every term by the shape it comes from, with `L` the ladder's length, `V` the
/// table's variables and `d` the batch's degree:
///
/// - the bus output: two absorbs;
/// - the ladder: per layer `i`, one draw, `i` rounds of three absorbs and a
///   draw, four absorbs and a draw — plus `gkr_verify_rows(L)` and its
///   constants;
/// - `V` draws for the zerocheck's weight point;
/// - the beta ladder, `challenge_powers_rows(num_roots)`;
/// - three claim absorbs and the batching draw, then `challenge_powers_rows(3)`
///   and TWO rows for `Σ lambda^i·claim_i`, because the zerocheck's claim is the
///   literal zero and contributes nothing;
/// - `V` rounds of `d` absorbs, a draw and `sumcheck_round_rows(d)`, plus the
///   round's interned Newton constants once;
/// - one selector per public factor, and two `eq`s for the weight tables;
/// - `combine_rows` and one multiply for the zerocheck rule's own weight;
/// - the bus statements, `claim_statements_cost`;
/// - two rows for `Σ lambda^i·rule_i`, the first weight being one, and the
///   refusal;
/// - `claim_reduce_rows`.
///
/// The `LFM_CONST` for the zero claim is interned like any other, and shares
/// with whatever else in the program needs a zero.
pub fn table_verify_cost(shape: &TableShape<'_>, entry: SpongeEntry) -> TableCost {
    let num_vars = shape.num_vars;
    let layers = shape.gkr_layers();
    let degree = shape.sumcheck_degree();
    let mut leg = Cost::default();
    let mut schedule = SpongeSchedule::new(entry);

    // The bus output.
    absorb_ext(&mut leg, &mut schedule, 2);

    // The ladder's transcript, then its arithmetic.
    for i in 0..layers {
        draw_ext(&mut leg, &mut schedule, 1);
        for _ in 0..i {
            absorb_ext(&mut leg, &mut schedule, GKR_SUMCHECK_DEGREE);
            draw_ext(&mut leg, &mut schedule, 1);
        }
        absorb_ext(&mut leg, &mut schedule, 4);
        draw_ext(&mut leg, &mut schedule, 1);
    }
    leg.ops(gkr_verify_rows(layers));

    // The zerocheck's weight point.
    draw_ext(&mut leg, &mut schedule, num_vars);

    // The beta ladder.
    leg.ops(challenge_powers_rows(shape.ir.num_roots()));

    // The batch: three claims, the batching draw, its powers, and the fold —
    // whose first term is the literal zero and costs nothing.
    absorb_ext(&mut leg, &mut schedule, RULES);
    draw_ext(&mut leg, &mut schedule, 1);
    leg.ops(challenge_powers_rows(RULES));
    leg.ops(RULES - 1);

    // The main sumcheck.
    for _ in 0..num_vars {
        absorb_ext(&mut leg, &mut schedule, degree);
        draw_ext(&mut leg, &mut schedule, 1);
        leg.ops(sumcheck_round_rows(degree));
    }

    // `values_at`.
    for selector in shape.ir.public_selectors() {
        selector_cost(*selector, num_vars, &mut leg);
    }
    leg.ops(2 * eq_eval_rows_again(num_vars));

    // The three rules, their fold and the refusal. `combine_rows` folds the
    // DAG's own constants in as ROWS; this form owns every constant in one pool
    // instead, so they come back out here and go in as values below — a DAG
    // constant that is also the `1` every `eq` seeds costs one row, not two.
    leg.ops(combine_rows(shape.ir) - dag_constant_rows(shape.ir));
    leg.op();
    leg.merge(&claim_statements_cost(shape.bus, num_vars));
    leg.ops(RULES - 1);
    leg.ops(ASSERT_ROWS);

    // `claim_reduce`. Its own form already carries its absorbs and draws as
    // ROWS, so only the sponge is replayed here.
    let sources = shape.sources();
    leg.ops(claim_reduce_rows(&sources, shape.num_columns, num_vars));
    for _ in 0..sources.len() {
        schedule.absorb(COORDINATES_PER_EXT);
    }
    schedule.draw_ext();
    for _ in 0..num_vars {
        for _ in 0..REDUCE_DEGREE {
            schedule.absorb(COORDINATES_PER_EXT);
        }
        schedule.draw_ext();
    }
    for _ in 0..shape.num_columns {
        schedule.absorb(COORDINATES_PER_EXT);
    }

    // The constants, by VALUE, because the pool is one per program and the
    // landed forms report their own as counts that cannot know what collides.
    for value in table_constants(shape, degree) {
        leg.constant(value);
    }
    // ★ And the SPONGE's own: `algebraic_leaf_hash` interns
    // `leaf_capacity(felts)` for the leaf it is about to hash
    // (`edsl.rs:676-692`), so the program pays one `LFM_CONST` per DISTINCT
    // leaf length it ever hashes. No per-hash form can carry this — the pool is
    // per PROGRAM and a hash does not know what other lengths the program will
    // use — which is why it is added here, off the finished schedule, and why
    // `SpongeHash::rows()` is right to leave it out.
    for hash in schedule.hashes() {
        leg.constant_word(leaf_capacity(hash.felts()));
    }

    TableCost { leg, schedule }
}

/// `absorb_ext` on both halves: one `Unpack` and three felts into the sponge.
fn absorb_ext(leg: &mut Cost, schedule: &mut SpongeSchedule, count: usize) {
    leg.ops(count * absorb_unpack_rows());
    for _ in 0..count {
        schedule.absorb(COORDINATES_PER_EXT);
    }
}

/// `sample_ext` on both halves: one `Pack` and three candidates.
fn draw_ext(leg: &mut Cost, schedule: &mut SpongeSchedule, count: usize) {
    leg.ops(count * sample_ext_rows());
    for _ in 0..count {
        schedule.draw_ext();
    }
}

/// `u_j = (r − j)/(j + 1)`'s two interned constants, for `j = 1 .. d−1` — the
/// same values `whir_poly::emit_newton_step` interns, written out here so the
/// pool can be predicted BY VALUE rather than by count.
///
/// Counting them would not do: `claim_reduce` runs at degree 2 and the GKR
/// ladder at 3, so `j = 1`'s pair is the same pair twice and the program pays
/// for it once. `sumcheck_round_consts` is the count for a leg standing alone.
fn newton_constants(degree: usize) -> Vec<FEE> {
    let d = degree.max(1);
    let mut values = Vec::with_capacity(2 * d.saturating_sub(1));
    for j in 1..d {
        let inv = FE::from((j + 1) as u64)
            .inv()
            .expect("j + 1 is a small nonzero Goldilocks element");
        values.push(FEE::new([inv, FE::zero(), FE::zero()]));
        values.push(FEE::new([
            FE::zero() - FE::from(j as u64) * inv,
            FE::zero(),
            FE::zero(),
        ]));
    }
    values
}

/// Every constant a table's verify interns, by value.
///
/// The `1` that seeds every `eq` and every challenge ladder; the `0` the
/// refusals divide by and the zerocheck's claim is; the Newton pairs of the
/// three sumcheck degrees a table runs — the ladder's 3, the batch's, and
/// `claim_reduce`'s 2 — which overlap; and the constraint DAG's own `Fixed`
/// values, which the AIR chose.
/// The rows `whir_program::steps_rows` charges for the DAG's CONSTANTS: one per
/// distinct `Fixed` value, and the zero a `Neg` subtracts from.
///
/// Counted the same way it counts them, so that subtracting it from
/// `combine_rows` leaves exactly the operations.
///
/// ⚠ AND THE EMPTY DAG'S ZERO, which is the one case where `combine_rows`'
/// single row is not an operation at all. A shape with NO ROOTS takes
/// `combine_rows`' early return of 1 (`whir_air.rs:74`), and what
/// `emit_combine` emits for it is `b.ext_const(&FEE::zero())`
/// (`whir_air.rs:125`) — an `LFM_CONST`, not an `LFM_XALU`. `combine_rows` is
/// in a rows-INCLUSIVE convention and is right; this function is the conversion
/// to the const-free one, so the zero has to be named HERE or the conversion
/// charges an operation nobody emits. The zero itself is already in
/// [`table_constants`], so naming it costs the pool nothing.
///
/// Found by the assembled walk's F1 over three tables whose AIRs carry
/// `EmptyConstraints`: it read exactly three rows under its prediction, one per
/// table. No fixture with an empty DAG existed before it.
fn dag_constant_rows(ir: &IrShape<GoldilocksField, GoldilocksExtension>) -> usize {
    if ir.root_steps().is_empty() {
        return 1;
    }
    let mut values: Vec<FEE> = Vec::new();
    let mut negates = false;
    for step in ir.steps_as_ops() {
        match step {
            multilinear::program::Op::Fixed(value) => {
                if !values.contains(&value) {
                    values.push(value);
                }
            }
            multilinear::program::Op::Neg(_) => negates = true,
            _ => {}
        }
    }
    values.len() + usize::from(negates && !values.contains(&FEE::zero()))
}

fn table_constants(shape: &TableShape<'_>, degree: usize) -> Vec<FEE> {
    let mut values = vec![FEE::one(), FEE::zero()];
    for d in [GKR_SUMCHECK_DEGREE, degree, REDUCE_DEGREE] {
        values.extend(newton_constants(d));
    }
    for step in shape.ir.steps_as_ops() {
        if let multilinear::program::Op::Fixed(value) = step {
            values.push(value);
        }
    }
    // The zero a `Neg` subtracts from is already in the list above.
    values
}
