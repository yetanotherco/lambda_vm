//! The batched argue as a machine leg (D-BATCH B-4a): the in-guest mirror of
//! `stark::multilinear_table::batched::verify_argue`.
//!
//! One lockstep GKR ladder per bin, one front-loaded constraint sumcheck over
//! every table, and a claim reduction only for a table with a shifted read —
//! where the per-table leg ([`super::whir_table::emit_table_verify`]) runs all
//! three per table. The plan is the host's own `argue_plan`, called, never
//! restated: the bins are part of the transcript.
//!
//! Like the per-table leg this DERIVES every challenge from the transcript it
//! drives, so executing an honest proof is the challenge-stream comparison:
//! a wrong challenge anywhere leaves one of the refusals below unsatisfiable —
//! each ladder step's relation, the constraint sumcheck's final equation, a
//! shifted table's reduction, and a column read twice claimed twice the same.
//!
//! ⛔ `check_preprocessed`, the bus balance and the openings are the caller's,
//! exactly as the host's `verify_argue` leaves them.

use multilinear::claim_reduce::FactorSource;
use multilinear::constraint_argument::FactorKind;
use stark::multilinear_table::{ArguePlan, weight_slots};

use crate::tables::types::FEE;

use super::algebraic_commit::leaf_capacity;
use super::builder::{Ext, LfmBuilder};
use super::whir_air::{combine_rows, emit_combine};
use super::whir_bus::{
    BusInputs, Cost, claim_statements_cost, emit_claim_statements, emit_eq_evals, eq_evals_cost,
};
use super::whir_gkr::GKR_SUMCHECK_DEGREE;
use super::whir_poly::{
    challenge_powers_rows, emit_challenge_powers, emit_eq_eval, emit_sumcheck_rounds,
    eq_eval_rows_again, sumcheck_round_rows,
};
use super::whir_reduce::{REDUCE_DEGREE, ReduceWires, claim_reduce_rows, emit_claim_reduce_verify};
use super::whir_table::{
    TableCost, TableShape, dag_constant_rows, emit_selector, newton_constants, selector_cost, weave,
};
use super::whir_transcript::{
    COORDINATES_PER_EXT, SpongeEntry, SpongeSchedule, WhirTranscript, absorb_unpack_rows,
    sample_ext_rows,
};

/// Rows an `assert_eq_ext` lowers to (`builder.rs:289-292`).
const ASSERT_ROWS: usize = 2;

/// One ladder step as wires: the shared sumcheck's rounds (`i` of them at step
/// `i`, three evaluations each) and every active tree's four halves.
pub struct LadderStepWires {
    pub sumcheck: Vec<Vec<Ext>>,
    /// `[p_lo, p_hi, q_lo, q_hi]` per active tree, in table order.
    pub halves: Vec<[Ext; 4]>,
}

/// The `BatchedArgue` as wires. Every field is proof data.
pub struct BatchedArgueWires<'a> {
    /// Per tree, `(p, q)`: one a table, or one for each chunk of a table the
    /// plan cuts ([`ArguePlan::chunks`]), table order then chunk order.
    pub bus_outputs: &'a [(Ext, Ext)],
    /// Per bin, per step.
    pub gkr: &'a [Vec<LadderStepWires>],
    /// `n_max` rounds of `D_max` evaluations.
    pub constraint: &'a [Vec<Ext>],
    /// Per table, its committed factors' values at `r_{<n_T}`.
    pub factor_values: &'a [Vec<Ext>],
    /// Per table, its reduction: present exactly for a table with a shift.
    pub reduces: &'a [Option<ReduceWires<'a>>],
}

/// What the leg leaves per table.
pub struct BatchedVerdictWires {
    /// Per table, its bus output: its one tree's, or — a cut table — its
    /// chunks' fractions added, `Σ p_j/q_j` as one `(p, q)`.
    pub bus_outputs: Vec<(Ext, Ext)>,
    /// Per table, the point its columns are claimed at and their values.
    pub tables: Vec<(Vec<Ext>, Vec<Ext>)>,
}

/// `Σ_a w_{2a}·(p_lo q_hi + p_hi q_lo) + w_{2a+1}·q_lo q_hi` with `w_0` the
/// literal one: `5|A| − 1` rows.
fn emit_relation(b: &mut LfmBuilder, weights: &[Ext], halves: &[[Ext; 4]]) -> Ext {
    let mut acc: Option<Ext> = None;
    for (a, h) in halves.iter().enumerate() {
        let [p_lo, p_hi, q_lo, q_hi] = *h;
        let cross = b.emul(p_lo, q_hi);
        let numerator = b.emul_add(p_hi, q_lo, cross);
        let denominator = b.emul(q_lo, q_hi);
        let with_numerator = match acc {
            None => numerator,
            Some(running) => b.emul_add(weights[2 * a], numerator, running),
        };
        acc = Some(b.emul_add(weights[2 * a + 1], denominator, with_numerator));
    }
    acc.expect("a ladder step has an active tree")
}

/// `lo + c·(hi − lo)`.
fn emit_line(b: &mut LfmBuilder, lo: Ext, hi: Ext, c: Ext) -> Ext {
    let spread = b.esub(hi, lo);
    b.emul_add(c, spread, lo)
}

/// A cut table's chunk claims, all at one point `P`, combined at `s'` into
/// the claim of its one padded tree at `s' ‖ P` — the host's
/// `combine_chunk_claims`: `p̃ = Σ_{j<c} eq(s', j)·p̃_j` and
/// `q̃ = Σ_{j<c} eq(s', j)·q̃_j + Σ_{c≤j<2^ℓ} eq(s', j)`. `c + 1 + ℓ + 2·(2^ℓ − 1)
/// + c + 2^ℓ − c` rows (the eq table's seed is a constant).
fn emit_combine_chunks(
    b: &mut LfmBuilder,
    chunks: &[(Vec<Ext>, Ext, Ext)],
    combine: &[Ext],
) -> (Vec<Ext>, Ext, Ext) {
    let weights = emit_eq_evals(b, combine);
    assert!(
        chunks.len() <= weights.len(),
        "the chunks fit the combination"
    );
    // One height in one bin: the ladder left every chunk at the same wires.
    assert!(
        chunks.iter().all(|(at, _, _)| at.len() == chunks[0].0.len()
            && at.iter().zip(&chunks[0].0).all(|(x, y)| x.0 == y.0)),
        "a cut table's chunks end at one point"
    );
    let mut p: Option<Ext> = None;
    let mut q: Option<Ext> = None;
    for ((_, pj, qj), w) in chunks.iter().zip(&weights) {
        p = Some(match p {
            None => b.emul(*w, *pj),
            Some(acc) => b.emul_add(*w, *pj, acc),
        });
        q = Some(match q {
            None => b.emul(*w, *qj),
            Some(acc) => b.emul_add(*w, *qj, acc),
        });
    }
    let p = p.expect("a cut table has chunks");
    let mut q = q.expect("a cut table has chunks");
    for w in &weights[chunks.len()..] {
        q = b.eadd(q, *w);
    }
    let point = combine.iter().chain(&chunks[0].0).copied().collect();
    (point, p, q)
}

/// A cut table's chunk outputs as one fraction: `p/q = Σ_j p_j/q_j`, three rows
/// a chunk after the first.
fn emit_add_fractions(b: &mut LfmBuilder, outputs: &[(Ext, Ext)]) -> (Ext, Ext) {
    let (mut p, mut q) = outputs[0];
    for &(pj, qj) in &outputs[1..] {
        let scaled = b.emul(p, qj);
        p = b.emul_add(pj, q, scaled);
        q = b.emul(q, qj);
    }
    (p, q)
}

/// ★ `verify_argue`, emitted.
///
/// `z`, `alpha_powers` and `beta` are the proof's shared challenges, drawn by
/// the roots block before this leg. `shapes` are the tables' structures, in
/// table order, and `plan` the host's `argue_plan` over them.
#[allow(clippy::too_many_arguments)]
pub fn emit_batched_argue(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    proof: &BatchedArgueWires<'_>,
    shapes: &[TableShape<'_>],
    plan: &ArguePlan,
    z: Ext,
    alpha_powers: &[Ext],
    beta: Ext,
) -> BatchedVerdictWires {
    let n = shapes.len();
    assert_eq!(plan.shapes.len(), n, "the plan is over these tables");
    assert_eq!(
        proof.bus_outputs.len(),
        plan.trees(),
        "one bus output per tree"
    );
    assert_eq!(proof.gkr.len(), plan.bins.len(), "one ladder per bin");
    assert_eq!(proof.factor_values.len(), n, "factor values per table");
    assert_eq!(proof.reduces.len(), n, "a reduction slot per table");

    // ── G: per bin, its trees' outputs, then its ladder, then each cut
    // table's chunks combined ──
    let mut claims: Vec<Option<(Vec<Ext>, Ext, Ext)>> = vec![None; n];
    let mut outputs: Vec<Option<(Ext, Ext)>> = vec![None; n];
    for (bin, ladder) in plan.bins.iter().zip(proof.gkr) {
        // The bin's trees: each table's chunks, in order, as `(table, tree)`.
        let members: Vec<(usize, usize)> = bin
            .iter()
            .flat_map(|&t| {
                let first = plan.first_tree(t);
                (0..plan.chunks[t].count).map(move |j| (t, first + j))
            })
            .collect();
        for &(_, tree) in &members {
            transcript.absorb_ext(b, proof.bus_outputs[tree].0);
            transcript.absorb_ext(b, proof.bus_outputs[tree].1);
        }
        let heights: Vec<usize> = members.iter().map(|&(t, _)| plan.tree_vars(t)).collect();
        let steps = heights.iter().copied().max().unwrap_or(0);
        assert_eq!(ladder.len(), steps, "the ladder runs to the tallest tree");
        let mut running: Vec<(Ext, Ext)> = members
            .iter()
            .map(|&(_, tree)| proof.bus_outputs[tree])
            .collect();
        let mut leaves: Vec<Option<(Vec<Ext>, Ext, Ext)>> = vec![None; members.len()];
        for a in 0..members.len() {
            if heights[a] == 0 {
                leaves[a] = Some((Vec::new(), running[a].0, running[a].1));
            }
        }
        let mut point: Vec<Ext> = Vec::new();
        for (i, step) in ladder.iter().enumerate() {
            let active: Vec<usize> = (0..members.len()).filter(|&a| heights[a] > i).collect();
            assert_eq!(step.halves.len(), active.len(), "halves per active tree");
            assert_eq!(step.sumcheck.len(), i, "step i's sumcheck has i rounds");
            let mu = transcript.sample_ext(b);
            let mut rounds = Vec::with_capacity(i);
            for evaluations in &step.sumcheck {
                assert_eq!(evaluations.len(), GKR_SUMCHECK_DEGREE);
                for e in evaluations {
                    transcript.absorb_ext(b, *e);
                }
                rounds.push(transcript.sample_ext(b));
            }
            for h in &step.halves {
                for v in h {
                    transcript.absorb_ext(b, *v);
                }
            }
            let c = transcript.sample_ext(b);

            let weights = emit_challenge_powers(b, mu, 2 * active.len());
            let mut claimed: Option<Ext> = None;
            for (k, &a) in active.iter().enumerate() {
                let (p, q) = running[a];
                let with_p = match claimed {
                    None => p,
                    Some(acc) => b.emul_add(weights[2 * k], p, acc),
                };
                claimed = Some(b.emul_add(weights[2 * k + 1], q, with_p));
            }
            let claimed = claimed.expect("a step has an active tree");
            let residual = emit_sumcheck_rounds(b, claimed, &step.sumcheck, &rounds);
            let eq_at = emit_eq_eval(b, &point, &rounds);
            let relation = emit_relation(b, &weights, &step.halves);
            let expected = b.emul(eq_at, relation);
            b.assert_eq_ext(expected, residual);

            point = std::iter::once(c).chain(rounds).collect();
            for (k, &a) in active.iter().enumerate() {
                let [p_lo, p_hi, q_lo, q_hi] = step.halves[k];
                running[a] = (emit_line(b, p_lo, p_hi, c), emit_line(b, q_lo, q_hi, c));
                if heights[a] == i + 1 {
                    leaves[a] = Some((point.clone(), running[a].0, running[a].1));
                }
            }
        }
        // Each table's claim and output: its one tree's, or its chunks'
        // combined at `s'`, drawn after the bin's ladder (the host's
        // `table_claim`).
        let mut at_tree = 0;
        for &t in bin {
            let count = plan.chunks[t].count;
            let chunks: Vec<(Vec<Ext>, Ext, Ext)> = leaves[at_tree..at_tree + count]
                .iter()
                .map(|c| c.clone().expect("every tree reaches its input layer"))
                .collect();
            let first = plan.first_tree(t);
            let tree_outputs = &proof.bus_outputs[first..first + count];
            at_tree += count;
            if count == 1 {
                claims[t] = chunks.into_iter().next();
                outputs[t] = Some(tree_outputs[0]);
                continue;
            }
            let combine: Vec<Ext> = (0..plan.chunks[t].combine_vars())
                .map(|_| transcript.sample_ext(b))
                .collect();
            claims[t] = Some(emit_combine_chunks(b, &chunks, &combine));
            outputs[t] = Some(emit_add_fractions(b, tree_outputs));
        }
    }
    let claims: Vec<(Vec<Ext>, Ext, Ext)> = claims
        .into_iter()
        .map(|c| c.expect("every table's tree reaches its input layer"))
        .collect();
    let outputs: Vec<(Ext, Ext)> = outputs
        .into_iter()
        .map(|o| o.expect("every table has its output"))
        .collect();

    // ── C: one front-loaded sumcheck ──
    let xi: Vec<Ext> = (0..plan.num_vars)
        .map(|_| transcript.sample_ext(b))
        .collect();
    for (_, p, q) in &claims {
        transcript.absorb_ext(b, *p);
        transcript.absorb_ext(b, *q);
    }
    let lambda = transcript.sample_ext(b);
    assert_eq!(proof.constraint.len(), plan.num_vars, "n_max rounds");
    let mut r: Vec<Ext> = Vec::with_capacity(plan.num_vars);
    for evaluations in proof.constraint {
        assert_eq!(evaluations.len(), plan.degree, "D_max evaluations a round");
        for e in evaluations {
            transcript.absorb_ext(b, *e);
        }
        r.push(transcript.sample_ext(b));
    }

    let lambda2 = b.emul(lambda, lambda);
    let lambda3 = b.emul(lambda2, lambda);
    let weights = emit_challenge_powers(b, lambda3, n);
    let mut claimed: Option<Ext> = None;
    for (t, (_, p, q)) in claims.iter().enumerate() {
        let inner = b.emul(lambda, *p);
        let inner = b.emul_add(lambda2, *q, inner);
        claimed = Some(match claimed {
            None => inner,
            Some(acc) => b.emul_add(weights[t], inner, acc),
        });
    }
    let claimed = claimed.expect("a proof argues a table");
    let residual = emit_sumcheck_rounds(b, claimed, proof.constraint, &r);

    // `Π_{j ≥ m} r_j` for each table's `m = n_T < n_max`, from the end.
    let lowest = shapes.iter().map(|s| s.num_vars).min().unwrap_or(0);
    let mut suffix: Vec<Option<Ext>> = vec![None; plan.num_vars + 1];
    for j in (lowest..plan.num_vars).rev() {
        suffix[j] = Some(match suffix[j + 1] {
            None => r[j],
            Some(rest) => b.emul(r[j], rest),
        });
    }
    let max_roots = shapes.iter().map(|s| s.ir.num_roots()).max().unwrap_or(0);
    let betas = emit_challenge_powers(b, beta, max_roots);
    let mut rebuilt: Option<Ext> = None;
    for (t, shape) in shapes.iter().enumerate() {
        let num_vars = shape.num_vars;
        let at = &r[..num_vars];
        let (weight_r, weight_z) = weight_slots(shape.kinds.len());
        let (claim_point, _, _) = &claims[t];
        let row_point = &claim_point[claim_point.len() - num_vars..];
        let mut public: Vec<Ext> = shape
            .ir
            .public_selectors()
            .iter()
            .map(|selector| emit_selector(b, *selector, at))
            .collect();
        public.push(emit_eq_eval(b, &xi[..num_vars], at));
        public.push(emit_eq_eval(b, row_point, at));
        let values = weave(shape.kinds, &proof.factor_values[t], &public);
        let combined = emit_combine(b, shape.ir, &betas[..shape.ir.num_roots()], &values);
        let zerocheck = b.emul(values[weight_r], combined);
        let bus = emit_claim_statements(
            b,
            shape.bus,
            &BusInputs {
                claim_point,
                num_row_vars: num_vars,
                z,
                alpha_powers,
                values: &values,
                weight: weight_z,
            },
        );
        let value = b.emul_add(lambda, bus.numerator, zerocheck);
        let value = b.emul_add(lambda2, bus.denominator, value);
        let term = match suffix[num_vars] {
            Some(padding) if num_vars < plan.num_vars => b.emul(value, padding),
            _ => value,
        };
        rebuilt = Some(match rebuilt {
            None => term,
            Some(acc) => b.emul_add(weights[t], term, acc),
        });
    }
    let rebuilt = rebuilt.expect("a proof argues a table");
    // The host's `BatchMismatch`.
    b.assert_eq_ext(rebuilt, residual);
    for values in proof.factor_values {
        for v in values {
            transcript.absorb_ext(b, *v);
        }
    }

    // ── Rd ──
    let mut tables = Vec::with_capacity(n);
    for (t, shape) in shapes.iter().enumerate() {
        let at = &r[..shape.num_vars];
        let sources = shape.sources();
        let factor_values = &proof.factor_values[t];
        match (plan.shapes[t].shifted, &proof.reduces[t]) {
            (true, Some(reduce)) => {
                let reduced = emit_claim_reduce_verify(
                    b,
                    transcript,
                    reduce,
                    &sources,
                    factor_values,
                    at,
                    shape.num_columns,
                );
                tables.push((reduced.point, reduced.column_values));
            }
            (false, None) => {
                let columns = emit_direct_columns(b, &sources, factor_values, shape.num_columns);
                tables.push((at.to_vec(), columns));
            }
            _ => panic!("table {t}: a reduction is present exactly for a table with a shift"),
        }
    }

    BatchedVerdictWires {
        bus_outputs: outputs,
        tables,
    }
}

/// A table's column claims when every committed factor reads its column
/// unshifted: each column's claim is its factor's value, and a column read
/// twice must be claimed the same both times (an assert each).
fn emit_direct_columns(
    b: &mut LfmBuilder,
    sources: &[FactorSource],
    factor_values: &[Ext],
    num_columns: usize,
) -> Vec<Ext> {
    assert_eq!(sources.len(), factor_values.len(), "one value per factor");
    let mut columns: Vec<Option<Ext>> = vec![None; num_columns];
    for (source, value) in sources.iter().zip(factor_values) {
        assert_eq!(source.offset, 0, "an unshifted table");
        match columns[source.column] {
            Some(claimed) => b.assert_eq_ext(claimed, *value),
            None => columns[source.column] = Some(*value),
        }
    }
    columns
        .into_iter()
        .map(|c| c.expect("every column is read by a factor"))
        .collect()
}

/// Each of a bin's trees' input variables, in the ladder's order: each
/// table's chunks, in table order.
fn tree_heights(plan: &ArguePlan, bin: &[usize]) -> Vec<usize> {
    bin.iter()
        .flat_map(|&t| std::iter::repeat_n(plan.tree_vars(t), plan.chunks[t].count))
        .collect()
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

/// ★ INSTRUCTIONS and permutations [`emit_batched_argue`] costs, from the
/// shapes and the plan alone — every term by the emitter's own shape:
///
/// - G, per bin: two absorbs per table; per step `i` with `A` active trees, a
///   draw, `i` rounds of three absorbs and a draw, `4A` absorbs and a draw;
///   then `challenge_powers_rows(2A)`, `2A − 1` rows for the claim, `i` degree-3
///   rounds, `eq` over `i`, `5A − 1` for the relation, one `Mul` by `eq`, the
///   assert, and `4A` for the two lines per tree;
/// - C: `n_max` draws for `ξ`, `2|T|` absorbs, the `λ` draw, `n_max` rounds of
///   `D_max` absorbs and a draw; `λ²`, `λ³`, `challenge_powers_rows(|T|)`,
///   `3|T| − 1` rows for the claim, `n_max` degree-`D_max` rounds, the suffix
///   products (`n_max − min n_T − 1` rows when some table is short), the beta
///   ladder once at the widest table's roots; per table its selectors, two
///   `eq`s, `combine_rows` less its constants, one `Mul`, the bus statements,
///   two rows for the rule, one for the padding when short, one to fold (none
///   for the first); the assert; one absorb per factor value;
/// - Rd: a shifted table's `claim_reduce_rows` and sponge; an unshifted one's
///   asserts for columns read twice.
pub fn batched_argue_cost(
    shapes: &[TableShape<'_>],
    plan: &ArguePlan,
    entry: SpongeEntry,
) -> TableCost {
    let n = shapes.len();
    let mut leg = Cost::default();
    let mut schedule = SpongeSchedule::new(entry);
    let one = FEE::one();
    let zero = FEE::zero();
    let mut degrees: Vec<usize> = Vec::new();

    // ── G ──
    for bin in &plan.bins {
        let heights = tree_heights(plan, bin);
        absorb_ext(&mut leg, &mut schedule, 2 * heights.len());
        let steps = heights.iter().copied().max().unwrap_or(0);
        for i in 0..steps {
            let active = heights.iter().filter(|&&h| h > i).count();
            draw_ext(&mut leg, &mut schedule, 1);
            for _ in 0..i {
                absorb_ext(&mut leg, &mut schedule, GKR_SUMCHECK_DEGREE);
                draw_ext(&mut leg, &mut schedule, 1);
            }
            absorb_ext(&mut leg, &mut schedule, 4 * active);
            draw_ext(&mut leg, &mut schedule, 1);

            leg.constant(one);
            leg.ops(challenge_powers_rows(2 * active));
            leg.ops(2 * active - 1);
            leg.ops(i * sumcheck_round_rows(GKR_SUMCHECK_DEGREE));
            if i > 0 {
                degrees.push(GKR_SUMCHECK_DEGREE);
            }
            leg.ops(eq_eval_rows_again(i));
            leg.ops(5 * active - 1);
            leg.op();
            leg.ops(ASSERT_ROWS);
            leg.constant(zero);
            leg.ops(4 * active);
        }
        // A cut table's `ℓ` draws, its eq table over them, the two sums and
        // the padding chunks' weights, and its outputs added as fractions.
        for &t in bin {
            let chunks = &plan.chunks[t];
            if chunks.count == 1 {
                continue;
            }
            let ell = chunks.combine_vars();
            draw_ext(&mut leg, &mut schedule, ell);
            eq_evals_cost(ell, &mut leg);
            leg.ops(2 * chunks.count + ((1usize << ell) - chunks.count));
            leg.ops(3 * (chunks.count - 1));
        }
    }

    // ── C ──
    draw_ext(&mut leg, &mut schedule, plan.num_vars);
    absorb_ext(&mut leg, &mut schedule, 2 * n);
    draw_ext(&mut leg, &mut schedule, 1);
    for _ in 0..plan.num_vars {
        absorb_ext(&mut leg, &mut schedule, plan.degree);
        draw_ext(&mut leg, &mut schedule, 1);
    }
    leg.ops(2);
    leg.constant(one);
    leg.ops(challenge_powers_rows(n));
    leg.ops(3 * n - 1);
    leg.ops(plan.num_vars * sumcheck_round_rows(plan.degree));
    if plan.num_vars > 0 {
        degrees.push(plan.degree);
    }
    let lowest = shapes.iter().map(|s| s.num_vars).min().unwrap_or(0);
    if lowest < plan.num_vars {
        leg.ops(plan.num_vars - lowest - 1);
    }
    let max_roots = shapes.iter().map(|s| s.ir.num_roots()).max().unwrap_or(0);
    leg.ops(challenge_powers_rows(max_roots));
    for (t, shape) in shapes.iter().enumerate() {
        let num_vars = shape.num_vars;
        for selector in shape.ir.public_selectors() {
            selector_cost(*selector, num_vars, &mut leg);
        }
        leg.ops(2 * eq_eval_rows_again(num_vars));
        leg.ops(combine_rows(shape.ir) - dag_constant_rows(shape.ir));
        for step in shape.ir.steps_as_ops() {
            if let multilinear::program::Op::Fixed(value) = step {
                leg.constant(value);
            }
        }
        if shape.ir.root_steps().is_empty() {
            leg.constant(zero);
        }
        let negates = shape
            .ir
            .steps_as_ops()
            .iter()
            .any(|step| matches!(step, multilinear::program::Op::Neg(_)));
        if negates {
            leg.constant(zero);
        }
        leg.op();
        leg.merge(&claim_statements_cost(shape.bus, num_vars));
        leg.ops(2);
        if num_vars < plan.num_vars {
            leg.op();
        }
        if t > 0 {
            leg.op();
        }
    }
    leg.ops(ASSERT_ROWS);
    leg.constant(zero);
    let factor_values: usize = shapes.iter().map(|s| s.sources().len()).sum();
    absorb_ext(&mut leg, &mut schedule, factor_values);

    // ── Rd ──
    for (t, shape) in shapes.iter().enumerate() {
        let sources = shape.sources();
        if plan.shapes[t].shifted {
            leg.ops(claim_reduce_rows(
                &sources,
                shape.num_columns,
                shape.num_vars,
            ));
            leg.constant(one);
            leg.constant(zero);
            degrees.push(REDUCE_DEGREE);
            for _ in 0..sources.len() {
                schedule.absorb(COORDINATES_PER_EXT);
            }
            schedule.draw_ext();
            for _ in 0..shape.num_vars {
                for _ in 0..REDUCE_DEGREE {
                    schedule.absorb(COORDINATES_PER_EXT);
                }
                schedule.draw_ext();
            }
            for _ in 0..shape.num_columns {
                schedule.absorb(COORDINATES_PER_EXT);
            }
        } else {
            let mut seen = vec![false; shape.num_columns];
            for source in &sources {
                if seen[source.column] {
                    leg.ops(ASSERT_ROWS);
                    leg.constant(zero);
                }
                seen[source.column] = true;
            }
        }
    }

    for d in degrees {
        for value in newton_constants(d) {
            leg.constant(value);
        }
    }
    for hash in schedule.hashes() {
        leg.constant_word(leaf_capacity(hash.felts()));
    }
    TableCost { leg, schedule }
}

/// The kinds a table's factors read — for a caller holding only the layout.
pub fn is_shifted(kinds: &[FactorKind]) -> bool {
    kinds
        .iter()
        .filter_map(FactorKind::source)
        .any(|source| source.offset != 0)
}

// ============================ inside a block leaf ===========================

/// A batched argue's wires, hinted from an arena with every count taken from
/// the tables' SHAPES and the plan — never from a proof — and OWNED, for
/// [`BatchedStore::reduces`] and [`BatchedArgueWires`] to borrow.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) struct BatchedStore {
    pub(crate) bus_outputs: Vec<(Ext, Ext)>,
    pub(crate) gkr: Vec<Vec<LadderStepWires>>,
    pub(crate) constraint: Vec<Vec<Ext>>,
    pub(crate) factor_values: Vec<Vec<Ext>>,
    /// Per table, a shifted table's reduction rounds and column values.
    reduce_parts: Vec<Option<ReduceParts>>,
}

/// A reduction's rounds and column values, as wires.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
type ReduceParts = (Vec<Vec<Ext>>, Vec<Ext>);

#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
impl BatchedStore {
    /// The reductions as wires, one slot per table.
    pub(crate) fn reduces(&self) -> Vec<Option<ReduceWires<'_>>> {
        self.reduce_parts
            .iter()
            .map(|part| {
                part.as_ref().map(|(rounds, columns)| ReduceWires {
                    sumcheck: rounds,
                    column_values: columns,
                })
            })
            .collect()
    }
}

/// How many words a batched argue over `shapes` binned by `plan` takes, in
/// [`hint_batched_shaped`]'s order: what [`push_batched_words`] must write.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn batched_words(shapes: &[TableShape<'_>], plan: &ArguePlan) -> usize {
    assert_eq!(
        plan.shapes.len(),
        shapes.len(),
        "the plan is over these tables"
    );
    let mut words = 2 * plan.trees();
    for bin in &plan.bins {
        let heights = tree_heights(plan, bin);
        let steps = heights.iter().copied().max().unwrap_or(0);
        for i in 0..steps {
            let active = heights.iter().filter(|&&h| h > i).count();
            words += i * GKR_SUMCHECK_DEGREE + 4 * active;
        }
    }
    words += plan.num_vars * plan.degree;
    words += shapes.iter().map(|s| s.sources().len()).sum::<usize>();
    for (t, shape) in shapes.iter().enumerate() {
        if plan.shapes[t].shifted {
            words += shape.num_vars * REDUCE_DEGREE + shape.num_columns;
        }
    }
    words
}

/// Hints a batched argue's wires from `arena` at `at`, in [`push_batched_words`]'
/// order: the bus outputs, each bin's ladder (each step's rounds, then its active
/// trees' halves), the constraint rounds, the factor values, then each shifted
/// table's reduction. Every count is `emit_batched_argue`'s own, from the shapes
/// and the plan, so a proof of another shape cannot be read into it.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn hint_batched_shaped(
    b: &mut LfmBuilder,
    arena: super::instr::ArenaId,
    at: &mut u32,
    shapes: &[TableShape<'_>],
    plan: &ArguePlan,
) -> BatchedStore {
    let mut take = |b: &mut LfmBuilder, count: usize| -> Vec<Ext> {
        (0..count)
            .map(|_| {
                let wire = b.hint_word(arena, *at).as_ext();
                *at += 1;
                wire
            })
            .collect()
    };
    let bus_outputs = (0..plan.trees())
        .map(|_| {
            let pq = take(b, 2);
            (pq[0], pq[1])
        })
        .collect();
    let gkr = plan
        .bins
        .iter()
        .map(|bin| {
            let heights = tree_heights(plan, bin);
            let steps = heights.iter().copied().max().unwrap_or(0);
            (0..steps)
                .map(|i| {
                    let active = heights.iter().filter(|&&h| h > i).count();
                    LadderStepWires {
                        sumcheck: (0..i).map(|_| take(b, GKR_SUMCHECK_DEGREE)).collect(),
                        halves: (0..active)
                            .map(|_| {
                                let h = take(b, 4);
                                [h[0], h[1], h[2], h[3]]
                            })
                            .collect(),
                    }
                })
                .collect()
        })
        .collect();
    let constraint = (0..plan.num_vars).map(|_| take(b, plan.degree)).collect();
    let factor_values = shapes.iter().map(|s| take(b, s.sources().len())).collect();
    let reduce_parts = shapes
        .iter()
        .enumerate()
        .map(|(t, shape)| {
            plan.shapes[t].shifted.then(|| {
                let rounds = (0..shape.num_vars)
                    .map(|_| take(b, REDUCE_DEGREE))
                    .collect();
                (rounds, take(b, shape.num_columns))
            })
        })
        .collect();
    BatchedStore {
        bus_outputs,
        gkr,
        constraint,
        factor_values,
        reduce_parts,
    }
}

/// A batched argue's words, in [`hint_batched_shaped`]'s order.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn push_batched_words(
    words: &mut Vec<super::word::LfmWord>,
    argue: &stark::multilinear_table::BatchedArgue<crate::tables::types::GoldilocksExtension>,
) {
    use super::word::ext_word;
    for (p, q) in &argue.bus_outputs {
        words.extend([ext_word(p), ext_word(q)]);
    }
    for ladder in &argue.gkr {
        for step in &ladder.layers {
            for round in &step.sumcheck.rounds {
                words.extend(round.evaluations.iter().map(ext_word));
            }
            for h in &step.halves {
                words.extend([h.p_lo, h.p_hi, h.q_lo, h.q_hi].iter().map(ext_word));
            }
        }
    }
    for round in &argue.constraint.rounds {
        words.extend(round.evaluations.iter().map(ext_word));
    }
    for values in &argue.factor_values {
        words.extend(values.iter().map(ext_word));
    }
    for reduce in argue.reduces.iter().flatten() {
        for round in &reduce.sumcheck.rounds {
            words.extend(round.evaluations.iter().map(ext_word));
        }
        words.extend(reduce.column_values.iter().map(ext_word));
    }
}

/// ★ A block group's tables, batched: [`emit_batched_argue`] on the group's
/// fork, then each table's preprocessed leg at the point its columns are
/// claimed at — the host's `check_preprocessed` after `verify_argue`, which
/// `block_verify_with` runs per table — leaving what the per-table walk leaves.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_batched_walk(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    store: &BatchedStore,
    shapes: &[TableShape<'_>],
    plan: &ArguePlan,
    plans: &[super::whir_epoch::PreprocessedPlan<'_>],
    slots: &[&[usize]],
    z: Ext,
    alpha_powers: &[Ext],
    beta: Ext,
) -> super::whir_epoch::TableWalk {
    assert_eq!(shapes.len(), plans.len(), "one preprocessed plan per table");
    assert_eq!(shapes.len(), slots.len(), "one slot map per table");
    let reduces = store.reduces();
    let verdict = emit_batched_argue(
        b,
        transcript,
        &BatchedArgueWires {
            bus_outputs: &store.bus_outputs,
            gkr: &store.gkr,
            constraint: &store.constraint,
            factor_values: &store.factor_values,
            reduces: &reduces,
        },
        shapes,
        plan,
        z,
        alpha_powers,
        beta,
    );
    let mut walk = super::whir_epoch::TableWalk {
        outputs: Vec::with_capacity(shapes.len()),
        points: Vec::with_capacity(shapes.len()),
        widths: Vec::with_capacity(shapes.len()),
        values: Vec::new(),
    };
    for (t, ((point, columns), output)) in verdict
        .tables
        .into_iter()
        .zip(verdict.bus_outputs)
        .enumerate()
    {
        let table = super::whir_table::TableVerdictWires {
            bus_output: output,
            point,
            column_values: columns,
        };
        super::whir_epoch::emit_preprocessed_leg(b, &plans[t], slots[t], &shapes[t], &table);
        walk.outputs.push(table.bus_output);
        walk.widths.push(table.column_values.len());
        walk.values.extend(table.column_values.iter().copied());
        walk.points.push(table.point);
    }
    walk
}
