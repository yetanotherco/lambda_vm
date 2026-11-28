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
use multilinear::gkr_lockstep::{LadderStep, Relation, ladder_steps};
use multilinear::whir_chain::GkrLadder;
use stark::multilinear_table::{ArguePlan, weight_slots};

use crate::tables::types::FEE;

use super::algebraic_commit::leaf_capacity;
use super::builder::{Ext, LfmBuilder};
use super::whir_air::{combine_rows, emit_combine};
use super::whir_bus::{
    BusInputs, Cost, claim_statements_cost, emit_claim_statements, emit_eq_evals, eq_evals_cost,
};
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

/// One ladder step as wires: the shared sumcheck's rounds (as many as the
/// step's parent layer has variables, each of the step's degree in
/// evaluations) and every active tree's halves — one under a one-layer
/// relation, `h₀` then `h₁` under a two-layer one
/// ([`multilinear::gkr_lockstep::ladder_steps`]).
pub struct LadderStepWires {
    pub sumcheck: Vec<Vec<Ext>>,
    /// `[p_lo, p_hi, q_lo, q_hi]` per half, the active trees in table order.
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

/// One half's fraction, `(p_lo q_hi + p_hi q_lo, q_lo q_hi)`: three rows.
fn emit_fraction(b: &mut LfmBuilder, h: [Ext; 4]) -> (Ext, Ext) {
    let [p_lo, p_hi, q_lo, q_hi] = h;
    let cross = b.emul(p_lo, q_hi);
    let numerator = b.emul_add(p_hi, q_lo, cross);
    let denominator = b.emul(q_lo, q_hi);
    (numerator, denominator)
}

/// `Σ_a w_{2a}·N_a + w_{2a+1}·Q_a` over a step's trees with `w_0` the literal
/// one, `(N_a, Q_a)` a one-layer tree's half's fraction (3 rows) or a
/// two-layer tree's `(N₀D₁ + N₁D₀, D₀D₁)` over its two halves (9 rows): the
/// host's `step_relation`. Each tree adds two rows to fold, the first one.
fn emit_relation(
    b: &mut LfmBuilder,
    weights: &[Ext],
    relations: &[(usize, Relation)],
    halves: &[[Ext; 4]],
) -> Ext {
    let mut acc: Option<Ext> = None;
    let mut at = 0;
    for (a, &(_, relation)) in relations.iter().enumerate() {
        let (numerator, denominator) = match relation {
            Relation::Single => emit_fraction(b, halves[at]),
            Relation::Double => {
                let (n0, d0) = emit_fraction(b, halves[at]);
                let (n1, d1) = emit_fraction(b, halves[at + 1]);
                let cross = b.emul(n0, d1);
                let numerator = b.emul_add(n1, d0, cross);
                let denominator = b.emul(d0, d1);
                (numerator, denominator)
            }
        };
        at += relation.halves();
        let with_numerator = match acc {
            None => numerator,
            Some(running) => b.emul_add(weights[2 * a], numerator, running),
        };
        acc = Some(b.emul_add(weights[2 * a + 1], denominator, with_numerator));
    }
    acc.expect("a ladder step has an active tree")
}

/// Rows [`emit_relation`] emits for a step's relations.
fn relation_rows(relations: &[(usize, Relation)]) -> usize {
    relations
        .iter()
        .map(|&(_, relation)| match relation {
            Relation::Single => 5,
            Relation::Double => 11,
        })
        .sum::<usize>()
        - 1
}

/// Rows the step's trees' next claims take: two lines a one-layer tree, six
/// (the plane: two lines at `c₁` and one at `c₂`, for `p` and for `q`) a
/// two-layer tree; two rows a line.
fn descend_rows(relations: &[(usize, Relation)]) -> usize {
    relations
        .iter()
        .map(|&(_, relation)| match relation {
            Relation::Single => 4,
            Relation::Double => 12,
        })
        .sum()
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
    ladder: GkrLadder,
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

    // ── G ──
    let (claims, outputs) = emit_ladders(b, transcript, proof.bus_outputs, proof.gkr, plan, ladder);
    let pairs: Vec<TablePairWires<'_>> = shapes
        .iter()
        .zip(&claims)
        .map(|(shape, claim)| TablePairWires {
            global: Some(PairWires {
                claim,
                bus: shape.bus,
                z,
                alpha_powers,
            }),
            local: None,
        })
        .collect();
    let tables = emit_argue_rest(b, transcript, proof, shapes, plan, &pairs, false, beta);
    BatchedVerdictWires {
        bus_outputs: outputs,
        tables,
    }
}

/// One bus pair of a table in the C phase, as wires: the claim its class's
/// ladders left (`(point, p̃, q̃)`), its interactions in that class, and the
/// class's challenges.
#[derive(Clone, Copy)]
pub(crate) struct PairWires<'a> {
    pub(crate) claim: &'a (Vec<Ext>, Ext, Ext),
    pub(crate) bus: &'a [stark::multilinear_logup::InteractionShape<
        crate::tables::types::GoldilocksExtension,
    >],
    pub(crate) z: Ext,
    pub(crate) alpha_powers: &'a [Ext],
}

/// A table's bus pairs in the C phase — the host's `TablePairs`: the
/// global pair at `λ, λ²` and the local pair at `λ³, λ⁴`, each when the
/// table has an interaction in that class. Every table has at least one.
#[derive(Clone, Copy)]
pub(crate) struct TablePairWires<'a> {
    pub(crate) global: Option<PairWires<'a>>,
    pub(crate) local: Option<PairWires<'a>>,
}

/// A tree's (or a cut table's combined) claim after its ladder: its point,
/// `p̃` and `q̃`.
pub(crate) type ClassClaim = (Vec<Ext>, Ext, Ext);

/// ★ Phase G over one LogUp class, emitted — the host's `verify_ladders`: per
/// bin of `plan`, its trees' outputs absorbed, its ladder, each cut table's
/// chunks combined at its `s'`. Returns per plan entry its claim
/// `(point, p̃, q̃)` and its output (a cut table's chunks added as one
/// fraction). `plan` is over the class's members, so its entry `k` is the
/// class's `k`-th member.
pub(crate) fn emit_ladders(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    bus_outputs: &[(Ext, Ext)],
    gkr: &[Vec<LadderStepWires>],
    plan: &ArguePlan,
    ladder: GkrLadder,
) -> (Vec<ClassClaim>, Vec<(Ext, Ext)>) {
    let n = plan.shapes.len();
    assert_eq!(bus_outputs.len(), plan.trees(), "one bus output per tree");
    assert_eq!(gkr.len(), plan.bins.len(), "one ladder per bin");
    let proof = LadderWires { bus_outputs, gkr };

    // ── G: per bin, its trees' outputs, then its ladder, then each cut
    // table's chunks combined ──
    let mut claims: Vec<Option<(Vec<Ext>, Ext, Ext)>> = vec![None; n];
    let mut outputs: Vec<Option<(Ext, Ext)>> = vec![None; n];
    let form = ladder;
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
        // Every step's relations, degree and counts: the host's own schedule,
        // from the heights and the format alone.
        let schedule = ladder_steps(&heights, form);
        assert_eq!(
            ladder.len(),
            schedule.len(),
            "the ladder runs to the tallest tree"
        );
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
        for (wires, step) in ladder.iter().zip(&schedule) {
            let i = step.parent;
            assert_eq!(
                wires.halves.len(),
                step.halves(),
                "the step's halves, as its relations send them"
            );
            assert_eq!(wires.sumcheck.len(), i, "a step over layer i has i rounds");
            let mu = transcript.sample_ext(b);
            let mut rounds = Vec::with_capacity(i);
            for evaluations in &wires.sumcheck {
                assert_eq!(evaluations.len(), step.degree, "the step's degree");
                for e in evaluations {
                    transcript.absorb_ext(b, *e);
                }
                rounds.push(transcript.sample_ext(b));
            }
            for h in &wires.halves {
                for v in h {
                    transcript.absorb_ext(b, *v);
                }
            }
            // `c₁`, then `c₂` — its own draw, never `c₁` again — when a tree
            // of the step descends two layers: on the diagonal `c₂ = c₁` the
            // plane cannot see `p₀₁ + δ, p₁₀ − δ`.
            let c1 = transcript.sample_ext(b);
            let c2 = step
                .draws_second_challenge()
                .then(|| transcript.sample_ext(b));

            let weights = emit_challenge_powers(b, mu, 2 * step.relations.len());
            let mut claimed: Option<Ext> = None;
            for (k, &(a, _)) in step.relations.iter().enumerate() {
                let (p, q) = running[a];
                let with_p = match claimed {
                    None => p,
                    Some(acc) => b.emul_add(weights[2 * k], p, acc),
                };
                claimed = Some(b.emul_add(weights[2 * k + 1], q, with_p));
            }
            let claimed = claimed.expect("a step has an active tree");
            let residual = emit_sumcheck_rounds(b, claimed, &wires.sumcheck, &rounds);
            let eq_at = emit_eq_eval(b, &point, &rounds);
            let relation = emit_relation(b, &weights, &step.relations, &wires.halves);
            let expected = b.emul(eq_at, relation);
            b.assert_eq_ext(expected, residual);

            let single_point: Vec<Ext> =
                std::iter::once(c1).chain(rounds.iter().copied()).collect();
            let double_point: Option<Vec<Ext>> =
                c2.map(|c2| [c1, c2].into_iter().chain(rounds.iter().copied()).collect());
            let mut at = 0;
            for &(a, relation) in &step.relations {
                let reached = match relation {
                    Relation::Single => {
                        let [p_lo, p_hi, q_lo, q_hi] = wires.halves[at];
                        running[a] = (emit_line(b, p_lo, p_hi, c1), emit_line(b, q_lo, q_hi, c1));
                        &single_point
                    }
                    Relation::Double => {
                        let c2 = c2.expect("a two-layer step draws c₂");
                        let [p0_lo, p0_hi, q0_lo, q0_hi] = wires.halves[at];
                        let [p1_lo, p1_hi, q1_lo, q1_hi] = wires.halves[at + 1];
                        let p0 = emit_line(b, p0_lo, p0_hi, c1);
                        let p1 = emit_line(b, p1_lo, p1_hi, c1);
                        let p = emit_line(b, p0, p1, c2);
                        let q0 = emit_line(b, q0_lo, q0_hi, c1);
                        let q1 = emit_line(b, q1_lo, q1_hi, c1);
                        let q = emit_line(b, q0, q1, c2);
                        running[a] = (p, q);
                        double_point.as_ref().expect("a two-layer step's point")
                    }
                };
                at += relation.halves();
                if heights[a] == i + relation.layers() {
                    leaves[a] = Some((reached.clone(), running[a].0, running[a].1));
                }
            }
            point = double_point.unwrap_or(single_point);
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
    (claims, outputs)
}

/// The ladders' messages a class carries: every tree's output and one
/// ladder a bin.
struct LadderWires<'a> {
    bus_outputs: &'a [(Ext, Ext)],
    gkr: &'a [Vec<LadderStepWires>],
}

/// ★ Phases C and Rd of the host's `verify_argue_classed`, emitted, from the
/// claims every table's pairs left: `ξ`, every global claim then every local
/// claim (table order each), `λ`, one front-loaded sumcheck of
/// `Σ_t w_t·f_t` with `w_t = λ^{5t}` when `local` (the group argues local
/// lookups) and `λ^{3t}` otherwise, each table's rules rebuilt at the
/// sumcheck's point (`C`, then its global pair at `λ, λ²` and its local pair
/// at `λ³, λ⁴` when present, each pair at its own `eq` weight slot), the
/// factor values absorbed, then each table's reduction. `plan` is the plan
/// over every table of the group (its `n_max`, `D_max` and shifts; the
/// host's `full_shapes`). Returns each table's point and column claims.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_argue_rest(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    proof: &BatchedArgueWires<'_>,
    shapes: &[TableShape<'_>],
    plan: &ArguePlan,
    pairs: &[TablePairWires<'_>],
    local: bool,
    beta: Ext,
) -> Vec<(Vec<Ext>, Vec<Ext>)> {
    let n = shapes.len();
    assert_eq!(plan.shapes.len(), n, "the plan is over these tables");
    assert_eq!(pairs.len(), n, "one pair set per table");
    assert!(
        pairs
            .iter()
            .all(|p| p.global.is_some() || p.local.is_some()),
        "every table argues in a class"
    );
    assert!(
        local || pairs.iter().all(|p| p.local.is_none()),
        "a local pair only in a group that argues local lookups"
    );
    assert_eq!(proof.factor_values.len(), n, "factor values per table");
    assert_eq!(proof.reduces.len(), n, "a reduction slot per table");

    // ── C: one front-loaded sumcheck ──
    let xi: Vec<Ext> = (0..plan.num_vars)
        .map(|_| transcript.sample_ext(b))
        .collect();
    for pair in pairs
        .iter()
        .filter_map(|p| p.global)
        .chain(pairs.iter().filter_map(|p| p.local))
    {
        let (_, p, q) = pair.claim;
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
    // A table's share `λ^{st}`: stride 3 (`C, N, D`), or 5 in a group that
    // argues local lookups (`N^L, D^L` after them) — the host's
    // `table_weights`, by the group's kind, never by the table's.
    let (lambda4, stride) = if local {
        let lambda4 = b.emul(lambda3, lambda);
        let lambda5 = b.emul(lambda4, lambda);
        (Some(lambda4), lambda5)
    } else {
        (None, lambda3)
    };
    #[cfg(test)]
    let stride = match (local, super::whir_block::local_faults::get()) {
        (true, super::whir_block::local_faults::Fault::TodaysWeights) => lambda3,
        _ => stride,
    };
    let weights = emit_challenge_powers(b, stride, n);
    // `λ^j` for a pair's `j`: the global pair's `(1, 2)`, the local's `(3, 4)`.
    let pair_powers = |first: usize| -> (Ext, Ext) {
        match first {
            1 => (lambda, lambda2),
            _ => (lambda3, lambda4.expect("a local pair in a local group")),
        }
    };
    let mut claimed: Option<Ext> = None;
    for (t, pair) in pairs.iter().enumerate() {
        let mut inner: Option<Ext> = None;
        for (first, pair) in [(1, pair.global), (3, pair.local)] {
            let Some(pair) = pair else { continue };
            let (_, p, q) = pair.claim;
            let (lp, lq) = pair_powers(first);
            let with_p = match inner {
                None => b.emul(lp, *p),
                Some(acc) => b.emul_add(lp, *p, acc),
            };
            inner = Some(b.emul_add(lq, *q, with_p));
        }
        let inner = inner.expect("every table argues in a class");
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
        let present: Vec<(usize, PairWires<'_>)> = [(1, pairs[t].global), (3, pairs[t].local)]
            .into_iter()
            .filter_map(|(first, pair)| pair.map(|pair| (first, pair)))
            .collect();
        let mut public: Vec<Ext> = shape
            .ir
            .public_selectors()
            .iter()
            .map(|selector| emit_selector(b, *selector, at))
            .collect();
        public.push(emit_eq_eval(b, &xi[..num_vars], at));
        // Each pair's `eq` weight at its claim's row point, in pair order:
        // the weight slots after the zerocheck's.
        for (_, pair) in &present {
            let (claim_point, _, _) = pair.claim;
            let row_point = &claim_point[claim_point.len() - num_vars..];
            public.push(emit_eq_eval(b, row_point, at));
        }
        let values = weave(shape.kinds, &proof.factor_values[t], &public);
        let combined = emit_combine(b, shape.ir, &betas[..shape.ir.num_roots()], &values);
        let zerocheck = b.emul(values[weight_r], combined);
        let mut value = zerocheck;
        for (k, (first, pair)) in present.iter().enumerate() {
            let (claim_point, _, _) = pair.claim;
            let bus = emit_claim_statements(
                b,
                pair.bus,
                &BusInputs {
                    claim_point,
                    num_row_vars: num_vars,
                    z: pair.z,
                    alpha_powers: pair.alpha_powers,
                    values: &values,
                    weight: weight_z + k,
                },
            );
            let (ln, ld) = pair_powers(*first);
            value = b.emul_add(ln, bus.numerator, value);
            value = b.emul_add(ld, bus.denominator, value);
        }
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
    tables
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
/// shapes, the plan and the ladder alone — every term by the emitter's own
/// shape:
///
/// - G, per bin: two absorbs per table; per step over layer `i` with `A`
///   active trees sending `H` halves at degree `d` (the host's
///   `ladder_steps`), a draw, `i` rounds of `d` absorbs and a draw, `4H`
///   absorbs and a draw (two when a tree descends two layers); then
///   `challenge_powers_rows(2A)`, `2A − 1` rows for the claim, `i` degree-`d`
///   rounds, `eq` over `i`, the relation (5 rows a one-layer tree, 11 a
///   two-layer one, less one), one `Mul` by `eq`, the assert, and the next
///   claims (4 rows a one-layer tree, 12 a two-layer one);
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
    ladder: GkrLadder,
    entry: SpongeEntry,
) -> TableCost {
    let pairs: Vec<CostPairs<'_>> = shapes
        .iter()
        .map(|shape| CostPairs {
            global: Some(shape.bus),
            local: None,
        })
        .collect();
    let mut leg = Cost::default();
    let mut schedule = SpongeSchedule::new(entry);
    let mut degrees: Vec<usize> = Vec::new();
    ladders_cost(&mut leg, &mut schedule, &mut degrees, plan, ladder);
    argue_rest_cost(
        &mut leg,
        &mut schedule,
        &mut degrees,
        shapes,
        plan,
        &pairs,
        false,
    );
    finish_cost(leg, schedule, degrees)
}

/// [`batched_argue_cost`] for a group under local lookups
/// ([`emit_argue_rest`] with `local`): G over the global class's plan
/// (`global`), then C and Rd over every table of `plan` with each table's
/// pairs. The local ladders run on the group's own sponge: their cost is
/// [`local_argue_cost`]'s.
pub(crate) fn batched_argue_cost_classed(
    shapes: &[TableShape<'_>],
    plan: &ArguePlan,
    global: &ArguePlan,
    pairs: &[CostPairs<'_>],
    ladder: GkrLadder,
    entry: SpongeEntry,
) -> TableCost {
    let mut leg = Cost::default();
    let mut schedule = SpongeSchedule::new(entry);
    let mut degrees: Vec<usize> = Vec::new();
    ladders_cost(&mut leg, &mut schedule, &mut degrees, global, ladder);
    argue_rest_cost(
        &mut leg,
        &mut schedule,
        &mut degrees,
        shapes,
        plan,
        pairs,
        true,
    );
    finish_cost(leg, schedule, degrees)
}

/// A table's bus pairs as the cost reads them: its interactions in each
/// class it argues in.
#[derive(Clone, Copy)]
pub(crate) struct CostPairs<'a> {
    pub(crate) global: Option<
        &'a [stark::multilinear_logup::InteractionShape<
            crate::tables::types::GoldilocksExtension,
        >],
    >,
    pub(crate) local: Option<
        &'a [stark::multilinear_logup::InteractionShape<
            crate::tables::types::GoldilocksExtension,
        >],
    >,
}

impl<'a> CostPairs<'a> {
    fn count(&self) -> usize {
        usize::from(self.global.is_some()) + usize::from(self.local.is_some())
    }

    fn buses(
        &self,
    ) -> impl Iterator<
        Item = &'a [stark::multilinear_logup::InteractionShape<
            crate::tables::types::GoldilocksExtension,
        >],
    > {
        self.global.into_iter().chain(self.local)
    }
}

/// Phase G's cost over one class's `plan` ([`emit_ladders`]).
fn ladders_cost(
    leg: &mut Cost,
    schedule: &mut SpongeSchedule,
    degrees: &mut Vec<usize>,
    plan: &ArguePlan,
    ladder: GkrLadder,
) {
    let one = FEE::one();
    let zero = FEE::zero();
    for bin in &plan.bins {
        let heights = tree_heights(plan, bin);
        absorb_ext(leg, schedule, 2 * heights.len());
        for step in ladder_steps(&heights, ladder) {
            let LadderStep {
                parent: i,
                degree,
                ref relations,
            } = step;
            let active = relations.len();
            draw_ext(leg, schedule, 1);
            for _ in 0..i {
                absorb_ext(leg, schedule, degree);
                draw_ext(leg, schedule, 1);
            }
            absorb_ext(leg, schedule, 4 * step.halves());
            draw_ext(
                leg,
                schedule,
                1 + usize::from(step.draws_second_challenge()),
            );

            leg.constant(one);
            leg.ops(challenge_powers_rows(2 * active));
            leg.ops(2 * active - 1);
            leg.ops(i * sumcheck_round_rows(degree));
            if i > 0 {
                degrees.push(degree);
            }
            leg.ops(eq_eval_rows_again(i));
            leg.ops(relation_rows(relations));
            leg.op();
            leg.ops(ASSERT_ROWS);
            leg.constant(zero);
            leg.ops(descend_rows(relations));
        }
        // A cut table's `ℓ` draws, its eq table over them, the two sums and
        // the padding chunks' weights, and its outputs added as fractions.
        for &t in bin {
            let chunks = &plan.chunks[t];
            if chunks.count == 1 {
                continue;
            }
            let ell = chunks.combine_vars();
            draw_ext(leg, schedule, ell);
            eq_evals_cost(ell, leg);
            leg.ops(2 * chunks.count + ((1usize << ell) - chunks.count));
            leg.ops(3 * (chunks.count - 1));
        }
    }
}

/// Phases C and Rd's cost ([`emit_argue_rest`]).
fn argue_rest_cost(
    leg: &mut Cost,
    schedule: &mut SpongeSchedule,
    degrees: &mut Vec<usize>,
    shapes: &[TableShape<'_>],
    plan: &ArguePlan,
    pairs: &[CostPairs<'_>],
    local: bool,
) {
    let one = FEE::one();
    let zero = FEE::zero();
    // ── C ──
    let n = shapes.len();
    let lanes: usize = pairs.iter().map(|p| p.count()).sum();
    draw_ext(leg, schedule, plan.num_vars);
    absorb_ext(leg, schedule, 2 * lanes);
    draw_ext(leg, schedule, 1);
    for _ in 0..plan.num_vars {
        absorb_ext(leg, schedule, plan.degree);
        draw_ext(leg, schedule, 1);
    }
    // λ², λ³, and under local lookups λ⁴ and λ⁵.
    leg.ops(if local { 4 } else { 2 });
    leg.constant(one);
    leg.ops(challenge_powers_rows(n));
    leg.ops(2 * lanes + n - 1);
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
            selector_cost(*selector, num_vars, leg);
        }
        leg.ops((1 + pairs[t].count()) * eq_eval_rows_again(num_vars));
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
        for bus in pairs[t].buses() {
            leg.merge(&claim_statements_cost(bus, num_vars));
            leg.ops(2);
        }
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
    absorb_ext(leg, schedule, factor_values);

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
}

/// The Newton constants of every sumcheck degree and the sponge's
/// capacities, added to a leg's cost.
fn finish_cost(mut leg: Cost, schedule: SpongeSchedule, degrees: Vec<usize>) -> TableCost {
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
pub(crate) fn batched_words(
    shapes: &[TableShape<'_>],
    plan: &ArguePlan,
    ladder: GkrLadder,
) -> usize {
    batched_words_classed(shapes, plan, plan, ladder)
}

/// [`batched_words`] for a group whose global class argues its ladders under
/// `ladders` (the class's plan; `plan` is over every table).
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn batched_words_classed(
    shapes: &[TableShape<'_>],
    plan: &ArguePlan,
    ladders: &ArguePlan,
    ladder: GkrLadder,
) -> usize {
    assert_eq!(
        plan.shapes.len(),
        shapes.len(),
        "the plan is over these tables"
    );
    let mut words = local_words(ladders, ladder);
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
    ladder: GkrLadder,
) -> BatchedStore {
    hint_batched_classed(b, arena, at, shapes, plan, plan, ladder)
}

/// [`hint_batched_shaped`] for a group whose global class argues its ladders
/// under `ladders` (the class's plan; `plan` is over every table).
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn hint_batched_classed(
    b: &mut LfmBuilder,
    arena: super::instr::ArenaId,
    at: &mut u32,
    shapes: &[TableShape<'_>],
    plan: &ArguePlan,
    ladders: &ArguePlan,
    ladder: GkrLadder,
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
    let bus_outputs = (0..ladders.trees())
        .map(|_| {
            let pq = take(b, 2);
            (pq[0], pq[1])
        })
        .collect();
    let gkr = ladders
        .bins
        .iter()
        .map(|bin| {
            ladder_steps(&tree_heights(ladders, bin), ladder)
                .iter()
                .map(|step| LadderStepWires {
                    sumcheck: (0..step.parent).map(|_| take(b, step.degree)).collect(),
                    halves: (0..step.halves())
                        .map(|_| {
                            let h = take(b, 4);
                            [h[0], h[1], h[2], h[3]]
                        })
                        .collect(),
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

/// One LogUp class of a group's tables, as the leg reads it — the host's
/// `ClassTables::split`: the tables that argue in it (group order), each
/// one's interactions in the class (its AIR's order), and the class's plan
/// (`argue_plan_packed` over the members' class statements, the host's own).
pub(crate) struct ClassShapes {
    pub(crate) members: Vec<usize>,
    pub(crate) buses: Vec<
        Vec<stark::multilinear_logup::InteractionShape<crate::tables::types::GoldilocksExtension>>,
    >,
    pub(crate) plan: ArguePlan,
}

impl ClassShapes {
    /// Where table `t` sits among the members, when it argues in the class.
    pub(crate) fn member(&self, t: usize) -> Option<usize> {
        self.members.iter().position(|&m| m == t)
    }
}

/// A group's local argue's wires — every local tree's output, then one
/// ladder a bin of the local plan — hinted with every count from the plan,
/// never from a proof.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) struct LocalStore {
    pub(crate) bus_outputs: Vec<(Ext, Ext)>,
    pub(crate) gkr: Vec<Vec<LadderStepWires>>,
}

/// Words a local argue over `plan` takes, in [`hint_local_shaped`]'s order:
/// what [`push_local_words`] must write.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn local_words(plan: &ArguePlan, ladder: GkrLadder) -> usize {
    let mut words = 2 * plan.trees();
    for bin in &plan.bins {
        for step in ladder_steps(&tree_heights(plan, bin), ladder) {
            words += step.parent * step.degree + 4 * step.halves();
        }
    }
    words
}

/// Hints a local argue's wires from `arena` at `at`, in [`push_local_words`]'
/// order: the local trees' outputs, then each bin's ladder (each step's
/// rounds, then its active trees' halves).
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn hint_local_shaped(
    b: &mut LfmBuilder,
    arena: super::instr::ArenaId,
    at: &mut u32,
    plan: &ArguePlan,
    ladder: GkrLadder,
) -> LocalStore {
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
            ladder_steps(&tree_heights(plan, bin), ladder)
                .iter()
                .map(|step| LadderStepWires {
                    sumcheck: (0..step.parent).map(|_| take(b, step.degree)).collect(),
                    halves: (0..step.halves())
                        .map(|_| {
                            let h = take(b, 4);
                            [h[0], h[1], h[2], h[3]]
                        })
                        .collect(),
                })
                .collect()
        })
        .collect();
    LocalStore { bus_outputs, gkr }
}

/// A local argue's words, in [`hint_local_shaped`]'s order.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn push_local_words(
    words: &mut Vec<super::word::LfmWord>,
    argue: &stark::multilinear_table::batched::LocalArgue<
        crate::tables::types::GoldilocksExtension,
    >,
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
}

/// The local ladders' cost ([`emit_ladders`] over the local `plan`), on the
/// group's own sponge `schedule` — continued from its commit-time transcript
/// by the caller — and the schedule left after them.
pub(crate) fn local_ladders_cost(
    plan: &ArguePlan,
    ladder: GkrLadder,
    mut schedule: SpongeSchedule,
) -> TableCost {
    let mut leg = Cost::default();
    let mut degrees: Vec<usize> = Vec::new();
    ladders_cost(&mut leg, &mut schedule, &mut degrees, plan, ladder);
    finish_cost(leg, schedule, degrees)
}

/// What a group's local argue hands its C phase, as wires: per table, its
/// local claim (`None` off the local class) and its local interactions, and
/// the group's `(z_g, α_g^p)`.
pub(crate) struct LocalPairs<'a> {
    pub(crate) claims: &'a [Option<(Vec<Ext>, Ext, Ext)>],
    pub(crate) class: &'a ClassShapes,
    pub(crate) z: Ext,
    pub(crate) alpha_powers: &'a [Ext],
}

/// ★ A block group's tables under local lookups: G over the global class on
/// the group's fork (`global`), then C and Rd over every table with its
/// local pair (`local`, from the group's own ladders) — the host's
/// `verify_argue_classed` — and each table's preprocessed leg, as
/// [`emit_batched_walk`]. The walk's outputs are the global class's members'
/// (the block's bus), in member order.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_batched_walk_classed(
    b: &mut LfmBuilder,
    transcript: &mut WhirTranscript,
    store: &BatchedStore,
    shapes: &[TableShape<'_>],
    plan: &ArguePlan,
    global: &ClassShapes,
    local: &LocalPairs<'_>,
    ladder: GkrLadder,
    plans: &[super::whir_epoch::PreprocessedPlan<'_>],
    slots: &[&[usize]],
    z: Ext,
    alpha_powers: &[Ext],
    beta: Ext,
) -> super::whir_epoch::TableWalk {
    let n = shapes.len();
    assert_eq!(n, plans.len(), "one preprocessed plan per table");
    assert_eq!(n, slots.len(), "one slot map per table");
    assert_eq!(local.claims.len(), n, "a local claim slot per table");
    let (claims, outputs) = emit_ladders(
        b,
        transcript,
        &store.bus_outputs,
        &store.gkr,
        &global.plan,
        ladder,
    );
    let pairs: Vec<TablePairWires<'_>> = (0..n)
        .map(|t| TablePairWires {
            global: global.member(t).map(|k| PairWires {
                claim: &claims[k],
                bus: &global.buses[k],
                z,
                alpha_powers,
            }),
            local: local.claims[t].as_ref().map(|claim| PairWires {
                claim,
                bus: &local.class.buses[local
                    .class
                    .member(t)
                    .expect("a table with a local claim is a local member")],
                z: local.z,
                alpha_powers: local.alpha_powers,
            }),
        })
        .collect();
    let reduces = store.reduces();
    let tables = emit_argue_rest(
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
        &pairs,
        true,
        beta,
    );
    let mut walk = super::whir_epoch::TableWalk {
        outputs,
        points: Vec::with_capacity(n),
        widths: Vec::with_capacity(n),
        values: Vec::new(),
    };
    for (t, (point, columns)) in tables.into_iter().enumerate() {
        super::whir_epoch::emit_preprocessed_leg_at(
            b, &plans[t], slots[t], &shapes[t], &point, &columns,
        );
        walk.widths.push(columns.len());
        walk.values.extend(columns.iter().copied());
        walk.points.push(point);
    }
    walk
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
    ladder: GkrLadder,
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
        ladder,
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
