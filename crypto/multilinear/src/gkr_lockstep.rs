//! LogUp-GKR over several fraction trees at once, in **lockstep** (D-BATCH §2.2,
//! phase G): one sumcheck per step for every tree still descending.
//!
//! The trees stay separate and keep their own heights. What is shared is the
//! ladder: they are aligned at the OUTPUT, so step `i` proves layer `i + 1` of
//! every tree with more than `i` input variables, all against the same point.
//! Every challenge is shared, so the point every active tree sits at is the
//! same one, one `eq` table serves them all, and one sumcheck batches them with
//! a fresh `μ`:
//!
//! ```text
//! claim_i = Σ_a μ^{2a}·p̄_a + μ^{2a+1}·q̄_a
//!         = Σ_x eq(P, x) · Σ_a [ μ^{2a}·(p_lo q_hi + p_hi q_lo) + μ^{2a+1}·q_lo q_hi ]_a(x)
//! ```
//!
//! With one tree this is [`gkr::prove`](crate::gkr::prove) step for step, `μ`
//! in the place of its `λ`. A tree leaves the ladder at the step that reaches
//! its input layer, with its input claim at the point the ladder is at then —
//! which is what the caller settles against its trace.
//!
//! Soundness is SWIRL Thm 4.1.1 per step (every polynomial has `i` variables)
//! plus the GKR line through `lo`/`hi` at `c`: a false claim on any active tree
//! dooms the step, and the batching costs `(2|A_i| − 1)/|F|`.
//!
//! The bus outputs are NOT absorbed here: the caller absorbs them where the
//! format puts them (per bin, before its ladder).
//!
//! # Two layers a step ([`GkrLadder::Double`])
//!
//! Under the two-layer ladder a step proves layer `ℓ` against its
//! grandchildren, layer `ℓ + 2`, and the layer between is never sent. Write
//! layer `ℓ + 2` as `(b₁, b₂, x)`, `b₁` its first variable, and its four
//! quarters `p_{b₁b₂}(x)`, `q_{b₁b₂}(x)`. For each `b₂` the tree sends one
//! [`TreeHalves`] `h_{b₂} = (p_{0b₂}, p_{1b₂}, q_{0b₂}, q_{1b₂})` at the
//! sumcheck's point `ρ`: today's shape, `lo`/`hi` meaning `b₁`. Its fraction
//! `(N_{b₂}, D_{b₂}) = (p_lo q_hi + p_hi q_lo, q_lo q_hi)` is layer `ℓ + 1`
//! at `(b₂, ρ)`, and layer `ℓ` is their fold:
//!
//! ```text
//! claim = Σ_x eq(P, x) · Σ_a [ μ^{2a}·(N₀D₁ + N₁D₀) + μ^{2a+1}·D₀D₁ ]_a(x)   (degree 5)
//! ```
//!
//! After the halves, `c₁` then `c₂` are drawn, and the tree's next claim is
//! layer `ℓ + 2` at `(c₁, c₂, ρ)`: the line through each half's `lo`/`hi` at
//! `c₁`, then the line through `h₀`'s and `h₁`'s at `c₂`. A tree of odd
//! height takes today's relation in the step that reaches its input layer,
//! and leaves at `(c₁, ρ)`. [`ladder_steps`] derives every step's relations,
//! degree and counts from the heights and the format alone, on both sides.
//!
//! Soundness is the same argument with the layer map `fold ∘ fold`: a round
//! errs with probability `5/|F|`, the plane through the four quarters (degree
//! 2 in `(c₁, c₂)`) with `2/|F|`, the batching as above. Two independent
//! challenges are load-bearing: on the diagonal `c₂ = c₁` the plane cannot
//! tell `p_{01} + δ, p_{10} − δ` from the truth.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{element::FieldElement, traits::IsField};

pub use crate::whir_chain::GkrLadder;
use crate::{
    Error, challenge_powers,
    eq::{eq_eval, eq_mle},
    gkr::{FractionLayer, GkrClaim},
    mle::Mle,
    poly::SumcheckPolynomial,
    sumcheck::{self, SumcheckProof},
};

/// One tree's four restricted values at the end of a step.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct TreeHalves<F: IsField> {
    pub p_lo: FieldElement<F>,
    pub p_hi: FieldElement<F>,
    pub q_lo: FieldElement<F>,
    pub q_hi: FieldElement<F>,
}

/// One step of the ladder: the shared sumcheck, and the halves of every tree
/// active in it, in the caller's tree order.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct LockstepLayer<F: IsField> {
    pub sumcheck: SumcheckProof<F>,
    pub halves: Vec<TreeHalves<F>>,
}

/// The whole ladder, output end first: `max_T k_T` steps.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct LockstepProof<F: IsField> {
    pub layers: Vec<LockstepLayer<F>>,
}

/// Which trees step `i` proves: those with more than `i` input variables, in
/// the caller's order. Both sides call this.
pub fn active(input_vars: &[usize], step: usize) -> Vec<usize> {
    (0..input_vars.len())
        .filter(|&t| input_vars[t] > step)
        .collect()
}

/// How one tree descends in a ladder step ([`ladder_steps`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relation {
    /// From the step's layer `ℓ` to `ℓ + 1`: today's relation, one
    /// [`TreeHalves`], degree 3.
    Single,
    /// From `ℓ` to `ℓ + 2`: two [`TreeHalves`] (`h₀`, `h₁`), degree 5.
    Double,
}

impl Relation {
    /// The tree layers it descends.
    pub const fn layers(self) -> usize {
        match self {
            Self::Single => 1,
            Self::Double => 2,
        }
    }

    /// The [`TreeHalves`] it sends at the step's end.
    pub const fn halves(self) -> usize {
        self.layers()
    }

    /// Its round polynomial's degree, `eq(P, ·)` included.
    pub const fn degree(self) -> usize {
        match self {
            Self::Single => 3,
            Self::Double => 5,
        }
    }
}

/// One step of a ladder, as both sides derive it from the trees' heights and
/// the format: never from a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LadderStep {
    /// `ℓ`: the layer every active tree's claim is on when the step starts,
    /// so the step's sumcheck rounds and its point's length.
    pub parent: usize,
    /// Values a round: the highest degree among the step's relations.
    pub degree: usize,
    /// The trees the step proves, in the caller's order, each with how it
    /// descends.
    pub relations: Vec<(usize, Relation)>,
}

impl LadderStep {
    /// Whether the step draws `c₂`: when any of its trees descends two
    /// layers.
    pub fn draws_second_challenge(&self) -> bool {
        self.relations
            .iter()
            .any(|&(_, relation)| relation == Relation::Double)
    }

    /// The [`TreeHalves`] the step sends, every tree's in tree order.
    pub fn halves(&self) -> usize {
        self.relations
            .iter()
            .map(|&(_, relation)| relation.halves())
            .sum()
    }
}

/// ★ THE LADDER'S SCHEDULE, the one spelling every side reads: the prover,
/// the host verifier, and every reader that lays a ladder out.
///
/// [`GkrLadder::Single`]: a step at each `ℓ = 0, 1, …, H − 1` (`H` the
/// tallest tree's input variables), every tree with more than `ℓ` input
/// variables descending one layer at degree 3 — today's ladder.
///
/// [`GkrLadder::Double`]: a step at each `ℓ = 0, 2, 4, … < H`. A tree with
/// `k ≥ ℓ + 2` input variables descends two layers; one with `k = ℓ + 1`
/// (odd height, at the step that reaches its input layer) descends one. The
/// step's degree is 5 when any tree descends two layers, else 3.
pub fn ladder_steps(input_vars: &[usize], form: GkrLadder) -> Vec<LadderStep> {
    let height = input_vars.iter().copied().max().unwrap_or(0);
    let stride = match form {
        GkrLadder::Single => 1,
        GkrLadder::Double => 2,
    };
    (0..height)
        .step_by(stride)
        .map(|parent| {
            let relations: Vec<(usize, Relation)> = active(input_vars, parent)
                .into_iter()
                .map(|t| {
                    let relation = match form {
                        GkrLadder::Double if input_vars[t] >= parent + 2 => Relation::Double,
                        _ => Relation::Single,
                    };
                    (t, relation)
                })
                .collect();
            let degree = relations
                .iter()
                .map(|&(_, relation)| relation.degree())
                .max()
                .unwrap_or(Relation::Single.degree());
            LadderStep {
                parent,
                degree,
                relations,
            }
        })
        .collect()
}

/// `Σ_a w_{2a}·(p_lo q_hi + p_hi q_lo) + w_{2a+1}·q_lo q_hi` over the trees'
/// values, laid out four a tree.
fn relation<F: IsField>(
    weights: &[FieldElement<F>],
    halves: &[FieldElement<F>],
) -> FieldElement<F> {
    halves
        .chunks(4)
        .zip(weights.chunks(2))
        .fold(FieldElement::zero(), |acc, (v, w)| {
            let numerator = &v[0] * &v[3] + &v[1] * &v[2];
            let denominator = &v[2] * &v[3];
            acc + &w[0] * numerator + &w[1] * denominator
        })
}

/// One half's fraction, `(p_lo q_hi + p_hi q_lo, q_lo q_hi)`, its four
/// values laid out `p_lo, p_hi, q_lo, q_hi`.
fn fraction<F: IsField>(v: &[FieldElement<F>]) -> (FieldElement<F>, FieldElement<F>) {
    (&v[0] * &v[3] + &v[1] * &v[2], &v[2] * &v[3])
}

/// The two-layer relation's fraction over two halves' eight values
/// (`h₀` then `h₁`): `(N₀D₁ + N₁D₀, D₀D₁)`, the fold of the halves'
/// fractions.
fn double_fraction<F: IsField>(v: &[FieldElement<F>]) -> (FieldElement<F>, FieldElement<F>) {
    let (n0, d0) = fraction(&v[..4]);
    let (n1, d1) = fraction(&v[4..8]);
    (&n0 * &d1 + &n1 * &d0, d0 * d1)
}

/// A step's residual without its `eq`: `Σ_a w_{2a}·N_a + w_{2a+1}·Q_a` over
/// its trees in order, `(N_a, Q_a)` each tree's fraction under its relation,
/// read off `values` (four a single relation, eight a double one).
fn step_relation<F: IsField>(
    relations: &[(usize, Relation)],
    weights: &[FieldElement<F>],
    values: &[FieldElement<F>],
) -> FieldElement<F> {
    let mut at = 0;
    let mut acc = FieldElement::zero();
    for (&(_, relation), w) in relations.iter().zip(weights.chunks(2)) {
        let width = 4 * relation.halves();
        let (numerator, denominator) = match relation {
            Relation::Single => fraction(&values[at..at + width]),
            Relation::Double => double_fraction(&values[at..at + width]),
        };
        acc = acc + &w[0] * numerator + &w[1] * denominator;
        at += width;
    }
    acc
}

/// The step's polynomial: `eq(P, ·)` and four halves per active tree.
struct LockstepRelation<F: IsField> {
    /// `[eq, (p_lo, p_hi, q_lo, q_hi) per tree]`.
    polys: Vec<Mle<F>>,
    /// `μ` powers, two per tree.
    weights: Vec<FieldElement<F>>,
}

impl<F: IsField + 'static> SumcheckPolynomial<F> for LockstepRelation<F> {
    fn num_vars(&self) -> usize {
        self.polys[0].num_vars()
    }

    fn degree(&self) -> usize {
        3
    }

    fn polys(&self) -> &[Mle<F>] {
        &self.polys
    }

    fn combine(&self, v: &[FieldElement<F>]) -> FieldElement<F> {
        &v[0] * relation(&self.weights, &v[1..])
    }

    fn fix_first_variable(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        for p in &mut self.polys {
            p.fix_first_variable_in_place(r)?;
        }
        Ok(())
    }
}

/// One tree's two-layer relation: `eq(P, ·)·(w_p·N + w_q·Q)` with
/// `(N, Q) = (N₀D₁ + N₁D₀, D₀D₁)` over its two halves' factors.
struct DoubleRelation<F: IsField> {
    /// `[eq, h₀ (p_lo, p_hi, q_lo, q_hi), h₁ (p_lo, p_hi, q_lo, q_hi)]`.
    polys: Vec<Mle<F>>,
    /// `[w_p, w_q]`.
    weights: Vec<FieldElement<F>>,
}

impl<F: IsField + 'static> SumcheckPolynomial<F> for DoubleRelation<F> {
    fn num_vars(&self) -> usize {
        self.polys[0].num_vars()
    }

    fn degree(&self) -> usize {
        Relation::Double.degree()
    }

    fn polys(&self) -> &[Mle<F>] {
        &self.polys
    }

    fn combine(&self, v: &[FieldElement<F>]) -> FieldElement<F> {
        let (numerator, denominator) = double_fraction(&v[1..9]);
        &v[0] * (&self.weights[0] * numerator + &self.weights[1] * denominator)
    }

    fn fix_first_variable(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        for p in &mut self.polys {
            p.fix_first_variable_in_place(r)?;
        }
        Ok(())
    }
}

/// `(1 − c)·lo + c·hi`.
fn line<F: IsField>(
    lo: &FieldElement<F>,
    hi: &FieldElement<F>,
    c: &FieldElement<F>,
) -> FieldElement<F> {
    lo + c * &(hi - lo)
}

/// A two-layer relation's next claim, layer `ℓ + 2` at `(c₁, c₂, ρ)`, from
/// its halves at `ρ`: the line through each half's `lo`/`hi` at `c₁` (the
/// first variable), then the line through `h₀`'s and `h₁`'s at `c₂` (the
/// second).
fn plane<F: IsField>(
    h0: &TreeHalves<F>,
    h1: &TreeHalves<F>,
    c1: &FieldElement<F>,
    c2: &FieldElement<F>,
) -> (FieldElement<F>, FieldElement<F>) {
    (
        line(
            &line(&h0.p_lo, &h0.p_hi, c1),
            &line(&h1.p_lo, &h1.p_hi, c1),
            c2,
        ),
        line(
            &line(&h0.q_lo, &h0.q_hi, c1),
            &line(&h1.q_lo, &h1.q_hi, c1),
            c2,
        ),
    )
}

/// The step's point challenges, drawn once its halves are absorbed: `c₁`,
/// then `c₂` when a tree of the step descends two layers. Both sides draw
/// them here.
fn draw_point_challenges<F, T>(
    step: &LadderStep,
    transcript: &mut T,
) -> (FieldElement<F>, Option<FieldElement<F>>)
where
    F: IsField,
    T: IsTranscript<F>,
{
    let c1: FieldElement<F> = transcript.sample_field_element();
    let c2 = step
        .draws_second_challenge()
        .then(|| transcript.sample_field_element());
    (c1, c2)
}

/// Moves every tree of `step` to its claim on the layer it descended to, from
/// its halves (in tree order, as many as its relation sends) and the step's
/// challenges; records the input claim of each tree that reached its input
/// layer. Returns the point every tree still descending is at: `(c₁, c₂, ρ)`,
/// or `(c₁, ρ)` in a step without a two-layer relation. Both sides move here.
#[allow(clippy::too_many_arguments)]
fn descend<F: IsField>(
    step: &LadderStep,
    halves: &[TreeHalves<F>],
    c1: &FieldElement<F>,
    c2: Option<&FieldElement<F>>,
    rho: &[FieldElement<F>],
    input_vars: &[usize],
    claims: &mut [(FieldElement<F>, FieldElement<F>)],
    inputs: &mut [Option<GkrClaim<F>>],
) -> Result<Vec<FieldElement<F>>, Error> {
    let shape = || Error::ArgueShapeMismatch {
        part: "ladder step",
    };
    if halves.len() != step.halves() {
        return Err(shape());
    }
    let single_point = || -> Vec<FieldElement<F>> {
        std::iter::once(c1.clone())
            .chain(rho.iter().cloned())
            .collect()
    };
    let double_point = |c2: &FieldElement<F>| -> Vec<FieldElement<F>> {
        [c1.clone(), c2.clone()]
            .into_iter()
            .chain(rho.iter().cloned())
            .collect()
    };
    let mut at = 0;
    for &(t, relation) in &step.relations {
        let claim = match relation {
            Relation::Single => {
                let h = &halves[at];
                (line(&h.p_lo, &h.p_hi, c1), line(&h.q_lo, &h.q_hi, c1))
            }
            Relation::Double => plane(&halves[at], &halves[at + 1], c1, c2.ok_or_else(shape)?),
        };
        at += relation.halves();
        if input_vars[t] == step.parent + relation.layers() {
            let point = match relation {
                Relation::Single => single_point(),
                Relation::Double => double_point(c2.ok_or_else(shape)?),
            };
            inputs[t] = Some(GkrClaim {
                point,
                p: claim.0.clone(),
                q: claim.1.clone(),
            });
        }
        claims[t] = claim;
    }
    Ok(match c2 {
        Some(c2) => double_point(c2),
        None => single_point(),
    })
}

/// One tree's side of one ladder step: its relation
/// `eq(P, ·)·(w_p·(p_lo q_hi + p_hi q_lo) + w_q·q_lo q_hi)` over layer `i + 1`
/// ([`Relation::Single`]), or the two-layer one over layer `i + 2`
/// ([`Relation::Double`]), stepped by the shared challenges.
pub trait TreeStep<F: IsField> {
    /// This round's polynomial at `1, ..=d`, `d` its relation's degree
    /// ([`Relation::degree`]).
    fn round(&mut self) -> Result<Vec<FieldElement<F>>, Error>;
    /// Binds this round's variable to `r`.
    fn bind(&mut self, r: &FieldElement<F>) -> Result<(), Error>;
    /// The halves once every variable is bound: one under a single relation,
    /// `h₀` then `h₁` under a double one.
    fn halves(&self) -> Result<Vec<TreeHalves<F>>, Error>;
    /// Queues this round's work without waiting for it, where there is a
    /// device to queue on: the ladder queues every active tree's round before
    /// it reads any back, so a round costs one wait, not one a tree.
    fn prefetch(&mut self) -> Result<(), Error> {
        Ok(())
    }
    /// The cube this step still has on a device; zero on the host.
    fn device_cube(&self) -> usize {
        0
    }
    /// Takes the step's factors to the host, from here on stepped there.
    fn to_host(&mut self) -> Result<(), Error> {
        Ok(())
    }
    /// The host relation this step is stepped on now, or `None` while its
    /// rounds run on a device: a ladder with many such steps may run their
    /// rounds and binds side by side ([`host_rounds_parallel`]).
    fn host_step(&mut self) -> Option<&mut HostStep<F>> {
        None
    }
}

/// A tree the ladder descends: where its layers are is its own business — on
/// the host, or on a device that steps them where they lie — and the ladder
/// cannot tell, which is what makes a device ladder the host's bytes.
pub trait LadderTree<F: IsField> {
    /// `k`: its input layer's variables.
    fn input_vars(&self) -> usize;
    /// The output fraction.
    fn output(&self) -> (FieldElement<F>, FieldElement<F>);
    /// Layer `layer`'s step (`layer ≥ 1`), at `point` (`layer − 1`
    /// coordinates), weighted `[w_p, w_q]`, from the tree's claim `(p, q)` on
    /// the layer above.
    fn step(
        &self,
        layer: usize,
        point: &[FieldElement<F>],
        weights: [FieldElement<F>; 2],
        claim: &(FieldElement<F>, FieldElement<F>),
    ) -> Result<Box<dyn TreeStep<F> + '_>, Error>;
    /// The two-layer step whose factors are layer `layer`'s quarters
    /// (`layer ≥ 2`), at `point` (`layer − 2` coordinates), weighted
    /// `[w_p, w_q]`, from the tree's claim `(p, q)` on layer `layer − 2`
    /// ([`Relation::Double`]).
    fn step_double(
        &self,
        layer: usize,
        point: &[FieldElement<F>],
        weights: [FieldElement<F>; 2],
        claim: &(FieldElement<F>, FieldElement<F>),
    ) -> Result<Box<dyn TreeStep<F> + '_>, Error>;
}

/// The host's relation for one tree's step.
enum HostRelation<F: IsField> {
    Single(LockstepRelation<F>),
    Double(DoubleRelation<F>),
}

/// The step over a layer the host holds: one tree's relation.
pub struct HostStep<F: IsField>(HostRelation<F>);

impl<F: IsField + 'static> HostStep<F>
where
    FieldElement<F>: Send + Sync,
{
    /// Today's step over five factors, `eq` first.
    fn single(polys: Vec<Mle<F>>, weights: &[FieldElement<F>; 2]) -> Self {
        Self(HostRelation::Single(LockstepRelation {
            polys,
            weights: weights.to_vec(),
        }))
    }

    /// The two-layer step over nine factors, `eq` first, then `h₀`'s and
    /// `h₁`'s `p_lo, p_hi, q_lo, q_hi`.
    fn double(polys: Vec<Mle<F>>, weights: &[FieldElement<F>; 2]) -> Self {
        Self(HostRelation::Double(DoubleRelation {
            polys,
            weights: weights.to_vec(),
        }))
    }

    /// The two-layer step over `grandchild`, layer `ℓ + 2` at `point`
    /// (`ℓ` coordinates): its quarters `b = 2b₁ + b₂` at offset `b·2^ℓ`,
    /// laid out `h₀ = (q₀₀, q₁₀)`, `h₁ = (q₀₁, q₁₁)` for `p` and `q` alike.
    fn over_double(
        grandchild: &FractionLayer<F>,
        point: &[FieldElement<F>],
        weights: [FieldElement<F>; 2],
    ) -> Result<Self, Error> {
        if grandchild.num_vars() != point.len() + 2 {
            return Err(Error::VariableCountMismatch {
                expected: point.len() + 2,
                got: grandchild.num_vars(),
            });
        }
        let quarter = grandchild.p.len() / 4;
        let cells = |m: &Mle<F>, b: usize| -> Result<Mle<F>, Error> {
            Mle::new(m.evals()[b * quarter..(b + 1) * quarter].to_vec())
        };
        let mut polys = Vec::with_capacity(9);
        polys.push(eq_mle(point)?);
        for b2 in 0..2 {
            for m in [&grandchild.p, &grandchild.q] {
                polys.push(cells(m, b2)?);
                polys.push(cells(m, 2 + b2)?);
            }
        }
        Ok(Self(HostRelation::Double(DoubleRelation {
            polys,
            weights: weights.to_vec(),
        })))
    }

    fn over(
        next: &FractionLayer<F>,
        point: &[FieldElement<F>],
        weights: [FieldElement<F>; 2],
    ) -> Result<Self, Error> {
        let half = next.p.len() / 2;
        let mut polys = Vec::with_capacity(5);
        polys.push(eq_mle(point)?);
        for m in [&next.p, &next.q] {
            polys.push(Mle::new(m.evals()[..half].to_vec())?);
            polys.push(Mle::new(m.evals()[half..].to_vec())?);
        }
        Ok(Self::single(polys, &weights))
    }
}

impl<F: IsField + 'static> TreeStep<F> for HostStep<F>
where
    FieldElement<F>: Send + Sync,
{
    fn round(&mut self) -> Result<Vec<FieldElement<F>>, Error> {
        Ok(match &self.0 {
            HostRelation::Single(relation) => sumcheck::round_evaluations(relation, 3, false),
            HostRelation::Double(relation) => {
                sumcheck::round_evaluations(relation, Relation::Double.degree(), false)
            }
        })
    }

    fn bind(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        match &mut self.0 {
            HostRelation::Single(relation) => relation.fix_first_variable(r),
            HostRelation::Double(relation) => relation.fix_first_variable(r),
        }
    }

    fn host_step(&mut self) -> Option<&mut HostStep<F>> {
        Some(self)
    }

    fn halves(&self) -> Result<Vec<TreeHalves<F>>, Error> {
        let polys = match &self.0 {
            HostRelation::Single(relation) => &relation.polys,
            HostRelation::Double(relation) => &relation.polys,
        };
        let bound = |slot: usize| -> Result<FieldElement<F>, Error> {
            polys
                .get(slot)
                .and_then(Mle::as_constant)
                .cloned()
                .ok_or(Error::NoVariablesLeft)
        };
        (0..(polys.len() - 1) / 4)
            .map(|h| {
                Ok(TreeHalves {
                    p_lo: bound(1 + 4 * h)?,
                    p_hi: bound(2 + 4 * h)?,
                    q_lo: bound(3 + 4 * h)?,
                    q_hi: bound(4 + 4 * h)?,
                })
            })
            .collect()
    }
}

/// The step over `next`, a layer the host holds: what [`HostLayers`] steps
/// over, for a tree whose layer is written here only when it is needed (a
/// card tree whose input layer the card could not write again).
pub fn host_step<F: IsField + 'static>(
    next: &FractionLayer<F>,
    point: &[FieldElement<F>],
    weights: [FieldElement<F>; 2],
) -> Result<Box<dyn TreeStep<F>>, Error>
where
    FieldElement<F>: Send + Sync,
{
    Ok(Box::new(HostStep::over(next, point, weights)?))
}

/// The two-layer step over `grandchild`, a layer the host holds: what
/// [`host_step`] is to the one-layer step.
pub fn host_step_double<F: IsField + 'static>(
    grandchild: &FractionLayer<F>,
    point: &[FieldElement<F>],
    weights: [FieldElement<F>; 2],
) -> Result<Box<dyn TreeStep<F>>, Error>
where
    FieldElement<F>: Send + Sync,
{
    Ok(Box::new(HostStep::over_double(grandchild, point, weights)?))
}

/// ⛔ A departure a NEGATIVE test arms in one tree of a ladder ([`Faulted`]):
/// the prover departs at the tree's step over one layer (by default layer 2,
/// a tree's first two-layer step) and steps honestly after, from the claims
/// its departure leaves. Each is refused. Never armed in production.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepFault {
    /// Its two halves sent `h₁, h₀`: the step's residual is symmetric in
    /// them and passes; the plane is not.
    SwapHalves,
    /// Each half's `lo` and `hi` traded (`p_lo ↔ p_hi`, `q_lo ↔ q_hi`): the
    /// residual passes again; the lines do not.
    SwapLoHi,
    /// The step after it proved at its point with the first two coordinates
    /// traded, `(c₂, c₁, ρ)`: the plane read the other way round.
    SwapPlane,
}

/// `tree` with a [`StepFault`] armed (negative tests only).
#[doc(hidden)]
pub struct Faulted<'a, F: IsField> {
    pub tree: &'a dyn LadderTree<F>,
    pub fault: StepFault,
    /// The layer whose step departs: the step's factors are its cells.
    pub layer: usize,
    /// Set once the departing step is open, until the next one opens.
    departed: std::cell::Cell<bool>,
}

impl<'a, F: IsField> Faulted<'a, F> {
    /// `fault` at the tree's first two-layer step, the one over layer 2.
    pub fn new(tree: &'a dyn LadderTree<F>, fault: StepFault) -> Self {
        Self::at(tree, fault, 2)
    }

    /// `fault` at the tree's step over `layer`.
    pub fn at(tree: &'a dyn LadderTree<F>, fault: StepFault, layer: usize) -> Self {
        Self {
            tree,
            fault,
            layer,
            departed: std::cell::Cell::new(false),
        }
    }

    /// `point` with its first two coordinates traded, under
    /// [`StepFault::SwapPlane`] on the step after the departing one.
    fn point(&self, point: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
        let mut point = point.to_vec();
        if self.departed.replace(false) && self.fault == StepFault::SwapPlane && point.len() >= 2 {
            point.swap(0, 1);
        }
        point
    }

    /// The step over `layer`, departing there.
    fn depart<'s>(&self, layer: usize, step: Box<dyn TreeStep<F> + 's>) -> Box<dyn TreeStep<F> + 's>
    where
        F: 's,
    {
        if layer != self.layer {
            return step;
        }
        self.departed.set(true);
        match self.fault {
            StepFault::SwapPlane => step,
            fault => Box::new(Departed { inner: step, fault }),
        }
    }
}

/// A [`Faulted`] tree's departing step: its halves altered, its rounds the
/// honest step's.
struct Departed<'a, F: IsField> {
    inner: Box<dyn TreeStep<F> + 'a>,
    fault: StepFault,
}

impl<F: IsField> TreeStep<F> for Departed<'_, F> {
    fn round(&mut self) -> Result<Vec<FieldElement<F>>, Error> {
        self.inner.round()
    }

    fn bind(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        self.inner.bind(r)
    }

    fn halves(&self) -> Result<Vec<TreeHalves<F>>, Error> {
        let mut halves = self.inner.halves()?;
        match self.fault {
            StepFault::SwapHalves => halves.reverse(),
            StepFault::SwapLoHi => {
                for h in &mut halves {
                    std::mem::swap(&mut h.p_lo, &mut h.p_hi);
                    std::mem::swap(&mut h.q_lo, &mut h.q_hi);
                }
            }
            StepFault::SwapPlane => {}
        }
        Ok(halves)
    }

    fn prefetch(&mut self) -> Result<(), Error> {
        self.inner.prefetch()
    }

    fn device_cube(&self) -> usize {
        self.inner.device_cube()
    }

    fn to_host(&mut self) -> Result<(), Error> {
        self.inner.to_host()
    }

    fn host_step(&mut self) -> Option<&mut HostStep<F>> {
        self.inner.host_step()
    }
}

impl<F: IsField + 'static> LadderTree<F> for Faulted<'_, F> {
    fn input_vars(&self) -> usize {
        self.tree.input_vars()
    }

    fn output(&self) -> (FieldElement<F>, FieldElement<F>) {
        self.tree.output()
    }

    fn step(
        &self,
        layer: usize,
        point: &[FieldElement<F>],
        weights: [FieldElement<F>; 2],
        claim: &(FieldElement<F>, FieldElement<F>),
    ) -> Result<Box<dyn TreeStep<F> + '_>, Error> {
        let step = self.tree.step(layer, &self.point(point), weights, claim)?;
        Ok(self.depart(layer, step))
    }

    fn step_double(
        &self,
        layer: usize,
        point: &[FieldElement<F>],
        weights: [FieldElement<F>; 2],
        claim: &(FieldElement<F>, FieldElement<F>),
    ) -> Result<Box<dyn TreeStep<F> + '_>, Error> {
        let step = self
            .tree
            .step_double(layer, &self.point(point), weights, claim)?;
        Ok(self.depart(layer, step))
    }
}

/// A tree's layers held here, output first.
pub struct HostLayers<'a, F: IsField>(pub &'a [FractionLayer<F>]);

impl<F: IsField + 'static> LadderTree<F> for HostLayers<'_, F>
where
    FieldElement<F>: Send + Sync,
{
    fn input_vars(&self) -> usize {
        self.0.len().saturating_sub(1)
    }

    fn output(&self) -> (FieldElement<F>, FieldElement<F>) {
        (
            self.0[0].p.evals()[0].clone(),
            self.0[0].q.evals()[0].clone(),
        )
    }

    fn step(
        &self,
        layer: usize,
        point: &[FieldElement<F>],
        weights: [FieldElement<F>; 2],
        _claim: &(FieldElement<F>, FieldElement<F>),
    ) -> Result<Box<dyn TreeStep<F> + '_>, Error> {
        let next = self.0.get(layer).ok_or(Error::NoVariablesLeft)?;
        Ok(Box::new(HostStep::over(next, point, weights)?))
    }

    fn step_double(
        &self,
        layer: usize,
        point: &[FieldElement<F>],
        weights: [FieldElement<F>; 2],
        _claim: &(FieldElement<F>, FieldElement<F>),
    ) -> Result<Box<dyn TreeStep<F> + '_>, Error> {
        let grandchild = self.0.get(layer).ok_or(Error::NoVariablesLeft)?;
        Ok(Box::new(HostStep::over_double(grandchild, point, weights)?))
    }
}

/// The step over a layer a device holds: its rounds on the card while the cube
/// is worth a launch, then the host's over the factors it hands back — today's
/// crossover ([`crate::HOST_CUBE_DIRECT`]).
struct DeviceStep<F: IsField> {
    session: Option<crate::gpu::LayerSession>,
    host: Option<HostStep<F>>,
    weights: [FieldElement<F>; 2],
}

impl<F: IsField + 'static> DeviceStep<F>
where
    FieldElement<F>: Send + Sync,
{
    /// The card's factors, folded so far, brought here: the step goes on on
    /// the host's relation over them.
    fn hand_over(&mut self) -> Result<(), Error> {
        if let Some(session) = self.session.take() {
            let factors = session.factors::<F>()?;
            if factors.len() != 5 {
                return Err(Error::DeviceFailed {
                    stage: "layer factors",
                });
            }
            self.host = Some(HostStep::single(factors, &self.weights));
        }
        Ok(())
    }

    fn current(&mut self) -> Result<&mut dyn TreeStep<F>, Error> {
        match (self.session.as_mut(), self.host.as_mut()) {
            (Some(session), _) => Ok(session as &mut dyn TreeStep<F>),
            (None, Some(host)) => Ok(host as &mut dyn TreeStep<F>),
            (None, None) => Err(Error::DeviceFailed {
                stage: "layer session",
            }),
        }
    }
}

impl<F: IsField + 'static> TreeStep<F> for crate::gpu::LayerSession {
    fn round(&mut self) -> Result<Vec<FieldElement<F>>, Error> {
        crate::gpu::LayerSession::round::<F>(self)
    }

    fn bind(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        self.fold(r)
    }

    fn halves(&self) -> Result<Vec<TreeHalves<F>>, Error> {
        let factors = self.factors::<F>()?;
        let one = |slot: usize| -> Result<FieldElement<F>, Error> {
            factors
                .get(slot)
                .and_then(|f| f.as_constant().cloned())
                .ok_or(Error::NoVariablesLeft)
        };
        Ok(vec![TreeHalves {
            p_lo: one(1)?,
            p_hi: one(2)?,
            q_lo: one(3)?,
            q_hi: one(4)?,
        }])
    }
}

impl<F: IsField + 'static> TreeStep<F> for DeviceStep<F>
where
    FieldElement<F>: Send + Sync,
{
    fn round(&mut self) -> Result<Vec<FieldElement<F>>, Error> {
        self.current()?.round()
    }

    fn bind(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        self.current()?.bind(r)
    }

    fn halves(&self) -> Result<Vec<TreeHalves<F>>, Error> {
        match (&self.session, &self.host) {
            (Some(session), _) => TreeStep::<F>::halves(session),
            (None, Some(host)) => host.halves(),
            (None, None) => Err(Error::NoVariablesLeft),
        }
    }

    fn prefetch(&mut self) -> Result<(), Error> {
        match self.session.as_mut() {
            Some(session) => session.enqueue(),
            None => Ok(()),
        }
    }

    fn device_cube(&self) -> usize {
        self.session
            .as_ref()
            .map_or(0, |session| 1usize << session.num_vars())
    }

    fn to_host(&mut self) -> Result<(), Error> {
        self.hand_over()
    }

    fn host_step(&mut self) -> Option<&mut HostStep<F>> {
        if self.session.is_some() {
            return None;
        }
        self.host.as_mut()
    }
}

/// What a [`GruenStep`] needs of the card: Gruen's rounds over one layer's
/// halves ([`crate::gpu::GruenSession`]).
trait GruenCard<F: IsField> {
    /// The rounds it runs, `J`.
    fn rounds(&self) -> usize;
    /// Queues the next round, folding by `previous`, without waiting for it.
    fn enqueue(&mut self, previous: Option<&FieldElement<F>>, want_h0: bool) -> Result<(), Error>;
    /// The round's `H(1)`, `H(2)` — and `H(0)` with `want_h0`.
    fn sums(
        &mut self,
        previous: Option<&FieldElement<F>>,
        want_h0: bool,
    ) -> Result<Vec<FieldElement<F>>, Error>;
    /// After the last round: `p_lo, p_hi, q_lo, q_hi` bound to it.
    fn finish(&mut self, previous: Option<&FieldElement<F>>) -> Result<Vec<Mle<F>>, Error>;
}

impl<F: IsField + 'static> GruenCard<F> for crate::gpu::GruenSession {
    fn rounds(&self) -> usize {
        crate::gpu::GruenSession::rounds(self)
    }

    fn enqueue(&mut self, previous: Option<&FieldElement<F>>, want_h0: bool) -> Result<(), Error> {
        crate::gpu::GruenSession::enqueue(self, previous, want_h0)
    }

    fn sums(
        &mut self,
        previous: Option<&FieldElement<F>>,
        want_h0: bool,
    ) -> Result<Vec<FieldElement<F>>, Error> {
        crate::gpu::GruenSession::sums(self, previous, want_h0)
    }

    fn finish(&mut self, previous: Option<&FieldElement<F>>) -> Result<Vec<Mle<F>>, Error> {
        crate::gpu::GruenSession::finish(self, previous)
    }
}

impl<F: IsField + 'static> GruenCard<F> for crate::gpu::Gruen2Session {
    fn rounds(&self) -> usize {
        crate::gpu::Gruen2Session::rounds(self)
    }

    fn enqueue(&mut self, previous: Option<&FieldElement<F>>, want_h0: bool) -> Result<(), Error> {
        crate::gpu::Gruen2Session::enqueue(self, previous, want_h0)
    }

    fn sums(
        &mut self,
        previous: Option<&FieldElement<F>>,
        want_h0: bool,
    ) -> Result<Vec<FieldElement<F>>, Error> {
        crate::gpu::Gruen2Session::sums(self, previous, want_h0)
    }

    fn finish(&mut self, previous: Option<&FieldElement<F>>) -> Result<Vec<Mle<F>>, Error> {
        crate::gpu::Gruen2Session::finish(self, previous)
    }
}

/// The step over a device layer on Gruen's rounds ([`crate::gkr_gruen`], on
/// with `gkr_gruen::argue_gkr_gruen` as today's per-table layers are): the
/// tree's relation is `w_p·eq·(p_lo q_hi + p_hi q_lo + λ'·q_lo q_hi)` with
/// `λ' = w_q/w_p`, so the card sums its `H(1)`, `H(2)` under `λ'`, the tree's
/// own chain (from its claim `p + λ'·q`) makes the message a per-table layer
/// would send, and the step sends it times `w_p` — the field elements
/// [`HostStep`] sums. At the card's tail it hands the host today's five
/// factors, `eq` included, and steps on there.
struct GruenStep<F: IsField, C> {
    session: Option<C>,
    /// Whether the session is a two-layer step's: its card sums `H(1)..H(4)`
    /// of a quartic and hands back eight factors, and the chain's messages
    /// are of degree 5.
    double: bool,
    chain: crate::gkr_gruen::Chain<F>,
    /// The layer's point `u`, one coordinate a round.
    point: Vec<FieldElement<F>>,
    /// `1/(1 − u_j)` for each card round; `None` sums `H(0)` on the card.
    lefts: Vec<Option<FieldElement<F>>>,
    /// The card rounds run so far.
    round: usize,
    /// This round's message, unscaled, until its challenge binds it.
    message: Option<crate::gkr_gruen::Message<F>>,
    /// The last round's challenge, which the next card pass folds by.
    previous: Option<FieldElement<F>>,
    host: Option<HostStep<F>>,
    weights: [FieldElement<F>; 2],
}

/// `λ' = w_q/w_p`, or `None` when `w_p` has no inverse.
fn gruen_lambda<F: IsField>(weights: &[FieldElement<F>; 2]) -> Option<FieldElement<F>> {
    weights[0].inv().ok().map(|inverse| &weights[1] * &inverse)
}

impl<F: IsField + 'static> GruenStep<F, crate::gpu::GruenSession>
where
    FieldElement<F>: Send + Sync,
{
    /// `None` when the step cannot run this way — `w_p` with no inverse, or a
    /// card that declines the session — and before anything moves, so the
    /// caller opens today's session instead.
    fn open(
        device: &crate::gpu::DeviceTree,
        layer: usize,
        point: &[FieldElement<F>],
        weights: &[FieldElement<F>; 2],
        claim: &(FieldElement<F>, FieldElement<F>),
    ) -> Result<Option<Self>, Error> {
        let Some(lambda) = gruen_lambda(weights) else {
            return Ok(None);
        };
        let tail = crate::gkr_gruen::gkr_gruen_tail();
        let Some(session) = device.gruen_session(layer, point, &lambda, tail) else {
            return Ok(None);
        };
        Self::over(session, point, weights, claim)
    }
}

impl<F: IsField + 'static> GruenStep<F, crate::gpu::Gruen2Session>
where
    FieldElement<F>: Send + Sync,
{
    /// The two-layer step over layer `layer`'s quarters on the card, or
    /// `None` — before anything moves — when it cannot run this way.
    fn open_double(
        device: &crate::gpu::DeviceTree,
        layer: usize,
        point: &[FieldElement<F>],
        weights: &[FieldElement<F>; 2],
        claim: &(FieldElement<F>, FieldElement<F>),
    ) -> Result<Option<Self>, Error> {
        let Some(lambda) = gruen_lambda(weights) else {
            return Ok(None);
        };
        let tail = crate::gkr_gruen::gkr_gruen_tail();
        let Some(session) = device.gruen2_session(layer, point, &lambda, tail) else {
            return Ok(None);
        };
        Self::over_double(session, point, weights, claim)
    }
}

impl<F: IsField + 'static, C: GruenCard<F>> GruenStep<F, C>
where
    FieldElement<F>: Send + Sync,
{
    /// The step over `session`, opened under `λ' = w_q/w_p` at `point`.
    fn over(
        session: C,
        point: &[FieldElement<F>],
        weights: &[FieldElement<F>; 2],
        claim: &(FieldElement<F>, FieldElement<F>),
    ) -> Result<Option<Self>, Error> {
        Self::over_kind(session, point, weights, claim, false)
    }

    /// The two-layer step over `session`, opened under `λ'` at `point`.
    fn over_double(
        session: C,
        point: &[FieldElement<F>],
        weights: &[FieldElement<F>; 2],
        claim: &(FieldElement<F>, FieldElement<F>),
    ) -> Result<Option<Self>, Error> {
        Self::over_kind(session, point, weights, claim, true)
    }

    fn over_kind(
        session: C,
        point: &[FieldElement<F>],
        weights: &[FieldElement<F>; 2],
        claim: &(FieldElement<F>, FieldElement<F>),
        double: bool,
    ) -> Result<Option<Self>, Error> {
        let Some(lambda) = gruen_lambda(weights) else {
            return Ok(None);
        };
        let Some(chain) = crate::gkr_gruen::Chain::new(&claim.0 + &lambda * &claim.1) else {
            return Ok(None);
        };
        let lefts = crate::gkr_gruen::inverse_lefts(&point[..session.rounds()]);
        let mut step = Self {
            session: Some(session),
            double,
            chain,
            point: point.to_vec(),
            lefts,
            round: 0,
            message: None,
            previous: None,
            host: None,
            weights: weights.clone(),
        };
        if step.lefts.is_empty() {
            step.hand_over()?;
        }
        Ok(Some(step))
    }

    /// After the card's last round: its halves, and `κ·eq(u_{J..})` for the
    /// weight, as the host's relation.
    fn hand_over(&mut self) -> Result<(), Error> {
        let Some(mut session) = self.session.take() else {
            return Ok(());
        };
        let halves = session.finish(self.previous.as_ref())?;
        let rounds = self.lefts.len();
        let eq: Vec<FieldElement<F>> = crate::eq::eq_evals(&self.point[rounds..])
            .iter()
            .map(|v| self.chain.kappa() * v)
            .collect();
        let factors = if self.double { 9 } else { 5 };
        let mut polys = Vec::with_capacity(factors);
        polys.push(Mle::new(eq)?);
        polys.extend(halves);
        if polys.len() != factors {
            return Err(Error::DeviceFailed {
                stage: "layer factors",
            });
        }
        crate::gkr_gruen::note_layer(rounds as u64, false);
        self.host = Some(if self.double {
            crate::gkr_gruen::note_two_layer();
            HostStep::double(polys, &self.weights)
        } else {
            HostStep::single(polys, &self.weights)
        });
        Ok(())
    }
}

impl<F: IsField + 'static, C: GruenCard<F>> TreeStep<F> for GruenStep<F, C>
where
    FieldElement<F>: Send + Sync,
{
    fn round(&mut self) -> Result<Vec<FieldElement<F>>, Error> {
        let Some(session) = self.session.as_mut() else {
            return self.host.as_mut().ok_or(Error::NoVariablesLeft)?.round();
        };
        let left = self.lefts[self.round].as_ref();
        let sums = session.sums(self.previous.as_ref(), left.is_none())?;
        if left.is_none() {
            crate::gkr_gruen::note_direct_h0();
        }
        let message = if self.double {
            if sums.len() < 4 {
                return Err(Error::DeviceFailed {
                    stage: "gruen message",
                });
            }
            self.chain.message_quartic(
                &self.point[self.round],
                left,
                [&sums[0], &sums[1], &sums[2], &sums[3]],
                sums.get(4).filter(|_| left.is_none()),
            )
        } else {
            self.chain.message(
                &self.point[self.round],
                left,
                &sums[0],
                &sums[1],
                sums.get(2).filter(|_| left.is_none()),
            )
        }
        .ok_or(Error::DeviceFailed {
            stage: "gruen message",
        })?;
        let sent = message.sent.iter().map(|v| v * &self.weights[0]).collect();
        self.message = Some(message);
        Ok(sent)
    }

    fn bind(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        if self.session.is_none() {
            return self.host.as_mut().ok_or(Error::NoVariablesLeft)?.bind(r);
        }
        let message = self.message.take().ok_or(Error::DeviceFailed {
            stage: "gruen message",
        })?;
        self.chain.advance(&self.point[self.round], &message, r);
        self.previous = Some(r.clone());
        self.round += 1;
        if self.round == self.lefts.len() {
            self.hand_over()?;
        }
        Ok(())
    }

    fn halves(&self) -> Result<Vec<TreeHalves<F>>, Error> {
        self.host.as_ref().ok_or(Error::NoVariablesLeft)?.halves()
    }

    fn prefetch(&mut self) -> Result<(), Error> {
        let want_h0 = self.lefts.get(self.round).is_some_and(Option::is_none);
        match self.session.as_mut() {
            Some(session) => session.enqueue(self.previous.as_ref(), want_h0),
            None => Ok(()),
        }
    }

    fn host_step(&mut self) -> Option<&mut HostStep<F>> {
        if self.session.is_some() {
            return None;
        }
        self.host.as_mut()
    }

    // `device_cube` stays zero and `to_host` does nothing: a Gruen step leaves
    // the card at its own tail (`LAMBDA_VM_ARGUE_GKR_GRUEN_TAIL`), the cube
    // today's per-table layers leave it at.
}

/// `eq·(w_p·(p_lo q_hi + p_hi q_lo) + w_q·q_lo q_hi)` as the device's program:
/// the batched step's relation for one tree.
fn tree_program<F: IsField + 'static>(
    weights: &[FieldElement<F>; 2],
) -> Result<crate::program::Program<F>, Error> {
    use crate::program::Builder;
    let mut b = Builder::<F>::new();
    let eq = b.var(0);
    let p_lo = b.var(1);
    let p_hi = b.var(2);
    let q_lo = b.var(3);
    let q_hi = b.var(4);
    let first = b.mul(p_lo, q_hi);
    let second = b.mul(p_hi, q_lo);
    let numerator = b.add(first, second);
    let denominator = b.mul(q_lo, q_hi);
    let wp = b.fixed(weights[0].clone());
    let wq = b.fixed(weights[1].clone());
    let a = b.mul(wp, numerator);
    let c = b.mul(wq, denominator);
    let sum = b.add(a, c);
    let root = b.mul(eq, sum);
    b.finish(root)
}

/// A table's tree as the argue builds it: its levels here, or on a device.
impl<F: IsField + 'static> LadderTree<F> for crate::gkr::FractionTree<F>
where
    FieldElement<F>: Send + Sync,
{
    fn input_vars(&self) -> usize {
        self.num_layers().saturating_sub(1)
    }

    fn output(&self) -> (FieldElement<F>, FieldElement<F>) {
        crate::gkr::FractionTree::output(self)
    }

    fn step(
        &self,
        layer: usize,
        point: &[FieldElement<F>],
        weights: [FieldElement<F>; 2],
        claim: &(FieldElement<F>, FieldElement<F>),
    ) -> Result<Box<dyn TreeStep<F> + '_>, Error> {
        let Some(device) = self.device() else {
            return Ok(Box::new(HostStep::over(self.layer(layer), point, weights)?));
        };
        // A level near the output comes here whole, as today's prefix does:
        // its rounds are over cubes a core walks in microseconds. Never the
        // input layer, which a tree that gave it back writes again.
        let prefix = crate::HOST_CUBE_DIRECT.trailing_zeros() as usize + 1;
        if layer <= prefix && layer + 1 < self.num_layers() {
            let (p, q) = device
                .layer_to_host::<F>(layer)
                .ok_or(Error::DeviceFailed {
                    stage: "layer to host",
                })?;
            let next = FractionLayer::new(p, q)?;
            return Ok(Box::new(HostStep::over(&next, point, weights)?));
        }
        if crate::gkr_gruen::argue_gkr_gruen()
            && let Some(step) = GruenStep::open(device, layer, point, &weights, claim)?
        {
            return Ok(Box::new(step));
        }
        let program = tree_program(&weights)?;
        let session =
            device
                .layer_session(layer, point, &program, 3)
                .ok_or(Error::DeviceFailed {
                    stage: "layer session",
                })?;
        Ok(Box::new(DeviceStep {
            session: Some(session),
            host: None,
            weights,
        }))
    }

    /// The two-layer step: on the card's Gruen rounds over layer `layer`'s
    /// quarters ([`crate::gpu::Gruen2Session`]) when the card holds the tree;
    /// on the host over a level near the output (the one-layer step's prefix
    /// rule: never the input layer), or wherever the card declines, over the
    /// layer brought here whole. A layer the card can neither step nor hand
    /// over (an input layer the tree keeps apart from its levels) fails here,
    /// and the caller that can write it on the host does
    /// (`stark::multilinear_table::batched`'s input-on-host tree).
    fn step_double(
        &self,
        layer: usize,
        point: &[FieldElement<F>],
        weights: [FieldElement<F>; 2],
        claim: &(FieldElement<F>, FieldElement<F>),
    ) -> Result<Box<dyn TreeStep<F> + '_>, Error> {
        let Some(device) = self.device() else {
            return Ok(Box::new(HostStep::over_double(
                self.layer(layer),
                point,
                weights,
            )?));
        };
        let prefix = crate::HOST_CUBE_DIRECT.trailing_zeros() as usize + 1;
        let near_the_output = layer <= prefix && layer + 1 < self.num_layers();
        if !near_the_output
            && crate::gkr_gruen::argue_gkr_gruen()
            && let Some(step) = GruenStep::open_double(device, layer, point, &weights, claim)?
        {
            return Ok(Box::new(step));
        }
        let (p, q) = device
            .layer_to_host::<F>(layer)
            .ok_or(Error::DeviceFailed {
                stage: "layer to host",
            })?;
        let grandchild = FractionLayer::new(p, q)?;
        Ok(Box::new(HostStep::over_double(
            &grandchild,
            point,
            weights,
        )?))
    }
}

/// Proves every tree's layers in lockstep, on the host, stepping by `form`.
///
/// `trees[t]` is tree `t`'s layers, output first (`trees[t][0]` has no
/// variables, the last is the input layer). Returns the ladder and each tree's
/// input-layer claim, in tree order. A tree with no variables at all is its
/// own input claim at the empty point and takes no step.
pub fn prove<F, T>(
    trees: &[&[FractionLayer<F>]],
    form: GkrLadder,
    transcript: &mut T,
) -> Result<(LockstepProof<F>, Vec<GkrClaim<F>>), Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
    FieldElement<F>: Send + Sync,
{
    for tree in trees {
        for (i, layer) in tree.iter().enumerate() {
            if layer.num_vars() != i {
                return Err(Error::VariableCountMismatch {
                    expected: i,
                    got: layer.num_vars(),
                });
            }
        }
        if tree.is_empty() {
            return Err(Error::EmptyPolynomial);
        }
    }
    let hosts: Vec<HostLayers<'_, F>> = trees.iter().map(|t| HostLayers(t)).collect();
    let trees: Vec<&dyn LadderTree<F>> = hosts.iter().map(|t| t as &dyn LadderTree<F>).collect();
    prove_trees(&trees, form, transcript)
}

/// A tree's own message extended to the step's degree: `own` is its round
/// polynomial at `1, ..=d`, `claim` the sum it carries into the round (so its
/// value at 0 is `claim − own(1)`); returns the values at `0..=d` and the
/// message at `1, ..=degree`. A tree whose relation is of lower degree than
/// its step's sends this, which is its own polynomial's values there.
fn extend<F: IsField>(
    claim: &FieldElement<F>,
    own: Vec<FieldElement<F>>,
    degree: usize,
) -> (Vec<FieldElement<F>>, Vec<FieldElement<F>>) {
    let mut nodes = Vec::with_capacity(own.len() + 1);
    nodes.push(claim - &own[0]);
    nodes.extend(own);
    let message = (1..=degree)
        .map(|t| match nodes.get(t) {
            Some(v) => v.clone(),
            None => sumcheck::interpolate(&nodes, &FieldElement::from(t as u64)),
        })
        .collect();
    (nodes, message)
}

/// [`prove`] over trees wherever their layers are.
pub fn prove_trees<F, T>(
    trees: &[&dyn LadderTree<F>],
    form: GkrLadder,
    transcript: &mut T,
) -> Result<(LockstepProof<F>, Vec<GkrClaim<F>>), Error>
where
    F: IsField + 'static,
    FieldElement<F>: Send + Sync,
    T: IsTranscript<F>,
{
    let side_by_side = argue_host_parallel();
    let mut host_times = HostRoundTimes::start(trees.len());
    let input_vars: Vec<usize> = trees.iter().map(|t| t.input_vars()).collect();
    let schedule = ladder_steps(&input_vars, form);

    let mut claims: Vec<(FieldElement<F>, FieldElement<F>)> =
        trees.iter().map(|tree| tree.output()).collect();
    let mut inputs: Vec<Option<GkrClaim<F>>> = (0..trees.len())
        .map(|t| {
            (input_vars[t] == 0).then(|| GkrClaim {
                point: Vec::new(),
                p: claims[t].0.clone(),
                q: claims[t].1.clone(),
            })
        })
        .collect();
    let mut point: Vec<FieldElement<F>> = Vec::new();
    let mut layers = Vec::with_capacity(schedule.len());

    for step in &schedule {
        let i = step.parent;
        let mu: FieldElement<F> = transcript.sample_field_element();
        let weights = challenge_powers(&mu, 2 * step.relations.len());
        let mut stepping: Vec<Box<dyn TreeStep<F> + '_>> = step
            .relations
            .iter()
            .zip(weights.chunks(2))
            .map(|(&(t, relation), w)| {
                let w = [w[0].clone(), w[1].clone()];
                match relation {
                    Relation::Single => trees[t].step(i + 1, &point, w, &claims[t]),
                    Relation::Double => trees[t].step_double(i + 2, &point, w, &claims[t]),
                }
            })
            .collect::<Result<_, _>>()?;
        // A tree whose relation is of lower degree than the step's carries
        // its own running sum, to extend its message to the step's degree
        // ([`extend`]); today's ladder has none.
        let mut running: Vec<Option<FieldElement<F>>> = step
            .relations
            .iter()
            .zip(weights.chunks(2))
            .map(|(&(t, relation), w)| {
                (relation.degree() < step.degree)
                    .then(|| &w[0] * &claims[t].0 + &w[1] * &claims[t].1)
            })
            .collect();

        let mut rounds = Vec::with_capacity(i);
        let mut z = Vec::with_capacity(i);
        for _ in 0..i {
            // The crossover on the step's SUMMED cube: while the active
            // trees' device cubes together are worth a launch, they stay on
            // the card; below it every one comes here — a batched host round
            // then walks all of them at the cost of one.
            let cube: usize = stepping
                .iter()
                .map(|tree_step| tree_step.device_cube())
                .sum();
            if cube > 0 && cube <= crate::HOST_CUBE_DIRECT {
                for tree_step in stepping.iter_mut() {
                    tree_step.to_host()?;
                }
            }
            for tree_step in stepping.iter_mut() {
                tree_step.prefetch()?;
            }
            // The host steps' rounds side by side, when there are enough of
            // them: each tree's message is its own, and the sum below still
            // adds them in tree order.
            let parallel = side_by_side && stepping.len() >= HOST_PARALLEL_MIN_STEPS;
            let mut ahead = if parallel {
                host_times.time(|| host_rounds_parallel(&mut stepping))?
            } else {
                Vec::new()
            };
            let mut sent = vec![FieldElement::<F>::zero(); step.degree];
            let mut nodes: Vec<Option<Vec<FieldElement<F>>>> = vec![None; stepping.len()];
            for (k, tree_step) in stepping.iter_mut().enumerate() {
                let own = match ahead.get_mut(k).and_then(Option::take) {
                    Some(own) => own,
                    None if host_times.on && tree_step.host_step().is_some() => {
                        host_times.time(|| tree_step.round())?
                    }
                    None => tree_step.round()?,
                };
                let degree = step.relations[k].1.degree();
                if own.len() != degree {
                    return Err(Error::RoundDegreeMismatch {
                        round: rounds.len(),
                        expected: degree,
                        got: own.len(),
                    });
                }
                let own = match &running[k] {
                    Some(claim) => {
                        let (at, message) = extend(claim, own, step.degree);
                        nodes[k] = Some(at);
                        message
                    }
                    None => own,
                };
                for (s, v) in sent.iter_mut().zip(own) {
                    *s += v;
                }
            }
            for e in &sent {
                transcript.append_field_element(e);
            }
            let r: FieldElement<F> = transcript.sample_field_element();
            for (claim, at) in running.iter_mut().zip(&nodes) {
                if let (Some(claim), Some(at)) = (claim.as_mut(), at) {
                    *claim = sumcheck::interpolate(at, &r);
                }
            }
            let bound = if parallel {
                host_times.time(|| host_binds_parallel(&mut stepping, &r))?
            } else {
                Vec::new()
            };
            for (k, tree_step) in stepping.iter_mut().enumerate() {
                if bound.get(k).copied().unwrap_or(false) {
                    continue;
                }
                if host_times.on && tree_step.host_step().is_some() {
                    host_times.time(|| tree_step.bind(&r))?;
                } else {
                    tree_step.bind(&r)?;
                }
            }
            rounds.push(sumcheck::RoundProof { evaluations: sent });
            z.push(r);
        }

        let mut halves = Vec::with_capacity(step.halves());
        for (tree_step, &(_, relation)) in stepping.iter().zip(&step.relations) {
            let own = tree_step.halves()?;
            if own.len() != relation.halves() {
                return Err(Error::ArgueShapeMismatch {
                    part: "ladder step",
                });
            }
            for h in own {
                for v in [&h.p_lo, &h.p_hi, &h.q_lo, &h.q_hi] {
                    transcript.append_field_element(v);
                }
                halves.push(h);
            }
        }
        drop(stepping);
        let (c1, c2) = draw_point_challenges(step, transcript);
        point = descend(
            step,
            &halves,
            &c1,
            c2.as_ref(),
            &z,
            &input_vars,
            &mut claims,
            &mut inputs,
        )?;
        layers.push(LockstepLayer {
            sumcheck: SumcheckProof { rounds },
            halves,
        });
    }

    host_times.print(side_by_side);
    let inputs = inputs
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(Error::NoVariablesLeft)?;
    Ok((LockstepProof { layers }, inputs))
}

/// The fewest steps in a round whose host rounds go side by side
/// ([`argue_host_parallel`]): the many-table groups' ladders (hundreds of
/// trees) qualify, a usual group's (a handful) keeps one thread.
pub const HOST_PARALLEL_MIN_STEPS: usize = 32;

/// The fewest host steps one task takes: a small layer's round is a microsecond
/// or two, so a task must hold enough of them to pay for itself.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
const HOST_PARALLEL_CHUNK: usize = 16;

static HOST_PARALLEL_FORCED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether a ladder round with at least [`HOST_PARALLEL_MIN_STEPS`] steps runs
/// its host steps' rounds and binds side by side on the calling thread's pool
/// (on by default; `LAMBDA_VM_ARGUE_HOST_PARALLEL=0` turns it off, and so
/// does a build without the `parallel` feature). Each step's message and
/// bind are its own and the round's sum still adds the messages in tree
/// order, so the proof is the same. Read once, with a banner.
pub fn argue_host_parallel() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = match HOST_PARALLEL_FORCED.load(std::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => *ON.get_or_init(|| {
            let on = host_parallel_from(
                std::env::var("LAMBDA_VM_ARGUE_HOST_PARALLEL")
                    .ok()
                    .as_deref(),
            );
            eprintln!(
                "★ ARGUE HOST PARALLEL: {}",
                if on {
                    "on (the default; a ladder round's host steps side by side when it has many)"
                } else {
                    "off (one host step after another)"
                }
            );
            on
        }),
    };
    on && cfg!(feature = "parallel")
}

/// The knob's word: off for `0` only.
fn host_parallel_from(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim() != "0")
}

/// Overrides `LAMBDA_VM_ARGUE_HOST_PARALLEL` for the whole process — for a
/// test that proves both ways in one binary. `None` restores the environment's.
#[doc(hidden)]
pub fn force_argue_host_parallel(on: Option<bool>) {
    HOST_PARALLEL_FORCED.store(
        match on {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

static HOST_PARALLEL_PASSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many ladder rounds have run their host steps side by side in this
/// process (for a test that they did).
#[doc(hidden)]
pub fn host_parallel_passes() -> u64 {
    HOST_PARALLEL_PASSES.load(std::sync::atomic::Ordering::Relaxed)
}

/// The host steps among `stepping`, with their places.
fn host_steps<'s, F: IsField>(
    stepping: &'s mut [Box<dyn TreeStep<F> + '_>],
) -> (Vec<usize>, Vec<&'s mut HostStep<F>>) {
    let mut at = Vec::new();
    let mut hosts = Vec::new();
    for (k, step) in stepping.iter_mut().enumerate() {
        if let Some(host) = step.host_step() {
            at.push(k);
            hosts.push(host);
        }
    }
    (at, hosts)
}

/// The host steps' messages this round, computed side by side, at their
/// places among `stepping`; `None` where a step is on a device.
fn host_rounds_parallel<F>(
    stepping: &mut [Box<dyn TreeStep<F> + '_>],
) -> Result<Vec<Option<Vec<FieldElement<F>>>>, Error>
where
    F: IsField + 'static,
    FieldElement<F>: Send + Sync,
{
    let n = stepping.len();
    let (at, mut hosts) = host_steps(stepping);
    HOST_PARALLEL_PASSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let owns: Vec<Vec<FieldElement<F>>> = side_by_side(&mut hosts, |host| host.round())?;
    let mut ahead: Vec<Option<Vec<FieldElement<F>>>> = (0..n).map(|_| None).collect();
    for (k, own) in at.into_iter().zip(owns) {
        ahead[k] = Some(own);
    }
    Ok(ahead)
}

/// Binds the host steps to `r` side by side; `true` at the places bound.
fn host_binds_parallel<F>(
    stepping: &mut [Box<dyn TreeStep<F> + '_>],
    r: &FieldElement<F>,
) -> Result<Vec<bool>, Error>
where
    F: IsField + 'static,
    FieldElement<F>: Send + Sync,
{
    let n = stepping.len();
    let (at, mut hosts) = host_steps(stepping);
    side_by_side(&mut hosts, |host| host.bind(r))?;
    let mut bound = vec![false; n];
    for k in at {
        bound[k] = true;
    }
    Ok(bound)
}

/// `work` on every host step, in chunks of at least [`HOST_PARALLEL_CHUNK`] on
/// the calling thread's pool; the results in step order, the first error in
/// step order.
#[cfg(feature = "parallel")]
fn side_by_side<F, R, W>(hosts: &mut [&mut HostStep<F>], work: W) -> Result<Vec<R>, Error>
where
    F: IsField + 'static,
    FieldElement<F>: Send + Sync,
    R: Send,
    W: Fn(&mut HostStep<F>) -> Result<R, Error> + Sync,
{
    use rayon::prelude::*;
    let chunk = hosts
        .len()
        .div_ceil(rayon::current_num_threads().max(1))
        .max(HOST_PARALLEL_CHUNK);
    let parts: Vec<Result<Vec<R>, Error>> = hosts
        .par_chunks_mut(chunk)
        .map(|part| part.iter_mut().map(|host| work(host)).collect())
        .collect();
    let mut out = Vec::with_capacity(hosts.len());
    for part in parts {
        out.extend(part?);
    }
    Ok(out)
}

/// Without `parallel`: one after another.
#[cfg(not(feature = "parallel"))]
fn side_by_side<F, R, W>(hosts: &mut [&mut HostStep<F>], work: W) -> Result<Vec<R>, Error>
where
    F: IsField + 'static,
    W: Fn(&mut HostStep<F>) -> Result<R, Error>,
{
    hosts.iter_mut().map(|host| work(host)).collect()
}

/// A measurement, never a byte of the proof
/// (`LAMBDA_VM_ARGUE_HOST_ROUND_TIMES=1`): one ladder's host rounds and binds
/// in wall seconds, printed as one `LADDER HOST ROUNDS` line at its end.
struct HostRoundTimes {
    on: bool,
    trees: usize,
    started: std::time::Instant,
    spent: f64,
    calls: u64,
}

impl HostRoundTimes {
    fn start(trees: usize) -> Self {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON.get_or_init(|| {
            std::env::var("LAMBDA_VM_ARGUE_HOST_ROUND_TIMES").is_ok_and(|v| v.trim() == "1")
        });
        Self {
            on,
            trees,
            started: std::time::Instant::now(),
            spent: 0.0,
            calls: 0,
        }
    }

    fn time<R>(&mut self, work: impl FnOnce() -> R) -> R {
        if !self.on {
            return work();
        }
        let at = std::time::Instant::now();
        let out = work();
        self.spent += at.elapsed().as_secs_f64();
        self.calls += 1;
        out
    }

    fn print(&self, side_by_side: bool) {
        if self.on {
            eprintln!(
                "LADDER HOST ROUNDS: trees {} · total {:.3} ms · host rounds and binds {:.3} ms ({} passes) · side by side {}",
                self.trees,
                self.started.elapsed().as_secs_f64() * 1e3,
                self.spent * 1e3,
                self.calls,
                if side_by_side { "on" } else { "off" },
            );
        }
    }
}

/// Verifies a ladder against the trees' claimed outputs and their input
/// heights (`input_vars[t]` = `k_t`, from the statement, never the proof).
/// Returns each tree's input-layer claim, in tree order.
pub fn verify<F, T>(
    proof: &LockstepProof<F>,
    outputs: &[(FieldElement<F>, FieldElement<F>)],
    input_vars: &[usize],
    form: GkrLadder,
    transcript: &mut T,
) -> Result<Vec<GkrClaim<F>>, Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
{
    verify_with(proof, outputs, input_vars, form, transcript, true)
}

/// [`verify`], with the step residual's check switchable — `false` only in
/// the mutation test that shows the check is load-bearing.
#[doc(hidden)]
pub fn verify_with<F, T>(
    proof: &LockstepProof<F>,
    outputs: &[(FieldElement<F>, FieldElement<F>)],
    input_vars: &[usize],
    form: GkrLadder,
    transcript: &mut T,
    check_residual: bool,
) -> Result<Vec<GkrClaim<F>>, Error>
where
    F: IsField + 'static,
    T: IsTranscript<F>,
{
    if outputs.len() != input_vars.len() {
        return Err(Error::ArgueShapeMismatch {
            part: "tree outputs",
        });
    }
    // Every step's relations, degree and counts: the statement's heights and
    // the verifier's own format, never the proof.
    let schedule = ladder_steps(input_vars, form);
    if proof.layers.len() != schedule.len() {
        return Err(Error::ArgueShapeMismatch { part: "ladder" });
    }

    let mut claims: Vec<(FieldElement<F>, FieldElement<F>)> = outputs.to_vec();
    let mut inputs: Vec<Option<GkrClaim<F>>> = (0..outputs.len())
        .map(|t| {
            (input_vars[t] == 0).then(|| GkrClaim {
                point: Vec::new(),
                p: claims[t].0.clone(),
                q: claims[t].1.clone(),
            })
        })
        .collect();
    let mut point: Vec<FieldElement<F>> = Vec::new();

    for (i, (step, layer)) in schedule.iter().zip(&proof.layers).enumerate() {
        if layer.halves.len() != step.halves() {
            return Err(Error::ArgueShapeMismatch {
                part: "ladder step",
            });
        }
        let mu: FieldElement<F> = transcript.sample_field_element();
        let weights = challenge_powers(&mu, 2 * step.relations.len());
        let claimed = step
            .relations
            .iter()
            .zip(weights.chunks(2))
            .fold(FieldElement::<F>::zero(), |acc, (&(t, _), w)| {
                acc + &w[0] * &claims[t].0 + &w[1] * &claims[t].1
            });

        let claim = sumcheck::verify(
            &layer.sumcheck,
            claimed,
            point.len(),
            step.degree,
            transcript,
        )?;

        let values: Vec<FieldElement<F>> = layer
            .halves
            .iter()
            .flat_map(|h| {
                [
                    h.p_lo.clone(),
                    h.p_hi.clone(),
                    h.q_lo.clone(),
                    h.q_hi.clone(),
                ]
            })
            .collect();
        let expected =
            eq_eval(&point, &claim.point)? * step_relation(&step.relations, &weights, &values);
        if check_residual && expected != claim.expected_evaluation {
            return Err(Error::LayerRelationMismatch { layer: i });
        }

        for v in &values {
            transcript.append_field_element(v);
        }
        let (c1, c2) = draw_point_challenges(step, transcript);
        point = descend(
            step,
            &layer.halves,
            &c1,
            c2.as_ref(),
            &claim.point,
            input_vars,
            &mut claims,
            &mut inputs,
        )?;
    }

    inputs
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(Error::ArgueShapeMismatch { part: "ladder" })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gkr::{self, FractionTree};
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"gkr-lockstep-test")
    }

    /// A random fraction layer of `2^num_vars` cells, reproducible from `seed`.
    fn random_layer(num_vars: usize, seed: u64) -> FractionLayer<F> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            FE::from(s)
        };
        let size = 1usize << num_vars;
        let p: Vec<FE> = (0..size).map(|_| next()).collect();
        let q: Vec<FE> = (0..size).map(|_| next()).collect();
        FractionLayer::new(Mle::new(p).unwrap(), Mle::new(q).unwrap()).unwrap()
    }

    /// Every layer, output first, by host folding.
    fn layers(input: FractionLayer<F>) -> Vec<FractionLayer<F>> {
        let mut layers = vec![input];
        while layers.last().unwrap().num_vars() > 0 {
            let next = layers.last().unwrap().fold().unwrap();
            layers.push(next);
        }
        layers.reverse();
        layers
    }

    /// A probe, never a gate: the host's lockstep over one Ext3 tree of
    /// `2^k` input cells, for `k` 10, 13, 15 and 16 (ms, best of three).
    #[test]
    #[ignore = "a timing probe"]
    fn host_ladder_seconds_by_tree_height() {
        use math::field::element::FieldElement as FEl;
        type E3 = FEl<Ext3>;
        for k in [10usize, 13, 15, 16] {
            let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ k as u64;
            let mut next = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                E3::from(s)
            };
            let p: Vec<E3> = (0..1usize << k).map(|_| next()).collect();
            let q: Vec<E3> = (0..1usize << k).map(|_| next()).collect();
            let input = FractionLayer::new(Mle::new(p).unwrap(), Mle::new(q).unwrap()).unwrap();
            let mut layers = vec![input];
            while layers.last().unwrap().num_vars() > 0 {
                let next = layers.last().unwrap().fold().unwrap();
                layers.push(next);
            }
            layers.reverse();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let mut t = DefaultTranscript::<Ext3>::new(b"probe");
                let start = std::time::Instant::now();
                prove(&[layers.as_slice()], GkrLadder::Single, &mut t).unwrap();
                best = best.min(start.elapsed().as_secs_f64());
            }
            println!(
                "HOST LADDER PROBE: 2^{k} input cells · {:.3} ms",
                best * 1e3
            );
        }
    }

    fn round_trip(heights: &[usize]) -> (Vec<Vec<FractionLayer<F>>>, LockstepProof<F>) {
        round_trip_as(heights, GkrLadder::Single)
    }

    /// Each tree's output fraction, in tree order.
    fn outputs_of(trees: &[Vec<FractionLayer<F>>]) -> Vec<(FE, FE)> {
        trees
            .iter()
            .map(|t| (t[0].p.evals()[0], t[0].q.evals()[0]))
            .collect()
    }

    /// Random trees of `heights`, every layer, output first.
    fn trees_of(heights: &[usize], seed: u64) -> Vec<Vec<FractionLayer<F>>> {
        heights
            .iter()
            .enumerate()
            .map(|(t, &k)| layers(random_layer(k, seed + t as u64)))
            .collect()
    }

    /// Whether every claim is its tree's input layer at its point.
    fn claims_are_the_inputs(trees: &[Vec<FractionLayer<F>>], claims: &[GkrClaim<F>]) -> Vec<bool> {
        trees
            .iter()
            .zip(claims)
            .map(|(tree, claim)| {
                let input = tree.last().unwrap();
                claim.point.len() == input.num_vars()
                    && input.p.evaluate(&claim.point).unwrap() == claim.p
                    && input.q.evaluate(&claim.point).unwrap() == claim.q
            })
            .collect()
    }

    fn round_trip_as(
        heights: &[usize],
        form: GkrLadder,
    ) -> (Vec<Vec<FractionLayer<F>>>, LockstepProof<F>) {
        let trees = trees_of(heights, 17);
        let refs: Vec<&[FractionLayer<F>]> = trees.iter().map(|t| t.as_slice()).collect();
        let mut prover = transcript();
        let (proof, claims) = prove(&refs, form, &mut prover).unwrap();

        let outputs = outputs_of(&trees);
        let mut verifier = transcript();
        let got = verify(&proof, &outputs, heights, form, &mut verifier).unwrap();
        assert_eq!(
            got, claims,
            "the prover and the verifier reach the same claims"
        );
        assert_eq!(prover.state(), verifier.state(), "and the same transcript");
        // Each claim is its tree's input layer at the claim point.
        for (tree, claim) in trees.iter().zip(&claims) {
            let input = tree.last().unwrap();
            assert_eq!(claim.point.len(), input.num_vars());
            assert_eq!(input.p.evaluate(&claim.point).unwrap(), claim.p);
            assert_eq!(input.q.evaluate(&claim.point).unwrap(), claim.q);
        }
        (trees, proof)
    }

    /// Trees of different heights, one of them without variables, share one
    /// ladder; each leaves it with its own input claim.
    #[test]
    fn trees_of_different_heights_round_trip_in_lockstep() {
        round_trip(&[5, 2, 0, 5, 3]);
        round_trip(&[1]);
        round_trip(&[0]);
        round_trip(&[4, 4]);
    }

    /// With one tree the ladder is today's GKR, message for message: `μ` is
    /// drawn where `λ` was and plays its role.
    #[test]
    fn one_tree_is_todays_gkr() {
        for k in 0..6 {
            let input = random_layer(k, 99 + k as u64);
            let tree = FractionTree::build(input.clone()).unwrap();
            let today = gkr::prove(&tree, &mut transcript()).unwrap();
            let layers = layers(input);
            let (proof, claims) =
                prove(&[layers.as_slice()], GkrLadder::Single, &mut transcript()).unwrap();
            assert_eq!(claims, vec![today.claim.clone()], "k = {k}");
            assert_eq!(proof.layers.len(), today.proof.layers.len());
            for (ours, theirs) in proof.layers.iter().zip(&today.proof.layers) {
                assert_eq!(ours.sumcheck, theirs.sumcheck, "k = {k}");
                let h = &ours.halves[0];
                assert_eq!(
                    [&h.p_lo, &h.p_hi, &h.q_lo, &h.q_hi],
                    [&theirs.p_lo, &theirs.p_hi, &theirs.q_lo, &theirs.q_hi]
                );
            }
        }
    }

    /// One tree's halves wrong at a mid step of a multi-tree ladder: the step's
    /// residual refuses it, and names the step.
    #[test]
    fn one_trees_halves_wrong_at_a_mid_step_are_refused() {
        let heights = [4, 3, 4];
        let (trees, proof) = round_trip(&heights);
        let outputs: Vec<(FE, FE)> = trees
            .iter()
            .map(|t| (t[0].p.evals()[0], t[0].q.evals()[0]))
            .collect();
        for tree in 0..3 {
            let mut bad = proof.clone();
            bad.layers[2].halves[tree].q_lo += FE::one();
            assert_eq!(
                verify(
                    &bad,
                    &outputs,
                    &heights,
                    GkrLadder::Single,
                    &mut transcript()
                )
                .unwrap_err(),
                Error::LayerRelationMismatch { layer: 2 },
                "tree {tree}"
            );
        }
        // A round value of the shared sumcheck, likewise.
        let mut bad = proof.clone();
        bad.layers[3].sumcheck.rounds[1].evaluations[2] += FE::one();
        assert_eq!(
            verify(
                &bad,
                &outputs,
                &heights,
                GkrLadder::Single,
                &mut transcript()
            )
            .unwrap_err(),
            Error::LayerRelationMismatch { layer: 3 }
        );
        // A wrong output of one tree.
        let mut wrong = outputs.clone();
        wrong[1].0 += FE::one();
        assert!(
            verify(
                &proof,
                &wrong,
                &heights,
                GkrLadder::Single,
                &mut transcript()
            )
            .is_err()
        );
    }

    /// The residual is load-bearing: made inert, the mid-step tamper above is
    /// no longer refused at its step.
    #[test]
    fn the_residual_check_made_inert_lets_a_bad_step_through() {
        let heights = [4, 3, 4];
        let (trees, proof) = round_trip(&heights);
        let outputs: Vec<(FE, FE)> = trees
            .iter()
            .map(|t| (t[0].p.evals()[0], t[0].q.evals()[0]))
            .collect();
        let mut bad = proof.clone();
        bad.layers[2].halves[1].q_lo += FE::one();
        assert!(
            verify_with(
                &bad,
                &outputs,
                &heights,
                GkrLadder::Single,
                &mut transcript(),
                false
            )
            .is_ok()
        );
    }

    /// The shape is the statement's: a ladder a step short, a step with a tree
    /// missing, or heights the proof was not made for are refused.
    #[test]
    fn a_ladder_of_the_wrong_shape_is_refused() {
        let heights = [4, 2];
        let (trees, proof) = round_trip(&heights);
        let outputs: Vec<(FE, FE)> = trees
            .iter()
            .map(|t| (t[0].p.evals()[0], t[0].q.evals()[0]))
            .collect();
        let mut short = proof.clone();
        short.layers.pop();
        assert!(
            verify(
                &short,
                &outputs,
                &heights,
                GkrLadder::Single,
                &mut transcript()
            )
            .is_err()
        );
        let mut missing = proof.clone();
        missing.layers[1].halves.pop();
        assert!(
            verify(
                &missing,
                &outputs,
                &heights,
                GkrLadder::Single,
                &mut transcript()
            )
            .is_err()
        );
        assert!(
            verify(
                &proof,
                &outputs,
                &[4, 3],
                GkrLadder::Single,
                &mut transcript()
            )
            .is_err()
        );
        assert!(
            verify(
                &proof,
                &outputs[..1],
                &heights,
                GkrLadder::Single,
                &mut transcript()
            )
            .is_err()
        );
    }

    /// Over the extension too, which is where the argue runs.
    #[test]
    fn the_ladder_runs_over_the_extension() {
        let lift = |layer: FractionLayer<F>| {
            let up = |m: &Mle<F>| {
                Mle::new(m.evals().iter().map(|v| v.to_extension::<Ext3>()).collect()).unwrap()
            };
            FractionLayer::new(up(&layer.p), up(&layer.q)).unwrap()
        };
        let trees: Vec<Vec<FractionLayer<Ext3>>> = [3usize, 1, 3]
            .iter()
            .enumerate()
            .map(|(t, &k)| {
                let mut layers = vec![lift(random_layer(k, t as u64))];
                while layers.last().unwrap().num_vars() > 0 {
                    let next = layers.last().unwrap().fold().unwrap();
                    layers.push(next);
                }
                layers.reverse();
                layers
            })
            .collect();
        let refs: Vec<&[FractionLayer<Ext3>]> = trees.iter().map(|t| t.as_slice()).collect();
        let mut prover = DefaultTranscript::<Ext3>::new(b"ext");
        let (proof, claims) = prove(&refs, GkrLadder::Single, &mut prover).unwrap();
        let outputs: Vec<_> = trees
            .iter()
            .map(|t| (t[0].p.evals()[0], t[0].q.evals()[0]))
            .collect();
        let got = verify(
            &proof,
            &outputs,
            &[3, 1, 3],
            GkrLadder::Single,
            &mut DefaultTranscript::<Ext3>::new(b"ext"),
        )
        .unwrap();
        for (a, b) in got.iter().zip(&claims) {
            assert!(a.point == b.point && a.p == b.p && a.q == b.q);
        }
        assert_eq!(got.len(), claims.len());
    }

    /// The card's Gruen rounds on the host, value for value — the arithmetic
    /// of `gkr_gruen::host_rounds`: the halves folded by the previous
    /// challenge on the way in, `H` summed under the split weight.
    struct HostGruen {
        /// `p_lo, p_hi, q_lo, q_hi`, `2^m` cells each.
        f: [Vec<FE>; 4],
        m: usize,
        low: usize,
        e_lo: Vec<FE>,
        e_hi: Vec<Vec<FE>>,
        lambda: FE,
        done: usize,
        pending: Option<Vec<FE>>,
    }

    impl HostGruen {
        fn new(next: &FractionLayer<F>, point: &[FE], lambda: FE, low: usize) -> Self {
            let half = next.p.len() / 2;
            let (p, q) = (next.p.evals(), next.q.evals());
            let m = point.len();
            let low = low.min(m);
            let rounds = m - low;
            Self {
                f: [
                    p[..half].to_vec(),
                    p[half..].to_vec(),
                    q[..half].to_vec(),
                    q[half..].to_vec(),
                ],
                m,
                low,
                e_lo: crate::eq::eq_evals(&point[rounds..]),
                e_hi: (0..rounds)
                    .map(|j| crate::eq::eq_evals(&point[j + 1..rounds]))
                    .collect(),
                lambda,
                done: 0,
                pending: None,
            }
        }

        /// Binds the last round's variable: `2·width` live cells to `width`.
        fn fold(&mut self, previous: Option<&FE>, width: usize) {
            if let Some(s) = previous {
                for g in self.f.iter_mut() {
                    for x in 0..width {
                        g[x] = g[x] + s * &(g[x + width] - g[x]);
                    }
                }
            }
        }
    }

    impl GruenCard<F> for HostGruen {
        fn rounds(&self) -> usize {
            self.m - self.low
        }

        fn enqueue(&mut self, previous: Option<&FE>, want_h0: bool) -> Result<(), Error> {
            if self.pending.is_some() {
                return Ok(());
            }
            let j = self.done;
            let quarter = 1usize << (self.m - j - 1);
            self.fold(previous, 2 * quarter);
            let h = |v: [&FE; 4]| v[0] * v[3] + v[1] * v[2] + self.lambda * (v[2] * v[3]);
            let mut sums = vec![FE::zero(); if want_h0 { 3 } else { 2 }];
            for x in 0..quarter {
                let w = self.e_hi[j][x >> self.low] * self.e_lo[x & ((1 << self.low) - 1)];
                let lo: [&FE; 4] = std::array::from_fn(|k| &self.f[k][x]);
                let hi: [&FE; 4] = std::array::from_fn(|k| &self.f[k][x + quarter]);
                let two: [FE; 4] = std::array::from_fn(|k| hi[k] + hi[k] - lo[k]);
                sums[0] += w * h(hi);
                sums[1] += w * h([&two[0], &two[1], &two[2], &two[3]]);
                if want_h0 {
                    sums[2] += w * h(lo);
                }
            }
            self.pending = Some(sums);
            self.done += 1;
            Ok(())
        }

        fn sums(&mut self, previous: Option<&FE>, want_h0: bool) -> Result<Vec<FE>, Error> {
            self.enqueue(previous, want_h0)?;
            self.pending.take().ok_or(Error::NoVariablesLeft)
        }

        fn finish(&mut self, previous: Option<&FE>) -> Result<Vec<Mle<F>>, Error> {
            let cells = 1usize << self.low;
            self.fold(previous, cells);
            self.f
                .iter()
                .map(|g| Mle::new(g[..cells].to_vec()))
                .collect()
        }
    }

    /// The card's two-layer Gruen rounds on the host, value for value: the
    /// eight factors (`h₀`'s `p_lo, p_hi, q_lo, q_hi`, then `h₁`'s) folded by
    /// the previous challenge on the way in, `H(1)..H(4)` (and `H(0)`) of
    /// `N + λQ` summed under the split weight. The relation is written out
    /// by the two folds, as the card's reference composes it, not by the
    /// card's algebra.
    struct HostGruen2 {
        f: Vec<Vec<FE>>,
        m: usize,
        low: usize,
        e_lo: Vec<FE>,
        e_hi: Vec<Vec<FE>>,
        lambda: FE,
        done: usize,
        pending: Option<Vec<FE>>,
    }

    impl HostGruen2 {
        fn new(grandchild: &FractionLayer<F>, point: &[FE], lambda: FE, low: usize) -> Self {
            let quarter = grandchild.p.len() / 4;
            let (p, q) = (grandchild.p.evals(), grandchild.q.evals());
            let cells = |m: &[FE], b: usize| m[b * quarter..(b + 1) * quarter].to_vec();
            let f = (0..2)
                .flat_map(|b2| {
                    [
                        cells(p, b2),
                        cells(p, b2 + 2),
                        cells(q, b2),
                        cells(q, b2 + 2),
                    ]
                })
                .collect();
            let m = point.len();
            let low = low.min(m);
            let rounds = m - low;
            Self {
                f,
                m,
                low,
                e_lo: crate::eq::eq_evals(&point[rounds..]),
                e_hi: (0..rounds)
                    .map(|j| crate::eq::eq_evals(&point[j + 1..rounds]))
                    .collect(),
                lambda,
                done: 0,
                pending: None,
            }
        }

        fn fold(&mut self, previous: Option<&FE>, width: usize) {
            if let Some(s) = previous {
                for g in self.f.iter_mut() {
                    for x in 0..width {
                        g[x] = g[x] + s * &(g[x + width] - g[x]);
                    }
                }
            }
        }

        /// `N + λQ` at one node from the eight factors' values there.
        fn parent(&self, v: &[FE]) -> FE {
            let (n, d) = double_fraction(v);
            n + self.lambda * d
        }
    }

    impl GruenCard<F> for HostGruen2 {
        fn rounds(&self) -> usize {
            self.m - self.low
        }

        fn enqueue(&mut self, previous: Option<&FE>, want_h0: bool) -> Result<(), Error> {
            if self.pending.is_some() {
                return Ok(());
            }
            let j = self.done;
            let quarter = 1usize << (self.m - j - 1);
            self.fold(previous, 2 * quarter);
            let mut sums = vec![FE::zero(); if want_h0 { 5 } else { 4 }];
            for x in 0..quarter {
                let w = self.e_hi[j][x >> self.low] * self.e_lo[x & ((1 << self.low) - 1)];
                for t in 0..5u64 {
                    if t == 0 && !want_h0 {
                        continue;
                    }
                    let node = FE::from(t);
                    let v: Vec<FE> = self
                        .f
                        .iter()
                        .map(|g| g[x] + node * (g[x + quarter] - g[x]))
                        .collect();
                    let row = if t == 0 { 4 } else { t as usize - 1 };
                    sums[row] += w * self.parent(&v);
                }
            }
            self.pending = Some(sums);
            self.done += 1;
            Ok(())
        }

        fn sums(&mut self, previous: Option<&FE>, want_h0: bool) -> Result<Vec<FE>, Error> {
            self.enqueue(previous, want_h0)?;
            self.pending.take().ok_or(Error::NoVariablesLeft)
        }

        fn finish(&mut self, previous: Option<&FE>) -> Result<Vec<Mle<F>>, Error> {
            let cells = 1usize << self.low;
            self.fold(previous, cells);
            self.f
                .iter()
                .map(|g| Mle::new(g[..cells].to_vec()))
                .collect()
        }
    }

    /// A host tree whose steps run Gruen's rounds down to a cube of `2^low`,
    /// as a device tree's steps do on the card — its two-layer steps too.
    struct GruenHostTree<'a> {
        layers: &'a [FractionLayer<F>],
        low: usize,
    }

    impl LadderTree<F> for GruenHostTree<'_> {
        fn input_vars(&self) -> usize {
            self.layers.len() - 1
        }

        fn output(&self) -> (FE, FE) {
            (self.layers[0].p.evals()[0], self.layers[0].q.evals()[0])
        }

        fn step(
            &self,
            layer: usize,
            point: &[FE],
            weights: [FE; 2],
            claim: &(FE, FE),
        ) -> Result<Box<dyn TreeStep<F> + '_>, Error> {
            let next = &self.layers[layer];
            let lambda = gruen_lambda(&weights).expect("μ ≠ 0 in these runs");
            let card = HostGruen::new(next, point, lambda, self.low);
            let step = GruenStep::over(card, point, &weights, claim)?.expect("a Gruen step");
            Ok(Box::new(step))
        }

        fn step_double(
            &self,
            layer: usize,
            point: &[FE],
            weights: [FE; 2],
            claim: &(FE, FE),
        ) -> Result<Box<dyn TreeStep<F> + '_>, Error> {
            let lambda = gruen_lambda(&weights).expect("μ ≠ 0 in these runs");
            let card = HostGruen2::new(&self.layers[layer], point, lambda, self.low);
            let step = GruenStep::over_double(card, point, &weights, claim)?.expect("a Gruen step");
            Ok(Box::new(step))
        }
    }

    /// ★ The ladder's Gruen steps send the host reference's bytes: every tree
    /// at every step, with `w_p = μ^{2a}` scaling its own chain's messages,
    /// from tails of one cell up to the whole layer, and through the rounds
    /// that sum `H(0)` instead of deriving it.
    #[test]
    fn gruen_steps_send_the_host_references_bytes() {
        let heights = [6usize, 3, 0, 6, 4, 1];
        let trees: Vec<Vec<FractionLayer<F>>> = heights
            .iter()
            .enumerate()
            .map(|(t, &k)| layers(random_layer(k, 41 + t as u64)))
            .collect();
        let refs: Vec<&[FractionLayer<F>]> = trees.iter().map(|t| t.as_slice()).collect();
        let mut host = transcript();
        let (want, want_claims) = prove(&refs, GkrLadder::Single, &mut host).unwrap();
        for direct_h0 in [false, true] {
            crate::gkr_gruen::force_gkr_gruen_direct_h0(direct_h0);
            for low in [0usize, 1, 2, 6] {
                let gruen: Vec<GruenHostTree<'_>> = trees
                    .iter()
                    .map(|layers| GruenHostTree { layers, low })
                    .collect();
                let dyns: Vec<&dyn LadderTree<F>> =
                    gruen.iter().map(|t| t as &dyn LadderTree<F>).collect();
                let mut prover = transcript();
                let (got, claims) = prove_trees(&dyns, GkrLadder::Single, &mut prover).unwrap();
                assert!(
                    got == want && claims == want_claims,
                    "low {low}, direct H(0) {direct_h0}: the Gruen ladder moved a byte"
                );
                assert_eq!(prover.state(), host.state());
            }
        }
        crate::gkr_gruen::force_gkr_gruen_direct_h0(false);
    }

    /// ★ A ladder of many trees — host trees and Gruen steps that hand their
    /// layers to the host part way (tails of one cell up to the whole layer)
    /// — proves the same bytes, claims and transcript with its host steps run
    /// side by side as one after another, and its rounds did go side by side.
    #[test]
    fn host_steps_side_by_side_prove_the_same_bytes() {
        let heights: Vec<usize> = (0..72)
            .map(|t| [6usize, 3, 0, 5, 4, 1, 6, 2][t % 8])
            .collect();
        let trees: Vec<Vec<FractionLayer<F>>> = heights
            .iter()
            .enumerate()
            .map(|(t, &k)| layers(random_layer(k, 301 + t as u64)))
            .collect();
        let host: Vec<HostLayers<'_, F>> = trees[..36].iter().map(|t| HostLayers(t)).collect();
        let gruen: Vec<GruenHostTree<'_>> = trees[36..]
            .iter()
            .enumerate()
            .map(|(t, layers)| GruenHostTree {
                layers,
                low: [0usize, 1, 2, 6][t % 4],
            })
            .collect();
        let dyns: Vec<&dyn LadderTree<F>> = host
            .iter()
            .map(|t| t as &dyn LadderTree<F>)
            .chain(gruen.iter().map(|t| t as &dyn LadderTree<F>))
            .collect();
        let mut runs = Vec::new();
        for on in [false, true] {
            force_argue_host_parallel(Some(on));
            let passes = host_parallel_passes();
            let mut prover = transcript();
            let (proof, claims) = prove_trees(&dyns, GkrLadder::Single, &mut prover).unwrap();
            let ran = host_parallel_passes() - passes;
            if on && cfg!(feature = "parallel") {
                assert!(ran > 0, "no round went side by side");
            } else {
                assert_eq!(ran, 0, "a round went side by side with the knob off");
            }
            let outputs: Vec<(FE, FE)> = trees
                .iter()
                .map(|t| (t[0].p.evals()[0], t[0].q.evals()[0]))
                .collect();
            let mut verifier = transcript();
            let got = verify(&proof, &outputs, &heights, GkrLadder::Single, &mut verifier).unwrap();
            assert_eq!(got, claims, "side by side {on}: the verifier's claims");
            assert_eq!(prover.state(), verifier.state());
            runs.push((proof, claims, prover.state()));
        }
        force_argue_host_parallel(None);
        assert!(
            runs[0] == runs[1],
            "the side-by-side host steps moved a byte"
        );
    }

    /// Every value of `proof`, in order, through a fresh transcript: its
    /// state is a digest of the proof's bytes.
    fn proof_digest(proof: &LockstepProof<Ext3>) -> String {
        let mut t = DefaultTranscript::<Ext3>::new(b"lockstep-proof-digest");
        for layer in &proof.layers {
            t.append_bytes(&(layer.sumcheck.rounds.len() as u64).to_le_bytes());
            for round in &layer.sumcheck.rounds {
                t.append_bytes(&(round.evaluations.len() as u64).to_le_bytes());
                for v in &round.evaluations {
                    t.append_field_element(v);
                }
            }
            t.append_bytes(&(layer.halves.len() as u64).to_le_bytes());
            for h in &layer.halves {
                for v in [&h.p_lo, &h.p_hi, &h.q_lo, &h.q_hi] {
                    t.append_field_element(v);
                }
            }
        }
        hex(&t.state())
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The heights the byte pin below proves over.
    const PINNED_HEIGHTS: [&[usize]; 7] = [
        &[5, 2, 0, 5, 3],
        &[6, 6],
        &[4, 3, 4],
        &[1],
        &[0],
        &[2, 2],
        &[7, 8],
    ];

    /// The lockstep ladder's bytes over the cubic extension, host trees and
    /// Gruen steps alike: the digest of each proof and of the prover's
    /// transcript after it.
    fn todays_ladder_digests() -> Vec<(String, String)> {
        let lift = |layer: FractionLayer<F>| {
            let up = |m: &Mle<F>| {
                Mle::new(m.evals().iter().map(|v| v.to_extension::<Ext3>()).collect()).unwrap()
            };
            FractionLayer::new(up(&layer.p), up(&layer.q)).unwrap()
        };
        let mut out = Vec::new();
        for (s, heights) in PINNED_HEIGHTS.iter().enumerate() {
            let trees: Vec<Vec<FractionLayer<Ext3>>> = heights
                .iter()
                .enumerate()
                .map(|(t, &k)| {
                    let mut layers = vec![lift(random_layer(k, 1000 * s as u64 + t as u64))];
                    while layers.last().unwrap().num_vars() > 0 {
                        let next = layers.last().unwrap().fold().unwrap();
                        layers.push(next);
                    }
                    layers.reverse();
                    layers
                })
                .collect();
            let refs: Vec<&[FractionLayer<Ext3>]> = trees.iter().map(|t| t.as_slice()).collect();
            let mut prover = DefaultTranscript::<Ext3>::new(b"lockstep-pin");
            let (proof, _) = prove(&refs, GkrLadder::Single, &mut prover).unwrap();
            out.push((proof_digest(&proof), hex(&prover.state())));
        }
        out
    }

    /// The digests of [`todays_ladder_digests`], computed before the
    /// two-layer step existed: (proof, transcript state) per height set.
    const PINNED_DIGESTS: [(&str, &str); 7] = [
        (
            "b9c7d0e954a00b9f1e0de0cffd84e9199349a25a7f98a6442d411e4f9d405f62",
            "da74881a6f34a81953c802abb882f5e15ffc907bbc7132da5fbb723a86b0a254",
        ),
        (
            "6d4dd66000c3edf1fe72debb063db15e722085f9fb952b11f686add8907662d4",
            "6b2a8b3a972d9360d570bf1c57b3a03b443858446ca9e4e40a7e51bf38256894",
        ),
        (
            "caa48e655b44f89b81fe6e4ddb56a07dbe2730fe5ea6dc7f3c75da175dba5d36",
            "edfb7a8ed3718849afd3f74d02191a1a4d17ee58a77293bccf3806ae482c62e5",
        ),
        (
            "486f56bfdb955a30d219d99dfb000f9a119ab2f35f3f8dac82ac47822110b4d6",
            "8c4f99c3d8beb5711ee90afabaa438b68394980400e5adac2c337aebcc1b9116",
        ),
        (
            "134e5a9f4f8ebb347c1d5a9acb8c9a7bde9a8dfcd9c4fd1c1fa124468c42ec6a",
            "c3ce8226ccba4d095005208bd3dbb9ae47aed90c352f4df76d532139ee7dab6d",
        ),
        (
            "80dce121313b14048f27c17b110dedfee54df0c5ad15a5e822a1ab984f222773",
            "60bbad644089d16c7ae81c808b9907eb369c7f8d0b44ef665bb202479521a2c6",
        ),
        (
            "be6fd6c01e06a176fc4b56c0523b0230d255cd9c63105884706c2d857214d730",
            "e61fbf0c0dc386e421196bd7e3f04fca76e05388ead318644c6e4b819dd76dc5",
        ),
    ];

    /// ★ Today's ladder bytes: the proof and the transcript after it, for
    /// every height set above, as they were before the two-layer step
    /// existed.
    #[test]
    fn the_single_ladder_proves_todays_bytes() {
        let got = todays_ladder_digests();
        assert_eq!(got.len(), PINNED_DIGESTS.len());
        for (((proof, state), (proof_pin, state_pin)), heights) in
            got.iter().zip(PINNED_DIGESTS).zip(PINNED_HEIGHTS)
        {
            assert_eq!(proof, proof_pin, "{heights:?}: the proof's bytes");
            assert_eq!(state, state_pin, "{heights:?}: the transcript");
        }
    }

    // ── Two tree layers a step ─────────────────────────────────────────────

    /// The height sets the two-layer tests prove over: even and odd heights,
    /// trees without variables, a single tree of each parity, and a ladder
    /// whose last step has only one-layer relations.
    const DOUBLE_HEIGHTS: [&[usize]; 9] = [
        &[5, 2, 0, 5, 3],
        &[6, 6],
        &[4, 3, 4],
        &[1],
        &[0],
        &[2, 2],
        &[7, 8],
        &[3],
        &[9, 1, 6],
    ];

    /// ★ The schedule: today's ladder step for step under the one-layer form,
    /// and under the two-layer form a step at every other layer, each tree
    /// descending two layers but at the step that reaches an odd height's
    /// input, which takes today's relation.
    #[test]
    fn the_schedule_is_todays_under_single_and_steps_two_layers_under_double() {
        for heights in DOUBLE_HEIGHTS {
            let single = ladder_steps(heights, GkrLadder::Single);
            let height = heights.iter().copied().max().unwrap_or(0);
            assert_eq!(single.len(), height, "{heights:?}");
            for (i, step) in single.iter().enumerate() {
                assert_eq!(step.parent, i);
                assert_eq!(step.degree, 3);
                assert!(!step.draws_second_challenge());
                let trees: Vec<usize> = step.relations.iter().map(|&(t, _)| t).collect();
                assert_eq!(trees, active(heights, i));
                assert!(step.relations.iter().all(|&(_, r)| r == Relation::Single));
            }
            let double = ladder_steps(heights, GkrLadder::Double);
            assert_eq!(double.len(), height.div_ceil(2), "{heights:?}");
            for (j, step) in double.iter().enumerate() {
                assert_eq!(step.parent, 2 * j);
                let trees: Vec<usize> = step.relations.iter().map(|&(t, _)| t).collect();
                assert_eq!(trees, active(heights, 2 * j));
                for &(t, relation) in &step.relations {
                    let want = if heights[t] >= step.parent + 2 {
                        Relation::Double
                    } else {
                        Relation::Single
                    };
                    assert_eq!(relation, want, "{heights:?} step {j} tree {t}");
                }
                let any_double = step.relations.iter().any(|&(_, r)| r == Relation::Double);
                assert_eq!(step.degree, if any_double { 5 } else { 3 });
                assert_eq!(step.draws_second_challenge(), any_double);
            }
        }
        // [5, 2, 0, 5, 3] by hand: steps at 0, 2, 4.
        let steps = ladder_steps(&[5, 2, 0, 5, 3], GkrLadder::Double);
        use Relation::{Double as D, Single as S};
        assert_eq!(
            steps
                .iter()
                .map(|s| (s.parent, s.degree, s.relations.clone()))
                .collect::<Vec<_>>(),
            vec![
                (0, 5, vec![(0, D), (1, D), (3, D), (4, D)]),
                (2, 5, vec![(0, D), (3, D), (4, S)]),
                (4, 3, vec![(0, S), (3, S)]),
            ]
        );
    }

    /// ★ Round trips under the two-layer form at every height set: the
    /// prover and the verifier reach the same claims and transcript, each
    /// claim is its tree's input layer at its point, and the proof has the
    /// schedule's shape (rounds, values a round, halves a step).
    #[test]
    fn double_ladders_round_trip_at_even_and_odd_heights() {
        for heights in DOUBLE_HEIGHTS {
            let (_, proof) = round_trip_as(heights, GkrLadder::Double);
            let schedule = ladder_steps(heights, GkrLadder::Double);
            assert_eq!(proof.layers.len(), schedule.len());
            for (layer, step) in proof.layers.iter().zip(&schedule) {
                assert_eq!(layer.sumcheck.rounds.len(), step.parent);
                assert!(
                    layer
                        .sumcheck
                        .rounds
                        .iter()
                        .all(|r| r.evaluations.len() == step.degree)
                );
                assert_eq!(layer.halves.len(), step.halves());
            }
        }
    }

    /// Over the extension too, host trees and the card's Gruen steps side by
    /// side: the bytes do not depend on where a step runs.
    #[test]
    fn double_gruen_and_side_by_side_steps_prove_the_host_references_bytes() {
        let heights: Vec<usize> = (0..72)
            .map(|t| [6usize, 3, 0, 5, 4, 1, 7, 2][t % 8])
            .collect();
        let trees = trees_of(&heights, 501);
        let refs: Vec<&[FractionLayer<F>]> = trees.iter().map(|t| t.as_slice()).collect();
        let mut host = transcript();
        let (want, want_claims) = prove(&refs, GkrLadder::Double, &mut host).unwrap();
        let outputs = outputs_of(&trees);
        let got = verify(
            &want,
            &outputs,
            &heights,
            GkrLadder::Double,
            &mut transcript(),
        )
        .unwrap();
        assert_eq!(got, want_claims);
        for on in [false, true] {
            force_argue_host_parallel(Some(on));
            for low in [0usize, 1, 2, 6] {
                let gruen: Vec<GruenHostTree<'_>> = trees
                    .iter()
                    .map(|layers| GruenHostTree { layers, low })
                    .collect();
                let dyns: Vec<&dyn LadderTree<F>> =
                    gruen.iter().map(|t| t as &dyn LadderTree<F>).collect();
                let mut prover = transcript();
                let (proof, claims) = prove_trees(&dyns, GkrLadder::Double, &mut prover).unwrap();
                assert!(
                    proof == want && claims == want_claims,
                    "side by side {on}, low {low}: the two-layer ladder moved a byte"
                );
                assert_eq!(prover.state(), host.state());
            }
        }
        force_argue_host_parallel(None);
    }

    /// ★ The two-layer ladder's Gruen steps send the host reference's bytes:
    /// every tree at every step, two-layer steps on the quartic chain and the
    /// odd trees' one-layer steps beside them, from tails of one cell up to
    /// the whole layer, and through the rounds that sum `H(0)` instead of
    /// deriving it — and the steps did run on the two-layer rounds.
    #[test]
    fn double_gruen_steps_send_the_host_references_bytes() {
        let heights = [6usize, 3, 0, 6, 4, 1, 7, 2];
        let trees = trees_of(&heights, 811);
        let refs: Vec<&[FractionLayer<F>]> = trees.iter().map(|t| t.as_slice()).collect();
        let mut host = transcript();
        let (want, want_claims) = prove(&refs, GkrLadder::Double, &mut host).unwrap();
        for direct_h0 in [false, true] {
            crate::gkr_gruen::force_gkr_gruen_direct_h0(direct_h0);
            for low in [0usize, 1, 2, 6] {
                let gruen: Vec<GruenHostTree<'_>> = trees
                    .iter()
                    .map(|layers| GruenHostTree { layers, low })
                    .collect();
                let dyns: Vec<&dyn LadderTree<F>> =
                    gruen.iter().map(|t| t as &dyn LadderTree<F>).collect();
                let before = crate::gkr_gruen::gruen2_layers();
                let mut prover = transcript();
                let (got, claims) = prove_trees(&dyns, GkrLadder::Double, &mut prover).unwrap();
                assert!(
                    got == want && claims == want_claims,
                    "low {low}, direct H(0) {direct_h0}: the two-layer Gruen ladder moved a byte"
                );
                assert_eq!(prover.state(), host.state());
                assert!(
                    crate::gkr_gruen::gruen2_layers() > before,
                    "low {low}: no step ran the two-layer rounds"
                );
            }
        }
        crate::gkr_gruen::force_gkr_gruen_direct_h0(false);
    }

    /// The quartic chain's message is the host's own round: from a step's
    /// `H(1)..H(4)` (and its claim, or `H(0)`), `s(1..5)` equal the
    /// two-layer relation's round evaluations, every round of a layer.
    #[test]
    fn the_quartic_chains_messages_are_the_host_rounds() {
        let grandchild = random_layer(7, 12);
        let point: Vec<FE> = (0..5).map(|i| FE::from(211 + 13 * i as u64)).collect();
        let weights = [FE::from(5u64), FE::from(17u64)];
        let lambda = gruen_lambda(&weights).unwrap();
        for direct_h0 in [false, true] {
            crate::gkr_gruen::force_gkr_gruen_direct_h0(direct_h0);
            let mut host = HostStep::over_double(&grandchild, &point, weights).unwrap();
            let claim = match &host.0 {
                HostRelation::Double(relation) => relation.sum_over_hypercube(),
                HostRelation::Single(_) => unreachable!(),
            };
            let card = HostGruen2::new(&grandchild, &point, lambda, 0);
            let mut step = GruenStep::over_double(
                card,
                &point,
                &weights,
                &(claim * weights[0].inv().unwrap(), FE::zero()),
            )
            .unwrap()
            .unwrap();
            for round in 0..point.len() {
                let want = host.round().unwrap();
                let got = step.round().unwrap();
                assert_eq!(got, want, "round {round}, direct H(0) {direct_h0}");
                let r = FE::from(9_000 + round as u64);
                host.bind(&r).unwrap();
                step.bind(&r).unwrap();
            }
            assert_eq!(step.halves().unwrap(), host.halves().unwrap());
        }
        crate::gkr_gruen::force_gkr_gruen_direct_h0(false);
    }

    /// Two runs of the two-layer ladder are the same bytes.
    #[test]
    fn the_double_ladder_is_deterministic() {
        let (_, a) = round_trip_as(&[7, 8, 3], GkrLadder::Double);
        let (_, b) = round_trip_as(&[7, 8, 3], GkrLadder::Double);
        assert_eq!(a, b);
    }

    /// ★ The two-layer relation at Boolean points is today's two folds
    /// composed, cell by cell, and along any variable it is a quartic: its
    /// fifth differences vanish and its fourth do not.
    #[test]
    fn the_double_relation_is_two_folds_composed_and_quartic() {
        for l in 0..5usize {
            let grandchild = random_layer(l + 2, 77 + l as u64);
            let parent = grandchild.fold().unwrap().fold().unwrap();
            let quarter = 1usize << l;
            let cell = |m: &Mle<F>, b: usize, x: usize| m.evals()[b * quarter + x];
            let values_at = |x: usize| -> Vec<FE> {
                let mut v = Vec::with_capacity(8);
                for b2 in 0..2 {
                    for m in [&grandchild.p, &grandchild.q] {
                        v.push(cell(m, b2, x));
                        v.push(cell(m, 2 + b2, x));
                    }
                }
                v
            };
            for x in 0..quarter {
                let (n, d) = double_fraction(&values_at(x));
                assert_eq!(
                    (n, d),
                    (parent.p.evals()[x], parent.q.evals()[x]),
                    "ℓ {l}, x {x}"
                );
            }
            if l == 0 {
                continue;
            }
            // Along the first variable of x: lo at x, hi at x + quarter/2.
            let lambda = FE::from(0x1234_5678u64);
            for x in 0..quarter / 2 {
                let (lo, hi) = (values_at(x), values_at(x + quarter / 2));
                let at = |t: u64| -> FE {
                    let t = FE::from(t);
                    let v: Vec<FE> = lo.iter().zip(&hi).map(|(a, b)| a + t * (b - a)).collect();
                    let (n, d) = double_fraction(&v);
                    n + lambda * d
                };
                let mut diffs: Vec<FE> = (0..7).map(at).collect();
                let mut orders = Vec::new();
                for _ in 0..5 {
                    diffs = diffs.windows(2).map(|w| w[1] - w[0]).collect();
                    orders.push(diffs.clone());
                }
                assert!(orders[4].iter().all(|d| *d == FE::zero()), "degree ≤ 4");
                assert!(orders[3].iter().any(|d| *d != FE::zero()), "degree 4");
            }
        }
    }

    /// A one-layer relation's message in a degree-5 step: its own cubic,
    /// extended from its running sum, is the host's own evaluation at
    /// `1..=5`.
    #[test]
    fn a_single_relation_extended_to_degree_five_is_its_own_polynomial() {
        let next = random_layer(5, 5);
        let point: Vec<FE> = (0..4).map(|i| FE::from(91 + i as u64)).collect();
        let weights = [FE::from(3u64), FE::from(11u64)];
        let mut step = HostStep::over(&next, &point, weights).unwrap();
        let HostRelation::Single(relation) = &step.0 else {
            unreachable!()
        };
        let mut claim = relation.sum_over_hypercube();
        for round in 0..4 {
            let HostRelation::Single(relation) = &step.0 else {
                unreachable!()
            };
            let direct = sumcheck::round_evaluations(relation, 5, false);
            let own = step.round().unwrap();
            assert_eq!(own.len(), 3);
            let (nodes, message) = extend(&claim, own, 5);
            assert_eq!(message, direct, "round {round}");
            let r = FE::from(1000 + round as u64);
            claim = sumcheck::interpolate(&nodes, &r);
            step.bind(&r).unwrap();
        }
    }

    /// Every layer of each tree, output first.
    type Trees = Vec<Vec<FractionLayer<F>>>;

    /// A round trip's trees, their outputs and the proof.
    fn outputs_and_proof(
        heights: &[usize],
        form: GkrLadder,
    ) -> (Trees, Vec<(FE, FE)>, LockstepProof<F>) {
        let (trees, proof) = round_trip_as(heights, form);
        let outputs = outputs_of(&trees);
        (trees, outputs, proof)
    }

    fn verify_double(
        proof: &LockstepProof<F>,
        outputs: &[(FE, FE)],
        heights: &[usize],
    ) -> Result<Vec<GkrClaim<F>>, Error> {
        verify(
            proof,
            outputs,
            heights,
            GkrLadder::Double,
            &mut transcript(),
        )
    }

    /// ★ Each of a two-layer tree's eight values, one off at a mid step, is
    /// refused by that step's residual; so is each of an odd tree's four in
    /// the mixed step that reaches its input; and so is any value at step 0,
    /// which has no rounds.
    #[test]
    fn each_value_of_a_double_step_one_off_is_refused_at_its_step() {
        // [4, 3, 4]: step 1 (ℓ = 2) holds tree 0 (two layers, halves 0 and 1),
        // tree 1 (odd: one layer, half 2) and tree 2 (halves 3 and 4).
        let heights = [4, 3, 4];
        let (_, outputs, proof) = outputs_and_proof(&heights, GkrLadder::Double);
        let schedule = ladder_steps(&heights, GkrLadder::Double);
        assert_eq!(
            schedule[1].relations,
            vec![
                (0, Relation::Double),
                (1, Relation::Single),
                (2, Relation::Double)
            ]
        );
        for (j, layer) in proof.layers.iter().enumerate() {
            for half in 0..layer.halves.len() {
                for value in 0..4 {
                    let mut bad = proof.clone();
                    let h = &mut bad.layers[j].halves[half];
                    *[&mut h.p_lo, &mut h.p_hi, &mut h.q_lo, &mut h.q_hi][value] += FE::one();
                    assert_eq!(
                        verify_double(&bad, &outputs, &heights).unwrap_err(),
                        Error::LayerRelationMismatch { layer: j },
                        "step {j}, half {half}, value {value}"
                    );
                }
            }
        }
        // Step 0 with no rounds: [2, 2], eight values a tree.
        let heights = [2, 2];
        let (_, outputs, proof) = outputs_and_proof(&heights, GkrLadder::Double);
        assert_eq!(proof.layers.len(), 1);
        assert!(proof.layers[0].sumcheck.rounds.is_empty());
        assert_eq!(proof.layers[0].halves.len(), 4);
        for half in 0..4 {
            for value in 0..4 {
                let mut bad = proof.clone();
                let h = &mut bad.layers[0].halves[half];
                *[&mut h.p_lo, &mut h.p_hi, &mut h.q_lo, &mut h.q_hi][value] += FE::one();
                assert_eq!(
                    verify_double(&bad, &outputs, &heights).unwrap_err(),
                    Error::LayerRelationMismatch { layer: 0 },
                    "half {half}, value {value}"
                );
            }
        }
    }

    /// A ladder's last step when all its trees descend one layer (the tallest
    /// tree odd): each of its values one off is refused at that step, and so is
    /// a round removed from or added to a mixed step, by name.
    #[test]
    fn a_final_one_layer_step_and_a_steps_round_count_are_refused() {
        // [5, 4]: steps at ℓ = 0 (both two layers), 2 (both two layers, the
        // second leaving) and 4 (the first, one layer, degree 3).
        let heights = [5, 4];
        let (_, outputs, proof) = outputs_and_proof(&heights, GkrLadder::Double);
        let schedule = ladder_steps(&heights, GkrLadder::Double);
        assert_eq!(schedule[2].relations, vec![(0, Relation::Single)]);
        assert_eq!(schedule[2].degree, 3);
        for value in 0..4 {
            let mut bad = proof.clone();
            let h = &mut bad.layers[2].halves[0];
            *[&mut h.p_lo, &mut h.p_hi, &mut h.q_lo, &mut h.q_hi][value] += FE::one();
            assert_eq!(
                verify_double(&bad, &outputs, &heights).unwrap_err(),
                Error::LayerRelationMismatch { layer: 2 },
                "value {value}"
            );
        }
        // [4, 3, 4]: step 1 (ℓ = 2) is mixed and runs two rounds.
        let heights = [4, 3, 4];
        let (_, outputs, proof) = outputs_and_proof(&heights, GkrLadder::Double);
        let mut short = proof.clone();
        short.layers[1].sumcheck.rounds.pop();
        assert_eq!(
            verify_double(&short, &outputs, &heights).unwrap_err(),
            Error::RoundCountMismatch {
                expected: 2,
                got: 1
            }
        );
        let mut long = proof.clone();
        let extra = long.layers[1].sumcheck.rounds[0].clone();
        long.layers[1].sumcheck.rounds.push(extra);
        assert_eq!(
            verify_double(&long, &outputs, &heights).unwrap_err(),
            Error::RoundCountMismatch {
                expected: 2,
                got: 3
            }
        );
    }

    /// The residual is load-bearing under the two-layer form too: made inert,
    /// the mid-step tamper above is no longer refused at its step.
    #[test]
    fn the_residual_check_made_inert_lets_a_bad_double_step_through() {
        let heights = [4, 3, 4];
        let (_, outputs, proof) = outputs_and_proof(&heights, GkrLadder::Double);
        let mut bad = proof.clone();
        bad.layers[1].halves[0].q_lo += FE::one();
        assert!(
            verify_with(
                &bad,
                &outputs,
                &heights,
                GkrLadder::Double,
                &mut transcript(),
                false
            )
            .is_ok()
        );
    }

    /// A round value one off in a degree-5 round is refused.
    #[test]
    fn a_degree_five_round_value_one_off_is_refused() {
        let heights = [6, 5];
        let (_, outputs, proof) = outputs_and_proof(&heights, GkrLadder::Double);
        for j in 1..proof.layers.len() {
            for round in [0, proof.layers[j].sumcheck.rounds.len() - 1] {
                for value in 0..5 {
                    let mut bad = proof.clone();
                    bad.layers[j].sumcheck.rounds[round].evaluations[value] += FE::one();
                    assert_eq!(
                        verify_double(&bad, &outputs, &heights).unwrap_err(),
                        Error::LayerRelationMismatch { layer: j },
                        "step {j}, round {round}, value {value}"
                    );
                }
            }
        }
    }

    /// ★ A round of the wrong length is refused by name: 3 or 6 values in a
    /// degree-5 round, 5 in a step whose relations are all one-layer.
    #[test]
    fn a_round_of_the_wrong_length_is_refused() {
        let heights = [5, 4];
        let (_, outputs, proof) = outputs_and_proof(&heights, GkrLadder::Double);
        // Step 1 (ℓ = 2) is degree 5; step 2 (ℓ = 4) has tree 0 alone, odd.
        let schedule = ladder_steps(&heights, GkrLadder::Double);
        assert_eq!((schedule[1].degree, schedule[2].degree), (5, 3));
        for len in [3usize, 6] {
            let mut bad = proof.clone();
            bad.layers[1].sumcheck.rounds[1]
                .evaluations
                .resize(len, FE::one());
            assert_eq!(
                verify_double(&bad, &outputs, &heights).unwrap_err(),
                Error::RoundDegreeMismatch {
                    round: 1,
                    expected: 5,
                    got: len
                }
            );
        }
        let mut bad = proof.clone();
        bad.layers[2].sumcheck.rounds[0]
            .evaluations
            .resize(5, FE::one());
        assert_eq!(
            verify_double(&bad, &outputs, &heights).unwrap_err(),
            Error::RoundDegreeMismatch {
                round: 0,
                expected: 3,
                got: 5
            }
        );
    }

    /// ★ The counts are the schedule's: a two-layer tree sent as one half, a
    /// one-layer tree as two, a tree too many or too few, a step too many or
    /// too few: refused as the wrong shape.
    #[test]
    fn halves_of_the_wrong_count_are_refused() {
        let heights = [4, 3, 4];
        let (_, outputs, proof) = outputs_and_proof(&heights, GkrLadder::Double);
        let shape = Error::ArgueShapeMismatch {
            part: "ladder step",
        };
        // Step 1: tree 0 two halves (0, 1), tree 1 one (2), tree 2 two (3, 4).
        let mut one_half = proof.clone();
        one_half.layers[1].halves.remove(1);
        assert_eq!(
            verify_double(&one_half, &outputs, &heights).unwrap_err(),
            shape
        );
        let mut two_halves = proof.clone();
        let extra = two_halves.layers[1].halves[2].clone();
        two_halves.layers[1].halves.insert(2, extra);
        assert_eq!(
            verify_double(&two_halves, &outputs, &heights).unwrap_err(),
            shape
        );
        let mut missing = proof.clone();
        missing.layers[0].halves.truncate(4);
        assert_eq!(
            verify_double(&missing, &outputs, &heights).unwrap_err(),
            shape
        );
        let mut extra_tree = proof.clone();
        let more = extra_tree.layers[0].halves[..2].to_vec();
        extra_tree.layers[0].halves.extend(more);
        assert_eq!(
            verify_double(&extra_tree, &outputs, &heights).unwrap_err(),
            shape
        );
        let ladder = Error::ArgueShapeMismatch { part: "ladder" };
        let mut short = proof.clone();
        short.layers.pop();
        assert_eq!(
            verify_double(&short, &outputs, &heights).unwrap_err(),
            ladder
        );
        let mut long = proof.clone();
        let last = long.layers[1].clone();
        long.layers.push(last);
        assert_eq!(
            verify_double(&long, &outputs, &heights).unwrap_err(),
            ladder
        );
    }

    /// A μ cancellation: one tree's numerator moved up and another's down by
    /// the same amount, so the unweighted step relation holds; the weights
    /// tell them apart.
    #[test]
    fn values_moved_between_two_trees_are_refused() {
        let heights = [4, 4];
        let (_, outputs, proof) = outputs_and_proof(&heights, GkrLadder::Double);
        let unweighted = |halves: &[TreeHalves<F>]| -> FE {
            let values: Vec<FE> = halves
                .iter()
                .flat_map(|h| [h.p_lo, h.p_hi, h.q_lo, h.q_hi])
                .collect();
            let (n0, _) = double_fraction(&values[..8]);
            let (n1, _) = double_fraction(&values[8..]);
            n0 + n1
        };
        let mut bad = proof.clone();
        let before = unweighted(&bad.layers[1].halves);
        // N_a is linear in each p: moving p_lo of h₀ by δ moves N_a by
        // δ·q_hi(h₀)·D₁. Pick the second tree's δ' to cancel the first's.
        let (h, k) = (&bad.layers[1].halves[..2], &bad.layers[1].halves[2..]);
        let slope = |h: &[TreeHalves<F>]| h[0].q_hi * h[1].q_lo * h[1].q_hi;
        let delta = FE::from(7u64);
        let delta2 = -(delta * slope(h)) * slope(k).inv().unwrap();
        bad.layers[1].halves[0].p_lo += delta;
        bad.layers[1].halves[2].p_lo += delta2;
        assert_eq!(unweighted(&bad.layers[1].halves), before, "the sum held");
        assert_eq!(
            verify_double(&bad, &outputs, &heights).unwrap_err(),
            Error::LayerRelationMismatch { layer: 1 }
        );
    }

    /// ★ A proof under one form is refused under the other, both ways.
    #[test]
    fn a_ladder_under_the_other_form_is_refused() {
        for heights in [&[4usize, 3, 4][..], &[6, 6], &[2, 2], &[5]] {
            let (_, outputs, single) = outputs_and_proof(heights, GkrLadder::Single);
            let (_, _, double) = outputs_and_proof(heights, GkrLadder::Double);
            assert!(
                verify(
                    &single,
                    &outputs,
                    heights,
                    GkrLadder::Double,
                    &mut transcript()
                )
                .is_err(),
                "{heights:?}: a one-layer ladder under the two-layer verifier"
            );
            assert!(
                verify(
                    &double,
                    &outputs,
                    heights,
                    GkrLadder::Single,
                    &mut transcript()
                )
                .is_err(),
                "{heights:?}: a two-layer ladder under the one-layer verifier"
            );
        }
        // A ladder of one step, on its own: [1] is the same proof both ways
        // (one step, one-layer relation, no rounds) and is accepted both ways.
        let (_, outputs, single) = outputs_and_proof(&[1], GkrLadder::Single);
        let (_, _, double) = outputs_and_proof(&[1], GkrLadder::Double);
        assert_eq!(single, double);
        assert!(verify_double(&single, &outputs, &[1]).is_ok());
    }

    /// A ladder proved with one tree [`Faulted`].
    /// A faulted ladder's trees, proof and verdict.
    type Departure = (Trees, LockstepProof<F>, Result<Vec<GkrClaim<F>>, Error>);

    fn faulted(heights: &[usize], target: usize, fault: StepFault) -> Departure {
        faulted_at(heights, target, fault, 2)
    }

    fn faulted_at(heights: &[usize], target: usize, fault: StepFault, layer: usize) -> Departure {
        let trees = trees_of(heights, 61);
        let hosts: Vec<HostLayers<'_, F>> = trees.iter().map(|t| HostLayers(t)).collect();
        let armed = Faulted::at(&hosts[target], fault, layer);
        let dyns: Vec<&dyn LadderTree<F>> = hosts
            .iter()
            .enumerate()
            .map(|(t, h)| {
                if t == target {
                    &armed as &dyn LadderTree<F>
                } else {
                    h as &dyn LadderTree<F>
                }
            })
            .collect();
        let mut prover = transcript();
        let (proof, _) = prove_trees(&dyns, GkrLadder::Double, &mut prover).unwrap();
        let outputs = outputs_of(&trees);
        let verdict = verify_double(&proof, &outputs, heights);
        (trees, proof, verdict)
    }

    /// ★ The relation is symmetric in a tree's two halves and in each half's
    /// `lo`/`hi`, so a prover that swaps them passes its step's residual; the
    /// plane is not symmetric, so the swap is refused one step later — or,
    /// for a tree that leaves at that step, its input claim is false and the
    /// caller's check of it against the trace (the argue's constraint phase)
    /// refuses it.
    #[test]
    fn swapped_halves_pass_their_step_and_are_refused_after_it() {
        for fault in [StepFault::SwapHalves, StepFault::SwapLoHi] {
            // A tree that goes on: refused at step 1, so step 0's residual held.
            let (_, _, verdict) = faulted(&[4, 4], 0, fault);
            assert_eq!(
                verdict.unwrap_err(),
                Error::LayerRelationMismatch { layer: 1 },
                "{fault:?}"
            );
            // A tree that leaves at step 0: the ladder verifies, and its input
            // claim is false while its neighbour's is true.
            let (trees, _, verdict) = faulted(&[2, 4], 0, fault);
            let claims = verdict.unwrap_or_else(|e| panic!("{fault:?}: {e:?}"));
            assert_eq!(
                claims_are_the_inputs(&trees, &claims),
                vec![false, true],
                "{fault:?}"
            );
        }
        // An odd tree that goes on after its swapped two-layer step: refused
        // at the mixed step. [3, 4]: tree 0 takes one layer at step 1.
        let (_, _, verdict) = faulted(&[3, 4], 0, StepFault::SwapLoHi);
        assert_eq!(
            verdict.unwrap_err(),
            Error::LayerRelationMismatch { layer: 1 }
        );
        // Its one half swapped in that mixed step, the one that reaches its
        // input (over layer 3): the step passes, the input claim is false.
        let (trees, _, verdict) = faulted_at(&[3, 4], 0, StepFault::SwapLoHi, 3);
        let claims = verdict.expect("the mixed step's residual is symmetric too");
        assert_eq!(claims_are_the_inputs(&trees, &claims), vec![false, true]);
    }

    /// The plane's coordinates in the other order: a prover that proves the
    /// next layer at `(c₂, c₁, ρ)` is refused there.
    #[test]
    fn a_step_proved_at_the_swapped_plane_is_refused() {
        for heights in [&[4usize, 4][..], &[3, 4], &[6]] {
            let (_, _, verdict) = faulted(heights, 0, StepFault::SwapPlane);
            assert_eq!(
                verdict.unwrap_err(),
                Error::LayerRelationMismatch { layer: 1 },
                "{heights:?}"
            );
        }
    }

    /// A tree whose two-layer step at layer 2 sends forged values: its
    /// output claimed `p + ε`, and `p₀₁ + δ`, `p₁₀ − δ` with
    /// `δ = ε / (q₀₀·q₁₁·(q₁₀ − q₀₁))` so step 0's residual holds for the
    /// false claim. Every other step honest.
    struct Diagonal<'a> {
        layers: &'a [FractionLayer<F>],
        epsilon: FE,
    }

    impl Diagonal<'_> {
        fn forge(&self, halves: &mut [TreeHalves<F>]) {
            let (h0, h1) = (&halves[0], &halves[1]);
            let delta = self.epsilon * (h0.q_lo * h1.q_hi * (h0.q_hi - h1.q_lo)).inv().unwrap();
            // p₀₁ is h₁'s lo, p₁₀ is h₀'s hi.
            halves[1].p_lo += delta;
            halves[0].p_hi = halves[0].p_hi - delta;
        }
    }

    impl LadderTree<F> for Diagonal<'_> {
        fn input_vars(&self) -> usize {
            self.layers.len() - 1
        }

        fn output(&self) -> (FE, FE) {
            let (p, q) = HostLayers(self.layers).output();
            (p + self.epsilon, q)
        }

        fn step(
            &self,
            layer: usize,
            point: &[FE],
            weights: [FE; 2],
            claim: &(FE, FE),
        ) -> Result<Box<dyn TreeStep<F> + '_>, Error> {
            let _ = claim;
            Ok(Box::new(HostStep::over(
                &self.layers[layer],
                point,
                weights,
            )?))
        }

        fn step_double(
            &self,
            layer: usize,
            point: &[FE],
            weights: [FE; 2],
            _claim: &(FE, FE),
        ) -> Result<Box<dyn TreeStep<F> + '_>, Error> {
            let step = HostStep::over_double(&self.layers[layer], point, weights)?;
            if layer != 2 {
                return Ok(Box::new(step));
            }
            struct Forged<'a>(HostStep<F>, &'a Diagonal<'a>);
            impl TreeStep<F> for Forged<'_> {
                fn round(&mut self) -> Result<Vec<FE>, Error> {
                    self.0.round()
                }
                fn bind(&mut self, r: &FE) -> Result<(), Error> {
                    self.0.bind(r)
                }
                fn halves(&self) -> Result<Vec<TreeHalves<F>>, Error> {
                    let mut halves = self.0.halves()?;
                    self.1.forge(&mut halves);
                    Ok(halves)
                }
            }
            Ok(Box::new(Forged(step, self)))
        }
    }

    /// ★ Why `c₂` is drawn on its own: the forgery above keeps the plane on
    /// its diagonal, `c₂ = c₁`, exactly where it was — so a verifier that
    /// took `c₂ := c₁` would carry the true claim on and accept a false
    /// output. With two challenges its next step refuses it.
    #[test]
    fn the_diagonal_forgery_of_a_false_output_is_refused() {
        let layers = layers(random_layer(4, 404));
        let forger = Diagonal {
            layers: &layers,
            epsilon: FE::from(12345u64),
        };
        // The forged values: step 0's residual holds for the false output,
        // and the plane on its diagonal is the truth.
        let honest = HostStep::over_double(&layers[2], &[], [FE::one(), FE::zero()])
            .unwrap()
            .halves()
            .unwrap();
        let mut forged = honest.clone();
        forger.forge(&mut forged);
        let values = |h: &[TreeHalves<F>]| -> Vec<FE> {
            h.iter()
                .flat_map(|h| [h.p_lo, h.p_hi, h.q_lo, h.q_hi])
                .collect()
        };
        let (n, q) = double_fraction(&values(&forged));
        assert_eq!((n, q), forger.output(), "the residual holds for p + ε");
        for c in [FE::from(5u64), FE::from(0xdead_beefu64)] {
            assert_eq!(
                plane(&forged[0], &forged[1], &c, &c),
                plane(&honest[0], &honest[1], &c, &c),
                "on the diagonal the forgery is invisible"
            );
            let other = c + FE::one();
            assert_ne!(
                plane(&forged[0], &forged[1], &c, &other),
                plane(&honest[0], &honest[1], &c, &other),
                "off it, it is not"
            );
        }
        // The ladder: proved by the forger, refused at its next step.
        let mut prover = transcript();
        let (proof, _) = prove_trees(
            &[&forger as &dyn LadderTree<F>],
            GkrLadder::Double,
            &mut prover,
        )
        .unwrap();
        assert_eq!(
            verify_double(&proof, &[forger.output()], &[4]).unwrap_err(),
            Error::LayerRelationMismatch { layer: 1 }
        );
        // A tree that leaves at the forged step: its input claim is false.
        let short = layers_of_two();
        let forger = Diagonal {
            layers: &short,
            epsilon: FE::from(999u64),
        };
        let mut prover = transcript();
        let (proof, _) = prove_trees(
            &[&forger as &dyn LadderTree<F>],
            GkrLadder::Double,
            &mut prover,
        )
        .unwrap();
        let claims = verify_double(&proof, &[forger.output()], &[2]).unwrap();
        assert_eq!(
            claims_are_the_inputs(std::slice::from_ref(&short), &claims),
            vec![false]
        );
    }

    fn layers_of_two() -> Vec<FractionLayer<F>> {
        layers(random_layer(2, 202))
    }

    /// The knob is on by default and off for `0` only.
    #[test]
    fn the_host_parallel_knob_is_on_by_default_and_off_for_zero_only() {
        for on in [None, Some(""), Some("1"), Some(" 1 "), Some("yes")] {
            assert!(host_parallel_from(on), "{on:?}");
        }
        for off in [Some("0"), Some(" 0\n")] {
            assert!(!host_parallel_from(off), "{off:?}");
        }
    }

    /// A probe, never a gate: a ladder of `n` host trees over the cubic
    /// extension of `2^k` input cells (`k` 6 and 10: the many-table group's
    /// host part — its first steps whole, its later ones' tails), one host
    /// step after another against side by side (ms, best of five).
    #[test]
    #[ignore = "a timing probe"]
    fn host_steps_side_by_side_seconds() {
        use math::field::element::FieldElement as FEl;
        type E3 = FEl<Ext3>;
        for k in [6usize, 10] {
            for n in [8usize, 64, 192, 366] {
                let trees: Vec<Vec<FractionLayer<Ext3>>> = (0..n)
                    .map(|t| {
                        let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ (t as u64 * 7919 + k as u64);
                        let mut next = || {
                            s ^= s << 13;
                            s ^= s >> 7;
                            s ^= s << 17;
                            E3::from(s)
                        };
                        let p: Vec<E3> = (0..1usize << k).map(|_| next()).collect();
                        let q: Vec<E3> = (0..1usize << k).map(|_| next()).collect();
                        let mut layers = vec![
                            FractionLayer::new(Mle::new(p).unwrap(), Mle::new(q).unwrap()).unwrap(),
                        ];
                        while layers.last().unwrap().num_vars() > 0 {
                            let next = layers.last().unwrap().fold().unwrap();
                            layers.push(next);
                        }
                        layers.reverse();
                        layers
                    })
                    .collect();
                let host: Vec<HostLayers<'_, Ext3>> = trees.iter().map(|t| HostLayers(t)).collect();
                let dyns: Vec<&dyn LadderTree<Ext3>> =
                    host.iter().map(|t| t as &dyn LadderTree<Ext3>).collect();
                let mut best = [f64::MAX; 2];
                for _ in 0..5 {
                    for (j, on) in [false, true].into_iter().enumerate() {
                        force_argue_host_parallel(Some(on));
                        let mut t = DefaultTranscript::<Ext3>::new(b"probe");
                        let start = std::time::Instant::now();
                        prove_trees(&dyns, GkrLadder::Single, &mut t).unwrap();
                        best[j] = best[j].min(start.elapsed().as_secs_f64());
                    }
                }
                force_argue_host_parallel(None);
                println!(
                    "HOST STEPS PROBE: {n} trees of 2^{k} cells · one after another {:.3} ms · side by side {:.3} ms",
                    best[0] * 1e3,
                    best[1] * 1e3
                );
            }
        }
    }
}
