//! Stage 1 of the argue, on the host: a table's zerocheck batch — its
//! constraint rule beside the bus's two — with the round polynomials computed
//! the stage-1 way, and every message the one today's rounds send
//! (`thoughts/zf/gap2/fix2/D-ARGUE.md` §2).
//!
//! The batch is `eq(r,x)·C(x) + eq(ρ,x)·(γ·N(x) + γ²·D(x))`: `C` the table's
//! roots batched by `β`, `N` and `D` the bus's two sides at the GKR claim's row
//! point `ρ`. Three things change how its rounds are computed, none what they
//! are:
//!
//! - **the bus as one column** (S1-2). Both bus rules are affine in the factors
//!   with constant coefficients, so `γ·N + γ²·D` is one table
//!   `L = a₀ + Σ a_k·f_k`, built once ([`BusColumn`]), and the rounds read one
//!   value where they read every interaction's side;
//! - **the weights pulled out** (S1-4, Gruen). Round `j`'s polynomial is
//!   `E^r_j·eq₁(r_j,X)·A_j(X) + E^ρ_j·eq₁(ρ_j,X)·B_j(X)`, `A_j` the constraint's
//!   sum under `eq(r_{>j}, ·)` and `B_j` the bus column's under `eq(ρ_{>j}, ·)`.
//!   `A_j` is walked at `X ∈ {0, 2..d}` and `A_j(1)` is taken from the claim the
//!   round carries in; `B_j` is linear;
//! - **rounds 0 and 1 in the base field** (S1-5). Before the first challenge every
//!   factor is a base column, so one pass over the 4-row groups `(a, b, x'')`
//!   evaluates `C` on the grid `{0..d}²` — each factor's grid by additions from
//!   its group's four rows — and both messages come out of it. On a trace that
//!   satisfies its AIR the four boolean corners are zero, and are not computed.
//!
//! A round polynomial is a function of the batch, the challenges drawn before it
//! and the round, so an exact algorithm for it sends what today sends (§2.7).
//! Where this one would not — a trace that breaks its AIR, under the corner skip
//! — both proofs are rejected; [`Options::check_corners`] computes the corners
//! anyway and refuses the first that is not zero.
//!
//! Host only. It is the reference the device kernels are checked against, and
//! what each variant does is counted by phase as it runs ([`Counters`]): priced
//! from each walk's step mix, which is what a kernel walking the same program
//! executes, whatever shortcut the host takes to the same values. [`model`]
//! states the same counts in closed form for any height, and a test holds the
//! two together.

use std::collections::BTreeMap;

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};

use crate::{
    Error, challenge_powers,
    eq::eq_evals,
    logup::{self, Interaction},
    mle::Mle,
    program::{self, Builder, Op, Program},
    sumcheck::{RoundProof, SumcheckProof, interpolate},
};

/// A program's steps by kind: what one walk of it executes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mix {
    pub var: u64,
    pub fixed: u64,
    pub add: u64,
    pub sub: u64,
    pub mul: u64,
    pub neg: u64,
}

impl Mix {
    pub fn of<V: IsField>(steps: &[Op<V>]) -> Self {
        let mut mix = Self::default();
        for step in steps {
            match step {
                Op::Var(_) => mix.var += 1,
                Op::Fixed(_) => mix.fixed += 1,
                Op::Add(..) => mix.add += 1,
                Op::Sub(..) => mix.sub += 1,
                Op::Mul(..) => mix.mul += 1,
                Op::Neg(_) => mix.neg += 1,
            }
        }
        mix
    }

    pub fn steps(&self) -> u64 {
        self.var + self.fixed + self.add + self.sub + self.mul + self.neg
    }

    /// Additions, subtractions and negations: one add each.
    pub fn adds(&self) -> u64 {
        self.add + self.sub + self.neg
    }
}

/// Work, by kind.
///
/// A read interpolated to a node `t ≥ 1` — `lo + t·(hi − lo)` — is counted
/// apart from the multiply it is made of, so one count can be priced as
/// today's kernel does it (an extension multiply by `t`) or as S1-1 does (a
/// base one): [`Ops::units`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ops {
    pub base_mul: u64,
    pub base_add: u64,
    pub ext_mul: u64,
    pub ext_add: u64,
    /// A base value times an extension one.
    pub mixed_mul: u64,
    /// `lo + t·(hi − lo)` at a node `t ≥ 1`.
    pub node_reads: u64,
    /// Program steps an interpreter walks: its per-step overhead.
    pub steps: u64,
    /// Factor and table bytes read.
    pub bytes: u64,
}

impl Ops {
    pub fn add(&mut self, other: &Ops) {
        self.base_mul += other.base_mul;
        self.base_add += other.base_add;
        self.ext_mul += other.ext_mul;
        self.ext_add += other.ext_add;
        self.mixed_mul += other.mixed_mul;
        self.node_reads += other.node_reads;
        self.steps += other.steps;
        self.bytes += other.bytes;
    }

    /// In D-ARGUE §2.8's units, where one Goldilocks product or one reduction
    /// is a unit: a base multiply 2, an extension multiply 12 (nine products,
    /// three reductions), a base-by-extension one 6, a base add ½, an
    /// extension add 1½. A read at a node is two extension adds and a multiply
    /// by the node: an extension one today (15), a base one under S1-1
    /// (`int_nodes`, 9). An interpreter adds 4 a step.
    pub fn units(&self, int_nodes: bool, interpreted: bool) -> f64 {
        let read = if int_nodes { 9.0 } else { 15.0 };
        let step = if interpreted { 4.0 } else { 0.0 };
        2.0 * self.base_mul as f64
            + 0.5 * self.base_add as f64
            + 12.0 * self.ext_mul as f64
            + 1.5 * self.ext_add as f64
            + 6.0 * self.mixed_mul as f64
            + read * self.node_reads as f64
            + step * self.steps as f64
    }
}

/// What a variant did, by phase.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    /// `L`, built once (S1-2).
    pub bus_column: Ops,
    /// The `eq` weight tables.
    pub weights: Ops,
    /// Rounds 0 and 1's pass over the base columns (S1-5).
    pub grid: Ops,
    /// Every other round's walks.
    pub rounds: Ops,
    /// The folds between rounds, the grid's double fold among them.
    pub fold: Ops,
    /// The host's part of a round: its message out of the round's sums.
    pub assembly: Ops,
    /// The values at the point of the factors no round folded.
    pub bound: Ops,
}

impl Counters {
    pub fn phases(&self) -> [(&'static str, &Ops); 7] {
        [
            ("bus column", &self.bus_column),
            ("weights", &self.weights),
            ("grid", &self.grid),
            ("rounds", &self.rounds),
            ("fold", &self.fold),
            ("assembly", &self.assembly),
            ("bound", &self.bound),
        ]
    }

    pub fn total(&self) -> Ops {
        let mut total = Ops::default();
        for (_, ops) in self.phases() {
            total.add(ops);
        }
        total
    }

    pub fn units(&self, int_nodes: bool, interpreted: bool) -> f64 {
        self.total().units(int_nodes, interpreted)
    }
}

/// A table's constraint part: the roots' DAG over the factor values in the
/// base field, each root's step, and the factor slot of each root's selector.
///
/// A table's own constraint set reads no challenge, and a constant it reads is
/// a lifted base one unless the AIR says otherwise — so the DAG runs in the
/// base field until the first fold, which is what rounds 0 and 1 run on.
#[derive(Clone, Debug)]
pub struct Constraints<F: IsField> {
    steps: Vec<Op<F>>,
    roots: Vec<u32>,
    selectors: Vec<Option<usize>>,
    degree: usize,
}

impl<F: IsField + 'static> Constraints<F> {
    /// The DAG as the extension holds it (`IrShape::steps_as_ops` and its
    /// roots and selectors), brought down to the base field.
    ///
    /// `degree` is the constraint's — `C`'s, a root's DAG degree plus its
    /// selector, at most — and the batch's round degree is one more.
    ///
    /// Refuses a constant with an extension part ([`Error::NotBaseField`]): the
    /// precondition the base rounds rest on.
    pub fn from_extension<E>(
        steps: &[Op<E>],
        roots: &[u32],
        selectors: &[Option<usize>],
        degree: usize,
    ) -> Result<Self, Error>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        if roots.len() != selectors.len() {
            return Err(Error::VariableCountMismatch {
                expected: roots.len(),
                got: selectors.len(),
            });
        }
        let mut base = Vec::with_capacity(steps.len());
        for (at, step) in steps.iter().enumerate() {
            let operand = |a: u32| -> Result<u32, Error> {
                if a as usize >= at {
                    return Err(Error::UnknownPolynomial {
                        index: a as usize,
                        len: at,
                    });
                }
                Ok(a)
            };
            base.push(match *step {
                Op::Fixed(ref c) => {
                    let limbs = c.clone().to_subfield_vec::<F>();
                    let (first, rest) = limbs.split_first().ok_or(Error::EmptyPolynomial)?;
                    if rest.iter().any(|limb| *limb != FieldElement::<F>::zero()) {
                        return Err(Error::NotBaseField { step: at });
                    }
                    Op::Fixed(first.clone())
                }
                Op::Var(i) => Op::Var(i),
                Op::Add(a, b) => Op::Add(operand(a)?, operand(b)?),
                Op::Sub(a, b) => Op::Sub(operand(a)?, operand(b)?),
                Op::Mul(a, b) => Op::Mul(operand(a)?, operand(b)?),
                Op::Neg(a) => Op::Neg(operand(a)?),
            });
        }
        if let Some(&root) = roots.iter().find(|&&root| root as usize >= steps.len()) {
            return Err(Error::UnknownPolynomial {
                index: root as usize,
                len: steps.len(),
            });
        }
        Ok(Self {
            steps: base,
            roots: roots.to_vec(),
            selectors: selectors.to_vec(),
            degree,
        })
    }

    pub fn steps(&self) -> &[Op<F>] {
        &self.steps
    }

    pub fn roots(&self) -> &[u32] {
        &self.roots
    }

    pub fn selectors(&self) -> &[Option<usize>] {
        &self.selectors
    }

    /// `C`'s degree in the factors, a selector counted.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// The factor slots the constraint part reads, its DAG's and its
    /// selectors', ascending.
    pub fn reads(&self) -> Vec<usize> {
        let mut reads: Vec<usize> = self
            .steps
            .iter()
            .filter_map(|step| match step {
                Op::Var(i) => Some(*i as usize),
                _ => None,
            })
            .chain(self.selectors.iter().flatten().copied())
            .collect();
        reads.sort_unstable();
        reads.dedup();
        reads
    }

    /// Roots with a selector: each costs one more multiply in the combine.
    pub fn selected(&self) -> usize {
        self.selectors.iter().flatten().count()
    }

    /// The DAG with every factor read through `read`, in whichever field the
    /// values are in; each step's value lands in `out`.
    fn run<V>(&self, read: &impl Fn(usize) -> FieldElement<V>, out: &mut Vec<FieldElement<V>>)
    where
        F: IsSubFieldOf<V>,
        V: IsField,
    {
        out.clear();
        for step in &self.steps {
            let value: FieldElement<V> = match *step {
                Op::Fixed(ref c) => c.clone().to_extension::<V>(),
                Op::Var(i) => read(i as usize),
                Op::Add(a, b) => &out[a as usize] + &out[b as usize],
                Op::Sub(a, b) => &out[a as usize] - &out[b as usize],
                Op::Mul(a, b) => &out[a as usize] * &out[b as usize],
                Op::Neg(a) => -&out[a as usize],
            };
            out.push(value);
        }
    }

    /// `C = Σ β_i·sel_i·root_i` at a base-field point: each root and its
    /// selector in the base field, `β_i` the only extension factor.
    fn combine_base<E>(
        &self,
        betas: &[FieldElement<E>],
        read: &impl Fn(usize) -> FieldElement<F>,
        out: &[FieldElement<F>],
    ) -> FieldElement<E>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let mut sum = FieldElement::<E>::zero();
        for ((&root, selector), beta) in self.roots.iter().zip(&self.selectors).zip(betas) {
            let term: FieldElement<F> = match selector {
                Some(slot) => &read(*slot) * &out[root as usize],
                None => out[root as usize].clone(),
            };
            let scaled: FieldElement<E> = &term * beta;
            sum += scaled;
        }
        sum
    }

    /// The same once the factors are folded into the extension.
    fn combine_ext<E>(
        &self,
        betas: &[FieldElement<E>],
        read: &impl Fn(usize) -> FieldElement<E>,
        out: &[FieldElement<E>],
    ) -> FieldElement<E>
    where
        E: IsField,
    {
        let mut sum = FieldElement::<E>::zero();
        for ((&root, selector), beta) in self.roots.iter().zip(&self.selectors).zip(betas) {
            let term: FieldElement<E> = match selector {
                Some(slot) => &read(*slot) * &out[root as usize],
                None => out[root as usize].clone(),
            };
            let scaled: FieldElement<E> = &term * beta;
            sum += scaled;
        }
        sum
    }

    /// Today's zerocheck rule — `weight · Σ β_i·sel_i·root_i` over the
    /// extension, the weight at factor slot `weight` — written the way
    /// `IrShape::program` writes it.
    pub fn zerocheck_program<E>(
        &self,
        betas: &[FieldElement<E>],
        weight: usize,
    ) -> Result<Program<E>, Error>
    where
        F: IsSubFieldOf<E>,
        E: IsField + 'static,
    {
        if betas.len() != self.roots.len() {
            return Err(Error::VariableCountMismatch {
                expected: self.roots.len(),
                got: betas.len(),
            });
        }
        let mut builder = Builder::<E>::new();
        for step in &self.steps {
            match *step {
                Op::Fixed(ref c) => builder.fixed(c.clone().to_extension::<E>()),
                Op::Var(i) => builder.var(i as usize),
                Op::Add(a, b) => builder.add(a, b),
                Op::Sub(a, b) => builder.sub(a, b),
                Op::Mul(a, b) => builder.mul(a, b),
                Op::Neg(a) => builder.neg(a),
            };
        }
        let terms: Vec<(u32, FieldElement<E>)> = self
            .roots
            .iter()
            .zip(betas)
            .zip(&self.selectors)
            .map(|((&root, beta), selector)| {
                let term = match selector {
                    Some(slot) => {
                        let s = builder.var(*slot);
                        builder.mul(root, s)
                    }
                    None => root,
                };
                (term, beta.clone())
            })
            .collect();
        let sum = builder.weighted_sum(&terms);
        let w = builder.var(weight);
        let root = builder.mul(w, sum);
        builder.finish(root)
    }
}

/// The bus's two rules as one column's coefficients (S1-2):
/// `λ_N·N + λ_D·D = a₀ + Σ_k a_k·f_k`.
///
/// `N = Σ_i w_i·num_i(f)` and `D = Σ_i w_i·den_i(f) + padding`, with `w` the
/// interaction weights `eq(interaction point, i)` and `padding` the weight of
/// the `0/1` slots — as [`logup::claim_statements`] writes them — and every
/// side affine in the factors with constant coefficients. So the combination
/// is affine too, and its coefficients are sums over the interactions, made
/// once per table after the batching challenge is drawn.
///
/// It is not a proof value: the verifier checks the batch against its own two
/// rules, which this equals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BusColumn<E: IsField> {
    constant: FieldElement<E>,
    /// `(factor slot, coefficient)`, the slots ascending and distinct.
    terms: Vec<(usize, FieldElement<E>)>,
}

impl<E: IsField + 'static> BusColumn<E>
where
    FieldElement<E>: Send + Sync,
{
    /// `numerator·N + denominator·D` at `interaction_point`: the batch's
    /// weights on the bus's two rules.
    pub fn new(
        interactions: &[Interaction<E>],
        interaction_point: &[FieldElement<E>],
        numerator: &FieldElement<E>,
        denominator: &FieldElement<E>,
    ) -> Result<Self, Error> {
        if interactions.is_empty() {
            return Err(Error::EmptyPolynomial);
        }
        let weights = eq_evals(interaction_point);
        if weights.len() < interactions.len() {
            return Err(Error::VariableCountMismatch {
                expected: interactions.len().next_power_of_two().trailing_zeros() as usize,
                got: interaction_point.len(),
            });
        }
        let padding = weights[interactions.len()..]
            .iter()
            .fold(FieldElement::<E>::zero(), |acc, w| acc + w);
        let mut constant = denominator * &padding;
        let mut coefficients: BTreeMap<usize, FieldElement<E>> = BTreeMap::new();
        for (interaction, w) in interactions.iter().zip(&weights) {
            for (side, scale) in [
                (&interaction.numerator, w * numerator),
                (&interaction.denominator, w * denominator),
            ] {
                constant += &scale * side.constant_term();
                for (slot, coefficient) in side.terms() {
                    let term = &scale * coefficient;
                    *coefficients.entry(*slot).or_insert_with(FieldElement::zero) += term;
                }
            }
        }
        Ok(Self {
            constant,
            terms: coefficients.into_iter().collect(),
        })
    }

    /// A column given by its coefficients — a test's forgery, for one.
    pub fn from_parts(constant: FieldElement<E>, terms: Vec<(usize, FieldElement<E>)>) -> Self {
        Self { constant, terms }
    }

    pub fn constant(&self) -> &FieldElement<E> {
        &self.constant
    }

    pub fn terms(&self) -> &[(usize, FieldElement<E>)] {
        &self.terms
    }

    /// `L` at one point, from every factor's value there.
    pub fn evaluate(&self, values: &[FieldElement<E>]) -> FieldElement<E> {
        self.terms
            .iter()
            .fold(self.constant.clone(), |acc, (slot, a)| {
                acc + a * &values[*slot]
            })
    }

    /// `L`'s table over the cube, from the base factors: a base-by-extension
    /// multiply-add per term a row.
    pub fn table<F>(&self, factors: &[Mle<F>], ops: &mut Ops) -> Result<Vec<FieldElement<E>>, Error>
    where
        F: IsSubFieldOf<E> + 'static,
    {
        let size = factors
            .first()
            .map(Mle::len)
            .ok_or(Error::EmptyPolynomial)?;
        for (slot, _) in &self.terms {
            let factor = factors.get(*slot).ok_or(Error::UnknownPolynomial {
                index: *slot,
                len: factors.len(),
            })?;
            if factor.len() != size {
                return Err(Error::VariableCountMismatch {
                    expected: size.trailing_zeros() as usize,
                    got: factor.num_vars(),
                });
            }
        }
        let mut table = vec![self.constant.clone(); size];
        for (slot, a) in &self.terms {
            for (cell, value) in table.iter_mut().zip(factors[*slot].evals()) {
                let term: FieldElement<E> = value * a;
                *cell += term;
            }
        }
        column_ops(ops, size as u64, self.terms.len() as u64);
        Ok(table)
    }
}

/// How a zerocheck batch's rounds are computed. Each variant adds a technique
/// to the one before it; every one sends the same messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// Today's rounds: the whole batch program walked at every node `1..=D`
    /// of every index, the weights two factors among the rest, every factor
    /// folded every round.
    Today,
    /// S1-2: the bus's two rules replaced by the column `L`; otherwise today's
    /// rounds, the factors nothing reads left out of them.
    Bus,
    /// S1-2 and S1-4: no weight in the walk; `C` at `{0, 2..d}`, `A_j(1)` from
    /// the claim, `B_j` from `L`'s two halves.
    Gruen,
    /// S1-2, S1-4 and S1-5: rounds 0 and 1 from one base-field pass over the
    /// 4-row groups; S1-4 from round 2.
    Grid,
}

/// A variant and its switches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    pub variant: Variant,
    /// [`Variant::Today`] and [`Variant::Bus`]: walk the batch program on
    /// demand ([`Program::on_demand`]), as the head does for a big batch. The
    /// values are the same; the reads, and so the counts, are not.
    pub on_demand: bool,
    /// [`Variant::Grid`]: the grid's four boolean corners are taken as zero —
    /// what they are on a trace that satisfies its AIR.
    pub skip_corners: bool,
    /// [`Variant::Grid`]: compute the corners anyway and refuse the first that
    /// is not zero ([`Error::ConstraintViolated`]) — XCHECK's self-check.
    pub check_corners: bool,
    /// ⛔ A FAULT, for a negative control: `L`'s term at this index off by one.
    pub bus_fault: Option<usize>,
}

impl Options {
    pub fn new(variant: Variant) -> Self {
        Self {
            variant,
            on_demand: false,
            skip_corners: false,
            check_corners: false,
            bus_fault: None,
        }
    }
}

/// What the batch is about, beside the factors: the constraint part, its
/// batching powers, the bus, and the two points.
pub struct Statement<'a, F: IsField, E: IsField> {
    pub constraints: &'a Constraints<F>,
    /// `β^i`, one per root.
    pub betas: &'a [FieldElement<E>],
    pub interactions: &'a [Interaction<E>],
    /// The GKR input-layer claim's point: the interaction bits, then the rows'
    /// — whose tail is `ρ`.
    pub claim_point: &'a [FieldElement<E>],
    /// The zerocheck's point.
    pub r: &'a [FieldElement<E>],
}

/// The rounds, the point they drew, and every factor's value there in slot
/// order.
pub type Proved<E> = (SumcheckProof<E>, Vec<FieldElement<E>>, Vec<FieldElement<E>>);

/// The batch's sumcheck the way `options` says, from the claims on.
///
/// Absorbs `claims` — the zerocheck's, then the bus's `p̃` and `q̃` — draws the
/// batching challenge and runs every round, as [`crate::batch::prove`] does
/// over today's `Batched`; the proof, the point and the transcript after it
/// are that one's. `factors` are the table's, in the base field and slot
/// order: the committed views and the public selectors.
///
/// The claims must be the batch's true sums. Today's rounds never read them;
/// S1-4 takes `A_j(1)` from the claim each round carries in, which is the true
/// one exactly when the first is.
pub fn prove<F, E, T>(
    statement: &Statement<'_, F, E>,
    factors: &[Mle<F>],
    claims: &[FieldElement<E>; 3],
    options: Options,
    transcript: &mut T,
    counters: &mut Counters,
) -> Result<Proved<E>, Error>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
    FieldElement<E>: Send + Sync,
    T: IsTranscript<E>,
{
    let num_vars = factors
        .first()
        .map(Mle::num_vars)
        .ok_or(Error::EmptyPolynomial)?;
    for factor in factors {
        if factor.num_vars() != num_vars {
            return Err(Error::VariableCountMismatch {
                expected: num_vars,
                got: factor.num_vars(),
            });
        }
    }
    if statement.r.len() != num_vars {
        return Err(Error::VariableCountMismatch {
            expected: num_vars,
            got: statement.r.len(),
        });
    }
    let expected = logup::input_layer_vars(statement.interactions.len(), num_vars);
    if statement.claim_point.len() != expected {
        return Err(Error::VariableCountMismatch {
            expected,
            got: statement.claim_point.len(),
        });
    }
    if statement.betas.len() != statement.constraints.roots.len() {
        return Err(Error::VariableCountMismatch {
            expected: statement.constraints.roots.len(),
            got: statement.betas.len(),
        });
    }
    if let Some(&slot) = statement
        .constraints
        .reads()
        .iter()
        .find(|&&slot| slot >= factors.len())
    {
        return Err(Error::UnknownPolynomial {
            index: slot,
            len: factors.len(),
        });
    }

    for claim in claims {
        transcript.append_field_element(claim);
    }
    let lambdas = challenge_powers(&transcript.sample_field_element(), 3);
    let claim = lambdas
        .iter()
        .zip(claims)
        .fold(FieldElement::<E>::zero(), |acc, (l, c)| acc + l * c);
    let degree = (statement.constraints.degree + 1).max(2);
    let (interaction_point, _) = statement
        .claim_point
        .split_at(statement.claim_point.len() - num_vars);
    let bus = || -> Result<BusColumn<E>, Error> {
        let mut bus = BusColumn::new(
            statement.interactions,
            interaction_point,
            &lambdas[1],
            &lambdas[2],
        )?;
        if let Some(term) = options.bus_fault {
            let len = bus.terms.len();
            let (_, a) = bus
                .terms
                .get_mut(term)
                .ok_or(Error::UnknownPolynomial { index: term, len })?;
            *a += FieldElement::<E>::one();
        }
        Ok(bus)
    };

    match options.variant {
        Variant::Today => {
            let program = batch_program(statement, num_vars, factors.len(), &lambdas)?;
            let rho = &statement.claim_point[interaction_point.len()..];
            let extra = vec![
                eq_table(statement.r, &mut counters.weights),
                eq_table(rho, &mut counters.weights),
            ];
            walk(
                &program,
                options.on_demand,
                factors,
                extra,
                true,
                degree,
                transcript,
                counters,
            )
        }
        Variant::Bus => {
            let bus = bus()?;
            let program = column_program(statement, factors.len())?;
            let rho = &statement.claim_point[interaction_point.len()..];
            let extra = vec![
                eq_table(statement.r, &mut counters.weights),
                eq_table(rho, &mut counters.weights),
                bus.table(factors, &mut counters.bus_column)?,
            ];
            walk(
                &program,
                options.on_demand,
                factors,
                extra,
                false,
                degree,
                transcript,
                counters,
            )
        }
        Variant::Gruen | Variant::Grid => {
            let bus = bus()?;
            gruen(
                statement, factors, &bus, claim, degree, options, transcript, counters,
            )
        }
    }
}

/// Today's batch over `factors` trace factors, the weights at the two slots
/// after them: the zerocheck rule and the bus's two, combined with `lambdas`
/// as `batch::prove_resident` combines them.
pub fn batch_program<F, E>(
    statement: &Statement<'_, F, E>,
    num_vars: usize,
    factors: usize,
    lambdas: &[FieldElement<E>],
) -> Result<Program<E>, Error>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    let (weight_r, weight_z) = (factors, factors + 1);
    let zerocheck = statement
        .constraints
        .zerocheck_program(statement.betas, weight_r)?;
    let rules = logup::claim_statements(
        statement.interactions,
        statement.claim_point,
        num_vars,
        weight_z,
    )?;
    program::combine(
        &[
            &zerocheck,
            rules.numerator.program().ok_or(Error::EmptyPolynomial)?,
            rules.denominator.program().ok_or(Error::EmptyPolynomial)?,
        ],
        lambdas,
    )
}

/// S1-2's batch over `factors` trace factors: the weights at the two slots
/// after them, `L` at the next.
pub fn column_program<F, E>(
    statement: &Statement<'_, F, E>,
    factors: usize,
) -> Result<Program<E>, Error>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    let zerocheck = statement
        .constraints
        .zerocheck_program(statement.betas, factors)?;
    bus_batch(&zerocheck, factors + 1, factors + 2)
}

/// S1-2's batch: today's zerocheck rule plus the row weight times `L`, at
/// factor slots `weight` and `column`.
pub fn bus_batch<E: IsField + 'static>(
    zerocheck: &Program<E>,
    weight: usize,
    column: usize,
) -> Result<Program<E>, Error> {
    let mut builder = Builder::<E>::new();
    let w = builder.var(weight);
    let l = builder.var(column);
    let root = builder.mul(w, l);
    let bus = builder.finish(root)?;
    program::combine(
        &[zerocheck, &bus],
        &[FieldElement::one(), FieldElement::one()],
    )
}

/// `eq(point, ·)`'s table, counted as [`eq_evals`] builds it.
fn eq_table<E: IsField>(point: &[FieldElement<E>], ops: &mut Ops) -> Vec<FieldElement<E>>
where
    FieldElement<E>: Send + Sync,
{
    eq_ops(ops, point.len() as u64);
    eq_evals(point)
}

/// `eq(a_{>j+1}, ·)` from `eq(a_{>j}, ·)`: the two halves added, since
/// `eq₁(a_{j+1}, 0) + eq₁(a_{j+1}, 1) = 1`.
fn halve<E: IsField>(table: &mut Vec<FieldElement<E>>, ops: &mut Ops) {
    let half = table.len() / 2;
    let (lo, hi) = table.split_at_mut(half);
    for (l, h) in lo.iter_mut().zip(hi.iter()) {
        *l = &*l + h;
    }
    table.truncate(half);
    ops.ext_add += half as u64;
}

/// `eq₁(a, t) = (1 − a)(1 − t) + a·t`.
fn eq1<E: IsField>(a: &FieldElement<E>, t: &FieldElement<E>) -> FieldElement<E> {
    let one = FieldElement::<E>::one();
    (&one - a) * (&one - t) + a * t
}

/// A table folded on its first variable: `lo + s·(hi − lo)`.
fn fold<E: IsField>(table: &mut Vec<FieldElement<E>>, s: &FieldElement<E>, ops: &mut Ops) {
    let half = table.len() / 2;
    let (lo, hi) = table.split_at_mut(half);
    for (l, h) in lo.iter_mut().zip(hi.iter()) {
        *l = &*l + s * &(h - &*l);
    }
    table.truncate(half);
    fold_ops(ops, 1, half as u64);
}

/// A factor's value at `point` from its base table: the first fold lifts, the
/// rest stay up — `Mle::evaluate_in`'s host path.
fn bound_at<F, E>(
    factor: &Mle<F>,
    point: &[FieldElement<E>],
    ops: &mut Ops,
) -> Result<FieldElement<E>, Error>
where
    F: IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    bound_ops(ops, 1, factor.len() as u64);
    factor.evaluate_in_on_host(point)
}

// ── what each piece of work costs ────────────────────────────────────────────
//
// Every count below is linear in its last argument, so the executed paths call
// them once per round with that round's size, and [`model`] once per phase with
// the sizes summed over its rounds. What the two can disagree on is the sums,
// which is what `the_model_is_what_the_reference_counts` holds.

/// `eq(point, ·)` over `len` variables: two multiplies a cell, a subtraction a
/// variable.
fn eq_ops(ops: &mut Ops, len: u64) {
    ops.ext_mul += 2 * ((1u64 << len) - 1);
    ops.ext_add += len;
}

/// `L` over `cells` rows of `terms` terms: a base-by-extension multiply-add a
/// term, a base read.
fn column_ops(ops: &mut Ops, cells: u64, terms: u64) {
    ops.mixed_mul += cells * terms;
    ops.ext_add += cells * terms;
    ops.bytes += 8 * cells * terms;
}

/// `tables` extension tables folded over `indices` indices.
fn fold_ops(ops: &mut Ops, tables: u64, indices: u64) {
    ops.ext_mul += tables * indices;
    ops.ext_add += 2 * tables * indices;
    ops.bytes += 48 * tables * indices;
}

/// `count` base factors of `cells` rows evaluated at a point, as
/// [`bound_at`] does.
fn bound_ops(ops: &mut Ops, count: u64, cells: u64) {
    let half = cells / 2;
    if half == 0 {
        return;
    }
    ops.base_add += count * half;
    ops.mixed_mul += count * half;
    ops.ext_add += count * half;
    ops.bytes += count * 16 * half;
    let rest = half - 1;
    ops.ext_mul += count * rest;
    ops.ext_add += count * 2 * rest;
    ops.bytes += count * 48 * rest;
}

/// `walks` walks of a batch program at nodes `t ≥ 1` — every read
/// interpolated, the sum accumulated — as today's kernel walks them.
fn walk_ops(ops: &mut Ops, mix: &Mix, walks: u64) {
    ops.node_reads += walks * mix.var;
    ops.bytes += walks * 48 * mix.var;
    ops.ext_add += walks * (mix.adds() + 1);
    ops.ext_mul += walks * mix.mul;
    ops.steps += walks * mix.steps();
}

/// S1-4 over `indices` indices: `C` walked at the `d` nodes `0, 2..=d` (node
/// 0's reads plain, the rest interpolated), combined and weighted; `B`'s two
/// sums.
fn gruen_ops(ops: &mut Ops, mix: &Mix, roots: u64, selected: u64, d: u64, indices: u64) {
    let walks = indices * d;
    let interpolated = indices * (d - 1);
    ops.node_reads += interpolated * mix.var;
    ops.bytes += indices * 24 * mix.var + interpolated * 48 * mix.var;
    ops.ext_add += walks * (mix.adds() + roots + 1) + 2 * indices;
    ops.ext_mul += walks * (mix.mul + roots + selected + 1) + 2 * indices;
    ops.steps += walks * (mix.steps() + 2 * roots + selected);
    // The two weights, and `L`'s two halves.
    ops.bytes += indices * (24 + 24 + 48);
}

/// `rounds` messages out of their parts ([`message`]) — the host's side of a
/// round. Approximate, and far below anything a round walks: `A(D)` priced as
/// `(d+1)²` multiplies, each node's `g(t)` as eight.
fn message_ops(ops: &mut Ops, d: u64, rounds: u64) {
    ops.ext_mul += rounds * ((d + 1) * (d + 1) + 8 * (d + 1));
    ops.ext_add += rounds * (6 * (d + 1) + 1);
}

/// `A(1)` from the claim, `rounds` times.
fn derive_ops(ops: &mut Ops, rounds: u64) {
    ops.ext_mul += 6 * rounds;
    ops.ext_add += 5 * rounds;
}

/// Round 1's `A₁(Y)` for each of the `d + 1` values of `Y`: a Lagrange form
/// in `s₀` over the grid's first variable.
fn lagrange_ops(ops: &mut Ops, d: u64) {
    ops.ext_mul += (d + 1) * (d + 1) * (d + 1);
}

/// The double fold over `groups` groups: each of `reads` base factors'
/// four rows into one, and `L`'s.
fn double_fold_ops(ops: &mut Ops, groups: u64, reads: u64) {
    ops.ext_mul += 4;
    ops.mixed_mul += 4 * groups * reads;
    ops.ext_add += 3 * groups * (reads + 1);
    ops.bytes += 32 * groups * reads + 96 * groups;
    ops.ext_mul += 4 * groups;
}

/// The grid pass over `groups` groups: each read factor's grid by additions,
/// the walks at the points computed, the sums at the points kept, and `U`.
#[allow(clippy::too_many_arguments)]
fn grid_ops(
    ops: &mut Ops,
    groups: u64,
    reads: u64,
    mix: &Mix,
    roots: u64,
    selected: u64,
    d: u64,
    options: Options,
) {
    let side = d + 1;
    let computed_corners = !options.skip_corners || options.check_corners;
    let computed = side * side - if computed_corners { 0 } else { 4 };
    let kept = side * side - if options.skip_corners { 4 } else { 0 };
    ops.base_add += groups * reads * d * (d + 3);
    ops.bytes += groups * reads * 32;
    let walks = groups * computed;
    ops.base_add += walks * mix.adds();
    ops.base_mul += walks * (mix.mul + selected);
    ops.mixed_mul += walks * roots;
    ops.ext_add += walks * roots;
    ops.steps += walks * (mix.steps() + 2 * roots + selected);
    ops.ext_mul += groups * kept;
    ops.ext_add += groups * kept;
    // `U`, and the two weights.
    ops.ext_mul += 4 * groups;
    ops.ext_add += 4 * groups;
    ops.bytes += groups * (96 + 48);
}

/// Today's round loop over `program`: at every index, every node `1..=degree`,
/// every factor interpolated and the program walked. A table is folded when
/// `fold_all` or when the program reads it; a trace factor none folded is
/// bound at the end.
///
/// `extra` are the tables past the trace's factors (the weights, `L`), in
/// their slots.
#[allow(clippy::too_many_arguments)]
fn walk<F, E, T>(
    program: &Program<E>,
    on_demand: bool,
    factors: &[Mle<F>],
    extra: Vec<Vec<FieldElement<E>>>,
    fold_all: bool,
    degree: usize,
    transcript: &mut T,
    counters: &mut Counters,
) -> Result<Proved<E>, Error>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
    T: IsTranscript<E>,
{
    let walked = if on_demand {
        program.on_demand()
    } else {
        program.clone()
    };
    let mix = Mix::of(walked.steps());
    let width = factors.len() + extra.len();
    let mut read = vec![false; width];
    for step in walked.steps() {
        if let Op::Var(i) = step {
            let slot = *i as usize;
            if slot >= width {
                return Err(Error::UnknownPolynomial {
                    index: slot,
                    len: width,
                });
            }
            read[slot] = true;
        }
    }
    let mut extra = extra.into_iter();
    let mut tables: Vec<Option<Vec<FieldElement<E>>>> = (0..width)
        .map(|slot| match factors.get(slot) {
            Some(factor) if fold_all || read[slot] => Some(
                factor
                    .evals()
                    .iter()
                    .map(|v| v.clone().to_extension::<E>())
                    .collect(),
            ),
            Some(_) => None,
            None => extra.next(),
        })
        .collect();
    let num_vars = factors.first().map(Mle::num_vars).unwrap_or(0);
    let nodes: Vec<FieldElement<E>> = (1..=degree as u64).map(FieldElement::from).collect();
    let mut values = vec![FieldElement::<E>::zero(); width];
    let mut scratch = Vec::new();
    let mut rounds = Vec::with_capacity(num_vars);
    let mut point = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        let len = tables.iter().flatten().map(Vec::len).next().unwrap_or(1);
        let half = len / 2;
        let mut sums = vec![FieldElement::<E>::zero(); degree];
        for i in 0..half {
            for (sum, t) in sums.iter_mut().zip(&nodes) {
                for (value, table) in values.iter_mut().zip(&tables) {
                    if let Some(table) = table {
                        let lo = &table[i];
                        *value = lo + t * &(&table[i + half] - lo);
                    }
                }
                *sum += walked.eval(&values, &mut scratch);
            }
        }
        walk_ops(&mut counters.rounds, &mix, (half * degree) as u64);
        for e in &sums {
            transcript.append_field_element(e);
        }
        let s = transcript.sample_field_element();
        for table in tables.iter_mut().flatten() {
            fold(table, &s, &mut counters.fold);
        }
        rounds.push(RoundProof { evaluations: sums });
        point.push(s);
    }
    let bound = factors
        .iter()
        .zip(&tables)
        .map(|(factor, table)| match table {
            Some(table) => Ok(table[0].clone()),
            None => bound_at(factor, &point, &mut counters.bound),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((SumcheckProof { rounds }, point, bound))
}

/// S1-4's rounds, from round 0 or — under [`Variant::Grid`], two variables up —
/// from round 2, after S1-5's pass has sent rounds 0 and 1.
#[allow(clippy::too_many_arguments)]
fn gruen<F, E, T>(
    statement: &Statement<'_, F, E>,
    factors: &[Mle<F>],
    bus: &BusColumn<E>,
    claim: FieldElement<E>,
    degree: usize,
    options: Options,
    transcript: &mut T,
    counters: &mut Counters,
) -> Result<Proved<E>, Error>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
    FieldElement<E>: Send + Sync,
    T: IsTranscript<E>,
{
    let constraints = statement.constraints;
    let num_vars = factors.first().map(Mle::num_vars).unwrap_or(0);
    let r = statement.r;
    let rho = &statement.claim_point[statement.claim_point.len() - num_vars..];
    // `C`'s degree as the rounds see it: a constant constraint is a line.
    let d = constraints.degree.max(1);
    debug_assert_eq!(
        degree,
        d + 1,
        "the batch's degree is the constraint's plus one"
    );
    let reads = constraints.reads();
    let mix = Mix::of(constraints.steps());
    let (roots, selected) = (
        constraints.roots.len() as u64,
        constraints.selected() as u64,
    );
    let one = FieldElement::<E>::one();
    let big = FieldElement::<E>::from(degree as u64);

    let mut l = bus.table(factors, &mut counters.bus_column)?;
    let mut claim = claim;
    let (mut e_r, mut e_rho) = (one.clone(), one.clone());
    let mut rounds = Vec::with_capacity(num_vars);
    let mut point = Vec::with_capacity(num_vars);
    // A round's message sent and absorbed, its challenge drawn, and the claim
    // the next round carries — what the verifier computes from the message.
    let mut send = |g: Vec<FieldElement<E>>,
                    claim: &mut FieldElement<E>,
                    rounds: &mut Vec<RoundProof<E>>,
                    point: &mut Vec<FieldElement<E>>| {
        for e in &g {
            transcript.append_field_element(e);
        }
        let s = transcript.sample_field_element();
        let mut all = Vec::with_capacity(g.len() + 1);
        all.push(&*claim - &g[0]);
        all.extend(g.iter().cloned());
        *claim = interpolate(&all, &s);
        rounds.push(RoundProof { evaluations: g });
        point.push(s.clone());
        s
    };

    // The tables of the factors the constraint part reads, by slot, as the
    // rounds have folded them; the rest are bound at the end.
    let mut tables: Vec<Option<Vec<FieldElement<E>>>> = vec![None; factors.len()];
    let (mut eq_r, mut eq_rho, start) = if options.variant == Variant::Grid && num_vars >= 2 {
        let eq_r = eq_table(&r[2..], &mut counters.weights);
        let eq_rho = eq_table(&rho[2..], &mut counters.weights);
        let (t, u) = grid_pass(
            statement, factors, &reads, &l, &eq_r, &eq_rho, d, options, counters,
        )?;
        let node = |k: usize| FieldElement::<E>::from(k as u64);
        // Round 0: `A₀(X) = Σ_c eq₁(r₁, c)·T(X, c)`, `B₀(b) = Σ_c eq₁(ρ₁, c)·U(b, c)`.
        let a0: Vec<FieldElement<E>> = (0..=d)
            .map(|a| (&one - &r[1]) * &t[a][0] + &r[1] * &t[a][1])
            .collect();
        let b0: [FieldElement<E>; 2] =
            core::array::from_fn(|b| (&one - &rho[1]) * &u[b][0] + &rho[1] * &u[b][1]);
        let g0 = message(&r[0], &rho[0], &e_r, &e_rho, &a0, &b0, &big);
        message_ops(&mut counters.assembly, d as u64, 1);
        let s0 = send(g0, &mut claim, &mut rounds, &mut point);
        e_r = &e_r * eq1(&r[0], &s0);
        e_rho = &e_rho * eq1(&rho[0], &s0);
        // Round 1: `A₁(Y) = Σ_a ℓ_a(s₀)·T(a, Y)` — Lagrange in the first
        // variable, exact since `C` has degree at most `d` in it — and
        // `B₁(c) = (1 − s₀)·U(0, c) + s₀·U(1, c)`.
        let a1: Vec<FieldElement<E>> = (0..=d)
            .map(|b| {
                let column: Vec<FieldElement<E>> = (0..=d).map(|a| t[a][b].clone()).collect();
                interpolate(&column, &s0)
            })
            .collect();
        lagrange_ops(&mut counters.assembly, d as u64);
        let b1: [FieldElement<E>; 2] =
            core::array::from_fn(|c| (&one - &s0) * &u[0][c] + &s0 * &u[1][c]);
        let g1 = message(&r[1], &rho[1], &e_r, &e_rho, &a1, &b1, &big);
        message_ops(&mut counters.assembly, d as u64, 1);
        let s1 = send(g1, &mut claim, &mut rounds, &mut point);
        e_r = &e_r * eq1(&r[1], &s1);
        e_rho = &e_rho * eq1(&rho[1], &s1);
        // Each group's four rows to one, base to extension: the two folds by
        // `s₀` and `s₁` as one, `Σ_{a,b} eq₁(s₀, a)·eq₁(s₁, b)·f(a, b, x'')`.
        let w: [[FieldElement<E>; 2]; 2] = core::array::from_fn(|a| {
            core::array::from_fn(|b| eq1(&s0, &node(a)) * eq1(&s1, &node(b)))
        });
        let quarter = l.len() / 4;
        let at = |a: usize, b: usize, x: usize| a * 2 * quarter + b * quarter + x;
        for &slot in &reads {
            let f = factors[slot].evals();
            tables[slot] = Some(
                (0..quarter)
                    .map(|x| {
                        let mut v = FieldElement::<E>::zero();
                        for (a, row) in w.iter().enumerate() {
                            for (b, wab) in row.iter().enumerate() {
                                let term: FieldElement<E> = &f[at(a, b, x)] * wab;
                                v += term;
                            }
                        }
                        v
                    })
                    .collect(),
            );
        }
        l = (0..quarter)
            .map(|x| {
                let mut v = FieldElement::<E>::zero();
                for (a, row) in w.iter().enumerate() {
                    for (b, wab) in row.iter().enumerate() {
                        v += &l[at(a, b, x)] * wab;
                    }
                }
                v
            })
            .collect();
        double_fold_ops(&mut counters.fold, quarter as u64, reads.len() as u64);
        let (mut eq_r, mut eq_rho) = (eq_r, eq_rho);
        if num_vars > 2 {
            halve(&mut eq_r, &mut counters.weights);
            halve(&mut eq_rho, &mut counters.weights);
        }
        (eq_r, eq_rho, 2)
    } else {
        for &slot in &reads {
            tables[slot] = Some(
                factors[slot]
                    .evals()
                    .iter()
                    .map(|v| v.clone().to_extension::<E>())
                    .collect(),
            );
        }
        let from = 1.min(num_vars);
        let eq_r = eq_table(&r[from..], &mut counters.weights);
        let eq_rho = eq_table(&rho[from..], &mut counters.weights);
        (eq_r, eq_rho, 0)
    };

    let mut values = vec![FieldElement::<E>::zero(); factors.len()];
    let mut scratch: Vec<FieldElement<E>> = Vec::new();
    // `A` is walked at `0` and `2..=d`; `A(1)` comes from the claim.
    let walked: Vec<usize> = core::iter::once(0).chain(2..=d).collect();
    for j in start..num_vars {
        let half = l.len() / 2;
        // Every read factor at index `i`, extended to the node `k` — by a base
        // multiply, the node being an integer (S1-1).
        let at_node = |values: &mut Vec<FieldElement<E>>,
                       tables: &[Option<Vec<FieldElement<E>>>],
                       i: usize,
                       k: usize| {
            let node = FieldElement::<F>::from(k as u64);
            for &slot in &reads {
                let table = tables[slot].as_ref().expect("a read factor is folded");
                let lo = &table[i];
                values[slot] = if k == 0 {
                    lo.clone()
                } else {
                    let delta = &table[i + half] - lo;
                    let step: FieldElement<E> = &node * &delta;
                    lo + step
                };
            }
        };
        let mut a = vec![FieldElement::<E>::zero(); d + 1];
        let mut b = [FieldElement::<E>::zero(), FieldElement::<E>::zero()];
        for i in 0..half {
            for &k in &walked {
                at_node(&mut values, &tables, i, k);
                let read = |slot: usize| values[slot].clone();
                constraints.run(&read, &mut scratch);
                let c = constraints.combine_ext(statement.betas, &read, &scratch);
                a[k] += &eq_r[i] * &c;
            }
            b[0] += &eq_rho[i] * &l[i];
            b[1] += &eq_rho[i] * &l[i + half];
        }
        gruen_ops(
            &mut counters.rounds,
            &mix,
            roots,
            selected,
            d as u64,
            half as u64,
        );

        // `A(1)` from the claim the round carries in, `g(0) + g(1)`:
        // `E^r·((1 − r_j)·A(0) + r_j·A(1)) + E^ρ·((1 − ρ_j)·B(0) + ρ_j·B(1))`.
        let (r_j, rho_j) = (&r[j], &rho[j]);
        let divisor = &e_r * r_j;
        let bus_part = &e_rho * ((&one - rho_j) * &b[0] + rho_j * &b[1]);
        derive_ops(&mut counters.assembly, 1);
        a[1] = match divisor.inv() {
            Ok(inverse) => (&claim - &e_r * (&one - r_j) * &a[0] - bus_part) * inverse,
            // `E^r_j·r_j` vanished, so the claim says nothing about `A(1)` and
            // it is walked. A random point makes this `(j + 1)/|F|`.
            Err(_) => {
                let mut direct = FieldElement::<E>::zero();
                for (i, weight) in eq_r.iter().enumerate().take(half) {
                    at_node(&mut values, &tables, i, 1);
                    let read = |slot: usize| values[slot].clone();
                    constraints.run(&read, &mut scratch);
                    let c = constraints.combine_ext(statement.betas, &read, &scratch);
                    direct += weight * &c;
                }
                walk_ops(&mut counters.rounds, &mix, half as u64);
                direct
            }
        };
        let g = message(r_j, rho_j, &e_r, &e_rho, &a, &b, &big);
        message_ops(&mut counters.assembly, d as u64, 1);
        let s = send(g, &mut claim, &mut rounds, &mut point);
        e_r = &e_r * eq1(r_j, &s);
        e_rho = &e_rho * eq1(rho_j, &s);
        for table in tables.iter_mut().flatten() {
            fold(table, &s, &mut counters.fold);
        }
        fold(&mut l, &s, &mut counters.fold);
        if j + 1 < num_vars {
            halve(&mut eq_r, &mut counters.weights);
            halve(&mut eq_rho, &mut counters.weights);
        }
    }

    let bound = factors
        .iter()
        .zip(&tables)
        .map(|(factor, table)| match table {
            Some(table) => Ok(table[0].clone()),
            None => bound_at(factor, &point, &mut counters.bound),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((SumcheckProof { rounds }, point, bound))
}

/// A round's message from its parts: `A` at `0..=d` (the constraint's, degree
/// `d`), `B` at `0` and `1` (the bus column's, linear), and the weights'
/// products so far. `g(t) = E^r·eq₁(r_j, t)·A(t) + E^ρ·eq₁(ρ_j, t)·B(t)` for
/// `t = 1..=D`, `D = d + 1`, with `A(D)` from the Lagrange form over `0..=d`.
fn message<E: IsField>(
    r_j: &FieldElement<E>,
    rho_j: &FieldElement<E>,
    e_r: &FieldElement<E>,
    e_rho: &FieldElement<E>,
    a: &[FieldElement<E>],
    b: &[FieldElement<E>; 2],
    big: &FieldElement<E>,
) -> Vec<FieldElement<E>> {
    let d = a.len() - 1;
    let a_big = interpolate(a, big);
    let slope = &b[1] - &b[0];
    (1..=d + 1)
        .map(|t| {
            let node = FieldElement::<E>::from(t as u64);
            let at = if t <= d { &a[t] } else { &a_big };
            let bt = &b[0] + &node * &slope;
            e_r * eq1(r_j, &node) * at + e_rho * eq1(rho_j, &node) * bt
        })
        .collect()
}

/// S1-5's pass: `T(a, b) = Σ_{x''} eq(r_{>1}, x'')·C(a, b, x'')` over the grid
/// `{0..d}²` and `U(b, c) = Σ_{x''} eq(ρ_{>1}, x'')·L(b, c, x'')` over
/// `{0,1}²`, group `x''`'s rows being `a·N/2 + b·N/4 + x''`.
///
/// Each read factor's grid comes from its group's four rows by additions — the
/// first variable extended from the two rows of each `b`, then the second
/// from each `a`'s two — and `C` is walked in the base field at every point
/// that is not a corner, or at every point when the corners are not skipped
/// or are checked.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn grid_pass<F, E>(
    statement: &Statement<'_, F, E>,
    factors: &[Mle<F>],
    reads: &[usize],
    l: &[FieldElement<E>],
    eq_r: &[FieldElement<E>],
    eq_rho: &[FieldElement<E>],
    d: usize,
    options: Options,
    counters: &mut Counters,
) -> Result<(Vec<Vec<FieldElement<E>>>, [[FieldElement<E>; 2]; 2]), Error>
where
    F: IsField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    let constraints = statement.constraints;
    let quarter = l.len() / 4;
    let side = d + 1;
    let mut t = vec![vec![FieldElement::<E>::zero(); side]; side];
    let mut u: [[FieldElement<E>; 2]; 2] = Default::default();
    let mut grid: Vec<Vec<FieldElement<F>>> = vec![Vec::new(); factors.len()];
    for &slot in reads {
        grid[slot] = vec![FieldElement::<F>::zero(); side * side];
    }
    let mut scratch: Vec<FieldElement<F>> = Vec::new();
    let row = |a: usize, b: usize, x: usize| a * 2 * quarter + b * quarter + x;
    let computed_corners = !options.skip_corners || options.check_corners;
    for x in 0..quarter {
        for &slot in reads {
            let f = factors[slot].evals();
            let g = &mut grid[slot];
            for b in 0..2 {
                g[b] = f[row(0, b, x)].clone();
                g[side + b] = f[row(1, b, x)].clone();
                let delta = &g[side + b] - &g[b];
                for a in 2..side {
                    g[a * side + b] = &g[(a - 1) * side + b] + &delta;
                }
            }
            for a in 0..side {
                let delta = &g[a * side + 1] - &g[a * side];
                for b in 2..side {
                    g[a * side + b] = &g[a * side + b - 1] + &delta;
                }
            }
        }
        for a in 0..side {
            for b in 0..side {
                let corner = a < 2 && b < 2;
                if corner && !computed_corners {
                    continue;
                }
                let read = |slot: usize| grid[slot][a * side + b].clone();
                constraints.run::<F>(&read, &mut scratch);
                let c: FieldElement<E> = constraints.combine_base(statement.betas, &read, &scratch);
                if corner {
                    if options.check_corners && c != FieldElement::<E>::zero() {
                        return Err(Error::ConstraintViolated { row: row(a, b, x) });
                    }
                    if options.skip_corners {
                        continue;
                    }
                }
                t[a][b] += &eq_r[x] * &c;
            }
        }
        for (b, lane) in u.iter_mut().enumerate() {
            for (c, cell) in lane.iter_mut().enumerate() {
                *cell += &eq_rho[x] * &l[row(b, c, x)];
            }
        }
    }
    grid_ops(
        &mut counters.grid,
        quarter as u64,
        reads.len() as u64,
        &Mix::of(constraints.steps()),
        constraints.roots.len() as u64,
        constraints.selected() as u64,
        d as u64,
        options,
    );
    Ok((t, u))
}

/// What the op model needs to know about a table's batch.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub num_vars: usize,
    /// `C`'s degree, a selector counted.
    pub constraint_degree: usize,
    /// The table's factors: its committed views and public selectors.
    pub factors: usize,
    /// The factors the constraint part reads ([`Constraints::reads`]).
    pub constraint_reads: usize,
    /// The trace factors S1-2's batch program reads.
    pub bus_batch_reads: usize,
    /// `L`'s terms: the distinct factors the bus reads.
    pub bus_terms: usize,
    pub roots: usize,
    /// Roots with a selector.
    pub selected: usize,
    /// The constraint DAG's steps.
    pub dag: Mix,
    /// Today's batch program, as the rounds walk it.
    pub batch: Mix,
    /// S1-2's batch program ([`bus_batch`]), as the rounds walk it.
    pub bus_batch: Mix,
}

impl Shape {
    /// `statement`'s shape over `factors` trace factors, at the height of its
    /// zerocheck point: the batch programs as [`prove`] builds them with
    /// `lambdas`, walked on demand when `on_demand`.
    ///
    /// A program's step mix depends on the challenges only where one is zero
    /// or one — a coefficient `Builder::weighted_sum` drops, a product
    /// `Program::simplify` folds — which a random challenge makes negligible.
    pub fn of<F, E>(
        statement: &Statement<'_, F, E>,
        factors: usize,
        lambdas: &[FieldElement<E>],
        on_demand: bool,
    ) -> Result<Self, Error>
    where
        F: IsField + IsSubFieldOf<E> + 'static,
        E: IsField + 'static,
        FieldElement<E>: Send + Sync,
    {
        if lambdas.len() != 3 {
            return Err(Error::VariableCountMismatch {
                expected: 3,
                got: lambdas.len(),
            });
        }
        let num_vars = statement.r.len();
        let walked = |program: Program<E>| {
            if on_demand {
                program.on_demand()
            } else {
                program
            }
        };
        let batch = walked(batch_program(statement, num_vars, factors, lambdas)?);
        let column = walked(column_program(statement, factors)?);
        let mut read: Vec<usize> = column
            .steps()
            .iter()
            .filter_map(|step| match step {
                Op::Var(i) if (*i as usize) < factors => Some(*i as usize),
                _ => None,
            })
            .collect();
        read.sort_unstable();
        read.dedup();
        let split = statement.claim_point.len().saturating_sub(num_vars);
        let bus = BusColumn::new(
            statement.interactions,
            &statement.claim_point[..split],
            &lambdas[1],
            &lambdas[2],
        )?;
        let constraints = statement.constraints;
        Ok(Self {
            num_vars,
            constraint_degree: constraints.degree,
            factors,
            constraint_reads: constraints.reads().len(),
            bus_batch_reads: read.len(),
            bus_terms: bus.terms.len(),
            roots: constraints.roots.len(),
            selected: constraints.selected(),
            dag: Mix::of(&constraints.steps),
            batch: Mix::of(batch.steps()),
            bus_batch: Mix::of(column.steps()),
        })
    }
}

/// Each variant's counts at `shape`'s height, in closed form: what [`prove`]
/// counts as it runs, for a table the host cannot afford to run.
///
/// A round `j` works on `half_j = 2^{n−1−j}` indices, so rounds `from..n` on
/// `2^{n−from} − 1` of them. Assumes no `E^r_j·r_j` vanishes, which a random
/// point makes negligible.
pub fn model(shape: &Shape, options: Options) -> Counters {
    let n = shape.num_vars as u64;
    let cells = 1u64 << n;
    // Indices over rounds `from..n`.
    let over = |from: u64| (cells >> from.min(n)).saturating_sub(1);
    let d = shape.constraint_degree.max(1) as u64;
    let degree = d + 1;
    let (roots, selected) = (shape.roots as u64, shape.selected as u64);
    let factors = shape.factors as u64;
    let reads = shape.constraint_reads as u64;
    let mut c = Counters::default();
    match options.variant {
        Variant::Today => {
            eq_ops(&mut c.weights, n);
            eq_ops(&mut c.weights, n);
            walk_ops(&mut c.rounds, &shape.batch, degree * over(0));
            fold_ops(&mut c.fold, factors + 2, over(0));
        }
        Variant::Bus => {
            column_ops(&mut c.bus_column, cells, shape.bus_terms as u64);
            eq_ops(&mut c.weights, n);
            eq_ops(&mut c.weights, n);
            walk_ops(&mut c.rounds, &shape.bus_batch, degree * over(0));
            fold_ops(&mut c.fold, shape.bus_batch_reads as u64 + 3, over(0));
            bound_ops(&mut c.bound, factors - shape.bus_batch_reads as u64, cells);
        }
        Variant::Gruen | Variant::Grid => {
            column_ops(&mut c.bus_column, cells, shape.bus_terms as u64);
            let start = if options.variant == Variant::Grid && n >= 2 {
                let groups = cells / 4;
                eq_ops(&mut c.weights, n - 2);
                eq_ops(&mut c.weights, n - 2);
                grid_ops(
                    &mut c.grid,
                    groups,
                    reads,
                    &shape.dag,
                    roots,
                    selected,
                    d,
                    options,
                );
                message_ops(&mut c.assembly, d, 2);
                lagrange_ops(&mut c.assembly, d);
                double_fold_ops(&mut c.fold, groups, reads);
                2
            } else {
                eq_ops(&mut c.weights, n.saturating_sub(1));
                eq_ops(&mut c.weights, n.saturating_sub(1));
                0
            };
            // The halvings: each weight is built for the first round that
            // walks it, and halved into every round's after that.
            c.weights.ext_add += 2 * if start == 2 { over(2) } else { over(1) };
            gruen_ops(&mut c.rounds, &shape.dag, roots, selected, d, over(start));
            let rounds = n.saturating_sub(start);
            derive_ops(&mut c.assembly, rounds);
            message_ops(&mut c.assembly, d, rounds);
            fold_ops(&mut c.fold, reads + 1, over(start));
            bound_ops(&mut c.bound, factors - reads, cells);
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        batch::{self, Rule},
        eq::eq_mle,
        logup::Affine,
        selector::Selector,
    };
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::{
        extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3,
        goldilocks::GoldilocksField as Gl,
    };

    type FB = FieldElement<Gl>;
    type FX = FieldElement<Ext3>;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn base(&mut self) -> FB {
            FB::from(self.next())
        }

        fn ext(&mut self) -> FX {
            FX::new([self.base(), self.base(), self.base()])
        }

        fn exts(&mut self, n: usize) -> Vec<FX> {
            (0..n).map(|_| self.ext()).collect()
        }
    }

    // The table's factors: four columns, a column's next-row view, the
    // transition's selector, and a column only the bus reads.
    const A: usize = 0;
    const B: usize = 1;
    const C: usize = 2;
    const D: usize = 3;
    const A_NEXT: usize = 4;
    const SELECTOR: usize = 5;
    const BUS_ONLY: usize = 6;
    const WIDTH: usize = 7;

    struct Table {
        constraints: Constraints<Gl>,
        interactions: Vec<Interaction<Ext3>>,
        betas: Vec<FX>,
    }

    /// A table whose constraint part has degree `degree`. At 1: `c = a + b`.
    /// From 2: `a_next = a + 1` under the selector (off on the last row), and
    /// `c = a·b + a`, beside a root that is zero whatever the trace (a
    /// negation and a constant in it). At 3: `d = a·b·c` too. Three bus
    /// interactions, so the fourth slot is padding.
    fn table(degree: usize, rng: &mut Rng) -> Table {
        let mut b = Builder::<Ext3>::new();
        let a = b.var(A);
        let bv = b.var(B);
        let c = b.var(C);
        let d = b.var(D);
        let next = b.var(A_NEXT);
        let one = b.fixed(FX::one());
        let seven = b.fixed(FX::from(7u64));
        let mut roots = Vec::new();
        let mut selectors = Vec::new();
        if degree == 1 {
            let sum = b.add(a, bv);
            roots.push(b.sub(c, sum));
            selectors.push(None);
        } else {
            let step = b.sub(next, a);
            roots.push(b.sub(step, one));
            selectors.push(Some(SELECTOR));
            let ab = b.mul(a, bv);
            let root = b.sub(c, ab);
            roots.push(b.sub(root, a));
            selectors.push(None);
            let negated = b.neg(ab);
            let zero = b.add(negated, ab);
            roots.push(b.mul(zero, seven));
            selectors.push(None);
            if degree == 3 {
                let abc = b.mul(ab, c);
                roots.push(b.sub(d, abc));
                selectors.push(None);
            }
        }
        let program = b.finish_verbatim(roots[0]).unwrap();
        let constraints =
            Constraints::<Gl>::from_extension(program.steps(), &roots, &selectors, degree).unwrap();
        let (alpha, z) = (rng.ext(), rng.ext());
        let interactions = vec![
            Interaction::new(
                Affine::factor(BUS_ONLY),
                Affine::new(vec![(A, alpha), (B, alpha * alpha)], z),
            ),
            Interaction::new(
                Affine::new(vec![(SELECTOR, -FX::one())], FX::zero()),
                Affine::new(vec![(C, alpha)], z + FX::from(3u64)),
            ),
            Interaction::new(
                Affine::constant(FX::from(2u64)),
                Affine::new(vec![(A, alpha * alpha * alpha), (D, FX::from(5u64))], z),
            ),
        ];
        let beta = rng.ext();
        let mut betas = vec![FX::one()];
        for _ in 1..roots.len() {
            let next = betas[betas.len() - 1] * beta;
            betas.push(next);
        }
        Table {
            constraints,
            interactions,
            betas,
        }
    }

    /// The table's factors over `2^n` rows: a trace that satisfies it, or
    /// random columns. The next-row view and the selector are what they are
    /// either way.
    fn factors(n: usize, degree: usize, satisfying: bool, rng: &mut Rng) -> Vec<Mle<Gl>> {
        let size = 1usize << n;
        let random = |rng: &mut Rng| -> Vec<FB> { (0..size).map(|_| rng.base()).collect() };
        let a: Vec<FB> = if satisfying {
            (0..size as u64).map(|i| FB::from(i + 7)).collect()
        } else {
            random(rng)
        };
        let b = random(rng);
        let c: Vec<FB> = match (satisfying, degree) {
            (true, 1) => a.iter().zip(&b).map(|(x, y)| x + y).collect(),
            (true, _) => a.iter().zip(&b).map(|(x, y)| x * y + x).collect(),
            (false, _) => random(rng),
        };
        let d: Vec<FB> = if satisfying && degree == 3 {
            (0..size).map(|i| a[i] * b[i] * c[i]).collect()
        } else {
            random(rng)
        };
        let next: Vec<FB> = (0..size).map(|i| a[(i + 1) % size]).collect();
        let bus_only = random(rng);
        let selector = Selector::except_last(1).table::<Gl>(n).unwrap();
        vec![
            Mle::new(a).unwrap(),
            Mle::new(b).unwrap(),
            Mle::new(c).unwrap(),
            Mle::new(d).unwrap(),
            Mle::new(next).unwrap(),
            selector,
            Mle::new(bus_only).unwrap(),
        ]
    }

    fn transcript() -> DefaultTranscript<Ext3> {
        DefaultTranscript::<Ext3>::new(b"fused-argue")
    }

    fn lift(factors: &[Mle<Gl>]) -> Vec<Mle<Ext3>> {
        factors
            .iter()
            .map(|f| {
                Mle::new(f.evals().iter().map(|v| v.to_extension::<Ext3>()).collect()).unwrap()
            })
            .collect()
    }

    /// Today's three rules and their factor list, as `multilinear_table::prove`
    /// builds them: the trace's factors lifted, then `eq(r, ·)` and `eq(ρ, ·)`.
    fn todays(
        table: &Table,
        factors: &[Mle<Gl>],
        r: &[FX],
        claim_point: &[FX],
    ) -> (Vec<Mle<Ext3>>, Vec<Rule<'static, Ext3>>) {
        let n = factors[0].num_vars();
        let zerocheck = Rule::compiled(
            table.constraints.degree() + 1,
            table
                .constraints
                .zerocheck_program(&table.betas, WIDTH)
                .unwrap(),
        );
        let interactions: &'static [Interaction<Ext3>] =
            Box::leak(table.interactions.clone().into_boxed_slice());
        let bus = logup::claim_statements(interactions, claim_point, n, WIDTH + 1).unwrap();
        let mut polys = lift(factors);
        polys.push(eq_mle(r).unwrap());
        polys.push(eq_mle(&bus.row_point).unwrap());
        (polys, vec![zerocheck, bus.numerator, bus.denominator])
    }

    /// Each rule's sum over the cube: the claims an honest batch carries.
    fn true_claims(polys: &[Mle<Ext3>], rules: &[Rule<'_, Ext3>]) -> [FX; 3] {
        let size = polys[0].len();
        core::array::from_fn(|k| {
            (0..size).fold(FX::zero(), |acc, i| {
                let values: Vec<FX> = polys.iter().map(|p| p.evals()[i]).collect();
                acc + rules[k].apply(&values)
            })
        })
    }

    struct Case {
        table: Table,
        factors: Vec<Mle<Gl>>,
        r: Vec<FX>,
        claim_point: Vec<FX>,
    }

    fn case(n: usize, degree: usize, satisfying: bool, seed: u64) -> Case {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1);
        let table = table(degree, &mut rng);
        let factors = factors(n, degree, satisfying, &mut rng);
        let r = rng.exts(n);
        let claim_point = rng.exts(logup::input_layer_vars(table.interactions.len(), n));
        Case {
            table,
            factors,
            r,
            claim_point,
        }
    }

    impl Case {
        fn statement(&self) -> Statement<'_, Gl, Ext3> {
            Statement {
                constraints: &self.table.constraints,
                betas: &self.table.betas,
                interactions: &self.table.interactions,
                claim_point: &self.claim_point,
                r: &self.r,
            }
        }

        /// The claims: the true sums, or with the zerocheck's forced to zero —
        /// what a prover claims whatever its trace.
        fn claims(&self, zerocheck_zero: bool) -> [FX; 3] {
            let (polys, rules) = todays(&self.table, &self.factors, &self.r, &self.claim_point);
            let mut claims = true_claims(&polys, &rules);
            if zerocheck_zero {
                claims[0] = FX::zero();
            }
            claims
        }

        /// Today's proof through `batch::prove`, and the transcript after it.
        fn today(
            &self,
            claims: &[FX; 3],
        ) -> (SumcheckProof<Ext3>, Vec<FX>, DefaultTranscript<Ext3>) {
            let (polys, rules) = todays(&self.table, &self.factors, &self.r, &self.claim_point);
            let mut t = transcript();
            let (proof, point) = batch::prove(polys, rules, claims, &mut t).unwrap();
            (proof, point, t)
        }

        fn fused(
            &self,
            claims: &[FX; 3],
            options: Options,
        ) -> (
            Result<Proved<Ext3>, Error>,
            DefaultTranscript<Ext3>,
            Counters,
        ) {
            let mut t = transcript();
            let mut counters = Counters::default();
            let proved = prove(
                &self.statement(),
                &self.factors,
                claims,
                options,
                &mut t,
                &mut counters,
            );
            (proved, t, counters)
        }

        /// The first round at which `options` sends another message than
        /// today's, or `None`; with every other part of the result checked
        /// when there is none.
        fn parts_at(&self, claims: &[FX; 3], options: Options) -> Option<usize> {
            let (proof, point, mut t_today) = self.today(claims);
            let (proved, mut t, _) = self.fused(claims, options);
            let (fused, fused_point, bound) = proved.unwrap();
            let parted = proof
                .rounds
                .iter()
                .zip(&fused.rounds)
                .position(|(a, b)| a.evaluations != b.evaluations);
            if parted.is_some() {
                return parted;
            }
            assert_eq!(fused.rounds.len(), proof.rounds.len(), "{options:?}");
            assert_eq!(fused_point, point, "{options:?}: the point");
            assert_eq!(t.state(), t_today.state(), "{options:?}: the transcript");
            assert_eq!(
                t.sample_field_element(),
                t_today.sample_field_element(),
                "{options:?}: the next challenge"
            );
            for (slot, (factor, value)) in self.factors.iter().zip(&bound).enumerate() {
                assert_eq!(
                    *value,
                    factor.evaluate_in(&point).unwrap(),
                    "{options:?}: factor {slot} at the point"
                );
            }
            None
        }
    }

    fn every(satisfying: bool) -> Vec<Options> {
        let mut all = Vec::new();
        for variant in [Variant::Today, Variant::Bus] {
            for on_demand in [false, true] {
                all.push(Options {
                    on_demand,
                    ..Options::new(variant)
                });
            }
        }
        all.push(Options::new(Variant::Gruen));
        all.push(Options::new(Variant::Grid));
        if satisfying {
            all.push(Options {
                skip_corners: true,
                ..Options::new(Variant::Grid)
            });
            all.push(Options {
                skip_corners: true,
                check_corners: true,
                ..Options::new(Variant::Grid)
            });
        }
        all
    }

    /// ★ Every variant sends today's messages — the same rounds, point,
    /// transcript and next challenge as `batch::prove` over today's batch, and
    /// every factor's value at the point — at every height from none to 2^7, at
    /// each constraint degree, on random columns (the corners computed) and on
    /// a trace that satisfies the table (the corners skipped, and checked).
    #[test]
    fn every_variant_sends_todays_messages() {
        for degree in 1..=3 {
            for n in 0..=7 {
                for satisfying in [false, true] {
                    let case = case(n, degree, satisfying, (degree * 100 + n) as u64);
                    let claims = case.claims(false);
                    if satisfying {
                        assert_eq!(claims[0], FX::zero(), "a satisfying trace's zerocheck");
                    }
                    for options in every(satisfying) {
                        assert_eq!(
                            case.parts_at(&claims, options),
                            None,
                            "degree {degree}, n {n}, satisfying {satisfying}: {options:?}"
                        );
                    }
                }
            }
        }
    }

    /// ⛔ The corner skip is only what today sends on a trace that satisfies
    /// its AIR: on random columns the skipped corners are not zero and the
    /// first message moves, and the corner check refuses at the first row.
    #[test]
    fn the_corner_skip_parts_from_today_on_random_columns() {
        for degree in 1..=3 {
            let case = case(5, degree, false, 7 + degree as u64);
            let claims = case.claims(false);
            let skip = Options {
                skip_corners: true,
                ..Options::new(Variant::Grid)
            };
            assert_eq!(case.parts_at(&claims, skip), Some(0), "degree {degree}");
            let checked = Options {
                check_corners: true,
                ..skip
            };
            assert_eq!(
                case.fused(&claims, checked).0.unwrap_err(),
                Error::ConstraintViolated { row: 0 },
                "degree {degree}"
            );
        }
    }

    /// ⛔ NEGATIVE CONTROL: one cell of a satisfying trace flipped. The prover
    /// still claims zero for the zerocheck; today's rounds send the broken
    /// trace's sums and the stage-1 ones do not, from round 0 — both proofs
    /// rejected, with different bytes (D-ARGUE §2.7 (a)). The corner check
    /// names the row.
    #[test]
    fn a_flipped_cell_parts_the_messages_and_the_check_names_its_row() {
        for degree in 1..=3 {
            let mut case = case(5, degree, true, 40 + degree as u64);
            let row = 5;
            let mut c = case.factors[C].evals().to_vec();
            c[row] += FB::one();
            case.factors[C] = Mle::new(c).unwrap();
            let claims = case.claims(true);
            for options in [
                Options::new(Variant::Gruen),
                Options {
                    skip_corners: true,
                    ..Options::new(Variant::Grid)
                },
            ] {
                assert_eq!(
                    case.parts_at(&claims, options),
                    Some(0),
                    "degree {degree}: {options:?}"
                );
            }
            let checked = Options {
                skip_corners: true,
                check_corners: true,
                ..Options::new(Variant::Grid)
            };
            assert_eq!(
                case.fused(&claims, checked).0.unwrap_err(),
                Error::ConstraintViolated { row },
                "degree {degree}"
            );
        }
    }

    /// ⛔ NEGATIVE CONTROL: one coefficient of `L` wrong. Every variant that
    /// reads `L` parts from today at round 0.
    #[test]
    fn a_wrong_bus_coefficient_parts_the_messages() {
        let case = case(6, 3, true, 77);
        let claims = case.claims(false);
        for fault in 0..3 {
            for options in [
                Options::new(Variant::Bus),
                Options::new(Variant::Gruen),
                Options {
                    skip_corners: true,
                    ..Options::new(Variant::Grid)
                },
            ] {
                let faulty = Options {
                    bus_fault: Some(fault),
                    ..options
                };
                assert_eq!(case.parts_at(&claims, faulty), Some(0), "{faulty:?}");
            }
        }
    }

    /// ★ The model is what the reference counts, phase by phase, for every
    /// variant at every height it runs — which is what lets the model price a
    /// table at a height the host cannot run.
    #[test]
    fn the_model_is_what_the_reference_counts() {
        for degree in 1..=3 {
            for n in [0, 1, 2, 3, 4, 6] {
                for satisfying in [false, true] {
                    let case = case(n, degree, satisfying, 900 + (degree * 10 + n) as u64);
                    let claims = case.claims(false);
                    let mut t = transcript();
                    for claim in &claims {
                        t.append_field_element(claim);
                    }
                    let lambdas = challenge_powers(&t.sample_field_element(), 3);
                    for options in every(satisfying) {
                        let shape =
                            Shape::of(&case.statement(), WIDTH, &lambdas, options.on_demand)
                                .unwrap();
                        let (proved, _, counters) = case.fused(&claims, options);
                        proved.unwrap();
                        assert_eq!(
                            model(&shape, options),
                            counters,
                            "degree {degree}, n {n}: {options:?}"
                        );
                    }
                }
            }
        }
    }

    /// `L` is the bus's two rules, weighted: at a random point it is
    /// `λ_N·N + λ_D·D`, the rules read with a row weight of one.
    #[test]
    fn the_bus_column_is_the_two_rules() {
        let mut rng = Rng(0x5eed);
        let table = table(3, &mut rng);
        let n = 4;
        let claim_point = rng.exts(logup::input_layer_vars(table.interactions.len(), n));
        let (numerator, denominator) = (rng.ext(), rng.ext());
        let split = claim_point.len() - n;
        let column = BusColumn::new(
            &table.interactions,
            &claim_point[..split],
            &numerator,
            &denominator,
        )
        .unwrap();
        let rules = logup::claim_statements(&table.interactions, &claim_point, n, WIDTH).unwrap();
        for _ in 0..4 {
            let mut values = rng.exts(WIDTH);
            values.push(FX::one());
            let expected = numerator * rules.numerator.apply(&values)
                + denominator * rules.denominator.apply(&values);
            assert_eq!(column.evaluate(&values), expected);
        }
        // Every factor the bus reads is one term.
        let slots: Vec<usize> = column.terms().iter().map(|(slot, _)| *slot).collect();
        assert_eq!(slots, [A, B, C, D, SELECTOR, BUS_ONLY]);
    }

    /// A constraint constant with an extension part cannot run in the base
    /// field, and says which step it is.
    #[test]
    fn a_constant_with_an_extension_part_is_refused() {
        let mut b = Builder::<Ext3>::new();
        let a = b.var(0);
        let odd = b.fixed(FX::new([FB::one(), FB::one(), FB::zero()]));
        let root = b.mul(a, odd);
        let program = b.finish_verbatim(root).unwrap();
        assert_eq!(
            Constraints::<Gl>::from_extension(program.steps(), &[root], &[None], 1).unwrap_err(),
            Error::NotBaseField { step: 1 }
        );
    }
}
