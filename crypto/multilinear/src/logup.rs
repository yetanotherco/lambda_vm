//! A bus as a fraction tree over the trace, and the rules that settle its
//! input-layer claim against the trace's factors.
//!
//! A bus balances when `Σ ± mult / (z − fingerprint) = 0`. Both parts are
//! **affine** in the trace columns — a multiplicity is a linear combination of
//! flag columns, a fingerprint is limbs weighted by powers of two and bus
//! elements weighted by powers of a challenge — so neither is ever a column of
//! its own. [`FractionTree`](crate::gkr::FractionTree) proves the sum, and what
//! it leaves is one claim about the input layer.
//!
//! That claim reduces to **two rules**, whatever the number of interactions.
//! The input layer is indexed by `(interaction, row)` with the interaction in
//! the high variables, so the claim point splits and
//! `p(hi ‖ lo) = Σ_i eq(hi, i)·p_i(lo)`: the interaction weights are constants
//! the verifier computes, leaving a single affine combination of factors per
//! side. Both rules are degree 2 once the row weight is counted.

use math::field::{element::FieldElement, traits::IsField};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    Error,
    batch::Rule,
    eq::eq_evals,
    gkr::FractionLayer,
    mle::Mle,
    program::{Builder, Program},
};

/// An affine expression over the sumcheck factors: `Σ c_j·f_{s_j} + k`.
#[derive(Clone, Debug)]
pub struct Affine<E: IsField> {
    terms: Vec<(usize, FieldElement<E>)>,
    constant: FieldElement<E>,
}

impl<E: IsField + 'static> Affine<E> {
    /// `terms` pairs a factor slot with its coefficient.
    pub fn new(terms: Vec<(usize, FieldElement<E>)>, constant: FieldElement<E>) -> Self {
        Self { terms, constant }
    }

    pub fn constant(value: FieldElement<E>) -> Self {
        Self::new(Vec::new(), value)
    }

    /// A single factor read with coefficient one.
    pub fn factor(slot: usize) -> Self {
        Self::new(vec![(slot, FieldElement::one())], FieldElement::zero())
    }

    pub fn terms(&self) -> &[(usize, FieldElement<E>)] {
        &self.terms
    }

    /// The `k` of `Σ c_j·f_{s_j} + k`.
    ///
    /// Read-only, and named `constant_term` rather than `constant` because that
    /// name is already the CONSTRUCTOR above. A caller that emits this
    /// expression as code — the WHIR recursion's in-guest verifier — needs the
    /// coefficients and this, where [`Affine::evaluate`] gives it only the
    /// answer.
    pub fn constant_term(&self) -> &FieldElement<E> {
        &self.constant
    }

    /// The expression's value, given every factor's value at a point.
    pub fn evaluate(&self, values: &[FieldElement<E>]) -> FieldElement<E> {
        self.terms
            .iter()
            .fold(self.constant.clone(), |acc, (slot, coefficient)| {
                acc + coefficient * &values[*slot]
            })
    }

    /// The expression as steps over the factor values, returning the step
    /// holding it.
    pub fn emit(&self, builder: &mut Builder<E>) -> u32 {
        if self.terms.is_empty() {
            return builder.fixed(self.constant.clone());
        }
        let terms: Vec<(u32, FieldElement<E>)> = self
            .terms
            .iter()
            .map(|(slot, coefficient)| (builder.var(*slot), coefficient.clone()))
            .collect();
        let sum = builder.weighted_sum(&terms);
        if self.constant == FieldElement::zero() {
            return sum;
        }
        let constant = builder.fixed(self.constant.clone());
        builder.add(sum, constant)
    }

    /// The expression's table over the cube. `factors` must not be empty: its
    /// first entry sets the height.
    pub fn table(&self, factors: &[Mle<E>]) -> Result<Mle<E>, Error>
    where
        FieldElement<E>: Send + Sync,
    {
        let size = factors.first().ok_or(Error::EmptyPolynomial)?.len();
        let mut evals = vec![self.constant.clone(); size];
        for (slot, coefficient) in &self.terms {
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
            // One pass per term over the whole column. The bus builds one of
            // these per interaction, so it is a hot loop on a real table.
            #[cfg(feature = "parallel")]
            evals
                .par_iter_mut()
                .zip(factor.evals().par_iter())
                .for_each(|(slot, value)| *slot += coefficient * value);
            #[cfg(not(feature = "parallel"))]
            for (slot, value) in evals.iter_mut().zip(factor.evals()) {
                *slot += coefficient * value;
            }
        }
        Mle::new(evals)
    }
}

/// One bus interaction on one table.
///
/// `numerator` is the **signed** multiplicity — a receiver's sign is already in
/// its coefficients — and `denominator` is `z − fingerprint`.
#[derive(Clone, Debug)]
pub struct Interaction<E: IsField> {
    pub numerator: Affine<E>,
    pub denominator: Affine<E>,
}

impl<E: IsField> Interaction<E> {
    pub fn new(numerator: Affine<E>, denominator: Affine<E>) -> Self {
        Self {
            numerator,
            denominator,
        }
    }
}

/// The number of variables the input layer spans: the rows plus the bits that
/// index the interaction.
pub fn input_layer_vars(interactions: usize, num_row_vars: usize) -> usize {
    num_row_vars + interactions.next_power_of_two().trailing_zeros() as usize
}

/// The fraction tree's input layer, indexed by `(interaction, row)` with the
/// interaction in the high variables.
///
/// Interactions are padded up to a power of two with `0/1`, which the tree adds
/// without moving the sum.
pub fn input_layer<E: IsField + 'static>(
    interactions: &[Interaction<E>],
    factors: &[Mle<E>],
) -> Result<FractionLayer<E>, Error> {
    if interactions.is_empty() {
        return Err(Error::EmptyPolynomial);
    }
    let size = factors.first().ok_or(Error::EmptyPolynomial)?.len();
    let slots = interactions.len().next_power_of_two();

    let mut p = Vec::with_capacity(slots * size);
    let mut q = Vec::with_capacity(slots * size);
    for interaction in interactions {
        p.extend(interaction.numerator.table(factors)?.into_evals());
        q.extend(interaction.denominator.table(factors)?.into_evals());
    }
    for _ in interactions.len()..slots {
        p.extend(std::iter::repeat_n(FieldElement::<E>::zero(), size));
        q.extend(std::iter::repeat_n(FieldElement::<E>::one(), size));
    }

    FractionLayer::new(Mle::new(p)?, Mle::new(q)?)
}

/// A table's input layer as a plan over its base columns (D-BATCH M1-2): for
/// each interaction's two sides, its terms — where the column a factor reads
/// starts in the table's run of columns (`column · rows`) and its shift, mod
/// `rows` — with their ext3 coefficients, and the side's constant, all as raw
/// limbs for the card.
///
/// Side `2i` is interaction `i`'s numerator, `2i + 1` its denominator;
/// `side_start[s] .. side_start[s + 1]` are side `s`'s terms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputPlan {
    pub side_start: Vec<u32>,
    pub terms: Vec<u64>,
    pub coeffs: Vec<u64>,
    pub constants: Vec<u64>,
}

impl InputPlan {
    pub fn interactions(&self) -> usize {
        self.side_start.len() / 2
    }
}

/// The plan for `interactions` over factors whose sources are `sources` (one
/// per slot: the committed column and shift a factor reads, `None` for a
/// public factor). `None` when a side reads a public factor, when a slot is
/// unknown, or when `E` is not the cubic extension the card works in.
pub fn input_plan<E: IsField + 'static>(
    interactions: &[Interaction<E>],
    sources: &[Option<crate::claim_reduce::FactorSource>],
    rows: usize,
) -> Option<InputPlan> {
    let mut plan = InputPlan {
        side_start: vec![0],
        terms: Vec::new(),
        coeffs: Vec::new(),
        constants: Vec::new(),
    };
    for interaction in interactions {
        for side in [&interaction.numerator, &interaction.denominator] {
            for (slot, coefficient) in side.terms() {
                let source = (*sources.get(*slot)?)?;
                plan.terms.push((source.column * rows) as u64);
                plan.terms.push((source.offset % rows.max(1)) as u64);
                plan.coeffs
                    .extend_from_slice(&crate::gpu::ext3_raw(coefficient)?);
            }
            plan.constants
                .extend_from_slice(&crate::gpu::ext3_raw(side.constant_term())?);
            plan.side_start
                .push(u32::try_from(plan.terms.len() / 2).ok()?);
        }
    }
    Some(plan)
}

/// The input layer and the tree above it, written straight from the table's
/// resident base columns ([`crate::gpu::input_layer_tree_from_columns`]),
/// under `LAMBDA_VM_ARGUE_GKR_INPUT`. `None` when the columns are not on the
/// card, a side reads a public factor, or the card declines — the caller then
/// takes today's path.
pub fn resident_tree_from_columns<F: IsField + 'static, E: IsField + 'static>(
    interactions: &[Interaction<E>],
    trace: &crate::constraint_argument::TraceData<F, E>,
) -> Option<crate::gkr::FractionTree<E>> {
    if interactions.is_empty() {
        return None;
    }
    let (store, first) = trace.resident_shared()?;
    let rows = trace.columns().first()?.len();
    let sources: Vec<_> = trace
        .kinds()
        .iter()
        .map(crate::constraint_argument::FactorKind::source)
        .collect();
    let plan = input_plan(interactions, &sources, rows)?;
    let tree = crate::gpu::input_layer_tree_from_columns(
        store,
        first,
        trace.columns().len(),
        rows,
        &plan,
    )?;
    crate::gkr::FractionTree::from_device(tree).ok()
}

/// The input layer and the tree above it, built where the factors already are.
///
/// The layer is `interactions × rows` fractions — the biggest thing a table's
/// argument builds — and every one of its cells is an affine expression over
/// the factors, which is a program the device can run.
///
/// `None` when the device declines; the caller then builds the layer here.
pub fn resident_tree<E: IsField + 'static>(
    interactions: &[Interaction<E>],
    factors: std::sync::Arc<crate::gpu::DeviceFactors>,
) -> Option<crate::gkr::FractionTree<E>> {
    if interactions.is_empty() {
        return None;
    }
    let emit = |side: &Affine<E>| {
        let mut builder = Builder::<E>::new();
        let root = side.emit(&mut builder);
        builder.finish(root).ok()
    };
    let numerators: Vec<Program<E>> = interactions
        .iter()
        .map(|i| emit(&i.numerator))
        .collect::<Option<_>>()?;
    let denominators: Vec<Program<E>> = interactions
        .iter()
        .map(|i| emit(&i.denominator))
        .collect::<Option<_>>()?;

    let tree = crate::gpu::input_layer_tree(factors, numerators, denominators)?;
    crate::gkr::FractionTree::from_device(tree).ok()
}

/// A device fraction tree built ahead of its turn, its output fraction left
/// unread so the fold kernels stay in flight — see [`resident_tree_deferred`].
/// Finalized at the consume site with [`Self::into_tree`].
pub struct PrefetchedTree<E: IsField> {
    device: crate::gpu::DeviceTree,
    _marker: core::marker::PhantomData<E>,
}

impl<E: IsField + 'static> PrefetchedTree<E> {
    /// Read the output fraction (the sync the prefetch deferred) and wrap the
    /// tree for `prove`. `None` on a device read failure — the caller then
    /// builds host-side; the transcript has not moved here, so that is sound.
    pub fn into_tree(self) -> Option<crate::gkr::FractionTree<E>> {
        crate::gkr::FractionTree::from_device(self.device).ok()
    }
}

/// The prefetch sibling of [`resident_tree`]: builds the device tree WITHOUT
/// reading its output, so the fold kernels overlap the caller's argue instead
/// of being waited on. `None` when the device declines (as [`resident_tree`])
/// OR there is no non-evicting VRAM headroom for a second tree — prefetch is
/// subordinate to argue and to the round-2 retention, displacing neither.
pub fn resident_tree_deferred<E: IsField + 'static>(
    interactions: &[Interaction<E>],
    factors: std::sync::Arc<crate::gpu::DeviceFactors>,
) -> Option<PrefetchedTree<E>> {
    if interactions.is_empty() {
        return None;
    }
    let emit = |side: &Affine<E>| {
        let mut builder = Builder::<E>::new();
        let root = side.emit(&mut builder);
        builder.finish(root).ok()
    };
    let numerators: Vec<Program<E>> = interactions
        .iter()
        .map(|i| emit(&i.numerator))
        .collect::<Option<_>>()?;
    let denominators: Vec<Program<E>> = interactions
        .iter()
        .map(|i| emit(&i.denominator))
        .collect::<Option<_>>()?;

    let device = crate::gpu::input_layer_tree_deferred(factors, numerators, denominators)?;
    Some(PrefetchedTree {
        device,
        _marker: core::marker::PhantomData,
    })
}

/// What the batch needs to settle a bus's input-layer claim.
pub struct BusStatements<'a, E: IsField> {
    pub numerator: Rule<'a, E>,
    pub denominator: Rule<'a, E>,
    /// The row half of the claim point — where the weight table belongs.
    pub row_point: Vec<FieldElement<E>>,
}

/// Turns a GKR input-layer claim into two rules over the trace's factors.
///
/// `weight` is the factor slot holding `eq(row_point, ·)`, which the caller
/// adds as a public factor. Both sides must build these from the same
/// interactions: they are the bus's structure, not proof data.
pub fn claim_statements<'a, E: IsField + 'static>(
    interactions: &'a [Interaction<E>],
    claim_point: &[FieldElement<E>],
    num_row_vars: usize,
    weight: usize,
) -> Result<BusStatements<'a, E>, Error> {
    let expected = input_layer_vars(interactions.len(), num_row_vars);
    if claim_point.len() != expected {
        return Err(Error::VariableCountMismatch {
            expected,
            got: claim_point.len(),
        });
    }
    let (interaction_point, row_point) = claim_point.split_at(claim_point.len() - num_row_vars);

    // Weight per interaction; the tail belongs to the 0/1 padding slots, whose
    // denominators are one and whose numerators vanish.
    let weights = eq_evals(interaction_point);
    let padding = weights[interactions.len()..]
        .iter()
        .fold(FieldElement::<E>::zero(), |acc, w| acc + w);
    let live = weights[..interactions.len()].to_vec();

    // `Σ_i w_i · side_i(f)` (plus the padding, where it belongs), times the row
    // weight — the shape both sides of the bus statement take.
    let weighted = |sides: Vec<&Affine<E>>, constant: Option<FieldElement<E>>| {
        let mut builder = Builder::<E>::new();
        let mut terms: Vec<(u32, FieldElement<E>)> = sides
            .into_iter()
            .zip(&live)
            .map(|(side, w)| (side.emit(&mut builder), w.clone()))
            .collect();
        if let Some(constant) = constant {
            let step = builder.fixed(constant);
            terms.push((step, FieldElement::one()));
        }
        let sum = builder.weighted_sum(&terms);
        let row = builder.var(weight);
        let root = builder.mul(row, sum);
        builder.finish(root)
    };

    let numerator: Program<E> =
        weighted(interactions.iter().map(|i| &i.numerator).collect(), None)?;
    let denominator: Program<E> = weighted(
        interactions.iter().map(|i| &i.denominator).collect(),
        Some(padding),
    )?;
    let numerator = Rule::compiled(2, numerator);
    let denominator = Rule::compiled(2, denominator);

    Ok(BusStatements {
        numerator,
        denominator,
        row_point: row_point.to_vec(),
    })
}

#[cfg(test)]
mod input_plan_tests {
    use super::*;
    use crate::claim_reduce::FactorSource;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;

    type EE = FieldElement<Ext3>;

    fn next(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn ext(seed: &mut u64) -> EE {
        EE::new([next(seed).into(), next(seed).into(), next(seed).into()])
    }

    /// What `gkr_input_from_columns` computes, on the host: the kernel's
    /// arithmetic, cell by cell, from the plan's raw limbs.
    fn run_plan(
        plan: &InputPlan,
        columns: &[u64],
        rows: usize,
        padding: bool,
    ) -> (Vec<EE>, Vec<EE>) {
        let interactions = plan.interactions();
        let slots = if padding {
            interactions.next_power_of_two()
        } else {
            interactions
        };
        let raw = |limbs: &[u64]| crate::gpu::ext3_from_raw::<Ext3>(limbs);
        let (mut p, mut q) = (Vec::new(), Vec::new());
        for s in 0..slots {
            for r in 0..rows {
                if s >= interactions {
                    p.push(EE::zero());
                    q.push(EE::one());
                    continue;
                }
                let side = |which: usize| {
                    let mut acc = raw(&plan.constants[which * 3..which * 3 + 3]);
                    for t in plan.side_start[which] as usize..plan.side_start[which + 1] as usize {
                        let mut k = r + plan.terms[2 * t + 1] as usize;
                        if k >= rows {
                            k -= rows;
                        }
                        let value =
                            FieldElement::<Gl>::from(columns[plan.terms[2 * t] as usize + k]);
                        acc += raw(&plan.coeffs[t * 3..t * 3 + 3]) * value.to_extension::<Ext3>();
                    }
                    acc
                };
                p.push(side(2 * s));
                q.push(side(2 * s + 1));
            }
        }
        (p, q)
    }

    /// ★ The plan's cells are the lifted path's: for interactions over direct
    /// and shifted columns, padded or not, the kernel's arithmetic on the plan
    /// equals `input_layer` over the lifted factors, cell for cell.
    #[test]
    fn the_plan_writes_the_lifted_input_layer() {
        let mut seed = 0x5EED_u64;
        for (rows, width, count) in [(1usize, 2usize, 1usize), (8, 3, 3), (32, 5, 5), (64, 7, 9)] {
            let columns: Vec<Vec<u64>> = (0..width)
                .map(|_| {
                    (0..rows)
                        .map(|_| next(&mut seed) % 0xFFFF_FFFF_0000_0001)
                        .collect()
                })
                .collect();
            // Every column direct, then two shifted views, then a public slot.
            let mut sources: Vec<Option<FactorSource>> =
                (0..width).map(|c| Some(FactorSource::direct(c))).collect();
            sources.push(Some(FactorSource::shifted(0, 1)));
            sources.push(Some(FactorSource::shifted(width - 1, 3)));
            sources.push(None);
            let factors: Vec<Mle<Ext3>> = sources
                .iter()
                .map(|source| {
                    let evals = (0..rows)
                        .map(|r| match source {
                            Some(src) => FieldElement::<Gl>::from(
                                columns[src.column][(r + src.offset) % rows],
                            )
                            .to_extension::<Ext3>(),
                            None => EE::from(7u64),
                        })
                        .collect();
                    Mle::new(evals).unwrap()
                })
                .collect();
            let committed = sources.len() - 1;
            let affine = |seed: &mut u64| {
                let terms = (0..(next(seed) % 4) as usize)
                    .map(|_| ((next(seed) as usize) % committed, ext(seed)))
                    .collect();
                Affine::new(terms, ext(seed))
            };
            let interactions: Vec<Interaction<Ext3>> = (0..count)
                .map(|_| Interaction::new(affine(&mut seed), affine(&mut seed)))
                .collect();
            let plan = input_plan(&interactions, &sources, rows).expect("a plan over columns");
            let flat: Vec<u64> = columns.concat();
            let want = input_layer(&interactions, &factors).unwrap();
            let (p, q) = run_plan(&plan, &flat, rows, true);
            assert_eq!(p, want.p.evals(), "rows {rows}: the numerators differ");
            assert_eq!(q, want.q.evals(), "rows {rows}: the denominators differ");
            let (p, _) = run_plan(&plan, &flat, rows, false);
            assert_eq!(p.len(), count * rows, "unpadded: the real cells only");
            assert_eq!(p[..], want.p.evals()[..count * rows]);
        }
    }

    /// A side that reads a public factor has no plan: the caller keeps
    /// today's path.
    #[test]
    fn a_public_factor_has_no_plan() {
        let sources = vec![Some(FactorSource::direct(0)), None];
        let interactions = vec![Interaction::new(
            Affine::factor(1),
            Affine::constant(EE::one()),
        )];
        assert_eq!(input_plan::<Ext3>(&interactions, &sources, 4), None);
        let interactions = vec![Interaction::new(
            Affine::factor(0),
            Affine::constant(EE::one()),
        )];
        assert!(input_plan::<Ext3>(&interactions, &sources, 4).is_some());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    use crate::{
        batch,
        eq::eq_mle,
        gkr::{self, FractionTree},
    };

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"logup-test")
    }

    /// Factor layout: the value, the two half-flags, and the row weight.
    const VALUE: usize = 0;
    const LOW: usize = 1;
    const HIGH: usize = 2;
    const WEIGHT: usize = 3;

    /// The low half sends each value and the high half receives the same one,
    /// so the bus balances *because of the values*, not by construction.
    fn columns(num_vars: usize) -> Vec<Mle<F>> {
        let size = 1usize << num_vars;
        let half = size / 2;
        let value: Vec<FE> = (0..size).map(|i| FE::from((i % half) as u64 + 1)).collect();
        let low: Vec<FE> = (0..size)
            .map(|i| if i < half { FE::one() } else { FE::zero() })
            .collect();
        let high: Vec<FE> = low.iter().map(|v| FE::one() - v).collect();
        vec![
            Mle::new(value).unwrap(),
            Mle::new(low).unwrap(),
            Mle::new(high).unwrap(),
        ]
    }

    /// `+low / (z − alpha·value)` against `−high / (z − alpha·value)`.
    fn balanced(z: FE, alpha: FE, _num_vars: usize) -> Vec<Interaction<F>> {
        vec![
            Interaction::new(Affine::factor(LOW), Affine::new(vec![(VALUE, -alpha)], z)),
            Interaction::new(
                Affine::new(vec![(HIGH, -FE::one())], FE::zero()),
                Affine::new(vec![(VALUE, -alpha)], z),
            ),
        ]
    }

    #[test]
    fn an_affine_expression_evaluates_and_tabulates_alike() {
        let factors = columns(3);
        let expr = Affine::new(vec![(VALUE, FE::from(7)), (LOW, FE::from(11))], FE::from(5));
        let table = expr.table(&factors).unwrap();

        for i in 0..factors[0].len() {
            let values: Vec<FE> = factors.iter().map(|f| f.evals()[i]).collect();
            assert_eq!(table.evals()[i], expr.evaluate(&values), "row {i}");
        }
    }

    #[test]
    fn the_input_layer_lays_interactions_out_row_by_row() {
        let num_vars = 3;
        let size = 1usize << num_vars;
        let factors = columns(num_vars);
        let interactions = balanced(FE::from(97), FE::from(31), num_vars);
        let layer = input_layer(&interactions, &factors).unwrap();

        assert_eq!(layer.num_vars(), input_layer_vars(2, num_vars));
        for (i, interaction) in interactions.iter().enumerate() {
            let p = interaction.numerator.table(&factors).unwrap();
            let q = interaction.denominator.table(&factors).unwrap();
            for row in 0..size {
                // The interaction sits in the high variables.
                assert_eq!(layer.p.evals()[i * size + row], p.evals()[row]);
                assert_eq!(layer.q.evals()[i * size + row], q.evals()[row]);
            }
        }
    }

    #[test]
    fn padding_slots_are_the_zero_fraction() {
        let num_vars = 2;
        let size = 1usize << num_vars;
        let factors = columns(num_vars);
        let z = FE::from(97);
        let alpha = FE::from(31);
        // Three interactions pad up to four slots.
        let mut interactions = balanced(z, alpha, num_vars);
        interactions.push(Interaction::new(
            Affine::constant(FE::zero()),
            Affine::constant(z),
        ));

        let layer = input_layer(&interactions, &factors).unwrap();
        assert_eq!(layer.num_vars(), num_vars + 2);
        for row in 0..size {
            assert_eq!(layer.p.evals()[3 * size + row], FE::zero());
            assert_eq!(layer.q.evals()[3 * size + row], FE::one());
        }
    }

    #[test]
    fn a_balanced_bus_has_a_zero_output_numerator() {
        let num_vars = 3;
        let factors = columns(num_vars);
        let interactions = balanced(FE::from(97), FE::from(31), num_vars);
        let tree = FractionTree::build(input_layer(&interactions, &factors).unwrap()).unwrap();

        let (p, q) = tree.output();
        assert_eq!(p, FE::zero());
        assert_ne!(q, FE::zero());
    }

    #[test]
    fn an_unbalanced_bus_does_not() {
        let num_vars = 3;
        let factors = columns(num_vars);
        let z = FE::from(97);
        let alpha = FE::from(31);
        // Only the send: nothing cancels it.
        let interactions = vec![balanced(z, alpha, num_vars).remove(0)];
        let tree = FractionTree::build(input_layer(&interactions, &factors).unwrap()).unwrap();

        assert_ne!(tree.output().0, FE::zero());
    }

    #[test]
    fn a_value_received_that_was_never_sent_unbalances_the_bus() {
        // The multisets are what balance, so moving one value apart is enough.
        let num_vars = 3;
        let mut factors = columns(num_vars);
        let mut value = factors[0].evals().to_vec();
        value[5] += FE::one();
        factors[0] = Mle::new(value).unwrap();

        let interactions = balanced(FE::from(97), FE::from(31), num_vars);
        let tree = FractionTree::build(input_layer(&interactions, &factors).unwrap()).unwrap();
        assert_ne!(tree.output().0, FE::zero());
    }

    /// The identity the batch leans on: each rule sums over the row cube to the
    /// claim the GKR handed back.
    #[test]
    fn the_input_claim_is_what_the_rules_sum_to() {
        for num_vars in 1..=4usize {
            let factors = columns(num_vars);
            let interactions = balanced(FE::from(97), FE::from(31), num_vars);
            let tree = FractionTree::build(input_layer(&interactions, &factors).unwrap()).unwrap();
            let out = gkr::prove(&tree, &mut transcript()).unwrap();

            let statements =
                claim_statements(&interactions, &out.claim.point, num_vars, WEIGHT).unwrap();
            let mut with_weight = factors.clone();
            with_weight.push(eq_mle(&statements.row_point).unwrap());

            let sum = |rule: &Rule<'_, F>| {
                (0..(1usize << num_vars)).fold(FE::zero(), |acc, i| {
                    let values: Vec<FE> = with_weight.iter().map(|f| f.evals()[i]).collect();
                    acc + rule.apply(&values)
                })
            };
            assert_eq!(sum(&statements.numerator), out.claim.p, "n={num_vars}");
            assert_eq!(sum(&statements.denominator), out.claim.q, "n={num_vars}");
        }
    }

    /// Same identity with the padding in play, where a wrong padding weight
    /// would show up in the denominator.
    #[test]
    fn the_claim_holds_with_padded_interactions() {
        let num_vars = 3;
        let factors = columns(num_vars);
        let z = FE::from(97);
        let alpha = FE::from(31);
        let mut interactions = balanced(z, alpha, num_vars);
        interactions.push(Interaction::new(
            Affine::constant(FE::zero()),
            Affine::constant(z),
        ));

        let tree = FractionTree::build(input_layer(&interactions, &factors).unwrap()).unwrap();
        let out = gkr::prove(&tree, &mut transcript()).unwrap();
        let statements =
            claim_statements(&interactions, &out.claim.point, num_vars, WEIGHT).unwrap();

        let mut with_weight = factors;
        with_weight.push(eq_mle(&statements.row_point).unwrap());
        let sum = |rule: &Rule<'_, F>| {
            (0..(1usize << num_vars)).fold(FE::zero(), |acc, i| {
                let values: Vec<FE> = with_weight.iter().map(|f| f.evals()[i]).collect();
                acc + rule.apply(&values)
            })
        };
        assert_eq!(sum(&statements.numerator), out.claim.p);
        assert_eq!(sum(&statements.denominator), out.claim.q);
    }

    /// Two rules, one batched sumcheck, whatever the interaction count.
    #[test]
    fn the_bus_costs_two_rules_in_one_sumcheck() {
        let num_vars = 3;
        let factors = columns(num_vars);
        let interactions = balanced(FE::from(97), FE::from(31), num_vars);
        let tree = FractionTree::build(input_layer(&interactions, &factors).unwrap()).unwrap();
        let output = tree.output();

        let mut prover = transcript();
        let out = gkr::prove(&tree, &mut prover).unwrap();
        let statements =
            claim_statements(&interactions, &out.claim.point, num_vars, WEIGHT).unwrap();
        let mut with_weight = factors.clone();
        with_weight.push(eq_mle(&statements.row_point).unwrap());
        let claims = [out.claim.p, out.claim.q];

        let (proof, point) = batch::prove(
            with_weight.clone(),
            vec![statements.numerator, statements.denominator],
            &claims,
            &mut prover,
        )
        .unwrap();
        assert_eq!(proof.rounds.len(), num_vars);

        let mut verifier = transcript();
        let claim = gkr::verify(&out.proof, output, &mut verifier).unwrap();
        let statements = claim_statements(&interactions, &claim.point, num_vars, WEIGHT).unwrap();
        let row_point = statements.row_point.clone();

        let checked = batch::verify(
            &proof,
            &[statements.numerator, statements.denominator],
            &claims,
            |at: &[FE]| {
                // Only the trace factors travel; the weight is recomputed.
                let mut values: Vec<FE> = factors
                    .iter()
                    .map(|f| f.evaluate(at))
                    .collect::<Result<_, _>>()?;
                values.push(crate::eq::eq_eval(&row_point, at)?);
                Ok(values)
            },
            num_vars,
            &mut verifier,
        )
        .unwrap();
        assert_eq!(checked, point);
    }

    #[test]
    fn a_claim_point_of_the_wrong_arity_is_rejected() {
        let interactions = balanced(FE::from(97), FE::from(31), 3);
        let result = claim_statements(&interactions, &[FE::one(); 3], 3, WEIGHT);
        assert_eq!(
            result.err(),
            Some(Error::VariableCountMismatch {
                expected: 4,
                got: 3
            })
        );
    }

    #[test]
    fn an_empty_bus_is_rejected() {
        assert_eq!(
            input_layer::<F>(&[], &columns(3)).unwrap_err(),
            Error::EmptyPolynomial
        );
    }
}
