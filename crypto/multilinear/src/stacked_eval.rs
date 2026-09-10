//! Settling many column claims against **few** commitments.
//!
//! [`stacking`](crate::stacking) puts each column in a subcube of a shared
//! polynomial, so `column(z) = stacked(prefix ‖ z)` with no protocol. What is
//! left is a multi-point claim on one committed polynomial, and one weighted
//! sumcheck settles it:
//!
//! ```text
//! Σ_i gamma^i·column_i(z) = Σ_x [Σ_i gamma^i·eq(prefix_i ‖ z, x)]·stacked(x)
//! ```
//!
//! The columns of one table share a height and — once the claim reduction has
//! run — a point, so the weight **factors**: a slot table over the prefix bits
//! times `eq(z, ·)` over the column's own variables. Building it is then one
//! pass over the stacked cube instead of one per column, which is what makes
//! this cheap enough to be worth doing.
//!
//! The claim is settled by [`whir_chain`](crate::whir_chain), so the stack pays
//! **one** round schedule and query set for the whole trace — where a
//! per-column path would pay one each. That is where stacking earns its keep;
//! with a single-round WHIR it only changed the shape.
//!
//! The stacked polynomials stay in the **base field**: a trace is base-field,
//! and nothing extension-valued touches it until the first fold. The claims and
//! the weight are in the extension, which is what the mixed evaluation is for.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
    },
    traits::AsBytes,
};

use crate::{
    Error, challenge_powers,
    eq::{eq_eval, eq_evals},
    mle::Mle,
    stacking::StackedLayout,
    whir::Domain,
    whir_chain::{self, ChainConfig, ChainProof},
    whir_commit::{CodewordCommitment, Commitment},
};

/// The stacked polynomials, committed. Base-field, like the trace they hold.
pub struct StackedCommitment<F: IsFFTField + IsPrimeField>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    layout: StackedLayout,
    polys: Vec<Mle<F>>,
    commitments: Vec<CodewordCommitment<F>>,
    domain: Domain<F>,
}

impl<F: IsFFTField + IsPrimeField + Send + Sync> StackedCommitment<F>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    /// Packs `columns` into `layout` and commits each stacked polynomial.
    ///
    /// Every stacked polynomial has `n_stack` variables, so they share one
    /// evaluation domain. Takes the columns as they already are — copying them
    /// into `Vec`s first would be one more resident copy of the whole trace.
    pub fn commit(
        layout: StackedLayout,
        columns: &[Mle<F>],
        config: &ChainConfig,
    ) -> Result<Self, Error> {
        let polys = layout.stack(columns)?;
        let mut commitments = Vec::with_capacity(polys.len());
        let mut domain = None;
        for poly in &polys {
            let (commitment, d) = whir_chain::commit::<F>(poly, config)?;
            commitments.push(commitment);
            domain = Some(d);
        }
        Ok(Self {
            layout,
            polys,
            commitments,
            domain: domain.ok_or(Error::EmptyPolynomial)?,
        })
    }

    pub fn roots(&self) -> Vec<Commitment> {
        self.commitments.iter().map(|c| c.root()).collect()
    }

    pub fn domain(&self) -> &Domain<F> {
        &self.domain
    }

    pub fn layout(&self) -> &StackedLayout {
        &self.layout
    }
}

/// One weighted, chained evaluation proof per stacked polynomial.
#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct StackedProof<F: IsField, E: IsField> {
    pub polys: Vec<ChainProof<F, E>>,
}

impl<F: IsField, E: IsField> StackedProof<F, E> {
    /// Codeword elements the openings carry, across every stacked polynomial.
    pub fn opened_elements(&self) -> usize {
        self.polys.iter().map(ChainProof::opened_elements).sum()
    }
}

/// The columns one stacked polynomial holds: `(column index, slot)`, plus the
/// height they share.
///
/// Errors if they do not share it: the weight only factors when they do.
fn columns_in(layout: &StackedLayout, poly: usize) -> Result<(usize, Vec<(usize, usize)>), Error> {
    let mut num_vars = None;
    let mut slots = Vec::new();
    for (column, place) in layout.placements().iter().enumerate() {
        if place.poly != poly {
            continue;
        }
        match num_vars {
            None => num_vars = Some(place.num_vars),
            Some(m) if m != place.num_vars => {
                return Err(Error::VariableCountMismatch {
                    expected: m,
                    got: place.num_vars,
                });
            }
            _ => {}
        }
        slots.push((column, place.offset >> place.num_vars));
    }
    Ok((num_vars.unwrap_or(layout.n_stack()), slots))
}

/// `Σ_i gamma^i·eq(prefix_i ‖ point, ·)` as a table, built as the tensor of the
/// slot weights and `eq(point, ·)`.
fn weight_table<E: IsField>(
    layout: &StackedLayout,
    poly: usize,
    point: &[FieldElement<E>],
    weights: &[FieldElement<E>],
) -> Result<Mle<E>, Error> {
    let (num_vars, slots) = columns_in(layout, poly)?;
    if point.len() != num_vars {
        return Err(Error::VariableCountMismatch {
            expected: num_vars,
            got: point.len(),
        });
    }
    // On the cube `eq(prefix_i, ·)` is the indicator of slot `i`, so the high
    // half of the weight is just the batching coefficient in that slot.
    let mut slot_weight = vec![FieldElement::<E>::zero(); 1usize << (layout.n_stack() - num_vars)];
    for (column, slot) in slots {
        slot_weight[slot] = weights[column].clone();
    }

    let eq_low = eq_evals(point);
    let mut table = Vec::with_capacity(1usize << layout.n_stack());
    for w in &slot_weight {
        table.extend(eq_low.iter().map(|e| w * e));
    }
    Mle::new(table)
}

/// The same weight at an arbitrary point, in closed form.
fn weight_at<E: IsField>(
    layout: &StackedLayout,
    poly: usize,
    point: &[FieldElement<E>],
    weights: &[FieldElement<E>],
    at: &[FieldElement<E>],
) -> Result<FieldElement<E>, Error> {
    let (num_vars, slots) = columns_in(layout, poly)?;
    if at.len() != layout.n_stack() {
        return Err(Error::VariableCountMismatch {
            expected: layout.n_stack(),
            got: at.len(),
        });
    }
    let (high, low) = at.split_at(layout.n_stack() - num_vars);

    let mut prefix = FieldElement::<E>::zero();
    for (column, _) in slots {
        let bits = layout
            .placement(column)
            .expect("column came from this layout")
            .prefix_bits();
        let corner: Vec<FieldElement<E>> = bits
            .into_iter()
            .map(|b| {
                if b {
                    FieldElement::<E>::one()
                } else {
                    FieldElement::<E>::zero()
                }
            })
            .collect();
        prefix += &weights[column] * eq_eval(&corner, high)?;
    }
    Ok(prefix * eq_eval(point, low)?)
}

/// The claimed sum for one stacked polynomial: its columns' values, batched.
fn claimed<E: IsField>(
    layout: &StackedLayout,
    poly: usize,
    values: &[FieldElement<E>],
    weights: &[FieldElement<E>],
) -> Result<FieldElement<E>, Error> {
    let (_, slots) = columns_in(layout, poly)?;
    Ok(slots
        .into_iter()
        .fold(FieldElement::<E>::zero(), |acc, (column, _)| {
            acc + &weights[column] * &values[column]
        }))
}

/// Proves that every column takes its claimed value at the shared `point`.
///
/// The claims are absorbed before the batching challenge, so the prover cannot
/// pick them after seeing it.
pub fn prove<F, E, T>(
    stacked: &StackedCommitment<F>,
    point: &[FieldElement<E>],
    values: &[FieldElement<E>],
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<StackedProof<F, E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
{
    let layout = &stacked.layout;
    if values.len() != layout.placements().len() {
        return Err(Error::QueryCountMismatch {
            expected: layout.placements().len(),
            got: values.len(),
        });
    }
    for value in values {
        transcript.append_field_element(value);
    }
    let weights = challenge_powers(&transcript.sample_field_element(), values.len());

    let mut polys = Vec::with_capacity(stacked.polys.len());
    for (i, (poly, commitment)) in stacked.polys.iter().zip(&stacked.commitments).enumerate() {
        polys.push(whir_chain::prove_weighted::<F, E, T>(
            poly,
            weight_table(layout, i, point, &weights)?,
            commitment,
            &stacked.domain,
            config,
            transcript,
        )?);
    }

    Ok(StackedProof { polys })
}

/// Verifies the column claims against the stacked commitments.
///
/// The layout is public and derived from the column heights, so it is not part
/// of the proof.
#[allow(clippy::too_many_arguments)]
pub fn verify<F, E, T>(
    proof: &StackedProof<F, E>,
    layout: &StackedLayout,
    roots: &[Commitment],
    point: &[FieldElement<E>],
    values: &[FieldElement<E>],
    domain: &Domain<F>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<(), Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
{
    if values.len() != layout.placements().len() {
        return Err(Error::QueryCountMismatch {
            expected: layout.placements().len(),
            got: values.len(),
        });
    }
    if proof.polys.len() != layout.num_polys() || roots.len() != layout.num_polys() {
        return Err(Error::QueryCountMismatch {
            expected: layout.num_polys(),
            got: proof.polys.len().min(roots.len()),
        });
    }
    for value in values {
        transcript.append_field_element(value);
    }
    let weights = challenge_powers(&transcript.sample_field_element(), values.len());

    for (i, (eval_proof, root)) in proof.polys.iter().zip(roots).enumerate() {
        whir_chain::verify_weighted::<F, E, T, _>(
            eval_proof,
            root,
            |at: &[FieldElement<E>]| weight_at(layout, i, point, &weights, at),
            claimed(layout, i, values, &weights)?,
            layout.n_stack(),
            domain,
            config,
            transcript,
        )
        .map_err(|_| Error::ColumnOpeningRejected { column: i })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::goldilocks::GoldilocksField as F;

    use crate::whir_chain::GrindBits;

    type FE = FieldElement<F>;

    fn transcript() -> DefaultTranscript<F> {
        DefaultTranscript::<F>::new(b"stacked-eval-test")
    }

    fn config() -> ChainConfig {
        ChainConfig {
            log_blowup: 2,
            log_folding: 2,
            num_queries: 3,
            grind: GrindBits::default(),
        }
    }

    fn column(num_vars: usize, seed: u64) -> Mle<F> {
        Mle::new(
            (0..(1u64 << num_vars))
                .map(|i| FE::from(i.wrapping_add(seed).wrapping_mul(6364136223846793005) >> 11))
                .collect(),
        )
        .unwrap()
    }

    fn point(num_vars: usize) -> Vec<FE> {
        (0..num_vars).map(|i| FE::from(101 + i as u64)).collect()
    }

    /// `count` columns of one height, and the layout that packs them.
    fn stack_of(count: usize, num_vars: usize, n_stack: usize) -> (StackedLayout, Vec<Mle<F>>) {
        let columns: Vec<Mle<F>> = (0..count)
            .map(|i| column(num_vars, 7 + i as u64 * 13))
            .collect();
        let layout = StackedLayout::build(&vec![num_vars; count], n_stack).unwrap();
        (layout, columns)
    }

    fn values(columns: &[Mle<F>], at: &[FE]) -> Vec<FE> {
        columns.iter().map(|c| c.evaluate(at).unwrap()).collect()
    }

    fn run(
        layout: StackedLayout,
        columns: &[Mle<F>],
        at: &[FE],
        claimed: &[FE],
    ) -> Result<usize, Error> {
        let stacked = StackedCommitment::<F>::commit(layout, columns, &config())?;
        let roots = stacked.roots();
        let proof = prove(&stacked, at, claimed, &config(), &mut transcript())?;

        verify(
            &proof,
            stacked.layout(),
            &roots,
            at,
            claimed,
            stacked.domain(),
            &config(),
            &mut transcript(),
        )?;
        Ok(roots.len())
    }

    /// The point of the module: many columns, one commitment, one opening.
    #[test]
    fn every_column_settles_against_one_commitment() {
        let num_vars = 3;
        let (layout, columns) = stack_of(6, num_vars, 6);
        assert_eq!(layout.num_polys(), 1);

        let at = point(num_vars);
        let claimed = values(&columns, &at);
        assert_eq!(run(layout, &columns, &at, &claimed).unwrap(), 1);
    }

    #[test]
    fn columns_overflowing_the_stack_use_more_polynomials() {
        let num_vars = 3;
        // Two slots per stacked polynomial, so six columns need three.
        let (layout, columns) = stack_of(6, num_vars, 4);
        assert_eq!(layout.num_polys(), 3);

        let at = point(num_vars);
        let claimed = values(&columns, &at);
        assert_eq!(run(layout, &columns, &at, &claimed).unwrap(), 3);
    }

    #[test]
    fn unused_slots_do_not_disturb_the_claims() {
        // Three columns in a four-slot stack: the padding is zero and carries no
        // weight.
        let num_vars = 3;
        let (layout, columns) = stack_of(3, num_vars, 5);
        assert_eq!(layout.num_polys(), 1);

        let at = point(num_vars);
        let claimed = values(&columns, &at);
        run(layout, &columns, &at, &claimed).unwrap();
    }

    #[test]
    fn a_forged_column_value_is_rejected() {
        let num_vars = 3;
        let (layout, columns) = stack_of(4, num_vars, 5);
        let at = point(num_vars);
        let mut claimed = values(&columns, &at);
        claimed[2] += FE::one();

        assert!(run(layout, &columns, &at, &claimed).is_err());
    }

    #[test]
    fn swapping_two_column_values_is_rejected() {
        let num_vars = 3;
        let (layout, columns) = stack_of(4, num_vars, 5);
        let at = point(num_vars);
        let mut claimed = values(&columns, &at);
        claimed.swap(0, 1);

        assert!(run(layout, &columns, &at, &claimed).is_err());
    }

    /// Right values, wrong point: the weight pins where they were taken.
    #[test]
    fn values_from_another_point_are_rejected() {
        let num_vars = 3;
        let (layout, columns) = stack_of(4, num_vars, 5);
        let at = point(num_vars);
        let elsewhere: Vec<FE> = at.iter().map(|x| x + FE::one()).collect();
        let claimed = values(&columns, &elsewhere);

        assert!(run(layout, &columns, &at, &claimed).is_err());
    }

    /// The crux: the prover folds the weight's table and the verifier evaluates
    /// its closed form, so they must be the same polynomial.
    #[test]
    fn the_weight_table_matches_its_closed_form() {
        let num_vars = 2;
        let n_stack = 4;
        let (layout, _) = stack_of(3, num_vars, n_stack);
        let at = point(num_vars);
        let weights: Vec<FE> = (0..3).map(|i| FE::from(3 + i as u64)).collect();

        let table = weight_table(&layout, 0, &at, &weights).unwrap();
        let off_cube: Vec<FE> = (0..n_stack).map(|i| FE::from(31 + i as u64)).collect();
        assert_eq!(
            table.evaluate(&off_cube).unwrap(),
            weight_at(&layout, 0, &at, &weights, &off_cube).unwrap()
        );
    }

    /// And the weight really does read each column out of its own subcube.
    #[test]
    fn the_weight_picks_each_column_out_of_its_slot() {
        let num_vars = 2;
        let (layout, columns) = stack_of(3, num_vars, 4);
        let at = point(num_vars);
        let weights: Vec<FE> = (0..3).map(|i| FE::from(3 + i as u64)).collect();

        let stacked = layout.stack(&columns).unwrap();
        let table = weight_table(&layout, 0, &at, &weights).unwrap();
        // Σ_x w(x)·stacked(x) must be the batched column values.
        let summed = table
            .evals()
            .iter()
            .zip(stacked[0].evals())
            .fold(FE::zero(), |acc, (w, v)| acc + w * v);
        let expected = values(&columns, &at)
            .iter()
            .zip(&weights)
            .fold(FE::zero(), |acc, (v, w)| acc + v * w);
        assert_eq!(summed, expected);
    }

    #[test]
    fn mixed_heights_in_one_stack_are_rejected() {
        let layout = StackedLayout::build(&[3, 2], 5).unwrap();
        let weights = [FE::one(), FE::one()];
        assert!(matches!(
            weight_table(&layout, 0, &point(3), &weights).err(),
            Some(Error::VariableCountMismatch { .. })
        ));
    }

    #[test]
    fn a_value_per_column_is_required() {
        let num_vars = 3;
        let (layout, columns) = stack_of(4, num_vars, 5);
        let at = point(num_vars);
        let claimed = values(&columns, &at);

        let stacked = StackedCommitment::<F>::commit(layout, &columns, &config()).unwrap();
        assert!(matches!(
            prove(&stacked, &at, &claimed[..3], &config(), &mut transcript()).err(),
            Some(Error::QueryCountMismatch {
                expected: 4,
                got: 3
            })
        ));
    }

    #[test]
    fn a_proof_replayed_under_another_transcript_is_rejected() {
        let num_vars = 3;
        let (layout, columns) = stack_of(4, num_vars, 5);
        let at = point(num_vars);
        let claimed = values(&columns, &at);

        let stacked = StackedCommitment::<F>::commit(layout, &columns, &config()).unwrap();
        let roots = stacked.roots();
        let proof = prove(&stacked, &at, &claimed, &config(), &mut transcript()).unwrap();

        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        assert!(
            verify(
                &proof,
                stacked.layout(),
                &roots,
                &at,
                &claimed,
                stacked.domain(),
                &config(),
                &mut other,
            )
            .is_err()
        );
    }

    /// The field tower in use: a base-field domain with extension-valued
    /// columns.
    #[test]
    fn the_stack_settles_over_a_field_tower() {
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
        type ExtE = FieldElement<Ext>;

        let num_vars = 3;
        let count = 5;
        // The columns are base-field, like a trace; the point is not.
        let columns: Vec<Mle<F>> = (0..count)
            .map(|i| {
                Mle::new(
                    (0..(1u64 << num_vars))
                        .map(|r| FE::from(r * 7 + 1 + i as u64 * 100))
                        .collect(),
                )
                .unwrap()
            })
            .collect();
        let layout = StackedLayout::build(&vec![num_vars; count], 6).unwrap();
        let at: Vec<ExtE> = (0..num_vars).map(|i| ExtE::from(101 + i as u64)).collect();
        let claimed: Vec<ExtE> = columns
            .iter()
            .map(|c| c.evaluate_in(&at).unwrap())
            .collect();

        let stacked = StackedCommitment::<F>::commit(layout, &columns, &config()).unwrap();
        let roots = stacked.roots();
        assert_eq!(roots.len(), 1);

        let mut prover = DefaultTranscript::<Ext>::new(b"tower");
        let proof = prove::<F, Ext, _>(&stacked, &at, &claimed, &config(), &mut prover).unwrap();

        let mut verifier = DefaultTranscript::<Ext>::new(b"tower");
        verify::<F, Ext, _>(
            &proof,
            stacked.layout(),
            &roots,
            &at,
            &claimed,
            stacked.domain(),
            &config(),
            &mut verifier,
        )
        .unwrap();
    }
}
