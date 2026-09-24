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

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    Error, challenge_powers,
    eq::{eq_eval, eq_evals_into},
    mle::Mle,
    stacking::{Placement, StackedLayout},
    whir::Domain,
    whir_chain::{self, ChainConfig, ChainProof},
    whir_commit::{CodewordCommitment, Commitment},
    whir_hash::WhirHash,
};

/// The stacked polynomials, committed. Base-field, like the trace they hold.
pub struct StackedCommitment<F: IsFFTField + IsPrimeField + 'static, H: WhirHash>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    layout: StackedLayout,
    commitments: Vec<CodewordCommitment<F, H>>,
    domain: Domain<F>,
    /// The room the commits and the openings take turns with, promised once
    /// for the whole group. Lives as long as the commitments do, because the
    /// openings are the last thing that uses it.
    _room: Option<crate::gpu::DeviceRoom>,
}

impl<F: IsFFTField + IsPrimeField + Send + Sync + 'static, H: WhirHash> StackedCommitment<F, H>
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
        columns: &[&Mle<F>],
        resident: Option<(&crate::gpu::ResidentColumns, usize)>,
        config: &ChainConfig,
    ) -> Result<Self, Error> {
        // The stacked polynomials are not built here: each is its columns at
        // their offsets, and both the commit and the opening write those
        // straight into the device's buffer. Assembling a copy on the way is a
        // pass over every byte about to be uploaded.
        let sources: Vec<whir_chain::Stacked<'_, F>> = (0..layout.num_polys())
            .map(|poly| whir_chain::Stacked {
                parts: layout
                    .parts_of(poly)
                    .into_iter()
                    .map(|(column, offset)| (columns[column], offset))
                    .collect(),
                resident: resident.map(|(store, first)| {
                    (
                        store,
                        layout
                            .parts_of(poly)
                            .into_iter()
                            .map(|(column, offset)| (first + column, offset))
                            .collect(),
                    )
                }),
                num_vars: layout.n_stack(),
            })
            .collect();
        // The commits and the openings take turns with the same working set —
        // two commits in flight, then one opening at a time — so the group
        // promises one turn's worth rather than every polynomial promising its
        // own. For eight of them that is nine codewords of room instead of
        // sixteen, and what the difference buys is the widest tables getting a
        // device at all. If the card will not promise it, each commitment
        // promises its own, which is the conservative accounting.
        //
        // ★ AND THE LEAF LAYERS ARE NOW IN THIS NUMBER, as a named term rather
        // than as a surprise. A commitment that is opened keeps the leaf layer
        // its commit hashed — `num_leaves × 32` = `C · 2^(5−k)` bytes at fold
        // width `k`, a QUARTER of a base codeword at the production `k = 4` —
        // so a group of `n` commitments holds `n · C/4` more than the codewords
        // alone. It is not reserved here, because the width is not known until
        // the tree is built: each codeword grows THIS reservation when it
        // captures (`DeviceCodeword::capture_leaves`), and `grow` refuses
        // without changing anything when the budget will not take it. A refused
        // retention costs the leaf pass again and nothing else.
        //
        // ⚠ A TREE is still not in this number and must never be. H4 kept the
        // whole node array — twice these bytes — and the card reached 96%, after
        // which commits fell back to the host at ~1.5 GiB each. The layer is
        // half of what that held and two thirds of what it saved.
        let room = sources.first().and_then(|poly| {
            let codeword_bytes = (1u64 << (poly.num_vars() + config.log_blowup)) * 8;
            crate::gpu::reserve_room(codeword_bytes)
        });
        let transient = room.is_none();
        // A commit spends most of its wall time waiting on a device — the tree
        // coming back — with the next polynomial's transform not yet launched.
        // Committing in pairs covers that wait and no more: a third in flight
        // has nothing left to hide behind, and each one holds a codeword and a
        // tree on the device while it runs. Every polynomial has the same
        // variable count, so they share a domain.
        let commit = |poly: &whir_chain::Stacked<'_, F>| {
            whir_chain::commit_stacked::<F, H>(poly, config, transient)
        };
        let mut domain = None;
        let mut commitments = Vec::with_capacity(sources.len());
        for pair in sources.chunks(2) {
            #[cfg(feature = "parallel")]
            let built = pair
                .par_iter()
                .map(commit)
                .collect::<Result<Vec<_>, Error>>()?;
            #[cfg(not(feature = "parallel"))]
            let built = pair.iter().map(commit).collect::<Result<Vec<_>, Error>>()?;
            for (commitment, d) in built {
                commitments.push(commitment);
                domain = Some(d);
            }
        }
        Ok(Self {
            layout,
            commitments,
            domain: domain.ok_or(Error::EmptyPolynomial)?,
            _room: room,
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

/// Where each column is claimed.
///
/// One table's sumcheck leaves every one of its columns at the **same** point,
/// which is [`Shared`](Self::Shared). Stacking several tables into one
/// commitment does not: each settles at its own, and the weight stops
/// factoring into slots times a single `eq`.
#[derive(Clone, Copy, Debug)]
pub enum Claimed<'a, E: IsField> {
    Shared(&'a [FieldElement<E>]),
    PerColumn(&'a [Vec<FieldElement<E>>]),
}

impl<E: IsField> Claimed<'_, E> {
    fn point(&self, column: usize) -> Result<&[FieldElement<E>], Error> {
        match self {
            Self::Shared(point) => Ok(point),
            Self::PerColumn(points) => {
                points
                    .get(column)
                    .map(Vec::as_slice)
                    .ok_or(Error::UnknownPolynomial {
                        index: column,
                        len: points.len(),
                    })
            }
        }
    }
}

/// The columns one stacked polynomial holds, as `(column index, placement)`.
fn columns_in(layout: &StackedLayout, poly: usize) -> impl Iterator<Item = (usize, &Placement)> {
    layout
        .placements()
        .iter()
        .enumerate()
        .filter(move |(_, place)| place.poly == poly)
}

/// One column's share of a stacked polynomial's weight: the subcube it owns,
/// the point it is claimed at, and the scale it carries.
///
/// The weight is the sum of these, and each lands in its own range — which is
/// what lets whoever builds it write the cells where they go, here or on a
/// device.
pub struct WeightShare<'a, E: IsField> {
    pub offset: usize,
    pub point: &'a [FieldElement<E>],
    pub scale: FieldElement<E>,
}

/// The shares of `poly`'s weight, one per column it holds.
fn weight_shares<'a, E: IsField>(
    layout: &StackedLayout,
    poly: usize,
    points: &'a Claimed<'a, E>,
    weights: &'a [FieldElement<E>],
) -> Result<Vec<WeightShare<'a, E>>, Error> {
    columns_in(layout, poly)
        .map(|(column, place)| {
            let point = points.point(column)?;
            if point.len() != place.num_vars {
                return Err(Error::VariableCountMismatch {
                    expected: place.num_vars,
                    got: point.len(),
                });
            }
            Ok(WeightShare {
                offset: place.offset,
                point,
                scale: weights[column].clone(),
            })
        })
        .collect()
}

/// `Σ_i gamma^i·eq(prefix_i ‖ point_i, ·)` as a table.
///
/// Each column owns a contiguous subcube of the stack, so its share of the
/// weight is written straight into that range and the gaps stay zero. Columns
/// of different heights cost nothing extra here — the range is just shorter.
pub fn weight_table<E: IsField + 'static>(
    shares: &[WeightShare<'_, E>],
    n_stack: usize,
) -> Result<Mle<E>, Error>
where
    FieldElement<E>: Send + Sync,
{
    let mut table = vec![FieldElement::<E>::zero(); 1usize << n_stack];
    for share in shares {
        let cells = 1usize << share.point.len();
        eq_evals_into(
            share.point,
            &share.scale,
            &mut table[share.offset..share.offset + cells],
        )?;
    }
    Mle::new(table)
}

/// The same weight at an arbitrary point, in closed form.
///
/// A column's prefix picks its subcube out of the stack, so its term is the
/// prefix indicator at the high variables times `eq(point_i, ·)` at the low
/// ones — and how many are "high" is the column's own business, which is what
/// lets heights differ.
fn weight_at<E: IsField>(
    layout: &StackedLayout,
    poly: usize,
    points: &Claimed<'_, E>,
    weights: &[FieldElement<E>],
    at: &[FieldElement<E>],
) -> Result<FieldElement<E>, Error> {
    if at.len() != layout.n_stack() {
        return Err(Error::VariableCountMismatch {
            expected: layout.n_stack(),
            got: at.len(),
        });
    }
    let mut total = FieldElement::<E>::zero();
    for (column, place) in columns_in(layout, poly) {
        let corner: Vec<FieldElement<E>> = place
            .prefix_bits()
            .into_iter()
            .map(|b| {
                if b {
                    FieldElement::<E>::one()
                } else {
                    FieldElement::<E>::zero()
                }
            })
            .collect();
        let (high, low) = at.split_at(corner.len());
        total += &weights[column] * eq_eval(&corner, high)? * eq_eval(points.point(column)?, low)?;
    }
    Ok(total)
}

/// The claimed sum for one stacked polynomial: its columns' values, batched.
fn claimed<E: IsField>(
    layout: &StackedLayout,
    poly: usize,
    values: &[FieldElement<E>],
    weights: &[FieldElement<E>],
) -> Result<FieldElement<E>, Error> {
    Ok(
        columns_in(layout, poly).fold(FieldElement::<E>::zero(), |acc, (column, _)| {
            acc + &weights[column] * &values[column]
        }),
    )
}

/// Proves that every column takes its claimed value at the shared `point`.
///
/// The claims are absorbed before the batching challenge, so the prover cannot
/// pick them after seeing it.
pub fn prove<F, E, T, H>(
    stacked: &StackedCommitment<F, H>,
    columns: &[&Mle<F>],
    resident: Option<(&crate::gpu::ResidentColumns, usize)>,
    point: &Claimed<'_, E>,
    values: &[FieldElement<E>],
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<StackedProof<F, E>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: IsTranscript<E>,
    H: WhirHash,
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

    let mut polys = Vec::with_capacity(stacked.commitments.len());
    for (i, commitment) in stacked.commitments.iter().enumerate() {
        // The polynomial is its columns at their offsets, handed over as they
        // are — see `StackedCommitment::commit`.
        let poly = whir_chain::Stacked {
            parts: layout
                .parts_of(i)
                .into_iter()
                .map(|(column, offset)| (columns[column], offset))
                .collect(),
            resident: resident.map(|(store, first)| {
                (
                    store,
                    layout
                        .parts_of(i)
                        .into_iter()
                        .map(|(column, offset)| (first + column, offset))
                        .collect(),
                )
            }),
            num_vars: layout.n_stack(),
        };
        // The weight goes down as its shares: a device writes them into its own
        // buffer, and the host materializes the table only if none does.
        polys.push(whir_chain::prove_shared::<F, E, T, H>(
            &poly,
            &weight_shares(layout, i, point, &weights)?,
            layout.n_stack(),
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
pub fn verify<F, E, T, H>(
    proof: &StackedProof<F, E>,
    layout: &StackedLayout,
    roots: &[Commitment],
    point: &Claimed<'_, E>,
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
    H: WhirHash,
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
        whir_chain::verify_weighted::<F, E, T, _, H>(
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

    use crate::{whir_chain::GrindBits, whir_hash::KeccakWhir};

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
            format: crate::whir_chain::ChainFormat::DEFAULT,
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
        let stacked = StackedCommitment::<F, KeccakWhir>::commit(
            layout,
            &crate::stacking::borrow(columns),
            None,
            &config(),
        )?;
        let roots = stacked.roots();
        let proof = prove(
            &stacked,
            &crate::stacking::borrow(columns),
            None,
            &Claimed::Shared(at),
            claimed,
            &config(),
            &mut transcript(),
        )?;

        verify::<F, _, _, KeccakWhir>(
            &proof,
            stacked.layout(),
            &roots,
            &Claimed::Shared(at),
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

        let claimed = Claimed::Shared(&at);
        let shares = weight_shares(&layout, 0, &claimed, &weights).unwrap();
        let table = weight_table(&shares, layout.n_stack()).unwrap();
        let off_cube: Vec<FE> = (0..n_stack).map(|i| FE::from(31 + i as u64)).collect();
        assert_eq!(
            table.evaluate(&off_cube).unwrap(),
            weight_at(&layout, 0, &Claimed::Shared(&at), &weights, &off_cube).unwrap()
        );
    }

    /// And the weight really does read each column out of its own subcube.
    #[test]
    fn the_weight_picks_each_column_out_of_its_slot() {
        let num_vars = 2;
        let (layout, columns) = stack_of(3, num_vars, 4);
        let at = point(num_vars);
        let weights: Vec<FE> = (0..3).map(|i| FE::from(3 + i as u64)).collect();

        let stacked = layout.stack(&crate::stacking::borrow(&columns)).unwrap();
        let claimed = Claimed::Shared(&at);
        let shares = weight_shares(&layout, 0, &claimed, &weights).unwrap();
        let table = weight_table(&shares, layout.n_stack()).unwrap();
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
            weight_shares(&layout, 0, &Claimed::Shared(&point(3)), &weights).err(),
            Some(Error::VariableCountMismatch { .. })
        ));
    }

    #[test]
    fn a_value_per_column_is_required() {
        let num_vars = 3;
        let (layout, columns) = stack_of(4, num_vars, 5);
        let at = point(num_vars);
        let claimed = values(&columns, &at);

        let stacked = StackedCommitment::<F, KeccakWhir>::commit(
            layout,
            &crate::stacking::borrow(&columns),
            None,
            &config(),
        )
        .unwrap();
        assert!(matches!(
            prove(
                &stacked,
                &crate::stacking::borrow(&columns),
                None,
                &Claimed::Shared(&at),
                &claimed[..3],
                &config(),
                &mut transcript()
            )
            .err(),
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

        let stacked = StackedCommitment::<F, KeccakWhir>::commit(
            layout,
            &crate::stacking::borrow(&columns),
            None,
            &config(),
        )
        .unwrap();
        let roots = stacked.roots();
        let proof = prove(
            &stacked,
            &crate::stacking::borrow(&columns),
            None,
            &Claimed::Shared(&at),
            &claimed,
            &config(),
            &mut transcript(),
        )
        .unwrap();

        let mut other = DefaultTranscript::<F>::new(b"a-different-statement");
        assert!(
            verify::<F, _, _, KeccakWhir>(
                &proof,
                stacked.layout(),
                &roots,
                &Claimed::Shared(&at),
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

        let stacked = StackedCommitment::<F, KeccakWhir>::commit(
            layout,
            &crate::stacking::borrow(&columns),
            None,
            &config(),
        )
        .unwrap();
        let roots = stacked.roots();
        assert_eq!(roots.len(), 1);

        let mut prover = DefaultTranscript::<Ext>::new(b"tower");
        let proof = prove::<F, Ext, _, KeccakWhir>(
            &stacked,
            &crate::stacking::borrow(&columns),
            None,
            &Claimed::Shared(&at),
            &claimed,
            &config(),
            &mut prover,
        )
        .unwrap();

        let mut verifier = DefaultTranscript::<Ext>::new(b"tower");
        verify::<F, Ext, _, KeccakWhir>(
            &proof,
            stacked.layout(),
            &roots,
            &Claimed::Shared(&at),
            &claimed,
            stacked.domain(),
            &config(),
            &mut verifier,
        )
        .unwrap();
    }

    /// Columns of **different heights**, each claimed at **its own point**,
    /// settled against one commitment.
    ///
    /// This is what stacking separate tables together needs and what a shared
    /// point cannot express: their sumchecks end wherever they end. The
    /// weight stops factoring into slots times a single `eq`, so the check
    /// that it still describes the same claim is the whole point of the test.
    #[test]
    fn columns_of_different_heights_settle_at_their_own_points() {
        let columns = vec![column(3, 7), column(1, 11), column(2, 13)];
        let n_stack = 4;
        let layout = StackedLayout::build(&[3, 1, 2], n_stack).unwrap();

        // A point per column, each of that column's own width.
        let points: Vec<Vec<FE>> = vec![
            vec![FE::from(2), FE::from(3), FE::from(5)],
            vec![FE::from(7)],
            vec![FE::from(11), FE::from(13)],
        ];
        let claimed: Vec<FE> = columns
            .iter()
            .zip(&points)
            .map(|(c, p)| c.evaluate(p).unwrap())
            .collect();

        let stacked = StackedCommitment::<F, KeccakWhir>::commit(
            layout,
            &crate::stacking::borrow(&columns),
            None,
            &config(),
        )
        .unwrap();
        let roots = stacked.roots();
        let at = Claimed::PerColumn(&points);

        let proof = prove(
            &stacked,
            &crate::stacking::borrow(&columns),
            None,
            &at,
            &claimed,
            &config(),
            &mut transcript(),
        )
        .unwrap();
        verify::<F, _, _, KeccakWhir>(
            &proof,
            stacked.layout(),
            &roots,
            &at,
            &claimed,
            stacked.domain(),
            &config(),
            &mut transcript(),
        )
        .unwrap();

        // And a wrong value for one column is rejected, so the per-point
        // weight is really tying each claim to its own column.
        let mut tampered = claimed.clone();
        tampered[1] += FE::one();
        let proof = prove(
            &stacked,
            &crate::stacking::borrow(&columns),
            None,
            &at,
            &tampered,
            &config(),
            &mut transcript(),
        )
        .unwrap();
        assert!(
            verify::<F, _, _, KeccakWhir>(
                &proof,
                stacked.layout(),
                &roots,
                &at,
                &tampered,
                stacked.domain(),
                &config(),
                &mut transcript(),
            )
            .is_err()
        );
    }
}
