//! WHIR's commitment-time out-of-domain sample, on a stack.
//!
//! WHIR's commitment does not end at the Merkle root. The verifier then draws
//! an out-of-domain point `z0` and the prover answers the committed
//! polynomial's value there, `y = F(ood_point(z0))`. A committed word may be
//! close to more than one codeword; the answer names one of them, since two
//! distinct codewords agree at a random point outside the domain only with
//! probability `(2^n − 1)/|E|`. Every challenge drawn after the answers then
//! argues about one codeword per committed word.
//!
//! The opening carries the answer as one more claim, which is the successor
//! rounds' out-of-domain step ([`crate::whir_chain`]) moved to round 0: with a
//! batching challenge `γ` drawn after the answer, the chain proves
//!
//! ```text
//! claim + γ·y   under the weight   w + γ·eq(ood_point(z0), ·)
//! ```
//!
//! What lives here:
//! - the answers of a stack from its columns' own values: a stacked polynomial
//!   is zero outside its columns, so `F_i(p) = Σ_c eq(corner_c, p_high)·c(p_low)`
//!   over the columns `c` it holds, `p_low` the last `h_c` coordinates of `p`
//!   ([`answers`]);
//! - the overlay one chain polynomial carries ([`Overlay`]);
//! - the overlay's share of the first sumcheck rounds while the opening's
//!   weight is still in share form on a card ([`PendingOverlay`]). That share
//!   needs `F(q ‖ p[K..])` for the `2^K` Boolean prefixes `q`, which the same
//!   column values give ([`partial`]), so the card never writes
//!   `γ·eq(p, ·)` at the stack's full width.

use math::field::{element::FieldElement, traits::IsField};

use crate::{Error, eq::eq_evals, mle::Mle, stacking::StackedLayout, whir_chain::MAX_FOLD};

/// `eq(a, x)` for one variable: `(1 − a)(1 − x) + a·x`.
fn eq1<E: IsField>(a: &FieldElement<E>, x: &FieldElement<E>) -> FieldElement<E> {
    let one = FieldElement::<E>::one();
    (&one - a) * (&one - x) + a * x
}

/// `eq(point, bits)` for Boolean `bits`: the product of `point_j` where the bit
/// is set and `1 − point_j` where it is not.
fn eq_bits<E: IsField>(point: &[FieldElement<E>], bits: &[bool]) -> FieldElement<E> {
    point
        .iter()
        .zip(bits)
        .fold(FieldElement::<E>::one(), |acc, (p, &bit)| {
            if bit {
                acc * p
            } else {
                acc * (FieldElement::<E>::one() - p)
            }
        })
}

/// Refuses `columns` unless it holds one value per column of `layout`, and
/// `point` unless it spans the stack.
fn check_shape<E: IsField>(
    layout: &StackedLayout,
    point: &[FieldElement<E>],
    columns: &[FieldElement<E>],
) -> Result<(), Error> {
    if point.len() != layout.n_stack() {
        return Err(Error::VariableCountMismatch {
            expected: layout.n_stack(),
            got: point.len(),
        });
    }
    if columns.len() != layout.placements().len() {
        return Err(Error::QueryCountMismatch {
            expected: layout.placements().len(),
            got: columns.len(),
        });
    }
    Ok(())
}

/// The point column `c` of `layout` is read at for the stack's value at
/// `point`: its last `h_c` coordinates.
pub fn column_point<'a, E: IsField>(
    layout: &StackedLayout,
    column: usize,
    point: &'a [FieldElement<E>],
) -> &'a [FieldElement<E>] {
    let vars = layout.placements()[column].num_vars;
    &point[point.len() - vars..]
}

/// ★ Every stacked polynomial's value at `point`, from `columns`: column `c`'s
/// value at [`column_point`], one per column of `layout` in placement order.
///
/// `F_i(point) = Σ_{c ∈ i} eq(corner_c, point[..n − h_c])·columns[c]`: the
/// closed form `stacked_eval`'s weight is evaluated in, read the other way.
pub fn answers<E: IsField>(
    layout: &StackedLayout,
    point: &[FieldElement<E>],
    columns: &[FieldElement<E>],
) -> Result<Vec<FieldElement<E>>, Error> {
    check_shape(layout, point, columns)?;
    let mut out = vec![FieldElement::<E>::zero(); layout.num_polys()];
    for (place, value) in layout.placements().iter().zip(columns) {
        let corner = place.prefix_bits();
        out[place.poly] += eq_bits(&point[..corner.len()], &corner) * value;
    }
    Ok(out)
}

/// The variables a [`partial`] table of polynomial `poly` may bind: at most
/// [`MAX_FOLD`], the widest a lean opening's first rounds run, and no more
/// than its shortest column prefix — a column taller than `2^(n − K)` spans
/// several of the table's prefixes, and its value at one of them is not a
/// value the commit evaluated.
pub fn partial_vars(layout: &StackedLayout, poly: usize) -> usize {
    layout
        .placements()
        .iter()
        .filter(|place| place.poly == poly)
        .map(|place| place.n_stack - place.num_vars)
        .fold(MAX_FOLD.min(layout.n_stack()), usize::min)
}

/// `F_poly(q ‖ point[k..])` for every Boolean prefix `q ∈ {0,1}^k`, the first
/// variable the most significant bit of the index (the [`Mle`] convention):
/// what [`PendingOverlay`] reads its rounds from. Folding it at `point[..k]`
/// gives the answer.
///
/// `k` at most [`partial_vars`]: each column then lies under one prefix, the
/// top `k` bits of its corner, and contributes
/// `eq(point[k..n − h_c], corner[k..])·columns[c]` there.
pub fn partial<E: IsField>(
    layout: &StackedLayout,
    poly: usize,
    point: &[FieldElement<E>],
    columns: &[FieldElement<E>],
    k: usize,
) -> Result<Vec<FieldElement<E>>, Error> {
    check_shape(layout, point, columns)?;
    if k > partial_vars(layout, poly) {
        return Err(Error::VariableCountMismatch {
            expected: partial_vars(layout, poly),
            got: k,
        });
    }
    let mut table = vec![FieldElement::<E>::zero(); 1usize << k];
    for (place, value) in layout.placements().iter().zip(columns) {
        if place.poly != poly {
            continue;
        }
        let corner = place.prefix_bits();
        let prefix = corner[..k]
            .iter()
            .fold(0usize, |acc, &bit| (acc << 1) | usize::from(bit));
        table[prefix] += eq_bits(&point[k..corner.len()], &corner[k..]) * value;
    }
    Ok(table)
}

/// One chain polynomial's commit-time claim, as its opening carries it:
/// `scale·eq(point, ·)` on its weight, and with it `scale·F(point)` on its
/// claim.
#[derive(Clone, Debug)]
pub struct Overlay<'a, E: IsField> {
    /// `ood_point(z0, n)`, the whole stack's variables.
    pub point: &'a [FieldElement<E>],
    /// The polynomial's batching weight, `γ^(i+1)`.
    pub scale: FieldElement<E>,
    /// [`partial`] at `K = log2(len)`: what a card's first `K` rounds need.
    pub partial: Vec<FieldElement<E>>,
}

impl<E: IsField + 'static> Overlay<'_, E>
where
    FieldElement<E>: Send + Sync,
{
    /// The variables [`partial`](Self::partial) spans.
    pub fn partial_vars(&self) -> usize {
        self.partial.len().trailing_zeros() as usize
    }

    /// `scale·eq(point, ·)` as a table, for a weight built on the host.
    pub fn table(&self) -> Result<Mle<E>, Error> {
        let mut table = vec![FieldElement::<E>::zero(); 1usize << self.point.len()];
        crate::eq::eq_evals_into(self.point, &self.scale, &mut table)?;
        Mle::new(table)
    }

    /// `weight + scale·eq(point, ·)`.
    pub fn add_to(&self, weight: &Mle<E>) -> Result<Mle<E>, Error> {
        let overlay = self.table()?;
        if overlay.len() != weight.len() {
            return Err(Error::VariableCountMismatch {
                expected: weight.num_vars(),
                got: self.point.len(),
            });
        }
        Mle::new(
            weight
                .evals()
                .iter()
                .zip(overlay.evals())
                .map(|(w, o)| w + o)
                .collect(),
        )
    }
}

/// An [`Overlay`] not yet written into its opening's weight: while the weight
/// is in share form on a card, each round's share of `scale·eq(point, ·)·F` is
/// computed here, from the [`partial`] table, and added to the card's round
/// evaluations; once the table's variables are bound — or the caller needs the
/// weight whole — what is left, `scale·eq(point[..b], r)·eq(point[b..], ·)`, is
/// added to the weight where it lies ([`residual`](Self::residual)).
///
/// Round `j` binds variable `j`. With the first `j` variables bound at `r` and
/// `A_x = F(r, x, point[j+1..])`, the overlay's round polynomial is
///
/// ```text
/// scale·eq(point[..j], r)·eq(point_j, X)·((1 − X)·A_0 + X·A_1)
/// ```
///
/// since `Σ_rest eq(point[j+1..], rest)·F(r, X, rest) = F(r, X, point[j+1..])`.
#[derive(Clone, Debug)]
pub struct PendingOverlay<E: IsField> {
    point: Vec<FieldElement<E>>,
    scale: FieldElement<E>,
    /// `F(r, ·, point[K..])` over the variables `bound..K`, the bound ones at
    /// their challenges.
    table: Vec<FieldElement<E>>,
    /// `eq(point[..bound], r)`.
    at: FieldElement<E>,
    bound: usize,
}

impl<E: IsField + 'static> PendingOverlay<E>
where
    FieldElement<E>: Send + Sync,
{
    pub fn new(overlay: &Overlay<'_, E>) -> Self {
        Self {
            point: overlay.point.to_vec(),
            scale: overlay.scale.clone(),
            table: overlay.partial.clone(),
            at: FieldElement::<E>::one(),
            bound: 0,
        }
    }

    /// Variables this can still take a round over.
    pub fn rounds_left(&self) -> usize {
        self.table.len().trailing_zeros() as usize
    }

    /// The overlay's round polynomial at `nodes`, for the next unbound
    /// variable. Refused when the table has none left.
    pub fn round(&self, nodes: &[FieldElement<E>]) -> Result<Vec<FieldElement<E>>, Error> {
        let left = self.rounds_left();
        if left == 0 {
            return Err(Error::NoVariablesLeft);
        }
        let half = self.table.len() / 2;
        let tail = eq_evals(&self.point[self.bound + 1..self.bound + left]);
        let (lo, hi) = self.table.split_at(half);
        let mut a0 = FieldElement::<E>::zero();
        let mut a1 = FieldElement::<E>::zero();
        for ((e, l), h) in tail.iter().zip(lo).zip(hi) {
            a0 += e * l;
            a1 += e * h;
        }
        let lead = &self.scale * &self.at;
        let p = &self.point[self.bound];
        Ok(nodes
            .iter()
            .map(|x| &lead * eq1(p, x) * (&a0 + x * (&a1 - &a0)))
            .collect())
    }

    /// Binds the next variable at `r`.
    pub fn bind(&mut self, r: &FieldElement<E>) -> Result<(), Error> {
        if self.rounds_left() == 0 {
            return Err(Error::NoVariablesLeft);
        }
        self.at = &self.at * eq1(&self.point[self.bound], r);
        let half = self.table.len() / 2;
        let (lo, hi) = self.table.split_at(half);
        self.table = lo.iter().zip(hi).map(|(l, h)| l + r * (h - l)).collect();
        self.bound += 1;
        Ok(())
    }

    /// What is left to add to the weight bound at the challenges so far:
    /// `scale·at·eq(point[bound..], ·)`, as `(point[bound..], scale·at)`.
    pub fn residual(&self) -> (&[FieldElement<E>], FieldElement<E>) {
        (&self.point[self.bound..], &self.scale * &self.at)
    }

    /// Variables bound so far.
    pub fn bound(&self) -> usize {
        self.bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::{Composed, SumcheckPolynomial},
        sumcheck,
    };
    use math::field::{
        extensions_goldilocks::Degree3GoldilocksExtensionField as Ext, goldilocks::GoldilocksField,
    };

    type F = GoldilocksField;
    type FE = FieldElement<F>;
    type EE = FieldElement<Ext>;

    fn ext(seed: u64) -> EE {
        EE::new([
            FE::from(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 3),
            FE::from(seed.wrapping_mul(0xc2b2_ae3d_27d4_eb4f) >> 5),
            FE::from(seed.wrapping_mul(0x1656_67b1_9e37_79f9) >> 7),
        ])
    }

    /// Mixed heights with gaps the stacking leaves zero: three polynomials of
    /// nine variables.
    fn stack() -> (StackedLayout, Vec<Mle<F>>) {
        let heights = [3usize, 5, 4, 2, 6, 5, 9, 7, 7];
        let layout = StackedLayout::build(&heights, 9).unwrap();
        let columns = heights
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                Mle::new(
                    (0..(1u64 << m))
                        .map(|r| {
                            FE::from(
                                r.wrapping_add(31 * i as u64 + 5)
                                    .wrapping_mul(6364136223846793005)
                                    >> 9,
                            )
                        })
                        .collect(),
                )
                .unwrap()
            })
            .collect();
        (layout, columns)
    }

    fn product(v: &[EE]) -> EE {
        &v[0] * &v[1]
    }

    fn lifted(f: &Mle<F>) -> Mle<Ext> {
        Mle::new(f.evals().iter().map(|v| v.to_extension::<Ext>()).collect()).unwrap()
    }

    fn column_values(layout: &StackedLayout, columns: &[Mle<F>], point: &[EE]) -> Vec<EE> {
        columns
            .iter()
            .enumerate()
            .map(|(c, column)| column.evaluate_in(column_point(layout, c, point)).unwrap())
            .collect()
    }

    #[test]
    fn the_answers_are_the_stacks_values_at_the_point() {
        let (layout, columns) = stack();
        assert_eq!(layout.num_polys(), 3, "the stack this test means");
        let point = crate::whir_chain::ood_point(&ext(0x5eed), layout.n_stack());
        let values = column_values(&layout, &columns, &point);
        let got = answers(&layout, &point, &values).unwrap();
        let stacked = layout.stack(&crate::stacking::borrow(&columns)).unwrap();
        for (poly, (answer, f)) in got.iter().zip(&stacked).enumerate() {
            assert_eq!(*answer, f.evaluate_in(&point).unwrap(), "polynomial {poly}");
        }
        assert!(
            answers(&layout, &point, &values[1..]).is_err(),
            "one value per column"
        );
    }

    #[test]
    fn the_partial_table_is_the_stack_at_each_prefix_and_folds_to_the_answer() {
        let (layout, columns) = stack();
        let point = crate::whir_chain::ood_point(&ext(0xab), layout.n_stack());
        let values = column_values(&layout, &columns, &point);
        let stacked = layout.stack(&crate::stacking::borrow(&columns)).unwrap();
        let ys = answers(&layout, &point, &values).unwrap();
        for poly in 0..layout.num_polys() {
            let k = partial_vars(&layout, poly);
            let table = partial(&layout, poly, &point, &values, k).unwrap();
            for (q, value) in table.iter().enumerate() {
                let mut at: Vec<EE> = (0..k)
                    .map(|j| {
                        if (q >> (k - 1 - j)) & 1 == 1 {
                            EE::one()
                        } else {
                            EE::zero()
                        }
                    })
                    .collect();
                at.extend_from_slice(&point[k..]);
                assert_eq!(
                    *value,
                    stacked[poly].evaluate_in(&at).unwrap(),
                    "poly {poly} prefix {q}"
                );
            }
            assert_eq!(
                Mle::new(table).unwrap().evaluate(&point[..k]).unwrap(),
                ys[poly]
            );
            assert!(
                partial(&layout, poly, &point, &values, k + 1).is_err(),
                "no wider than its shortest column prefix"
            );
        }
        // The polynomial holding the 2^9 column fills the stack: no prefix.
        assert_eq!(partial_vars(&layout, 1), 0);
    }

    /// ★ The pending overlay's rounds are the overlay's share of the host's
    /// rounds over `(w + scale·eq(p, ·))·F`, and what is left after them is
    /// the overlay's bound weight: so a card adding them to its rounds over `w`
    /// proves what the host proves over the whole weight.
    #[test]
    fn the_pending_rounds_are_the_overlays_share_of_the_whole_weight() {
        let (layout, columns) = stack();
        let stacked = layout.stack(&crate::stacking::borrow(&columns)).unwrap();
        let point = crate::whir_chain::ood_point(&ext(0x77), layout.n_stack());
        let values = column_values(&layout, &columns, &point);
        let poly = 0;
        let k = partial_vars(&layout, poly);
        assert!(k >= 3, "the test wants a few rounds");
        let overlay = Overlay {
            point: &point,
            scale: ext(0x5ca1e),
            partial: partial(&layout, poly, &point, &values, k).unwrap(),
        };
        let message = lifted(&stacked[poly]);
        let weight = Mle::new((0..message.len() as u64).map(ext).collect()).unwrap();

        let mut whole = Composed::new(
            vec![overlay.add_to(&weight).unwrap(), message.clone()],
            product,
            2,
        )
        .unwrap();
        let mut shares = Composed::new(vec![weight, message], product, 2).unwrap();
        let mut pending = PendingOverlay::new(&overlay);
        let nodes = [EE::one(), EE::from(2u64)];
        for round in 0..k {
            let want = sumcheck::round_evaluations(&whole, 2, false);
            let base = sumcheck::round_evaluations(&shares, 2, false);
            let extra = pending.round(&nodes).unwrap();
            let got: Vec<EE> = base.iter().zip(&extra).map(|(b, e)| b + e).collect();
            assert_eq!(got, want, "round {round}");
            let r = ext(1000 + round as u64);
            whole.fix_first_variable(&r).unwrap();
            shares.fix_first_variable(&r).unwrap();
            pending.bind(&r).unwrap();
        }
        assert!(pending.round(&nodes).is_err(), "the table is spent");
        // The residual completes the share-form weight to the whole one.
        let (rest, scale) = pending.residual();
        let mut factors = shares.into_polys();
        let bound_shares = factors.remove(0);
        let completed = Overlay {
            point: rest,
            scale,
            partial: Vec::new(),
        }
        .add_to(&bound_shares)
        .unwrap();
        assert_eq!(completed.evals(), whole.into_polys().remove(0).evals());
    }
}
