//! Multilinear extensions, held as their `2^n` hypercube evaluations.
//!
//! Index `i` is read with **variable 0 as the most significant bit**. Every
//! fold in this crate assumes that.

use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::Error;

/// A multilinear polynomial held by its hypercube evaluations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mle<F: IsField> {
    evals: Vec<FieldElement<F>>,
    num_vars: usize,
}

impl<F: IsField + 'static> Mle<F> {
    /// Builds an MLE from `2^n` evaluations in hypercube order.
    pub fn new(evals: Vec<FieldElement<F>>) -> Result<Self, Error> {
        let len = evals.len();
        if !len.is_power_of_two() {
            return Err(Error::NotPowerOfTwo(len));
        }
        Ok(Self {
            num_vars: len.trailing_zeros() as usize,
            evals,
        })
    }

    /// The constant polynomial on zero variables.
    pub fn constant(value: FieldElement<F>) -> Self {
        Self {
            evals: vec![value],
            num_vars: 0,
        }
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        self.evals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.evals.is_empty()
    }

    pub fn evals(&self) -> &[FieldElement<F>] {
        &self.evals
    }

    pub fn into_evals(self) -> Vec<FieldElement<F>> {
        self.evals
    }

    /// Fixes variable 0 to `r`, returning a polynomial on `n - 1` variables.
    ///
    /// `new[j] = (1 - r)·old[j] + r·old[j + 2^(n-1)]`, which is the multilinear
    /// interpolation between the two halves of the table.
    pub fn fix_first_variable(&self, r: &FieldElement<F>) -> Result<Self, Error> {
        if self.num_vars == 0 {
            return Err(Error::NoVariablesLeft);
        }
        let half = self.evals.len() / 2;
        let evals = (0..half)
            .map(|j| {
                let lo = &self.evals[j];
                let hi = &self.evals[j + half];
                // lo + r·(hi - lo) — one multiplication instead of two.
                lo + r * &(hi - lo)
            })
            .collect();
        Ok(Self {
            evals,
            num_vars: self.num_vars - 1,
        })
    }

    /// Fixes variable 0 in place. Same arithmetic as [`Self::fix_first_variable`],
    /// but reuses the allocation — sumcheck folds once per round.
    pub fn fix_first_variable_in_place(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        if self.num_vars == 0 {
            return Err(Error::NoVariablesLeft);
        }
        let half = self.evals.len() / 2;
        let (lo, hi) = self.evals.split_at_mut(half);
        // Every index is independent, and the halves are disjoint slices, so the
        // split is what lets the rows go out to the pool at all.
        let fold = |(a, b): (&mut FieldElement<F>, &FieldElement<F>)| *a = &*a + r * &(b - &*a);
        #[cfg(feature = "parallel")]
        if half >= crate::SERIAL_BELOW {
            lo.par_iter_mut().zip(hi.par_iter()).for_each(fold);
        } else {
            lo.iter_mut().zip(hi.iter()).for_each(fold);
        }
        #[cfg(not(feature = "parallel"))]
        lo.iter_mut().zip(hi.iter()).for_each(fold);
        self.evals.truncate(half);
        self.num_vars -= 1;
        Ok(())
    }

    /// Fixes the **last** variable to `r`, returning a polynomial on `n - 1`
    /// variables.
    ///
    /// The last variable is the low bit of the index, so this pairs `2j` with
    /// `2j + 1`. Codeword folding binds variables from this end, which is why
    /// it exists alongside [`Self::fix_first_variable_in_place`].
    pub fn fix_last_variable_in_place(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        if self.num_vars == 0 {
            return Err(Error::NoVariablesLeft);
        }
        let half = self.evals.len() / 2;
        for j in 0..half {
            let lo = self.evals[2 * j].clone();
            let delta = &self.evals[2 * j + 1] - &lo;
            self.evals[j] = lo + r * &delta;
        }
        self.evals.truncate(half);
        self.num_vars -= 1;
        Ok(())
    }

    /// Evaluates the extension at an arbitrary point in `F^n`.
    pub fn evaluate(&self, point: &[FieldElement<F>]) -> Result<FieldElement<F>, Error> {
        if point.len() != self.num_vars {
            return Err(Error::VariableCountMismatch {
                expected: self.num_vars,
                got: point.len(),
            });
        }
        Self::evaluate_at(&self.evals, point)
    }

    /// The extension of `evals` at `point`, without owning an [`Mle`].
    ///
    /// The first fold reads the slice and writes the half-size buffer the rest
    /// fold in place, so the table is never copied at full width. A caller
    /// holding a slice of a bigger table — a GKR layer's half, say — evaluates
    /// it without materializing it at all.
    pub fn evaluate_at(
        evals: &[FieldElement<F>],
        point: &[FieldElement<F>],
    ) -> Result<FieldElement<F>, Error> {
        if evals.len() != 1usize << point.len() {
            return Err(Error::VariableCountMismatch {
                expected: point.len(),
                got: evals.len().trailing_zeros() as usize,
            });
        }
        if let Some(value) = crate::gpu::evaluate_mle(evals, point) {
            return Ok(value);
        }
        let Some((first, rest)) = point.split_first() else {
            return Ok(evals[0].clone());
        };

        let half = evals.len() / 2;
        let (lo, hi) = evals.split_at(half);
        let combine = |(l, h): (&FieldElement<F>, &FieldElement<F>)| l + first * &(h - l);
        #[cfg(feature = "parallel")]
        let mut current: Vec<FieldElement<F>> = if half >= crate::SERIAL_BELOW {
            lo.par_iter().zip(hi.par_iter()).map(combine).collect()
        } else {
            lo.iter().zip(hi.iter()).map(combine).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let mut current: Vec<FieldElement<F>> = lo.iter().zip(hi.iter()).map(combine).collect();

        for r in rest {
            let half = current.len() / 2;
            let (lo, hi) = current.split_at_mut(half);
            let fold = |(a, b): (&mut FieldElement<F>, &FieldElement<F>)| *a = &*a + r * &(b - &*a);
            #[cfg(feature = "parallel")]
            if half >= crate::SERIAL_BELOW {
                lo.par_iter_mut().zip(hi.par_iter()).for_each(fold);
            } else {
                lo.iter_mut().zip(hi.iter()).for_each(fold);
            }
            #[cfg(not(feature = "parallel"))]
            lo.iter_mut().zip(hi.iter()).for_each(fold);
            current.truncate(half);
        }
        Ok(current.swap_remove(0))
    }

    /// The extension at a point in a **larger** field.
    ///
    /// A trace column lives in the base field while the challenges do not, so
    /// this is how a committed column answers a claim: the first fold lifts,
    /// the rest stay up. Lifting the whole table first would instead cost its
    /// size times the extension degree.
    pub fn evaluate_in<E>(&self, point: &[FieldElement<E>]) -> Result<FieldElement<E>, Error>
    where
        F: IsSubFieldOf<E>,
        E: IsField + 'static,
    {
        if point.len() != self.num_vars {
            return Err(Error::VariableCountMismatch {
                expected: self.num_vars,
                got: point.len(),
            });
        }
        if let Some(value) = crate::gpu::evaluate_mle(&self.evals, point) {
            return Ok(value);
        }
        let Some((first, rest)) = point.split_first() else {
            return Ok(self.evals[0].clone().to_extension::<E>());
        };

        let half = self.evals.len() / 2;
        let mut current: Vec<FieldElement<E>> = (0..half)
            .map(|j| {
                let lo = &self.evals[j];
                let hi = &self.evals[j + half];
                // The base element on the left: the only direction the tower
                // gives.
                lo.clone().to_extension::<E>() + (hi - lo) * first
            })
            .collect();

        for r in rest {
            let half = current.len() / 2;
            for j in 0..half {
                let delta = &current[j + half] - &current[j];
                current[j] = &current[j] + r * &delta;
            }
            current.truncate(half);
        }
        Ok(current.into_iter().next().expect("one value remains"))
    }

    /// Several columns' extensions at one point in a larger field.
    ///
    /// [`evaluate_in`](Self::evaluate_in) folds each column on its own — half
    /// the cube lifted and then folded down, per column. Sharing the point,
    /// the `eq(point, ·)` table is built **once** and each column is a dot
    /// product against it, a base-by-extension multiplication per nonzero
    /// cell; a preprocessed table is mostly zeros, and those cost nothing.
    pub fn evaluate_many_in<E>(
        columns: &[&Mle<F>],
        point: &[FieldElement<E>],
    ) -> Result<Vec<FieldElement<E>>, Error>
    where
        F: IsSubFieldOf<E>,
        E: IsField + 'static,
        FieldElement<E>: Send + Sync,
    {
        if let Some(column) = columns.iter().find(|c| c.num_vars != point.len()) {
            return Err(Error::VariableCountMismatch {
                expected: column.num_vars,
                got: point.len(),
            });
        }
        let eq = crate::eq::eq_evals(point);
        let zero = FieldElement::<F>::zero();
        Ok(columns
            .iter()
            .map(|column| {
                column
                    .evals
                    .iter()
                    .zip(&eq)
                    .filter(|(v, _)| **v != zero)
                    .fold(FieldElement::<E>::zero(), |acc, (v, e)| acc + v * e)
            })
            .collect())
    }

    /// The single remaining evaluation, once every variable has been fixed.
    pub fn as_constant(&self) -> Option<&FieldElement<F>> {
        (self.num_vars == 0).then(|| &self.evals[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    fn mle(vals: &[u64]) -> Mle<F> {
        Mle::new(vals.iter().map(|v| FE::from(*v)).collect()).unwrap()
    }

    #[test]
    fn rejects_non_power_of_two() {
        let evals: Vec<FE> = (0..3).map(FE::from).collect();
        assert_eq!(Mle::new(evals).unwrap_err(), Error::NotPowerOfTwo(3));
    }

    #[test]
    fn num_vars_is_log2_of_the_table() {
        assert_eq!(mle(&[1]).num_vars(), 0);
        assert_eq!(mle(&[1, 2]).num_vars(), 1);
        assert_eq!(mle(&[1, 2, 3, 4]).num_vars(), 2);
        assert_eq!(mle(&[0; 256]).num_vars(), 8);
    }

    #[test]
    fn agrees_with_the_table_on_hypercube_corners() {
        // f(x0, x1) with x0 the high bit: [f(00), f(01), f(10), f(11)]
        let f = mle(&[7, 11, 13, 17]);
        for (i, expected) in [7u64, 11, 13, 17].iter().enumerate() {
            let x0 = FE::from(((i >> 1) & 1) as u64);
            let x1 = FE::from((i & 1) as u64);
            assert_eq!(f.evaluate(&[x0, x1]).unwrap(), FE::from(*expected));
        }
    }

    #[test]
    fn is_multilinear_in_each_variable() {
        // A multilinear polynomial is affine along every axis, so the midpoint
        // evaluation is the average of the two endpoints.
        let f = mle(&[7, 11, 13, 17]);
        let two_inv = FE::from(2).inv().unwrap();
        let x1 = FE::from(5);

        let at_0 = f.evaluate(&[FE::zero(), x1]).unwrap();
        let at_1 = f.evaluate(&[FE::one(), x1]).unwrap();
        let at_mid = f.evaluate(&[two_inv, x1]).unwrap();

        assert_eq!(at_mid, (at_0 + at_1) * two_inv);
    }

    #[test]
    fn fixing_a_variable_matches_evaluating_it() {
        let f = mle(&[3, 5, 8, 13, 21, 34, 55, 89]);
        let r = FE::from(42);
        let folded = f.fix_first_variable(&r).unwrap();

        assert_eq!(folded.num_vars(), 2);
        for a in 0..2u64 {
            for b in 0..2u64 {
                let rest = [FE::from(a), FE::from(b)];
                let via_fold = folded.evaluate(&rest).unwrap();
                let direct = f.evaluate(&[r, FE::from(a), FE::from(b)]).unwrap();
                assert_eq!(via_fold, direct);
            }
        }
    }

    #[test]
    fn in_place_fold_matches_the_allocating_one() {
        let f = mle(&[3, 5, 8, 13, 21, 34, 55, 89]);
        let r = FE::from(9);
        let expected = f.fix_first_variable(&r).unwrap();

        let mut g = f;
        g.fix_first_variable_in_place(&r).unwrap();
        assert_eq!(g, expected);
    }

    #[test]
    fn folding_every_variable_leaves_the_evaluation() {
        let f = mle(&[3, 5, 8, 13]);
        let point = [FE::from(6), FE::from(7)];
        let expected = f.evaluate(&point).unwrap();

        let mut g = f;
        for r in &point {
            g.fix_first_variable_in_place(r).unwrap();
        }
        assert_eq!(g.num_vars(), 0);
        assert_eq!(g.as_constant().unwrap(), &expected);
    }

    #[test]
    fn folding_a_constant_is_an_error() {
        let mut f = Mle::<F>::constant(FE::from(4));
        assert_eq!(
            f.fix_first_variable_in_place(&FE::from(1)).unwrap_err(),
            Error::NoVariablesLeft
        );
    }

    #[test]
    fn evaluate_rejects_a_point_of_the_wrong_arity() {
        let f = mle(&[1, 2, 3, 4]);
        assert_eq!(
            f.evaluate(&[FE::from(1)]).unwrap_err(),
            Error::VariableCountMismatch {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn fixing_the_last_variable_matches_evaluating_it() {
        let f = mle(&[3, 5, 8, 13, 21, 34, 55, 89]);
        let r = FE::from(23);
        let mut folded = f.clone();
        folded.fix_last_variable_in_place(&r).unwrap();

        assert_eq!(folded.num_vars(), 2);
        for a in 0..2u64 {
            for b in 0..2u64 {
                let via_fold = folded.evaluate(&[FE::from(a), FE::from(b)]).unwrap();
                let direct = f.evaluate(&[FE::from(a), FE::from(b), r]).unwrap();
                assert_eq!(via_fold, direct, "a={a}, b={b}");
            }
        }
    }

    #[test]
    fn evaluating_in_a_larger_field_matches_lifting_first() {
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
        type ExtE = FieldElement<Ext>;

        for num_vars in 0..=4usize {
            let f = mle(&(0..(1u64 << num_vars))
                .map(|i| i.wrapping_mul(6364136223846793005) >> 13)
                .collect::<Vec<_>>());
            let point: Vec<ExtE> = (0..num_vars).map(|i| ExtE::from(101 + i as u64)).collect();

            let lifted =
                Mle::new(f.evals().iter().map(|v| v.to_extension::<Ext>()).collect()).unwrap();

            assert_eq!(
                f.evaluate_in(&point).unwrap(),
                lifted.evaluate(&point).unwrap(),
                "num_vars={num_vars}"
            );
        }
    }

    #[test]
    fn evaluating_in_the_same_field_is_evaluating() {
        let f = mle(&[3, 5, 8, 13]);
        let point = [FE::from(6), FE::from(7)];
        assert_eq!(f.evaluate_in(&point).unwrap(), f.evaluate(&point).unwrap());
    }

    #[test]
    fn first_and_last_folds_bind_different_ends() {
        let f = mle(&[1, 2, 3, 4]);
        let r = FE::from(5);
        let mut by_first = f.clone();
        by_first.fix_first_variable_in_place(&r).unwrap();
        let mut by_last = f;
        by_last.fix_last_variable_in_place(&r).unwrap();
        assert_ne!(by_first, by_last);
    }
}
