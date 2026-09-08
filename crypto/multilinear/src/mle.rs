//! Multilinear extensions, held as their `2^n` hypercube evaluations.
//!
//! Index `i` is read with **variable 0 as the most significant bit**. Every
//! fold in this crate assumes that.

use math::field::{element::FieldElement, traits::IsField};

use crate::Error;

/// A multilinear polynomial held by its hypercube evaluations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mle<F: IsField> {
    evals: Vec<FieldElement<F>>,
    num_vars: usize,
}

impl<F: IsField> Mle<F> {
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
        for j in 0..half {
            let delta = &self.evals[j + half] - &self.evals[j];
            self.evals[j] = &self.evals[j] + r * &delta;
        }
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
        let mut current = self.clone();
        for r in point {
            current.fix_first_variable_in_place(r)?;
        }
        Ok(current.evals[0].clone())
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
