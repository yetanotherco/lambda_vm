//! A sum of products of multilinear polynomials — one implementation of
//! [`SumcheckPolynomial`](crate::poly::SumcheckPolynomial).

use math::field::{element::FieldElement, traits::IsField};

use crate::{Error, mle::Mle, poly::SumcheckPolynomial};

/// One monomial: `coefficient · Π_{i ∈ factors} poly[i]`.
///
/// A term with no factors is a constant. Repeating an index raises that
/// polynomial to a power, which is how a squared column is expressed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term<F: IsField> {
    pub coefficient: FieldElement<F>,
    pub factors: Vec<usize>,
}

impl<F: IsField> Term<F> {
    pub fn new(coefficient: FieldElement<F>, factors: Vec<usize>) -> Self {
        Self {
            coefficient,
            factors,
        }
    }

    /// A term that is just one polynomial, with coefficient one.
    pub fn single(index: usize) -> Self {
        Self::new(FieldElement::<F>::one(), vec![index])
    }
}

/// A sum of [`Term`]s over a shared set of multilinear polynomials.
#[derive(Clone, Debug)]
pub struct VirtualPolynomial<F: IsField> {
    polys: Vec<Mle<F>>,
    terms: Vec<Term<F>>,
    num_vars: usize,
}

impl<F: IsField> VirtualPolynomial<F> {
    /// Builds the polynomial, checking that every factor resolves and that all
    /// operands agree on the number of variables.
    pub fn new(polys: Vec<Mle<F>>, terms: Vec<Term<F>>) -> Result<Self, Error> {
        if terms.is_empty() {
            return Err(Error::EmptyPolynomial);
        }
        let num_vars = polys.first().map(|p| p.num_vars()).unwrap_or(0);
        for p in &polys {
            if p.num_vars() != num_vars {
                return Err(Error::VariableCountMismatch {
                    expected: num_vars,
                    got: p.num_vars(),
                });
            }
        }
        for term in &terms {
            for &index in &term.factors {
                if index >= polys.len() {
                    return Err(Error::UnknownPolynomial {
                        index,
                        len: polys.len(),
                    });
                }
            }
        }
        Ok(Self {
            polys,
            terms,
            num_vars,
        })
    }

    pub fn terms(&self) -> &[Term<F>] {
        &self.terms
    }

    /// The value of each factor once every variable has been fixed.
    pub fn factor_constants(&self) -> Option<Vec<FieldElement<F>>> {
        self.polys
            .iter()
            .map(|p| p.as_constant().cloned())
            .collect()
    }
}

impl<F: IsField> SumcheckPolynomial<F> for VirtualPolynomial<F> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// The largest number of factors in any term.
    fn degree(&self) -> usize {
        self.terms
            .iter()
            .map(|t| t.factors.len())
            .max()
            .unwrap_or(0)
    }

    fn polys(&self) -> &[Mle<F>] {
        &self.polys
    }

    fn combine(&self, values: &[FieldElement<F>]) -> FieldElement<F> {
        self.terms.iter().fold(FieldElement::zero(), |acc, term| {
            let product = term
                .factors
                .iter()
                .fold(term.coefficient.clone(), |p, &f| p * &values[f]);
            acc + product
        })
    }

    fn fix_first_variable(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        for p in &mut self.polys {
            p.fix_first_variable_in_place(r)?;
        }
        self.num_vars -= 1;
        Ok(())
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

    /// `f = 2·a·b + 3·c` on two variables.
    fn sample() -> VirtualPolynomial<F> {
        let a = mle(&[1, 2, 3, 4]);
        let b = mle(&[5, 6, 7, 8]);
        let c = mle(&[9, 10, 11, 12]);
        VirtualPolynomial::new(
            vec![a, b, c],
            vec![
                Term::new(FE::from(2), vec![0, 1]),
                Term::new(FE::from(3), vec![2]),
            ],
        )
        .unwrap()
    }

    #[test]
    fn degree_is_the_longest_term() {
        assert_eq!(sample().degree(), 2);
    }

    #[test]
    fn index_evaluation_applies_the_term_structure() {
        let f = sample();
        for i in 0..4usize {
            let (a, b, c) = (
                FE::from(1 + i as u64),
                FE::from(5 + i as u64),
                FE::from(9 + i as u64),
            );
            let expected = FE::from(2) * a * b + FE::from(3) * c;
            assert_eq!(f.eval_at_index(i), expected);
        }
    }

    #[test]
    fn hypercube_sum_matches_the_terms() {
        let f = sample();
        let expected = (0..4).fold(FE::zero(), |acc, i| acc + f.eval_at_index(i));
        assert_eq!(f.sum_over_hypercube(), expected);
    }

    #[test]
    fn evaluate_off_cube_matches_combine_of_factor_values() {
        let f = sample();
        let point = [FE::from(17), FE::from(23)];
        let values: Vec<FE> = f
            .polys()
            .iter()
            .map(|p| p.evaluate(&point).unwrap())
            .collect();
        assert_eq!(f.evaluate(&point).unwrap(), f.combine(&values));
    }

    #[test]
    fn folding_agrees_with_evaluating_the_fixed_variable() {
        let mut f = sample();
        let r = FE::from(31);
        let rest = FE::from(37);
        let expected = f.evaluate(&[r, rest]).unwrap();

        f.fix_first_variable(&r).unwrap();
        assert_eq!(f.num_vars(), 1);
        assert_eq!(f.evaluate(&[rest]).unwrap(), expected);
    }

    #[test]
    fn repeated_factor_is_a_power() {
        let a = mle(&[2, 3, 5, 7]);
        let f = VirtualPolynomial::new(vec![a], vec![Term::new(FE::one(), vec![0, 0])]).unwrap();
        assert_eq!(f.degree(), 2);
        assert_eq!(f.eval_at_index(2), FE::from(25));
    }

    #[test]
    fn constant_term_has_degree_zero() {
        let f = VirtualPolynomial::new(vec![mle(&[1, 1])], vec![Term::new(FE::from(9), vec![])])
            .unwrap();
        assert_eq!(f.degree(), 0);
        assert_eq!(f.eval_at_index(0), FE::from(9));
        assert_eq!(f.sum_over_hypercube(), FE::from(18));
    }

    #[test]
    fn rejects_a_factor_that_does_not_resolve() {
        let err = VirtualPolynomial::new(vec![mle(&[1, 2])], vec![Term::new(FE::one(), vec![3])])
            .unwrap_err();
        assert_eq!(err, Error::UnknownPolynomial { index: 3, len: 1 });
    }

    #[test]
    fn rejects_operands_of_differing_arity() {
        let err = VirtualPolynomial::new(
            vec![mle(&[1, 2]), mle(&[1, 2, 3, 4])],
            vec![Term::single(0)],
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::VariableCountMismatch {
                expected: 1,
                got: 2
            }
        );
    }

    #[test]
    fn rejects_an_empty_term_list() {
        let err = VirtualPolynomial::new(vec![mle(&[1, 2])], vec![]).unwrap_err();
        assert_eq!(err, Error::EmptyPolynomial);
    }
}
