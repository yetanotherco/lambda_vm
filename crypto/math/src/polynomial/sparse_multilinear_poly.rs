#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::field::{element::FieldElement, traits::IsField};
use crate::polynomial::error::MultilinearError;
use alloc::vec::Vec;

pub struct SparseMultilinearPolynomial<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    num_vars: usize,
    evals: Vec<(usize, FieldElement<F>)>,
}

impl<F: IsField> SparseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Creates a new sparse multilinear polynomial
    pub fn new(num_vars: usize, evals: Vec<(usize, FieldElement<F>)>) -> Self {
        SparseMultilinearPolynomial { num_vars, evals }
    }

    /// Returns the number of variables of the sparse multilinear polynomial
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Computes the eq extension polynomial of the polynomial.
    /// return 1 when a == r, otherwise return 0.
    /// Chi is the Lagrange basis polynomial evaluated at r
    /// a indicates the binary decomposition of the index of the Lagrange basis polynomial
    fn compute_chi(a: &[bool], r: &[FieldElement<F>]) -> Result<FieldElement<F>, MultilinearError> {
        if a.len() != r.len() {
            return Err(MultilinearError::ChisAndEvalsLengthMismatch(
                a.len(),
                r.len(),
            ));
        }
        let mut chi_i = FieldElement::one();
        for (aj, rj) in a.iter().zip(r.iter()) {
            if *aj {
                chi_i *= rj;
            } else {
                chi_i *= FieldElement::<F>::one() - rj;
            }
        }
        Ok(chi_i)
    }

    // Takes O(n log n)
    /// Evaluates the multilinear polynomial at a point r
    pub fn evaluate(&self, r: &[FieldElement<F>]) -> Result<FieldElement<F>, MultilinearError> {
        if r.len() != self.num_vars() {
            return Err(MultilinearError::IncorrectNumberofEvaluationPoints(
                r.len(),
                self.num_vars(),
            ));
        }

        #[cfg(feature = "parallel")]
        let iter = (0..self.evals.len()).into_par_iter();

        #[cfg(not(feature = "parallel"))]
        let iter = 0..self.evals.len();

        Ok(iter
            .map(|i| {
                let bits = get_bits(self.evals[i].0, r.len());
                let mut chi_i = FieldElement::<F>::one();
                for j in 0..r.len() {
                    if bits[j] {
                        chi_i *= &r[j];
                    } else {
                        chi_i *= FieldElement::<F>::one() - &r[j];
                    }
                }
                chi_i * &self.evals[i].1
            })
            .sum())
    }

    // Takes O(n log n)
    /// Evaluates the multilinear polynomial at a point r
    pub fn evaluate_with(
        num_vars: usize,
        evals: &[(usize, FieldElement<F>)],
        r: &[FieldElement<F>],
    ) -> Result<FieldElement<F>, MultilinearError> {
        if r.len() != num_vars {
            return Err(MultilinearError::IncorrectNumberofEvaluationPoints(
                r.len(),
                num_vars,
            ));
        }

        #[cfg(feature = "parallel")]
        let iter = (0..evals.len()).into_par_iter();

        #[cfg(not(feature = "parallel"))]
        let iter = 0..evals.len();
        Ok(iter
            .map(|i| {
                let bits = get_bits(evals[i].0, r.len());
                SparseMultilinearPolynomial::compute_chi(&bits, r).unwrap() * &evals[i].1
            })
            .sum())
    }
}

/// Returns the bit decomposition (Vec<bool>) of the `index` of an evaluation within the sparse multilinear polynomial.
fn get_bits(n: usize, num_bits: usize) -> Vec<bool> {
    (0..num_bits)
        .map(|shift_amount| (n & (1 << (num_bits - shift_amount - 1))) > 0)
        .collect::<Vec<bool>>()
}
