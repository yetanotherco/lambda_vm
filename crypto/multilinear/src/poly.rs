//! What sumcheck needs from a polynomial: its multilinear factors plus a rule
//! for combining their values. Keeps a constraint DAG out of expanded form,
//! which would be exponential in the nesting depth.

use math::field::{element::FieldElement, traits::IsField};

use crate::{Error, mle::Mle, program::Program};

/// A polynomial over the hypercube, presented as multilinear factors plus a
/// rule for combining their values.
pub trait SumcheckPolynomial<F: IsField + 'static> {
    /// Variables left to bind.
    fn num_vars(&self) -> usize;

    /// Total degree, which bounds each round polynomial's degree.
    ///
    /// Must be an upper bound: the verifier checks the round polynomial has
    /// exactly `degree + 1` evaluations, so understating it rejects honest
    /// proofs and overstating it only costs proof size.
    fn degree(&self) -> usize;

    /// The multilinear factors, in the order [`combine`](Self::combine) indexes.
    fn polys(&self) -> &[Mle<F>];

    /// The polynomial's value, given each factor's value at the same point.
    fn combine(&self, values: &[FieldElement<F>]) -> FieldElement<F>;

    /// The same, with a scratch buffer the caller owns. The sumcheck calls this
    /// once per cube index per interpolation node, so an implementation backed
    /// by a program has nowhere to put its steps that is not the caller's.
    fn combine_in(
        &self,
        values: &[FieldElement<F>],
        scratch: &mut Vec<FieldElement<F>>,
    ) -> FieldElement<F> {
        let _ = scratch;
        self.combine(values)
    }

    /// Binds variable 0 to `r` in every factor.
    fn fix_first_variable(&mut self, r: &FieldElement<F>) -> Result<(), Error>;

    /// The program this polynomial is, when it can say: the description a
    /// device runs in place of [`combine`](Self::combine).
    ///
    /// An implementation that offers one must also accept its factors back
    /// through [`accept_folded`](Self::accept_folded) — the device binds them
    /// there, and the sumcheck's contract is that the polynomial comes back
    /// folded.
    fn program(&self) -> Option<&Program<F>> {
        None
    }

    /// Takes factors bound elsewhere, in the order [`polys`](Self::polys)
    /// returns them.
    fn accept_folded(&mut self, polys: Vec<Mle<F>>) -> Result<(), Error> {
        let _ = polys;
        Err(Error::DeviceFailed {
            stage: "write-back",
        })
    }

    /// Value at hypercube index `i`.
    fn eval_at_index(&self, i: usize) -> FieldElement<F> {
        let values: Vec<FieldElement<F>> =
            self.polys().iter().map(|p| p.evals()[i].clone()).collect();
        self.combine(&values)
    }

    /// Sum over the whole hypercube. Reference implementation — the point of
    /// sumcheck is to avoid paying this.
    fn sum_over_hypercube(&self) -> FieldElement<F> {
        (0..(1usize << self.num_vars()))
            .fold(FieldElement::zero(), |acc, i| acc + self.eval_at_index(i))
    }

    /// Value at an arbitrary point, extending every factor multilinearly.
    fn evaluate(&self, point: &[FieldElement<F>]) -> Result<FieldElement<F>, Error> {
        let values: Vec<FieldElement<F>> = self
            .polys()
            .iter()
            .map(|p| p.evaluate(point))
            .collect::<Result<_, _>>()?;
        Ok(self.combine(&values))
    }
}

/// Factors plus a closure that combines them. The closure holds no trace data,
/// so the verifier can carry the same one.
pub struct Composed<F: IsField, C> {
    polys: Vec<Mle<F>>,
    combine: C,
    degree: usize,
    num_vars: usize,
    /// The same rule as straight-line code, when the caller can say what it is.
    program: Option<Program<F>>,
}

impl<F: IsField + 'static, C> Composed<F, C>
where
    C: Fn(&[FieldElement<F>]) -> FieldElement<F> + 'static,
{
    /// `degree` must upper-bound the closure's total degree in the factors.
    pub fn new(polys: Vec<Mle<F>>, combine: C, degree: usize) -> Result<Self, Error> {
        let num_vars = polys.first().map(|p| p.num_vars()).unwrap_or(0);
        for p in &polys {
            if p.num_vars() != num_vars {
                return Err(Error::VariableCountMismatch {
                    expected: num_vars,
                    got: p.num_vars(),
                });
            }
        }
        Ok(Self {
            polys,
            combine,
            degree,
            num_vars,
            program: None,
        })
    }

    /// The same, saying which program the closure is. The two must agree: the
    /// device runs the program and the host may run either.
    pub fn with_program(mut self, program: Program<F>) -> Self {
        self.program = Some(program);
        self
    }
}

impl<F: IsField, C> Composed<F, C> {
    /// The factors, dropping the rule.
    ///
    /// A chained WHIR needs them back after each group of rounds: it rebuilds
    /// its weight from the folded one, so it cannot keep the polynomial.
    pub fn into_polys(self) -> Vec<Mle<F>> {
        self.polys
    }
}

impl<F: IsField + 'static, C> SumcheckPolynomial<F> for Composed<F, C>
where
    C: Fn(&[FieldElement<F>]) -> FieldElement<F>,
{
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn polys(&self) -> &[Mle<F>] {
        &self.polys
    }

    fn combine(&self, values: &[FieldElement<F>]) -> FieldElement<F> {
        (self.combine)(values)
    }

    fn combine_in(
        &self,
        values: &[FieldElement<F>],
        scratch: &mut Vec<FieldElement<F>>,
    ) -> FieldElement<F> {
        match &self.program {
            Some(program) => program.eval(values, scratch),
            None => (self.combine)(values),
        }
    }

    fn fix_first_variable(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        for p in &mut self.polys {
            p.fix_first_variable_in_place(r)?;
        }
        self.num_vars -= 1;
        Ok(())
    }

    fn program(&self) -> Option<&Program<F>> {
        self.program.as_ref()
    }

    fn accept_folded(&mut self, polys: Vec<Mle<F>>) -> Result<(), Error> {
        if polys.len() != self.polys.len() {
            return Err(Error::VariableCountMismatch {
                expected: self.polys.len(),
                got: polys.len(),
            });
        }
        self.num_vars = polys.first().map(Mle::num_vars).unwrap_or(0);
        self.polys = polys;
        Ok(())
    }
}

/// Multiplies another polynomial by `eq(r, ·)`, appended as one more factor.
///
/// This is what turns a sumcheck into a zerocheck, and it works for any
/// underlying polynomial rather than only the sum-of-products one.
#[derive(Debug)]
pub struct EqScaled<F: IsField + 'static, P: SumcheckPolynomial<F>> {
    inner: P,
    /// `inner`'s factors followed by the `eq` table — the layout `combine` and
    /// the sumcheck prover both index.
    polys: Vec<Mle<F>>,
}

impl<F: IsField, P: SumcheckPolynomial<F>> EqScaled<F, P> {
    /// Wraps `inner` with the `eq(r, ·)` table.
    pub fn new(inner: P, eq: Mle<F>) -> Result<Self, Error> {
        if eq.num_vars() != inner.num_vars() {
            return Err(Error::VariableCountMismatch {
                expected: inner.num_vars(),
                got: eq.num_vars(),
            });
        }
        let mut polys = inner.polys().to_vec();
        polys.push(eq);
        Ok(Self { inner, polys })
    }

    pub fn into_inner(self) -> P {
        self.inner
    }
}

impl<F: IsField, P: SumcheckPolynomial<F>> SumcheckPolynomial<F> for EqScaled<F, P> {
    fn num_vars(&self) -> usize {
        self.inner.num_vars()
    }

    fn degree(&self) -> usize {
        self.inner.degree() + 1
    }

    fn polys(&self) -> &[Mle<F>] {
        &self.polys
    }

    fn combine(&self, values: &[FieldElement<F>]) -> FieldElement<F> {
        let (inner_values, eq_value) = values.split_at(values.len() - 1);
        self.inner.combine(inner_values) * &eq_value[0]
    }

    fn fix_first_variable(&mut self, r: &FieldElement<F>) -> Result<(), Error> {
        // `inner` keeps its own copies of the factors, so both views must fold.
        self.inner.fix_first_variable(r)?;
        for p in &mut self.polys {
            p.fix_first_variable_in_place(r)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField as F;

    use crate::{
        eq::eq_mle,
        virtual_poly::{Term, VirtualPolynomial},
    };

    type FE = FieldElement<F>;

    fn mle(vals: &[u64]) -> Mle<F> {
        Mle::new(vals.iter().map(|v| FE::from(*v)).collect()).unwrap()
    }

    fn inner() -> VirtualPolynomial<F> {
        VirtualPolynomial::new(
            vec![mle(&[1, 2, 3, 4]), mle(&[5, 6, 7, 8])],
            vec![Term::new(FE::from(2), vec![0, 1])],
        )
        .unwrap()
    }

    #[test]
    fn eq_scaling_adds_one_to_the_degree() {
        let r = vec![FE::from(3), FE::from(5)];
        let scaled = EqScaled::new(inner(), eq_mle(&r).unwrap()).unwrap();
        assert_eq!(inner().degree(), 2);
        assert_eq!(scaled.degree(), 3);
    }

    #[test]
    fn eq_scaling_multiplies_pointwise_on_the_cube() {
        let r = vec![FE::from(3), FE::from(5)];
        let eq = eq_mle(&r).unwrap();
        let base = inner();
        let scaled = EqScaled::new(inner(), eq.clone()).unwrap();

        for i in 0..4 {
            assert_eq!(
                scaled.eval_at_index(i),
                base.eval_at_index(i) * eq.evals()[i]
            );
        }
    }

    #[test]
    fn eq_scaled_sum_of_a_multilinear_polynomial_is_its_extension_at_r() {
        // Σ_x eq(r,x)·f(x) = f̃(r) — the defining property of eq, and it holds
        // only when f is itself multilinear.
        let linear = VirtualPolynomial::new(
            vec![mle(&[1, 2, 3, 4])],
            vec![Term::new(FE::from(2), vec![0])],
        )
        .unwrap();
        let r = vec![FE::from(11), FE::from(13)];
        let scaled = EqScaled::new(linear.clone(), eq_mle(&r).unwrap()).unwrap();

        assert_eq!(scaled.sum_over_hypercube(), linear.evaluate(&r).unwrap());
    }

    #[test]
    fn above_degree_one_the_eq_weighted_sum_is_not_the_pointwise_product() {
        // For a product of two columns, `evaluate` gives ã(r)·b̃(r) while the
        // eq-weighted sum gives the multilinear extension of the *function*
        // a·b. Those are different polynomials, and conflating them is an easy
        // way to write a wrong soundness argument.
        let r = vec![FE::from(11), FE::from(13)];
        let scaled = EqScaled::new(inner(), eq_mle(&r).unwrap()).unwrap();

        assert_ne!(scaled.sum_over_hypercube(), inner().evaluate(&r).unwrap());
    }

    #[test]
    fn folding_keeps_both_views_in_step() {
        let r = vec![FE::from(2), FE::from(4)];
        let mut scaled = EqScaled::new(inner(), eq_mle(&r).unwrap()).unwrap();
        let challenge = FE::from(9);
        let rest = FE::from(21);

        let expected = scaled.evaluate(&[challenge, rest]).unwrap();
        scaled.fix_first_variable(&challenge).unwrap();

        assert_eq!(scaled.num_vars(), 1);
        assert_eq!(scaled.evaluate(&[rest]).unwrap(), expected);
    }

    #[test]
    fn rejects_an_eq_table_of_the_wrong_arity() {
        let r = vec![FE::from(3)];
        let err = EqScaled::new(inner(), eq_mle(&r).unwrap()).unwrap_err();
        assert_eq!(
            err,
            Error::VariableCountMismatch {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn a_composed_polynomial_applies_its_closure() {
        // 2ab + 3c, written as a closure instead of terms.
        let polys = vec![
            mle(&[1, 2, 3, 4]),
            mle(&[5, 6, 7, 8]),
            mle(&[9, 10, 11, 12]),
        ];
        let composed = Composed::new(
            polys,
            |v: &[FE]| FE::from(2) * v[0] * v[1] + FE::from(3) * v[2],
            2,
        )
        .unwrap();

        assert_eq!(composed.degree(), 2);
        for i in 0..4 {
            let (a, b, c) = (
                FE::from(1 + i as u64),
                FE::from(5 + i as u64),
                FE::from(9 + i as u64),
            );
            assert_eq!(
                composed.eval_at_index(i),
                FE::from(2) * a * b + FE::from(3) * c
            );
        }
    }

    #[test]
    fn a_composed_polynomial_matches_the_equivalent_terms() {
        let polys = vec![mle(&[1, 2, 3, 4]), mle(&[5, 6, 7, 8])];
        let by_terms =
            VirtualPolynomial::new(polys.clone(), vec![Term::new(FE::from(2), vec![0, 1])])
                .unwrap();
        let by_closure = Composed::new(polys, |v: &[FE]| FE::from(2) * v[0] * v[1], 2).unwrap();

        let point = [FE::from(19), FE::from(23)];
        assert_eq!(
            by_terms.evaluate(&point).unwrap(),
            by_closure.evaluate(&point).unwrap()
        );
        assert_eq!(
            by_terms.sum_over_hypercube(),
            by_closure.sum_over_hypercube()
        );
    }

    #[test]
    fn composed_rejects_factors_of_differing_arity() {
        let result = Composed::new(vec![mle(&[1, 2]), mle(&[1, 2, 3, 4])], |v: &[FE]| v[0], 1);
        assert!(matches!(
            result.err(),
            Some(Error::VariableCountMismatch {
                expected: 1,
                got: 2
            })
        ));
    }
}
