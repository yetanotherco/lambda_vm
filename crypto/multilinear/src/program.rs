//! A batch's rule as straight-line code over the factor values.
//!
//! A [`Rule`](crate::batch::Rule) is a closure, which is what makes a statement
//! easy to state and impossible to hand to a device. Every rule this crate
//! proves is nonetheless *data* — a compiled constraint program, an affine
//! expression over factor slots, a fixed formula — so a rule can carry the
//! program it is, and the sumcheck can run that instead: one description, run
//! by the host loop and by the kernel that replaces it.
//!
//! Values are whatever the sumcheck's field is. Two evaluators of the same
//! program may sum in different orders, which the field does not distinguish:
//! Goldilocks compares and serializes canonically, so equal values are equal
//! everywhere the protocol looks.

use math::field::{element::FieldElement, traits::IsField};

use crate::Error;

/// One step. Operands are indices of earlier steps; `Var` indexes the factor
/// values the sumcheck supplies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op<E: IsField> {
    Fixed(FieldElement<E>),
    Var(u32),
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Neg(u32),
}

/// Straight-line code and the step holding its value.
#[derive(Clone, Debug)]
pub struct Program<E: IsField> {
    steps: Vec<Op<E>>,
    root: u32,
}

impl<E: IsField> Program<E> {
    pub fn steps(&self) -> &[Op<E>] {
        &self.steps
    }

    pub fn root(&self) -> u32 {
        self.root
    }

    /// The program's value, given every factor's value at one point.
    ///
    /// `scratch` is the caller's, because this runs once per cube index per
    /// interpolation node: a buffer of its own per call would be one heap
    /// allocation per evaluation.
    pub fn eval(
        &self,
        values: &[FieldElement<E>],
        scratch: &mut Vec<FieldElement<E>>,
    ) -> FieldElement<E> {
        scratch.clear();
        scratch.reserve(self.steps.len());
        for step in &self.steps {
            let v = match *step {
                Op::Fixed(ref c) => c.clone(),
                Op::Var(i) => values[i as usize].clone(),
                Op::Add(a, b) => &scratch[a as usize] + &scratch[b as usize],
                Op::Sub(a, b) => &scratch[a as usize] - &scratch[b as usize],
                Op::Mul(a, b) => &scratch[a as usize] * &scratch[b as usize],
                Op::Neg(a) => -&scratch[a as usize],
            };
            scratch.push(v);
        }
        scratch[self.root as usize].clone()
    }

    /// The highest factor slot the program reads, or `None` when it reads none.
    pub fn max_slot(&self) -> Option<u32> {
        self.steps
            .iter()
            .filter_map(|step| match step {
                Op::Var(i) => Some(*i),
                _ => None,
            })
            .max()
    }
}

/// Emits steps, handing back the index of each one.
#[derive(Debug, Default)]
pub struct Builder<E: IsField> {
    steps: Vec<Op<E>>,
}

impl<E: IsField> Builder<E> {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    fn push(&mut self, op: Op<E>) -> u32 {
        self.steps.push(op);
        (self.steps.len() - 1) as u32
    }

    pub fn fixed(&mut self, value: FieldElement<E>) -> u32 {
        self.push(Op::Fixed(value))
    }

    pub fn var(&mut self, slot: usize) -> u32 {
        self.push(Op::Var(slot as u32))
    }

    pub fn add(&mut self, a: u32, b: u32) -> u32 {
        self.push(Op::Add(a, b))
    }

    pub fn sub(&mut self, a: u32, b: u32) -> u32 {
        self.push(Op::Sub(a, b))
    }

    pub fn mul(&mut self, a: u32, b: u32) -> u32 {
        self.push(Op::Mul(a, b))
    }

    pub fn neg(&mut self, a: u32) -> u32 {
        self.push(Op::Neg(a))
    }

    /// `Σ terms`, or a zero step when there are none.
    pub fn sum(&mut self, terms: &[u32]) -> u32 {
        match terms.split_first() {
            None => self.fixed(FieldElement::zero()),
            Some((&first, rest)) => rest.iter().fold(first, |acc, &t| self.add(acc, t)),
        }
    }

    /// `Σ coefficient_i · step_i`, folding a coefficient of one away.
    pub fn weighted_sum(&mut self, terms: &[(u32, FieldElement<E>)]) -> u32 {
        let scaled: Vec<u32> = terms
            .iter()
            .map(|(step, coefficient)| {
                if *coefficient == FieldElement::one() {
                    *step
                } else {
                    let c = self.fixed(coefficient.clone());
                    self.mul(c, *step)
                }
            })
            .collect();
        self.sum(&scaled)
    }

    /// Copies `program`'s steps in, renumbering its operands, and returns where
    /// its root landed.
    pub fn splice(&mut self, program: &Program<E>) -> u32 {
        let base = self.steps.len() as u32;
        for step in &program.steps {
            let shifted = match *step {
                Op::Fixed(ref c) => Op::Fixed(c.clone()),
                Op::Var(i) => Op::Var(i),
                Op::Add(a, b) => Op::Add(a + base, b + base),
                Op::Sub(a, b) => Op::Sub(a + base, b + base),
                Op::Mul(a, b) => Op::Mul(a + base, b + base),
                Op::Neg(a) => Op::Neg(a + base),
            };
            self.steps.push(shifted);
        }
        base + program.root
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The program computing `root`.
    pub fn finish(self, root: u32) -> Result<Program<E>, Error> {
        if root as usize >= self.steps.len() {
            return Err(Error::UnknownPolynomial {
                index: root as usize,
                len: self.steps.len(),
            });
        }
        Ok(Program {
            steps: self.steps,
            root,
        })
    }
}

/// `Σ_i lambda_i · program_i`, the batch as one program.
pub fn combine<E: IsField>(
    programs: &[&Program<E>],
    lambdas: &[FieldElement<E>],
) -> Result<Program<E>, Error> {
    if programs.len() != lambdas.len() {
        return Err(Error::VariableCountMismatch {
            expected: programs.len(),
            got: lambdas.len(),
        });
    }
    let mut builder = Builder::<E>::new();
    let terms: Vec<(u32, FieldElement<E>)> = programs
        .iter()
        .zip(lambdas)
        .map(|(program, lambda)| (builder.splice(program), lambda.clone()))
        .collect();
    let root = builder.weighted_sum(&terms);
    builder.finish(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    fn values(n: usize) -> Vec<FE> {
        (0..n as u64).map(|i| FE::from(i * 7 + 3)).collect()
    }

    #[test]
    fn a_program_evaluates_its_root() {
        // (f0 + f1) · f2 − 5
        let mut b = Builder::<F>::new();
        let f0 = b.var(0);
        let f1 = b.var(1);
        let f2 = b.var(2);
        let sum = b.add(f0, f1);
        let product = b.mul(sum, f2);
        let five = b.fixed(FE::from(5u64));
        let root = b.sub(product, five);
        let program = b.finish(root).unwrap();

        let v = values(3);
        let mut scratch = Vec::new();
        assert_eq!(
            program.eval(&v, &mut scratch),
            (v[0] + v[1]) * v[2] - FE::from(5u64)
        );
        assert_eq!(program.max_slot(), Some(2));
    }

    #[test]
    fn a_spliced_program_keeps_its_meaning() {
        let mut b = Builder::<F>::new();
        let f0 = b.var(0);
        let f1 = b.var(1);
        let root = b.mul(f0, f1);
        let inner = b.finish(root).unwrap();

        let mut outer = Builder::<F>::new();
        // A step before the splice, so the renumbering is not a no-op.
        let f2 = outer.var(2);
        let spliced = outer.splice(&inner);
        let root = outer.add(spliced, f2);
        let program = outer.finish(root).unwrap();

        let v = values(3);
        let mut scratch = Vec::new();
        assert_eq!(program.eval(&v, &mut scratch), v[0] * v[1] + v[2]);
    }

    #[test]
    fn a_combined_program_is_the_weighted_sum() {
        let mut b = Builder::<F>::new();
        let root = b.var(0);
        let first = b.finish(root).unwrap();
        let mut b = Builder::<F>::new();
        let root = b.var(1);
        let second = b.finish(root).unwrap();

        let lambdas = vec![FE::one(), FE::from(9u64)];
        let program = combine(&[&first, &second], &lambdas).unwrap();
        let v = values(2);
        let mut scratch = Vec::new();
        assert_eq!(program.eval(&v, &mut scratch), v[0] + FE::from(9u64) * v[1]);
    }

    #[test]
    fn a_weighted_sum_of_nothing_is_zero() {
        let mut b = Builder::<F>::new();
        let root = b.weighted_sum(&[]);
        let program = b.finish(root).unwrap();
        let mut scratch = Vec::new();
        assert_eq!(program.eval(&[], &mut scratch), FE::zero());
    }

    #[test]
    fn a_root_past_the_end_is_rejected() {
        let mut b = Builder::<F>::new();
        b.var(0);
        assert!(b.finish(7).is_err());
    }
}
