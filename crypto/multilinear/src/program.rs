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

    /// The same program, emitted on demand: each step where the step that
    /// uses it first needs it, the operand that holds more values evaluated
    /// first, and every read — a factor's value or a constant — emitted again
    /// at each use instead of once and held.
    ///
    /// A value lives from its step to its last use. [`simplify`](Self::simplify)
    /// makes every read of a column or a constant one step, and a batch is
    /// written piece by piece: every root, then their weighted sum; every
    /// interaction's side, then theirs. So a batch holds each shared read from
    /// its first use to its last, and each root or side until the sum at the
    /// end. On the widest precompile's batch that is 2,763 values a thread
    /// (`thoughts/zf/gap2/fix2/I-GFS.md` §15.5). On demand, a sum's terms are
    /// added in as they are made, which is the running sum a lean builder
    /// would write, and a read lives for one step.
    ///
    /// The order is the Sethi–Ullman one: of two operands, the one needing more
    /// values goes first, so the other is held while less is live. A step is
    /// the same operation on the same operands as before, so every value is
    /// the same, to the limb. A reload costs a load, plus the extension to the
    /// interpolation node for a factor. Nothing is simplified again, which
    /// would merge the reads back.
    pub fn on_demand(&self) -> Self {
        let n = self.steps.len();
        let reads = |op: &Op<E>| matches!(op, Op::Var(_) | Op::Fixed(_));
        // Values each step needs to be evaluated as a tree (a read, one).
        let mut need = vec![0u32; n];
        for (k, op) in self.steps.iter().enumerate() {
            need[k] = match *op {
                Op::Var(_) | Op::Fixed(_) => 1,
                Op::Neg(a) => need[a as usize],
                Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => {
                    let (x, y) = (need[a as usize], need[b as usize]);
                    if x == y {
                        x.saturating_add(1)
                    } else {
                        x.max(y)
                    }
                }
            };
        }
        let mut steps: Vec<Op<E>> = Vec::with_capacity(n * 3 / 2);
        // Where each old step landed, once it has; a read has no one place.
        let mut at: Vec<Option<u32>> = vec![None; n];
        let read = |steps: &mut Vec<Op<E>>, at: &[Option<u32>], k: u32| -> u32 {
            match &self.steps[k as usize] {
                Op::Var(i) => {
                    steps.push(Op::Var(*i));
                    (steps.len() - 1) as u32
                }
                Op::Fixed(c) => {
                    steps.push(Op::Fixed(c.clone()));
                    (steps.len() - 1) as u32
                }
                _ => at[k as usize].expect("an operand is emitted before its user"),
            }
        };
        if reads(&self.steps[self.root as usize]) {
            let root = read(&mut steps, &at, self.root);
            return Self { steps, root };
        }
        // A step, and whether its operands are already under way.
        let mut stack: Vec<(u32, bool)> = vec![(self.root, false)];
        while let Some((k, expanded)) = stack.pop() {
            if at[k as usize].is_some() {
                continue;
            }
            let op = &self.steps[k as usize];
            let operands = match *op {
                Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => [Some(a), Some(b)],
                Op::Neg(a) => [Some(a), None],
                Op::Var(_) | Op::Fixed(_) => unreachable!("a read is emitted at its use"),
            };
            if !expanded {
                stack.push((k, true));
                let mut pending: Vec<u32> = operands
                    .into_iter()
                    .flatten()
                    .filter(|&o| !reads(&self.steps[o as usize]) && at[o as usize].is_none())
                    .collect();
                // The one needing more goes first, so it goes on top; a tie
                // keeps the written order.
                if let [a, b] = pending[..]
                    && need[b as usize] > need[a as usize]
                {
                    pending.swap(0, 1);
                }
                stack.extend(pending.into_iter().rev().map(|o| (o, false)));
                continue;
            }
            let emitted = match *op {
                Op::Add(a, b) => {
                    let (a, b) = (read(&mut steps, &at, a), read(&mut steps, &at, b));
                    Op::Add(a, b)
                }
                Op::Sub(a, b) => {
                    let (a, b) = (read(&mut steps, &at, a), read(&mut steps, &at, b));
                    Op::Sub(a, b)
                }
                Op::Mul(a, b) => {
                    let (a, b) = (read(&mut steps, &at, a), read(&mut steps, &at, b));
                    Op::Mul(a, b)
                }
                Op::Neg(a) => Op::Neg(read(&mut steps, &at, a)),
                Op::Var(_) | Op::Fixed(_) => unreachable!("a read is emitted at its use"),
            };
            steps.push(emitted);
            at[k as usize] = Some((steps.len() - 1) as u32);
        }
        let root = at[self.root as usize].expect("the root is emitted");
        Self { steps, root }
    }

    /// The same program with each step emitted once and nothing dead in it.
    ///
    /// Straight-line code over a field: two steps with the same operator over
    /// the same operands compute the same value, adding zero or multiplying by
    /// one computes nothing, and a step the root does not reach is not
    /// computed at all. A batch's program is spliced together out of pieces
    /// that share structure — the same alpha powers, the same column read by
    /// several interactions — so what this removes is not a mistake in any one
    /// of them, it is the seam between them.
    ///
    /// The kernel walks this program once per cube index per interpolation
    /// node, through a slot file in global memory, so a step removed here is
    /// removed from every one of those walks.
    pub fn simplify(&self) -> Self
    where
        FieldElement<E>: PartialEq,
    {
        use std::collections::HashMap;

        let zero = FieldElement::<E>::zero();
        let one = FieldElement::<E>::one();
        let mut steps: Vec<Op<E>> = Vec::with_capacity(self.steps.len());
        // Where each old step ended up.
        let mut moved: Vec<u32> = Vec::with_capacity(self.steps.len());
        // Steps that are not constants, keyed by what they compute.
        let mut seen: HashMap<(u8, u32, u32), u32> = HashMap::new();
        // Constants, which have to be compared by value.
        let mut constants: Vec<(u32, FieldElement<E>)> = Vec::new();

        let constant_of = |steps: &[Op<E>], at: u32| match &steps[at as usize] {
            Op::Fixed(c) => Some(c.clone()),
            _ => None,
        };

        for step in &self.steps {
            let emit = |steps: &mut Vec<Op<E>>,
                        seen: &mut HashMap<(u8, u32, u32), u32>,
                        constants: &mut Vec<(u32, FieldElement<E>)>,
                        op: Op<E>|
             -> u32 {
                if let Op::Fixed(value) = &op {
                    if let Some((at, _)) = constants.iter().find(|(_, c)| c == value) {
                        return *at;
                    }
                    steps.push(op.clone());
                    let at = (steps.len() - 1) as u32;
                    constants.push((at, value.clone()));
                    return at;
                }
                // Addition and multiplication do not care which operand came
                // first, so the key does not either.
                let key = match op {
                    Op::Var(i) => (0, i, 0),
                    Op::Add(a, b) => (1, a.min(b), a.max(b)),
                    Op::Sub(a, b) => (2, a, b),
                    Op::Mul(a, b) => (3, a.min(b), a.max(b)),
                    Op::Neg(a) => (4, a, 0),
                    Op::Fixed(_) => unreachable!("handled above"),
                };
                if let Some(at) = seen.get(&key) {
                    return *at;
                }
                steps.push(op);
                let at = (steps.len() - 1) as u32;
                seen.insert(key, at);
                at
            };

            let at = match *step {
                Op::Fixed(ref c) => {
                    emit(&mut steps, &mut seen, &mut constants, Op::Fixed(c.clone()))
                }
                Op::Var(i) => emit(&mut steps, &mut seen, &mut constants, Op::Var(i)),
                Op::Neg(a) => {
                    let a = moved[a as usize];
                    match constant_of(&steps, a) {
                        Some(c) => emit(&mut steps, &mut seen, &mut constants, Op::Fixed(-c)),
                        None => emit(&mut steps, &mut seen, &mut constants, Op::Neg(a)),
                    }
                }
                Op::Add(a, b) => {
                    let (a, b) = (moved[a as usize], moved[b as usize]);
                    match (constant_of(&steps, a), constant_of(&steps, b)) {
                        (Some(x), Some(y)) => {
                            emit(&mut steps, &mut seen, &mut constants, Op::Fixed(x + y))
                        }
                        (Some(x), None) if x == zero => b,
                        (None, Some(y)) if y == zero => a,
                        _ => emit(&mut steps, &mut seen, &mut constants, Op::Add(a, b)),
                    }
                }
                Op::Sub(a, b) => {
                    let (a, b) = (moved[a as usize], moved[b as usize]);
                    match (constant_of(&steps, a), constant_of(&steps, b)) {
                        (Some(x), Some(y)) => {
                            emit(&mut steps, &mut seen, &mut constants, Op::Fixed(x - y))
                        }
                        (None, Some(y)) if y == zero => a,
                        _ => emit(&mut steps, &mut seen, &mut constants, Op::Sub(a, b)),
                    }
                }
                Op::Mul(a, b) => {
                    let (a, b) = (moved[a as usize], moved[b as usize]);
                    match (constant_of(&steps, a), constant_of(&steps, b)) {
                        (Some(x), Some(y)) => {
                            emit(&mut steps, &mut seen, &mut constants, Op::Fixed(x * y))
                        }
                        (Some(x), _) if x == zero => emit(
                            &mut steps,
                            &mut seen,
                            &mut constants,
                            Op::Fixed(zero.clone()),
                        ),
                        (_, Some(y)) if y == zero => emit(
                            &mut steps,
                            &mut seen,
                            &mut constants,
                            Op::Fixed(zero.clone()),
                        ),
                        (Some(x), None) if x == one => b,
                        (None, Some(y)) if y == one => a,
                        _ => emit(&mut steps, &mut seen, &mut constants, Op::Mul(a, b)),
                    }
                }
            };
            moved.push(at);
        }

        let root = moved[self.root as usize];
        Self { steps, root }.prune()
    }

    /// Drops the steps the root does not reach, renumbering the rest.
    fn prune(self) -> Self {
        let mut live = vec![false; self.steps.len()];
        live[self.root as usize] = true;
        for i in (0..self.steps.len()).rev() {
            if !live[i] {
                continue;
            }
            match self.steps[i] {
                Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => {
                    live[a as usize] = true;
                    live[b as usize] = true;
                }
                Op::Neg(a) => live[a as usize] = true,
                Op::Fixed(_) | Op::Var(_) => {}
            }
        }
        let mut moved = vec![0u32; self.steps.len()];
        let mut steps = Vec::with_capacity(self.steps.len());
        for (i, step) in self.steps.into_iter().enumerate() {
            if !live[i] {
                continue;
            }
            let shifted = match step {
                Op::Fixed(c) => Op::Fixed(c),
                Op::Var(v) => Op::Var(v),
                Op::Add(a, b) => Op::Add(moved[a as usize], moved[b as usize]),
                Op::Sub(a, b) => Op::Sub(moved[a as usize], moved[b as usize]),
                Op::Mul(a, b) => Op::Mul(moved[a as usize], moved[b as usize]),
                Op::Neg(a) => Op::Neg(moved[a as usize]),
            };
            steps.push(shifted);
            moved[i] = (steps.len() - 1) as u32;
        }
        let root = moved[self.root as usize];
        Self { steps, root }
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

    /// The program computing `root`, with each step emitted once.
    ///
    /// A builder is written for whoever emits — a rule at a time, a piece at a
    /// time — and the pieces overlap: the same alpha power, the same column,
    /// the same difference. [`simplify`](Program::simplify) is run here because
    /// this is where a program stops being written and starts being walked,
    /// and it is walked once per cube index per interpolation node.
    pub fn finish(self, root: u32) -> Result<Program<E>, Error>
    where
        FieldElement<E>: PartialEq,
    {
        if root as usize >= self.steps.len() {
            return Err(Error::UnknownPolynomial {
                index: root as usize,
                len: self.steps.len(),
            });
        }
        Ok(Program {
            steps: self.steps,
            root,
        }
        .simplify())
    }

    /// The program as it was emitted, step for step.
    pub fn finish_verbatim(self, root: u32) -> Result<Program<E>, Error> {
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

#[cfg(test)]
mod simplify_tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField as F;

    type FE = FieldElement<F>;

    /// Every simplification has to be invisible to the evaluator: same value at
    /// every point, fewer steps to get there.
    fn agrees(program: &Program<F>, slots: usize) {
        let small = program.simplify();
        assert!(
            small.steps().len() <= program.steps().len(),
            "simplifying grew the program"
        );
        let mut scratch = Vec::new();
        for seed in 0..6u64 {
            let values: Vec<FE> = (0..slots)
                .map(|i| FE::from((seed + 1) * (i as u64 + 3) + 7))
                .collect();
            let before = program.eval(&values, &mut scratch);
            let after = small.eval(&values, &mut scratch);
            assert_eq!(before, after, "seed {seed}");
        }
    }

    /// A program on demand is the same program: every read becomes one step
    /// per use, the root included when it is a read, a shared step is still
    /// emitted once, and the value at every point is the same.
    #[test]
    fn a_program_on_demand_is_the_same_program() {
        let mut b = Builder::<F>::new();
        let x = b.var(0);
        let y = b.var(1);
        let seven = b.fixed(FE::from(7u64));
        let xy = b.mul(x, y);
        let a = b.add(xy, seven);
        let d = b.sub(a, x);
        let n = b.neg(d);
        let shared = b.mul(n, seven);
        let root = b.mul(shared, shared);
        let program = b.finish(root).unwrap();
        let demand = program.on_demand();
        let count = |p: &Program<F>, read: bool| {
            p.steps()
                .iter()
                .filter(|op| matches!(op, Op::Var(_) | Op::Fixed(_)) == read)
                .count()
        };
        // x twice, y once, seven twice: one read per use. The six other steps
        // are six still: `shared` is used twice and made once.
        assert_eq!((count(&program, true), count(&demand, true)), (3, 5));
        assert_eq!((count(&program, false), count(&demand, false)), (6, 6));
        let mut scratch = Vec::new();
        for seed in 0..6u64 {
            let values: Vec<FE> = (0..2).map(|i| FE::from((seed + 1) * (i + 3) + 7)).collect();
            assert_eq!(
                program.eval(&values, &mut scratch),
                demand.eval(&values, &mut scratch),
                "seed {seed}"
            );
        }
        // A program that is a single read is that read.
        let mut b = Builder::<F>::new();
        let only = b.var(3);
        let read = b.finish(only).unwrap().on_demand();
        assert_eq!(read.steps(), &[Op::Var(3)]);
        assert_eq!(read.root(), 0);
    }

    /// ★ What the schedule is for: a weighted sum written after its terms
    /// holds every term until the sum; on demand it holds the running sum and
    /// the term being made. Eight products of two reads each, summed with
    /// weights, as `weighted_sum` writes them after the terms.
    #[test]
    fn a_sum_on_demand_holds_one_term_at_a_time() {
        let mut b = Builder::<F>::new();
        let terms: Vec<(u32, FE)> = (0..8u32)
            .map(|i| {
                let (p, q) = (b.var(2 * i as usize), b.var(2 * i as usize + 1));
                (b.mul(p, q), FE::from(u64::from(i) + 2))
            })
            .collect();
        let root = b.weighted_sum(&terms);
        let program = b.finish(root).unwrap();
        let demand = program.on_demand();
        let live = |p: &Program<F>| {
            // A value is live from its step to its last use.
            let mut last = vec![0usize; p.steps().len()];
            for (k, op) in p.steps().iter().enumerate() {
                match *op {
                    Op::Add(a, c) | Op::Sub(a, c) | Op::Mul(a, c) => {
                        last[a as usize] = k;
                        last[c as usize] = k;
                    }
                    Op::Neg(a) => last[a as usize] = k,
                    Op::Var(_) | Op::Fixed(_) => {}
                }
            }
            last[p.root() as usize] = p.steps().len();
            (0..p.steps().len())
                .map(|k| (0..=k).filter(|&j| last[j] >= k).count())
                .max()
                .unwrap()
        };
        assert!(
            live(&program) >= 8,
            "written after its terms, the sum holds its eight terms: {}",
            live(&program)
        );
        assert!(
            live(&demand) <= 4,
            "on demand it holds the running sum, a term and its two reads: {}",
            live(&demand)
        );
        let values: Vec<FE> = (0..16u64).map(|i| FE::from(i * 5 + 1)).collect();
        let mut scratch = Vec::new();
        assert_eq!(
            program.eval(&values, &mut scratch),
            demand.eval(&values, &mut scratch)
        );
    }

    #[test]
    fn a_repeated_step_is_emitted_once() {
        let mut b = Builder::<F>::new();
        let x = b.var(0);
        let y = b.var(1);
        let first = b.mul(x, y);
        // The same product again, and the same one with its operands swapped.
        let second = b.mul(x, y);
        let third = b.mul(y, x);
        let sum = b.add(first, second);
        let root = b.add(sum, third);
        let program = b.finish_verbatim(root).unwrap();

        agrees(&program, 2);
        // one var, one var, one product, two sums
        assert_eq!(program.simplify().steps().len(), 5);
    }

    #[test]
    fn constants_fold_and_units_disappear() {
        let mut b = Builder::<F>::new();
        let x = b.var(0);
        let one = b.fixed(FE::one());
        let zero = b.fixed(FE::zero());
        let scaled = b.mul(x, one);
        let shifted = b.add(scaled, zero);
        let two = b.fixed(FE::from(2));
        let three = b.fixed(FE::from(3));
        let six = b.mul(two, three);
        let root = b.add(shifted, six);
        let program = b.finish_verbatim(root).unwrap();

        agrees(&program, 1);
        let small = program.simplify();
        // `x`, the constant six, and their sum.
        assert_eq!(small.steps().len(), 3);
    }

    #[test]
    fn what_the_root_does_not_reach_is_dropped() {
        let mut b = Builder::<F>::new();
        let x = b.var(0);
        let y = b.var(1);
        let _dead = b.mul(x, y);
        let root = b.add(x, x);
        let program = b.finish_verbatim(root).unwrap();

        agrees(&program, 2);
        assert_eq!(program.simplify().steps().len(), 2);
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
