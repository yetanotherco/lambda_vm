//! Bridges an AIR's constraint IR to the hypercube: one factor per distinct
//! `(main, offset, col)` trace read, `offset` as a cyclic rotation, and row
//! domains as [`Selector`]s.
//!
//! Not sound on its own: nothing yet forces a rotated factor to be the shift of
//! the column it claims to shift. Values are all lifted to the extension field.

use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};
use multilinear::{Error as MlError, mle::Mle, poly::SumcheckPolynomial, selector::Selector};
use std::collections::BTreeMap;

use crate::constraint_ir::ir::{ConstraintProgram, Op};
use crate::constraints::builder::ConstraintMeta;

/// Identifies a trace read: main-vs-aux, frame-step offset, column.
///
/// `row` is not part of the key — every table in this VM reads row 0 of each
/// frame step, which the IR interpreter asserts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LeafKey {
    pub main: bool,
    pub offset: u8,
    pub col: u16,
}

/// The multilinear factors an IR program reads, one per distinct trace leaf.
///
/// Construction is `O(leaves · 2^n)`: each leaf's table is a rotation of its
/// column.
#[derive(Clone, Debug)]
pub struct TraceLeaves<E: IsField> {
    /// Leaf -> index into `polys`, kept ordered so the layout is deterministic.
    index: BTreeMap<LeafKey, usize>,
    polys: Vec<Mle<E>>,
    num_vars: usize,
}

impl<E: IsField> TraceLeaves<E> {
    /// Materializes one MLE per leaf in `program`.
    ///
    /// `main_column` and `aux_column` return a column's values indexed by step;
    /// both must return exactly `2^num_vars` entries.
    pub fn build<F>(
        program: &ConstraintProgram<F, E>,
        num_vars: usize,
        mut main_column: impl FnMut(u16) -> Vec<FieldElement<E>>,
        mut aux_column: impl FnMut(u16) -> Vec<FieldElement<E>>,
    ) -> Result<Self, MlError>
    where
        F: IsField,
    {
        let size = 1usize << num_vars;
        let mut index = BTreeMap::new();
        let mut polys = Vec::new();

        for op in &program.nodes {
            let Op::Var {
                main, offset, col, ..
            } = *op
            else {
                continue;
            };
            let key = LeafKey { main, offset, col };
            if index.contains_key(&key) {
                continue;
            }

            let column = if main {
                main_column(col)
            } else {
                aux_column(col)
            };
            if column.len() != size {
                return Err(MlError::NotPowerOfTwo(column.len()));
            }
            // offset = k reads k steps ahead; on the cube that is a cyclic shift.
            let shift = offset as usize % size;
            let rotated = (0..size)
                .map(|s| column[(s + shift) % size].clone())
                .collect();

            index.insert(key, polys.len());
            polys.push(Mle::new(rotated)?);
        }

        Ok(Self {
            index,
            polys,
            num_vars,
        })
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn polys(&self) -> &[Mle<E>] {
        &self.polys
    }

    pub fn len(&self) -> usize {
        self.polys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.polys.is_empty()
    }

    pub fn index_of(&self, key: &LeafKey) -> Option<usize> {
        self.index.get(key).copied()
    }
}

/// Uniform values an IR program may read: LogUp challenges, alpha powers and
/// the table offset. Constant across the whole trace, so they are scalars here
/// rather than polynomials.
#[derive(Clone, Debug)]
pub struct Uniforms<E: IsField> {
    pub rap_challenges: Vec<FieldElement<E>>,
    pub logup_alpha_powers: Vec<FieldElement<E>>,
    pub logup_table_offset: FieldElement<E>,
}

/// Hand-written so the field itself need not be `Default`.
impl<E: IsField> Default for Uniforms<E> {
    fn default() -> Self {
        Self {
            rap_challenges: Vec::new(),
            logup_alpha_powers: Vec::new(),
            logup_table_offset: FieldElement::zero(),
        }
    }
}

/// Per-node degree of an IR program, in the trace variables.
///
/// Constants and uniforms are degree 0, a trace read is degree 1, products add
/// and sums take the max. Used to size the sumcheck round polynomials.
fn node_degrees<F: IsField, E: IsField>(program: &ConstraintProgram<F, E>) -> Vec<usize> {
    let mut deg: Vec<usize> = Vec::with_capacity(program.nodes.len());
    for op in &program.nodes {
        let d = match *op {
            Op::ConstBase(_)
            | Op::ConstExt(_)
            | Op::RapChallenge { .. }
            | Op::AlphaPow { .. }
            | Op::TableOffset => 0,
            Op::Var { .. } => 1,
            Op::Add(a, b) | Op::Sub(a, b) => deg[a as usize].max(deg[b as usize]),
            Op::Mul(a, b) => deg[a as usize] + deg[b as usize],
            Op::Neg(a) | Op::Embed(a) => deg[a as usize],
        };
        deg.push(d);
    }
    deg
}

/// An AIR's constraints as one polynomial: `C = Σ_i beta^i · s_i(x) · C_i(x)`.
///
/// Factors are the trace leaves followed by one table per distinct non-trivial
/// selector, in that order.
pub struct IrPolynomial<'a, F: IsField, E: IsField> {
    shape: IrShape<'a, F, E>,
    polys: Vec<Mle<E>>,
}

/// The constraint's *structure*, with no trace data in it.
///
/// [`combine`](Self::combine) turns factor values into the batched constraint,
/// and needs nothing but this — so the verifier can hold one and rebuild
/// `C(point)` from values it learned through the commitment scheme, without
/// ever seeing a column.
#[derive(Clone)]
pub struct IrShape<'a, F: IsField, E: IsField> {
    program: &'a ConstraintProgram<F, E>,
    leaf_index: BTreeMap<LeafKey, usize>,
    uniforms: Uniforms<E>,
    /// Roots to batch, in order.
    roots: Vec<u32>,
    /// Index into the factor list of each root's selector, or `None` when it
    /// applies on every step and the multiplication can be skipped.
    selector_of_root: Vec<Option<usize>>,
    /// `beta^i` for each selected root.
    beta_powers: Vec<FieldElement<E>>,
    degree: usize,
    num_vars: usize,
}

impl<'a, F, E> IrPolynomial<'a, F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    /// Batches every constraint in `program`, taking each one's row domain from
    /// `meta`.
    pub fn new(
        program: &'a ConstraintProgram<F, E>,
        leaves: TraceLeaves<E>,
        uniforms: Uniforms<E>,
        beta: FieldElement<E>,
        meta: &[ConstraintMeta],
    ) -> Result<Self, MlError> {
        let roots: Vec<u32> = program.roots.clone();
        let selectors: Vec<Selector> = meta
            .iter()
            .map(|m| Selector::except_last(m.end_exemptions))
            .collect();
        Self::with_roots(program, leaves, uniforms, beta, roots, &selectors)
    }

    /// Batches every constraint with no row exemptions at all.
    ///
    /// Only correct for AIRs whose constraints really do hold on every step,
    /// wrap included.
    pub fn new_unselected(
        program: &'a ConstraintProgram<F, E>,
        leaves: TraceLeaves<E>,
        uniforms: Uniforms<E>,
        beta: FieldElement<E>,
    ) -> Result<Self, MlError> {
        let roots: Vec<u32> = program.roots.clone();
        let selectors = vec![Selector::ALL; roots.len()];
        Self::with_roots(program, leaves, uniforms, beta, roots, &selectors)
    }

    /// Batches the listed roots with the matching selectors.
    pub fn with_roots(
        program: &'a ConstraintProgram<F, E>,
        leaves: TraceLeaves<E>,
        uniforms: Uniforms<E>,
        beta: FieldElement<E>,
        roots: Vec<u32>,
        selectors: &[Selector],
    ) -> Result<Self, MlError> {
        if selectors.len() != roots.len() {
            return Err(MlError::VariableCountMismatch {
                expected: roots.len(),
                got: selectors.len(),
            });
        }
        let num_vars = leaves.num_vars;
        let degrees = node_degrees(program);

        let TraceLeaves {
            index: leaf_index,
            mut polys,
            ..
        } = leaves;

        // One table per distinct non-trivial selector, shared across roots.
        let mut selector_slot: BTreeMap<usize, usize> = BTreeMap::new();
        let mut selector_of_root = Vec::with_capacity(roots.len());
        for selector in selectors {
            if selector.is_trivial() {
                selector_of_root.push(None);
                continue;
            }
            let slot = match selector_slot.get(&selector.end_exemptions) {
                Some(&i) => i,
                None => {
                    let i = polys.len();
                    polys.push(selector.table(num_vars)?);
                    selector_slot.insert(selector.end_exemptions, i);
                    i
                }
            };
            selector_of_root.push(Some(slot));
        }

        // A selector is multilinear, so it costs exactly one degree.
        let degree = roots
            .iter()
            .zip(selectors)
            .map(|(&r, sel)| degrees[r as usize] + usize::from(!sel.is_trivial()))
            .max()
            .unwrap_or(0);

        let mut beta_powers = Vec::with_capacity(roots.len());
        let mut acc = FieldElement::<E>::one();
        for _ in 0..roots.len() {
            beta_powers.push(acc.clone());
            acc *= &beta;
        }

        Ok(Self {
            shape: IrShape {
                program,
                leaf_index,
                uniforms,
                roots,
                selector_of_root,
                beta_powers,
                degree,
                num_vars,
            },
            polys,
        })
    }

    /// The structure alone, for the verifier.
    pub fn shape(&self) -> &IrShape<'a, F, E> {
        &self.shape
    }
}

impl<F, E> IrShape<'_, F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// The batched constraint, given each factor's value at a point.
    pub fn combine(&self, values: &[FieldElement<E>]) -> FieldElement<E> {
        let nodes = self.run(values);
        self.roots
            .iter()
            .zip(&self.beta_powers)
            .zip(&self.selector_of_root)
            .fold(FieldElement::zero(), |acc, ((&root, beta_pow), sel)| {
                let mut term = &nodes[root as usize] * beta_pow;
                if let Some(slot) = sel {
                    term *= &values[*slot];
                }
                acc + term
            })
    }

    /// Runs the DAG with each trace leaf taking the supplied value.
    fn run(&self, values: &[FieldElement<E>]) -> Vec<FieldElement<E>> {
        let mut nodes: Vec<FieldElement<E>> = Vec::with_capacity(self.program.nodes.len());
        for op in &self.program.nodes {
            let v = match *op {
                Op::ConstBase(idx) => {
                    let base = self.program.base_consts[idx as usize].clone();
                    base.to_extension::<E>()
                }
                Op::ConstExt(idx) => self.program.ext_consts[idx as usize].clone(),
                Op::Var {
                    main, offset, col, ..
                } => {
                    let key = LeafKey { main, offset, col };
                    let i = *self
                        .leaf_index
                        .get(&key)
                        .expect("every Var leaf was materialized at build time");
                    values[i].clone()
                }
                Op::RapChallenge { idx } => self.uniforms.rap_challenges[idx as usize].clone(),
                Op::AlphaPow { idx } => self.uniforms.logup_alpha_powers[idx as usize].clone(),
                Op::TableOffset => self.uniforms.logup_table_offset.clone(),
                Op::Add(a, b) => &nodes[a as usize] + &nodes[b as usize],
                Op::Sub(a, b) => &nodes[a as usize] - &nodes[b as usize],
                Op::Mul(a, b) => &nodes[a as usize] * &nodes[b as usize],
                Op::Neg(a) => -&nodes[a as usize],
                Op::Embed(a) => nodes[a as usize].clone(),
            };
            nodes.push(v);
        }
        nodes
    }
}

impl<F, E> SumcheckPolynomial<E> for IrPolynomial<'_, F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    fn num_vars(&self) -> usize {
        self.shape.num_vars
    }

    fn degree(&self) -> usize {
        self.shape.degree
    }

    fn polys(&self) -> &[Mle<E>] {
        &self.polys
    }

    fn combine(&self, values: &[FieldElement<E>]) -> FieldElement<E> {
        self.shape.combine(values)
    }

    fn fix_first_variable(&mut self, r: &FieldElement<E>) -> Result<(), MlError> {
        for p in &mut self.polys {
            p.fix_first_variable_in_place(r)?;
        }
        self.shape.num_vars -= 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
    use math::field::goldilocks::GoldilocksField as Fp;
    use multilinear::zerocheck;

    use crate::constraints::builder::{
        CaptureBuilder, ConstraintBuilder, ConstraintSet, RowDomain,
    };

    type ExtE = FieldElement<Ext>;

    const COL_A: usize = 0;
    const COL_B: usize = 1;
    const COL_C: usize = 2;

    /// Two constraints in the shape real tables use:
    ///
    /// - idx 0, degree 1, reads the **next** step: `next(a) − a − b`
    /// - idx 1, degree 2, current step only: `a·a − c`
    ///
    /// Unlike the Fibonacci examples these hold cyclically, so no wrap-around
    /// exemption is needed — selectors are not modelled yet (see module docs).
    struct SampleSet;

    impl ConstraintSet<Fp, Ext> for SampleSet {
        fn max_degree(&self) -> usize {
            2
        }

        fn eval<B: ConstraintBuilder<Fp, Ext>>(&self, b: &mut B) {
            let a = b.main(0, COL_A);
            let b_col = b.main(0, COL_B);
            let a_next = b.main(1, COL_A);
            b.emit_base(0, a_next - a.clone() - b_col);

            let c = b.main(0, COL_C);
            b.emit_base(1, a.clone() * a - c);
        }
    }

    fn program() -> ConstraintProgram<Fp, Ext> {
        let mut cb = CaptureBuilder::<Fp, Ext>::new();
        SampleSet.eval(&mut cb);
        cb.finish(2).0
    }

    /// Columns satisfying both constraints on every step, wrap included.
    fn satisfying_columns(num_vars: usize) -> [Vec<ExtE>; 3] {
        let size = 1usize << num_vars;
        let a: Vec<ExtE> = (0..size as u64)
            .map(|i| ExtE::from(i.wrapping_mul(7).wrapping_add(3)))
            .collect();
        // b is the cyclic forward difference, so `next(a) − a − b` vanishes
        // including across the wrap.
        let b: Vec<ExtE> = (0..size).map(|i| a[(i + 1) % size] - a[i]).collect();
        let c: Vec<ExtE> = a.iter().map(|x| x * x).collect();
        [a, b, c]
    }

    fn leaves_from(columns: &[Vec<ExtE>; 3], num_vars: usize) -> TraceLeaves<Ext> {
        let prog = program();
        TraceLeaves::build(
            &prog,
            num_vars,
            |col| columns[col as usize].clone(),
            |_| unreachable!("this set has no aux reads"),
        )
        .unwrap()
    }

    fn transcript() -> DefaultTranscript<Ext> {
        DefaultTranscript::<Ext>::new(b"multilinear-air-test")
    }

    #[test]
    fn one_leaf_per_distinct_trace_read() {
        let num_vars = 4;
        let leaves = leaves_from(&satisfying_columns(num_vars), num_vars);
        // a@0, b@0, a@1, c@0 — the next-step read of `a` is its own factor.
        assert_eq!(leaves.len(), 4);
        assert!(
            leaves
                .index_of(&LeafKey {
                    main: true,
                    offset: 1,
                    col: COL_A as u16
                })
                .is_some()
        );
    }

    #[test]
    fn an_offset_leaf_is_the_rotation_of_its_column() {
        let num_vars = 3;
        let columns = satisfying_columns(num_vars);
        let leaves = leaves_from(&columns, num_vars);
        let size = 1usize << num_vars;

        let cur = leaves
            .index_of(&LeafKey {
                main: true,
                offset: 0,
                col: COL_A as u16,
            })
            .unwrap();
        let next = leaves
            .index_of(&LeafKey {
                main: true,
                offset: 1,
                col: COL_A as u16,
            })
            .unwrap();

        for s in 0..size {
            assert_eq!(leaves.polys()[cur].evals()[s], columns[COL_A][s]);
            assert_eq!(
                leaves.polys()[next].evals()[s],
                columns[COL_A][(s + 1) % size],
                "step {s}"
            );
        }
    }

    #[test]
    fn degree_comes_from_the_dag() {
        let num_vars = 3;
        let prog = program();
        let leaves = leaves_from(&satisfying_columns(num_vars), num_vars);
        let poly = IrPolynomial::new_unselected(&prog, leaves, Uniforms::default(), ExtE::from(5))
            .unwrap();
        // `a·a − c` is the degree-2 constraint; batching does not raise it.
        assert_eq!(poly.degree(), 2);
    }

    #[test]
    fn a_satisfying_trace_zerochecks() {
        let num_vars = 5;
        let prog = program();
        let leaves = leaves_from(&satisfying_columns(num_vars), num_vars);
        let poly = IrPolynomial::new_unselected(&prog, leaves, Uniforms::default(), ExtE::from(5))
            .unwrap();

        // The batched constraint really is zero on every step.
        assert_eq!(poly.sum_over_hypercube(), ExtE::zero());
        let degree = poly.degree();

        let proof = zerocheck::prove(poly, &mut transcript()).unwrap().proof;
        let claim = zerocheck::verify(&proof, num_vars, degree, &mut transcript()).unwrap();

        // Rebuild the polynomial to check the residual claim against it.
        let leaves = leaves_from(&satisfying_columns(num_vars), num_vars);
        let poly = IrPolynomial::new_unselected(&prog, leaves, Uniforms::default(), ExtE::from(5))
            .unwrap();
        assert_eq!(
            claim.constraint_evaluation().unwrap(),
            poly.evaluate(&claim.point).unwrap()
        );
    }

    #[test]
    fn a_single_broken_step_is_rejected() {
        let num_vars = 5;
        let prog = program();
        let mut columns = satisfying_columns(num_vars);
        columns[COL_C][9] += ExtE::one();

        let leaves = leaves_from(&columns, num_vars);
        let poly = IrPolynomial::new_unselected(&prog, leaves, Uniforms::default(), ExtE::from(5))
            .unwrap();
        assert_ne!(poly.sum_over_hypercube(), ExtE::zero());
        let degree = poly.degree();

        let proof = zerocheck::prove(poly, &mut transcript()).unwrap().proof;
        assert!(zerocheck::verify(&proof, num_vars, degree, &mut transcript()).is_err());
    }

    #[test]
    fn violations_that_cancel_in_the_sum_are_still_caught() {
        // Perturbing a[3] breaks `next(a) − a − b` twice: at step 2 the
        // next-step read is one too high, at step 3 the current-step read is.
        // The two violations are equal and opposite, so the plain sum over the
        // cube stays zero — this is exactly the case eq(r, ·) exists to catch,
        // and it is reachable from an ordinary AIR, not just a contrived one.
        let num_vars = 4;
        let prog = program();
        let mut columns = satisfying_columns(num_vars);
        columns[COL_A][3] += ExtE::one();
        columns[COL_C][3] = columns[COL_A][3] * columns[COL_A][3]; // keep idx 1 satisfied

        let build = |cols: &[Vec<ExtE>; 3]| {
            IrPolynomial::new_unselected(
                &prog,
                leaves_from(cols, num_vars),
                Uniforms::default(),
                ExtE::from(5),
            )
            .unwrap()
        };

        let poly = build(&columns);
        assert_eq!(
            poly.sum_over_hypercube(),
            ExtE::zero(),
            "the violations were expected to cancel"
        );
        // But the constraint is genuinely nonzero somewhere.
        let nonzero_steps = (0..(1usize << num_vars))
            .filter(|&i| poly.eval_at_index(i) != ExtE::zero())
            .count();
        assert_eq!(nonzero_steps, 2);

        let degree = poly.degree();
        let proof = zerocheck::prove(poly, &mut transcript()).unwrap().proof;

        match zerocheck::verify(&proof, num_vars, degree, &mut transcript()) {
            Err(_) => {}
            Ok(claim) => {
                // If the rounds happened to line up, the residual claim must
                // still disagree with the real polynomial.
                assert_ne!(
                    claim.constraint_evaluation(),
                    Some(build(&columns).evaluate(&claim.point).unwrap()),
                    "a violated constraint produced a consistent claim"
                );
            }
        }
    }

    #[test]
    fn batching_covers_every_constraint() {
        // With only the degree-2 root selected, a violation of the degree-1
        // one must go unnoticed — which is what makes the batched version's
        // rejection meaningful.
        let num_vars = 4;
        let prog = program();
        let mut columns = satisfying_columns(num_vars);
        columns[COL_B][2] += ExtE::one(); // breaks idx 0 only

        let only_second = IrPolynomial::with_roots(
            &prog,
            leaves_from(&columns, num_vars),
            Uniforms::default(),
            ExtE::from(5),
            vec![prog.roots[1]],
            &[Selector::ALL],
        )
        .unwrap();
        assert_eq!(only_second.sum_over_hypercube(), ExtE::zero());

        let batched = IrPolynomial::new_unselected(
            &prog,
            leaves_from(&columns, num_vars),
            Uniforms::default(),
            ExtE::from(5),
        )
        .unwrap();
        assert_ne!(batched.sum_over_hypercube(), ExtE::zero());
    }

    // ---------------------------------------------------------------
    // Selectors: an AIR whose transition constraint cannot hold on the
    // wrap-around step, which is the shape every real table has.
    // ---------------------------------------------------------------

    /// The 2-column Fibonacci recurrence, verbatim from
    /// `examples::fibonacci_2_columns`: both constraints read the next step
    /// and therefore carry one end exemption.
    struct FibSet;

    impl ConstraintSet<Fp, Ext> for FibSet {
        fn max_degree(&self) -> usize {
            1
        }

        fn eval<B: ConstraintBuilder<Fp, Ext>>(&self, b: &mut B) {
            let s0_0 = b.main(0, 0);
            let s0_1 = b.main(0, 1);
            let s1_0 = b.main(1, 0);
            let s1_1 = b.main(1, 1);

            b.emit_base_rows(
                0,
                RowDomain::except_last(1),
                s1_0.clone() - s0_0 - s0_1.clone(),
            );
            b.emit_base_rows(1, RowDomain::except_last(1), s1_1 - s0_1 - s1_0);
        }
    }

    fn fib_program() -> (ConstraintProgram<Fp, Ext>, Vec<ConstraintMeta>) {
        let mut cb = CaptureBuilder::<Fp, Ext>::new();
        FibSet.eval(&mut cb);
        let prog = cb.finish(2).0;
        let meta = vec![
            ConstraintMeta::base(0).with_end_exemptions(1),
            ConstraintMeta::base(1).with_end_exemptions(1),
        ];
        (prog, meta)
    }

    /// A genuine Fibonacci trace: the recurrence holds on every step except
    /// the wrap, exactly where the exemption applies.
    fn fib_columns(num_vars: usize) -> [Vec<ExtE>; 2] {
        let size = 1usize << num_vars;
        let mut c0 = vec![ExtE::one()];
        let mut c1 = vec![ExtE::one()];
        for i in 1..size {
            // s0_{i} = s0_{i-1} + s1_{i-1}; s1_{i} = s1_{i-1} + s0_{i}
            let next0 = c0[i - 1] + c1[i - 1];
            let next1 = c1[i - 1] + next0;
            c0.push(next0);
            c1.push(next1);
        }
        [c0, c1]
    }

    fn fib_leaves(columns: &[Vec<ExtE>; 2], num_vars: usize) -> TraceLeaves<Ext> {
        let (prog, _) = fib_program();
        TraceLeaves::build(
            &prog,
            num_vars,
            |col| columns[col as usize].clone(),
            |_| unreachable!("no aux reads"),
        )
        .unwrap()
    }

    #[test]
    fn without_a_selector_the_wrap_step_breaks_fibonacci() {
        // Establishes that the exemption is load-bearing: unselected, the
        // constraint is violated precisely at the wrap.
        let num_vars = 4;
        let (prog, _) = fib_program();
        let columns = fib_columns(num_vars);
        let poly = IrPolynomial::new_unselected(
            &prog,
            fib_leaves(&columns, num_vars),
            Uniforms::default(),
            ExtE::from(5),
        )
        .unwrap();

        let last = (1usize << num_vars) - 1;
        for i in 0..last {
            assert_eq!(poly.eval_at_index(i), ExtE::zero(), "step {i}");
        }
        assert_ne!(poly.eval_at_index(last), ExtE::zero(), "wrap step");
    }

    #[test]
    fn with_the_selector_a_real_fibonacci_air_zerochecks() {
        let num_vars = 5;
        let (prog, meta) = fib_program();
        let columns = fib_columns(num_vars);

        let build = || {
            IrPolynomial::new(
                &prog,
                fib_leaves(&columns, num_vars),
                Uniforms::default(),
                ExtE::from(5),
                &meta,
            )
            .unwrap()
        };

        let poly = build();
        // The selector masks the wrap, so the batched constraint vanishes
        // on the whole cube.
        for i in 0..(1usize << num_vars) {
            assert_eq!(poly.eval_at_index(i), ExtE::zero(), "step {i}");
        }
        let degree = poly.degree();

        let proof = zerocheck::prove(poly, &mut transcript()).unwrap().proof;
        let claim = zerocheck::verify(&proof, num_vars, degree, &mut transcript()).unwrap();
        assert_eq!(
            claim.constraint_evaluation().unwrap(),
            build().evaluate(&claim.point).unwrap()
        );
    }

    #[test]
    fn the_selector_costs_exactly_one_degree() {
        let num_vars = 4;
        let (prog, meta) = fib_program();
        let columns = fib_columns(num_vars);

        let unselected = IrPolynomial::new_unselected(
            &prog,
            fib_leaves(&columns, num_vars),
            Uniforms::default(),
            ExtE::from(5),
        )
        .unwrap();
        let selected = IrPolynomial::new(
            &prog,
            fib_leaves(&columns, num_vars),
            Uniforms::default(),
            ExtE::from(5),
            &meta,
        )
        .unwrap();

        assert_eq!(unselected.degree(), 1);
        assert_eq!(selected.degree(), 2);
    }

    #[test]
    fn a_selector_does_not_hide_a_violation_inside_the_active_range() {
        let num_vars = 5;
        let (prog, meta) = fib_program();
        let mut columns = fib_columns(num_vars);
        columns[0][7] += ExtE::one();

        let poly = IrPolynomial::new(
            &prog,
            fib_leaves(&columns, num_vars),
            Uniforms::default(),
            ExtE::from(5),
            &meta,
        )
        .unwrap();
        let degree = poly.degree();

        let proof = zerocheck::prove(poly, &mut transcript()).unwrap().proof;
        assert!(zerocheck::verify(&proof, num_vars, degree, &mut transcript()).is_err());
    }

    #[test]
    fn roots_sharing_an_exemption_share_one_selector_table() {
        // Both Fibonacci constraints exempt one step, so exactly one selector
        // table is materialized on top of the four trace leaves.
        let num_vars = 3;
        let (prog, meta) = fib_program();
        let columns = fib_columns(num_vars);
        let leaves = fib_leaves(&columns, num_vars);
        let num_leaves = leaves.len();

        let poly =
            IrPolynomial::new(&prog, leaves, Uniforms::default(), ExtE::from(5), &meta).unwrap();
        assert_eq!(poly.polys().len(), num_leaves + 1);
    }

    // ---------------------------------------------------------------
    // The whole argument over a real captured AIR.
    // ---------------------------------------------------------------

    /// The same Fibonacci set, captured with the extension set to the base
    /// field.
    ///
    /// WHIR's evaluation domain is a two-adic subgroup, so it lives in the base
    /// field; a codeword whose *values* are extension elements needs `encode`
    /// and `fold_codeword` generalized over a field tower. That generalization
    /// is the outstanding base/extension work, so the end-to-end test below
    /// runs where both coincide. Nothing about the argument changes — only how
    /// wide the arithmetic is.
    fn fib_program_base() -> (ConstraintProgram<Fp, Fp>, Vec<ConstraintMeta>) {
        struct FibBase;
        impl ConstraintSet<Fp, Fp> for FibBase {
            fn max_degree(&self) -> usize {
                1
            }
            fn eval<B: ConstraintBuilder<Fp, Fp>>(&self, b: &mut B) {
                let s0_0 = b.main(0, 0);
                let s0_1 = b.main(0, 1);
                let s1_0 = b.main(1, 0);
                let s1_1 = b.main(1, 1);
                b.emit_base_rows(
                    0,
                    RowDomain::except_last(1),
                    s1_0.clone() - s0_0 - s0_1.clone(),
                );
                b.emit_base_rows(1, RowDomain::except_last(1), s1_1 - s0_1 - s1_0);
            }
        }

        let mut cb = CaptureBuilder::<Fp, Fp>::new();
        FibBase.eval(&mut cb);
        let meta = vec![
            ConstraintMeta::base(0).with_end_exemptions(1),
            ConstraintMeta::base(1).with_end_exemptions(1),
        ];
        (cb.finish(2).0, meta)
    }

    type FpE = FieldElement<Fp>;

    fn fib_columns_base(num_vars: usize) -> [Vec<FpE>; 2] {
        let size = 1usize << num_vars;
        let mut c0 = vec![FpE::one()];
        let mut c1 = vec![FpE::one()];
        for i in 1..size {
            let next0 = c0[i - 1] + c1[i - 1];
            let next1 = c1[i - 1] + next0;
            c0.push(next0);
            c1.push(next1);
        }
        [c0, c1]
    }

    /// Runs the full argument over the captured AIR and returns the verdict.
    ///
    /// **Caveat, and it is not small.** The `next`-step read is committed as its
    /// own polynomial, so nothing here forces it to be the rotation of the
    /// column it claims to shift — a prover free to choose both could satisfy
    /// this with unrelated tables. Closing that needs the rotation kernel
    /// (`multilinear::eq::rot_eval`) wired as its own argument.
    fn argue_fib(columns: &[Vec<FpE>; 2], num_vars: usize) -> Result<(), multilinear::Error> {
        use multilinear::{
            constraint_argument::{self, CommittedTrace, TraceClaim},
            whir_eval::EvalConfig,
        };

        let (prog, meta) = fib_program_base();
        let leaves = TraceLeaves::build(
            &prog,
            num_vars,
            |col| columns[col as usize].clone(),
            |_| unreachable!("no aux reads"),
        )?;
        let poly = IrPolynomial::new(&prog, leaves, Uniforms::default(), FpE::from(5), &meta)?;
        let degree = poly.degree();
        let factors = poly.polys().to_vec();
        let shape = poly.shape().clone();

        let config = EvalConfig {
            log_blowup: 2,
            num_queries: 3,
        };
        let trace = CommittedTrace::commit(factors, &config)?;
        let roots = trace.roots();

        let mut prover_transcript = DefaultTranscript::<Fp>::new(b"air-argument");
        let proof = constraint_argument::prove(
            &trace,
            |v: &[FpE]| shape.combine(v),
            degree,
            &config,
            &mut prover_transcript,
        )?;

        let mut verifier_transcript = DefaultTranscript::<Fp>::new(b"air-argument");
        constraint_argument::verify(
            &proof,
            TraceClaim {
                roots: &roots,
                domain: trace.domain(),
                num_vars,
            },
            |v: &[FpE]| shape.combine(v),
            degree,
            &config,
            &mut verifier_transcript,
        )
    }

    /// The composition this whole crate exists for: an AIR captured by the same
    /// builder the prover uses, zerochecked over the hypercube, every factor
    /// settled against a WHIR commitment.
    #[test]
    fn a_real_air_argues_end_to_end_against_commitments() {
        let num_vars = 4;
        let columns = fib_columns_base(num_vars);
        argue_fib(&columns, num_vars).unwrap();
    }

    #[test]
    fn a_real_air_with_a_broken_row_is_rejected_end_to_end() {
        let num_vars = 4;
        let mut columns = fib_columns_base(num_vars);
        columns[1][6] += FpE::one();
        assert!(argue_fib(&columns, num_vars).is_err());
    }
}
