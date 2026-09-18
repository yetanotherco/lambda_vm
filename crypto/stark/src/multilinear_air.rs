//! Bridges an AIR's constraint IR to the hypercube: one committed column per
//! distinct `(main, col)` trace read, `offset` as a cyclic shift *view* of that
//! column, and row domains as [`Selector`]s.
//!
//! Nothing here has to be taken on trust. A shifted factor gets no commitment
//! of its own â [`claim_reduce`] binds its value to the column it shifts â and
//! a selector gets none either: it is a *public* factor, which the verifier
//! evaluates through [`Selector::evaluate`] rather than reading out of a
//! commitment. Only the trace columns are committed, and they stay in the base
//! field they are; the sumcheck's factors are lifted for it.

use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};
use multilinear::{
    Error as MlError,
    claim_reduce::{self, FactorSource},
    constraint_argument::FactorKind,
    mle::Mle,
    poly::SumcheckPolynomial,
    program::{Builder, Op as ProgramOp, Program},
    selector::Selector,
};
use std::collections::BTreeMap;
use std::marker::PhantomData;

use crate::constraint_ir::ir::{ConstraintProgram, Op};
use crate::constraints::builder::ConstraintMeta;

/// Identifies a trace read: main-vs-aux, frame-step offset, column.
///
/// `row` is not part of the key â every table in this VM reads row 0 of each
/// frame step, which the IR interpreter asserts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LeafKey {
    pub main: bool,
    pub offset: u8,
    pub col: u16,
}

/// Identifies a committed column: the same read with its offset dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ColumnKey {
    pub main: bool,
    pub col: u16,
}

impl LeafKey {
    pub fn column(&self) -> ColumnKey {
        ColumnKey {
            main: self.main,
            col: self.col,
        }
    }
}

/// Which column each sumcheck factor reads and how many steps ahead â the
/// structure of [`TraceLeaves`] with none of the trace in it.
///
/// One slot per distinct `(main, offset, col)` read, one column per distinct
/// `(main, col)`. A read at offset `k` is a **view**: the shifted table the
/// sumcheck folds is derived from the column, and the column is the only thing
/// that gets committed.
///
/// The verifier holds one of these: every slot assignment comes from the
/// program, so both sides derive the same one and only the columns' values are
/// missing. Building it lives here once â if the two sides laid out slots
/// separately they could disagree, and every claim would be about the wrong
/// table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafLayout {
    /// Leaf -> factor slot, kept ordered so the layout is deterministic.
    index: BTreeMap<LeafKey, usize>,
    /// Column key -> index into the columns.
    column_index: BTreeMap<ColumnKey, usize>,
    /// Column index -> which column it is, so materialization order is
    /// explicit rather than a side effect of discovery order.
    column_keys: Vec<ColumnKey>,
    /// One entry per factor, in factor order.
    sources: Vec<FactorSource>,
    num_vars: usize,
}

impl LeafLayout {
    /// Records every trace read in `program`.
    pub fn build<F, E>(program: &ConstraintProgram<F, E>, num_vars: usize) -> Self
    where
        F: IsField,
        E: IsField,
    {
        Self::build_live(program, &vec![true; program.nodes.len()], num_vars)
    }

    /// The same, restricted to the reads a [`live_nodes`] mask keeps â so a
    /// table's dropped LogUp constraints do not drag their auxiliary columns
    /// in.
    pub fn build_live<F, E>(
        program: &ConstraintProgram<F, E>,
        live: &[bool],
        num_vars: usize,
    ) -> Self
    where
        F: IsField,
        E: IsField,
    {
        Self::build_live_over(program, live, num_vars, 0)
    }

    /// The same, with the table's first `num_main_columns` registered up front.
    ///
    /// Which columns a table commits, and in **what order**, is then a function
    /// of the table and not of the AIR that reads it: two AIRs over the same
    /// trace — an epoch's local-to-global bookend and the cross-epoch one, say
    /// — commit the same polynomial, which is the only way one proof can say to
    /// another that it committed that table.
    pub fn build_live_over<F, E>(
        program: &ConstraintProgram<F, E>,
        live: &[bool],
        num_vars: usize,
        num_main_columns: usize,
    ) -> Self
    where
        F: IsField,
        E: IsField,
    {
        let mut layout = Self {
            index: BTreeMap::new(),
            column_index: BTreeMap::new(),
            column_keys: Vec::new(),
            sources: Vec::new(),
            num_vars,
        };

        for col in 0..num_main_columns {
            layout.register_main(col as u16);
        }

        for (id, op) in program.nodes.iter().enumerate() {
            if !live.get(id).copied().unwrap_or(false) {
                continue;
            }
            let Op::Var {
                main, offset, col, ..
            } = *op
            else {
                continue;
            };
            layout.record(LeafKey { main, offset, col });
        }

        layout
    }

    /// Assigns `key` a factor slot, and its column an index if this is the
    /// first read of it. Returns the slot, which an already-recorded key keeps.
    fn record(&mut self, key: LeafKey) -> usize {
        if let Some(&slot) = self.index.get(&key) {
            return slot;
        }
        let column = match self.column_index.get(&key.column()) {
            Some(&i) => i,
            None => {
                let i = self.column_keys.len();
                self.column_keys.push(key.column());
                self.column_index.insert(key.column(), i);
                i
            }
        };
        let slot = self.sources.len();
        self.index.insert(key, slot);
        self.sources
            .push(FactorSource::shifted(column, key.offset as usize));
        slot
    }

    /// Registers main column `col`, read unshifted, as a factor â returning the
    /// slot it already had if the constraints read it too.
    ///
    /// A bus interaction reads columns the constraints may not, and the two
    /// must share one factor when they overlap: that is what makes a column
    /// read by a constraint and by a fingerprint fold once.
    pub fn register_main(&mut self, col: u16) -> usize {
        self.record(LeafKey {
            main: true,
            offset: 0,
            col,
        })
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// The columns to commit, in the order they must be materialized.
    pub fn column_keys(&self) -> &[ColumnKey] {
        &self.column_keys
    }

    /// How many columns get committed.
    pub fn num_columns(&self) -> usize {
        self.column_keys.len()
    }

    /// What each factor reads.
    pub fn sources(&self) -> &[FactorSource] {
        &self.sources
    }

    /// Number of sumcheck factors.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Factor slot of a trace read.
    pub fn index_of(&self, key: &LeafKey) -> Option<usize> {
        self.index.get(key).copied()
    }

    /// Column slot of a trace read.
    pub fn column_of(&self, key: &ColumnKey) -> Option<usize> {
        self.column_index.get(key).copied()
    }

    /// Fills the layout in, one call per column in [`column_keys`] order.
    ///
    /// [`column_keys`]: Self::column_keys
    pub fn materialize<V: IsField + 'static>(
        self,
        mut column: impl FnMut(ColumnKey) -> Vec<FieldElement<V>>,
    ) -> Result<TraceLeaves<V>, MlError> {
        let size = 1usize << self.num_vars;
        let mut columns = Vec::with_capacity(self.column_keys.len());
        for &key in &self.column_keys {
            let values = column(key);
            if values.len() != size {
                return Err(MlError::NotPowerOfTwo(values.len()));
            }
            columns.push(Mle::new(values)?);
        }
        Ok(TraceLeaves {
            layout: self,
            columns,
        })
    }
}

/// A [`LeafLayout`] with its columns materialized.
///
/// `V` is the columns' value field: the trace's, which is the base one.
#[derive(Clone, Debug)]
pub struct TraceLeaves<V: IsField> {
    pub(crate) layout: LeafLayout,
    pub(crate) columns: Vec<Mle<V>>,
}

impl<V: IsField + 'static> TraceLeaves<V> {
    /// Materializes one MLE per distinct column in `program` and records the
    /// offset each factor reads it at.
    ///
    /// `main_column` and `aux_column` return a column's values indexed by step;
    /// both must return exactly `2^num_vars` entries.
    pub fn build<F, E>(
        program: &ConstraintProgram<F, E>,
        num_vars: usize,
        main_column: impl FnMut(u16) -> Vec<FieldElement<V>>,
        aux_column: impl FnMut(u16) -> Vec<FieldElement<V>>,
    ) -> Result<Self, MlError>
    where
        F: IsField + 'static,
        E: IsField + 'static,
    {
        Self::build_live(
            program,
            &vec![true; program.nodes.len()],
            num_vars,
            main_column,
            aux_column,
        )
    }

    /// The same, restricted to the reads a [`live_nodes`] mask keeps.
    pub fn build_live<F, E>(
        program: &ConstraintProgram<F, E>,
        live: &[bool],
        num_vars: usize,
        mut main_column: impl FnMut(u16) -> Vec<FieldElement<V>>,
        mut aux_column: impl FnMut(u16) -> Vec<FieldElement<V>>,
    ) -> Result<Self, MlError>
    where
        F: IsField + 'static,
        E: IsField + 'static,
    {
        LeafLayout::build_live(program, live, num_vars).materialize(|key| {
            if key.main {
                main_column(key.col)
            } else {
                aux_column(key.col)
            }
        })
    }

    /// The slot assignment alone.
    pub fn layout(&self) -> &LeafLayout {
        &self.layout
    }

    pub fn num_vars(&self) -> usize {
        self.layout.num_vars
    }

    /// The columns to commit.
    pub fn columns(&self) -> &[Mle<V>] {
        &self.columns
    }

    /// What each factor reads.
    pub fn sources(&self) -> &[FactorSource] {
        &self.layout.sources
    }

    /// The factor tables, every shifted read materialized.
    pub fn factors(&self) -> Result<Vec<Mle<V>>, MlError> {
        self.layout
            .sources
            .iter()
            .map(|s| claim_reduce::materialize(&self.columns, s))
            .collect()
    }

    /// Number of sumcheck factors.
    pub fn len(&self) -> usize {
        self.layout.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layout.is_empty()
    }

    /// Factor slot of a trace read.
    pub fn index_of(&self, key: &LeafKey) -> Option<usize> {
        self.layout.index_of(key)
    }

    /// Column slot of a trace read.
    pub fn column_of(&self, key: &ColumnKey) -> Option<usize> {
        self.layout.column_of(key)
    }

    /// Registers main column `col`, read unshifted, as a factor â returning the
    /// slot it already had if the constraints read it too. `values` is only
    /// consulted the first time the column appears.
    pub fn register_main(
        &mut self,
        col: u16,
        values: impl FnOnce() -> Vec<FieldElement<V>>,
    ) -> Result<usize, MlError> {
        let slot = self.layout.register_main(col);
        if self.layout.column_keys.len() > self.columns.len() {
            let values = values();
            if values.len() != 1usize << self.layout.num_vars {
                return Err(MlError::NotPowerOfTwo(values.len()));
            }
            self.columns.push(Mle::new(values)?);
        }
        Ok(slot)
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

/// The nodes reachable from `roots`, marked in one reverse pass â the node list
/// is topologically ordered, so an operand always has a lower id.
///
/// A real table's program carries its LogUp constraints after the base prefix.
/// The multilinear path replaces those with a bus, so their roots are dropped â
/// and their subtrees are a large part of the DAG and read auxiliary columns
/// and challenges this path does not have. Filtering by reachability is what
/// keeps them out.
pub fn live_nodes<F: IsField, E: IsField>(
    program: &ConstraintProgram<F, E>,
    roots: &[u32],
) -> Vec<bool> {
    let mut live = vec![false; program.nodes.len()];
    for &root in roots {
        live[root as usize] = true;
    }
    for (id, op) in program.nodes.iter().enumerate().rev() {
        if !live[id] {
            continue;
        }
        match *op {
            Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => {
                live[a as usize] = true;
                live[b as usize] = true;
            }
            Op::Neg(a) | Op::Embed(a) => live[a as usize] = true,
            _ => {}
        }
    }
    live
}

/// Compiles the subgraph the roots reach into straight-line steps.
///
/// Runs once, at build time, so the evaluator never touches the `BTreeMap`, a
/// dead node, or a slot it does not need. Returns the steps and, per root, the
/// step holding its value.
fn compile<F, E>(
    program: &ConstraintProgram<F, E>,
    live: &[bool],
    leaf_index: &BTreeMap<LeafKey, usize>,
    uniforms: &Uniforms<E>,
    roots: &[u32],
) -> Result<(Vec<Step<E>>, Vec<u32>), MlError>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let mut steps = Vec::new();
    // Node id -> the step holding its value. `Embed` maps to its operand's,
    // which is why this is a mapping and not just a running count.
    let mut at = vec![u32::MAX; program.nodes.len()];
    let step_of = |at: &[u32], id: u32| -> Result<u32, MlError> {
        let slot = at[id as usize];
        if slot == u32::MAX {
            // An operand of a live node is live, so this cannot happen unless
            // the node list stopped being topologically ordered.
            return Err(MlError::UnknownPolynomial {
                index: id as usize,
                len: program.nodes.len(),
            });
        }
        Ok(slot)
    };

    for (id, op) in program.nodes.iter().enumerate() {
        if !live.get(id).copied().unwrap_or(false) {
            continue;
        }
        let step = match *op {
            Op::ConstBase(idx) => Step::Fixed(
                program.base_consts[idx as usize]
                    .clone()
                    .to_extension::<E>(),
            ),
            Op::ConstExt(idx) => Step::Fixed(program.ext_consts[idx as usize].clone()),
            Op::RapChallenge { idx } => Step::Fixed(uniforms.rap_challenges[idx as usize].clone()),
            Op::AlphaPow { idx } => Step::Fixed(uniforms.logup_alpha_powers[idx as usize].clone()),
            Op::TableOffset => Step::Fixed(uniforms.logup_table_offset.clone()),
            Op::Var {
                main, offset, col, ..
            } => {
                let key = LeafKey { main, offset, col };
                let slot = *leaf_index
                    .get(&key)
                    .expect("every Var leaf was materialized at build time");
                Step::Var(slot as u32)
            }
            Op::Add(a, b) => Step::Add(step_of(&at, a)?, step_of(&at, b)?),
            Op::Sub(a, b) => Step::Sub(step_of(&at, a)?, step_of(&at, b)?),
            Op::Mul(a, b) => Step::Mul(step_of(&at, a)?, step_of(&at, b)?),
            Op::Neg(a) => Step::Neg(step_of(&at, a)?),
            // A no-op on values: alias the operand rather than copy it.
            Op::Embed(a) => {
                at[id] = step_of(&at, a)?;
                continue;
            }
        };
        at[id] = steps.len() as u32;
        steps.push(step);
    }

    let root_steps = roots
        .iter()
        .map(|&r| step_of(&at, r))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((steps, root_steps))
}

/// `beta^i` for `i` in `0..n` — the coefficients that batch a program's roots
/// into one constraint.
///
/// Kept out of [`IrShape`] on purpose: `beta` is drawn once the trace is
/// committed, and a shape is built before that.
pub fn beta_powers<E: IsField>(beta: &FieldElement<E>, n: usize) -> Vec<FieldElement<E>> {
    let mut powers = Vec::with_capacity(n);
    let mut acc = FieldElement::<E>::one();
    for _ in 0..n {
        powers.push(acc.clone());
        acc *= beta;
    }
    powers
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

/// An AIR's constraints as one polynomial: `C = Î£_i beta^i Â· s_i(x) Â· C_i(x)`.
///
/// Factors are the trace leaves followed by one table per distinct non-trivial
/// selector, in that order.
pub struct IrPolynomial<F: IsField, E: IsField> {
    shape: IrShape<F, E>,
    beta_powers: Vec<FieldElement<E>>,
    polys: Vec<Mle<E>>,
    layout: CommitLayout<F, E>,
}

/// What the argument needs from an [`IrPolynomial`]: the columns to commit,
/// where every factor's table comes from, and the public tables in factor
/// order.
///
/// The columns are the trace, so base-field; the public tables are computed and
/// live where the challenges do.
#[derive(Clone, Debug)]
pub struct CommitLayout<F: IsField, E: IsField> {
    pub columns: Vec<Mle<F>>,
    pub kinds: Vec<FactorKind>,
    pub public_tables: Vec<Mle<E>>,
}

/// The constraint's *structure*, with no trace data in it.
///
/// [`combine`](Self::combine) turns factor values into the batched constraint,
/// and needs nothing but this â so the verifier can hold one and rebuild
/// `C(point)` from values it learned through the commitment scheme, without
/// ever seeing a column.
#[derive(Clone)]
/// Owned: once the program is compiled there is nothing left to borrow from
/// it, and a shape that borrows nothing is one less lifetime to thread.
pub struct IrShape<F: IsField, E: IsField> {
    /// Compiling folds the base field away — constants are lifted once — so
    /// nothing here holds an `F`, but the shape is still the shape of a program
    /// over the tower and says so.
    field: PhantomData<F>,
    /// The program compiled to straight-line steps.
    steps: Vec<Step<E>>,
    /// Each batched root's step, in `roots` order.
    root_steps: Vec<u32>,
    /// Roots to batch, in order.
    roots: Vec<u32>,
    /// Index into the factor list of each root's selector, or `None` when it
    /// applies on every step and the multiplication can be skipped.
    selector_of_root: Vec<Option<usize>>,
    /// The public factors, in factor order — what the verifier recomputes.
    public_selectors: Vec<Selector>,
    degree: usize,
    num_vars: usize,
}

/// One step of the compiled constraint program.
///
/// The IR is a DAG over every node the AIR emits, and evaluating it used to
/// mean walking that whole list per hypercube index: a `BTreeMap` probe for
/// each variable read, a pushed zero for each node the selected roots do not
/// reach, and a buffer the width of the entire program — tens of thousands of
/// extension elements on the precompile tables. This is the reachable subgraph
/// renumbered so operands are dense indices, resolved once.
///
/// `Embed` leaves no step: it aliases its operand's.
#[derive(Clone, Debug)]
enum Step<E: IsField> {
    /// Known before the trace: a constant or a uniform.
    Fixed(FieldElement<E>),
    /// A factor value, by slot.
    Var(u32),
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Neg(u32),
}

impl<F, E> IrPolynomial<F, E>
where
    F: IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    /// Batches every constraint in `program`, taking each one's row domain from
    /// `meta`.
    pub fn new(
        program: &ConstraintProgram<F, E>,
        leaves: TraceLeaves<F>,
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
        program: &ConstraintProgram<F, E>,
        leaves: TraceLeaves<F>,
        uniforms: Uniforms<E>,
        beta: FieldElement<E>,
    ) -> Result<Self, MlError> {
        let roots: Vec<u32> = program.roots.clone();
        let selectors = vec![Selector::ALL; roots.len()];
        Self::with_roots(program, leaves, uniforms, beta, roots, &selectors)
    }

    /// Batches the listed roots with the matching selectors.
    pub fn with_roots(
        program: &ConstraintProgram<F, E>,
        leaves: TraceLeaves<F>,
        uniforms: Uniforms<E>,
        beta: FieldElement<E>,
        roots: Vec<u32>,
        selectors: &[Selector],
    ) -> Result<Self, MlError> {
        let TraceLeaves { layout, columns } = leaves;
        let (shape, kinds) = IrShape::build(program, &layout, uniforms, roots, selectors)?;
        let beta_powers = beta_powers(&beta, shape.num_roots());
        let public_tables = shape.public_tables()?;

        let mut public = public_tables.iter().cloned();
        let polys = kinds
            .iter()
            .map(|kind| match kind {
                // The sumcheck's factors share a field, so a base view is
                // lifted for it.
                FactorKind::Committed(s) => {
                    let view = claim_reduce::materialize(&columns, s)?;
                    Mle::new(
                        view.evals()
                            .iter()
                            .map(|v| v.clone().to_extension::<E>())
                            .collect(),
                    )
                }
                FactorKind::Public => public.next().ok_or(MlError::EmptyPolynomial),
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            shape,
            beta_powers,
            polys,
            layout: CommitLayout {
                columns,
                kinds,
                public_tables,
            },
        })
    }

    /// The structure alone, for the verifier.
    pub fn shape(&self) -> &IrShape<F, E> {
        &self.shape
    }

    /// The columns to commit and, for each sumcheck factor, where its table
    /// comes from. Selectors are public factors, so they are not columns.
    pub fn committed(&self) -> (&[Mle<F>], &[FactorKind]) {
        (&self.layout.columns, &self.layout.kinds)
    }

    /// The whole layout, dropping the factor tables â the argument rebuilds
    /// those from the columns and the public tables.
    pub fn into_layout(self) -> CommitLayout<F, E> {
        self.layout
    }

    /// The structure and the layout together, dropping the factor tables.
    pub fn into_shape_and_layout(self) -> (IrShape<F, E>, CommitLayout<F, E>) {
        (self.shape, self.layout)
    }
}

impl<F, E> IrShape<F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField + 'static,
{
    /// The batched constraint's structure, from the program and a slot
    /// assignment alone.
    ///
    /// Returns the factor kinds alongside it: the trace leaves in slot order,
    /// then one public factor per distinct non-trivial selector. **The verifier
    /// builds this too** â the whole point of taking a [`LeafLayout`] rather
    /// than a materialized [`TraceLeaves`] â so the two sides cannot disagree
    /// on what a factor slot means.
    pub fn build(
        program: &ConstraintProgram<F, E>,
        leaves: &LeafLayout,
        uniforms: Uniforms<E>,
        roots: Vec<u32>,
        selectors: &[Selector],
    ) -> Result<(Self, Vec<FactorKind>), MlError> {
        if selectors.len() != roots.len() {
            return Err(MlError::VariableCountMismatch {
                expected: roots.len(),
                got: selectors.len(),
            });
        }
        let num_vars = leaves.num_vars;
        let degrees = node_degrees(program);
        let live = live_nodes(program, &roots);

        let mut kinds: Vec<FactorKind> = leaves
            .sources
            .iter()
            .copied()
            .map(FactorKind::Committed)
            .collect();

        // One table per distinct non-trivial selector, shared across roots, and
        // public: the verifier evaluates it in closed form instead of reading
        // it out of a commitment.
        let mut selector_slot: BTreeMap<usize, usize> = BTreeMap::new();
        let mut selector_of_root = Vec::with_capacity(roots.len());
        let mut public_selectors = Vec::new();
        for selector in selectors {
            if selector.is_trivial() {
                selector_of_root.push(None);
                continue;
            }
            let slot = match selector_slot.get(&selector.end_exemptions) {
                Some(&i) => i,
                None => {
                    let factor = kinds.len();
                    kinds.push(FactorKind::Public);
                    public_selectors.push(*selector);
                    selector_slot.insert(selector.end_exemptions, factor);
                    factor
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

        let (steps, root_steps) = compile(program, &live, &leaves.index, &uniforms, &roots)?;

        Ok((
            Self {
                field: PhantomData,
                steps,
                root_steps,
                roots,
                selector_of_root,
                public_selectors,
                degree,
                num_vars,
            },
            kinds,
        ))
    }

    /// The public factors' selectors, in the order they appear in the kinds.
    ///
    /// [`public_values`](Self::public_values) evaluates these and hands back
    /// the answers; a caller that is EMITTING that evaluation as code needs the
    /// selectors themselves, and [`public_tables`](Self::public_tables) is not
    /// a substitute — it builds a `2^num_vars` table per selector, which is the
    /// pass the closed form exists to avoid.
    pub fn public_selectors(&self) -> &[Selector] {
        &self.public_selectors
    }

    /// The public factors' tables, in the order they appear in the kinds.
    ///
    /// Only the prover needs them: the verifier evaluates the same selectors in
    /// closed form through [`public_values`](Self::public_values), which is why
    /// building the structure does not build these.
    pub fn public_tables(&self) -> Result<Vec<Mle<E>>, MlError> {
        self.public_selectors
            .iter()
            .map(|s| s.table(self.num_vars))
            .collect()
    }
}

impl<F, E> IrShape<F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField + 'static,
{
    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// How many roots are batched — the length `beta_powers` must have.
    pub fn num_roots(&self) -> usize {
        self.roots.len()
    }

    /// The compiled DAG, in `multilinear::program`'s vocabulary.
    ///
    /// `Step` and `program::Op` are the same six variants over the same dense
    /// operand numbering, so this is that conversion — stated once, and in the
    /// crate that owns `Step`, rather than a second spelling of the DAG in
    /// every consumer that has to walk it. `Step` itself stays private.
    ///
    /// Read together with [`root_steps`](Self::root_steps) and
    /// [`selector_of_root`](Self::selector_of_root) by the recursion emitter,
    /// which rebuilds [`combine`](Self::combine) with the batching coefficients
    /// as RUNTIME wires: `program` interns them as `Fixed` steps
    /// (`Builder::weighted_sum`), which is correct for a host that already has
    /// `beta` and wrong for a verifier that draws it from a transcript.
    pub fn steps_as_ops(&self) -> Vec<ProgramOp<E>> {
        self.steps
            .iter()
            .map(|step| match *step {
                Step::Fixed(ref c) => ProgramOp::Fixed(c.clone()),
                Step::Var(i) => ProgramOp::Var(i),
                Step::Add(a, b) => ProgramOp::Add(a, b),
                Step::Sub(a, b) => ProgramOp::Sub(a, b),
                Step::Mul(a, b) => ProgramOp::Mul(a, b),
                Step::Neg(a) => ProgramOp::Neg(a),
            })
            .collect()
    }

    /// Each batched root's step index, in `roots` order — the steps
    /// [`combine`](Self::combine) reads out of the DAG.
    pub fn root_steps(&self) -> &[u32] {
        &self.root_steps
    }

    /// Each batched root's selector factor, or `None` where the root applies on
    /// every step and the multiplication is skipped.
    pub fn selector_of_root(&self) -> &[Option<usize>] {
        &self.selector_of_root
    }

    /// The batched constraint, given each factor's value at a point.
    ///
    /// `beta_powers` batches the roots and is **not** part of the structure:
    /// the challenge behind it is drawn after the trace is committed, which a
    /// layout built before there is a transcript cannot know. Size it with
    /// [`num_roots`](Self::num_roots).
    /// The zerocheck rule as straight-line code over the factor values:
    /// [`combine`](Self::combine) times the weight at factor slot `weight`.
    ///
    /// The steps go in first and unchanged, so a step's index is its operand
    /// number; the batched roots, their selectors and the weight follow.
    pub fn program(
        &self,
        beta_powers: &[FieldElement<E>],
        weight: usize,
    ) -> Result<Program<E>, MlError> {
        if beta_powers.len() != self.roots.len() {
            return Err(MlError::VariableCountMismatch {
                expected: self.roots.len(),
                got: beta_powers.len(),
            });
        }
        let mut builder = Builder::<E>::new();
        for step in &self.steps {
            let emitted = match *step {
                Step::Fixed(ref c) => builder.fixed(c.clone()),
                Step::Var(i) => builder.var(i as usize),
                Step::Add(a, b) => builder.add(a, b),
                Step::Sub(a, b) => builder.sub(a, b),
                Step::Mul(a, b) => builder.mul(a, b),
                Step::Neg(a) => builder.neg(a),
            };
            debug_assert_eq!(
                emitted as usize,
                builder.len() - 1,
                "a step's index is its operand number"
            );
        }
        let terms: Vec<(u32, FieldElement<E>)> = self
            .root_steps
            .iter()
            .zip(beta_powers)
            .zip(&self.selector_of_root)
            .map(|((&root, beta), selector)| {
                let term = match selector {
                    Some(slot) => {
                        let s = builder.var(*slot);
                        builder.mul(root, s)
                    }
                    None => root,
                };
                (term, beta.clone())
            })
            .collect();
        let sum = builder.weighted_sum(&terms);
        let w = builder.var(weight);
        let root = builder.mul(w, sum);
        builder.finish(root)
    }

    pub fn combine(
        &self,
        beta_powers: &[FieldElement<E>],
        values: &[FieldElement<E>],
    ) -> FieldElement<E> {
        let nodes = self.run(values);
        self.root_steps
            .iter()
            .zip(beta_powers)
            .zip(&self.selector_of_root)
            .fold(FieldElement::zero(), |acc, ((&root, beta_pow), sel)| {
                let mut term = &nodes[root as usize] * beta_pow;
                if let Some(slot) = sel {
                    term *= &values[*slot];
                }
                acc + term
            })
    }

    /// The public factors' values at `point`, in factor order.
    ///
    /// A selector is structure, not data, so the verifier computes it here
    /// rather than believing a commitment for it.
    pub fn public_values(
        &self,
        point: &[FieldElement<E>],
    ) -> Result<Vec<FieldElement<E>>, MlError> {
        self.public_selectors
            .iter()
            .map(|s| s.evaluate(point))
            .collect()
    }

    /// Runs the DAG with each trace leaf taking the supplied value.
    fn run(&self, values: &[FieldElement<E>]) -> Vec<FieldElement<E>> {
        let mut out: Vec<FieldElement<E>> = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            let v = match *step {
                Step::Fixed(ref c) => c.clone(),
                Step::Var(i) => values[i as usize].clone(),
                Step::Add(a, b) => &out[a as usize] + &out[b as usize],
                Step::Sub(a, b) => &out[a as usize] - &out[b as usize],
                Step::Mul(a, b) => &out[a as usize] * &out[b as usize],
                Step::Neg(a) => -&out[a as usize],
            };
            out.push(v);
        }
        out
    }
}

impl<F, E> SumcheckPolynomial<E> for IrPolynomial<F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField + 'static,
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
        self.shape.combine(&self.beta_powers, values)
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
    /// The trace's own field: a column is base, only the challenges are not.
    type BaseE = FieldElement<Fp>;

    const COL_A: usize = 0;
    const COL_B: usize = 1;
    const COL_C: usize = 2;

    /// Two constraints in the shape real tables use:
    ///
    /// - idx 0, degree 1, reads the **next** step: `next(a) â a â b`
    /// - idx 1, degree 2, current step only: `aÂ·a â c`
    ///
    /// Unlike the Fibonacci examples these hold cyclically, so no wrap-around
    /// exemption is needed â selectors are not modelled yet (see module docs).
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
    fn satisfying_columns(num_vars: usize) -> [Vec<BaseE>; 3] {
        let size = 1usize << num_vars;
        let a: Vec<BaseE> = (0..size as u64)
            .map(|i| BaseE::from(i.wrapping_mul(7).wrapping_add(3)))
            .collect();
        // b is the cyclic forward difference, so `next(a) â a â b` vanishes
        // including across the wrap.
        let b: Vec<BaseE> = (0..size).map(|i| a[(i + 1) % size] - a[i]).collect();
        let c: Vec<BaseE> = a.iter().map(|x| x * x).collect();
        [a, b, c]
    }

    fn leaves_from(columns: &[Vec<BaseE>; 3], num_vars: usize) -> TraceLeaves<Fp> {
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
        // a@0, b@0, a@1, c@0 â the next-step read of `a` is its own factor.
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
    fn an_offset_leaf_is_a_shift_view_of_one_committed_column() {
        let num_vars = 3;
        let columns = satisfying_columns(num_vars);
        let leaves = leaves_from(&columns, num_vars);
        let factors = leaves.factors().unwrap();
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

        // Both factors read the same committed column.
        assert_eq!(leaves.sources()[cur].column, leaves.sources()[next].column);
        assert_eq!(leaves.sources()[cur].offset, 0);
        assert_eq!(leaves.sources()[next].offset, 1);

        for s in 0..size {
            assert_eq!(factors[cur].evals()[s], columns[COL_A][s]);
            assert_eq!(
                factors[next].evals()[s],
                columns[COL_A][(s + 1) % size],
                "step {s}"
            );
        }
    }

    #[test]
    fn a_next_step_read_adds_no_commitment() {
        // Four factors â a@0, b@0, a@1, c@0 â over three columns: the next-step
        // read of `a` rides on `a`'s commitment instead of taking its own.
        let num_vars = 3;
        let leaves = leaves_from(&satisfying_columns(num_vars), num_vars);
        assert_eq!(leaves.len(), 4);
        assert_eq!(leaves.columns().len(), 3);
    }

    #[test]
    fn registering_a_column_the_constraints_read_reuses_its_factor() {
        let num_vars = 3;
        let columns = satisfying_columns(num_vars);
        let mut leaves = leaves_from(&columns, num_vars);
        let before = (leaves.len(), leaves.columns().len());

        let slot = leaves
            .register_main(COL_A as u16, || unreachable!("already a factor"))
            .unwrap();

        assert_eq!(
            slot,
            leaves
                .index_of(&LeafKey {
                    main: true,
                    offset: 0,
                    col: COL_A as u16
                })
                .unwrap()
        );
        assert_eq!((leaves.len(), leaves.columns().len()), before);
    }

    #[test]
    fn registering_a_column_no_constraint_reads_adds_one_of_each() {
        // A bus can read a column the constraints ignore, and then it does need
        // a commitment.
        let num_vars = 3;
        let columns = satisfying_columns(num_vars);
        let mut leaves = leaves_from(&columns, num_vars);
        let (factors, committed) = (leaves.len(), leaves.columns().len());

        let bus_column: Vec<BaseE> = (0..(1u64 << num_vars)).map(BaseE::from).collect();
        let slot = leaves.register_main(9, || bus_column.clone()).unwrap();

        assert_eq!(slot, factors);
        assert_eq!(leaves.len(), factors + 1);
        assert_eq!(leaves.columns().len(), committed + 1);
        assert_eq!(leaves.factors().unwrap()[slot].evals(), &bus_column[..]);
    }

    /// Reads column 0 only at the next step, so its column is committed with no
    /// unshifted factor of its own.
    struct AheadOnlySet;

    impl ConstraintSet<Fp, Ext> for AheadOnlySet {
        fn max_degree(&self) -> usize {
            1
        }

        fn eval<B: ConstraintBuilder<Fp, Ext>>(&self, b: &mut B) {
            let next = b.main(1, COL_A);
            let other = b.main(0, COL_B);
            b.emit_base(0, next - other);
        }
    }

    #[test]
    fn registering_a_column_only_read_ahead_reuses_its_commitment() {
        let num_vars = 3;
        let columns = satisfying_columns(num_vars);
        let mut cb = CaptureBuilder::<Fp, Ext>::new();
        AheadOnlySet.eval(&mut cb);
        let prog = cb.finish(1).0;

        let mut leaves = TraceLeaves::build(
            &prog,
            num_vars,
            |col| columns[col as usize].clone(),
            |_| unreachable!("no aux reads"),
        )
        .unwrap();
        // Two factors â A ahead, B here â over two columns.
        assert_eq!((leaves.len(), leaves.columns().len()), (2, 2));

        let slot = leaves
            .register_main(COL_A as u16, || unreachable!("already committed"))
            .unwrap();

        // A new factor, but no new commitment: the shifted read already put
        // column A in.
        assert_eq!(slot, 2);
        assert_eq!((leaves.len(), leaves.columns().len()), (3, 2));
        assert_eq!(leaves.factors().unwrap()[slot].evals(), &columns[COL_A][..]);
    }

    #[test]
    fn a_registered_column_of_the_wrong_height_is_rejected() {
        let num_vars = 3;
        let columns = satisfying_columns(num_vars);
        let mut leaves = leaves_from(&columns, num_vars);
        assert_eq!(
            leaves
                .register_main(9, || vec![BaseE::zero(); 3])
                .unwrap_err(),
            MlError::NotPowerOfTwo(3)
        );
    }

    #[test]
    fn degree_comes_from_the_dag() {
        let num_vars = 3;
        let prog = program();
        let leaves = leaves_from(&satisfying_columns(num_vars), num_vars);
        let poly = IrPolynomial::new_unselected(&prog, leaves, Uniforms::default(), ExtE::from(5))
            .unwrap();
        // `aÂ·a â c` is the degree-2 constraint; batching does not raise it.
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
        columns[COL_C][9] += BaseE::one();

        let build = || {
            IrPolynomial::new_unselected(
                &prog,
                leaves_from(&columns, num_vars),
                Uniforms::default(),
                ExtE::from(5),
            )
            .unwrap()
        };
        let poly = build();
        assert_ne!(poly.sum_over_hypercube(), ExtE::zero());
        let degree = poly.degree();

        // `g(0)` is derived from the claim, so the lie surfaces in the residual
        // rather than inside a round â and discharging that is the caller's.
        let proof = zerocheck::prove(poly, &mut transcript()).unwrap().proof;
        let claim = zerocheck::verify(&proof, num_vars, degree, &mut transcript()).unwrap();
        assert_ne!(
            claim.constraint_evaluation(),
            Some(build().evaluate(&claim.point).unwrap()),
            "a violated constraint produced a consistent claim"
        );
    }

    #[test]
    fn violations_that_cancel_in_the_sum_are_still_caught() {
        // Perturbing a[3] breaks `next(a) â a â b` twice: at step 2 the
        // next-step read is one too high, at step 3 the current-step read is.
        // The two violations are equal and opposite, so the plain sum over the
        // cube stays zero â this is exactly the case eq(r, Â·) exists to catch,
        // and it is reachable from an ordinary AIR, not just a contrived one.
        let num_vars = 4;
        let prog = program();
        let mut columns = satisfying_columns(num_vars);
        columns[COL_A][3] += BaseE::one();
        columns[COL_C][3] = columns[COL_A][3] * columns[COL_A][3]; // keep idx 1 satisfied

        let build = |cols: &[Vec<BaseE>; 3]| {
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
        // one must go unnoticed â which is what makes the batched version's
        // rejection meaningful.
        let num_vars = 4;
        let prog = program();
        let mut columns = satisfying_columns(num_vars);
        columns[COL_B][2] += BaseE::one(); // breaks idx 0 only

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
    fn fib_columns(num_vars: usize) -> [Vec<BaseE>; 2] {
        let size = 1usize << num_vars;
        let mut c0 = vec![BaseE::one()];
        let mut c1 = vec![BaseE::one()];
        for i in 1..size {
            // s0_{i} = s0_{i-1} + s1_{i-1}; s1_{i} = s1_{i-1} + s0_{i}
            let next0 = c0[i - 1] + c1[i - 1];
            let next1 = c1[i - 1] + next0;
            c0.push(next0);
            c1.push(next1);
        }
        [c0, c1]
    }

    fn fib_leaves(columns: &[Vec<BaseE>; 2], num_vars: usize) -> TraceLeaves<Fp> {
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
        columns[0][7] += BaseE::one();

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
        let degree = poly.degree();

        let proof = zerocheck::prove(poly, &mut transcript()).unwrap().proof;
        let claim = zerocheck::verify(&proof, num_vars, degree, &mut transcript()).unwrap();
        assert_ne!(
            claim.constraint_evaluation(),
            Some(build().evaluate(&claim.point).unwrap()),
            "a violated constraint produced a consistent claim"
        );
    }

    #[test]
    fn the_fibonacci_air_commits_only_its_two_columns() {
        // Five factors: both columns at the current and the next step, plus the
        // shared selector. Two commitments: the next-step reads are views of
        // the columns, and the selector is public.
        let num_vars = 3;
        let (prog, meta) = fib_program();
        let columns = fib_columns(num_vars);
        let poly = IrPolynomial::new(
            &prog,
            fib_leaves(&columns, num_vars),
            Uniforms::default(),
            ExtE::from(5),
            &meta,
        )
        .unwrap();

        let (committed, kinds) = poly.committed();
        assert_eq!(kinds.len(), 5);
        assert_eq!(committed.len(), 2);
        assert_eq!(
            kinds.iter().filter(|k| **k == FactorKind::Public).count(),
            1
        );
    }

    #[test]
    fn the_verifier_recomputes_the_selector_it_never_receives() {
        // The public value the argument uses is the selector's closed form, and
        // it is the same table the prover folded.
        let num_vars = 3;
        let (prog, meta) = fib_program();
        let columns = fib_columns(num_vars);
        let poly = IrPolynomial::new(
            &prog,
            fib_leaves(&columns, num_vars),
            Uniforms::default(),
            ExtE::from(5),
            &meta,
        )
        .unwrap();

        let shape = poly.shape().clone();
        let point: Vec<ExtE> = (0..num_vars).map(|i| ExtE::from(31 + i as u64)).collect();
        let public = shape.public_values(&point).unwrap();
        assert_eq!(public.len(), 1);

        let table = poly.into_layout().public_tables.pop().unwrap();
        assert_eq!(public[0], table.evaluate(&point).unwrap());
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

    /// The compiled rule must agree with the walk it replaces: same roots,
    /// same beta powers, same selectors, times the weight.
    #[test]
    fn the_zerocheck_program_agrees_with_the_walk() {
        let num_vars = 3;
        let (prog, meta) = fib_program();
        let columns = fib_columns(num_vars);
        let leaves = fib_leaves(&columns, num_vars);
        let poly =
            IrPolynomial::new(&prog, leaves, Uniforms::default(), ExtE::from(5), &meta).unwrap();
        let shape = poly.shape().clone();
        let betas = beta_powers(&ExtE::from(5), shape.num_roots());

        // One more factor than the shape reads: the last is the weight.
        let width = poly.polys().len();
        let values: Vec<ExtE> = (0..=width)
            .map(|i| ExtE::from((i as u64).wrapping_mul(2654435761) + 11))
            .collect();
        let program = shape.program(&betas, width).unwrap();
        let mut scratch = Vec::new();
        assert_eq!(
            program.eval(&values, &mut scratch),
            values[width] * shape.combine(&betas, &values[..width])
        );
    }

    /// A beta power per root, or the rule would batch the wrong number of them.
    #[test]
    fn a_program_with_the_wrong_beta_count_is_rejected() {
        let num_vars = 3;
        let (prog, meta) = fib_program();
        let columns = fib_columns(num_vars);
        let leaves = fib_leaves(&columns, num_vars);
        let poly =
            IrPolynomial::new(&prog, leaves, Uniforms::default(), ExtE::from(5), &meta).unwrap();
        let shape = poly.shape().clone();
        assert!(shape.program(&[ExtE::one()], poly.polys().len()).is_err());
    }

    // ---------------------------------------------------------------
    // The whole argument over a real captured AIR.
    // ---------------------------------------------------------------

    /// Runs the full argument over the captured AIR and returns the verdict.
    ///
    /// The `next`-step read is a view of the column it shifts: it gets no
    /// commitment, and the shift kernel binds its value to that column.
    fn argue_fib(columns: &[Vec<BaseE>; 2], num_vars: usize) -> Result<(), multilinear::Error> {
        use multilinear::{
            constraint_argument::{self, CommittedTrace, TraceClaim},
            whir_chain::{ChainConfig, GrindBits},
            whir_hash::KeccakWhir,
        };

        let (prog, meta) = fib_program();
        let leaves = TraceLeaves::build(
            &prog,
            num_vars,
            |col| columns[col as usize].clone(),
            |_| unreachable!("no aux reads"),
        )?;
        let poly = IrPolynomial::new(&prog, leaves, Uniforms::default(), ExtE::from(5), &meta)?;
        let degree = poly.degree();
        let shape = poly.shape().clone();
        let layout = poly.into_layout();

        let config = ChainConfig {
            log_blowup: 2,
            log_folding: 2,
            num_queries: 3,
            grind: GrindBits::default(),
        };
        // Domain in the base field, columns in the degree-3 extension.
        let n_stack = constraint_argument::one_stack(num_vars, layout.columns.len());
        let trace = CommittedTrace::<Fp, Ext>::commit_views(
            layout.columns,
            layout.kinds,
            layout.public_tables,
            n_stack,
            &config,
        )?;
        let roots = trace.roots();
        let betas = beta_powers(&ExtE::from(5), shape.num_roots());

        let mut prover_transcript = DefaultTranscript::<Ext>::new(b"air-argument");
        let proof = constraint_argument::prove::<Fp, Ext, _, _, KeccakWhir>(
            &trace,
            |v: &[ExtE]| shape.combine(&betas, v),
            degree,
            &config,
            &mut prover_transcript,
        )?;

        let mut verifier_transcript = DefaultTranscript::<Ext>::new(b"air-argument");
        constraint_argument::verify::<Fp, Ext, _, _, _, KeccakWhir>(
            &proof,
            TraceClaim {
                roots: &roots,
                kinds: trace.kinds(),
                layout: trace.layout(),
                domain: trace.domain(),
                num_vars,
            },
            |v: &[ExtE]| shape.combine(&betas, v),
            |point: &[ExtE]| shape.public_values(point),
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
        let columns = fib_columns(num_vars);
        argue_fib(&columns, num_vars).unwrap();
    }

    #[test]
    fn a_real_air_with_a_broken_row_is_rejected_end_to_end() {
        let num_vars = 4;
        let mut columns = fib_columns(num_vars);
        columns[1][6] += BaseE::one();
        assert!(argue_fib(&columns, num_vars).is_err());
    }
}
