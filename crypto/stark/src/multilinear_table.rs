//! The argument for one table of this VM: its own constraints zerochecked, its
//! buses proved by LogUp-GKR, and both discharged in **one** sumcheck against
//! one commitment per column.
//!
//! Where the univariate prover commits auxiliary columns and proves LogUp with
//! transition constraints, this proves it with a fraction tree. So a table's
//! program is used in two parts: `roots[..num_base]` — the table's own
//! constraints — are zerochecked, and the LogUp roots after them are
//! **dropped**, the bus replacing them. Dropping them drops every auxiliary
//! read along with them, which is why only main columns are committed here.
//!
//! The bus balance is not a table's own business: its contribution is a
//! fraction, and only the sum over every table in the proof has to vanish.

use math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
    },
    traits::AsBytes,
};
use multilinear::{
    Error as MlError,
    batch::Rule,
    claim_reduce,
    constraint_argument::{self, ConstraintCore, FactorKind, TraceData},
    eq::{eq_eval, eq_mle},
    gkr::{self, FractionTree, GkrProof},
    logup,
    mle::Mle,
    stacked_eval::{self, Claimed, StackedCommitment, StackedProof},
    stacking::StackedLayout,
    whir::Domain,
    whir_chain::ChainConfig,
    whir_commit::Commitment,
};

use crate::constraint_ir::ir::ConstraintProgram;
use crate::constraints::builder::ConstraintMeta;
use crate::lookup::BusInteraction;
use crate::multilinear_air;
use crate::multilinear_air::{ColumnKey, IrShape, LeafLayout, Uniforms, live_nodes};
use crate::multilinear_logup;
use multilinear::selector::Selector;

/// A table's structure, with no trace in it: what the factors are, what gets
/// committed, and over what domain.
///
/// **Both sides build one.** Everything here comes from the program, the
/// constraint metadata, the bus interactions and the shape of the trace — never
/// from its values — so a verifier holding the same arguments derives the same
/// factor slots, the same stack and the same domain. Only the commitment roots
/// come out of the proof, and [`statement`](Self::statement) is where they join.
///
/// Keeping the construction in one place is the point: if the two sides laid
/// out factors separately they could assign a column a different slot, and
/// every claim would then be about a different table than the one committed.
pub struct TableLayout<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E>,
    E: IsField,
{
    shape: IrShape<F, E>,
    interactions: &'a [BusInteraction],
    leaves: LeafLayout,
    /// Main column -> the factor that reads it unshifted.
    slot_of: Vec<usize>,
    kinds: Vec<FactorKind>,
    num_vars: usize,
}

impl<'a, F, E> TableLayout<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E>,
    E: IsField,
{
    /// Lays out a table's factors and its stack.
    ///
    /// Every main column is committed, read or not — the trace is the trace,
    /// and a bus reads columns the constraints may not.
    ///
    /// `uniforms` is normally [`Uniforms::default`]: a table's own constraint
    /// set is base-rooted, so it reads no challenge.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        program: &'a ConstraintProgram<F, E>,
        meta: &[ConstraintMeta],
        interactions: &'a [BusInteraction],
        num_main_columns: usize,
        num_vars: usize,
        uniforms: Uniforms<E>,
    ) -> Result<Self, MlError> {
        if interactions.is_empty() {
            // A table with no bus is just `constraint_argument::prove`.
            return Err(MlError::EmptyPolynomial);
        }
        if meta.len() < program.num_base {
            return Err(MlError::VariableCountMismatch {
                expected: program.num_base,
                got: meta.len(),
            });
        }

        let roots = program.roots[..program.num_base].to_vec();
        let live = live_nodes(program, &roots);
        let mut leaves = LeafLayout::build_live(program, &live, num_vars);

        let slot_of: Vec<usize> = (0..num_main_columns)
            .map(|col| leaves.register_main(col as u16))
            .collect();

        let selectors: Vec<Selector> = meta[..program.num_base]
            .iter()
            .map(|m| Selector::except_last(m.end_exemptions))
            .collect();
        let (shape, kinds) = IrShape::build(program, &leaves, uniforms, roots, &selectors)?;

        Ok(Self {
            shape,
            interactions,
            leaves,
            slot_of,
            kinds,
            num_vars,
        })
    }

    /// The statement this layout describes, with no preprocessed columns to
    /// check. Purely structural: the commitment roots are not in it, because
    /// they come out of the proof.
    pub fn statement(&self) -> TableStatement<'_, F, E> {
        self.statement_with_preprocessed(&[])
    }

    /// The same, with the table's preprocessed columns for the verifier to
    /// check the claimed openings against — see [`TableStatement::preprocessed`].
    pub fn statement_with_preprocessed<'s>(
        &'s self,
        preprocessed: &'s [Mle<F>],
    ) -> TableStatement<'s, F, E> {
        TableStatement {
            shape: &self.shape,
            interactions: self.interactions,
            slot_of: &self.slot_of,
            preprocessed,
            kinds: &self.kinds,
            num_vars: self.num_vars,
        }
    }

    pub fn shape(&self) -> &IrShape<F, E> {
        &self.shape
    }

    /// Main column -> the factor that reads it unshifted.
    pub fn slot_of(&self) -> &[usize] {
        &self.slot_of
    }

    pub fn kinds(&self) -> &[FactorKind] {
        &self.kinds
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// How many columns get committed — main columns only.
    pub fn num_columns(&self) -> usize {
        self.leaves.num_columns()
    }

    /// The columns to commit, in the order they must be materialized.
    pub fn column_keys(&self) -> &[ColumnKey] {
        self.leaves.column_keys()
    }
}

/// One table's structure and its trace, with **no commitment of its own**.
///
/// A table used to carry its own, so a proof of N tables paid N openings — and
/// the opening is almost all of a proof. Here the tables hand their columns to
/// [`CommittedTables`], which commits every one of them together.
pub struct CommittedTable<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    layout: TableLayout<'a, F, E>,
    trace: TraceData<F, E>,
}

impl<'a, F, E> CommittedTable<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    /// Lays out the factors and materializes every main column.
    ///
    /// `main_column` returns a column's values by step, `2^num_vars` of them.
    /// Auxiliary columns are never asked for: the only constraints that read
    /// them are the LogUp ones, and the bus replaces those.
    pub fn new(
        program: &'a ConstraintProgram<F, E>,
        meta: &[ConstraintMeta],
        interactions: &'a [BusInteraction],
        num_main_columns: usize,
        num_vars: usize,
        uniforms: Uniforms<E>,
        main_column: impl FnMut(u16) -> Vec<FieldElement<F>>,
    ) -> Result<Self, MlError> {
        let layout = TableLayout::new(
            program,
            meta,
            interactions,
            num_main_columns,
            num_vars,
            uniforms,
        )?;
        Self::from_layout(layout, main_column)
    }

    /// The same against a layout already built — the same one the verifier will
    /// rebuild, so the statement and the trace cannot describe different
    /// tables.
    pub fn from_layout(
        layout: TableLayout<'a, F, E>,
        mut main_column: impl FnMut(u16) -> Vec<FieldElement<F>>,
    ) -> Result<Self, MlError> {
        let size = 1usize << layout.num_vars();
        let mut columns = Vec::with_capacity(layout.num_columns());
        for key in layout.column_keys() {
            assert!(
                key.main,
                "the bus replaces every constraint that reads an auxiliary column"
            );
            let values = main_column(key.col);
            if values.len() != size {
                return Err(MlError::NotPowerOfTwo(values.len()));
            }
            columns.push(Mle::new(values)?);
        }

        let trace = TraceData::new(
            columns,
            layout.kinds().to_vec(),
            layout.shape().public_tables()?,
        )?;
        Ok(Self { layout, trace })
    }

    /// The structure alone — what the verifier holds.
    pub fn layout(&self) -> &TableLayout<'a, F, E> {
        &self.layout
    }

    /// This table's statement.
    pub fn statement(&self) -> TableStatement<'_, F, E> {
        self.layout.statement()
    }

    pub fn kinds(&self) -> &[FactorKind] {
        self.trace.kinds()
    }

    pub fn shape(&self) -> &IrShape<F, E> {
        self.layout.shape()
    }

    /// Main column -> the factor that reads it unshifted.
    pub fn slot_of(&self) -> &[usize] {
        self.layout.slot_of()
    }

    pub fn num_vars(&self) -> usize {
        self.trace.num_vars()
    }

    /// How many columns get committed — main columns only.
    pub fn num_committed_columns(&self) -> usize {
        self.trace.columns().len()
    }
}

/// Every table's columns, committed **once**.
///
/// The opening is almost the whole proof, and a commitment costs one however
/// many columns it holds, so N tables committing separately pay N times for
/// what one commitment settles. Here the columns of every table go into one
/// stack, each keeps the height it has, and one opening answers all of them —
/// which is what [`Claimed::PerColumn`] exists for, since each table's sumcheck
/// ends at its own point.
///
/// [`Claimed::PerColumn`]: multilinear::stacked_eval::Claimed::PerColumn
pub struct CommittedTables<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    tables: Vec<CommittedTable<'a, F, E>>,
    stacked: StackedCommitment<F>,
    roots: Vec<Commitment>,
}

/// Where each table's columns start in the global column order.
///
/// Both sides derive it from the table shapes alone, so a claim about table
/// `i`'s column `c` means the same thing to each.
pub fn column_offsets(widths: &[usize]) -> Vec<usize> {
    widths
        .iter()
        .scan(0usize, |at, w| {
            let start = *at;
            *at += w;
            Some(start)
        })
        .collect()
}

/// How wide one stacked polynomial may get, in variables.
///
/// This is the knob between proof size and peak memory, and it is a resource
/// tradeoff rather than an optimization. Every extra variable halves the number
/// of polynomials — and so the number of openings, which is nearly all of a
/// proof — but doubles the biggest single allocation: a polynomial of `n`
/// variables is encoded over `2^(n + log_blowup)` field elements, and the NTT
/// wants the coefficients and the codeword alive at once.
///
/// 25 puts one codeword at 1 GiB in the base field, which fits alongside the
/// rest of the prover on a 32 GiB machine. A proof of ethrex 10tx lands in a
/// handful of polynomials rather than one per table.
pub const MAX_STACK_VARS: usize = 25;

/// The stack every table's columns share, from their shapes alone.
///
/// `shapes` is `(columns, height in variables)` per table, in table order. The
/// heights differ and that is fine: a column takes the subcube it needs and the
/// next one starts after it, so stacking wastes less than each table rounding
/// up to its own power of two.
///
/// Columns that do not fit spill into another polynomial — see
/// [`MAX_STACK_VARS`], which is what stops one proof-sized allocation.
pub fn global_layout(shapes: &[(usize, usize)]) -> Result<StackedLayout, MlError> {
    let heights: Vec<usize> = shapes
        .iter()
        .flat_map(|&(width, num_vars)| std::iter::repeat_n(num_vars, width))
        .collect();
    let cells: usize = heights.iter().map(|&m| 1usize << m).sum();
    let want = cells.next_power_of_two().trailing_zeros() as usize;
    // Never narrower than the tallest column, or it would not fit at all.
    let tallest = heights.iter().copied().max().unwrap_or(0);
    let n_stack = want.min(MAX_STACK_VARS).max(tallest);
    StackedLayout::build(&heights, n_stack)
}

impl<'a, F, E> CommittedTables<'a, F, E>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    /// Commits every table's columns into one stack.
    pub fn commit(
        tables: Vec<CommittedTable<'a, F, E>>,
        config: &ChainConfig,
    ) -> Result<Self, MlError> {
        let shapes: Vec<(usize, usize)> = tables
            .iter()
            .map(|t| (t.num_committed_columns(), t.num_vars()))
            .collect();
        let layout = global_layout(&shapes)?;

        let columns: Vec<Mle<F>> = tables
            .iter()
            .flat_map(|t| t.trace.columns().iter().cloned())
            .collect();
        let stacked = StackedCommitment::<F>::commit(layout, &columns, config)?;
        let roots = stacked.roots();
        Ok(Self {
            tables,
            stacked,
            roots,
        })
    }

    pub fn tables(&self) -> &[CommittedTable<'a, F, E>] {
        &self.tables
    }

    /// One root per stacked polynomial, for the whole proof.
    pub fn roots(&self) -> &[Commitment] {
        &self.roots
    }

    pub fn layout(&self) -> &StackedLayout {
        self.stacked.layout()
    }

    pub fn domain(&self) -> &Domain<F> {
        self.stacked.domain()
    }

    /// What the one opening settles against.
    pub fn stacked(&self) -> &StackedCommitment<F> {
        &self.stacked
    }
}

/// What the verifier holds: the table's structure and what was committed.
pub struct TableStatement<'a, F: IsFFTField + IsPrimeField, E: IsField> {
    pub shape: &'a IrShape<F, E>,
    pub interactions: &'a [BusInteraction],
    /// Main column -> the factor that reads it unshifted.
    pub slot_of: &'a [usize],
    /// The table's **preprocessed** columns, `0..n` of the main trace: the ones
    /// the program determines and the verifier can therefore recompute.
    ///
    /// Empty on the proving side, and empty for a table that has none. The
    /// commitment only says the prover stayed consistent with what it
    /// committed; these are what say it committed the right thing.
    pub preprocessed: &'a [Mle<F>],
    pub kinds: &'a [FactorKind],
    pub num_vars: usize,
}

/// Every field is a reference or a length, so copying is free — spelled out
/// rather than derived, which would demand `F: Copy` and `E: Copy` of the field
/// markers.
impl<F: IsFFTField + IsPrimeField, E: IsField> Clone for TableStatement<'_, F, E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: IsFFTField + IsPrimeField, E: IsField> Copy for TableStatement<'_, F, E> {}

/// A table's argument.
#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct TableProof<E: IsField> {
    pub gkr: GkrProof<E>,
    /// The bus's output fraction. A table's own contribution need not vanish —
    /// the balance is over every table in the proof — so both halves travel and
    /// checking the sum is the caller's. Lying about them yields an input-layer
    /// claim the trace does not answer, so nothing has to be taken on trust.
    pub bus_output: (FieldElement<E>, FieldElement<E>),
    /// The table's argument up to its columns' claims. The opening that
    /// settles them is [`MultiProof::columns`], shared with every other table.
    pub constraint: ConstraintCore<E>,
}

/// Every table's argument, and the **one** opening that settles all of them.
#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct MultiProof<F: IsField, E: IsField> {
    /// One per stacked polynomial. The verifier absorbs these before any
    /// challenge is drawn, so a prover cannot choose a commitment after seeing
    /// one — and carrying them here is what makes the proof self-contained.
    pub roots: Vec<Commitment>,
    pub tables: Vec<TableProof<E>>,
    /// Every table's columns, at each one's own point, against one stack.
    pub columns: StackedProof<F, E>,
}

/// The table's share of the bus, `p/q`.
///
/// `None` when the denominator vanished, which a random challenge makes
/// negligible.
pub fn contribution<E: IsField>(
    output: &(FieldElement<E>, FieldElement<E>),
) -> Option<FieldElement<E>> {
    output.1.inv().ok().map(|inv| &output.0 * inv)
}

/// The factor slots: the trace's factors, then `eq(r, ·)` for the zerocheck and
/// `eq(row, ·)` for the bus's two claims.
fn weights(num_trace_factors: usize) -> (usize, usize) {
    (num_trace_factors, num_trace_factors + 1)
}

/// Proves the table: its constraints vanish and its bus sums to what the proof
/// says, in one sumcheck.
///
/// Stops at the claims about the table's columns and hands back the point they
/// are claimed at; the caller settles every table's claims against the shared
/// stack in one opening.
///
/// `z` and `alpha` are the LogUp challenges, **shared across every table** in a
/// multi-table proof. The caller must have absorbed the commitment roots and
/// drawn them, identically on both sides.
pub fn prove<F, E, T>(
    table: &CommittedTable<'_, F, E>,
    z: &FieldElement<E>,
    alpha: &FieldElement<E>,
    beta: &FieldElement<E>,
    transcript: &mut T,
) -> Result<(TableProof<E>, Vec<FieldElement<E>>), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>,
{
    let interactions = multilinear_logup::interactions(
        table.layout.interactions,
        table.slot_of().len(),
        z,
        alpha,
        |col| slot(table.slot_of(), col),
    )?;

    // The input layer reads the trace's factors; they are materialized here,
    // used and dropped rather than held for the proof.
    let tree = {
        let factors = table.trace.factors()?;
        FractionTree::build(logup::input_layer(&interactions, &factors)?)?
    };
    let bus_output = tree.output();
    transcript.append_field_element(&bus_output.0);
    transcript.append_field_element(&bus_output.1);
    let gkr_out = gkr::prove(&tree, transcript)?;

    let num_vars = table.num_vars();
    let r: Vec<FieldElement<E>> = (0..num_vars)
        .map(|_| transcript.sample_field_element())
        .collect();

    let (weight_r, weight_z) = weights(table.kinds().len());
    let bus = logup::claim_statements(&interactions, &gkr_out.claim.point, num_vars, weight_z)?;

    let shape = table.shape();
    let betas = multilinear_air::beta_powers(beta, shape.num_roots());
    let zerocheck = Rule::compiled(shape.degree() + 1, shape.program(&betas, weight_r)?);

    let (constraint, point) = constraint_argument::prove_core::<F, E, T>(
        &table.trace,
        vec![eq_mle(&r)?, eq_mle(&bus.row_point)?],
        vec![zerocheck, bus.numerator, bus.denominator],
        &[
            FieldElement::zero(),
            gkr_out.claim.p.clone(),
            gkr_out.claim.q.clone(),
        ],
        transcript,
    )?;

    Ok((
        TableProof {
            gkr: gkr_out.proof,
            bus_output,
            constraint,
        },
        point,
    ))
}
/// What verifying one table leaves for the caller: its share of the bus, and
/// the claims its columns are left at.
pub type TableVerdict<E> = (
    (FieldElement<E>, FieldElement<E>),
    claim_reduce::ReducedClaim<E>,
);

/// Verifies the table and returns its bus output together with the claims its
/// columns are left at, for the caller to sum and to settle against the shared
/// stack.
pub fn verify<E, T>(
    proof: &TableProof<E>,
    statement: TableStatement<'_, impl IsFFTField + IsPrimeField + IsSubFieldOf<E> + Sync, E>,
    z: &FieldElement<E>,
    alpha: &FieldElement<E>,
    beta: &FieldElement<E>,
    transcript: &mut T,
) -> Result<TableVerdict<E>, MlError>
where
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>,
{
    let interactions = multilinear_logup::interactions(
        statement.interactions,
        statement.slot_of.len(),
        z,
        alpha,
        |col| slot(statement.slot_of, col),
    )?;

    transcript.append_field_element(&proof.bus_output.0);
    transcript.append_field_element(&proof.bus_output.1);
    let gkr_claim = gkr::verify(&proof.gkr, proof.bus_output.clone(), transcript)?;

    let num_vars = statement.num_vars;
    let r: Vec<FieldElement<E>> = (0..num_vars)
        .map(|_| transcript.sample_field_element())
        .collect();

    let (weight_r, weight_z) = weights(statement.kinds.len());
    let bus = logup::claim_statements(&interactions, &gkr_claim.point, num_vars, weight_z)?;
    let row_point = bus.row_point.clone();

    let shape = statement.shape;
    let betas = multilinear_air::beta_powers(beta, shape.num_roots());
    let zerocheck = Rule::compiled(shape.degree() + 1, shape.program(&betas, weight_r)?);

    let reduced = constraint_argument::verify_core(
        &proof.constraint,
        statement.kinds,
        statement.slot_of.len(),
        num_vars,
        &[zerocheck, bus.numerator, bus.denominator],
        &[
            FieldElement::zero(),
            gkr_claim.p.clone(),
            gkr_claim.q.clone(),
        ],
        |at: &[FieldElement<E>]| {
            // The selectors, then the two weight tables.
            let mut values = shape.public_values(at)?;
            values.push(eq_eval(&r, at)?);
            values.push(eq_eval(&row_point, at)?);
            Ok(values)
        },
        transcript,
    )?;

    check_preprocessed(statement, &reduced)?;

    Ok((proof.bus_output.clone(), reduced))
}

/// Checks the table's preprocessed columns against what the proof claims for
/// them at the reduced point.
///
/// The commitment binds the prover to the columns it committed, not to the
/// *right* ones — nothing else in the argument says a preprocessed table is the
/// one the program implies. So the verifier evaluates its own copy at the point
/// the proof settled on and demands the same value. Costs one pass over each
/// such column, which is what recomputing a preprocessed commitment costs on
/// the univariate side.
fn check_preprocessed<F, E>(
    statement: TableStatement<'_, F, E>,
    reduced: &claim_reduce::ReducedClaim<E>,
) -> Result<(), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E>,
    E: IsField,
{
    for (col, column) in statement.preprocessed.iter().enumerate() {
        let factor = slot(statement.slot_of, col)?;
        // A preprocessed column is read unshifted by construction: `TableLayout`
        // registers every main column that way. Anything else means the two
        // sides disagree about the layout, which is not a claim to compare.
        let source = statement
            .kinds
            .get(factor)
            .and_then(FactorKind::source)
            .filter(|s| s.offset == 0)
            .ok_or(MlError::UnknownPolynomial {
                index: factor,
                len: statement.kinds.len(),
            })?;
        let claimed =
            reduced
                .column_values
                .get(source.column)
                .ok_or(MlError::UnknownPolynomial {
                    index: source.column,
                    len: reduced.column_values.len(),
                })?;
        if column.evaluate_in(&reduced.point)? != *claimed {
            return Err(MlError::EvaluationMismatch);
        }
    }
    Ok(())
}
/// Proves every table in one transcript, against one commitment.
///
/// The LogUp challenges are drawn **once**, after the commitment roots are
/// absorbed: sharing them is what lets one table's send be another's receive,
/// and absorbing the roots first is what stops a prover from choosing a bus
/// after seeing them.
///
/// Each table's sumcheck leaves its columns claimed at a point of its own.
/// Those go into **one** opening at the end, which is what makes a proof of
/// many tables cost about what a proof of one does.
pub fn multi_prove<F, E, T>(
    committed: &CommittedTables<'_, F, E>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<MultiProof<F, E>, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>,
{
    for root in committed.roots() {
        transcript.append_bytes(root);
    }
    let z: FieldElement<E> = transcript.sample_field_element();
    let alpha: FieldElement<E> = transcript.sample_field_element();
    let beta: FieldElement<E> = transcript.sample_field_element();

    let mut tables = Vec::with_capacity(committed.tables().len());
    // One point and one claimed value per **column**, in the global column
    // order the stack was built in.
    let mut points: Vec<Vec<FieldElement<E>>> = Vec::new();
    let mut values: Vec<FieldElement<E>> = Vec::new();
    for table in committed.tables() {
        let (proof, point) = prove(table, &z, &alpha, &beta, transcript)?;
        for _ in 0..table.num_committed_columns() {
            points.push(point.clone());
        }
        values.extend(proof.constraint.reduce.column_values.iter().cloned());
        tables.push(proof);
    }

    let columns = stacked_eval::prove::<F, E, T>(
        &committed.stacked,
        &Claimed::PerColumn(&points),
        &values,
        config,
        transcript,
    )?;

    Ok(MultiProof {
        roots: committed.roots().to_vec(),
        tables,
        columns,
    })
}

/// Verifies every table **and the bus balance across them**, then settles every
/// column against the one commitment.
///
/// A table's own contribution need not vanish, and neither does the sum: a bus
/// whose counterparty is the statement rather than another table leaves a
/// residue, so the caller says what it owes. For this VM that is the COMMIT
/// bus carrying the program's public output — `expected` is zero exactly when
/// the program outputs nothing.
#[allow(clippy::too_many_arguments)]
pub fn multi_verify<F, E, T>(
    proof: &MultiProof<F, E>,
    statements: &[TableStatement<'_, F, E>],
    layout: &StackedLayout,
    domain: &Domain<F>,
    expected: &FieldElement<E>,
    config: &ChainConfig,
    transcript: &mut T,
) -> Result<(), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>,
{
    if proof.tables.len() != statements.len() {
        return Err(MlError::QueryCountMismatch {
            expected: statements.len(),
            got: proof.tables.len(),
        });
    }
    for root in &proof.roots {
        transcript.append_bytes(root);
    }
    let z: FieldElement<E> = transcript.sample_field_element();
    let alpha: FieldElement<E> = transcript.sample_field_element();
    let beta: FieldElement<E> = transcript.sample_field_element();

    let mut balance = FieldElement::<E>::zero();
    let mut points: Vec<Vec<FieldElement<E>>> = Vec::new();
    let mut values: Vec<FieldElement<E>> = Vec::new();
    for (table, statement) in proof.tables.iter().zip(statements) {
        let (output, reduced) = verify(table, *statement, &z, &alpha, &beta, transcript)?;
        balance += contribution(&output).ok_or(MlError::BusImbalance)?;
        for _ in 0..statement.slot_of.len() {
            points.push(reduced.point.clone());
        }
        values.extend(reduced.column_values);
    }
    if balance != *expected {
        return Err(MlError::BusImbalance);
    }

    stacked_eval::verify::<F, E, T>(
        &proof.columns,
        layout,
        &proof.roots,
        &Claimed::PerColumn(&points),
        &values,
        domain,
        config,
        transcript,
    )
}

fn slot(slots: &[usize], column: usize) -> Result<usize, MlError> {
    slots
        .get(column)
        .copied()
        .ok_or(MlError::UnknownPolynomial {
            index: column,
            len: slots.len(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::{
        extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
        goldilocks::GoldilocksField as Fp,
    };

    use multilinear::whir_chain::GrindBits;

    use crate::constraints::builder::{ConstraintBuilder, ConstraintSet, EmptyConstraints};
    use crate::examples::multi_table_lookup::{
        new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
    };
    use crate::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};
    use crate::proof::options::ProofOptions;
    use crate::traits::AIR;

    type FE = FieldElement<Fp>;
    type ExtE = FieldElement<Ext>;
    type Air<CS> = AirWithBuses<Fp, Ext, NullBoundaryConstraintBuilder, (), CS>;

    fn config() -> ChainConfig {
        ChainConfig {
            log_blowup: 2,
            log_folding: 2,
            num_queries: 3,
            grind: GrindBits::default(),
        }
    }

    /// The ADD table's own constraint: every row really is an addition.
    struct AddConstraints;

    impl ConstraintSet<Fp, Ext> for AddConstraints {
        fn max_degree(&self) -> usize {
            1
        }

        fn eval<B: ConstraintBuilder<Fp, Ext>>(&self, b: &mut B) {
            let a = b.main(0, 0);
            let addend = b.main(0, 1);
            let sum = b.main(0, 2);
            b.emit_base(0, a + addend - sum);
        }
    }

    /// The MUL table's, one degree higher.
    struct MulConstraints;

    impl ConstraintSet<Fp, Ext> for MulConstraints {
        fn max_degree(&self) -> usize {
            2
        }

        fn eval<B: ConstraintBuilder<Fp, Ext>>(&self, b: &mut B) {
            let a = b.main(0, 0);
            let factor = b.main(0, 1);
            let product = b.main(0, 2);
            b.emit_base(0, a * factor - product);
        }
    }

    /// The same tables the multi-table completeness test proves, but with the
    /// lookup tables now constraining their own rows too.
    fn airs() -> (
        Air<EmptyConstraints>,
        Air<AddConstraints>,
        Air<MulConstraints>,
    ) {
        let options = ProofOptions::default_test_options();
        let cpu = new_cpu_air_with_lookup(&options);
        let add = AirWithBuses::new(
            4,
            AuxiliaryTraceBuildData {
                interactions: new_add_air_with_lookup(&options)
                    .bus_interactions()
                    .to_vec(),
            },
            &options,
            1,
            AddConstraints,
        );
        let mul = AirWithBuses::new(
            4,
            AuxiliaryTraceBuildData {
                interactions: new_mul_air_with_lookup(&options)
                    .bus_interactions()
                    .to_vec(),
            },
            &options,
            1,
            MulConstraints,
        );
        (cpu, add, mul)
    }

    fn base(values: &[u64]) -> Vec<FE> {
        values.iter().map(|v| FE::from(*v)).collect()
    }

    fn cpu_columns() -> Vec<Vec<FE>> {
        vec![
            base(&[1, 0, 1, 0, 1, 1, 0, 0]),
            base(&[0, 1, 0, 1, 0, 0, 1, 1]),
            base(&[1, 2, 3, 4, 5, 6, 7, 8]),
            base(&[10, 20, 30, 40, 50, 60, 70, 80]),
            base(&[11, 40, 33, 160, 55, 66, 490, 640]),
        ]
    }

    fn add_columns() -> Vec<Vec<FE>> {
        vec![
            base(&[1, 3, 5, 6]),
            base(&[10, 30, 50, 60]),
            base(&[11, 33, 55, 66]),
            base(&[1, 1, 1, 1]),
        ]
    }

    fn mul_columns() -> Vec<Vec<FE>> {
        vec![
            base(&[2, 4, 7, 8]),
            base(&[20, 40, 70, 80]),
            base(&[40, 160, 490, 640]),
            base(&[1, 1, 1, 1]),
        ]
    }

    fn table<'a, CS: ConstraintSet<Fp, Ext>>(
        air: &'a Air<CS>,
        columns: &[Vec<FE>],
    ) -> Result<CommittedTable<'a, Fp, Ext>, MlError> {
        let num_vars = columns[0].len().trailing_zeros() as usize;
        // The trace goes in as it is: base-field.
        let lifted = columns.to_vec();
        CommittedTable::new(
            air.constraint_program(),
            air.constraints_meta(),
            air.bus_interactions(),
            columns.len(),
            num_vars,
            Uniforms::default(),
            |col| lifted[col as usize].clone(),
        )
    }

    /// Proves and verifies all three tables through the multi-table entry
    /// points, which check the bus balance themselves.
    fn argue(
        cpu_cols: Vec<Vec<FE>>,
        add_cols: Vec<Vec<FE>>,
        mul_cols: Vec<Vec<FE>>,
    ) -> Result<(), MlError> {
        let (cpu_air, add_air, mul_air) = airs();
        let committed = CommittedTables::commit(
            vec![
                table(&cpu_air, &cpu_cols)?,
                table(&add_air, &add_cols)?,
                table(&mul_air, &mul_cols)?,
            ],
            &config(),
        )?;

        let mut prover = DefaultTranscript::<Ext>::new(b"multilinear-table");
        let proof = multi_prove(&committed, &config(), &mut prover)?;

        // Three tables of different heights, and **one** commitment with one
        // opening for all of them.
        assert_eq!(proof.roots.len(), 1);
        assert_eq!(proof.columns.polys.len(), 1);
        for (table, table_proof) in committed.tables().iter().zip(&proof.tables) {
            // Three statements — the constraint and the bus's two claims — in
            // one pass over the table's rows.
            assert_eq!(
                table_proof.constraint.sumcheck.rounds.len(),
                table.num_vars()
            );
        }

        let statements: Vec<TableStatement<'_, Fp, Ext>> =
            committed.tables().iter().map(|t| t.statement()).collect();

        let mut verifier = DefaultTranscript::<Ext>::new(b"multilinear-table");
        multi_verify(
            &proof,
            &statements,
            committed.layout(),
            committed.domain(),
            &ExtE::zero(),
            &config(),
            &mut verifier,
        )
    }

    /// The composition this whole port is aimed at, on the repo's own
    /// multi-table example: three tables of different heights, each with its
    /// own constraints *and* its buses, every one argued in a single sumcheck
    /// against one commitment per trace — and the buses balance, checked by the
    /// verifier rather than by the caller.
    #[test]
    fn three_tables_argue_and_their_buses_balance() {
        argue(cpu_columns(), add_columns(), mul_columns()).unwrap();
    }

    #[test]
    fn a_row_that_is_not_an_addition_is_rejected() {
        let mut columns = add_columns();
        columns[2][2] += FE::one();
        assert!(argue(cpu_columns(), columns, mul_columns()).is_err());
    }

    #[test]
    fn a_row_that_is_not_a_product_is_rejected() {
        let mut columns = mul_columns();
        columns[2][1] += FE::one();
        assert!(argue(cpu_columns(), add_columns(), columns).is_err());
    }

    /// Every table verifies on its own and the proof still fails: the balance
    /// is the sum, and that is what a missing receive breaks.
    #[test]
    fn a_receive_that_never_happened_leaves_the_bus_unbalanced() {
        let mut columns = add_columns();
        columns[3][2] = FE::zero();
        assert_eq!(
            argue(cpu_columns(), columns, mul_columns()).unwrap_err(),
            MlError::BusImbalance
        );
    }

    /// A row that still adds up, but is not the one the CPU dispatched: the
    /// table's own constraint holds and the bus does not. What makes this
    /// meaningful is that the fingerprint is read off the *committed* columns.
    #[test]
    fn a_coherent_row_the_cpu_never_dispatched_unbalances_the_bus() {
        let mut columns = add_columns();
        columns[0][0] = FE::from(2);
        columns[1][0] = FE::from(9); // 2 + 9 = 11, still an addition

        assert_eq!(
            argue(cpu_columns(), columns, mul_columns()).unwrap_err(),
            MlError::BusImbalance
        );
    }

    /// The LogUp roots really are being dropped: the program has more
    /// constraints than the base prefix, and none of the auxiliary columns they
    /// read gets committed.
    #[test]
    fn only_main_columns_are_committed() {
        let (cpu_air, add_air, mul_air) = airs();

        for (air_roots, num_base, aux_width, table, main_width) in [
            {
                let table = table(&cpu_air, &cpu_columns()).unwrap();
                let program = cpu_air.constraint_program();
                (
                    program.roots.len(),
                    program.num_base,
                    cpu_air.trace_layout().1,
                    table,
                    5,
                )
            },
            {
                let table = table(&add_air, &add_columns()).unwrap();
                let program = add_air.constraint_program();
                (
                    program.roots.len(),
                    program.num_base,
                    add_air.trace_layout().1,
                    table,
                    4,
                )
            },
            {
                let table = table(&mul_air, &mul_columns()).unwrap();
                let program = mul_air.constraint_program();
                (
                    program.roots.len(),
                    program.num_base,
                    mul_air.trace_layout().1,
                    table,
                    4,
                )
            },
        ] {
            assert!(
                air_roots > num_base,
                "the fixture must have LogUp constraints to drop"
            );
            assert!(
                aux_width > 0,
                "the univariate path commits auxiliary columns"
            );
            assert_eq!(table.num_committed_columns(), main_width);
        }
    }

    /// The multilinear query count is the univariate prover's own accounting, in
    /// the same regime — so these parameters are no weaker than the FRI ones the
    /// repo already ships. One round, so no union-bound margin.
    #[test]
    fn the_query_count_matches_the_univariate_provers() {
        use crate::proof::options::GoldilocksCubicProofOptions;
        use multilinear::whir_chain::ChainConfig;

        let univariate = GoldilocksCubicProofOptions::with_params(4, 128, 20).unwrap();
        let multilinear = ChainConfig::with_security(
            2,
            4,
            4,
            128,
            GrindBits {
                query: 20,
                ..GrindBits::default()
            },
        );

        assert_eq!(
            multilinear.num_queries, univariate.fri_number_of_queries,
            "the two regimes must agree"
        );
        assert_eq!(multilinear.grind.query, univariate.grinding_factor);
    }

    #[test]
    fn a_table_with_no_bus_is_not_this_argument() {
        let options = ProofOptions::default_test_options();
        let air: Air<AddConstraints> = AirWithBuses::new(
            4,
            AuxiliaryTraceBuildData {
                interactions: Vec::new(),
            },
            &options,
            1,
            AddConstraints,
        );
        assert_eq!(
            table(&air, &add_columns()).err(),
            Some(MlError::EmptyPolynomial)
        );
    }
}
