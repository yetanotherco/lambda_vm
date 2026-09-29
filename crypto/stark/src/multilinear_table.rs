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
use std::sync::Arc;

use multilinear::{
    Error as MlError,
    batch::{Rule, Weight},
    claim_reduce,
    constraint_argument::{self, ConstraintCore, FactorKind, TraceData},
    eq::{eq_eval, eq_mle},
    gkr::{self, FractionTree, GkrProof},
    logup,
    mle::Mle,
    stacked_eval::{self, Claimed, StackedCommitment, StackedProof},
    stacking::StackedLayout,
    whir::Domain,
    whir_chain::{ChainConfig, StackVars},
    whir_commit::Commitment,
    whir_hash::{KeccakWhir, WhirHash},
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
    E: IsField + 'static,
{
    /// Lays out a table's factors and its stack.
    ///
    /// Every main column is committed, read or not, **in index order** — the
    /// trace is the trace, a bus reads columns the constraints may not, and two
    /// AIRs over the same table have to commit the same polynomial.
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
        // The main columns are registered first, in index order, so what a
        // table commits and in what order is a function of the table alone —
        // not of which of them this AIR's constraints happen to read first.
        let mut leaves = LeafLayout::build_live_over(program, &live, num_vars, num_main_columns);

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
    /// Each column is asked for exactly once, in [`column_keys`] order, so a
    /// caller holding its own copy may hand it over rather than clone it —
    /// which on a real trace is the difference between one copy of it and two.
    ///
    /// [`column_keys`]: TableLayout::column_keys
    pub fn from_layout(
        layout: TableLayout<'a, F, E>,
        mut main_column: impl FnMut(u16) -> Vec<FieldElement<F>>,
    ) -> Result<Self, MlError> {
        debug_assert!(
            {
                let mut seen: Vec<u16> = layout.column_keys().iter().map(|k| k.col).collect();
                seen.sort_unstable();
                let asked = seen.len();
                seen.dedup();
                seen.len() == asked
            },
            "a caller may move its columns in, so each one must be asked for once",
        );
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

    /// The committed columns themselves. The opening needs them: a stacked
    /// polynomial is these at their offsets, and it is never assembled.
    pub fn columns(&self) -> &[Mle<F>] {
        self.trace.columns()
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
pub struct CommittedTables<'a, F, E, H = KeccakWhir>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    tables: Vec<CommittedTable<'a, F, E>>,
    /// The epoch's columns on the card, alive as long as the tables that read
    /// them. `None` when there is no device or it would not promise the room.
    store: Option<Arc<multilinear::gpu::ResidentColumns>>,
    /// The stacks the tables are committed in, in table order. One is the usual
    /// case; more than one exists so a table can have a commitment of its own —
    /// which is what binds the same table across two proofs, since a table has
    /// no root of its own when it shares a stack.
    groups: Vec<StackedCommitment<F, H>>,
    /// How many tables each group holds, in order.
    sizes: Vec<usize>,
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

/// The stack every table's columns share, from their shapes and the format's
/// stack cap alone.
///
/// `shapes` is `(columns, height in variables)` per table, in table order. The
/// heights differ and that is fine: a column takes the subcube it needs and the
/// next one starts after it, so stacking wastes less than each table rounding
/// up to its own power of two.
///
/// Columns that do not fit spill into another polynomial at `cap` — see
/// [`StackVars`], which is what stops one proof-sized allocation. The cap is
/// the config's (`ChainConfig::format.stack`), so the prover and the verifier
/// stack under the same one, and never a proof's.
pub fn global_layout(shapes: &[(usize, usize)], cap: StackVars) -> Result<StackedLayout, MlError> {
    let heights: Vec<usize> = shapes
        .iter()
        .flat_map(|&(width, num_vars)| std::iter::repeat_n(num_vars, width))
        .collect();
    StackedLayout::build(&heights, stack_height(shapes, cap))
}

/// The height, in variables, [`global_layout`] stacks `shapes` at under `cap`:
/// the smallest power of two that holds every column, at most `cap`, and never
/// narrower than the tallest column, or it would not fit at all.
///
/// ★ Monotone in the shapes: a group's height is at most that of any shape
/// list containing it, so every group of a proof stacks at most as tall as all
/// of its shapes stacked together would.
pub fn stack_height(shapes: &[(usize, usize)], cap: StackVars) -> usize {
    let cells: usize = shapes
        .iter()
        .map(|&(width, num_vars)| width << num_vars)
        .sum();
    let want = cells.next_power_of_two().trailing_zeros() as usize;
    let tallest = shapes
        .iter()
        .filter(|&&(width, _)| width > 0)
        .map(|&(_, num_vars)| num_vars)
        .max()
        .unwrap_or(0);
    want.min(cap.get()).max(tallest)
}

/// One layout per group, from the shapes, how the tables are split and the
/// stack cap.
pub fn global_layouts(
    shapes: &[(usize, usize)],
    sizes: &[usize],
    cap: StackVars,
) -> Result<Vec<StackedLayout>, MlError> {
    if sizes.iter().sum::<usize>() != shapes.len() {
        return Err(MlError::QueryCountMismatch {
            expected: shapes.len(),
            got: sizes.iter().sum(),
        });
    }
    let mut layouts = Vec::with_capacity(sizes.len());
    let mut at = 0usize;
    for &size in sizes {
        layouts.push(global_layout(&shapes[at..at + size], cap)?);
        at += size;
    }
    Ok(layouts)
}

impl<'a, F, E, H> CommittedTables<'a, F, E, H>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    /// Commits every table's columns into one stack.
    pub fn commit(
        tables: Vec<CommittedTable<'a, F, E>>,
        config: &ChainConfig,
    ) -> Result<Self, MlError> {
        let sizes = [tables.len()];
        Self::commit_grouped(tables, &sizes, config)
    }

    /// The same, with the tables split across several stacks.
    ///
    /// `sizes` is how many tables each group takes, in order, and must cover
    /// them all. A group of one is a table with a commitment of its own: the
    /// only way to say "this is the same table" to another proof, since a table
    /// that shares a stack has no root to compare.
    ///
    /// Everything else is unchanged — the tables are argued in one transcript
    /// against one set of roots, and each group is opened once.
    pub fn commit_grouped(
        mut tables: Vec<CommittedTable<'a, F, E>>,
        sizes: &[usize],
        config: &ChainConfig,
    ) -> Result<Self, MlError> {
        if sizes.iter().sum::<usize>() != tables.len() {
            return Err(MlError::QueryCountMismatch {
                expected: tables.len(),
                got: sizes.iter().sum(),
            });
        }
        // The device ledger's peak over the whole commit, under
        // `LAMBDA_VM_BASE_SPLIT=1` — `RESERVED HW`'s `commit`.
        let __rw_commit = multilinear::whir_split::open_reserved();
        // The epoch's columns, on the card once. Four things read them — the
        // commitment, the sumcheck's factors, the evaluation at the reduction
        // point and the opening's message — and each used to upload its own
        // copy.
        let all: Vec<&Mle<F>> = tables.iter().flat_map(|t| t.columns()).collect();
        let store = multilinear::gpu::upload_columns(&all).map(Arc::new);
        // Where each table's columns start in it.
        let mut firsts = Vec::with_capacity(tables.len());
        let mut column_at = 0usize;
        for table in &tables {
            firsts.push(column_at);
            column_at += table.num_committed_columns();
        }
        drop(all);

        let mut groups = Vec::with_capacity(sizes.len());
        let mut roots = Vec::new();
        let mut at = 0usize;
        for &size in sizes {
            let group = &tables[at..at + size];
            let shapes: Vec<(usize, usize)> = group
                .iter()
                .map(|t| (t.num_committed_columns(), t.num_vars()))
                .collect();
            let layout = global_layout(&shapes, config.format.stack)?;
            // By reference: the stack copies every column into its own buffer,
            // and the trace holds the originals for the rest of the proof.
            let columns: Vec<&Mle<F>> = group.iter().flat_map(|t| t.columns()).collect();
            let stacked = StackedCommitment::<F, H>::commit(
                layout,
                &columns,
                store.as_ref().map(|store| (&**store, firsts[at])),
                config,
            )?;
            roots.extend(stacked.roots());
            groups.push(stacked);
            at += size;
        }
        // Each table points at its own run, so its factors and its reduction
        // read them where they lie.
        if let Some(store) = &store {
            for (table, first) in tables.iter_mut().zip(&firsts) {
                table.trace.set_resident(store.clone(), *first);
            }
        }
        multilinear::whir_split::close_reserved(
            &multilinear::whir_split::RESERVED_COMMIT,
            __rw_commit,
        );
        Ok(Self {
            tables,
            store,
            groups,
            sizes: sizes.to_vec(),
            roots,
        })
    }

    pub fn tables(&self) -> &[CommittedTable<'a, F, E>] {
        &self.tables
    }

    /// Every group's roots, in order — what the transcript absorbs.
    pub fn roots(&self) -> &[Commitment] {
        &self.roots
    }

    /// How many tables each group holds.
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// The stacks, one per group.
    pub fn groups(&self) -> &[StackedCommitment<F, H>] {
        &self.groups
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

/// ★★★ THE ROOTS BLOCK: every root into the transcript, then the three shared
/// challenges — used by `multi_prove`, by `multi_verify`, AND by every caller
/// that REPLAYS the block on a fork of the transcript.
///
/// # Why it is one function
///
/// The two sides must absorb the same roots in the same order and draw the same
/// challenges after them. Written twice, they can drift — and a drift here is
/// silent in the worst way, because a consistently wrong order still verifies:
/// prover and verifier agree with each other and disagree with the
/// specification. Sharing the code makes that particular disagreement
/// unspellable rather than merely tested for.
///
/// # ⛔ The replays are callers too, and forgetting one is how this broke
///
/// A verifier that needs `z` and `alpha` BEFORE `multi_verify` runs — to compute
/// the COMMIT bus's counterparty, which is a function of them — replays this
/// block on a clone of the transcript. Such a replay is a third side of the same
/// agreement, and it is public for exactly that reason: when the block grew the
/// derived root, the two call sites inside this module grew with it and a
/// hand-rolled replay in the prover crate did not, so the counterparty was
/// computed at challenges no table had been checked at and every epoch that
/// published output failed on `BusImbalance`. The epochs that published nothing
/// could not see it, because the counterparty is zero there without reading
/// either challenge.
///
/// So: never spell this loop out again. Call [`absorb_roots`], pass the same
/// `derived` list the verification will be handed, and draw what you consume.
///
/// # The order, and why `derived` is last
///
/// The carried roots — the ones the proof actually contains — go in first, then
/// any derived root the verifier computed for itself. "Last" means last among
/// the roots, NOT after the challenges: a root absorbed after `z` is
/// indistinguishable IN `z` from a root never absorbed at all, so it would bind
/// nothing while looking wired from every counter and every byte gate.
///
/// That is the one placement no round-trip can catch, which is why
/// `the_roots_block_binds_the_derived_root_to_the_first_challenge` compares this
/// against an independently built transcript rather than against the other side.
pub fn absorb_roots_and_challenge<E, T>(
    transcript: &mut T,
    carried: &[Commitment],
    derived: &[Commitment],
) -> (FieldElement<E>, FieldElement<E>, FieldElement<E>)
where
    E: IsField + 'static,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>,
{
    absorb_roots::<E, T>(transcript, carried, derived);
    (
        transcript.sample_field_element(),
        transcript.sample_field_element(),
        transcript.sample_field_element(),
    )
}

/// The half of the roots block that can DRIFT: which roots, in which order.
///
/// Split out because a replay needs this half and not the other. The order is a
/// shared fact and is shared here; how many challenges are drawn afterwards is
/// the caller's own business, because a replay works on a fork it throws away
/// and only the first two challenges ever leave it.
///
/// ⚠ A challenge nobody reads is neither free nor invisible. Each is a sponge
/// squeeze; `hash_metrics` counts squeezes on every transcript instance, a
/// clone included; and the output buffer hands out four candidates per squeeze,
/// so an unread draw moves a pinned squeeze count by an amount that depends on
/// where the buffer happened to be. A replay draws exactly what it consumes,
/// and the order it draws in is checked against
/// [`absorb_roots_and_challenge`] by a test rather than by a comment.
pub fn absorb_roots<E, T>(transcript: &mut T, carried: &[Commitment], derived: &[Commitment])
where
    E: IsField + 'static,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>,
{
    for root in carried {
        transcript.append_bytes(root);
    }
    for root in derived {
        transcript.append_bytes(root);
    }
}

/// Where one stacked column of a prepared commitment is settled: a table, and
/// **which** of that table's preprocessed columns.
///
/// ★ THE TABLE INDEX IS WHAT GENERALISED; THE COLUMN INDEX IS WHAT KEEPS IT
/// HONEST. DECODE's prepared commitment covers ONE table's leading five columns.
/// The cross-epoch genesis stack covers one table PER DENSE PAGE, each settled
/// at its own reduced point — so the opening needed a table LIST, which is the
/// change. The column index rides along because a list of tables alone cannot
/// say WHICH columns of each, and the contract those columns must satisfy is
/// exact: see [`prefix_at`].
///
/// ⚠ A CROSS-EPOCH PAGE'S STACK CARRIES BOTH ITS COLUMNS, and that is a cost
/// paid to avoid changing a contract. A GLOBAL_MEMORY page presents
/// `[OFFSET, INIT]`; only INIT is worth an opening, since OFFSET is the identity
/// ramp whose extension costs `num_vars − 1` rows to check outright. But
/// `check_preprocessed` skips a PREFIX, and `{1}` is not one — so the stack
/// takes `{0, 1}`, the page loses its ramp, and `settled_out_of_band` stays a
/// count. On the block that is 51 rows lost against 282 gained, about 230 on a
/// hybrid costing order `10^5`.
///
/// ⚠ BOTH FIELDS ARE PART OF THE STATEMENT, NOT HINTS. The opening binds this
/// column to ONE table's reduced point, and a prover who could aim a pinned
/// column at another table's point — or at another of the same table's columns
/// — would be settling it against challenges it was never bound to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedColumn {
    /// Which table's reduced point this column is settled at.
    pub table: usize,
    /// Which of that table's PREPROCESSED columns it is, by index.
    pub column: usize,
}

/// Where `table`'s prepared columns sit in `at`, and how many there are —
/// `None` when the commitment does not cover that table at all.
///
/// ⛔⛔ **THE PREFIX CONTRACT, CHECKED RATHER THAN ASSUMED.**
/// [`check_preprocessed`] settles a table's FIRST `n` preprocessed columns:
/// it iterates `.skip(settled_out_of_band)`, and the in-guest mirror does the
/// same with `&columns[plan.settled..]`. So a commitment covering a table's
/// columns `{1}` — or `{0, 2}` — is INEXPRESSIBLE, and a caller that built one
/// would have the host skip the wrong columns while the opening settled others,
/// with every value gate still green.
///
/// A leading COUNT could not express that defect because it could not describe
/// the intent either. [`PreparedColumn`] can, so this is where the two meet:
/// the entries for one table must be CONTIGUOUS in `at` and must be exactly
/// `0..n`, and anything else is an error rather than a silent reinterpretation.
///
/// ⚠ CONTIGUITY IS PART OF IT, not a tidiness rule. The opening's claimed
/// values are taken as the run `values[start .. start + n]` of that table's own
/// columns, so entries interleaved with another table's would settle this
/// table's commitment against a slice that is not its own.
fn prefix_at(at: &[PreparedColumn], table: usize) -> Result<Option<(usize, usize)>, MlError> {
    let Some(first) = at.iter().position(|c| c.table == table) else {
        return Ok(None);
    };
    let n = at[first..].iter().take_while(|c| c.table == table).count();
    // Contiguous: nothing belonging to this table sits outside that run.
    if at.iter().filter(|c| c.table == table).count() != n {
        return Err(MlError::QueryCountMismatch {
            expected: n,
            got: at.iter().filter(|c| c.table == table).count(),
        });
    }
    // And the run is the prefix `0..n`, in order.
    for (offset, entry) in at[first..first + n].iter().enumerate() {
        if entry.column != offset {
            return Err(MlError::UnknownPolynomial {
                index: entry.column,
                len: n,
            });
        }
    }
    Ok(Some((first, n)))
}

/// One table's leading `columns` preprocessed columns — what a prepared
/// commitment over a single table's prefix covers, which is DECODE's case.
///
/// Returned OWNED because [`Prepared`] borrows the slice and a self-referential
/// struct cannot hand one out; the caller holds it for as long as the opening,
/// exactly as it already holds the borrowed column references.
pub fn leading_columns(table: usize, columns: usize) -> Vec<PreparedColumn> {
    (0..columns)
        .map(|column| PreparedColumn { table, column })
        .collect()
}

/// What the verifier needs to settle a prepared commitment it derived itself.
///
/// ⚠ `roots` are DERIVED — recomputed from the ELF by the verifier — never read
/// from the proof. A root taken from the proof would be a value absorbed before
/// it was checked.
pub struct PreparedCheck<'a, F>
where
    F: IsFFTField + IsPrimeField + 'static,
{
    /// Recomputed from the ELF, not carried in the proof.
    pub roots: &'a [Commitment],
    pub layout: &'a StackedLayout,
    pub domain: &'a Domain<F>,
    /// Where each stacked column is settled, parallel to the stack's columns.
    pub at: &'a [PreparedColumn],
}

/// A commitment built OUTSIDE this proof, opened at the points of the tables
/// whose preprocessed columns it covers.
///
/// DECODE's five preprocessed columns are ELF-derived: the same bytes in every
/// epoch of every run of that program. Committing them per epoch and then
/// re-evaluating their MLEs per epoch is work that depends on nothing the epoch
/// chose. This carries a commitment built once per ELF, so the epoch pays an
/// opening instead of five 2^20 folds.
///
/// ★ ONE STACK, SEVERAL TABLES, SEVERAL POINTS — and no new machinery for it.
/// The cross-epoch proof's genesis pages are one table each, so their INIT
/// columns are claimed at as many different reduced points as there are dense
/// pages. [`stacked_eval::Claimed::PerColumn`] has always resolved the point
/// per column — it is what every group opening in [`multi_prove`] already uses
/// — so the generality was in the opening all along and only this struct named
/// a single table. See [`PreparedColumn`].
pub struct Prepared<'a, F, H>
where
    F: IsFFTField + IsPrimeField + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
{
    /// Built once per ELF, outside this proof.
    pub commitment: &'a StackedCommitment<F, H>,
    /// The columns it was committed over, in that order.
    pub columns: &'a [&'a Mle<F>],
    /// Where each of those columns is settled, in the same order.
    pub at: &'a [PreparedColumn],
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
    /// Every table's columns, at each one's own point — one opening per
    /// commitment group, in group order.
    pub columns: Vec<StackedProof<F, E>>,
    /// ★★ The out-of-band preprocessed opening, when the proof carries one.
    ///
    /// DECODE's five preprocessed columns are ELF-derived, so their commitment
    /// is a function of the program alone and is built ONCE per ELF rather than
    /// per epoch. This opens it at DECODE's own reduced point, and the verifier
    /// compares the opened values against the ones DECODE's table settled on —
    /// which replaces re-evaluating five 2^20 MLEs on every epoch.
    ///
    /// ⚠ Its ROOT IS NOT IN [`roots`](Self::roots), and that is deliberate. The
    /// verifier derives the root from the ELF it already holds and absorbs the
    /// DERIVED value; a root carried here would be a second copy that a reader
    /// assumes is checked, and a value that has not been checked must never
    /// reach the transcript. `roots` therefore keeps exactly the group roots it
    /// always had, and the group indexing that slices it is untouched.
    pub preprocessed: Option<StackedProof<F, E>>,
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
///
/// Public because a verifier written OUTSIDE this crate — the WHIR recursion's
/// in-guest one — has to index the same two slots, and a second spelling of a
/// convention both sides must agree on is exactly the kind of drift nothing
/// would catch.
pub fn weight_slots(num_trace_factors: usize) -> (usize, usize) {
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
    prebuilt: Option<FractionTree<E>>,
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

    // The input layer reads the trace's factors. On a device they stay there
    // for the sumcheck too — they are the biggest thing the argument holds —
    // and the layer is written where they are; on the host they are
    // materialized here, used and dropped.
    //
    // `prebuilt` is a tree the pipeline built during the previous table's argue
    // (its output read here, at the consume site). Prefetch never changes WHAT
    // is built — same factors, same (z, alpha, beta) — only when, so a prebuilt
    // tree yields byte-for-byte the same proof as building it here now.
    let tree = match prebuilt {
        Some(tree) => tree,
        None => match table
            .trace
            .reside_from_columns()
            .and_then(|resident| logup::resident_tree(&interactions, resident))
        {
            Some(tree) => tree,
            // No device took them, so the host builds what it needs: the
            // factors, used here and by the sumcheck that follows.
            None => {
                let factors = table.trace.factors()?;
                FractionTree::build(logup::input_layer(&interactions, &factors)?)?
            }
        },
    };
    let bus_output = tree.output();
    transcript.append_field_element(&bus_output.0);
    transcript.append_field_element(&bus_output.1);
    let gkr_out = gkr::prove(&tree, transcript)?;

    let num_vars = table.num_vars();
    let r: Vec<FieldElement<E>> = (0..num_vars)
        .map(|_| transcript.sample_field_element())
        .collect();

    let (weight_r, weight_z) = weight_slots(table.kinds().len());
    let bus = logup::claim_statements(&interactions, &gkr_out.claim.point, num_vars, weight_z)?;

    let shape = table.shape();
    let betas = multilinear_air::beta_powers(beta, shape.num_roots());
    let zerocheck = Rule::compiled(shape.degree() + 1, shape.program(&betas, weight_r)?);

    // Under `LAMBDA_VM_ARGUE_DEVICE_TABLES` the weights go as their points and
    // the card builds them where the zerocheck folds them; the host builds them
    // only if the card turns the rounds down.
    let weights = if multilinear::gpu::argue_device_tables() {
        vec![Weight::Eq(r), Weight::Eq(bus.row_point.clone())]
    } else {
        vec![
            Weight::Table(eq_mle(&r)?),
            Weight::Table(eq_mle(&bus.row_point)?),
        ]
    };
    let (constraint, point) = constraint_argument::prove_core::<F, E, T>(
        &table.trace,
        weights,
        vec![zerocheck, bus.numerator, bus.denominator],
        &[
            FieldElement::zero(),
            gkr_out.claim.p.clone(),
            gkr_out.claim.q.clone(),
        ],
        transcript,
    )?;
    // The sumcheck folded the factors where they lay, so they are spent — and
    // the table outlives its own argument. Letting go of them here is what
    // keeps a proof from holding every table's at once.
    table.trace.release_device();

    Ok((
        TableProof {
            gkr: gkr_out.proof,
            bus_output,
            constraint,
        },
        point,
    ))
}
/// Whether the depth-1 tree prefetch is on. `LFM_WHIR_PREFETCH=1` (or `true`)
/// turns it on; unset or anything else is off, and off is byte-for-byte today's
/// serial path. Read once.
fn prefetch_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("LFM_WHIR_PREFETCH")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Build one table's fraction tree ahead of its turn, WITHOUT reading its output
/// (the deferred sync), so its kernels overlap the current table's argue. `None`
/// unless the table is device-resident AND a second tree fits without evicting
/// retention — prefetch never displaces argue or the round-2 retained layers.
/// The tree depends only on `(z, alpha)` via the interactions, not on `beta` or
/// the transcript, so building it early cannot change the proof.
fn try_prefetch_tree<F, E>(
    table: &CommittedTable<'_, F, E>,
    z: &FieldElement<E>,
    alpha: &FieldElement<E>,
) -> Option<logup::PrefetchedTree<E>>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let interactions = multilinear_logup::interactions(
        table.layout.interactions,
        table.slot_of().len(),
        z,
        alpha,
        |col| slot(table.slot_of(), col),
    )
    .ok()?;
    let resident = table.trace.reside_from_columns()?;
    logup::resident_tree_deferred(&interactions, resident)
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
    statement: TableStatement<
        '_,
        impl IsFFTField + IsPrimeField + IsSubFieldOf<E> + Sync + 'static,
        E,
    >,
    z: &FieldElement<E>,
    alpha: &FieldElement<E>,
    beta: &FieldElement<E>,
    transcript: &mut T,
    settled_out_of_band: usize,
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

    let (weight_r, weight_z) = weight_slots(statement.kinds.len());
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

    check_preprocessed(statement, &reduced, settled_out_of_band)?;

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
/// ★★ `settled_out_of_band` is how many of this table's LEADING preprocessed
/// columns a PREPARED OPENING already settled, and it is the whole saving: those
/// columns are tied to an ELF-derived commitment by an opening at this very
/// point, so evaluating their MLEs here would prove the same thing a second time
/// at `5 * 2^20` folds an epoch.
///
/// ⚠ A LEADING COUNT AND NOT A SET, deliberately, and a prepared commitment is
/// shaped to fit it rather than the other way round. A cross-epoch genesis page
/// presents `[OFFSET, INIT]` and only INIT is worth an opening — but `{1}` is
/// not a prefix, so the stack carries BOTH of that page's columns and settles
/// `2`. The page loses its OFFSET ramp, which on the block is 51 rows against
/// the 282 the three extra stack columns cost: about 230 rows on a hybrid
/// costing order `10^5`, in exchange for this contract not moving. See
/// [`prefix_at`], which is where a commitment that cannot be expressed as a
/// prefix is refused rather than silently reinterpreted.
///
/// ⚠ It must be driven by the same `PreparedCheck` value that drives the
/// opening, never by a flag a caller sets on its own — otherwise it is a switch
/// that turns off a check with nothing put in its place. The caller derives both
/// from one `at` slice; see `multi_verify`.
fn check_preprocessed<F, E>(
    statement: TableStatement<'_, F, E>,
    reduced: &claim_reduce::ReducedClaim<E>,
    settled_out_of_band: usize,
) -> Result<(), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
{
    if settled_out_of_band > statement.preprocessed.len() {
        return Err(MlError::QueryCountMismatch {
            expected: statement.preprocessed.len(),
            got: settled_out_of_band,
        });
    }
    for (col, column) in statement
        .preprocessed
        .iter()
        .enumerate()
        .skip(settled_out_of_band)
    {
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

/// `at` as validated `(table, count)` runs, in stack order.
///
/// ★ ONE WALK DOES BOTH JOBS: it is where the prefix and contiguity contracts
/// of [`prefix_at`] are enforced, and it is where the stack's COLUMN ORDER is
/// pinned. `at` must visit each table once, in the order the stack's columns
/// were committed, so run `k` of the commitment settles table `k`'s claims. A
/// reordering would settle one table's columns against another's commitment
/// while every individual value gate stayed green — the failure the AIR set's
/// single-derivation rule exists to prevent, one level down.
fn prepared_runs(at: &[PreparedColumn]) -> Result<Vec<(usize, usize)>, MlError> {
    let mut runs = Vec::new();
    let mut i = 0usize;
    while i < at.len() {
        let table = at[i].table;
        let (first, n) = prefix_at(at, table)?.ok_or(MlError::UnknownPolynomial {
            index: table,
            len: at.len(),
        })?;
        // The walk is in `at` order, so a table's run must begin where we are.
        // A table seen twice reaches here with `first` behind `i`.
        if first != i {
            return Err(MlError::QueryCountMismatch {
                expected: i,
                got: first,
            });
        }
        runs.push((table, n));
        i += n;
    }
    Ok(runs)
}

/// What [`prepared_claims`] hands back: the point each settled column is claimed
/// at, and the value it is claimed to take there, both in stack order.
///
/// ⚠ NAMED ONLY TO KEEP THAT SIGNATURE READABLE, and it carries no bound — a
/// bound on a type alias is not enforced, and `FieldElement`'s own is checked
/// at every use. That is the form `sumcheck::RoundGroup` and
/// `batch::ResidentProof` already take in this workspace.
type PreparedClaims<E> = (Vec<Vec<FieldElement<E>>>, Vec<FieldElement<E>>);

/// The points and claimed values a prepared opening is settled against, gathered
/// in stack order.
///
/// ★ A GATHER AND NOT ONE SLICE, which is the difference a multi-table prepared
/// commitment makes. A single table's prepared prefix is one contiguous run of
/// the global column order — `&points[at..at + width]`, as it always was. A
/// stack spanning three genesis pages is THREE such runs at three different
/// tables' offsets, and no single slice names them.
///
/// ⚠ Each run is still CONTIGUOUS and still that table's own leading columns,
/// which is what keeps [`check_preprocessed`]'s prefix contract intact: the
/// columns this settles are exactly the columns that check skips.
///
/// ⚠ The values are the ones THOSE TABLES' OWN arguments settled on. That is
/// what makes the opening a check on the pinned columns rather than on a second
/// copy of them, and it is why no separate equality is asserted anywhere.
fn prepared_claims<E: IsField>(
    runs: &[(usize, usize)],
    points: &[Vec<FieldElement<E>>],
    values: &[FieldElement<E>],
) -> Result<PreparedClaims<E>, MlError> {
    let mut at_points = Vec::new();
    let mut at_values = Vec::new();
    for &(start, n) in runs {
        let end = start + n;
        at_points.extend(
            points
                .get(start..end)
                .ok_or(MlError::QueryCountMismatch {
                    expected: end,
                    got: points.len(),
                })?
                .iter()
                .cloned(),
        );
        at_values.extend(
            values
                .get(start..end)
                .ok_or(MlError::QueryCountMismatch {
                    expected: end,
                    got: values.len(),
                })?
                .iter()
                .cloned(),
        );
    }
    Ok((at_points, at_values))
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
pub fn multi_prove<F, E, T, H>(
    committed: &CommittedTables<'_, F, E, H>,
    config: &ChainConfig,
    transcript: &mut T,
    prepared: Option<Prepared<'_, F, H>>,
) -> Result<MultiProof<F, E>, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    // ★★ The transcript's hash must BE the configuration's. Not "should": a
    // caller that passes a keccak transcript under an RPX `H` does not compile.
    //
    // `DefaultTranscript`'s hash parameter has a default, so `DefaultTranscript::<E>`
    // is a keccak transcript that looks like it names no hash. Every call site
    // wrote exactly that, and the RPX configuration therefore ran an RPX Merkle
    // backend, an RPX grind and a KECCAK sponge — through four measured A/Bs,
    // with no instrument disagreeing, because nothing failed: the proofs were
    // valid and the arms did differ from each other.
    //
    // Requiring each site to NAME a hash would not have caught it; a site can
    // name the wrong one. An equality the compiler checks is what makes the
    // half-configured arm unspellable, and it costs the several hundred STARK
    // call sites nothing, because they are not generic over `H`.
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>,
{
    // ⚠ The prepared root is NOT added to `MultiProof::roots`. The verifier
    // derives it from the ELF and absorbs the derived value; a copy carried in
    // the proof would be a field a reader assumes is checked.
    // `StackedCommitment::roots` builds a fresh `Vec`, so it is bound here
    // rather than borrowed from a temporary.
    // ── the argument's split, under `LAMBDA_VM_BASE_SPLIT=1` ──────────────
    // The guard counts concurrent proves so a record that could have mixed two
    // says so; it releases on DROP, which is what the `?`s below need.
    let _split = multilinear::whir_split::begin_prove();

    let prepared_roots: Vec<Commitment> = prepared
        .as_ref()
        .map(|p| p.commitment.roots())
        .unwrap_or_default();
    let __sp_challenge = multilinear::whir_split::mark();
    let (z, alpha, beta) =
        absorb_roots_and_challenge::<E, T>(transcript, committed.roots(), &prepared_roots);
    multilinear::whir_split::add(&multilinear::whir_split::CHALLENGE, __sp_challenge);

    let mut tables = Vec::with_capacity(committed.tables().len());
    // One point and one claimed value per **column**, in the global column
    // order the stack was built in.
    let mut points: Vec<Vec<FieldElement<E>>> = Vec::new();
    let mut values: Vec<FieldElement<E>> = Vec::new();
    // Where each table's columns start in the global column order. A prepared
    // commitment may span several tables, so every start is kept rather than
    // the one a single-table opening needed.
    let mut table_starts = Vec::with_capacity(committed.tables().len());
    // Prefetch state: when enabled, each iteration builds the NEXT table's tree
    // while this table's argue idles the GPU on per-round host round-trips.
    // Subordinate to argue and to retention (see `logup::resident_tree_deferred`),
    // gated behind LFM_WHIR_PREFETCH for a one-binary A/B — off is byte-exact.
    let prefetch = prefetch_enabled();
    let mut held: Option<logup::PrefetchedTree<E>> = None;
    // The device ledger's peak over the argument alone — `RESERVED HW`'s
    // `argue`. It opens on what the commits left promised (the codewords, the
    // layers they kept, and whatever room a group still holds) and rises by
    // what the tables reserve.
    let __rw_argue = multilinear::whir_split::open_reserved();
    for (__sp_at, table) in committed.tables().iter().enumerate() {
        table_starts.push(points.len());
        // ⛔ SERIAL, and the instrument says so rather than a reader inferring
        // it: this loop has no rayon and no `k`, so `ARGUE` is a WALL as well
        // as a sum. The per-table maximum is kept beside it because the sum
        // alone cannot tell fifty even tables from one that dominates, and
        // those two want opposite levers.
        // Depth-1 prefetch: finalize the tree built during the previous table's
        // argue (its output read here — the sync the prefetch deferred — and by
        // now its kernels have run), then kick off the NEXT table's tree so its
        // fold kernels overlap THIS table's per-round host round-trips. Both are
        // timed in this table's argue window, as the eager build is; a working
        // overlap shows up as a cheap finalize the next iteration.
        let __sp_table = multilinear::whir_split::mark();
        let this_tree = held.take().and_then(|p| p.into_tree());
        if prefetch && let Some(next) = committed.tables().get(__sp_at + 1) {
            held = try_prefetch_tree::<F, E>(next, &z, &alpha);
        }
        let (proof, point) = prove(table, &z, &alpha, &beta, transcript, this_tree)?;
        let __sp_secs = multilinear::whir_split::add(&multilinear::whir_split::ARGUE, __sp_table);
        multilinear::whir_split::note_table(__sp_at, __sp_secs);
        for _ in 0..table.num_committed_columns() {
            points.push(point.clone());
        }
        values.extend(proof.constraint.reduce.column_values.iter().cloned());
        tables.push(proof);
    }
    multilinear::whir_split::close_reserved(&multilinear::whir_split::RESERVED_ARGUE, __rw_argue);

    // One opening per group, over that group's columns. The points and values
    // are in the global column order, so a group takes the slice its tables
    // span.
    let mut columns = Vec::with_capacity(committed.groups().len());
    multilinear::whir_split::note_groups(committed.groups().len());
    // The ledger's peak over every opening — the groups' and the prepared
    // one — as `RESERVED HW`'s `open`.
    let __rw_open = multilinear::whir_split::open_reserved();
    // ⛔ CLEARED AT THE OPENING OF THE WINDOW, not merely read at its close.
    // The six are process-global accumulators, so whatever ran the chain
    // earlier in this process is still sitting in them; reading at the end
    // alone attributes that to this group loop. The first fixture run showed
    // it as a NEGATIVE remainder — the six summed to 16.40s inside an
    // `open_groups` of 12.77s, which is the one arithmetic a sum of parts
    // cannot produce honestly. Clearing here makes "group openings only" a
    // property of the window rather than an assumption about callers.
    let _ = multilinear::whir_split::take_chain();
    let __sp_groups = multilinear::whir_split::mark();
    let mut table_at = 0usize;
    let mut column_at = 0usize;
    for (group, &size) in committed.groups().iter().zip(committed.sizes()) {
        let width: usize = committed.tables()[table_at..table_at + size]
            .iter()
            .map(|t| t.num_committed_columns())
            .sum();
        // The same columns, in the same order, the group was committed over.
        let group_columns: Vec<&Mle<F>> = committed.tables()[table_at..table_at + size]
            .iter()
            .flat_map(|t| t.trace.columns())
            .collect();
        columns.push(stacked_eval::prove::<F, E, T, H>(
            group,
            &group_columns,
            committed.store.as_ref().map(|store| (&**store, column_at)),
            &Claimed::PerColumn(&points[column_at..column_at + width]),
            &values[column_at..column_at + width],
            config,
            transcript,
        )?);
        table_at += size;
        column_at += width;
    }

    // The out-of-band opening, each column at the reduced point of the table
    // whose preprocessed column it is. `PerColumn` because those points are in
    // general different — a cross-epoch genesis page is a table of its own — and
    // settling each column at its own table's point is what makes the opened
    // values comparable to that table's own claimed values.
    //
    // ⚠ A single-table prepared commitment is the special case where every one
    // of these points is the same one, and it produces the same weight shares
    // and therefore the same bytes as the `Shared` form it replaces:
    // `Claimed::point` hands back that one point for every column under either
    // variant, and `Claimed` reaches nothing but the weight.
    multilinear::whir_split::add(&multilinear::whir_split::OPEN_GROUPS, __sp_groups);
    // ⛔ THE SIX ARE TAKEN HERE AND NOWHERE LATER. The prepared opening below
    // runs the same chain and writes the same slots, so a take placed after it
    // would fold DECODE's chain into `open_groups` — and arm E would close, on
    // a number that is not what its name says. This is the only point at which
    // the slots hold the GROUP openings and nothing else.
    multilinear::whir_split::note_chain(multilinear::whir_split::take_chain());

    let __sp_prepared = multilinear::whir_split::mark();
    let preprocessed = match prepared {
        Some(prepared) => {
            if prepared.at.len() != prepared.columns.len() {
                return Err(MlError::QueryCountMismatch {
                    expected: prepared.columns.len(),
                    got: prepared.at.len(),
                });
            }
            let runs: Vec<(usize, usize)> = prepared_runs(prepared.at)?
                .into_iter()
                .map(|(table, n)| {
                    let start = *table_starts.get(table).ok_or(MlError::UnknownPolynomial {
                        index: table,
                        len: table_starts.len(),
                    })?;
                    Ok((start, n))
                })
                .collect::<Result<_, MlError>>()?;
            let (at_points, at_values) = prepared_claims(&runs, &points, &values)?;
            Some(stacked_eval::prove::<F, E, T, H>(
                prepared.commitment,
                prepared.columns,
                None,
                &Claimed::PerColumn(&at_points),
                &at_values,
                config,
                transcript,
            )?)
        }
        None => None,
    };
    multilinear::whir_split::add(&multilinear::whir_split::OPEN_PREPARED, __sp_prepared);
    multilinear::whir_split::close_reserved(&multilinear::whir_split::RESERVED_OPEN, __rw_open);

    Ok(MultiProof {
        roots: committed.roots().to_vec(),
        tables,
        columns,
        preprocessed,
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
pub fn multi_verify<F, E, T, H>(
    proof: &MultiProof<F, E>,
    statements: &[TableStatement<'_, F, E>],
    layouts: &[StackedLayout],
    domains: &[Domain<F>],
    sizes: &[usize],
    expected: &FieldElement<E>,
    config: &ChainConfig,
    transcript: &mut T,
    prepared: Option<PreparedCheck<'_, F>>,
) -> Result<(), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    // ★★ The transcript's hash must BE the configuration's. Not "should": a
    // caller that passes a keccak transcript under an RPX `H` does not compile.
    //
    // `DefaultTranscript`'s hash parameter has a default, so `DefaultTranscript::<E>`
    // is a keccak transcript that looks like it names no hash. Every call site
    // wrote exactly that, and the RPX configuration therefore ran an RPX Merkle
    // backend, an RPX grind and a KECCAK sponge — through four measured A/Bs,
    // with no instrument disagreeing, because nothing failed: the proofs were
    // valid and the arms did differ from each other.
    //
    // Requiring each site to NAME a hash would not have caught it; a site can
    // name the wrong one. An equality the compiler checks is what makes the
    // half-configured arm unspellable, and it costs the several hundred STARK
    // call sites nothing, because they are not generic over `H`.
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>,
{
    if proof.tables.len() != statements.len() {
        return Err(MlError::QueryCountMismatch {
            expected: statements.len(),
            got: proof.tables.len(),
        });
    }
    if layouts.len() != sizes.len()
        || domains.len() != sizes.len()
        || proof.columns.len() != sizes.len()
        || sizes.iter().sum::<usize>() != statements.len()
    {
        return Err(MlError::QueryCountMismatch {
            expected: sizes.len(),
            got: proof.columns.len(),
        });
    }
    let (z, alpha, beta) = absorb_roots_and_challenge::<E, T>(
        transcript,
        &proof.roots,
        prepared.as_ref().map(|p| p.roots).unwrap_or(&[]),
    );

    let mut balance = FieldElement::<E>::zero();
    let mut points: Vec<Vec<FieldElement<E>>> = Vec::new();
    let mut values: Vec<FieldElement<E>> = Vec::new();
    // Where each table's columns start in the global column order. A prepared
    // commitment may span several tables, so every start is kept.
    let mut table_starts = Vec::with_capacity(statements.len());
    for (index, (table, statement)) in proof.tables.iter().zip(statements).enumerate() {
        table_starts.push(points.len());
        // ★ ONE VALUE drives both halves: the count the opening settles is the
        // count `check_preprocessed` skips, and both come off the same `at`
        // slice. A second, independent knob would be a way to switch off a check
        // with nothing in its place.
        //
        // ⚠ `prefix_at` is what makes the COUNT safe here. `at` names a column
        // per stacked column, so it CAN describe a non-prefix that this count
        // could not express — and `prefix_at` refuses that rather than letting
        // the count silently reinterpret it as `0..n`.
        let settled = match prepared.as_ref() {
            Some(p) => prefix_at(p.at, index)?.map(|(_, n)| n).unwrap_or(0),
            None => 0,
        };
        // ⚠ THE ASSERT THAT MATTERS. The opening covers `settled` of this
        // table's columns; skipping more than that would drop a preprocessed
        // check nothing replaced.
        if settled > statement.preprocessed.len() {
            return Err(MlError::QueryCountMismatch {
                expected: statement.preprocessed.len(),
                got: settled,
            });
        }
        let (output, reduced) = verify(table, *statement, &z, &alpha, &beta, transcript, settled)?;
        balance += contribution(&output).ok_or(MlError::BusImbalance)?;
        for _ in 0..statement.slot_of.len() {
            points.push(reduced.point.clone());
        }
        values.extend(reduced.column_values);
    }
    if balance != *expected {
        return Err(MlError::BusImbalance);
    }

    // Each group settles its own columns against its own roots, in the order
    // the prover opened them.
    let mut statement_at = 0usize;
    let mut column_at = 0usize;
    let mut root_at = 0usize;
    for (((opening, layout), domain), &size) in
        proof.columns.iter().zip(layouts).zip(domains).zip(sizes)
    {
        let width: usize = statements[statement_at..statement_at + size]
            .iter()
            .map(|s| s.slot_of.len())
            .sum();
        let roots = proof
            .roots
            .get(root_at..root_at + layout.num_polys())
            .ok_or(MlError::QueryCountMismatch {
                expected: root_at + layout.num_polys(),
                got: proof.roots.len(),
            })?;
        stacked_eval::verify::<F, E, T, H>(
            opening,
            layout,
            roots,
            &Claimed::PerColumn(&points[column_at..column_at + width]),
            &values[column_at..column_at + width],
            domain,
            config,
            transcript,
        )?;
        statement_at += size;
        column_at += width;
        root_at += layout.num_polys();
    }

    // ★★★ THE PREPARED OPENING, and check (d) with it.
    //
    // (d) is NOT a separate assertion here, and that is deliberate. The values
    // handed to `stacked_eval::verify` are the ones EACH COLUMN'S OWN TABLE
    // settled on, gathered at that table's point — so the opening has to prove
    // the pinned commitment takes exactly those values at exactly those points.
    // Two copies of the same columns are tied by the check that already exists
    // rather than by an equality someone has to remember to write.
    //
    // The alternative shape — verify the opening against its own claimed
    // values, then assert those equal the tables' — is one line longer and one
    // line forgettable. This one cannot be omitted without deleting the call.
    if let Some(prepared) = prepared {
        let opening = proof
            .preprocessed
            .as_ref()
            .ok_or(MlError::QueryCountMismatch {
                expected: 1,
                got: 0,
            })?;
        let runs: Vec<(usize, usize)> = prepared_runs(prepared.at)?
            .into_iter()
            .map(|(table, n)| {
                let start = *table_starts.get(table).ok_or(MlError::UnknownPolynomial {
                    index: table,
                    len: table_starts.len(),
                })?;
                Ok((start, n))
            })
            .collect::<Result<_, MlError>>()?;
        let (at_points, at_values) = prepared_claims(&runs, &points, &values)?;
        stacked_eval::verify::<F, E, T, H>(
            opening,
            prepared.layout,
            prepared.roots,
            &Claimed::PerColumn(&at_points),
            &at_values,
            prepared.domain,
            config,
            transcript,
        )?;
    }
    Ok(())
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
            format: multilinear::whir_chain::ChainFormat::DEFAULT,
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
        let committed = CommittedTables::<_, _, KeccakWhir>::commit(
            vec![
                table(&cpu_air, &cpu_cols)?,
                table(&add_air, &add_cols)?,
                table(&mul_air, &mul_cols)?,
            ],
            &config(),
        )?;

        let mut prover = DefaultTranscript::<Ext>::new(b"multilinear-table");
        let proof = multi_prove(&committed, &config(), &mut prover, None)?;

        // Three tables of different heights, and **one** commitment with one
        // opening for all of them.
        assert_eq!(proof.roots.len(), 1);
        assert_eq!(proof.columns.len(), 1);
        assert_eq!(proof.columns[0].polys.len(), 1);
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
        multi_verify::<_, _, _, KeccakWhir>(
            &proof,
            &statements,
            std::slice::from_ref(committed.groups()[0].layout()),
            std::slice::from_ref(committed.groups()[0].domain()),
            committed.sizes(),
            &ExtE::zero(),
            &config(),
            &mut verifier,
            None,
        )
    }

    /// The composition this whole port is aimed at, on the repo's own
    /// multi-table example: three tables of different heights, each with its
    /// own constraints *and* its buses, every one argued in a single sumcheck
    /// against one commitment per trace — and the buses balance, checked by the
    /// verifier rather than by the caller.
    /// ★★★ THE ROOTS BLOCK BINDS THE DERIVED ROOT TO THE FIRST CHALLENGE.
    ///
    /// This is the guard for the one placement nothing else catches. Step 2's
    /// prover test sees a root that is never absorbed, and its sibling sees one
    /// aimed at the wrong table — but a root absorbed AFTER `z` left both green,
    /// because `alpha` and `beta` still move and the proof still differs.
    ///
    /// A round-trip cannot close it either: prover and verifier call the same
    /// block, so a consistently late absorb verifies on both sides. The two
    /// halves agree with each other and disagree with the specification.
    ///
    /// So this compares against an INDEPENDENTLY BUILT transcript rather than
    /// against the other side — the shape every check on this branch that held
    /// has in common. The reference absorbs carried-then-derived and draws three
    /// challenges; the production block must produce the same three.
    ///
    /// ⚠ The first challenge is the one that matters. Moving the absorb after
    /// `z` leaves `alpha` and `beta` correct, so a test comparing only the later
    /// two would pass on exactly the mutation this exists to catch.
    #[test]
    fn the_roots_block_binds_the_derived_root_to_the_first_challenge() {
        use crypto::fiat_shamir::is_transcript::IsTranscript;

        let carried: Vec<Commitment> = vec![[0x11; 32], [0x22; 32]];
        let derived: Vec<Commitment> = vec![[0xAB; 32]];

        // The specification, built by hand: carried roots, then the derived one,
        // then the three challenges.
        let mut reference = DefaultTranscript::<Ext>::new(b"roots-block");
        for root in carried.iter().chain(derived.iter()) {
            reference.append_bytes(root);
        }
        let want = (
            reference.sample_field_element(),
            reference.sample_field_element(),
            reference.sample_field_element(),
        );

        let mut got_transcript = DefaultTranscript::<Ext>::new(b"roots-block");
        let got = absorb_roots_and_challenge::<Ext, _>(&mut got_transcript, &carried, &derived);

        assert_eq!(
            got.0, want.0,
            "the FIRST challenge does not match a transcript that absorbed the \
             derived root before drawing it — the root is absorbed late, or not \
             at all, and binds nothing"
        );
        assert_eq!(got.1, want.1, "alpha diverged");
        assert_eq!(got.2, want.2, "beta diverged");

        // …and the reference really is sensitive to the placement, or the
        // assertions above hold for a reason that is not the ordering.
        let mut late = DefaultTranscript::<Ext>::new(b"roots-block");
        for root in &carried {
            late.append_bytes(root);
        }
        let late_z: ExtE = late.sample_field_element();
        assert_ne!(
            late_z, want.0,
            "drawing before the derived root gives the same first challenge as \
             drawing after it — this test cannot see the placement"
        );
    }

    #[test]
    fn three_tables_argue_and_their_buses_balance() {
        argue(cpu_columns(), add_columns(), mul_columns()).unwrap();
    }

    /// The same three tables at a height where the claim reduce's
    /// columns-on-the-card knob (`LAMBDA_VM_ARGUE_DEVICE_COLUMNS`) has work to
    /// move: CPU is 2^14 rows of 5 columns and ADD 2^14 of 4 — exactly 2^16
    /// cells, the threshold — so a card values both once they are resident,
    /// while MUL, 2^13 of 4, stays on the host either way. Every column is
    /// under 2^16 rows, so at the default the host walks all thirteen.
    ///
    /// Half the CPU's rows add and half multiply, each met by one receive; ADD
    /// is padded with rows that add up (0 + 0 = 0) and receive nothing.
    fn tall_columns() -> [Vec<Vec<FE>>; 3] {
        let (adds, muls) = (1u64 << 13, 1u64 << 13);
        let mut cpu = vec![Vec::new(); 5];
        let mut add = vec![Vec::new(); 4];
        let mut mul = vec![Vec::new(); 4];
        let push = |table: &mut Vec<Vec<FE>>, row: &[FE]| {
            assert_eq!(table.len(), row.len(), "a row of the table's width");
            for (column, value) in table.iter_mut().zip(row) {
                column.push(*value);
            }
        };
        for i in 0..adds {
            let (a, b) = (FE::from(i + 1), FE::from(3 * i + 7));
            let c = a + b;
            push(&mut cpu, &[FE::one(), FE::zero(), a, b, c]);
            push(&mut add, &[a, b, c, FE::one()]);
        }
        for _ in 0..adds {
            push(&mut add, &[FE::zero(); 4]);
        }
        for i in 0..muls {
            let (a, b) = (FE::from(i + 2), FE::from(5 * i + 3));
            let c = a * b;
            push(&mut cpu, &[FE::zero(), FE::one(), a, b, c]);
            push(&mut mul, &[a, b, c, FE::one()]);
        }
        [cpu, add, mul]
    }

    /// The tests below switch process-wide overrides — some arm a fault — so
    /// they never run at the same time.
    static ARGUE_OVERRIDES: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The three AIRs [`airs`] builds, borrowed as one.
    type ThreeAirs = (
        Air<EmptyConstraints>,
        Air<AddConstraints>,
        Air<MulConstraints>,
    );

    /// One arm of a knob's A/B: the commitment it proved against, the proof,
    /// the transcript after it, how many columns the card and the host valued
    /// at the reduction points, and how many challenge tables the card built,
    /// while it ran.
    struct TallArm<'a> {
        committed: CommittedTables<'a, Fp, Ext, KeccakWhir>,
        proof: MultiProof<Fp, Ext>,
        transcript: DefaultTranscript<Ext>,
        on_card: u64,
        on_host: u64,
        tables_on_card: u64,
    }

    /// Commits the tall tables afresh and proves them with
    /// `LAMBDA_VM_ARGUE_DEVICE_COLUMNS` and `LAMBDA_VM_ARGUE_DEVICE_TABLES` as
    /// asked.
    fn prove_tall<'a>(
        airs: &'a ThreeAirs,
        columns: &[Vec<Vec<FE>>; 3],
        on: bool,
        tables: bool,
    ) -> TallArm<'a> {
        let committed = CommittedTables::<_, _, KeccakWhir>::commit(
            vec![
                table(&airs.0, &columns[0]).unwrap(),
                table(&airs.1, &columns[1]).unwrap(),
                table(&airs.2, &columns[2]).unwrap(),
            ],
            &config(),
        )
        .unwrap();
        multilinear::gpu::force_argue_device_columns(Some(on));
        multilinear::gpu::force_argue_device_tables(Some(tables));
        let (card, host, built) = (
            multilinear::gpu::evaluate_calls(),
            multilinear::gpu::host_evaluate_calls(),
            multilinear::gpu::argue_tables_on_card(),
        );
        let mut transcript = DefaultTranscript::<Ext>::new(b"multilinear-table");
        let proof = multi_prove(&committed, &config(), &mut transcript, None);
        let (on_card, on_host, tables_on_card) = (
            multilinear::gpu::evaluate_calls() - card,
            multilinear::gpu::host_evaluate_calls() - host,
            multilinear::gpu::argue_tables_on_card() - built,
        );
        multilinear::gpu::force_argue_device_columns(None);
        multilinear::gpu::force_argue_device_tables(None);
        TallArm {
            committed,
            proof: proof.unwrap_or_else(|e| panic!("columns {on}, tables {tables}: {e:?}")),
            transcript,
            on_card,
            on_host,
            tables_on_card,
        }
    }

    fn verify_tall(
        committed: &CommittedTables<'_, Fp, Ext, KeccakWhir>,
        proof: &MultiProof<Fp, Ext>,
    ) -> Result<(), MlError> {
        let statements: Vec<TableStatement<'_, Fp, Ext>> =
            committed.tables().iter().map(|t| t.statement()).collect();
        let mut verifier = DefaultTranscript::<Ext>::new(b"multilinear-table");
        multi_verify::<_, _, _, KeccakWhir>(
            proof,
            &statements,
            std::slice::from_ref(committed.groups()[0].layout()),
            std::slice::from_ref(committed.groups()[0].domain()),
            committed.sizes(),
            &ExtE::zero(),
            &config(),
            &mut verifier,
            None,
        )
    }

    /// The first table whose argument's canonical bytes differ between two
    /// proofs — the name a failed identity gives — or `None`.
    fn first_table_that_differs(a: &MultiProof<Fp, Ext>, b: &MultiProof<Fp, Ext>) -> Option<usize> {
        let bytes = |p: &MultiProof<Fp, Ext>, k: usize| {
            p.tables.get(k).map(|t| bincode::serialize(t).unwrap())
        };
        (0..a.tables.len().max(b.tables.len())).find(|&k| bytes(a, k) != bytes(b, k))
    }

    /// Whether a device took this process's columns: `false` in a build or on
    /// a machine without one.
    fn a_device() -> bool {
        multilinear::gpu::reserve_budget() > 0
    }

    /// ★ `LAMBDA_VM_ARGUE_DEVICE_COLUMNS` moves no byte of the proof. The tall
    /// tables proved with the knob off and on give the same canonical bytes,
    /// table by table and whole, the same transcript after them and the same
    /// next challenge, and both verify.
    ///
    /// On a device the arms are also shown to have taken different paths —
    /// the card valued 9 columns with the knob on and none with it off — so the
    /// comparison is not of a path with itself. Without one, what is compared
    /// is the host walk against the same walk spread over the pool (under
    /// `parallel`). ⚠ Run it alone (`--exact`): the counters are process-wide.
    ///
    /// ```text
    /// cargo test --release -p stark --features cuda,parallel,multilinear/cuda,multilinear/parallel \
    ///     --lib -- multilinear_table::tests::the_argument_proves_the_same_bytes_with_its_columns_on_the_card --exact
    /// ```
    #[test]
    fn the_argument_proves_the_same_bytes_with_its_columns_on_the_card() {
        use crypto::fiat_shamir::is_transcript::IsTranscript;

        let _overrides = ARGUE_OVERRIDES.lock().unwrap_or_else(|e| e.into_inner());
        let airs = airs();
        let columns = tall_columns();
        let mut off = prove_tall(&airs, &columns, false, false);
        let mut on = prove_tall(&airs, &columns, true, false);

        if a_device() {
            assert_eq!(
                (off.on_card, off.on_host),
                (0, 13),
                "knob off: the host must value all 13 columns (card, host)"
            );
            assert_eq!(
                (on.on_card, on.on_host),
                (9, 4),
                "knob on: the card must value CPU's 5 and ADD's 4 (card, host)"
            );
            eprintln!(
                "argue device columns: the card valued {} columns",
                on.on_card
            );
        } else {
            assert_eq!(
                (off.on_card, on.on_card),
                (0, 0),
                "no device valued a column"
            );
            eprintln!("argue device columns: no device; the host walk compared with itself");
        }
        assert_eq!(
            first_table_that_differs(&off.proof, &on.proof),
            None,
            "a table's argument changed with its columns on the card"
        );
        assert_eq!(
            bincode::serialize(&off.proof).unwrap(),
            bincode::serialize(&on.proof).unwrap(),
            "the proofs' canonical bytes differ"
        );
        assert_eq!(
            off.transcript.state(),
            on.transcript.state(),
            "the transcripts parted"
        );
        assert_eq!(
            off.transcript.sample_field_element(),
            on.transcript.sample_field_element(),
            "the next challenge moved"
        );
        verify_tall(&off.committed, &off.proof).expect("the knob-off proof verifies");
        verify_tall(&on.committed, &on.proof).expect("the knob-on proof verifies");
    }

    /// ⛔ The identity above can fail. With the fault armed the card hands back
    /// its first column's value plus one; the knob-on proof must then differ
    /// from the knob-off one at CPU — the first table the card values — and
    /// must not verify. Needs a device: there is no card value to corrupt
    /// without one, and it says so rather than passing.
    #[test]
    fn a_wrong_card_value_changes_the_argument_and_fails_it() {
        let _overrides = ARGUE_OVERRIDES.lock().unwrap_or_else(|e| e.into_inner());
        let airs = airs();
        let columns = tall_columns();
        if !a_device() {
            eprintln!("argue device columns: SKIPPED, no device to corrupt a value on");
            return;
        }
        let off = prove_tall(&airs, &columns, false, false);
        multilinear::gpu::force_column_value_fault(true);
        let faulted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            prove_tall(&airs, &columns, true, false)
        }));
        multilinear::gpu::force_column_value_fault(false);
        let faulted = faulted.expect("the faulted arm proves");
        assert_eq!(faulted.on_card, 9, "the fault is on the card's path");
        assert_eq!(
            first_table_that_differs(&off.proof, &faulted.proof),
            Some(0),
            "the identity must name CPU, the first table the card valued"
        );
        assert_eq!(
            verify_tall(&faulted.committed, &faulted.proof),
            Err(MlError::ShiftedReadMismatch),
            "a corrupted column value must fail the claim reduce's check"
        );
    }

    /// ★ `LAMBDA_VM_ARGUE_DEVICE_TABLES` moves no byte of the proof, alone or
    /// with `LAMBDA_VM_ARGUE_DEVICE_COLUMNS`: the same canonical bytes table by
    /// table and whole, the same transcript and next challenge, and each proof
    /// verifies.
    ///
    /// On a device the arms are shown to take their own paths: the card builds
    /// 12 tables with the knob on — per table, the zerocheck's two `eq` weights
    /// and the reduce's shift table and batched column at the one offset these
    /// AIRs read — and none with it off. ⚠ Run it alone (`--exact`): the
    /// counters are process-wide.
    #[test]
    fn the_argument_proves_the_same_bytes_with_its_tables_on_the_card() {
        use crypto::fiat_shamir::is_transcript::IsTranscript;

        let _overrides = ARGUE_OVERRIDES.lock().unwrap_or_else(|e| e.into_inner());
        let airs = airs();
        let columns = tall_columns();
        let mut today = prove_tall(&airs, &columns, false, false);
        verify_tall(&today.committed, &today.proof).expect("today's proof verifies");
        let today_state = today.transcript.state();
        let today_next = today.transcript.sample_field_element();
        let device = a_device();
        for (on, tables) in [(false, true), (true, true)] {
            let what = format!("columns {on}, tables {tables}");
            let mut arm = prove_tall(&airs, &columns, on, tables);
            if device {
                assert_eq!(
                    (today.tables_on_card, arm.tables_on_card),
                    (0, 12),
                    "{what}: the tables the card built (today, this arm)"
                );
                let columns_moved = if on { (9, 4) } else { (0, 13) };
                assert_eq!(
                    (arm.on_card, arm.on_host),
                    columns_moved,
                    "{what}: the column values (card, host)"
                );
                eprintln!(
                    "argue device tables: {what}: the card built {} tables",
                    arm.tables_on_card
                );
            } else {
                assert_eq!(arm.tables_on_card, 0, "{what}: no device built a table");
                eprintln!(
                    "argue device tables: no device; the host's tables compared with themselves"
                );
            }
            assert_eq!(
                first_table_that_differs(&today.proof, &arm.proof),
                None,
                "{what}: a table's argument changed"
            );
            assert_eq!(
                bincode::serialize(&today.proof).unwrap(),
                bincode::serialize(&arm.proof).unwrap(),
                "{what}: the proofs' canonical bytes differ"
            );
            assert_eq!(
                today_state,
                arm.transcript.state(),
                "{what}: the transcripts parted"
            );
            assert_eq!(
                today_next,
                arm.transcript.sample_field_element(),
                "{what}: the next challenge moved"
            );
            verify_tall(&arm.committed, &arm.proof).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        }
    }

    /// ⛔ The identity above can fail. With a fault armed, the first cell of a
    /// table the card builds is overwritten: the zerocheck's `eq` weight, or the
    /// reduce's batched column. The proof must then differ from today's at CPU
    /// — the first table argued — and must not verify, each fault at its own
    /// check. Needs a device, and says so rather than passing without one.
    #[test]
    fn a_wrong_table_on_the_card_changes_the_argument_and_fails_it() {
        use multilinear::gpu::TableFault;

        let _overrides = ARGUE_OVERRIDES.lock().unwrap_or_else(|e| e.into_inner());
        let airs = airs();
        let columns = tall_columns();
        if !a_device() {
            eprintln!("argue device tables: SKIPPED, no device to corrupt a table on");
            return;
        }
        let today = prove_tall(&airs, &columns, false, false);
        for (fault, rejected) in [
            (TableFault::Eq, MlError::BatchMismatch),
            (TableFault::Batched, MlError::ShiftedReadMismatch),
        ] {
            multilinear::gpu::force_table_fault(Some(fault));
            let faulted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                prove_tall(&airs, &columns, false, true)
            }));
            multilinear::gpu::force_table_fault(None);
            let faulted = faulted.unwrap_or_else(|_| panic!("{fault:?}: the faulted arm proves"));
            assert_eq!(
                faulted.tables_on_card, 12,
                "{fault:?}: the fault is on the card's path"
            );
            assert_eq!(
                first_table_that_differs(&today.proof, &faulted.proof),
                Some(0),
                "{fault:?}: the identity must name CPU, the first table argued"
            );
            assert_eq!(
                verify_tall(&faulted.committed, &faulted.proof),
                Err(rejected),
                "{fault:?}: the proof over a corrupted table must not verify"
            );
        }
    }

    /// [`config`] with its stack cap at `cap`.
    fn config_at(cap: usize) -> ChainConfig {
        ChainConfig {
            format: multilinear::whir_chain::ChainFormat {
                stack: StackVars::new(cap).expect("a legal cap"),
                ..multilinear::whir_chain::ChainFormat::DEFAULT
            },
            ..config()
        }
    }

    /// The three tables' shapes, as `commit_grouped` reads them.
    fn three_shapes() -> Vec<(usize, usize)> {
        [cpu_columns(), add_columns(), mul_columns()]
            .iter()
            .map(|columns| (columns.len(), columns[0].len().trailing_zeros() as usize))
            .collect()
    }

    /// Proves the three tables stacked under `prove_cap`, then verifies under
    /// `verify_cap` with every layout and domain REBUILT from the shapes and the
    /// verifier's own config — as the production verifier does
    /// (`prover::multilinear_prove::stacks`), never taken from the proof or
    /// from the prover's commitment. Returns the proof's root count.
    fn argue_across(prove_cap: usize, verify_cap: usize) -> Result<usize, MlError> {
        let (cpu_air, add_air, mul_air) = airs();
        let prover_config = config_at(prove_cap);
        let committed = CommittedTables::<_, _, KeccakWhir>::commit(
            vec![
                table(&cpu_air, &cpu_columns())?,
                table(&add_air, &add_columns())?,
                table(&mul_air, &mul_columns())?,
            ],
            &prover_config,
        )?;
        let mut prover = DefaultTranscript::<Ext>::new(b"multilinear-table");
        let proof = multi_prove(&committed, &prover_config, &mut prover, None)?;

        let verifier_config = config_at(verify_cap);
        let layouts = global_layouts(
            &three_shapes(),
            committed.sizes(),
            verifier_config.format.stack,
        )?;
        let domains = layouts
            .iter()
            .map(|layout| Domain::<Fp>::new(layout.n_stack() + verifier_config.log_blowup))
            .collect::<Result<Vec<_>, _>>()?;
        let statements: Vec<TableStatement<'_, Fp, Ext>> =
            committed.tables().iter().map(|t| t.statement()).collect();
        let mut verifier = DefaultTranscript::<Ext>::new(b"multilinear-table");
        multi_verify::<_, _, _, KeccakWhir>(
            &proof,
            &statements,
            &layouts,
            &domains,
            committed.sizes(),
            &ExtE::zero(),
            &verifier_config,
            &mut verifier,
            None,
        )?;
        Ok(proof.roots.len())
    }

    /// ★ S2: THE STACK IS A FORMAT CONSTANT OF BOTH SIDES. The three tables hold
    /// 72 cells: one polynomial of 7 variables under a cap of 7, three of 5 under
    /// a cap of 5. Each proof verifies under its own cap — the honest control,
    /// first — and is REFUSED under the other, in both directions, as an error:
    /// a panic fails this test. The code path of 25 against 27, at a size a
    /// laptop proves.
    #[test]
    fn a_proof_stacked_under_one_cap_is_refused_under_another() {
        let polys = |cap: usize| {
            global_layout(&three_shapes(), StackVars::new(cap).unwrap())
                .unwrap()
                .num_polys()
        };
        assert_eq!(
            (polys(7), polys(5)),
            (1, 3),
            "the caps must stack differently"
        );
        assert_eq!(argue_across(7, 7).unwrap(), 1, "the honest control at 7");
        assert_eq!(argue_across(5, 5).unwrap(), 3, "the honest control at 5");
        assert!(
            argue_across(7, 5).is_err(),
            "a proof stacked under 7 verified under 5"
        );
        assert!(
            argue_across(5, 7).is_err(),
            "a proof stacked under 5 verified under 7"
        );
    }

    /// `stack_height` is `global_layout`'s height, and no group of a shape list
    /// stacks taller than the whole list: what lets the query count charge all
    /// of a proof's shapes and never undercharge one of its groups.
    #[test]
    fn the_stack_height_is_the_layouts_and_bounds_every_group() {
        let shapes = [(5, 3), (4, 2), (4, 2), (0, 9), (16, 4), (2, 6)];
        for cap in [1, 4, 5, 6, 7, 8, 9, 25, StackVars::WIDEST] {
            let cap = StackVars::new(cap).unwrap();
            let all = stack_height(&shapes, cap);
            assert_eq!(
                global_layout(&shapes, cap).unwrap().n_stack(),
                all,
                "{cap:?}"
            );
            for start in 0..shapes.len() {
                for end in start + 1..=shapes.len() {
                    assert!(
                        stack_height(&shapes[start..end], cap) <= all,
                        "{cap:?}: shapes[{start}..{end}] stack taller than all of them"
                    );
                }
            }
        }
        // The empty column does not count toward the tallest.
        assert_eq!(stack_height(&[(0, 9), (1, 2)], StackVars::LEGACY), 2);
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
