//! ★ THE BATCHED ARGUE (`ArgueFormat::Batched`, D-BATCH §2): one argument for
//! every table of a proof, where [`multi_prove`](super::multi_prove) runs one
//! per table.
//!
//! The roots and `(z, α, β)` are today's. Then, in the transcript's order:
//!
//! - **G**, per bin of [`argue_plan`]: the bin's bus outputs absorbed, then one
//!   LogUp-GKR ladder over its trees in lockstep
//!   ([`multilinear::gkr_lockstep`]). Each tree leaves with its input claim
//!   `(p̃_T, q̃_T)` at a point whose last `n_T` coordinates are the row point
//!   `ρ_T`, as today. The bus balance is checked once, over every table.
//! - **C**: a zerocheck point `ξ` of `n_max` coordinates, the claims `p̃_T, q̃_T`
//!   absorbed per table, one `λ`, and ONE front-loaded sumcheck
//!   ([`multilinear::front_loaded`]) of `Σ_t λ^{3t}·f_t`, with today's per-table
//!   rule `f_t = eq(ξ_{<n_t}, ·)·C_t + λ·eq(ρ_t, ·)·N_t + λ²·eq(ρ_t, ·)·D_t` —
//!   today's batch with `γ = λ`. Then every table's committed factor values at
//!   its prefix `r_{<n_t}`, absorbed per table.
//! - **Rd**: today's claim reduction, at `r_{<n_t}`, for a table with a shifted
//!   read only. A table whose committed factors are all unshifted reads its
//!   columns directly: its factor values ARE its column claims.
//! - **O**: the openings, the same code as the per-table format's.
//!
//! The plan — the bins and their order — is a function of the statement's
//! shapes and the format's bin cap, and nothing else, so the prover, the host
//! verifier and the in-guest emitter derive it from one function.
//!
//! This is the HOST reference: every tree and every round is built and run
//! here. A device prover must produce these bytes.

use super::*;

use multilinear::{
    front_loaded::{self, PaddingFault},
    gkr::{FractionLayer, GkrClaim},
    gkr_lockstep::{self, LockstepProof},
    logup::Interaction,
    sumcheck::{self, SumcheckProof},
};

/// One table's shape as the batched argue sees it: all it needs to plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArgueShape {
    /// `n_T`, the table's height in variables.
    pub num_vars: usize,
    /// `k_T`: the rows plus the bits that index its interactions.
    pub input_vars: usize,
    /// `D_T = max(d_C + 1, 2)`: the degree of its batched rule.
    pub degree: usize,
    /// Whether any committed factor reads a shifted row, which is what keeps
    /// its claim reduction.
    pub shifted: bool,
}

impl ArgueShape {
    /// The shape of a table's statement — structure only, so both sides get
    /// the same one.
    pub fn of<F, E>(statement: &TableStatement<'_, F, E>) -> Self
    where
        F: IsFFTField + IsPrimeField + IsSubFieldOf<E>,
        E: IsField + 'static,
    {
        Self {
            num_vars: statement.num_vars,
            input_vars: logup::input_layer_vars(statement.interactions.len(), statement.num_vars),
            degree: (statement.shape.degree() + 1).max(2),
            shifted: statement
                .kinds
                .iter()
                .filter_map(FactorKind::source)
                .any(|source| source.offset != 0),
        }
    }
}

/// The public schedule of a batched argue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArguePlan {
    pub shapes: Vec<ArgueShape>,
    /// The bins, in the order their ladders run; each in table order.
    pub bins: Vec<Vec<usize>>,
    /// `n_max`: the constraint sumcheck's rounds.
    pub num_vars: usize,
    /// `D_max`: its round degree.
    pub degree: usize,
}

/// ★★ THE PLAN, one function for every side (the `absorb_roots` lesson: two
/// spellings of a public ordering drift silently).
///
/// Bins: tables sorted by `(k_T desc, index asc)`, first-fit-decreasing under
/// `Σ 2^{k_T} ≤ 2^{bin_log_cells}`; a table with `k_T ≥ bin_log_cells` sits
/// alone. Bins run in the order they open, each in table order.
pub fn argue_plan(shapes: &[ArgueShape], bin_log_cells: u8) -> ArguePlan {
    let cap = 1u128 << bin_log_cells.min(127);
    let mut order: Vec<usize> = (0..shapes.len()).collect();
    order.sort_by(|&a, &b| {
        shapes[b]
            .input_vars
            .cmp(&shapes[a].input_vars)
            .then(a.cmp(&b))
    });
    // (tables, cells used, open to more)
    let mut bins: Vec<(Vec<usize>, u128, bool)> = Vec::new();
    for t in order {
        let cells = 1u128 << shapes[t].input_vars.min(127);
        if cells >= cap {
            bins.push((vec![t], cells, false));
            continue;
        }
        match bins
            .iter_mut()
            .find(|(_, used, open)| *open && used + cells <= cap)
        {
            Some((tables, used, _)) => {
                tables.push(t);
                *used += cells;
            }
            None => bins.push((vec![t], cells, true)),
        }
    }
    ArguePlan {
        shapes: shapes.to_vec(),
        bins: bins
            .into_iter()
            .map(|(mut tables, _, _)| {
                tables.sort_unstable();
                tables
            })
            .collect(),
        num_vars: shapes.iter().map(|s| s.num_vars).max().unwrap_or(0),
        degree: shapes.iter().map(|s| s.degree).max().unwrap_or(2),
    }
}

/// The batched argue's messages.
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
pub struct BatchedArgue<E: IsField> {
    /// Every table's bus output fraction, in table order.
    pub bus_outputs: Vec<(FieldElement<E>, FieldElement<E>)>,
    /// One ladder per bin, in plan order.
    pub gkr: Vec<LockstepProof<E>>,
    /// The front-loaded constraint sumcheck: `n_max` rounds of `D_max` values.
    pub constraint: SumcheckProof<E>,
    /// Per table, its committed factors' values at `r_{<n_T}`.
    pub factor_values: Vec<Vec<FieldElement<E>>>,
    /// Per table, its claim reduction — present exactly for a table with a
    /// shifted read.
    pub reduces: Vec<Option<claim_reduce::ReduceProof<E>>>,
}

/// A proof under the batched argue: [`MultiProof`] with the argue batched.
///
/// ⚠ A TYPE OF ITS OWN, not a variant inside [`MultiProof`], so today's proofs
/// keep their bytes: the byte gates serialize `MultiProof` through rkyv, which
/// writes a discriminant for an enum. The openings are the same code and the
/// same fields.
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
pub struct BatchedMultiProof<F: IsField, E: IsField> {
    pub roots: Vec<Commitment>,
    pub argue: BatchedArgue<E>,
    pub columns: Vec<StackedProof<F, E>>,
    pub preprocessed: Option<StackedProof<F, E>>,
}

/// ⛔ Faults a NEGATIVE test arms in the prover, each producing a proof the
/// verifier must refuse. [`multi_prove_batched`] arms none.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ProverFaults {
    /// This table's zerocheck weight takes the SUFFIX of `ξ`.
    pub suffix_xi: Option<usize>,
    /// This table's cross-table weight is `λ^{3t+1}`, not `λ^{3t}`.
    pub wrong_weight: Option<usize>,
    /// This table's `p̃` is absorbed off by one.
    pub tamper_claim: Option<usize>,
    /// This table is padded the `2^Δ` way, not by `Π X_j`.
    pub scaled_padding: Option<usize>,
    /// The prover plans its bins under this cap instead of the config's.
    pub bin_log_cells: Option<u8>,
}

/// Which of the batched verifier's checks run. Every one, except in the
/// mutation tests that show each is load-bearing.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct VerifierChecks {
    /// Each ladder step's residual.
    pub residual: bool,
    /// The constraint sumcheck's final equation.
    pub constraint: bool,
    /// `Σ p_T/q_T == expected`.
    pub balance: bool,
    /// A reduction present exactly for a table with a shifted read.
    pub reduce_shape: bool,
    /// The preprocessed columns at `r_{<n_T}`.
    pub preprocessed: bool,
}

impl VerifierChecks {
    pub const ALL: Self = Self {
        residual: true,
        constraint: true,
        balance: true,
        reduce_shape: true,
        preprocessed: true,
    };
}

/// Every level of a table's fraction tree, output first, built here.
fn host_tree<F, E>(
    interactions: &[Interaction<E>],
    trace: &TraceData<F, E>,
) -> Result<Vec<FractionLayer<E>>, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: Send + Sync,
{
    let factors = trace.factors()?;
    let mut layers = vec![logup::input_layer(interactions, &factors)?];
    while layers.last().expect("non-empty").num_vars() > 0 {
        let next = layers.last().expect("non-empty").fold()?;
        layers.push(next);
    }
    layers.reverse();
    Ok(layers)
}

/// A table's column claims when every committed factor reads its column
/// unshifted: each column's claim is its factor's value. A column read twice
/// must be claimed the same both times, and every column must be read.
fn direct_column_values<E: IsField>(
    kinds: &[FactorKind],
    factor_values: &[FieldElement<E>],
    num_columns: usize,
) -> Result<Vec<FieldElement<E>>, MlError> {
    let mut columns: Vec<Option<FieldElement<E>>> = vec![None; num_columns];
    let mut values = factor_values.iter();
    for source in kinds.iter().filter_map(FactorKind::source) {
        let value = values.next().ok_or(MlError::ArgueShapeMismatch {
            part: "factor values",
        })?;
        if source.offset != 0 {
            return Err(MlError::ArgueShapeMismatch { part: "reduction" });
        }
        let slot = columns
            .get_mut(source.column)
            .ok_or(MlError::UnknownPolynomial {
                index: source.column,
                len: num_columns,
            })?;
        match slot {
            Some(claimed) if claimed != value => return Err(MlError::EvaluationMismatch),
            Some(_) => {}
            None => *slot = Some(value.clone()),
        }
    }
    if values.next().is_some() {
        return Err(MlError::ArgueShapeMismatch {
            part: "factor values",
        });
    }
    columns
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(MlError::ArgueShapeMismatch {
            part: "column claims",
        })
}

/// A table's rules and its bus's row point.
type TableRules<'a, E> = (Vec<Rule<'a, E>>, Vec<FieldElement<E>>);

/// A table's three rules, today's: the zerocheck at `weight_r`, the bus's two
/// at `weight_z`, the bus claimed at `claim_point`.
fn table_rules<'a, F, E>(
    statement: &TableStatement<'_, F, E>,
    interactions: &'a [Interaction<E>],
    claim_point: &[FieldElement<E>],
    beta: &FieldElement<E>,
) -> Result<TableRules<'a, E>, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E>,
    E: IsField + 'static,
{
    let (weight_r, weight_z) = weight_slots(statement.kinds.len());
    let bus = logup::claim_statements(interactions, claim_point, statement.num_vars, weight_z)?;
    let shape = statement.shape;
    let betas = multilinear_air::beta_powers(beta, shape.num_roots());
    let zerocheck = Rule::compiled(shape.degree() + 1, shape.program(&betas, weight_r)?);
    Ok((
        vec![zerocheck, bus.numerator, bus.denominator],
        bus.row_point,
    ))
}

/// `λ^{3t}` per table — table `t`'s share of the batch.
fn table_weights<E: IsField>(lambda: &FieldElement<E>, tables: usize) -> Vec<FieldElement<E>> {
    let cube = lambda * lambda * lambda;
    multilinear::challenge_powers(&cube, tables)
}

/// Proves every table under the batched argue, against one commitment.
pub fn multi_prove_batched<F, E, T, H>(
    committed: &CommittedTables<'_, F, E, H>,
    config: &ChainConfig,
    transcript: &mut T,
    prepared: Option<Prepared<'_, F, H>>,
) -> Result<BatchedMultiProof<F, E>, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>,
{
    prove_batched_on(
        committed,
        config,
        transcript,
        prepared,
        ProverFaults::default(),
        Where::Device,
    )
}

/// Where the batched prover runs its trees and rounds.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Where {
    /// Every tree and every round here: the host reference.
    Host,
    /// On the card where it takes them — the trees where the factors are, the
    /// ladder's layers stepped there, the fused zerocheck rounds — and here
    /// where it declines. The same bytes as [`Where::Host`].
    Device,
}

/// [`multi_prove_batched`] as the host reference, with faults a negative test
/// arms.
#[doc(hidden)]
pub fn prove_batched_with<F, E, T, H>(
    committed: &CommittedTables<'_, F, E, H>,
    config: &ChainConfig,
    transcript: &mut T,
    prepared: Option<Prepared<'_, F, H>>,
    faults: ProverFaults,
) -> Result<BatchedMultiProof<F, E>, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>,
{
    prove_batched_on(committed, config, transcript, prepared, faults, Where::Host)
}

/// A bin's tree, where it was built.
enum BinTree<E: IsField> {
    Host(Vec<FractionLayer<E>>),
    Built(Box<multilinear::gkr::FractionTree<E>>),
}

/// [`multi_prove_batched`] on the host or the card, with faults.
#[doc(hidden)]
pub fn prove_batched_on<F, E, T, H>(
    committed: &CommittedTables<'_, F, E, H>,
    config: &ChainConfig,
    transcript: &mut T,
    prepared: Option<Prepared<'_, F, H>>,
    faults: ProverFaults,
    at: Where,
) -> Result<BatchedMultiProof<F, E>, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>,
{
    let _split = multilinear::whir_split::begin_prove();
    let ArgueFormat::Batched { bin_log_cells } = config.format.argue else {
        return Err(MlError::ArgueFormatMismatch);
    };
    check_settled_prefixes(committed, prepared.as_ref())?;

    let prepared_roots: Vec<Commitment> = prepared
        .as_ref()
        .map(|p| p.commitment.roots())
        .unwrap_or_default();
    let __sp_challenge = multilinear::whir_split::mark();
    let (z, alpha, beta) =
        absorb_roots_and_challenge::<E, T>(transcript, committed.roots(), &prepared_roots);
    multilinear::whir_split::add(&multilinear::whir_split::CHALLENGE, __sp_challenge);

    let tables = committed.tables();
    // The base split's `argue` and `RESERVED HW`'s, as `multi_prove` keeps
    // them — the whole batched argue as one window (it has no per-table loop).
    let __rw_argue = multilinear::whir_split::open_reserved();
    let __sp_argue = multilinear::whir_split::mark();
    let argued = prove_argue::<F, E, T>(
        tables,
        &z,
        &alpha,
        &beta,
        bin_log_cells,
        transcript,
        faults,
        at,
    );
    let __sp_secs = multilinear::whir_split::add(&multilinear::whir_split::ARGUE, __sp_argue);
    multilinear::whir_split::note_table(0, __sp_secs);
    multilinear::whir_split::close_reserved(&multilinear::whir_split::RESERVED_ARGUE, __rw_argue);
    let (argue, reduced) = argued?;
    let mut points: Vec<Vec<FieldElement<E>>> = Vec::new();
    let mut values: Vec<FieldElement<E>> = Vec::new();
    let mut table_starts = Vec::with_capacity(tables.len());
    for (table, claim) in tables.iter().zip(reduced) {
        table_starts.push(points.len());
        for _ in 0..table.num_committed_columns() {
            points.push(claim.point.clone());
        }
        values.extend(claim.column_values);
    }

    // ── O: the openings, as the per-table format's ────────────────────────
    let (columns, preprocessed) = prove_openings(
        committed,
        config,
        transcript,
        prepared,
        &points,
        &values,
        &table_starts,
    )?;

    Ok(BatchedMultiProof {
        roots: committed.roots().to_vec(),
        argue,
        columns,
        preprocessed,
    })
}

/// ★ The batched argue itself, from the shared challenges to every table's
/// column claims: phases G, C and Rd of the module docs. The roots block
/// before it and the openings after it are the caller's.
///
/// `pub` for the in-guest emitter's gate, which argues a few tables with no
/// commitment at all, as the per-table leg's gate calls `prove`.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn prove_argue<F, E, T>(
    tables: &[CommittedTable<'_, F, E>],
    z: &FieldElement<E>,
    alpha: &FieldElement<E>,
    beta: &FieldElement<E>,
    bin_log_cells: u8,
    transcript: &mut T,
    faults: ProverFaults,
    at: Where,
) -> Result<(BatchedArgue<E>, Vec<claim_reduce::ReducedClaim<E>>), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>,
{
    let beta = beta.clone();
    let statements: Vec<TableStatement<'_, F, E>> = tables.iter().map(|t| t.statement()).collect();
    let shapes: Vec<ArgueShape> = statements.iter().map(ArgueShape::of).collect();
    let plan = argue_plan(&shapes, faults.bin_log_cells.unwrap_or(bin_log_cells));
    let interactions: Vec<Vec<Interaction<E>>> = tables
        .iter()
        .map(|table| {
            multilinear_logup::interactions(
                table.layout.interactions,
                table.slot_of().len(),
                z,
                alpha,
                |col| slot(table.slot_of(), col),
            )
        })
        .collect::<Result<_, _>>()?;

    let mut split = ArgueSplit::default();
    // ── G: a ladder per bin, its outputs absorbed just before it ──────────
    let mut bus_outputs: Vec<Option<(FieldElement<E>, FieldElement<E>)>> = vec![None; tables.len()];
    let mut claims: Vec<Option<GkrClaim<E>>> = vec![None; tables.len()];
    let mut gkr = Vec::with_capacity(plan.bins.len());
    for bin in &plan.bins {
        let trees: Vec<BinTree<E>> = bin
            .iter()
            .map(|&t| -> Result<BinTree<E>, MlError> {
                let (trace, interactions) = (&tables[t].trace, &interactions[t]);
                if at == Where::Host {
                    return Ok(BinTree::Host(host_tree(interactions, trace)?));
                }
                // Where today's argue builds it: written from the base columns
                // with no lift (`LAMBDA_VM_ARGUE_NO_LIFT`), else from the
                // lifted factors on the card, else here — and a big host tree
                // still folds on a device. The same tree every way.
                let started = std::time::Instant::now();
                let on_card = (multilinear::gpu::argue_no_lift()
                    && trace.device_factors().is_none())
                .then(|| logup::resident_tree_from_columns(interactions, trace))
                .flatten()
                .or_else(|| {
                    let resident = trace.reside_from_columns();
                    let from_columns = (multilinear::gpu::argue_gkr_input() && resident.is_some())
                        .then(|| logup::resident_tree_from_columns(interactions, trace))
                        .flatten();
                    from_columns.or_else(|| {
                        resident.and_then(|resident| logup::resident_tree(interactions, resident))
                    })
                });
                let tree = match on_card {
                    Some(tree) => tree,
                    None => {
                        let factors = trace.factors()?;
                        multilinear::gkr::FractionTree::build(logup::input_layer(
                            interactions,
                            &factors,
                        )?)?
                    }
                };
                split.trees += started.elapsed().as_secs_f64();
                Ok(BinTree::Built(Box::new(tree)))
            })
            .collect::<Result<_, _>>()?;
        let hosts: Vec<Option<gkr_lockstep::HostLayers<'_, E>>> = trees
            .iter()
            .map(|tree| match tree {
                BinTree::Host(layers) => Some(gkr_lockstep::HostLayers(layers)),
                BinTree::Built(_) => None,
            })
            .collect();
        let refs: Vec<&dyn gkr_lockstep::LadderTree<E>> = trees
            .iter()
            .zip(&hosts)
            .map(|(tree, host)| match (tree, host) {
                (_, Some(host)) => host as &dyn gkr_lockstep::LadderTree<E>,
                (BinTree::Built(tree), None) => &**tree as &dyn gkr_lockstep::LadderTree<E>,
                (BinTree::Host(_), None) => unreachable!("a host tree has its layers"),
            })
            .collect();
        for (&t, tree) in bin.iter().zip(&refs) {
            let output = tree.output();
            transcript.append_field_element(&output.0);
            transcript.append_field_element(&output.1);
            bus_outputs[t] = Some(output);
        }
        let started = std::time::Instant::now();
        let (ladder, inputs) = gkr_lockstep::prove_trees(&refs, transcript)?;
        split.ladders += started.elapsed().as_secs_f64();
        for (&t, claim) in bin.iter().zip(inputs) {
            claims[t] = Some(claim);
        }
        gkr.push(ladder);
    }
    let bus_outputs: Vec<(FieldElement<E>, FieldElement<E>)> = bus_outputs
        .into_iter()
        .collect::<Option<_>>()
        .ok_or(MlError::ArgueShapeMismatch { part: "bins" })?;
    let claims: Vec<GkrClaim<E>> = claims
        .into_iter()
        .collect::<Option<_>>()
        .ok_or(MlError::ArgueShapeMismatch { part: "bins" })?;

    // ── C: one front-loaded sumcheck over every table ────────────────────
    let xi: Vec<FieldElement<E>> = (0..plan.num_vars)
        .map(|_| transcript.sample_field_element())
        .collect();
    for (t, claim) in claims.iter().enumerate() {
        let mut p = claim.p.clone();
        if faults.tamper_claim == Some(t) {
            p += FieldElement::<E>::one();
        }
        transcript.append_field_element(&p);
        transcript.append_field_element(&claim.q);
    }
    let lambda: FieldElement<E> = transcript.sample_field_element();
    let mut weights = table_weights(&lambda, tables.len());
    if let Some(t) = faults.wrong_weight {
        weights[t] = &weights[t] * &lambda;
    }
    let lambdas = vec![FieldElement::one(), lambda.clone(), &lambda * &lambda];
    let lambda2 = &lambda * &lambda;
    let started = std::time::Instant::now();
    // What each table's fused rounds read, held for as long as the rounds do:
    // the base columns on the card with no lift (`LAMBDA_VM_ARGUE_NO_LIFT`), the
    // lifted factors for a table they cannot read that way (lifted here, as the
    // per-table argue lifts it), or nothing, on the host.
    let mut constraints: Vec<Option<multilinear::fused::Constraints<F>>> =
        Vec::with_capacity(tables.len());
    let mut columns: Vec<Option<multilinear::gpu::ColumnFactors>> =
        Vec::with_capacity(tables.len());
    for table in tables {
        let shape = table.shape();
        let made = (at == Where::Device
            && multilinear::gpu_fused::argue_fused()
            && multilinear::gpu_fused::fused_takes_width(table.num_committed_columns()))
        .then(|| {
            multilinear::fused::Constraints::<F>::from_extension(
                &shape.steps_as_ops(),
                shape.root_steps(),
                shape.selector_of_root(),
                shape.degree(),
            )
            .ok()
        })
        .flatten();
        let column = (at == Where::Device
            && multilinear::gpu::argue_no_lift()
            && table.trace.device_factors().is_none())
        .then(|| made.as_ref().and_then(|_| table.trace.column_factors()))
        .flatten();
        if at == Where::Device && made.is_some() && column.is_none() {
            let _ = table.trace.reside_from_columns();
        }
        constraints.push(made);
        columns.push(column);
    }
    let device: Vec<Option<Arc<multilinear::gpu::DeviceFactors>>> = tables
        .iter()
        .map(|table| match at {
            Where::Host => None,
            Where::Device => table.trace.device_factors(),
        })
        .collect();
    let mut polys = Vec::with_capacity(tables.len());
    for (t, table) in tables.iter().enumerate() {
        let n = table.num_vars();
        let (rules, row_point) =
            table_rules(&statements[t], &interactions[t], &claims[t].point, &beta)?;
        let xi_t = if faults.suffix_xi == Some(t) {
            &xi[plan.num_vars - n..]
        } else {
            &xi[..n]
        };
        let betas = multilinear_air::beta_powers(&beta, table.shape().num_roots());
        let fused = constraints[t]
            .as_ref()
            .map(|constraints| multilinear::gpu_fused::FusedInput {
                constraints,
                betas: &betas,
                interactions: &interactions[t],
                claim_point: &claims[t].point,
                r: xi_t,
                columns: columns[t].as_ref(),
            });
        let claim = &lambda * &claims[t].p + &lambda2 * &claims[t].q;
        polys.push(front_loaded::TableRounds::new(
            rules,
            lambdas.clone(),
            &claim,
            [xi_t, &row_point],
            n,
            device[t].as_deref(),
            columns[t].as_ref(),
            fused,
            || table.trace.factors(),
        )?);
    }
    split.zc_setup += started.elapsed().as_secs_f64();
    let started = std::time::Instant::now();
    let (constraint, point) = front_loaded::prove_with(
        &mut polys,
        &weights,
        plan.degree,
        transcript,
        PaddingFault {
            scaled: faults.scaled_padding,
        },
    )?;
    split.zc_rounds += started.elapsed().as_secs_f64();
    let mut factor_values = Vec::with_capacity(tables.len());
    for (table, poly) in tables.iter().zip(&polys) {
        let values = table
            .kinds()
            .iter()
            .zip(poly.polys())
            .filter(|(kind, _)| kind.source().is_some())
            .map(|(_, factor)| factor.as_constant().cloned())
            .collect::<Option<Vec<_>>>()
            .ok_or(MlError::NoVariablesLeft)?;
        factor_values.push(values);
    }
    drop(polys);
    drop(device);
    drop(columns);
    for table in tables {
        table.trace.release_device();
    }
    for values in &factor_values {
        for value in values {
            transcript.append_field_element(value);
        }
    }

    // ── Rd: a shifted table's reduction; the others read their columns ────
    let started = std::time::Instant::now();
    let mut reduces = Vec::with_capacity(tables.len());
    let mut reduced = Vec::with_capacity(tables.len());
    for (t, table) in tables.iter().enumerate() {
        let at = &point[..table.num_vars()];
        let (column_point, column_values, reduce) = if plan.shapes[t].shifted {
            let sources: Vec<claim_reduce::FactorSource> = table
                .kinds()
                .iter()
                .filter_map(FactorKind::source)
                .collect();
            let (reduce, reduced_point) = claim_reduce::prove::<F, E, T>(
                table.columns(),
                &sources,
                &factor_values[t],
                at,
                table.trace.resident(),
                transcript,
            )?;
            (reduced_point, reduce.column_values.clone(), Some(reduce))
        } else {
            let column_values = direct_column_values(
                table.kinds(),
                &factor_values[t],
                table.num_committed_columns(),
            )?;
            (at.to_vec(), column_values, None)
        };
        reduced.push(claim_reduce::ReducedClaim {
            point: column_point,
            column_values,
        });
        reduces.push(reduce);
    }
    split.reduce += started.elapsed().as_secs_f64();
    if multilinear::whir_split::enabled() {
        eprintln!(
            "BATCHED ARGUE SPLIT: tables {} · bins {} · trees {:.3}s · ladders {:.3}s · zc setup \
             {:.3}s · zc rounds {:.3}s · reduce {:.3}s",
            tables.len(),
            plan.bins.len(),
            split.trees,
            split.ladders,
            split.zc_setup,
            split.zc_rounds,
            split.reduce
        );
        // The inputs of D-BATCH §3.2's error terms, measured on this argue.
        eprintln!(
            "BATCHED ARGUE BITS INPUTS: tables {} · trees a bin max {} · log2 cells {:.2} · \
             len_max {} · N_C max {} · D_max {} · n_max {}",
            tables.len(),
            plan.bins.iter().map(Vec::len).max().unwrap_or(0),
            plan.shapes
                .iter()
                .map(|shape| (shape.input_vars as f64).exp2())
                .sum::<f64>()
                .log2(),
            statements
                .iter()
                .flat_map(|s| s.interactions.iter().map(BusInteraction::num_bus_elements))
                .max()
                .unwrap_or(0),
            statements
                .iter()
                .map(|s| s.shape.num_roots())
                .max()
                .unwrap_or(0),
            plan.degree,
            plan.num_vars
        );
    }
    Ok((
        BatchedArgue {
            bus_outputs,
            gkr,
            constraint,
            factor_values,
            reduces,
        },
        reduced,
    ))
}

/// Where a batched argue's time goes, printed under `LAMBDA_VM_BASE_SPLIT=1`.
#[derive(Default)]
struct ArgueSplit {
    trees: f64,
    ladders: f64,
    zc_setup: f64,
    zc_rounds: f64,
    reduce: f64,
}

/// Verifies a batched proof: every table, the bus balance across them, and
/// every column against the one commitment. See [`multi_verify`].
#[allow(clippy::too_many_arguments)]
pub fn multi_verify_batched<F, E, T, H>(
    proof: &BatchedMultiProof<F, E>,
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
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>,
{
    multi_verify_batched_settled::<F, E, T, H>(
        proof, statements, layouts, domains, sizes, expected, config, transcript, prepared, false,
    )
}

/// [`multi_verify_batched`] with `exclude_settled`, as
/// [`multi_verify_settled`](super::multi_verify_settled).
#[allow(clippy::too_many_arguments)]
pub fn multi_verify_batched_settled<F, E, T, H>(
    proof: &BatchedMultiProof<F, E>,
    statements: &[TableStatement<'_, F, E>],
    layouts: &[StackedLayout],
    domains: &[Domain<F>],
    sizes: &[usize],
    expected: &FieldElement<E>,
    config: &ChainConfig,
    transcript: &mut T,
    prepared: Option<PreparedCheck<'_, F>>,
    exclude_settled: bool,
) -> Result<(), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>,
{
    verify_batched_with::<F, E, T, H>(
        proof,
        statements,
        layouts,
        domains,
        sizes,
        expected,
        config,
        transcript,
        prepared,
        exclude_settled,
        VerifierChecks::ALL,
    )
}

/// [`multi_verify_batched_settled`], with each new check switchable for the
/// mutation tests.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn verify_batched_with<F, E, T, H>(
    proof: &BatchedMultiProof<F, E>,
    statements: &[TableStatement<'_, F, E>],
    layouts: &[StackedLayout],
    domains: &[Domain<F>],
    sizes: &[usize],
    expected: &FieldElement<E>,
    config: &ChainConfig,
    transcript: &mut T,
    prepared: Option<PreparedCheck<'_, F>>,
    exclude_settled: bool,
    checks: VerifierChecks,
) -> Result<(), MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>
        + crypto::fiat_shamir::transcript_hash::HasTranscriptHash<Hash = <H as WhirHash>::Transcript>,
{
    // The variant and the bin cap are the verifier's config's, never the
    // proof's.
    let ArgueFormat::Batched { bin_log_cells } = config.format.argue else {
        return Err(MlError::ArgueFormatMismatch);
    };
    let argue = &proof.argue;
    let n = statements.len();
    if argue.bus_outputs.len() != n || argue.factor_values.len() != n || argue.reduces.len() != n {
        return Err(MlError::ArgueShapeMismatch { part: "tables" });
    }
    if layouts.len() != sizes.len()
        || domains.len() != sizes.len()
        || proof.columns.len() != sizes.len()
        || sizes.iter().sum::<usize>() != n
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

    let reduced_claims = verify_argue::<F, E, T>(
        argue,
        statements,
        bin_log_cells,
        &z,
        &alpha,
        &beta,
        transcript,
        checks,
    )?;
    let mut balance = FieldElement::<E>::zero();
    for output in &argue.bus_outputs {
        balance += contribution(output).ok_or(MlError::BusImbalance)?;
    }
    if checks.balance && balance != *expected {
        return Err(MlError::BusImbalance);
    }

    let mut points: Vec<Vec<FieldElement<E>>> = Vec::new();
    let mut values: Vec<FieldElement<E>> = Vec::new();
    let mut table_starts = Vec::with_capacity(n);
    let mut settled_counts = Vec::with_capacity(n);
    for (t, (statement, reduced)) in statements.iter().zip(reduced_claims).enumerate() {
        table_starts.push(points.len());
        let settled = settled_count(prepared.as_ref(), t, statement)?;
        settled_counts.push(settled);
        if checks.preprocessed {
            check_preprocessed(*statement, &reduced, settled)?;
        }
        for _ in 0..statement.slot_of.len() {
            points.push(reduced.point.clone());
        }
        values.extend(reduced.column_values);
    }

    // ── O ─────────────────────────────────────────────────────────────────
    verify_openings::<F, E, T, H>(
        &proof.roots,
        &proof.columns,
        proof.preprocessed.as_ref(),
        statements,
        layouts,
        domains,
        sizes,
        config,
        transcript,
        prepared,
        exclude_settled,
        &points,
        &values,
        &table_starts,
        &settled_counts,
    )
}

/// ★ The batched argue's verifying half, from the shared challenges to every
/// table's column claims (phases G, C, Rd). The bus balance, the preprocessed
/// checks and the openings are the caller's.
///
/// `pub` for the in-guest emitter's gate, which checks a leg against it.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn verify_argue<F, E, T>(
    argue: &BatchedArgue<E>,
    statements: &[TableStatement<'_, F, E>],
    bin_log_cells: u8,
    z: &FieldElement<E>,
    alpha: &FieldElement<E>,
    beta: &FieldElement<E>,
    transcript: &mut T,
    checks: VerifierChecks,
) -> Result<Vec<claim_reduce::ReducedClaim<E>>, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>,
{
    let n = statements.len();
    if argue.bus_outputs.len() != n || argue.factor_values.len() != n || argue.reduces.len() != n {
        return Err(MlError::ArgueShapeMismatch { part: "tables" });
    }
    let beta = beta.clone();
    let shapes: Vec<ArgueShape> = statements.iter().map(ArgueShape::of).collect();
    let plan = argue_plan(&shapes, bin_log_cells);
    if argue.gkr.len() != plan.bins.len() {
        return Err(MlError::ArgueShapeMismatch { part: "bins" });
    }
    let interactions: Vec<Vec<Interaction<E>>> = statements
        .iter()
        .map(|statement| {
            multilinear_logup::interactions(
                statement.interactions,
                statement.slot_of.len(),
                z,
                alpha,
                |col| slot(statement.slot_of, col),
            )
        })
        .collect::<Result<_, _>>()?;

    // ── G ─────────────────────────────────────────────────────────────────
    let mut claims: Vec<Option<GkrClaim<E>>> = vec![None; n];
    for (bin, ladder) in plan.bins.iter().zip(&argue.gkr) {
        let mut outputs = Vec::with_capacity(bin.len());
        for &t in bin {
            let output = &argue.bus_outputs[t];
            transcript.append_field_element(&output.0);
            transcript.append_field_element(&output.1);
            outputs.push(output.clone());
        }
        let heights: Vec<usize> = bin.iter().map(|&t| plan.shapes[t].input_vars).collect();
        let inputs =
            gkr_lockstep::verify_with(ladder, &outputs, &heights, transcript, checks.residual)?;
        for (&t, claim) in bin.iter().zip(inputs) {
            claims[t] = Some(claim);
        }
    }
    let claims: Vec<GkrClaim<E>> = claims
        .into_iter()
        .collect::<Option<_>>()
        .ok_or(MlError::ArgueShapeMismatch { part: "bins" })?;

    // ── C ─────────────────────────────────────────────────────────────────
    let xi: Vec<FieldElement<E>> = (0..plan.num_vars)
        .map(|_| transcript.sample_field_element())
        .collect();
    for claim in &claims {
        transcript.append_field_element(&claim.p);
        transcript.append_field_element(&claim.q);
    }
    let lambda: FieldElement<E> = transcript.sample_field_element();
    let weights = table_weights(&lambda, n);
    let lambda2 = &lambda * &lambda;
    let claimed = claims
        .iter()
        .zip(&weights)
        .fold(FieldElement::<E>::zero(), |acc, (claim, w)| {
            acc + w * (&lambda * &claim.p + &lambda2 * &claim.q)
        });
    let sumcheck = sumcheck::verify(
        &argue.constraint,
        claimed,
        plan.num_vars,
        plan.degree,
        transcript,
    )?;
    let point = &sumcheck.point;
    let mut rebuilt = FieldElement::<E>::zero();
    for (t, statement) in statements.iter().enumerate() {
        let num_vars = statement.num_vars;
        let at = &point[..num_vars];
        let (rules, row_point) = table_rules(statement, &interactions[t], &claims[t].point, &beta)?;
        let mut public = statement.shape.public_values(at)?;
        public.push(eq_eval(&xi[..num_vars], at)?);
        public.push(eq_eval(&row_point, at)?);
        let values = constraint_argument::weave(statement.kinds, &argue.factor_values[t], &public)?;
        let value = rules[0].apply(&values)
            + &lambda * rules[1].apply(&values)
            + &lambda2 * rules[2].apply(&values);
        rebuilt += &weights[t] * value * front_loaded::padding_product(point, num_vars);
    }
    if checks.constraint && rebuilt != sumcheck.expected_evaluation {
        return Err(MlError::BatchMismatch);
    }
    for values in &argue.factor_values {
        for value in values {
            transcript.append_field_element(value);
        }
    }

    // ── Rd ────────────────────────────────────────────────────────────────
    let mut reduced_claims = Vec::with_capacity(n);
    for (t, statement) in statements.iter().enumerate() {
        let at = &point[..statement.num_vars];
        let factor_values = &argue.factor_values[t];
        let reduced = match (plan.shapes[t].shifted, &argue.reduces[t]) {
            (true, Some(reduce)) => {
                let sources: Vec<claim_reduce::FactorSource> = statement
                    .kinds
                    .iter()
                    .filter_map(FactorKind::source)
                    .collect();
                claim_reduce::verify(
                    reduce,
                    &sources,
                    factor_values,
                    at,
                    statement.slot_of.len(),
                    transcript,
                )?
            }
            (false, None) => claim_reduce::ReducedClaim {
                point: at.to_vec(),
                column_values: direct_column_values(
                    statement.kinds,
                    factor_values,
                    statement.slot_of.len(),
                )?,
            },
            (false, Some(_)) if !checks.reduce_shape => claim_reduce::ReducedClaim {
                point: at.to_vec(),
                column_values: direct_column_values(
                    statement.kinds,
                    factor_values,
                    statement.slot_of.len(),
                )?,
            },
            _ => return Err(MlError::ArgueShapeMismatch { part: "reduction" }),
        };
        reduced_claims.push(reduced);
    }
    Ok(reduced_claims)
}

#[cfg(test)]
mod tests;
