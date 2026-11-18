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
//! Under a [`GkrPack::Chunks`] format a
//! table whose padding costs more than the trees it would add is argued as
//! `c` trees of `2^b` slots — its interactions' chunks, in order, all in one
//! bin. They end the ladder at one point `P`, and after the bin's ladder one
//! challenge `s'` combines their claims into today's one-tree claim at
//! `s' ‖ P` ([`combine_chunk_claims`]), so C, Rd and O are unchanged. The
//! cuts are [`chunking`]'s, a function of the shapes and the format's price;
//! the in-guest emitter reads no packed format.
//!
//! This is the HOST reference: every tree and every round is built and run
//! here. A device prover must produce these bytes.

use super::*;

use multilinear::{
    front_loaded::{self, PaddingFault},
    gkr::{FractionLayer, GkrClaim},
    gkr_lockstep::{self, LockstepProof},
    logup::{Affine, Interaction},
    sumcheck::{self, SumcheckProof},
    whir_chain::{GkrLadder, GkrPack},
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
    /// `m_T`, its interactions: what [`chunking`] cuts.
    pub interactions: usize,
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
            interactions: statement.interactions.len(),
        }
    }
}

/// How a table's interactions are laid into trees: `count` trees of
/// `2^slot_vars` slots, chunk `j` holding interactions
/// `[j·2^slot_vars, (j + 1)·2^slot_vars)` in order, the last padded `0/1`.
/// `count == 1` is today's one tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunking {
    /// `b`: the bits that index a slot within a chunk.
    pub slot_vars: usize,
    /// `c`: the chunks.
    pub count: usize,
}

impl Chunking {
    /// Today's: one tree over every interaction.
    pub fn whole(shape: &ArgueShape) -> Self {
        Self {
            slot_vars: shape.input_vars - shape.num_vars,
            count: 1,
        }
    }

    /// `ℓ`: the bits that index a chunk, `log2` of `count` rounded up to a
    /// power of two — the coordinates of the challenge that combines the
    /// chunks' claims. `slot_vars + combine_vars` is the table's `k_T − n_T`.
    pub fn combine_vars(&self) -> usize {
        self.count.next_power_of_two().trailing_zeros() as usize
    }

    /// The input cells of the table's trees together.
    fn cells(&self, shape: &ArgueShape) -> u128 {
        (self.count as u128) << (shape.num_vars + self.slot_vars).min(120)
    }

    /// Chunk `j`'s interactions.
    pub fn range(&self, j: usize, interactions: usize) -> std::ops::Range<usize> {
        let start = (j << self.slot_vars).min(interactions);
        start..((j + 1) << self.slot_vars).min(interactions)
    }
}

/// ★ How `pack` lays a table out, one function for every side.
///
/// [`GkrPack::Off`]: [`Chunking::whole`]. [`GkrPack::Chunks`]: over
/// `b ∈ [0, B]` (`2^B` today's slots), `c = ⌈m / 2^b⌉` chunks, the `b` that
/// minimises `c·2^{n+b} + (c − 1)·2^{tree_log_cells}` — the cells, and each
/// extra tree at its price — ties to the larger `b`, so `b = B` (today's tree)
/// unless cutting saves.
pub fn chunking(shape: &ArgueShape, pack: GkrPack) -> Chunking {
    let whole = Chunking::whole(shape);
    let GkrPack::Chunks { tree_log_cells } = pack else {
        return whole;
    };
    if shape.interactions < 2 {
        return whole;
    }
    let tree = 1u128 << tree_log_cells.min(120);
    let cost = |c: &Chunking| c.cells(shape) + (c.count as u128 - 1) * tree;
    let mut best = whole;
    for slot_vars in (0..whole.slot_vars).rev() {
        let candidate = Chunking {
            slot_vars,
            count: shape.interactions.div_ceil(1 << slot_vars),
        };
        if cost(&candidate) < cost(&best) {
            best = candidate;
        }
    }
    // The chunks and the all-padding ones after them fill today's slots
    // exactly, so the combined claim sits at a point of today's length.
    if best.slot_vars + best.combine_vars() != whole.slot_vars {
        return whole;
    }
    best
}

/// The public schedule of a batched argue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArguePlan {
    pub shapes: Vec<ArgueShape>,
    /// Each table's trees ([`chunking`]); one tree a table under
    /// [`GkrPack::Off`].
    pub chunks: Vec<Chunking>,
    /// The bins, in the order their ladders run; each in table order, a
    /// table's chunks adjacent and in order.
    pub bins: Vec<Vec<usize>>,
    /// `n_max`: the constraint sumcheck's rounds.
    pub num_vars: usize,
    /// `D_max`: its round degree.
    pub degree: usize,
}

impl ArguePlan {
    /// Every table's trees together: the bus outputs a proof carries.
    pub fn trees(&self) -> usize {
        self.chunks.iter().map(|c| c.count).sum()
    }

    /// Where table `t`'s first tree sits among the bus outputs (table order,
    /// then chunk order).
    pub fn first_tree(&self, t: usize) -> usize {
        self.chunks[..t].iter().map(|c| c.count).sum()
    }

    /// The input variables of each of table `t`'s trees.
    pub fn tree_vars(&self, t: usize) -> usize {
        self.shapes[t].num_vars + self.chunks[t].slot_vars
    }
}

/// [`argue_plan_packed`] with one tree a table: today's plan.
pub fn argue_plan(shapes: &[ArgueShape], bin_log_cells: u8) -> ArguePlan {
    argue_plan_packed(shapes, bin_log_cells, GkrPack::Off)
}

/// ★★ THE PLAN, one function for every side (the `absorb_roots` lesson: two
/// spellings of a public ordering drift silently).
///
/// Each table's trees are [`chunking`]'s under `pack`, one unit of
/// `c·2^{n_T+b}` input cells (`2^{k_T}` under [`GkrPack::Off`]). Bins: units
/// sorted by `(cells desc, index asc)`, first-fit-decreasing under
/// `Σ cells ≤ 2^{bin_log_cells}`; a unit at or over the cap sits alone. Bins
/// run in the order they open, each in table order.
pub fn argue_plan_packed(shapes: &[ArgueShape], bin_log_cells: u8, pack: GkrPack) -> ArguePlan {
    let cap = 1u128 << bin_log_cells.min(127);
    let chunks: Vec<Chunking> = shapes.iter().map(|shape| chunking(shape, pack)).collect();
    let unit = |t: usize| chunks[t].cells(&shapes[t]);
    let mut order: Vec<usize> = (0..shapes.len()).collect();
    order.sort_by(|&a, &b| unit(b).cmp(&unit(a)).then(a.cmp(&b)));
    // (tables, cells used, open to more)
    let mut bins: Vec<(Vec<usize>, u128, bool)> = Vec::new();
    for t in order {
        let cells = unit(t);
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
        chunks,
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
    /// Every tree's bus output fraction, in table order: one a table, or —
    /// under a [`GkrPack::Chunks`] format — one for each chunk of a cut table,
    /// in chunk order.
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
    /// This table's bus output numerator is replaced by the value
    /// that makes `Σ p/q = 0` over every table, absorbed and sent as if true,
    /// while its ladder runs on the TRUE tree (only step 0's residual sees it).
    /// A cut table's first chunk carries the forgery.
    pub balance_with: Option<usize>,
    /// The prover lays its trees out under this packing instead of the
    /// config's.
    pub pack: Option<GkrPack>,
    /// This cut table's first two chunks trade their interactions: honest
    /// trees, outputs and ladder, over the chunks in the wrong order.
    pub swap_chunks: Option<usize>,
    /// This cut table's last chunk puts mass in a padding slot, which the
    /// verifier takes as `0/1` whatever the prover built: the slot's
    /// numerator is the constant that brings the whole bus to zero (spread
    /// over the table's rows), and the trees, their outputs and the ladder are
    /// honest over the lists so changed. The table's last chunk must hold a
    /// padding slot of its own (a short last chunk).
    pub pad_mass: Option<usize>,
    /// `(table, fault, layer)`: this table's first tree departs at its ladder
    /// step over `layer` (2 is its first two-layer step), as
    /// [`gkr_lockstep::StepFault`] says, and steps honestly after.
    pub ladder: Option<(usize, gkr_lockstep::StepFault, usize)>,
    /// This table's bus output forged as under `balance_with`, and on a
    /// two-layer ladder its first tree's first step shifted along the
    /// diagonal ([`gkr_lockstep::StepFault::Diagonal`]) so that step's
    /// residual holds for the forged output; every later step honest. Only
    /// the plane's second, independent challenge refuses it.
    pub diagonal: Option<usize>,
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
    /// A block's commit-time out-of-domain answers batched into its chains
    /// (`multilinear_block::commit_ood_point`); off, the answers are absorbed
    /// and counted but no chain carries them.
    pub commit_ood: bool,
}

impl VerifierChecks {
    pub const ALL: Self = Self {
        residual: true,
        constraint: true,
        balance: true,
        reduce_shape: true,
        preprocessed: true,
        commit_ood: true,
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

/// Each of a table's trees' interactions under `chunks`: its own list when it
/// is one tree, else each chunk's in order. A last chunk too short to take
/// `2^b` slots by itself is padded with `0/1` interactions, which are the
/// padding's own cells, so its tree is the `2^b`-slot tree.
fn chunk_lists<'a, E: IsField + 'static>(
    interactions: &'a [Interaction<E>],
    chunks: &Chunking,
) -> Vec<std::borrow::Cow<'a, [Interaction<E>]>> {
    use std::borrow::Cow;
    if chunks.count == 1 {
        return vec![Cow::Borrowed(interactions)];
    }
    let slots = 1usize << chunks.slot_vars;
    (0..chunks.count)
        .map(|j| {
            let list = &interactions[chunks.range(j, interactions.len())];
            if list.len().next_power_of_two() == slots {
                return Cow::Borrowed(list);
            }
            let padding = Interaction::new(
                Affine::constant(FieldElement::zero()),
                Affine::constant(FieldElement::one()),
            );
            let mut padded = list.to_vec();
            padded.resize(slots / 2 + 1, padding);
            Cow::Owned(padded)
        })
        .collect()
}

/// The real interactions in a cut table's last chunk (the rest of its list
/// is [`chunk_lists`]' padding).
fn chunks_real_len(interactions: usize, chunks: &Chunking) -> usize {
    chunks.range(chunks.count - 1, interactions).len()
}

/// The table a tree belongs to, `tree` counted in table order then chunk
/// order.
fn table_of_tree(plan: &ArguePlan, tree: usize) -> usize {
    let mut first = 0;
    for (t, chunks) in plan.chunks.iter().enumerate() {
        first += chunks.count;
        if tree < first {
            return t;
        }
    }
    plan.chunks.len()
}

/// A table's GKR claim from its trees' input claims: its one tree's, or — a
/// cut table — the chunks' combined at `s'`, drawn here, once the chunks' last
/// line challenge has fixed their claims ([`combine_chunk_claims`]).
fn table_claim<E, T>(
    chunk_claims: &[GkrClaim<E>],
    chunks: &Chunking,
    transcript: &mut T,
) -> Result<GkrClaim<E>, MlError>
where
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: Send + Sync,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>,
{
    if chunk_claims.len() != chunks.count {
        return Err(MlError::ArgueShapeMismatch { part: "chunks" });
    }
    if chunks.count == 1 {
        return Ok(chunk_claims[0].clone());
    }
    let combine: Vec<FieldElement<E>> = (0..chunks.combine_vars())
        .map(|_| transcript.sample_field_element())
        .collect();
    combine_chunk_claims(chunk_claims, &combine)
}

/// ★ A cut table's claim: its `c` chunks' claims, all at one point `P`,
/// combined at `s'` (`ℓ` coordinates, `c ≤ 2^ℓ`) into the claim of today's one
/// tree at `s' ‖ P`:
///
/// - `p̃ = Σ_{j<c} eq(s', j)·p̃_j`;
/// - `q̃ = Σ_{j<c} eq(s', j)·q̃_j + Σ_{c≤j<2^ℓ} eq(s', j)` — chunks past the
///   last are all padding, `q ≡ 1`, and never built.
///
/// The padded layer's top `ℓ` slot bits index the chunk, so its multilinear
/// extension at `s' ‖ P` is exactly this combination of the chunks' at `P`.
/// A false chunk claim survives only if `s'` is a root of a nonzero
/// multilinear polynomial in `ℓ` variables: `≤ ℓ/|E|`.
#[doc(hidden)]
pub fn combine_chunk_claims<E>(
    chunk_claims: &[GkrClaim<E>],
    combine: &[FieldElement<E>],
) -> Result<GkrClaim<E>, MlError>
where
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: Send + Sync,
{
    let weights = multilinear::eq::eq_evals(combine);
    let first = chunk_claims
        .first()
        .ok_or(MlError::ArgueShapeMismatch { part: "chunks" })?;
    if chunk_claims.len() > weights.len()
        || chunk_claims.iter().any(|claim| claim.point != first.point)
    {
        return Err(MlError::ArgueShapeMismatch { part: "chunks" });
    }
    let mut p = FieldElement::<E>::zero();
    let mut q = FieldElement::<E>::zero();
    for (claim, w) in chunk_claims.iter().zip(&weights) {
        p += w * &claim.p;
        q += w * &claim.q;
    }
    for w in &weights[chunk_claims.len()..] {
        q += w.clone();
    }
    Ok(GkrClaim {
        point: combine.iter().chain(&first.point).cloned().collect(),
        p,
        q,
    })
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

/// A bin's tree as its build left it: built, or queued on the card with its
/// output unread (`LAMBDA_VM_ARGUE_ROUND_SLOT`), finalized once every tree of
/// the bin is queued.
enum Pending<E: IsField> {
    Ready(BinTree<E>),
    Queued(multilinear::logup::PrefetchedTree<E>),
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
    let ArgueFormat::Batched {
        bin_log_cells,
        pack,
        ladder,
    } = config.format.argue
    else {
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
        pack,
        ladder,
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
    pack: GkrPack,
    ladder: GkrLadder,
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
    let plan = argue_plan_packed(
        &shapes,
        faults.bin_log_cells.unwrap_or(bin_log_cells),
        faults.pack.unwrap_or(pack),
    );
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
    // Each tree's interactions, table order then chunk order: a whole table's
    // own list, or a chunk of it.
    let mut lists: Vec<std::borrow::Cow<'_, [Interaction<E>]>> = Vec::with_capacity(plan.trees());
    for (t, chunks) in plan.chunks.iter().enumerate() {
        let mut table_lists = chunk_lists(&interactions[t], chunks);
        if faults.swap_chunks == Some(t) && table_lists.len() > 1 {
            table_lists.swap(0, 1);
        }
        lists.extend(table_lists);
    }
    // The fault `pad_mass`: the bus's whole sum moved into a padding slot.
    if let Some(target) = faults.pad_mass {
        let mut sum = FieldElement::<E>::zero();
        for (i, list) in lists.iter().enumerate() {
            let t = table_of_tree(&plan, i);
            let layers = host_tree(list, &tables[t].trace)?;
            let (p, q) = gkr_lockstep::LadderTree::output(&gkr_lockstep::HostLayers(&layers));
            sum += p * q.inv().expect("q != 0");
        }
        let rows = FieldElement::<E>::from(1u64 << shapes[target].num_vars);
        let tree = plan.first_tree(target) + plan.chunks[target].count - 1;
        let real = chunks_real_len(interactions[target].len(), &plan.chunks[target]);
        let list = lists[tree].to_mut();
        assert!(
            plan.chunks[target].count > 1 && list.len() > real,
            "pad_mass: table {target}'s last chunk has no padding slot of its own"
        );
        let last = list.len() - 1;
        list[last] = Interaction::new(
            Affine::constant(-(sum * rows.inv().expect("rows != 0"))),
            Affine::constant(FieldElement::one()),
        );
    }

    // The faults `balance_with` and `diagonal`: the numerator that balances
    // the bus, from every true output, and how far it is from the true one.
    let fake_output: Option<(usize, FieldElement<E>)> =
        match faults.balance_with.or(faults.diagonal) {
            None => None,
            Some(target) => {
                let mut outputs = Vec::with_capacity(lists.len());
                for (i, list) in lists.iter().enumerate() {
                    let t = table_of_tree(&plan, i);
                    let layers = host_tree(list, &tables[t].trace)?;
                    outputs.push(gkr_lockstep::LadderTree::output(&gkr_lockstep::HostLayers(
                        &layers,
                    )));
                }
                let target = plan.first_tree(target);
                let mut rest = FieldElement::<E>::zero();
                for (i, (p, q)) in outputs.iter().enumerate() {
                    if i != target {
                        rest += p * q.inv().expect("q != 0");
                    }
                }
                Some((target, -(rest * &outputs[target].1)))
            }
        };
    let mut split = ArgueSplit {
        ledger_start: multilinear::gpu::ledger_reserved(),
        ..ArgueSplit::default()
    };
    let gruen_layers = multilinear::gkr_gruen::gruen_layers();
    let gruen2_layers = multilinear::gkr_gruen::gruen2_layers();
    let gruen_rounds = multilinear::gkr_gruen::gruen_rounds();
    // ── G: a ladder per bin, its outputs absorbed just before it ──────────
    let mut bus_outputs: Vec<Option<(FieldElement<E>, FieldElement<E>)>> = vec![None; plan.trees()];
    let mut claims: Vec<Option<GkrClaim<E>>> = vec![None; tables.len()];
    let mut gkr = Vec::with_capacity(plan.bins.len());
    // Under the round slot the columns' trees of a bin are queued one after
    // another and their outputs read once all are (each tree has its stream).
    let queue = multilinear::round_slot::argue_round_slot()
        != multilinear::round_slot::RoundSlotSetting::Off;
    let host_built = |t: usize, list: &[Interaction<E>]| -> Result<BinTree<E>, MlError> {
        // Lean: only the factors this tree's interactions read are lifted
        // (`LAMBDA_VM_ARGUE_HOST_TREE_LEAN`); the same layer either way.
        let layer = if logup::host_tree_lean() {
            logup::input_layer_lean(list, &tables[t].trace)?
        } else {
            logup::input_layer(list, &tables[t].trace.factors()?)?
        };
        Ok(BinTree::Built(Box::new(
            multilinear::gkr::FractionTree::build(layer)?,
        )))
    };
    for bin in &plan.bins {
        // The bin's trees: each table's chunks, in order.
        let members: Vec<(usize, usize)> = bin
            .iter()
            .flat_map(|&t| {
                let first = plan.first_tree(t);
                (0..plan.chunks[t].count).map(move |j| (t, first + j))
            })
            .collect();
        // A bin the card cannot carry whole hands its trees' input layers
        // back (`LAMBDA_VM_ARGUE_BIN_HAND_BACK`): each promises less, so the
        // bin's last trees are built on the card too. The same trees.
        let hand_back = at != Where::Host && multilinear::gpu::bin_hand_back() && {
            let carries: Vec<u64> = members
                .iter()
                .map(|&(t, i)| {
                    multilinear::gpu::carry_bytes(
                        lists[i].len().next_power_of_two() << tables[t].trace.num_vars(),
                    )
                })
                .collect();
            multilinear::gpu::bin_hand_back_forced()
                || multilinear::gpu::bin_cannot_carry(
                    multilinear::gpu::ledger_reserved(),
                    multilinear::gpu::reserve_budget(),
                    &carries,
                )
        };
        let handed_before = multilinear::gpu::handed_back_by_preference();
        let build_trims_before = multilinear::gpu::build_trims();
        let pending: Vec<Pending<E>> = multilinear::gpu::prefer_hand_back(hand_back, || {
            members
                .iter()
                .map(|&(t, i)| -> Result<Pending<E>, MlError> {
                    let (trace, interactions) = (&tables[t].trace, &lists[i][..]);
                    if at == Where::Host {
                        return Ok(Pending::Ready(BinTree::Host(host_tree(
                            interactions,
                            trace,
                        )?)));
                    }
                    let columns_tree = || {
                        if queue {
                            logup::resident_tree_from_columns_deferred(interactions, trace)
                                .map(Pending::Queued)
                        } else {
                            logup::resident_tree_from_columns(interactions, trace)
                                .map(|tree| Pending::Ready(BinTree::Built(Box::new(tree))))
                        }
                    };
                    // Where today's argue builds it: written from the base columns
                    // with no lift (`gpu::argue_no_lift`), else from the
                    // lifted factors on the card, else here — and a big host tree
                    // still folds on a device. The same tree every way.
                    let started = std::time::Instant::now();
                    let mut resident_seen = None;
                    let mut public_tried = false;
                    let mut public_why: Option<String> = None;
                    let on_card = (multilinear::gpu::argue_no_lift()
                        && trace.device_factors().is_none())
                    .then(columns_tree)
                    .flatten()
                    .or_else(|| {
                        // A chunk that reads a public factor (a selector) has no
                        // plan over the table's columns: built on the card from
                        // the columns it reads and its selectors
                        // (`LAMBDA_VM_ARGUE_PUBLIC_COLUMNS`), in a bin that hands
                        // its trees back. Elsewhere the table's factors are
                        // lifted as before, which costs less where they fit.
                        (logup::public_columns() && hand_back && trace.device_factors().is_none())
                            .then(|| {
                                public_tried = true;
                                logup::resident_tree_with_public(interactions, trace)
                                    .map_err(|why| public_why = Some(why))
                                    .ok()
                            })
                            .flatten()
                            .map(|tree| Pending::Ready(BinTree::Built(Box::new(tree))))
                    })
                    .or_else(|| {
                        let resident = trace.reside_from_columns();
                        resident_seen = Some(resident.is_some());
                        let from_columns = (multilinear::gpu::argue_gkr_input()
                            && resident.is_some())
                        .then(columns_tree)
                        .flatten();
                        from_columns.or_else(|| {
                            resident
                                .and_then(|resident| logup::resident_tree(interactions, resident))
                                .map(|tree| Pending::Ready(BinTree::Built(Box::new(tree))))
                        })
                    });
                    let tree = match on_card {
                        Some(tree) => {
                            split.trees_card += 1;
                            tree
                        }
                        None => {
                            split.trees_host += 1;
                            let why = match (public_tried, resident_seen) {
                                (true, _) => format!(
                                    "its public-columns card build was declined: {}",
                                    public_why.as_deref().unwrap_or("no reason given")
                                ),
                                (false, Some(false)) => {
                                    "no resident columns or factors for its table".to_string()
                                }
                                _ => "the card declined the build".to_string(),
                            };
                            multilinear::gpu::note_gkr_tree_host_built(
                                &why,
                                t,
                                interactions.len().next_power_of_two() << trace.num_vars(),
                            );
                            Pending::Ready(host_built(t, interactions)?)
                        }
                    };
                    split.trees += started.elapsed().as_secs_f64();
                    Ok(tree)
                })
                .collect::<Result<_, _>>()
        })?;
        if hand_back {
            split.bins_handed_back += 1;
            split.trees_handed_back +=
                multilinear::gpu::handed_back_by_preference() - handed_before;
        }
        // The queued trees' outputs, now that the bin's every tree is queued;
        // one that cannot be read is built here, before anything is absorbed.
        let started = std::time::Instant::now();
        let trees: Vec<BinTree<E>> = pending
            .into_iter()
            .zip(&members)
            .map(|(tree, &(t, i))| match tree {
                Pending::Ready(tree) => Ok(tree),
                Pending::Queued(queued) => match queued.into_tree() {
                    Some(tree) => Ok(BinTree::Built(Box::new(tree))),
                    None => {
                        split.trees_host_late += 1;
                        multilinear::gpu::note_gkr_tree_host_built(
                            "its queued card build could not be read",
                            t,
                            lists[i].len().next_power_of_two() << tables[t].trace.num_vars(),
                        );
                        host_built(t, &lists[i])
                    }
                },
            })
            .collect::<Result<_, MlError>>()?;
        split.trees += started.elapsed().as_secs_f64();
        split.ledger_trees = split.ledger_trees.max(multilinear::gpu::ledger_reserved());
        split.build_trims += multilinear::gpu::build_trims() - build_trims_before;
        split.pool_high_trees = split
            .pool_high_trees
            .max(multilinear::gpu::pool_reserved_high());
        let hosts: Vec<Option<gkr_lockstep::HostLayers<'_, E>>> = trees
            .iter()
            .map(|tree| match tree {
                BinTree::Host(layers) => Some(gkr_lockstep::HostLayers(layers)),
                BinTree::Built(_) => None,
            })
            .collect();
        // A built tree whose input layer the card cannot write again for its
        // last sumcheck has it written here instead (`InputOnHost`): the same
        // layer, so the same rounds.
        let writers: Vec<_> = members
            .iter()
            .map(|&(t, i)| {
                let (list, trace) = (&lists[i][..], &tables[t].trace);
                move || logup::input_layer_lean(list, trace)
            })
            .collect();
        let built: Vec<Option<InputOnHost<'_, E>>> = trees
            .iter()
            .zip(&members)
            .zip(&writers)
            .map(|((tree, &(t, i)), write)| match tree {
                BinTree::Built(tree) => Some(InputOnHost {
                    tree,
                    write,
                    table: t,
                    cells: lists[i].len().next_power_of_two() << tables[t].trace.num_vars(),
                }),
                BinTree::Host(_) => None,
            })
            .collect();
        let refs: Vec<&dyn gkr_lockstep::LadderTree<E>> = built
            .iter()
            .zip(&hosts)
            .map(|(built, host)| match (built, host) {
                (_, Some(host)) => host as &dyn gkr_lockstep::LadderTree<E>,
                (Some(built), None) => built as &dyn gkr_lockstep::LadderTree<E>,
                (None, None) => unreachable!("a host tree has its layers"),
            })
            .collect();
        // The faults `ladder` and `diagonal`: one tree of this bin departs
        // from the ladder.
        let armed: Option<(usize, gkr_lockstep::Faulted<'_, E>)> = faults
            .ladder
            .and_then(|(target, fault, layer)| {
                let tree = plan.first_tree(target);
                members
                    .iter()
                    .position(|&(_, i)| i == tree)
                    .map(|k| (k, gkr_lockstep::Faulted::at(refs[k], fault, layer)))
            })
            .or_else(|| {
                let (tree, fake) = fake_output.as_ref().filter(|_| faults.diagonal.is_some())?;
                let k = members.iter().position(|&(_, i)| i == *tree)?;
                let epsilon = fake - refs[k].output().0;
                Some((k, gkr_lockstep::Faulted::diagonal(refs[k], epsilon)))
            });
        let refs: Vec<&dyn gkr_lockstep::LadderTree<E>> = match &armed {
            Some((k, faulted)) => refs
                .iter()
                .enumerate()
                .map(|(j, &tree)| {
                    if j == *k {
                        faulted as &dyn gkr_lockstep::LadderTree<E>
                    } else {
                        tree
                    }
                })
                .collect(),
            None => refs,
        };
        for (&(_, i), tree) in members.iter().zip(&refs) {
            let mut output = tree.output();
            if let Some((target, p)) = &fake_output
                && *target == i
            {
                output.0 = p.clone();
            }
            transcript.append_field_element(&output.0);
            transcript.append_field_element(&output.1);
            bus_outputs[i] = Some(output);
        }
        let started = std::time::Instant::now();
        let trims_before = multilinear::gpu::rewrite_trims();
        let (steps, inputs) = gkr_lockstep::prove_trees(&refs, ladder, transcript)?;
        split.rewrite_trims += multilinear::gpu::rewrite_trims() - trims_before;
        split.ladders += started.elapsed().as_secs_f64();
        split.ledger_ladder = split.ledger_ladder.max(multilinear::gpu::ledger_reserved());
        split.pool_high_ladder = split
            .pool_high_ladder
            .max(multilinear::gpu::pool_reserved_high());
        let inputs: Vec<GkrClaim<E>> = inputs;
        let mut at_tree = 0;
        for &t in bin {
            let count = plan.chunks[t].count;
            let chunk_claims = &inputs[at_tree..at_tree + count];
            at_tree += count;
            claims[t] = Some(table_claim(chunk_claims, &plan.chunks[t], transcript)?);
        }
        gkr.push(steps);
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
    // the base columns on the card with no lift (`gpu::argue_no_lift`), the
    // lifted factors for a table they cannot read that way (lifted here, as the
    // per-table argue lifts it), or nothing, on the host.
    let mut constraints: Vec<Option<multilinear::fused::Constraints<F>>> =
        Vec::with_capacity(tables.len());
    let mut columns: Vec<Option<multilinear::gpu::ColumnFactors>> =
        Vec::with_capacity(tables.len());
    for table in tables {
        let shape = table.shape();
        let made = (at == Where::Device && multilinear::gpu_fused::argue_fused())
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
    split.ledger_zc = multilinear::gpu::ledger_reserved();
    // What an earlier argue on this thread left, so the line below is this one's.
    multilinear::gpu_fused::take_tail();
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
    LAST_ZC_SECS.with(|cell| cell.set(split.zc_setup + split.zc_rounds));
    let tail = multilinear::gpu_fused::take_tail();
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
            let (reduce, reduced_point) = claim_reduce::prove::<F, E, T, _>(
                &table.trace,
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
        // The Gruen count is the process's: an argue running beside this one
        // adds its layers too.
        eprintln!(
            "BATCHED ARGUE SPLIT: tables {} · bins {} · trees {:.3}s · ladders {:.3}s · zc setup \
             {:.3}s · zc rounds {:.3}s · reduce {:.3}s · Gruen layers (process) +{} · two-layer \
             Gruen layers (process) +{} · Gruen card rounds (process) +{}",
            tables.len(),
            plan.bins.len(),
            split.trees,
            split.ladders,
            split.zc_setup,
            split.zc_rounds,
            split.reduce,
            multilinear::gkr_gruen::gruen_layers() - gruen_layers,
            multilinear::gkr_gruen::gruen2_layers() - gruen2_layers,
            multilinear::gkr_gruen::gruen_rounds() - gruen_rounds
        );
        // The fused tables' later rounds by program length: the card's, and the
        // host's over what the card folded.
        eprintln!(
            "BATCHED ARGUE ZC TAIL: long tables {} (past the crossover {}) · card rounds {:.3}s · \
             host rounds {} in {:.3}s | short tables {} ({}) · card rounds {:.3}s · host rounds {} in {:.3}s",
            tail.tables[1],
            tail.on_card[1],
            tail.device[1],
            tail.host_rounds[1],
            tail.host[1],
            tail.tables[0],
            tail.on_card[0],
            tail.device[0],
            tail.host_rounds[0],
            tail.host[0],
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
        // The trees the ladders ran: one a table, or a cut table's chunks.
        eprintln!(
            "BATCHED ARGUE TREES: tables {} · trees {} · cut {} · tree log2 cells {:.2} · built on the card \
             {} · on the host {} (after a queued build {}) · ledger GiB: at the start {:.2} · the bin's trees \
             built {:.2} · after its ladder {:.2} · zerocheck set up {:.2} · handed back {} trees in {} bins · pool trimmed before {} rewrites · pool trimmed before {} \
             builds · pool reserved high GiB: the bin's trees built {:.2} · after its ladder {:.2}",
            tables.len(),
            plan.trees(),
            plan.chunks.iter().filter(|c| c.count > 1).count(),
            (0..tables.len())
                .map(|t| plan.chunks[t].count as f64 * (plan.tree_vars(t) as f64).exp2())
                .sum::<f64>()
                .log2(),
            split.trees_card - split.trees_host_late,
            split.trees_host + split.trees_host_late,
            split.trees_host_late,
            gib(split.ledger_start),
            gib(split.ledger_trees),
            gib(split.ledger_ladder),
            gib(split.ledger_zc),
            split.trees_handed_back,
            split.bins_handed_back,
            split.rewrite_trims,
            split.build_trims,
            gib(split.pool_high_trees),
            gib(split.pool_high_ladder)
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

/// A card-built tree in a bin's ladder whose input layer, when the card's step
/// over it fails (its layer handed back or kept without its padding, and the
/// card cannot write it again), is written here and stepped on the host: the
/// same layer, so the same rounds and bytes, where today the argue fails. Any
/// other layer's failure is the argue's, as before.
struct InputOnHost<'a, E: IsField> {
    tree: &'a multilinear::gkr::FractionTree<E>,
    write: &'a dyn Fn() -> Result<FractionLayer<E>, MlError>,
    table: usize,
    cells: usize,
}

impl<E> gkr_lockstep::LadderTree<E> for InputOnHost<'_, E>
where
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: Send + Sync,
{
    fn input_vars(&self) -> usize {
        gkr_lockstep::LadderTree::input_vars(self.tree)
    }

    fn output(&self) -> (FieldElement<E>, FieldElement<E>) {
        gkr_lockstep::LadderTree::output(self.tree)
    }

    fn step(
        &self,
        layer: usize,
        point: &[FieldElement<E>],
        weights: [FieldElement<E>; 2],
        claim: &(FieldElement<E>, FieldElement<E>),
    ) -> Result<Box<dyn gkr_lockstep::TreeStep<E> + '_>, MlError> {
        match gkr_lockstep::LadderTree::step(self.tree, layer, point, weights.clone(), claim) {
            Ok(step) => Ok(step),
            Err(failed) if layer == gkr_lockstep::LadderTree::input_vars(self.tree) => {
                let input = (self.write)()?;
                multilinear::gpu::note_input_written_on_host(
                    &format!("{failed:?}"),
                    self.table,
                    self.cells,
                );
                gkr_lockstep::host_step(&input, point, weights)
            }
            Err(failed) => Err(failed),
        }
    }

    fn step_double(
        &self,
        layer: usize,
        point: &[FieldElement<E>],
        weights: [FieldElement<E>; 2],
        claim: &(FieldElement<E>, FieldElement<E>),
    ) -> Result<Box<dyn gkr_lockstep::TreeStep<E> + '_>, MlError> {
        match gkr_lockstep::LadderTree::step_double(self.tree, layer, point, weights.clone(), claim)
        {
            Ok(step) => Ok(step),
            Err(failed) if layer == gkr_lockstep::LadderTree::input_vars(self.tree) => {
                let input = (self.write)()?;
                multilinear::gpu::note_input_written_on_host(
                    &format!("{failed:?}"),
                    self.table,
                    self.cells,
                );
                gkr_lockstep::host_step_double(&input, point, weights)
            }
            Err(failed) => Err(failed),
        }
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// What [`probe_constraint_sumcheck`] ran. A measurement; no proof holds it.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ConstraintProbe {
    pub tables: usize,
    /// Of them, the tables whose rounds the card ran.
    pub on_card: usize,
    /// Seconds: the rules, the weights and the card's sessions set up; the
    /// rounds; and of the long programs' later rounds, the host's.
    pub setup: f64,
    pub rounds: f64,
    pub host_tail: f64,
}

/// The stage [`probe_constraint_sumcheck`] names when a table's rounds would
/// have gone to the host and the probe was told not to take them there.
#[doc(hidden)]
pub const PROBE_DECLINED: &str = "a probe table the card declined";

/// ⛔ A MEASUREMENT, never part of a proof: the constraint sumcheck of
/// [`prove_argue`] (its phase C) over `tables`, with every challenge and every
/// input claim drawn from `transcript` — a throwaway the caller made for it —
/// and every message discarded. It costs what phase C costs over these
/// columns: the same rules, weights and rounds, where the batched argue would
/// run them. The claims' values are not the ladders' (no ladder runs), so the
/// messages are not a proof of anything; the work is the same.
///
/// Under [`Where::Device`] a table whose rounds the card declines is refused
/// ([`PROBE_DECLINED`]) instead of folded on the host; `host` lets the host
/// take it (a test without a card). Every table's device factors are let go
/// before it returns, whatever it returns.
///
/// `zc_only`: the bus's two rules weighted 0 and the claim 0 (`(1, 0, 0)`),
/// the tables batched by `λ^t` — the constraint sumcheck alone, as a phase A
/// that ran it apart from the bus would. Off, phase C's batch as it is.
#[doc(hidden)]
pub fn probe_constraint_sumcheck<F, E, T>(
    tables: &[CommittedTable<'_, F, E>],
    bin_log_cells: u8,
    pack: GkrPack,
    at: Where,
    host: bool,
    zc_only: bool,
    transcript: &mut T,
) -> Result<ConstraintProbe, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>,
{
    let probed =
        probe_constraint_rounds(tables, bin_log_cells, pack, at, host, zc_only, transcript);
    for table in tables {
        table.trace.release_device();
    }
    probed
}

/// [`probe_constraint_sumcheck`] before its tables let go of their device
/// factors: phase C of [`prove_argue`], line for line, over drawn claims.
#[allow(clippy::too_many_arguments)]
fn probe_constraint_rounds<F, E, T>(
    tables: &[CommittedTable<'_, F, E>],
    bin_log_cells: u8,
    pack: GkrPack,
    at: Where,
    host: bool,
    zc_only: bool,
    transcript: &mut T,
) -> Result<ConstraintProbe, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>,
{
    let started = std::time::Instant::now();
    let z: FieldElement<E> = transcript.sample_field_element();
    let alpha: FieldElement<E> = transcript.sample_field_element();
    let beta: FieldElement<E> = transcript.sample_field_element();
    let statements: Vec<TableStatement<'_, F, E>> = tables.iter().map(|t| t.statement()).collect();
    let shapes: Vec<ArgueShape> = statements.iter().map(ArgueShape::of).collect();
    let plan = argue_plan_packed(&shapes, bin_log_cells, pack);
    let interactions: Vec<Vec<Interaction<E>>> = tables
        .iter()
        .map(|table| {
            multilinear_logup::interactions(
                table.layout.interactions,
                table.slot_of().len(),
                &z,
                &alpha,
                |col| slot(table.slot_of(), col),
            )
        })
        .collect::<Result<_, _>>()?;
    // Where the ladders would leave each table's input claim: a point over its
    // input layer's variables, drawn here; the claimed values stay zero.
    let claims: Vec<GkrClaim<E>> = shapes
        .iter()
        .map(|shape| GkrClaim {
            point: (0..shape.input_vars)
                .map(|_| transcript.sample_field_element())
                .collect(),
            p: FieldElement::zero(),
            q: FieldElement::zero(),
        })
        .collect();

    // From here, phase C of `prove_argue` as it runs today.
    let xi: Vec<FieldElement<E>> = (0..plan.num_vars)
        .map(|_| transcript.sample_field_element())
        .collect();
    let lambda: FieldElement<E> = transcript.sample_field_element();
    let (weights, lambdas) = if zc_only {
        (
            multilinear::challenge_powers(&lambda, tables.len()),
            vec![
                FieldElement::one(),
                FieldElement::zero(),
                FieldElement::zero(),
            ],
        )
    } else {
        (
            table_weights(&lambda, tables.len()),
            vec![FieldElement::one(), lambda.clone(), &lambda * &lambda],
        )
    };
    let lambda2 = &lambda * &lambda;
    let programs: Vec<&IrShape<F, E>> = tables.iter().map(|t| t.shape()).collect();
    let (constraints, columns) = probe_sessions(tables, &programs, at);
    let device: Vec<Option<Arc<multilinear::gpu::DeviceFactors>>> = tables
        .iter()
        .map(|table| match at {
            Where::Host => None,
            Where::Device => table.trace.device_factors(),
        })
        .collect();
    let on_host = at == Where::Host || host;
    let mut polys = Vec::with_capacity(tables.len());
    for (t, table) in tables.iter().enumerate() {
        let n = table.num_vars();
        let (rules, row_point) =
            table_rules(&statements[t], &interactions[t], &claims[t].point, &beta)?;
        let xi_t = &xi[..n];
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
        let claim = if zc_only {
            FieldElement::zero()
        } else {
            &lambda * &claims[t].p + &lambda2 * &claims[t].q
        };
        polys.push(front_loaded::TableRounds::new(
            rules,
            lambdas.clone(),
            &claim,
            [xi_t, &row_point],
            n,
            device[t].as_deref(),
            columns[t].as_ref(),
            fused,
            || probe_host_factors(table, on_host),
        )?);
    }
    probe_rounds(polys, &weights, plan.degree, started, transcript)
}

/// Each table's fused program and its base columns on the card, as phase C
/// makes them, for a probe over `shapes` (a table's own program, or another
/// over its factors).
#[allow(clippy::type_complexity)]
fn probe_sessions<F, E>(
    tables: &[CommittedTable<'_, F, E>],
    shapes: &[&IrShape<F, E>],
    at: Where,
) -> (
    Vec<Option<multilinear::fused::Constraints<F>>>,
    Vec<Option<multilinear::gpu::ColumnFactors>>,
)
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let mut constraints = Vec::with_capacity(tables.len());
    let mut columns = Vec::with_capacity(tables.len());
    for (table, shape) in tables.iter().zip(shapes) {
        let made = (at == Where::Device && multilinear::gpu_fused::argue_fused())
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
    (constraints, columns)
}

/// A probe table's factors on the host, when the probe may take its rounds
/// there; else the refusal [`PROBE_DECLINED`] names.
fn probe_host_factors<F, E>(
    table: &CommittedTable<'_, F, E>,
    on_host: bool,
) -> Result<Vec<Mle<E>>, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    if on_host {
        table.trace.factors()
    } else {
        Err(MlError::DeviceFailed {
            stage: PROBE_DECLINED,
        })
    }
}

/// A probe's rounds over its tables' sessions, timed: `started` is when its
/// set-up began.
fn probe_rounds<E, T>(
    mut polys: Vec<front_loaded::TableRounds<'_, E>>,
    weights: &[FieldElement<E>],
    degree: usize,
    started: std::time::Instant,
    transcript: &mut T,
) -> Result<ConstraintProbe, MlError>
where
    E: IsField + Send + Sync + 'static,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>,
{
    let on_card = polys.iter().filter(|p| p.on_device()).count();
    let setup = started.elapsed().as_secs_f64();
    // What this thread's tail held before is not the probe's.
    multilinear::gpu_fused::take_tail();
    let started = std::time::Instant::now();
    front_loaded::prove_with(
        &mut polys,
        weights,
        degree,
        transcript,
        PaddingFault::default(),
    )?;
    let rounds = started.elapsed().as_secs_f64();
    let tail = multilinear::gpu_fused::take_tail();
    Ok(ConstraintProbe {
        tables: polys.len(),
        on_card,
        setup,
        rounds,
        host_tail: tail.host[0] + tail.host[1],
    })
}

/// ⛔ A MEASUREMENT, never part of a proof: the sumcheck a format that ran the
/// constraint sumcheck in phase A would run in phase B in its place — one
/// front-loaded sumcheck of `Σ_t λ^{3t}·[eq(r_t, ·)·Σ_j κ^j·f_j + λ·eq(ρ_t, ·)·N_t
/// + λ²·eq(ρ_t, ·)·D_t]`, the `f_j` the factors table `t`'s constraint program
/// reads ([`IrShape::reads_only`]), degree 2 — over `tables`, with every point,
/// challenge and claim drawn from `transcript` (a throwaway the caller made for
/// it) and every message discarded. The bus's rules are phase C's, at drawn
/// claim points; `r_t` is a drawn point's prefix, as phase A's would be.
///
/// Under [`Where::Device`] a table whose rounds the card declines is refused
/// ([`PROBE_DECLINED`]) unless `host`. Every table's device factors are let go
/// before it returns, whatever it returns.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn probe_merge_sumcheck<F, E, T>(
    tables: &[CommittedTable<'_, F, E>],
    z: &FieldElement<E>,
    alpha: &FieldElement<E>,
    bin_log_cells: u8,
    pack: GkrPack,
    at: Where,
    host: bool,
    transcript: &mut T,
) -> Result<ConstraintProbe, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>,
{
    let probed = probe_merge_rounds(tables, z, alpha, bin_log_cells, pack, at, host, transcript);
    for table in tables {
        table.trace.release_device();
    }
    probed
}

/// [`probe_merge_sumcheck`] before its tables let go of their device factors.
#[allow(clippy::too_many_arguments)]
fn probe_merge_rounds<F, E, T>(
    tables: &[CommittedTable<'_, F, E>],
    z: &FieldElement<E>,
    alpha: &FieldElement<E>,
    bin_log_cells: u8,
    pack: GkrPack,
    at: Where,
    host: bool,
    transcript: &mut T,
) -> Result<ConstraintProbe, MlError>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<E> + Send + Sync + 'static,
    E: IsField + Send + Sync + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
    T: crypto::fiat_shamir::is_transcript::IsTranscript<E>,
{
    let started = std::time::Instant::now();
    let statements: Vec<TableStatement<'_, F, E>> = tables.iter().map(|t| t.statement()).collect();
    let shapes: Vec<ArgueShape> = statements.iter().map(ArgueShape::of).collect();
    let plan = argue_plan_packed(&shapes, bin_log_cells, pack);
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
    // The ladders' input claims' points, and phase A's point, drawn here.
    let claims: Vec<Vec<FieldElement<E>>> = shapes
        .iter()
        .map(|shape| {
            (0..shape.input_vars)
                .map(|_| transcript.sample_field_element())
                .collect()
        })
        .collect();
    let r_a: Vec<FieldElement<E>> = (0..plan.num_vars)
        .map(|_| transcript.sample_field_element())
        .collect();
    let lambda: FieldElement<E> = transcript.sample_field_element();
    let kappa: FieldElement<E> = transcript.sample_field_element();
    let weights = table_weights(&lambda, tables.len());
    let lambdas = vec![FieldElement::one(), lambda.clone(), &lambda * &lambda];
    let reads: Vec<IrShape<F, E>> = tables.iter().map(|t| t.shape().reads_only().0).collect();
    let read_refs: Vec<&IrShape<F, E>> = reads.iter().collect();
    let (constraints, columns) = probe_sessions(tables, &read_refs, at);
    let device: Vec<Option<Arc<multilinear::gpu::DeviceFactors>>> = tables
        .iter()
        .map(|table| match at {
            Where::Host => None,
            Where::Device => table.trace.device_factors(),
        })
        .collect();
    let on_host = at == Where::Host || host;
    let zero = FieldElement::<E>::zero();
    let mut polys = Vec::with_capacity(tables.len());
    for (t, table) in tables.iter().enumerate() {
        let n = table.num_vars();
        let (weight_r, weight_z) = weight_slots(statements[t].kinds.len());
        let bus = logup::claim_statements(&interactions[t], &claims[t], n, weight_z)?;
        let kappas = multilinear::challenge_powers(&kappa, reads[t].num_roots());
        let merge = Rule::compiled(2, reads[t].program(&kappas, weight_r)?);
        let r_t = &r_a[..n];
        let fused = constraints[t]
            .as_ref()
            .map(|constraints| multilinear::gpu_fused::FusedInput {
                constraints,
                betas: &kappas,
                interactions: &interactions[t],
                claim_point: &claims[t],
                r: r_t,
                columns: columns[t].as_ref(),
            });
        polys.push(front_loaded::TableRounds::new(
            vec![merge, bus.numerator, bus.denominator],
            lambdas.clone(),
            &zero,
            [r_t, &bus.row_point],
            n,
            device[t].as_deref(),
            columns[t].as_ref(),
            fused,
            || probe_host_factors(table, on_host),
        )?);
    }
    probe_rounds(polys, &weights, 2, started, transcript)
}

thread_local! {
    /// The constraint sumcheck's seconds (set-up and rounds) of the last
    /// batched argue [`prove_argue`] ran on this thread.
    static LAST_ZC_SECS: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

/// The constraint sumcheck's seconds (phase C's set-up and rounds) of the last
/// batched argue this thread ran, and 0 from then on. A report.
#[doc(hidden)]
pub fn take_last_zc_secs() -> f64 {
    LAST_ZC_SECS.with(|cell| cell.replace(0.0))
}

/// Where a batched argue's time goes, printed under `LAMBDA_VM_BASE_SPLIT=1`.
#[derive(Default)]
struct ArgueSplit {
    trees: f64,
    /// Trees built on the card (queued or at once), and built on the host
    /// because the card path declined them (`host_built`): at once, or after
    /// a queued build could not be read.
    trees_card: usize,
    trees_host: usize,
    trees_host_late: usize,
    /// The device ledger's promise (bytes): when the argue began, the most
    /// any bin's trees held once built, the most after a bin's ladder (its
    /// trees still held), and once the zerocheck was set up.
    ledger_start: u64,
    ledger_trees: u64,
    ledger_ladder: u64,
    ledger_zc: u64,
    /// Bins whose trees handed their input layers back because the card
    /// could not carry them all (`LAMBDA_VM_ARGUE_BIN_HAND_BACK`), and the
    /// trees that did.
    bins_handed_back: usize,
    trees_handed_back: u64,
    /// Input layers written again after the pool handed its unused blocks
    /// back (`multilinear::gpu::rewrite_trims`).
    rewrite_trims: u64,
    /// Trees built after the pool handed its unused blocks back
    /// (`multilinear::gpu::build_trims`), and the pool's reserved high-water
    /// mark (bytes) once a bin's trees were built and after its ladder.
    build_trims: u64,
    pool_high_trees: u64,
    pool_high_ladder: u64,
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
    let ArgueFormat::Batched {
        bin_log_cells,
        pack,
        ladder,
    } = config.format.argue
    else {
        return Err(MlError::ArgueFormatMismatch);
    };
    let argue = &proof.argue;
    let n = statements.len();
    if argue.factor_values.len() != n || argue.reduces.len() != n {
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
        pack,
        ladder,
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
    pack: GkrPack,
    ladder: GkrLadder,
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
    let beta = beta.clone();
    let shapes: Vec<ArgueShape> = statements.iter().map(ArgueShape::of).collect();
    let plan = argue_plan_packed(&shapes, bin_log_cells, pack);
    // One bus output a tree: a table's, or each of a cut table's chunks'.
    if argue.bus_outputs.len() != plan.trees()
        || argue.factor_values.len() != n
        || argue.reduces.len() != n
    {
        return Err(MlError::ArgueShapeMismatch { part: "tables" });
    }
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
    for (bin, steps) in plan.bins.iter().zip(&argue.gkr) {
        let mut outputs = Vec::with_capacity(bin.len());
        let mut heights = Vec::with_capacity(bin.len());
        for &t in bin {
            let first = plan.first_tree(t);
            for output in &argue.bus_outputs[first..first + plan.chunks[t].count] {
                transcript.append_field_element(&output.0);
                transcript.append_field_element(&output.1);
                outputs.push(output.clone());
                heights.push(plan.tree_vars(t));
            }
        }
        let inputs = gkr_lockstep::verify_with(
            steps,
            &outputs,
            &heights,
            ladder,
            transcript,
            checks.residual,
        )?;
        let mut at_tree = 0;
        for &t in bin {
            let count = plan.chunks[t].count;
            let chunk_claims = inputs
                .get(at_tree..at_tree + count)
                .ok_or(MlError::ArgueShapeMismatch { part: "bins" })?;
            at_tree += count;
            claims[t] = Some(table_claim(chunk_claims, &plan.chunks[t], transcript)?);
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
