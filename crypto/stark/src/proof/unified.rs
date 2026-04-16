//! Unified multi-table proof with batched commitment and single FRI.
//!
//! Unlike [`StarkProof`](super::stark::StarkProof) which contains per-table FRI data,
//! this structure separates per-table OOD evaluations from shared commitment/FRI data.
//! All tables' columns are committed together in shared Merkle trees, and a single
//! FRI proof covers the combined DEEP composition polynomial.

use crypto::merkle_tree::proof::Proof;
use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};

use crate::{
    config::Commitment,
    fri::fri_decommit::FriDecommitment,
    lookup::BusPublicInputs,
    table::Table,
};

/// Per-table data in a unified proof: OOD evaluations and bus inputs.
/// Does NOT contain Merkle trees or FRI data — those are shared.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct TableOodData<E: IsField> {
    /// Length of this table's trace (before LDE).
    pub trace_length: usize,
    /// Number of main trace columns for this table.
    pub num_main_cols: usize,
    /// Number of aux trace columns (0 if no aux).
    pub num_aux_cols: usize,
    /// Whether this table has precomputed (preprocessed) columns.
    pub num_precomputed_cols: usize,
    /// Trace evaluations at the OOD point z: T(z) and T(gz).
    pub trace_ood_evaluations: Table<E>,
    /// Composition polynomial parts evaluated at z^N.
    pub composition_poly_parts_ood_evaluation: Vec<FieldElement<E>>,
    /// Bus interaction public inputs.
    pub bus_public_inputs: Option<BusPublicInputs<E>>,
}

/// Openings at a queried FRI index for the unified commitment.
///
/// Each query opens the shared main, aux, and composition Merkle trees
/// at the queried index and its symmetric partner.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct UnifiedQueryOpening<F: IsSubFieldOf<E>, E: IsField> {
    /// Main trace opening: Merkle proof + evaluations at (index, sym_index).
    pub main_proof: Proof<Commitment>,
    pub main_proof_sym: Proof<Commitment>,
    pub main_evaluations: Vec<FieldElement<F>>,
    pub main_evaluations_sym: Vec<FieldElement<F>>,
    /// Aux trace opening (if any tables have aux).
    pub aux_proof: Option<Proof<Commitment>>,
    pub aux_proof_sym: Option<Proof<Commitment>>,
    pub aux_evaluations: Vec<FieldElement<E>>,
    pub aux_evaluations_sym: Vec<FieldElement<E>>,
    /// Composition polynomial opening.
    pub composition_proof: Proof<Commitment>,
    pub composition_proof_sym: Proof<Commitment>,
    pub composition_evaluations: Vec<FieldElement<E>>,
    pub composition_evaluations_sym: Vec<FieldElement<E>>,
}

/// A unified multi-table STARK proof with batched commitments and single FRI.
///
/// Key differences from [`MultiProof`](super::stark::MultiProof):
/// - All main trace columns committed in ONE Merkle tree (not one per table)
/// - All aux columns in ONE Merkle tree
/// - All composition polynomial parts in ONE Merkle tree
/// - ONE FRI proof covering the combined DEEP composition polynomial
/// - Per-table data is limited to OOD evaluations and bus inputs
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct UnifiedMultiProof<F: IsSubFieldOf<E>, E: IsField> {
    // ====== Shared commitments ======
    /// Root of the batched main trace Merkle tree (all tables' main columns).
    pub main_trace_root: Commitment,
    /// Root of the batched auxiliary trace Merkle tree (all tables' aux columns).
    /// None if no table has auxiliary trace.
    pub aux_trace_root: Option<Commitment>,
    /// Root of the batched composition polynomial Merkle tree.
    pub composition_poly_root: Commitment,

    /// Precomputed commitment roots per preprocessed table.
    /// Vec of (table_index, commitment). Empty for non-preprocessed tables.
    pub precomputed_roots: Vec<(usize, Commitment)>,

    // ====== Per-table OOD data ======
    /// Per-table out-of-domain evaluations and metadata.
    pub table_data: Vec<TableOodData<E>>,

    // ====== Shared FRI proof ======
    /// FRI layer Merkle roots from the single FRI instance.
    pub fri_layers_merkle_roots: Vec<Commitment>,
    /// Final constant value from FRI folding.
    pub fri_last_value: FieldElement<E>,
    /// FRI decommitments (one per query).
    pub fri_query_list: Vec<FriDecommitment<E>>,

    // ====== Unified query openings ======
    /// Openings of the batched Merkle trees at each queried index.
    pub query_openings: Vec<UnifiedQueryOpening<F, E>>,

    // ====== Grinding ======
    pub nonce: Option<u64>,

    // ====== Layout ======
    /// Column layout: for each table, (start_main_col, num_main_cols, start_aux_col, num_aux_cols).
    /// Used by the verifier to extract per-table columns from unified openings.
    pub column_layout: Vec<ColumnRange>,
}

/// Describes where a table's columns live within the unified commitment.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ColumnRange {
    /// Starting column index in the batched main trace.
    pub main_start: usize,
    /// Number of main columns for this table.
    pub main_count: usize,
    /// Starting column index in the batched aux trace.
    pub aux_start: usize,
    /// Number of aux columns for this table.
    pub aux_count: usize,
    /// Starting column index in the batched composition polynomial.
    pub comp_start: usize,
    /// Number of composition polynomial parts for this table.
    pub comp_count: usize,
}
