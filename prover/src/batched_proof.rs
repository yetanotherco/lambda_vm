//! The batched proof, as a value.
//!
//! Kept apart from the passes that build it so the recursion guest, which
//! compiles the prover without `parallel`, can carry and verify one.

use math::field::element::FieldElement;

use crate::tables::types::{GoldilocksExtension, GoldilocksField};

/// One table's DEEP openings at its group's query indices.
pub type Open = stark::proof::stark::DeepPolynomialOpenings<GoldilocksField, GoldilocksExtension>;

/// A table's half of a batched proof: everything it contributes that is not a
/// FRI, which is now its group's business.
///
/// This is what the per-table `StarkProof` keeps once the layers, the final
/// polynomial, the queries and the nonce move to the group — the 57.9% of the
/// proof that stops being paid once per table.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct TablePublic {
    pub trace_rows: usize,
    pub main_root: stark::config::Commitment,
    pub precomputed_root: Option<stark::config::Commitment>,
    pub aux_root: Option<stark::config::Commitment>,
    pub composition_poly_root: stark::config::Commitment,
    pub trace_ood: stark::table::Table<GoldilocksExtension>,
    pub trace_ood_next: stark::table::Table<GoldilocksExtension>,
    pub parts_ood: Vec<FieldElement<GoldilocksExtension>>,
    pub bus_public_inputs: Option<stark::lookup::BusPublicInputs<GoldilocksExtension>>,
}

/// A batched proof: what the five passes produce, assembled.
///
/// Additive, not a replacement. `StarkProof` and `multi_verify` are untouched
/// and still produce byte-identical proofs; this is a second format alongside
/// them, for the path that folds one FRI per domain instead of one per table.
///
/// The split is the whole point. A table keeps what only it can answer for —
/// its roots, its out-of-domain values, its openings — and a group carries the
/// FRI those tables share. That is the 57.9% of a per-table proof that stops
/// being paid 227 times.
#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BatchedProof {
    /// Per table, in AIR order.
    pub tables: Vec<TablePublic>,
    /// The chunk layout the tables follow; the verifier rebuilds the AIRs from it.
    pub table_counts: crate::TableCounts,
    /// Per table, in AIR order: its rows at its group's indices.
    pub openings: Vec<Open>,
    /// Which group each table belongs to.
    pub group_of: Vec<usize>,
    /// The AIR indices in the order they were folded, which the verifier
    /// replays because a table's coefficient depends on every table before it.
    pub fold_order: Vec<usize>,
    /// Per group, in ascending domain: the FRI they share.
    pub groups: Vec<(usize, stark::prover::GroupFri<GoldilocksExtension>)>,
    /// The statement, which the verifier binds before absorbing any root.
    pub public_output: Vec<u8>,
    pub page_configs: Vec<crate::tables::page::PageConfig>,
}
