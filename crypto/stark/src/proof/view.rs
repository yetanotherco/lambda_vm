//! Borrowed views over a STARK proof that work identically whether the proof
//! is a real owned object or an rkyv-archived buffer.
//!
//! Each view is `Owned(&T)` or `Archived(&Archived<T>)`; scalar fields are
//! copied out (`to_native()` vs. a plain copy), field-element/commitment
//! arrays stay borrowed (`slice_as_native` vs. the `Vec`'s slice directly).
//! This lets the verifier be written once and run over either representation
//! with no serialization and no logic duplication.

use crate::config::Commitment;
use crate::frame::Frame;
use crate::fri::fri_decommit::{ArchivedFriDecommitment, FriDecommitment};
use crate::proof::stark::{
    ArchivedDeepPolynomialOpening, ArchivedPolynomialOpenings, ArchivedStarkProof,
    DeepPolynomialOpening, PolynomialOpenings, StarkProof,
};
use crate::table::{ArchivedTable, Table};
use math::field::element::{ArchivedFieldElement, FieldElement};
use math::field::traits::{IsField, IsSubFieldOf};

/// Deserializer used to materialize the (tiny) per-proof `PI` public inputs.
pub type PiDeserializer = rkyv::api::high::HighDeserializer<rkyv::rancor::Error>;

/// `&[FieldElement<G>]` view over an archived field-element vector (no copy).
#[inline]
pub(crate) fn evals<G: IsField>(
    v: &rkyv::vec::ArchivedVec<ArchivedFieldElement<G>>,
) -> &[FieldElement<G>]
where
    G::BaseType: rkyv::Archive,
{
    ArchivedFieldElement::slice_as_native(v.as_slice())
}

pub enum PolynomialOpeningsView<'a, F: IsField>
where
    F::BaseType: rkyv::Archive,
{
    Owned(&'a PolynomialOpenings<F>),
    Archived(&'a ArchivedPolynomialOpenings<F>),
}

// Manual Clone/Copy: the variants are plain references, so this holds for
// every `F`, regardless of whether `F` itself is `Clone`/`Copy`. A derive
// would add a spurious `F: Clone`/`F: Copy` bound.
impl<'a, F: IsField> Clone for PolynomialOpeningsView<'a, F>
where
    F::BaseType: rkyv::Archive,
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, F: IsField> Copy for PolynomialOpeningsView<'a, F> where F::BaseType: rkyv::Archive {}

impl<'a, F: IsField> PolynomialOpeningsView<'a, F>
where
    F::BaseType: rkyv::Archive,
{
    pub fn merkle_path(&self) -> &'a [Commitment] {
        match self {
            Self::Owned(p) => &p.proof.merkle_path,
            Self::Archived(p) => p.proof.merkle_path.as_slice(),
        }
    }

    pub fn evaluations(&self) -> &'a [FieldElement<F>] {
        match self {
            Self::Owned(p) => &p.evaluations,
            Self::Archived(p) => evals(&p.evaluations),
        }
    }

    pub fn evaluations_sym(&self) -> &'a [FieldElement<F>] {
        match self {
            Self::Owned(p) => &p.evaluations_sym,
            Self::Archived(p) => evals(&p.evaluations_sym),
        }
    }
}

pub enum DeepPolynomialOpeningView<'a, F: IsSubFieldOf<E>, E: IsField>
where
    F::BaseType: rkyv::Archive,
    E::BaseType: rkyv::Archive,
{
    Owned(&'a DeepPolynomialOpening<F, E>),
    Archived(&'a ArchivedDeepPolynomialOpening<F, E>),
}

impl<'a, F: IsSubFieldOf<E>, E: IsField> Clone for DeepPolynomialOpeningView<'a, F, E>
where
    F::BaseType: rkyv::Archive,
    E::BaseType: rkyv::Archive,
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, F: IsSubFieldOf<E>, E: IsField> Copy for DeepPolynomialOpeningView<'a, F, E>
where
    F::BaseType: rkyv::Archive,
    E::BaseType: rkyv::Archive,
{
}

impl<'a, F: IsSubFieldOf<E>, E: IsField> DeepPolynomialOpeningView<'a, F, E>
where
    F::BaseType: rkyv::Archive,
    E::BaseType: rkyv::Archive,
{
    pub fn composition_poly(&self) -> PolynomialOpeningsView<'a, E> {
        match self {
            Self::Owned(p) => PolynomialOpeningsView::Owned(&p.composition_poly),
            Self::Archived(p) => PolynomialOpeningsView::Archived(&p.composition_poly),
        }
    }

    pub fn main_trace_polys(&self) -> PolynomialOpeningsView<'a, F> {
        match self {
            Self::Owned(p) => PolynomialOpeningsView::Owned(&p.main_trace_polys),
            Self::Archived(p) => PolynomialOpeningsView::Archived(&p.main_trace_polys),
        }
    }

    pub fn precomputed_trace_polys(&self) -> Option<PolynomialOpeningsView<'a, F>> {
        match self {
            Self::Owned(p) => p
                .precomputed_trace_polys
                .as_ref()
                .map(PolynomialOpeningsView::Owned),
            Self::Archived(p) => p
                .precomputed_trace_polys
                .as_ref()
                .map(PolynomialOpeningsView::Archived),
        }
    }

    pub fn aux_trace_polys(&self) -> Option<PolynomialOpeningsView<'a, E>> {
        match self {
            Self::Owned(p) => p
                .aux_trace_polys
                .as_ref()
                .map(PolynomialOpeningsView::Owned),
            Self::Archived(p) => p
                .aux_trace_polys
                .as_ref()
                .map(PolynomialOpeningsView::Archived),
        }
    }
}

pub enum FriDecommitmentView<'a, E: IsField>
where
    E::BaseType: rkyv::Archive,
{
    Owned(&'a FriDecommitment<E>),
    Archived(&'a ArchivedFriDecommitment<E>),
}

impl<'a, E: IsField> Clone for FriDecommitmentView<'a, E>
where
    E::BaseType: rkyv::Archive,
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, E: IsField> Copy for FriDecommitmentView<'a, E> where E::BaseType: rkyv::Archive {}

impl<'a, E: IsField> FriDecommitmentView<'a, E>
where
    E::BaseType: rkyv::Archive,
{
    pub fn layers_auth_paths_len(&self) -> usize {
        match self {
            Self::Owned(p) => p.layers_auth_paths.len(),
            Self::Archived(p) => p.layers_auth_paths.len(),
        }
    }

    pub fn layer_auth_path(&self, i: usize) -> &'a [Commitment] {
        match self {
            Self::Owned(p) => &p.layers_auth_paths[i].merkle_path,
            Self::Archived(p) => p.layers_auth_paths[i].merkle_path.as_slice(),
        }
    }

    pub fn layers_evaluations_sym(&self) -> &'a [FieldElement<E>] {
        match self {
            Self::Owned(p) => &p.layers_evaluations_sym,
            Self::Archived(p) => evals(&p.layers_evaluations_sym),
        }
    }
}

pub enum StarkTableView<'a, F: IsField>
where
    F::BaseType: rkyv::Archive,
{
    Owned(&'a Table<F>),
    Archived(&'a ArchivedTable<F>),
}

impl<'a, F: IsField> Clone for StarkTableView<'a, F>
where
    F::BaseType: rkyv::Archive,
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, F: IsField> Copy for StarkTableView<'a, F> where F::BaseType: rkyv::Archive {}

impl<'a, F: IsField> StarkTableView<'a, F>
where
    F::BaseType: rkyv::Archive,
{
    pub fn width(&self) -> usize {
        match self {
            Self::Owned(t) => t.width,
            Self::Archived(t) => t.width(),
        }
    }

    pub fn height(&self) -> usize {
        match self {
            Self::Owned(t) => t.height,
            Self::Archived(t) => t.height(),
        }
    }

    pub fn get_row(&self, row_idx: usize) -> &'a [FieldElement<F>] {
        match self {
            Self::Owned(t) => t.get_row(row_idx),
            Self::Archived(t) => t.get_row(row_idx),
        }
    }

    pub fn row_major_data(&self) -> &'a [FieldElement<F>] {
        match self {
            Self::Owned(t) => t.row_major_data(),
            Self::Archived(t) => t.row_major_data(),
        }
    }

    /// `true` iff `width * height` matches the backing data length — the
    /// invariant `get_row` indexing relies on.
    pub fn dimensions_consistent(&self) -> bool {
        match self {
            Self::Owned(t) => t
                .width
                .checked_mul(t.height)
                .is_some_and(|n| n == t.row_major_data().len()),
            Self::Archived(t) => t.dimensions_consistent(),
        }
    }

    pub fn into_frame(&self, main_trace_columns: usize, step_size: usize) -> Frame<F, F>
    where
        F: IsSubFieldOf<F>,
    {
        match self {
            Self::Owned(t) => t.into_frame(main_trace_columns, step_size),
            Self::Archived(t) => t.into_frame(main_trace_columns, step_size),
        }
    }
}

pub enum StarkProofView<'a, F: IsSubFieldOf<E>, E: IsField, PI>
where
    F::BaseType: rkyv::Archive,
    E::BaseType: rkyv::Archive,
    PI: rkyv::Archive,
    <PI as rkyv::Archive>::Archived: rkyv::Deserialize<PI, PiDeserializer>,
{
    Owned(&'a StarkProof<F, E, PI>),
    Archived(&'a ArchivedStarkProof<F, E, PI>),
}

impl<'a, F: IsSubFieldOf<E>, E: IsField, PI> Clone for StarkProofView<'a, F, E, PI>
where
    F::BaseType: rkyv::Archive,
    E::BaseType: rkyv::Archive,
    PI: rkyv::Archive,
    <PI as rkyv::Archive>::Archived: rkyv::Deserialize<PI, PiDeserializer>,
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, F: IsSubFieldOf<E>, E: IsField, PI> Copy for StarkProofView<'a, F, E, PI>
where
    F::BaseType: rkyv::Archive,
    E::BaseType: rkyv::Archive,
    PI: rkyv::Archive,
    <PI as rkyv::Archive>::Archived: rkyv::Deserialize<PI, PiDeserializer>,
{
}

impl<'a, F: IsSubFieldOf<E>, E: IsField, PI> StarkProofView<'a, F, E, PI>
where
    F::BaseType: rkyv::Archive,
    E::BaseType: rkyv::Archive,
    PI: rkyv::Archive,
    <PI as rkyv::Archive>::Archived: rkyv::Deserialize<PI, PiDeserializer>,
{
    pub fn trace_length(&self) -> usize {
        match self {
            Self::Owned(p) => p.trace_length,
            Self::Archived(p) => p.trace_length.to_native() as usize,
        }
    }

    pub fn lde_trace_main_merkle_root(&self) -> &'a Commitment {
        match self {
            Self::Owned(p) => &p.lde_trace_main_merkle_root,
            Self::Archived(p) => &p.lde_trace_main_merkle_root,
        }
    }

    pub fn lde_trace_aux_merkle_root(&self) -> Option<&'a Commitment> {
        match self {
            Self::Owned(p) => p.lde_trace_aux_merkle_root.as_ref(),
            Self::Archived(p) => p.lde_trace_aux_merkle_root.as_ref(),
        }
    }

    pub fn lde_trace_precomputed_merkle_root(&self) -> Option<&'a Commitment> {
        match self {
            Self::Owned(p) => p.lde_trace_precomputed_merkle_root.as_ref(),
            Self::Archived(p) => p.lde_trace_precomputed_merkle_root.as_ref(),
        }
    }

    pub fn trace_ood_evaluations(&self) -> StarkTableView<'a, E> {
        match self {
            Self::Owned(p) => StarkTableView::Owned(&p.trace_ood_evaluations),
            Self::Archived(p) => StarkTableView::Archived(&p.trace_ood_evaluations),
        }
    }

    pub fn composition_poly_root(&self) -> &'a Commitment {
        match self {
            Self::Owned(p) => &p.composition_poly_root,
            Self::Archived(p) => &p.composition_poly_root,
        }
    }

    pub fn composition_poly_parts_ood_evaluation(&self) -> &'a [FieldElement<E>] {
        match self {
            Self::Owned(p) => &p.composition_poly_parts_ood_evaluation,
            Self::Archived(p) => evals(&p.composition_poly_parts_ood_evaluation),
        }
    }

    pub fn fri_layers_merkle_roots(&self) -> &'a [Commitment] {
        match self {
            Self::Owned(p) => &p.fri_layers_merkle_roots,
            Self::Archived(p) => p.fri_layers_merkle_roots.as_slice(),
        }
    }

    pub fn fri_final_poly_coeffs(&self) -> &'a [FieldElement<E>] {
        match self {
            Self::Owned(p) => &p.fri_final_poly_coeffs,
            Self::Archived(p) => evals(&p.fri_final_poly_coeffs),
        }
    }

    pub fn query_list_len(&self) -> usize {
        match self {
            Self::Owned(p) => p.query_list.len(),
            Self::Archived(p) => p.query_list.len(),
        }
    }

    pub fn query(&self, i: usize) -> FriDecommitmentView<'a, E> {
        match self {
            Self::Owned(p) => FriDecommitmentView::Owned(&p.query_list[i]),
            Self::Archived(p) => FriDecommitmentView::Archived(&p.query_list.as_slice()[i]),
        }
    }

    pub fn deep_poly_openings_len(&self) -> usize {
        match self {
            Self::Owned(p) => p.deep_poly_openings.len(),
            Self::Archived(p) => p.deep_poly_openings.len(),
        }
    }

    pub fn deep_poly_opening(&self, i: usize) -> DeepPolynomialOpeningView<'a, F, E> {
        match self {
            Self::Owned(p) => DeepPolynomialOpeningView::Owned(&p.deep_poly_openings[i]),
            Self::Archived(p) => {
                DeepPolynomialOpeningView::Archived(&p.deep_poly_openings.as_slice()[i])
            }
        }
    }

    pub fn nonce(&self) -> Option<u64> {
        match self {
            Self::Owned(p) => p.nonce,
            Self::Archived(p) => p.nonce.as_ref().map(|n| n.to_native()),
        }
    }

    /// The bus interaction's table contribution (L), if present. This is the
    /// only field of `BusPublicInputs` the verifier reads; both sides copy it
    /// out (it's a single field element, not worth a dedicated view type).
    pub fn bus_table_contribution(&self) -> Option<FieldElement<E>> {
        match self {
            Self::Owned(p) => p
                .bus_public_inputs
                .as_ref()
                .map(|b| b.table_contribution.clone()),
            Self::Archived(p) => p
                .bus_public_inputs
                .as_ref()
                .map(|b| b.table_contribution.as_native().clone()),
        }
    }

    pub fn has_bus_public_inputs(&self) -> bool {
        match self {
            Self::Owned(p) => p.bus_public_inputs.is_some(),
            Self::Archived(p) => p.bus_public_inputs.is_some(),
        }
    }

    /// Materializes the (tiny) `PI` public inputs: a clone on the owned side,
    /// an rkyv deserialize on the archived side.
    pub fn public_inputs(&self) -> Option<PI>
    where
        PI: Clone,
    {
        match self {
            Self::Owned(p) => Some(p.public_inputs.clone()),
            Self::Archived(p) => {
                rkyv::deserialize::<PI, rkyv::rancor::Error>(&p.public_inputs).ok()
            }
        }
    }
}
