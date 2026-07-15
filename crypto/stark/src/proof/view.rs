//! Read accessors over an rkyv-archived STARK proof.
//!
//! There is a single proof representation: the rkyv archive. The verifier reads
//! it in place (zero-copy); owned callers reach this path by serializing to
//! rkyv first (see `Verifier::verify`/`multi_verify`). These inherent accessors
//! on the `Archived*` proof types localize the archive->native conversions —
//! scalar fields are copied out (`to_native()`), field-element/commitment
//! arrays are viewed in place (`slice_as_native`/`as_slice`) — so the verifier
//! reads archived proof data through plain method calls.

use crate::config::Commitment;
use crate::fri::fri_decommit::{ArchivedFriDecommitment, FriDecommitment};
use crate::proof::stark::{
    ArchivedDeepPolynomialOpening, ArchivedPolynomialOpenings, ArchivedStarkProof,
    DeepPolynomialOpening, PolynomialOpenings, StarkProof,
};
use crate::table::ArchivedTable;
use math::field::element::{ArchivedFieldElement, FieldElement};
use math::field::traits::{IsField, IsSubFieldOf};

/// Deserializer used to materialize the (tiny) per-proof `PI` public inputs.
pub type PiDeserializer = rkyv::api::high::HighDeserializer<rkyv::rancor::Error>;

/// `&[FieldElement<G>]` view over an archived field-element vector (no copy).
#[inline]
fn evals<G: IsField>(v: &rkyv::vec::ArchivedVec<ArchivedFieldElement<G>>) -> &[FieldElement<G>]
where
    G::BaseType: math::field::element::NativeArchived,
{
    ArchivedFieldElement::slice_as_native(v.as_slice())
}

impl<F: IsField> ArchivedPolynomialOpenings<F>
where
    F::BaseType: math::field::element::NativeArchived,
{
    pub fn merkle_path(&self) -> &[Commitment] {
        self.proof.merkle_path.as_slice()
    }

    pub fn evaluations(&self) -> &[FieldElement<F>] {
        evals(&self.evaluations)
    }

    pub fn evaluations_sym(&self) -> &[FieldElement<F>] {
        evals(&self.evaluations_sym)
    }
}

impl<F: IsSubFieldOf<E>, E: IsField> ArchivedDeepPolynomialOpening<F, E>
where
    F::BaseType: math::field::element::NativeArchived,
    E::BaseType: math::field::element::NativeArchived,
{
    pub fn composition_poly(&self) -> &ArchivedPolynomialOpenings<E> {
        &self.composition_poly
    }

    pub fn main_trace_polys(&self) -> &ArchivedPolynomialOpenings<F> {
        &self.main_trace_polys
    }

    pub fn precomputed_trace_polys(&self) -> Option<&ArchivedPolynomialOpenings<F>> {
        self.precomputed_trace_polys.as_ref()
    }

    pub fn aux_trace_polys(&self) -> Option<&ArchivedPolynomialOpenings<E>> {
        self.aux_trace_polys.as_ref()
    }
}

impl<E: IsField> ArchivedFriDecommitment<E>
where
    E::BaseType: math::field::element::NativeArchived,
{
    pub fn layers_auth_paths_len(&self) -> usize {
        self.layers_auth_paths.len()
    }

    pub fn layer_auth_path(&self, i: usize) -> &[Commitment] {
        self.layers_auth_paths[i].merkle_path.as_slice()
    }

    pub fn layers_evaluations_sym(&self) -> &[FieldElement<E>] {
        evals(&self.layers_evaluations_sym)
    }
}

impl<F: IsSubFieldOf<E>, E: IsField, PI> ArchivedStarkProof<F, E, PI>
where
    F::BaseType: math::field::element::NativeArchived,
    E::BaseType: math::field::element::NativeArchived,
    PI: rkyv::Archive,
    <PI as rkyv::Archive>::Archived: rkyv::Deserialize<PI, PiDeserializer>,
{
    pub fn trace_length(&self) -> usize {
        self.trace_length.to_native() as usize
    }

    pub fn lde_trace_main_merkle_root(&self) -> &Commitment {
        &self.lde_trace_main_merkle_root
    }

    pub fn lde_trace_aux_merkle_root(&self) -> Option<&Commitment> {
        self.lde_trace_aux_merkle_root.as_ref()
    }

    pub fn lde_trace_precomputed_merkle_root(&self) -> Option<&Commitment> {
        self.lde_trace_precomputed_merkle_root.as_ref()
    }

    pub fn trace_ood_evaluations(&self) -> &ArchivedTable<E> {
        &self.trace_ood_evaluations
    }

    pub fn composition_poly_root(&self) -> &Commitment {
        &self.composition_poly_root
    }

    pub fn composition_poly_parts_ood_evaluation(&self) -> &[FieldElement<E>] {
        evals(&self.composition_poly_parts_ood_evaluation)
    }

    pub fn fri_layers_merkle_roots(&self) -> &[Commitment] {
        self.fri_layers_merkle_roots.as_slice()
    }

    pub fn fri_final_poly_coeffs(&self) -> &[FieldElement<E>] {
        evals(&self.fri_final_poly_coeffs)
    }

    pub fn query_list_len(&self) -> usize {
        self.query_list.len()
    }

    pub fn query(&self, i: usize) -> &ArchivedFriDecommitment<E> {
        &self.query_list.as_slice()[i]
    }

    pub fn deep_poly_openings_len(&self) -> usize {
        self.deep_poly_openings.len()
    }

    pub fn deep_poly_opening(&self, i: usize) -> &ArchivedDeepPolynomialOpening<F, E> {
        &self.deep_poly_openings.as_slice()[i]
    }

    pub fn nonce(&self) -> Option<u64> {
        self.nonce.as_ref().map(|n| n.to_native())
    }

    /// The bus interaction's table contribution (L), if present. This is the
    /// only field of `BusPublicInputs` the verifier reads; copied out (it's a
    /// single field element, not worth a dedicated view type).
    pub fn bus_table_contribution(&self) -> Option<FieldElement<E>> {
        self.bus_public_inputs
            .as_ref()
            .map(|b| b.table_contribution.as_native().clone())
    }

    pub fn has_bus_public_inputs(&self) -> bool {
        self.bus_public_inputs.is_some()
    }

    /// Materializes the (tiny) `PI` public inputs via an rkyv deserialize.
    pub fn public_inputs(&self) -> Option<PI> {
        rkyv::deserialize::<PI, rkyv::rancor::Error>(&self.public_inputs).ok()
    }
}

// ---------------------------------------------------------------------------
// Field-coverage guards.
//
// Each view above mirrors a proof struct field-by-field, but nothing in the
// type system links a struct field to a view accessor: a field added to one of
// these structs would compile with no accessor, and the verifier — which reads
// proof data only through the views — would silently ignore it. That is a
// soundness gap.
//
// These functions never run. They exhaustively destructure each backing struct
// *without* `..`, so adding a field turns the omission into a compile error
// (E0027, "pattern does not mention field ...") pointing right here. When one
// stops compiling, add the matching view accessor above, then bind the new
// field below to acknowledge it is covered.
//
// This enforces accessor *presence*, not arm symmetry: an accessor whose Owned
// and Archived arms read different (same-typed) fields still type-checks and is
// only caught by a behavioral test.
#[allow(dead_code)]
fn assert_stark_proof_view_is_exhaustive<F: IsSubFieldOf<E>, E: IsField, PI>(
    p: &StarkProof<F, E, PI>,
) {
    let StarkProof {
        trace_length: _,
        lde_trace_main_merkle_root: _,
        lde_trace_aux_merkle_root: _,
        lde_trace_precomputed_merkle_root: _,
        trace_ood_evaluations: _,
        composition_poly_root: _,
        composition_poly_parts_ood_evaluation: _,
        fri_layers_merkle_roots: _,
        fri_final_poly_coeffs: _,
        query_list: _,
        deep_poly_openings: _,
        nonce: _,
        bus_public_inputs: _,
        public_inputs: _,
    } = p;
}

#[allow(dead_code)]
fn assert_polynomial_openings_view_is_exhaustive<F: IsField>(p: &PolynomialOpenings<F>) {
    let PolynomialOpenings {
        proof: _,
        evaluations: _,
        evaluations_sym: _,
    } = p;
}

#[allow(dead_code)]
fn assert_deep_polynomial_opening_view_is_exhaustive<F: IsSubFieldOf<E>, E: IsField>(
    p: &DeepPolynomialOpening<F, E>,
) {
    let DeepPolynomialOpening {
        composition_poly: _,
        main_trace_polys: _,
        precomputed_trace_polys: _,
        aux_trace_polys: _,
    } = p;
}

#[allow(dead_code)]
fn assert_fri_decommitment_view_is_exhaustive<F: IsField>(p: &FriDecommitment<F>) {
    let FriDecommitment {
        layers_auth_paths: _,
        layers_evaluations_sym: _,
    } = p;
}
