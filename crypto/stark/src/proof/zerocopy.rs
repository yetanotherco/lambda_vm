//! Borrowed, zero-copy views over a STARK proof.
//!
//! The verifier reads a proof entirely through borrowed slices and references —
//! it never needs to *own* the proof data. [`StarkProofRef`] captures exactly
//! that read API, with two implementations:
//!
//! * `&StarkProof` — the conventional owned proof (borrows its own fields).
//! * `&ArchivedStarkProof` — an rkyv-archived proof read **in place** from its
//!   byte buffer, with no deserialization and no allocation.
//!
//! On a little-endian target an archived field element is bit-identical to a
//! native [`FieldElement`] (see
//! [`ArchivedFieldElement::slice_as_native`](math::field::element::ArchivedFieldElement::slice_as_native)),
//! so the archived implementation hands the verifier the same `&[FieldElement]`
//! slices the owned implementation does — the arithmetic code is shared verbatim.
//!
//! This module is only compiled with the `rkyv` feature; without it the verifier
//! uses `&StarkProof` directly.

use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};

use crate::config::Commitment;
use crate::frame::Frame;

// ============================================================================
// Borrowed views over the nested proof structures
// ============================================================================

/// Borrowed view of a [`PolynomialOpenings`](super::stark::PolynomialOpenings):
/// the two Merkle authentication paths (as `merkle_path` slices) and the two
/// evaluation slices.
pub struct PolynomialOpeningsRef<'a, F: IsField> {
    pub proof: &'a [Commitment],
    pub proof_sym: &'a [Commitment],
    pub evaluations: &'a [FieldElement<F>],
    pub evaluations_sym: &'a [FieldElement<F>],
}

/// Borrowed view of a [`DeepPolynomialOpening`](super::stark::DeepPolynomialOpening).
pub struct DeepPolynomialOpeningRef<'a, F: IsSubFieldOf<E>, E: IsField> {
    pub composition_poly: PolynomialOpeningsRef<'a, E>,
    pub main_trace_polys: PolynomialOpeningsRef<'a, F>,
    pub precomputed_trace_polys: Option<PolynomialOpeningsRef<'a, F>>,
    pub aux_trace_polys: Option<PolynomialOpeningsRef<'a, E>>,
}

/// Borrowed view of a [`FriDecommitment`](crate::fri::fri_decommit::FriDecommitment).
///
/// `layers_auth_paths` is one Merkle path (`&[Commitment]`) per FRI layer; access
/// layer `j` via [`Self::layer_auth_path`].
pub struct FriDecommitmentRef<'a, F: IsField> {
    /// Backing slices for each layer's authentication path.
    pub layer_paths: FriLayerPaths<'a>,
    pub layers_evaluations_sym: &'a [FieldElement<F>],
}

impl<'a, F: IsField> FriDecommitmentRef<'a, F> {
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layer_paths.len()
    }
    #[inline]
    pub fn layer_auth_path(&self, j: usize) -> &'a [Commitment] {
        self.layer_paths.path(j)
    }
}

use crypto::merkle_tree::proof::Proof;

/// Per-layer FRI auth paths, sourced from either an owned `Vec<Proof<Commitment>>`
/// or an archived `[ArchivedProof<Commitment>]`.
pub enum FriLayerPaths<'a> {
    Owned(&'a [Proof<Commitment>]),
    #[cfg(feature = "rkyv")]
    Archived(&'a [<Proof<Commitment> as rkyv::Archive>::Archived]),
}

impl<'a> FriLayerPaths<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            FriLayerPaths::Owned(v) => v.len(),
            #[cfg(feature = "rkyv")]
            FriLayerPaths::Archived(v) => v.len(),
        }
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    pub fn path(&self, j: usize) -> &'a [Commitment] {
        match self {
            FriLayerPaths::Owned(v) => &v[j].merkle_path,
            // `Commitment = [u8; 32]` archives to itself (align 1), so the
            // archived merkle_path slice IS a `&[Commitment]`.
            #[cfg(feature = "rkyv")]
            FriLayerPaths::Archived(v) => v[j].merkle_path.as_slice(),
        }
    }
}

/// Borrowed view of the out-of-domain trace evaluations [`Table`](crate::table::Table).
///
/// Holds a flat row-major slice plus dimensions, mirroring `Table`'s read API
/// (`width`, `height`, `get_row`, `into_frame`) without owning a `Vec`.
pub struct OodTableRef<'a, E: IsField> {
    data: &'a [FieldElement<E>],
    width: usize,
    height: usize,
}

impl<'a, E: IsField> OodTableRef<'a, E> {
    #[inline]
    pub fn new(data: &'a [FieldElement<E>], width: usize, height: usize) -> Self {
        Self {
            data,
            width,
            height,
        }
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    #[inline]
    pub fn get_row(&self, row_idx: usize) -> &[FieldElement<E>] {
        let start = row_idx * self.width;
        &self.data[start..start + self.width]
    }

    /// Build a [`Frame`] over this table, identical to `Table::into_frame`.
    /// Only the small OOD frame is materialized (bounded by `step_size × width`),
    /// never the whole proof.
    pub fn into_frame(&self, main_trace_columns: usize, step_size: usize) -> Frame<E, E>
    where
        E: IsSubFieldOf<E>,
    {
        crate::table::frame_from_rows(self.height, step_size, main_trace_columns, |row_idx| {
            self.get_row(row_idx)
        })
    }
}

// ============================================================================
// StarkProofRef: the verifier's read API over a proof
// ============================================================================

/// Everything the verifier reads from a single `StarkProof`, as borrowed views.
/// Implemented for both the owned `&StarkProof` and the archived
/// `&ArchivedStarkProof`.
pub trait StarkProofRef<'a, F: IsSubFieldOf<E>, E: IsField, PI> {
    fn trace_length(&self) -> usize;
    fn lde_trace_main_merkle_root(&self) -> &'a Commitment;
    fn lde_trace_aux_merkle_root(&self) -> Option<&'a Commitment>;
    fn lde_trace_precomputed_merkle_root(&self) -> Option<&'a Commitment>;
    fn trace_ood_evaluations(&self) -> OodTableRef<'a, E>;
    fn composition_poly_root(&self) -> &'a Commitment;
    fn composition_poly_parts_ood_evaluation(&self) -> &'a [FieldElement<E>];
    fn fri_layers_merkle_roots(&self) -> &'a [Commitment];
    fn fri_last_value(&self) -> &'a FieldElement<E>;
    fn query_list_len(&self) -> usize;
    fn query(&self, i: usize) -> FriDecommitmentRef<'a, E>;
    fn deep_poly_openings_len(&self) -> usize;
    fn deep_poly_opening(&self, i: usize) -> DeepPolynomialOpeningRef<'a, F, E>;
    fn nonce(&self) -> Option<u64>;
    /// `table_contribution` from `bus_public_inputs`, if present.
    fn bus_table_contribution(&self) -> Option<&'a FieldElement<E>>;
    fn has_bus_public_inputs(&self) -> bool;
    fn public_inputs(&self) -> &'a PI;
}

// ============================================================================
// Owned implementation: &StarkProof
// ============================================================================

use super::stark::StarkProof;

impl<'a, F: IsSubFieldOf<E>, E: IsField, PI> StarkProofRef<'a, F, E, PI>
    for &'a StarkProof<F, E, PI>
{
    #[inline]
    fn trace_length(&self) -> usize {
        (*self).trace_length
    }
    #[inline]
    fn lde_trace_main_merkle_root(&self) -> &'a Commitment {
        &(*self).lde_trace_main_merkle_root
    }
    #[inline]
    fn lde_trace_aux_merkle_root(&self) -> Option<&'a Commitment> {
        (*self).lde_trace_aux_merkle_root.as_ref()
    }
    #[inline]
    fn lde_trace_precomputed_merkle_root(&self) -> Option<&'a Commitment> {
        (*self).lde_trace_precomputed_merkle_root.as_ref()
    }
    #[inline]
    fn trace_ood_evaluations(&self) -> OodTableRef<'a, E> {
        let t = &(*self).trace_ood_evaluations;
        OodTableRef::new(t.data_slice(), t.width, t.height)
    }
    #[inline]
    fn composition_poly_root(&self) -> &'a Commitment {
        &(*self).composition_poly_root
    }
    #[inline]
    fn composition_poly_parts_ood_evaluation(&self) -> &'a [FieldElement<E>] {
        &(*self).composition_poly_parts_ood_evaluation
    }
    #[inline]
    fn fri_layers_merkle_roots(&self) -> &'a [Commitment] {
        &(*self).fri_layers_merkle_roots
    }
    #[inline]
    fn fri_last_value(&self) -> &'a FieldElement<E> {
        &(*self).fri_last_value
    }
    #[inline]
    fn query_list_len(&self) -> usize {
        (*self).query_list.len()
    }
    #[inline]
    fn query(&self, i: usize) -> FriDecommitmentRef<'a, E> {
        let q = &(*self).query_list[i];
        FriDecommitmentRef {
            layer_paths: FriLayerPaths::Owned(&q.layers_auth_paths),
            layers_evaluations_sym: &q.layers_evaluations_sym,
        }
    }
    #[inline]
    fn deep_poly_openings_len(&self) -> usize {
        (*self).deep_poly_openings.len()
    }
    fn deep_poly_opening(&self, i: usize) -> DeepPolynomialOpeningRef<'a, F, E> {
        let d = &(*self).deep_poly_openings[i];
        DeepPolynomialOpeningRef {
            composition_poly: polynomial_openings_ref(&d.composition_poly),
            main_trace_polys: polynomial_openings_ref(&d.main_trace_polys),
            precomputed_trace_polys: d
                .precomputed_trace_polys
                .as_ref()
                .map(polynomial_openings_ref),
            aux_trace_polys: d.aux_trace_polys.as_ref().map(polynomial_openings_ref),
        }
    }
    #[inline]
    fn nonce(&self) -> Option<u64> {
        (*self).nonce
    }
    #[inline]
    fn bus_table_contribution(&self) -> Option<&'a FieldElement<E>> {
        (*self)
            .bus_public_inputs
            .as_ref()
            .map(|bpi| &bpi.table_contribution)
    }
    #[inline]
    fn has_bus_public_inputs(&self) -> bool {
        (*self).bus_public_inputs.is_some()
    }
    #[inline]
    fn public_inputs(&self) -> &'a PI {
        &(*self).public_inputs
    }
}

#[inline]
fn polynomial_openings_ref<'a, G: IsField>(
    p: &'a super::stark::PolynomialOpenings<G>,
) -> PolynomialOpeningsRef<'a, G> {
    PolynomialOpeningsRef {
        proof: &p.proof.merkle_path,
        proof_sym: &p.proof_sym.merkle_path,
        evaluations: &p.evaluations,
        evaluations_sym: &p.evaluations_sym,
    }
}

// ============================================================================
// Zero-copy implementation: &ArchivedStarkProof (little-endian only)
// ============================================================================

#[cfg(feature = "rkyv")]
mod archived_impl {
    use super::*;
    use crate::proof::stark::{ArchivedPolynomialOpenings, ArchivedStarkProof};
    use math::field::element::ArchivedFieldElement;

    /// `&[FieldElement<G>]` view over an archived `ArchivedVec<ArchivedFieldElement<G>>`.
    #[inline]
    fn archived_evals<G: IsField>(
        v: &rkyv::vec::ArchivedVec<ArchivedFieldElement<G>>,
    ) -> &[FieldElement<G>]
    where
        G::BaseType: rkyv::Archive,
    {
        ArchivedFieldElement::slice_as_native(v.as_slice())
    }

    #[inline]
    fn archived_polynomial_openings_ref<G: IsField>(
        p: &ArchivedPolynomialOpenings<G>,
    ) -> PolynomialOpeningsRef<'_, G>
    where
        G::BaseType: rkyv::Archive,
    {
        PolynomialOpeningsRef {
            proof: p.proof.merkle_path.as_slice(),
            proof_sym: p.proof_sym.merkle_path.as_slice(),
            evaluations: archived_evals(&p.evaluations),
            evaluations_sym: archived_evals(&p.evaluations_sym),
        }
    }

    impl<'a, F: IsSubFieldOf<E>, E: IsField, PI> StarkProofRef<'a, F, E, PI>
        for &'a ArchivedStarkProof<F, E, PI>
    where
        F::BaseType: rkyv::Archive,
        E::BaseType: rkyv::Archive,
        StarkProof<F, E, PI>: rkyv::Archive<Archived = ArchivedStarkProof<F, E, PI>>,
        PI: rkyv::Archive<Archived = PI>,
    {
        #[inline]
        fn trace_length(&self) -> usize {
            (*self).trace_length.to_native() as usize
        }
        #[inline]
        fn lde_trace_main_merkle_root(&self) -> &'a Commitment {
            &(*self).lde_trace_main_merkle_root
        }
        #[inline]
        fn lde_trace_aux_merkle_root(&self) -> Option<&'a Commitment> {
            (*self).lde_trace_aux_merkle_root.as_ref()
        }
        #[inline]
        fn lde_trace_precomputed_merkle_root(&self) -> Option<&'a Commitment> {
            (*self).lde_trace_precomputed_merkle_root.as_ref()
        }
        #[inline]
        fn trace_ood_evaluations(&self) -> OodTableRef<'a, E> {
            let t = &(*self).trace_ood_evaluations;
            OodTableRef::new(
                archived_evals(&t.data),
                t.width.to_native() as usize,
                t.height.to_native() as usize,
            )
        }
        #[inline]
        fn composition_poly_root(&self) -> &'a Commitment {
            &(*self).composition_poly_root
        }
        #[inline]
        fn composition_poly_parts_ood_evaluation(&self) -> &'a [FieldElement<E>] {
            archived_evals(&(*self).composition_poly_parts_ood_evaluation)
        }
        #[inline]
        fn fri_layers_merkle_roots(&self) -> &'a [Commitment] {
            (*self).fri_layers_merkle_roots.as_slice()
        }
        #[inline]
        fn fri_last_value(&self) -> &'a FieldElement<E> {
            (*self).fri_last_value.as_native()
        }
        #[inline]
        fn query_list_len(&self) -> usize {
            (*self).query_list.len()
        }
        #[inline]
        fn query(&self, i: usize) -> FriDecommitmentRef<'a, E> {
            let q = &(*self).query_list[i];
            FriDecommitmentRef {
                layer_paths: FriLayerPaths::Archived(q.layers_auth_paths.as_slice()),
                layers_evaluations_sym: archived_evals(&q.layers_evaluations_sym),
            }
        }
        #[inline]
        fn deep_poly_openings_len(&self) -> usize {
            (*self).deep_poly_openings.len()
        }
        fn deep_poly_opening(&self, i: usize) -> DeepPolynomialOpeningRef<'a, F, E> {
            let d = &(*self).deep_poly_openings[i];
            DeepPolynomialOpeningRef {
                composition_poly: archived_polynomial_openings_ref(&d.composition_poly),
                main_trace_polys: archived_polynomial_openings_ref(&d.main_trace_polys),
                precomputed_trace_polys: d
                    .precomputed_trace_polys
                    .as_ref()
                    .map(archived_polynomial_openings_ref),
                aux_trace_polys: d
                    .aux_trace_polys
                    .as_ref()
                    .map(archived_polynomial_openings_ref),
            }
        }
        #[inline]
        fn nonce(&self) -> Option<u64> {
            (*self).nonce.as_ref().map(|n| n.to_native())
        }
        #[inline]
        fn bus_table_contribution(&self) -> Option<&'a FieldElement<E>> {
            (*self)
                .bus_public_inputs
                .as_ref()
                .map(|bpi| bpi.table_contribution.as_native())
        }
        #[inline]
        fn has_bus_public_inputs(&self) -> bool {
            (*self).bus_public_inputs.is_some()
        }
        #[inline]
        fn public_inputs(&self) -> &'a PI {
            &(*self).public_inputs
        }
    }
}
