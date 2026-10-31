//! The RPX256 Merkle backend — the algebraic sibling of
//! [`FieldElementVectorBackend`](super::field_element_vector::FieldElementVectorBackend).
//!
//! A sibling type rather than a reparameterisation of the byte backend, for the
//! reason the per-table branch gives: `FieldElementVectorBackend` is built
//! around a `digest::Digest` fed a byte stream, while this hashes FELTS with a
//! rate-8 overwrite duplex. Routing felts through bytes and back would work and
//! would be slower and less obvious; keeping them apart is what leaves the
//! keccak path untouched by this work.
//!
//! # What a leaf and a parent are
//!
//! **A leaf** is [`sponge_leaf`](crate::hash::rpx::sponge_leaf) over the felt
//! sequence the leaf's elements decompose to, in order — eight fresh felts per
//! permutation, capacity lane 0 the padding flag `len mod 8`, lane 1 the LEAF
//! domain.
//!
//! **A parent** is [`compress`](crate::hash::rpx::compress) — ONE permutation of
//! `[left ‖ right ‖ 0⁴]` at the zero domain, which makes it literally
//! `Rpx256::merge` and externally checkable against miden.
//!
//! **A node** is four canonical felts as 32 big-endian bytes, so
//! `IsMerkleTreeBackend::Node` is the same `[u8; 32]` every other backend in
//! this crate uses and no proof type changes width.
//!
//! # ⚠ The one contract a caller has to know
//!
//! `hash_data` on `&[a, b]` must equal what a FRI-style pair backend would
//! compute for the pair `[a, b]`, because the univariate prover commits layers
//! one way and verifies them the other. That invariant is why there is a single
//! implementation here rather than a "batched" and a "pair" one that could be
//! edited apart: both shapes go through [`hash_data_from_slices`], so there are
//! not two encodings to be shown equal. `tests::a_pair_leaf_is_the_two_element_vector`
//! pins it anyway, because "holds by construction" is a claim about today's
//! code.

use core::marker::PhantomData;

use alloc::vec::Vec;
use math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};

use crate::hash::rpx::{
    Fp, commitment_to_digest, compress, digest_to_commitment, element_felts, felts_from_bytes,
    sponge_leaf,
};
use crate::merkle_tree::traits::IsMerkleTreeBackend;

/// RPX256 over vectors of field elements.
#[derive(Clone, Debug)]
pub struct RpxVectorBackend<F> {
    /// `fn() -> F` rather than `F`, so the marker is unconditionally `Send` and
    /// `Sync` without an `unsafe impl`: a real epoch's base layer has millions
    /// of leaves, hashed in parallel.
    _marker: PhantomData<fn() -> F>,
}

impl<F> Default for RpxVectorBackend<F> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<F> RpxVectorBackend<F>
where
    F: IsField,
    FieldElement<F>: AsBytes,
{
    /// Leaf-hash the concatenation `a ‖ b` without materialising it.
    ///
    /// The single source of truth for the leaf's felt sequence: a plain leaf is
    /// the concatenation with an empty second slice, so the two shapes cannot
    /// disagree.
    pub fn hash_data_from_slices(a: &[FieldElement<F>], b: &[FieldElement<F>]) -> [u8; 32] {
        // Metric: a Merkle leaf finalize. `_direct` because this path builds no
        // `digest::Digest`, so it owes `total` as well — see `crate::hash_metrics`.
        crate::hash_metrics::count_merkle_direct();
        let mut felts: Vec<Fp> = Vec::with_capacity(a.len() + b.len());
        for e in a.iter().chain(b.iter()) {
            element_felts(e, &mut felts);
        }
        digest_to_commitment(&sponge_leaf(&felts))
    }

    /// Leaf-hash a byte buffer, rebuilding the felts it encodes.
    ///
    /// ⚠ Must equal [`hash_data`](IsMerkleTreeBackend::hash_data) on the
    /// elements those bytes encode — the one place an algebraic backend can
    /// silently disagree with itself, because the byte route has to rebuild
    /// what the felt route was handed.
    /// `tests::hash_bytes_agrees_with_hash_data` is the gate.
    pub fn hash_bytes(data: &[u8]) -> [u8; 32] {
        // Metric: a Merkle leaf finalize, as `hash_data_from_slices` is — the
        // two must agree on the digest, so they must agree on the count.
        crate::hash_metrics::count_merkle_direct();
        digest_to_commitment(&sponge_leaf(&felts_from_bytes(data)))
    }
}

impl<F> IsMerkleTreeBackend for RpxVectorBackend<F>
where
    F: IsField,
    FieldElement<F>: AsBytes + Sync + Send,
    Vec<FieldElement<F>>: Sync + Send,
{
    type Node = [u8; 32];
    type Data = Vec<FieldElement<F>>;

    fn hash_data(input: &Vec<FieldElement<F>>) -> [u8; 32] {
        Self::hash_data_from_slices(input, &[])
    }

    fn hash_new_parent(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        // Metric: a Merkle parent. One call, not a leaf count plus a node count:
        // this path does not flow through the leaf helper the way the byte
        // backend's parent flows through `hash_streamed`.
        crate::hash_metrics::count_merkle_node_direct();
        digest_to_commitment(&compress(
            &commitment_to_digest(left),
            &commitment_to_digest(right),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Base;

    type B = RpxVectorBackend<Base>;
    type E = RpxVectorBackend<Ext3>;

    fn base(n: usize) -> Vec<FieldElement<Base>> {
        (0..n as u64)
            .map(|i| FieldElement::from(i * 7 + 1))
            .collect()
    }

    fn ext(n: usize) -> Vec<FieldElement<Ext3>> {
        (0..n as u64)
            .map(|i| {
                FieldElement::<Ext3>::new([
                    FieldElement::from(i + 1),
                    FieldElement::from(i + 2),
                    FieldElement::from(i + 3),
                ])
            })
            .collect()
    }

    /// ★★ The `hash_bytes` / `hash_data` contract, on both fields and at the
    /// lengths the padding rule distinguishes.
    #[test]
    fn hash_bytes_agrees_with_hash_data() {
        for n in [0usize, 1, 7, 8, 9, 16, 17] {
            let leaf = base(n);
            let mut bytes = Vec::new();
            for e in &leaf {
                e.stream_bytes(&mut |b| bytes.extend_from_slice(b));
            }
            assert_eq!(B::hash_bytes(&bytes), B::hash_data(&leaf), "base, n = {n}");

            let leaf = ext(n);
            let mut bytes = Vec::new();
            for e in &leaf {
                e.stream_bytes(&mut |b| bytes.extend_from_slice(b));
            }
            assert_eq!(E::hash_bytes(&bytes), E::hash_data(&leaf), "ext3, n = {n}");
        }
    }

    /// ⚠ The invariant the univariate FRI path depends on: a two-element leaf
    /// is a pair.
    #[test]
    fn a_pair_leaf_is_the_two_element_vector() {
        let a = FieldElement::<Base>::from(11u64);
        let b = FieldElement::<Base>::from(22u64);
        assert_eq!(
            B::hash_data(&alloc::vec![a, b]),
            B::hash_data_from_slices(&[a], &[b])
        );
    }

    /// ✓ A leaf is order-sensitive and length-sensitive — so the equalities
    /// above are not equalities between constants.
    #[test]
    fn a_leaf_depends_on_the_order_and_the_length_of_its_elements() {
        let leaf = base(5);
        let mut swapped = leaf.clone();
        swapped.swap(0, 1);
        assert_ne!(B::hash_data(&leaf), B::hash_data(&swapped));

        let mut longer = leaf.clone();
        longer.push(FieldElement::from(0u64));
        assert_ne!(
            B::hash_data(&leaf),
            B::hash_data(&longer),
            "a trailing zero must not be invisible"
        );
    }

    /// ✓ An exact rate multiple spends no trailing permutation, so the eighth
    /// and ninth felts are not interchangeable at the block boundary.
    #[test]
    fn the_block_boundary_is_not_a_collision() {
        assert_ne!(B::hash_data(&base(8)), B::hash_data(&base(9)));
        assert_ne!(B::hash_data(&base(16)), B::hash_data(&base(17)));
    }

    /// ✓ A parent is order-sensitive.
    #[test]
    fn a_parent_depends_on_the_order_of_its_children() {
        let l = B::hash_data(&base(3));
        let r = B::hash_data(&base(4));
        assert_ne!(B::hash_new_parent(&l, &r), B::hash_new_parent(&r, &l));
    }

    /// ✓ A parent is NOT a leaf of the eight felts its children hold: the
    /// domains differ, which is the whole point of the capacity tag.
    #[test]
    fn a_parent_is_domain_separated_from_a_leaf() {
        use crate::hash::rpx::{commitment_to_digest, sponge_leaf};

        let l = B::hash_data(&base(3));
        let r = B::hash_data(&base(4));
        let parent = B::hash_new_parent(&l, &r);

        let mut felts = Vec::new();
        felts.extend_from_slice(&commitment_to_digest(&l));
        felts.extend_from_slice(&commitment_to_digest(&r));
        let as_leaf = digest_to_commitment(&sponge_leaf(&felts));

        assert_ne!(
            parent, as_leaf,
            "the LEAF domain must separate a leaf from a parent over the same felts"
        );
    }

    /// ✓ An extension leaf decomposes to three felts per element — checked
    /// against the felt sequence rather than against another backend call, so a
    /// decomposition that dropped a component would show.
    #[test]
    fn an_extension_leaf_absorbs_three_felts_per_element() {
        let leaf = ext(4);
        let mut felts = Vec::new();
        for e in &leaf {
            element_felts(e, &mut felts);
        }
        assert_eq!(felts.len(), 12, "four ext3 elements are twelve felts");
        assert_eq!(
            E::hash_data(&leaf),
            digest_to_commitment(&sponge_leaf(&felts))
        );
    }
}
