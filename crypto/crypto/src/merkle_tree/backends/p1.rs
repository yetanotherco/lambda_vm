//! ZisK's Poseidon1 Merkle backends (Goldilocks, width 16, rate 12; arity-4
//! trees) — the algebraic sibling of [`RpxVectorBackend`](super::rpx::RpxVectorBackend).
//!
//! The primitives are [`crate::hash::poseidon1_stark`] (checked against ZisK's
//! own code). What this module fixes is the encoding at the backend's seams,
//! the conventions [`super::rpx`] fixes for RPX:
//!
//! | seam | encoding |
//! |---|---|
//! | leaf | the elements' base felts in order ([`element_felts`]), then [`linear_hash`]: ZisK's chain with the width tag `[len, LEAF_DOMAIN, 0, 0]` as the first capacity |
//! | node | [`poseidon1_w16::compress4`] over four children; a node is four felts as 32 canonical big-endian bytes ([`digest_to_commitment`]) |
//! | padding child | the zero digest, i.e. 32 zero bytes |
//!
//! Nothing proves with these backends unless a caller names them; RPX stays
//! the default everywhere.

use core::marker::PhantomData;

use alloc::vec::Vec;
use math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};

use crate::hash::poseidon1_stark::linear_hash;
use crate::hash::poseidon1_w16;
use crate::hash::rpx::{
    Fp, commitment_to_digest, digest_to_commitment, element_felts, felts_from_bytes,
};
use crate::merkle_tree::traits::{IsLeafHasher, IsMerkleTreeBackend, IsStreamingLeafBackend};

/// ZisK's tagged leaf hash over felts, as a node.
pub fn leaf(felts: &[Fp]) -> [u8; 32] {
    digest_to_commitment(&linear_hash(felts))
}

/// ZisK's 4-ary node over four children.
pub fn node4(children: &[[u8; 32]; 4]) -> [u8; 32] {
    digest_to_commitment(&poseidon1_w16::compress4(
        &children.map(|c| commitment_to_digest(&c)),
    ))
}

/// The batched leaf backend: one leaf per vector of field elements.
#[derive(Clone, Debug)]
pub struct P1BatchBackend<F> {
    /// `fn() -> F`, so the marker is `Send` and `Sync` whatever `F` is.
    _marker: PhantomData<fn() -> F>,
}

impl<F> Default for P1BatchBackend<F> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

/// The pair backend: one leaf per fixed pair, the batched leaf of the
/// two-element vector.
#[derive(Clone, Debug)]
pub struct P1PairBackend<F> {
    _marker: PhantomData<fn() -> F>,
}

impl<F> Default for P1PairBackend<F> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<F> IsMerkleTreeBackend for P1BatchBackend<F>
where
    F: IsField + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    Vec<FieldElement<F>>: Sync + Send,
{
    type Node = [u8; 32];
    type Data = Vec<FieldElement<F>>;
    const ARITY: usize = 4;

    fn hash_data(input: &Vec<FieldElement<F>>) -> [u8; 32] {
        <Self as IsStreamingLeafBackend<F>>::hash_data_from_slices(input, &[])
    }

    /// A 4-ary tree has no binary parent; the two-child form is the 4-ary
    /// node over the pair and two padding digests (the shape of an odd-depth
    /// tree's top), so the method is defined everywhere.
    fn hash_new_parent(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        node4(&[*left, *right, [0u8; 32], [0u8; 32]])
    }

    fn hash_four(children: &[[u8; 32]; 4]) -> [u8; 32] {
        node4(children)
    }

    fn padding_node() -> Option<[u8; 32]> {
        Some([0u8; 32])
    }
}

impl<F> IsStreamingLeafBackend<F> for P1BatchBackend<F>
where
    F: IsField + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
    Vec<FieldElement<F>>: Sync + Send,
{
    /// Equals [`IsMerkleTreeBackend::hash_data`] on the elements `data`
    /// encodes (`tests::hash_bytes_agrees_with_hash_data`).
    fn hash_bytes(data: &[u8]) -> [u8; 32] {
        leaf(&felts_from_bytes(data))
    }

    fn hash_data_from_slices(a: &[FieldElement<F>], b: &[FieldElement<F>]) -> [u8; 32] {
        let mut felts = Vec::with_capacity(3 * (a.len() + b.len()));
        for e in a.iter().chain(b.iter()) {
            element_felts(e, &mut felts);
        }
        leaf(&felts)
    }

    type LeafHasher = P1LeafHasher<F>;

    fn leaf_hasher() -> Self::LeafHasher {
        P1LeafHasher {
            felts: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<F> IsMerkleTreeBackend for P1PairBackend<F>
where
    F: IsField + 'static,
    FieldElement<F>: AsBytes + Sync + Send,
{
    type Node = [u8; 32];
    type Data = [FieldElement<F>; 2];
    const ARITY: usize = 4;

    fn hash_data(input: &[FieldElement<F>; 2]) -> [u8; 32] {
        let mut felts = Vec::with_capacity(6);
        element_felts(&input[0], &mut felts);
        element_felts(&input[1], &mut felts);
        leaf(&felts)
    }

    /// As [`P1BatchBackend`]'s: the 4-ary node over the pair and two padding
    /// digests.
    fn hash_new_parent(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        node4(&[*left, *right, [0u8; 32], [0u8; 32]])
    }

    fn hash_four(children: &[[u8; 32]; 4]) -> [u8; 32] {
        node4(children)
    }

    fn padding_node() -> Option<[u8; 32]> {
        Some([0u8; 32])
    }
}

/// The incremental leaf hasher. The width tag needs the felt count, and the
/// row arrives in slices of field elements, so it buffers their felts.
pub struct P1LeafHasher<F> {
    felts: Vec<Fp>,
    _marker: PhantomData<fn() -> F>,
}

impl<F> IsLeafHasher<F> for P1LeafHasher<F>
where
    F: IsField,
    FieldElement<F>: AsBytes,
{
    type Node = [u8; 32];

    fn update(&mut self, data: &[FieldElement<F>]) {
        for e in data {
            element_felts(e, &mut self.felts);
        }
    }

    fn finalize(self) -> [u8; 32] {
        leaf(&self.felts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::poseidon1_stark::Merkle4;
    use crate::merkle_tree::merkle::MerkleTree;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Base;
    use math::field::traits::IsPrimeField;

    type B = P1BatchBackend<Base>;
    type E = P1BatchBackend<Ext3>;

    fn fe(i: u64) -> FieldElement<Base> {
        FieldElement::from(i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    fn fee(i: u64) -> FieldElement<Ext3> {
        FieldElement::<Ext3>::new([fe(3 * i), fe(3 * i + 1), fe(3 * i + 2)])
    }

    #[test]
    fn hash_bytes_agrees_with_hash_data() {
        for w in [1usize, 5, 12, 13, 40, 48, 64] {
            let row: Vec<FieldElement<Base>> = (0..w as u64).map(fe).collect();
            let bytes: Vec<u8> = row
                .iter()
                .flat_map(|f| Base::canonical(f.value()).to_be_bytes())
                .collect();
            assert_eq!(B::hash_bytes(&bytes), B::hash_data(&row), "width {w}");
            assert_eq!(B::hash_data(&row), leaf(&row));
        }
    }

    #[test]
    fn an_ext3_leaf_is_its_coefficients_in_order() {
        for w in [1usize, 2, 16] {
            let row: Vec<FieldElement<Ext3>> = (0..w as u64).map(fee).collect();
            let felts: Vec<Fp> = row.iter().flat_map(|e| e.value().to_vec()).collect();
            assert_eq!(E::hash_data(&row), leaf(&felts), "width {w}");
        }
    }

    #[test]
    fn the_leaf_hasher_is_the_batched_leaf() {
        let row: Vec<FieldElement<Ext3>> = (0..16u64).map(fee).collect();
        let mut h = E::leaf_hasher();
        h.update(&row[..5]);
        h.update(&row[5..]);
        assert_eq!(h.finalize(), E::hash_data(&row));
        assert_eq!(
            E::hash_data_from_slices(&row[..7], &row[7..]),
            E::hash_data(&row)
        );
    }

    #[test]
    fn the_pair_and_batched_families_agree_on_a_two_element_leaf() {
        // The univariate prover commits FRI layers with `Pair` and the
        // verifier authenticates them with `Batched`.
        for i in 0..8 {
            let (a, b) = (fee(2 * i), fee(2 * i + 1));
            assert_eq!(
                P1PairBackend::<Ext3>::hash_data(&[a, b]),
                E::hash_data(&vec![a, b])
            );
        }
    }

    #[test]
    fn a_tree_of_rows_is_the_oracles_tree() {
        for n in [1usize, 2, 8, 32, 128] {
            let rows: Vec<Vec<FieldElement<Base>>> = (0..n as u64)
                .map(|r| (0..7).map(|c| fe(r * 7 + c)).collect())
                .collect();
            let tree = MerkleTree::<B>::build(&rows).expect("non-empty");
            let leaves: Vec<_> = rows.iter().map(|r| linear_hash(r)).collect();
            let oracle = Merkle4::new(&leaves).expect("non-empty");
            assert_eq!(tree.root, digest_to_commitment(&oracle.root()), "{n} rows");
            let path = tree.get_proof_by_pos(n - 1).expect("in range");
            assert!(path.verify::<B>(&tree.root, n - 1, &rows[n - 1]));
        }
    }

    #[test]
    fn the_binary_parent_is_the_node_over_a_padded_pair() {
        let (a, b) = ([1u8; 32], [2u8; 32]);
        assert_eq!(
            B::hash_new_parent(&a, &b),
            B::hash_four(&[a, b, [0u8; 32], [0u8; 32]])
        );
    }
}
