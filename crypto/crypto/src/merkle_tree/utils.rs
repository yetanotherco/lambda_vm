use super::traits::IsMerkleTreeBackend;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Sibling of `node_index` in the flat node array, or `None` for the root (0).
pub fn get_sibling_pos(node_index: usize) -> Option<usize> {
    if node_index == 0 {
        return None;
    }
    if node_index.is_multiple_of(2) {
        Some(node_index - 1)
    } else {
        Some(node_index + 1)
    }
}

/// Parent of `node_index`. The root (0) has no parent and maps to itself.
pub fn get_parent_pos(node_index: usize) -> usize {
    if node_index == 0 {
        return node_index;
    }
    if node_index.is_multiple_of(2) {
        (node_index - 1) / 2
    } else {
        node_index / 2
    }
}

/// Fills the inner nodes of a Merkle tree bottom-up.
///
/// Precondition (caller-enforced, not checked): `nodes.len() + 1` is a power of
/// two and the last `leaves_len` entries of `nodes` already hold the leaf
/// hashes. Behaviour is unspecified for any other input.
pub fn build<B: IsMerkleTreeBackend>(nodes: &mut [B::Node], leaves_len: usize)
where
    B::Node: Clone,
{
    let mut level_begin_index = leaves_len - 1;
    let mut level_end_index = 2 * level_begin_index;
    while level_begin_index != level_end_index {
        let new_level_begin_index = level_begin_index / 2;
        let new_level_length = level_begin_index - new_level_begin_index;

        let (new_level_iter, children_iter) =
            nodes[new_level_begin_index..level_end_index + 1].split_at_mut(new_level_length);

        #[cfg(feature = "parallel")]
        let parent_and_children_zipped_iter = new_level_iter
            .into_par_iter()
            .zip(children_iter.par_chunks_exact(2));
        #[cfg(not(feature = "parallel"))]
        let parent_and_children_zipped_iter =
            new_level_iter.iter_mut().zip(children_iter.chunks_exact(2));

        parent_and_children_zipped_iter.for_each(|(new_parent, children)| {
            *new_parent = B::hash_new_parent(&children[0], &children[1]);
        });

        level_end_index = level_begin_index - 1;
        level_begin_index = new_level_begin_index;
    }
}
