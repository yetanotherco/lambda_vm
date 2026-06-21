use alloc::vec::Vec;

use super::traits::IsMerkleTreeBackend;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// =========================================================================
// Flat-array index arithmetic for an `arity`-ary complete tree.
//
// Layout (matches the binary case when `arity == 2`): node 0 is the root, and
// the children of node `i` are `arity*i + 1 ..= arity*i + arity`. The parent of a
// non-root node `i` is `(i - 1) / arity`, and `i`'s slot among its siblings is
// `(i - 1) % arity`. A node `i`'s sibling group is the `arity` consecutive nodes
// `parent*arity + 1 ..= parent*arity + arity`.
// =========================================================================

/// Parent index of a non-root node in an `arity`-ary tree.
#[inline]
pub fn parent_index_arity(node_index: usize, arity: usize) -> usize {
    (node_index - 1) / arity
}

/// Parent index of `node_index`; the root (index 0) returns itself to avoid
/// underflow (matching the historical `get_parent_pos` contract).
#[inline]
pub fn get_parent_pos_arity(node_index: usize, arity: usize) -> usize {
    if node_index == 0 {
        return node_index;
    }
    parent_index_arity(node_index, arity)
}

/// The `arity` children indices of an internal node `parent`, in order.
#[inline]
pub fn children_indices(parent: usize, arity: usize) -> impl Iterator<Item = usize> {
    (arity * parent + 1)..=(arity * parent + arity)
}

/// The sibling indices of `node_index` (the other `arity - 1` children of its
/// parent), in ascending order. Empty for the root.
#[inline]
pub fn sibling_indices(node_index: usize, arity: usize) -> Vec<usize> {
    if node_index == 0 {
        return Vec::new();
    }
    let parent = parent_index_arity(node_index, arity);
    children_indices(parent, arity)
        .filter(|&c| c != node_index)
        .collect()
}

// The list of values is completed repeating the last value to a power-of-`arity`
// length. `arity == 2` reproduces the historical power-of-two padding.
pub fn complete_until_power_of_arity<T: Clone>(mut values: Vec<T>, arity: usize) -> Vec<T> {
    while !is_power_of(values.len(), arity) {
        values.push(values[values.len() - 1].clone());
    }
    values
}

// ! NOTE !
// `x == 1` (arity^0) counts as a power, so the smallest tree (one leaf) is
// possible. Private; only used to pad the leaf count to a power of `arity`.
fn is_power_of(mut x: usize, arity: usize) -> bool {
    if x == 0 {
        return false;
    }
    while x.is_multiple_of(arity) {
        x /= arity;
    }
    x == 1
}

// ! CAUTION !
// Requires `leaves_len` to be a power of `B::ARITY`, the node buffer sized to
// `(leaves_len - 1) / (ARITY - 1) + leaves_len` total nodes, with the trailing
// `leaves_len` entries populated with the leaf hashes. Builds the inner nodes
// bottom-up, hashing each group of `ARITY` consecutive children into their
// parent. Takes no precautions for other cases.
pub fn build<B: IsMerkleTreeBackend>(nodes: &mut [B::Node], leaves_len: usize)
where
    B::Node: Clone,
{
    let arity = B::ARITY;
    // Number of inner nodes in an arity-ary complete tree with `leaves_len`
    // leaves is (leaves_len - 1) / (arity - 1). The leaf level begins at that
    // index; the level just processed spans [level_begin, level_end].
    let mut level_begin_index = (leaves_len - 1) / (arity - 1);
    let mut level_end_index = level_begin_index + leaves_len - 1;
    while level_begin_index != 0 {
        // Parent level indices: each parent at `p` hashes children
        // `arity*p+1 ..= arity*p+arity`. The parents of the current level
        // [level_begin, level_end] occupy [(level_begin-1)/arity, (level_end-1)/arity].
        let new_level_begin_index = (level_begin_index - 1) / arity;
        let new_level_length = level_begin_index - new_level_begin_index;

        let (new_level_iter, children_iter) =
            nodes[new_level_begin_index..level_end_index + 1].split_at_mut(new_level_length);

        #[cfg(feature = "parallel")]
        let parent_and_children_zipped_iter = new_level_iter
            .into_par_iter()
            .zip(children_iter.par_chunks_exact(arity));
        #[cfg(not(feature = "parallel"))]
        let parent_and_children_zipped_iter = new_level_iter
            .iter_mut()
            .zip(children_iter.chunks_exact(arity));

        parent_and_children_zipped_iter.for_each(|(new_parent, children)| {
            *new_parent = B::hash_children(children);
        });

        level_end_index = level_begin_index - 1;
        level_begin_index = new_level_begin_index;
    }
}
