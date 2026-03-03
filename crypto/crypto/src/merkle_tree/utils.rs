use alloc::vec::Vec;

use super::traits::IsMerkleTreeBackend;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Returns the parent index for a node in a tree of given arity stored in a flat array.
/// The root is at index 0. For any non-root node i, the parent is at (i-1)/arity.
pub fn parent_index(node_index: usize, arity: usize) -> usize {
    debug_assert!(node_index > 0, "root node has no parent");
    (node_index - 1) / arity
}

/// Returns the position (0..arity-1) of a node among its siblings.
pub fn child_position(node_index: usize, arity: usize) -> usize {
    debug_assert!(node_index > 0, "root node has no child position");
    (node_index - 1) % arity
}

/// Returns the indices of the sibling nodes that share the same parent.
/// The returned vector excludes `node_index` itself and has length `arity - 1`.
pub fn sibling_indices(node_index: usize, arity: usize) -> Vec<usize> {
    debug_assert!(node_index > 0, "root node has no siblings");
    let pos = child_position(node_index, arity);
    let first_child = node_index - pos;
    let mut result = Vec::with_capacity(arity - 1);
    for i in 0..arity {
        if i != pos {
            result.push(first_child + i);
        }
    }
    result
}

/// Pads a list of values to the next power of `arity` by repeating the last value.
pub fn complete_until_power_of_arity<T: Clone>(mut values: Vec<T>, arity: usize) -> Vec<T> {
    while !is_power_of(values.len(), arity) {
        values.push(values[values.len() - 1].clone());
    }
    values
}

/// Returns true if x is a power of `base` (including base^0 = 1).
pub fn is_power_of(x: usize, base: usize) -> bool {
    if x == 0 {
        return false;
    }
    let mut v = x;
    while v > 1 {
        if v % base != 0 {
            return false;
        }
        v /= base;
    }
    true
}

/// Rounds `n` up to the next power of `base` (including base^0 = 1).
pub fn next_power_of(n: usize, base: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p *= base;
    }
    p
}

/// For arity-2: internal_nodes = leaves - 1
/// For arity-4: internal_nodes = (leaves - 1) / 3
/// General formula: internal_nodes = (leaves - 1) / (arity - 1)
pub fn internal_node_count(leaves: usize, arity: usize) -> usize {
    (leaves - 1) / (arity - 1)
}

/// From total nodes, compute the number of leaves.
/// total = (arity * leaves - 1) / (arity - 1)
/// => leaves = ((arity - 1) * total + 1) / arity
pub fn num_leaves_from_total(total: usize, arity: usize) -> usize {
    ((arity - 1) * total + 1) / arity
}

/// Builds the internal nodes of a Merkle tree in-place.
///
/// `nodes` is a flat array. The last `leaves_len` entries are already populated
/// with leaf hashes. This function fills the internal nodes from bottom to top,
/// grouping children in chunks of `B::ARITY`.
pub fn build<B: IsMerkleTreeBackend>(nodes: &mut [B::Node], leaves_len: usize)
where
    B::Node: Clone,
{
    let arity = B::ARITY;
    let total = nodes.len();
    let mut level_begin = total - leaves_len;
    let mut level_end = total;

    while level_begin > 0 {
        let level_len = level_end - level_begin;
        let parent_level_len = level_len / arity;
        let parent_level_begin = level_begin - parent_level_len;

        let (parent_slice, children_slice) =
            nodes[parent_level_begin..level_end].split_at_mut(parent_level_len);

        #[cfg(feature = "parallel")]
        let parent_and_children_iter = parent_slice
            .into_par_iter()
            .zip(children_slice.par_chunks_exact(arity));
        #[cfg(not(feature = "parallel"))]
        let parent_and_children_iter = parent_slice
            .iter_mut()
            .zip(children_slice.chunks_exact(arity));

        parent_and_children_iter.for_each(|(parent, children)| {
            *parent = B::hash_new_parent(children);
        });

        level_end = level_begin;
        level_begin = parent_level_begin;
    }
}
