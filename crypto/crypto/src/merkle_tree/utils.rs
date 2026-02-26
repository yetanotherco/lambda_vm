use alloc::vec::Vec;

use super::traits::IsMerkleTreeBackend;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn sibling_index(node_index: usize) -> usize {
    if node_index.is_multiple_of(2) {
        node_index - 1
    } else {
        node_index + 1
    }
}

pub fn parent_index(node_index: usize) -> usize {
    if node_index.is_multiple_of(2) {
        (node_index - 1) / 2
    } else {
        node_index / 2
    }
}

/// Returns the sibling position for a given node index.
/// Returns `None` for the root node (index 0) since it has no sibling.
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

pub fn get_parent_pos(node_index: usize) -> usize {
    // Root node (index 0) has no parent, return itself to avoid underflow
    if node_index == 0 {
        return node_index;
    }
    if node_index.is_multiple_of(2) {
        (node_index - 1) / 2
    } else {
        node_index / 2
    }
}

// The list of values is completed repeating the last value to a power of two length
pub fn complete_until_power_of_two<T: Clone>(mut values: Vec<T>) -> Vec<T> {
    while !is_power_of_two(values.len()) {
        values.push(values[values.len() - 1].clone());
    }
    values
}

// ! NOTE !
// In this function we say 2^0 = 1 is a power of two.
// In turn, this makes the smallest tree of one leaf, possible.
// The function is private and is only used to ensure the tree
// has a power of 2 number of leaves.
fn is_power_of_two(x: usize) -> bool {
    (x & (x - 1)) == 0
}

// ! CAUTION !
// Make sure n=nodes.len()+1 is a power of two, and the last n/2 elements (leaves) are populated with hashes.
// This function takes no precautions for other cases.
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

// --- Arity-aware utilities ---

/// Parent index in an A-ary tree stored in a flat array (root at 0).
pub fn parent_index_arity(node_index: usize, arity: usize) -> usize {
    if node_index == 0 {
        return 0;
    }
    (node_index - 1) / arity
}

/// Indices of all siblings (same parent, excluding self). Empty for root.
pub fn sibling_indices_arity(node_index: usize, arity: usize) -> Vec<usize> {
    if node_index == 0 {
        return Vec::new();
    }
    let parent = parent_index_arity(node_index, arity);
    let first_child = parent * arity + 1;
    (first_child..first_child + arity)
        .filter(|&idx| idx != node_index)
        .collect()
}

/// Position of a node among its siblings (0..arity-1).
pub fn child_position(node_index: usize, arity: usize) -> usize {
    (node_index - 1) % arity
}

/// Total number of nodes in an A-ary tree with `num_leaves` leaves.
/// Formula: (A * L - 1) / (A - 1)
pub fn total_nodes_arity(num_leaves: usize, arity: usize) -> usize {
    (arity * num_leaves - 1) / (arity - 1)
}

/// Pad values to the next power of `arity`.
pub fn complete_until_power_of_arity<T: Clone>(mut values: Vec<T>, arity: usize) -> Vec<T> {
    let mut target = 1;
    while target < values.len() {
        target *= arity;
    }
    while values.len() < target {
        values.push(values[values.len() - 1].clone());
    }
    values
}

/// Build internal nodes of an A-ary tree. The flat array `nodes` has `total_nodes_arity(leaves_len, A)`
/// elements, with the last `leaves_len` positions already populated with leaf hashes.
/// Processes level by level from leaves to root, writing parents in-place.
pub fn build_arity<B: IsMerkleTreeBackend>(nodes: &mut [B::Node], leaves_len: usize)
where
    B::Node: Clone,
{
    let arity = B::ARITY;
    let total = nodes.len();

    let mut level_size = leaves_len;
    let mut level_start = total - leaves_len;

    while level_size > 1 {
        let parent_size = level_size / arity;
        let parent_start = level_start - parent_size;

        let (parents_slice, children_area) =
            nodes[parent_start..level_start + level_size].split_at_mut(parent_size);

        #[cfg(feature = "parallel")]
        let iter = parents_slice
            .into_par_iter()
            .zip(children_area.par_chunks_exact(arity));
        #[cfg(not(feature = "parallel"))]
        let iter = parents_slice
            .iter_mut()
            .zip(children_area.chunks_exact(arity));

        iter.for_each(|(parent, children)| {
            *parent = B::hash_children(children);
        });

        level_size = parent_size;
        level_start = parent_start;
    }
}
