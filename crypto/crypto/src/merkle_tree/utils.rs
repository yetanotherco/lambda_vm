use alloc::vec::Vec;

use super::traits::IsMerkleTreeBackend;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn parent_index(node_index: usize, arity: usize) -> usize {
    (node_index - 1) / arity
}

pub fn child_position(node_index: usize, arity: usize) -> usize {
    (node_index - 1) % arity
}

/// Returns the indices of all siblings of the given node (arity - 1 siblings).
pub fn sibling_indices(node_index: usize, arity: usize) -> Vec<usize> {
    let parent = parent_index(node_index, arity);
    let first_child = parent * arity + 1;
    (first_child..first_child + arity)
        .filter(|&i| i != node_index)
        .collect()
}

pub fn complete_until_power_of_arity<T: Clone>(mut values: Vec<T>, arity: usize) -> Vec<T> {
    if values.len() <= 1 {
        return values;
    }
    // Compute target length as the next power of arity >= values.len()
    let mut target = 1;
    while target < values.len() {
        target *= arity;
    }
    if target > values.len() {
        let pad = values.last().unwrap().clone();
        values.resize(target, pad);
    }
    values
}

pub fn internal_node_count(leaves: usize, arity: usize) -> usize {
    (leaves - 1) / (arity - 1)
}

pub fn num_leaves_from_total(total: usize, arity: usize) -> usize {
    ((arity - 1) * total + 1) / arity
}

/// Compute tree depth = log_arity(num_leaves)
pub fn compute_depth(num_leaves: usize, arity: usize) -> usize {
    if num_leaves <= 1 {
        return 0;
    }
    let mut depth = 0;
    let mut n = num_leaves;
    while n > 1 {
        n /= arity;
        depth += 1;
    }
    depth
}

pub fn build<B: IsMerkleTreeBackend>(nodes: &mut [B::Node], leaves_len: usize)
where
    B::Node: Clone,
{
    let arity = B::ARITY;
    let internal = internal_node_count(leaves_len, arity);
    // Build bottom-up, level by level
    // Leaves are at indices [internal..internal+leaves_len)
    // Parents at [(internal - parent_level_size)..internal)
    let mut level_start = internal; // start of current children level
    let mut level_size = leaves_len;

    while level_size > 1 {
        let parent_level_size = level_size / arity;
        let parent_start = level_start - parent_level_size;

        let (parents, children) =
            nodes[parent_start..level_start + level_size].split_at_mut(parent_level_size);

        #[cfg(feature = "parallel")]
        let iter = parents
            .into_par_iter()
            .zip(children.par_chunks_exact(arity));
        #[cfg(not(feature = "parallel"))]
        let iter = parents.iter_mut().zip(children.chunks_exact(arity));

        iter.for_each(|(parent, chunk)| {
            *parent = B::hash_new_parent(chunk);
        });

        level_size = parent_level_size;
        level_start = parent_start;
    }
}
