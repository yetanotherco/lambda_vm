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

        #[cfg(feature = "parallel")]
        {
            let (new_level_iter, children_iter) =
                nodes[new_level_begin_index..level_end_index + 1].split_at_mut(new_level_length);
            new_level_iter
                .into_par_iter()
                .zip(children_iter.par_chunks_exact(2))
                .for_each(|(new_parent, children)| {
                    *new_parent = B::hash_new_parent(&children[0], &children[1]);
                });
        }
        // Sequential path as an index loop (not `split_at_mut` +
        // `iter_mut().zip(chunks_exact)`) so the Rust->Lean extractors don't have
        // to model the `&mut` slice split or the `Zip`/`ChunksExact` iterator
        // adapters. Parents occupy `[new_level_begin_index, level_begin_index)`,
        // strictly below the children read at `level_begin_index + 2k (+1)`, so
        // the in-place writes never clobber a child still to be read.
        #[cfg(not(feature = "parallel"))]
        for k in 0..new_level_length {
            let child0 = level_begin_index + 2 * k;
            let parent = B::hash_new_parent(&nodes[child0], &nodes[child0 + 1]);
            nodes[new_level_begin_index + k] = parent;
        }

        level_end_index = level_begin_index - 1;
        level_begin_index = new_level_begin_index;
    }
}
