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

/// Level sizes of an arity-4 tree over `leaves_len ≥ 1` leaves, leaves first,
/// root (size 1) last: each level holds `⌈below / 4⌉` nodes. Only real nodes
/// are counted; a level whose length is not a multiple of four is padded with
/// the backend's padding digest when its parents are hashed, and those
/// padding children are never stored.
pub fn level_sizes4(leaves_len: usize) -> Vec<usize> {
    let mut sizes = vec![leaves_len];
    while let Some(&n) = sizes.last().filter(|&&n| n > 1) {
        sizes.push(n.div_ceil(4));
    }
    sizes
}

/// Where each level of [`level_sizes4`] starts in an arity-4 tree's node
/// vector, which stores the levels top-down (root at 0, leaves last).
pub fn level_offsets4(sizes: &[usize]) -> Vec<usize> {
    let mut offsets = vec![0; sizes.len()];
    for j in (0..sizes.len().saturating_sub(1)).rev() {
        offsets[j] = offsets[j + 1] + sizes[j + 1];
    }
    offsets
}

/// The leaf count of an arity-4 tree with `total` stored nodes, if `total` is
/// the node count of a tree over a power-of-two number of leaves. The count
/// is strictly increasing in the leaf count, so the answer is unique.
pub fn leaves_len4(total: usize) -> Option<usize> {
    let mut leaves = 1usize;
    loop {
        let n: usize = level_sizes4(leaves).iter().sum();
        if n == total {
            return Some(leaves);
        }
        if n > total {
            return None;
        }
        leaves = leaves.checked_mul(2)?;
    }
}

/// Builds every inner level of an arity-4 tree in place: `nodes` holds the
/// levels top-down per [`level_offsets4`], the leaf level already filled.
///
/// `None` (nothing built) when a level needs padding and the backend has no
/// padding digest ([`IsMerkleTreeBackend::padding_node`]).
pub fn build4<B: IsMerkleTreeBackend>(nodes: &mut [B::Node], leaves_len: usize) -> Option<()>
where
    B::Node: Clone,
{
    let sizes = level_sizes4(leaves_len);
    let offsets = level_offsets4(&sizes);
    // A level of `n > 1` nodes not a multiple of four ends in a short group,
    // which takes the padding digest; with no short group the placeholder is
    // never read.
    let pad = match B::padding_node() {
        Some(pad) => pad,
        None if sizes.iter().any(|&n| n > 1 && !n.is_multiple_of(4)) => return None,
        None => nodes.first()?.clone(),
    };
    for j in 0..sizes.len() - 1 {
        let (above, below) = nodes.split_at_mut(offsets[j]);
        let children = &below[..sizes[j]];
        let parents = &mut above[offsets[j + 1]..offsets[j + 1] + sizes[j + 1]];
        let parent_of = |p: usize| {
            let child = |c: usize| {
                children
                    .get(4 * p + c)
                    .cloned()
                    .unwrap_or_else(|| pad.clone())
            };
            B::hash_four(&[child(0), child(1), child(2), child(3)])
        };
        #[cfg(feature = "parallel")]
        parents
            .par_iter_mut()
            .enumerate()
            .for_each(|(p, out)| *out = parent_of(p));
        #[cfg(not(feature = "parallel"))]
        parents
            .iter_mut()
            .enumerate()
            .for_each(|(p, out)| *out = parent_of(p));
    }
    Some(())
}
