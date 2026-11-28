//! ⛔ MEASUREMENT ONLY, on no proving path: the card seconds of a WHIR tree's
//! leaf pass and inner levels, and of one proof-of-work search, under each
//! device hash — a microbenchmark of BLAKE3 against Poseidon1 at
//! the base proof's shapes. The kernels timed are the ones the prover
//! launches (`crate::whir::coset_leaf_kernel`, `crate::whir::build_inner_tree_levels`,
//! `crate::grinding`'s search), over a codeword of pseudo-random canonical
//! values the card writes itself.

use std::time::Instant;

use cudarc::driver::PushKernelArg;

use crate::Result;
use crate::device::{alloc_or_trim, backend};

/// One tree shape's card seconds, the median of the timed repetitions.
#[derive(Clone, Copy, Debug)]
pub struct TreeTiming {
    /// The leaf pass: one coset leaf of `2^log_folding` values per leaf.
    pub leaves_secs: f64,
    /// Every inner level above the leaves (`leaves − 1` parents).
    pub inner_secs: f64,
    pub num_leaves: usize,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// The leaf pass and the inner levels of a tree over a codeword of
/// `2^log_elements` values (base, or ext3 when `!base`) at `2^log_folding`
/// values a leaf, under `hash`, each the median of `reps` timed runs after
/// one untimed one. `reversed`: the base leaves in the coset leaf order
/// (Poseidon1 and BLAKE3 only).
pub fn tree(
    hash: crate::DeviceHash,
    base: bool,
    reversed: bool,
    log_elements: usize,
    log_folding: usize,
    reps: usize,
) -> Result<TreeTiming> {
    let be = backend()?;
    let stream = be.next_stream();
    let limbs = if base { 1 } else { 3 };
    let elements = 1usize << log_elements;
    let num_leaves = elements >> log_folding;
    // A Weyl sequence of canonical values, written by the host once.
    let host: Vec<u64> = (0..elements * limbs)
        .map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) % 0xFFFF_FFFF_0000_0001)
        .collect();
    let codeword = stream.clone_htod(&host)?;
    drop(host);
    let total_nodes = crate::tree_nodes(hash, num_leaves);
    // SAFETY: every byte read below is written first by the leaf pass or
    // the levels.
    let mut nodes = unsafe { alloc_or_trim::<u8>(&stream, total_nodes * 32) }?;
    let leaves_offset = crate::leaves_offset(hash, num_leaves) * 32;
    let num_leaves_u64 = num_leaves as u64;
    let block = 1u64 << log_folding;
    let leaf_bits = num_leaves.trailing_zeros();
    let (mut leaf_times, mut inner_times) = (Vec::new(), Vec::new());
    for rep in 0..=reps {
        let t = Instant::now();
        {
            let mut leaves = nodes.slice_mut(leaves_offset..leaves_offset + num_leaves * 32);
            if reversed {
                let (kernel, cfg) = match hash {
                    crate::DeviceHash::Poseidon1 => (
                        &crate::p1_stark::kernels()?.leaves_base_coset_rev,
                        crate::p1_stark::launch_cfg(num_leaves_u64),
                    ),
                    crate::DeviceHash::Blake3 => (
                        &crate::blake3_whir::kernels()?.leaves_base_coset_rev,
                        crate::blake3_whir::launch_cfg(num_leaves_u64),
                    ),
                    _ => return Err(crate::invalid_value()),
                };
                unsafe {
                    stream
                        .launch_builder(kernel)
                        .arg(&codeword)
                        .arg(&num_leaves_u64)
                        .arg(&block)
                        .arg(&leaf_bits)
                        .arg(&mut leaves)
                        .launch(cfg)?;
                }
            } else {
                let (kernel, cfg) = crate::whir::coset_leaf_kernel(be, hash, base, num_leaves_u64)?;
                unsafe {
                    stream
                        .launch_builder(kernel)
                        .arg(&codeword)
                        .arg(&num_leaves_u64)
                        .arg(&block)
                        .arg(&mut leaves)
                        .launch(cfg)?;
                }
            }
        }
        stream.synchronize()?;
        let leaves_secs = t.elapsed().as_secs_f64();
        let t = Instant::now();
        crate::whir::build_inner_tree_levels(stream.as_ref(), be, &mut nodes, num_leaves, hash)?;
        stream.synchronize()?;
        let inner_secs = t.elapsed().as_secs_f64();
        if rep > 0 {
            leaf_times.push(leaves_secs);
            inner_times.push(inner_secs);
        }
    }
    Ok(TreeTiming {
        leaves_secs: median(leaf_times),
        inner_secs: median(inner_times),
        num_leaves,
    })
}

/// Seconds one proof-of-work search at `factor` bits takes on the card, the
/// median over `seeds` inner hashes (canonical felts, so every arm reads them),
/// at the grind knobs in effect. `None` when the card declines a search.
pub fn grind(hash: crate::DeviceHash, factor: u8, seeds: usize) -> Option<f64> {
    let mut times = Vec::with_capacity(seeds);
    for s in 0..seeds as u64 {
        let inner: [u64; 4] = core::array::from_fn(|i| {
            (s * 4 + i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) % 0xFFFF_FFFF_0000_0001
        });
        let t = Instant::now();
        match hash {
            crate::DeviceHash::Blake3 => crate::grinding::generate_nonce_blake3_gpu(&inner, factor),
            crate::DeviceHash::Poseidon1 => crate::grinding::generate_nonce_p1_gpu(&inner, factor),
            crate::DeviceHash::Keccak256 => crate::grinding::generate_nonce_gpu(&inner, factor),
            _ => None,
        }?;
        times.push(t.elapsed().as_secs_f64());
    }
    Some(median(times))
}
