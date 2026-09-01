//! DEGREE-LANE EXPERIMENT (temporary, not for merge).
//!
//! Process-wide counters for Merkle hash work, behind the `hash-count`
//! feature. Every leaf hash in the tree backends funnels through
//! `hash_streamed` and every parent through `hash_new_parent_bytes`, so
//! instrumenting those two chokepoints captures all of it.
//!
//! Leaves are counted with their absorbed byte length because leaf width is
//! exactly where composition-polynomial part count shows up: a query opens one
//! leaf holding `2 * parts` extension elements. Parents are fixed-shape
//! 64-byte compressions and are counted by number alone.
//!
//! Counting is deterministic, so a single run is an exact reading — no
//! replicates. Never enable this feature for a timing run: the atomics sit in
//! the hot hashing path.

use core::sync::atomic::{AtomicU64, Ordering};

pub static LEAF_HASHES: AtomicU64 = AtomicU64::new(0);
pub static LEAF_BYTES: AtomicU64 = AtomicU64::new(0);
pub static PARENT_HASHES: AtomicU64 = AtomicU64::new(0);
/// Keccak-f permutations, the quantity an in-circuit verifier actually pays
/// (the guest's keccak accelerator ecall count). Absorbing `L` bytes at rate
/// `KECCAK_RATE` costs `ceil((L+1)/RATE)` permutations — the +1 is pad10*1,
/// which needs at least one byte. This is why extra composition parts can be
/// free: they widen a leaf, but cost nothing until the width crosses a block
/// boundary.
pub static PERMUTATIONS: AtomicU64 = AtomicU64::new(0);

/// Keccak-256 sponge rate: 200 - 2*32 bytes.
const KECCAK_RATE: u64 = 136;

#[inline]
fn permutations_for(bytes: u64) -> u64 {
    bytes / KECCAK_RATE + 1
}

#[inline]
pub fn record_leaf(bytes: usize) {
    LEAF_HASHES.fetch_add(1, Ordering::Relaxed);
    LEAF_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    PERMUTATIONS.fetch_add(permutations_for(bytes as u64), Ordering::Relaxed);
}

#[inline]
pub fn record_parent() {
    PARENT_HASHES.fetch_add(1, Ordering::Relaxed);
    // Always exactly two 32-byte nodes: 64 bytes, one permutation.
    PERMUTATIONS.fetch_add(1, Ordering::Relaxed);
}

/// `(leaf_hashes, leaf_bytes, parent_hashes, permutations)`.
pub fn read() -> (u64, u64, u64, u64) {
    (
        LEAF_HASHES.load(Ordering::Relaxed),
        LEAF_BYTES.load(Ordering::Relaxed),
        PARENT_HASHES.load(Ordering::Relaxed),
        PERMUTATIONS.load(Ordering::Relaxed),
    )
}

pub fn reset() {
    LEAF_HASHES.store(0, Ordering::Relaxed);
    LEAF_BYTES.store(0, Ordering::Relaxed);
    PARENT_HASHES.store(0, Ordering::Relaxed);
    PERMUTATIONS.store(0, Ordering::Relaxed);
}
