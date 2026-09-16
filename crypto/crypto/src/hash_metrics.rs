//! Host-only hash counters for measuring the cost of VERIFYING a proof — a
//! proxy for a recursive verifier's dominant work, which is hashing.
//!
//! ⚠ **The counters follow the proof's CONFIGURATION, not one hash.** They were
//! keccak-only when keccak was the only hash on the multilinear path. Under
//! [`crate::hash::rpx`] every Merkle counter would then have read ZERO — a
//! measurement that cannot fail, reporting "no hashing" for the arm whose whole
//! purpose is to change the hashing. The algebraic backend bumps them through
//! [`count_merkle_direct`] / [`count_merkle_node_direct`] instead, which also
//! bump `total`, because unlike the byte backends nothing downstream will.
//!
//! Behind the `hash-metrics` cargo feature: a normal build keeps
//! `PlatformKeccak256 = sha3::Keccak256` and every counter call compiles to
//! nothing, so the prover is provably unchanged. With the feature on (host only),
//! the host `PlatformKeccak256` wrapper counts, per keccak op:
//! * `total` — every finalize (leaf / node / transcript squeeze / program-id fold);
//!   `merkle`/`merkle_nodes`/`grinding` split it (keccak-only, disjoint subsets);
//! * `absorb_calls` / `absorb_bytes` — every `Update::update` (ABSORPTION). This is
//!   the guest's DOMINANT keccak cost — the many 8-byte `stream_bytes` absorbs in
//!   opening verification, not the finalize — so it is the dimension a block-
//!   absorption optimization moves. A finalize-only number would report such a
//!   change as zero improvement; `absorb_*` is what makes it visible.
//!
//! No enable/disable toggle and nothing in the verifier: counting is always on
//! under the feature, and a measuring caller just [`reset`]s before the verify
//! and reads [`snapshot`] after. Grinding is separated by counter, not excluded
//! at a call site, so there is no cross-thread race under a parallel verify.

/// Snapshot of the verify-hash counters (all zero without the `hash-metrics`
/// feature / on the guest). `total` is finalizes; `absorb_*` is absorption.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// Every keccak-256 finalize.
    pub total: u64,
    /// Merkle finalizes (keccak-guarded subset of `total`).
    pub merkle: u64,
    /// Merkle auth-path (parent) compressions (subset of `merkle`).
    pub merkle_nodes: u64,
    /// Grinding proof-of-work finalizes (subset of `total`).
    pub grinding: u64,
    /// Keccak absorb (`Update::update`) invocations.
    pub absorb_calls: u64,
    /// Bytes fed through absorb (`Sum of data.len()`).
    pub absorb_bytes: u64,
}

#[cfg(all(not(target_arch = "riscv64"), feature = "hash-metrics"))]
mod imp {
    use super::Counts;
    use core::sync::atomic::{AtomicU64, Ordering};

    static TOTAL: AtomicU64 = AtomicU64::new(0);
    static MERKLE: AtomicU64 = AtomicU64::new(0);
    static MERKLE_NODES: AtomicU64 = AtomicU64::new(0);
    static GRINDING: AtomicU64 = AtomicU64::new(0);
    static ABSORB_CALLS: AtomicU64 = AtomicU64::new(0);
    static ABSORB_BYTES: AtomicU64 = AtomicU64::new(0);

    /// Every keccak-256 finalize, from any site (host `PlatformKeccak256`).
    #[inline(always)]
    pub fn count_total() {
        TOTAL.fetch_add(1, Ordering::Relaxed);
    }

    /// A Merkle finalize (leaf or node), counted ONLY when the backend digest is
    /// the platform keccak wrapper — the one whose `finalize` also bumps
    /// [`count_total`]. This keeps `merkle` a strict subset of `total` for ANY
    /// `D` (a non-keccak backend, as in the crypto tests, does not go through the
    /// counted wrapper, so counting it here would let `merkle` exceed `total`).
    ///
    /// A hash that does not route its Merkle work through a `digest::Digest` at
    /// all — the algebraic backend sponges felts directly — uses
    /// [`count_merkle_direct`] instead, which keeps the same invariant by
    /// bumping both counters itself.
    #[inline(always)]
    pub fn count_merkle<D: 'static>() {
        if core::any::TypeId::of::<D>()
            == core::any::TypeId::of::<crate::hash::platform_keccak::PlatformKeccak256>()
        {
            MERKLE.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A Merkle parent (auth-path) compression. Subset of [`count_merkle`];
    /// same keccak-only guard.
    #[inline(always)]
    pub fn count_merkle_node<D: 'static>() {
        if core::any::TypeId::of::<D>()
            == core::any::TypeId::of::<crate::hash::platform_keccak::PlatformKeccak256>()
        {
            MERKLE_NODES.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A grinding (proof-of-work) finalize. Subset of [`count_total`]; a caller
    /// reports `total - grinding` to exclude the PoW check.
    #[inline(always)]
    pub fn count_grinding() {
        GRINDING.fetch_add(1, Ordering::Relaxed);
    }

    /// A keccak absorb (`Update::update`) of `nbytes` — the guest's dominant
    /// keccak cost, and the dimension a block-absorption optimization moves
    /// (finalizes do not change). Bumps the call count and the byte total.
    #[inline(always)]
    pub fn count_absorb(nbytes: usize) {
        ABSORB_CALLS.fetch_add(1, Ordering::Relaxed);
        ABSORB_BYTES.fetch_add(nbytes as u64, Ordering::Relaxed);
    }

    /// ★ A Merkle LEAF finalize by a hash whose Merkle work does not pass
    /// through a `digest::Digest` — the algebraic backend, which sponges felts
    /// directly and never builds a digest object.
    ///
    /// Bumps `total` as well as `merkle`, because nothing downstream will: for
    /// the byte backends `total` comes from the digest's own `finalize`, and
    /// there is no such call here. Doing both in one function is what keeps
    /// `merkle ⊆ total` true by construction rather than by two call sites
    /// agreeing.
    #[inline(always)]
    pub fn count_merkle_direct() {
        TOTAL.fetch_add(1, Ordering::Relaxed);
        MERKLE.fetch_add(1, Ordering::Relaxed);
    }

    /// ★ A Merkle PARENT by such a hash. Bumps `total`, `merkle` and
    /// `merkle_nodes`, so `merkle - merkle_nodes` is the leaf count on this path
    /// exactly as it is on the byte path.
    #[inline(always)]
    pub fn count_merkle_node_direct() {
        TOTAL.fetch_add(1, Ordering::Relaxed);
        MERKLE.fetch_add(1, Ordering::Relaxed);
        MERKLE_NODES.fetch_add(1, Ordering::Relaxed);
    }

    /// Zero all counters.
    pub fn reset() {
        TOTAL.store(0, Ordering::Relaxed);
        MERKLE.store(0, Ordering::Relaxed);
        MERKLE_NODES.store(0, Ordering::Relaxed);
        GRINDING.store(0, Ordering::Relaxed);
        ABSORB_CALLS.store(0, Ordering::Relaxed);
        ABSORB_BYTES.store(0, Ordering::Relaxed);
    }

    pub fn snapshot() -> Counts {
        Counts {
            total: TOTAL.load(Ordering::Relaxed),
            merkle: MERKLE.load(Ordering::Relaxed),
            merkle_nodes: MERKLE_NODES.load(Ordering::Relaxed),
            grinding: GRINDING.load(Ordering::Relaxed),
            absorb_calls: ABSORB_CALLS.load(Ordering::Relaxed),
            absorb_bytes: ABSORB_BYTES.load(Ordering::Relaxed),
        }
    }
}

// Feature off, or the riscv64 guest: every entry compiles to nothing.
#[cfg(any(target_arch = "riscv64", not(feature = "hash-metrics")))]
mod imp {
    use super::Counts;

    #[inline(always)]
    pub fn count_total() {}
    #[inline(always)]
    pub fn count_merkle<D: 'static>() {}
    #[inline(always)]
    pub fn count_merkle_node<D: 'static>() {}
    #[inline(always)]
    pub fn count_grinding() {}
    #[inline(always)]
    pub fn count_merkle_direct() {}
    #[inline(always)]
    pub fn count_merkle_node_direct() {}
    #[inline(always)]
    pub fn count_absorb(_nbytes: usize) {}
    pub fn reset() {}
    pub fn snapshot() -> Counts {
        Counts::default()
    }
}

pub use imp::{
    count_absorb, count_grinding, count_merkle, count_merkle_direct, count_merkle_node,
    count_merkle_node_direct, count_total, reset, snapshot,
};
