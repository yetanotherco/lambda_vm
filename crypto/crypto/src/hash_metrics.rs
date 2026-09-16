//! Host-only keccak-hash counters for measuring the cost of VERIFYING a proof
//! (a proxy for the recursion guest's dominant work: keccak hashing).
//!
//! Behind the `hash-metrics` cargo feature: a normal build keeps
//! `PlatformKeccak256 = sha3::Keccak256` and every counter call compiles to
//! nothing, so the prover is provably unchanged. With the feature on (host only),
//! the host `PlatformKeccak256` wrapper counts, per keccak op:
//! * `perms` — keccak-f permutations, THE unit the guest pays (one `keccak_permute`
//!   syscall each): `sum over hashes of (absorbed_bytes / 136 + 1)`. Guest-faithful
//!   regardless of host call shape — a 64-byte Merkle parent is 1 perm here (2
//!   updates + finalize) and 1 on the guest (`keccak256_pair`) — so it is
//!   self-validatable against a measured `keccak_permute` census;
//! * `total` — every finalize (leaf / node / transcript squeeze / program-id fold).
//!   `merkle` and `grinding` are disjoint keccak-only subsets of it (the remainder
//!   is the transcript / program-id fold); `merkle_nodes ⊆ merkle`, so
//!   leaves = `merkle − merkle_nodes`;
//! * `absorb_calls` / `absorb_bytes` — every `Update::update`. HOST-side absorption:
//!   the guest takes `keccak256_pair` for Merkle parents (0 updates), so its absorb
//!   calls ≈ `absorb_calls − 2·merkle_nodes`. Useful for spotting a block-absorption
//!   change (fewer, larger updates — same `perms`, fewer `absorb_calls`).
//!
//! No enable/disable toggle and nothing in the verifier: counting is always on
//! under the feature, and a measuring caller just [`reset`]s before the verify
//! and reads [`snapshot`] after. Grinding is separated by counter, not excluded
//! at a call site, so there is no cross-thread race under a parallel verify.

/// Snapshot of the verify-hash counters (all zero without the `hash-metrics`
/// feature / on the guest).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// keccak-f permutations — the unit the guest pays (one `keccak_permute` each).
    pub perms: u64,
    /// Every keccak-256 finalize.
    pub total: u64,
    /// Merkle finalizes (keccak-guarded subset of `total`).
    pub merkle: u64,
    /// Merkle auth-path (parent) compressions (subset of `merkle`).
    pub merkle_nodes: u64,
    /// Grinding proof-of-work finalizes (subset of `total`).
    pub grinding: u64,
    /// Keccak absorb (`Update::update`) invocations (host-side).
    pub absorb_calls: u64,
    /// Bytes fed through absorb (sum of `data.len()`).
    pub absorb_bytes: u64,
}

#[cfg(all(not(target_arch = "riscv64"), feature = "hash-metrics"))]
mod imp {
    use super::Counts;
    use core::sync::atomic::{AtomicU64, Ordering};

    /// keccak-256 rate in bytes (1088 bits): bytes absorbed per permutation.
    const RATE: usize = 136;

    static PERMS: AtomicU64 = AtomicU64::new(0);
    static TOTAL: AtomicU64 = AtomicU64::new(0);
    static MERKLE: AtomicU64 = AtomicU64::new(0);
    static MERKLE_NODES: AtomicU64 = AtomicU64::new(0);
    static GRINDING: AtomicU64 = AtomicU64::new(0);
    static ABSORB_CALLS: AtomicU64 = AtomicU64::new(0);
    static ABSORB_BYTES: AtomicU64 = AtomicU64::new(0);

    /// A keccak-256 finalize of a hash that absorbed `nbytes`. Bumps the finalize
    /// count and the permutation count (`nbytes / RATE + 1` — the full blocks
    /// absorbed plus the padded final block), the unit the guest pays.
    #[inline(always)]
    pub fn count_finalize(nbytes: usize) {
        TOTAL.fetch_add(1, Ordering::Relaxed);
        PERMS.fetch_add((nbytes / RATE + 1) as u64, Ordering::Relaxed);
    }

    /// A Merkle finalize (leaf or node), counted ONLY when the backend digest is
    /// the platform keccak wrapper — the one whose `finalize` also bumps
    /// [`count_finalize`]. This keeps `merkle` a strict subset of `total` for ANY
    /// `D` (a non-keccak backend, as in the crypto tests, does not go through the
    /// counted wrapper, so counting it here would let `merkle` exceed `total`).
    #[inline(always)]
    pub fn count_merkle<D: 'static>() {
        if core::any::TypeId::of::<D>()
            == core::any::TypeId::of::<crate::hash::platform_keccak::PlatformKeccak256>()
        {
            MERKLE.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A Merkle parent (auth-path) compression. Subset of [`count_merkle`];
    /// same keccak-only guard. Every parent must ALSO call [`count_merkle`] so
    /// `merkle_nodes ⊆ merkle`.
    #[inline(always)]
    pub fn count_merkle_node<D: 'static>() {
        if core::any::TypeId::of::<D>()
            == core::any::TypeId::of::<crate::hash::platform_keccak::PlatformKeccak256>()
        {
            MERKLE_NODES.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A grinding (proof-of-work) finalize. Subset of [`count_finalize`]; a caller
    /// reports `total - grinding` to exclude the PoW check.
    #[inline(always)]
    pub fn count_grinding() {
        GRINDING.fetch_add(1, Ordering::Relaxed);
    }

    /// A keccak absorb (`Update::update`) of `nbytes` (host-side). Bumps the call
    /// count and the byte total.
    #[inline(always)]
    pub fn count_absorb(nbytes: usize) {
        ABSORB_CALLS.fetch_add(1, Ordering::Relaxed);
        ABSORB_BYTES.fetch_add(nbytes as u64, Ordering::Relaxed);
    }

    /// Zero all counters.
    pub fn reset() {
        PERMS.store(0, Ordering::Relaxed);
        TOTAL.store(0, Ordering::Relaxed);
        MERKLE.store(0, Ordering::Relaxed);
        MERKLE_NODES.store(0, Ordering::Relaxed);
        GRINDING.store(0, Ordering::Relaxed);
        ABSORB_CALLS.store(0, Ordering::Relaxed);
        ABSORB_BYTES.store(0, Ordering::Relaxed);
    }

    pub fn snapshot() -> Counts {
        Counts {
            perms: PERMS.load(Ordering::Relaxed),
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
    pub fn count_finalize(_nbytes: usize) {}
    #[inline(always)]
    pub fn count_merkle<D: 'static>() {}
    #[inline(always)]
    pub fn count_merkle_node<D: 'static>() {}
    #[inline(always)]
    pub fn count_grinding() {}
    #[inline(always)]
    pub fn count_absorb(_nbytes: usize) {}
    pub fn reset() {}
    pub fn snapshot() -> Counts {
        Counts::default()
    }
}

pub use imp::{
    count_absorb, count_finalize, count_grinding, count_merkle, count_merkle_node, reset, snapshot,
};
