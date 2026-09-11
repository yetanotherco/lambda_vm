//! Host-only keccak-hash counters — a metric for the cost of VERIFYING a proof
//! (a proxy for the recursion guest's dominant work: keccak hashing).
//!
//! [`count_total`] fires on EVERY keccak-256 finalize (the host
//! [`crate::hash::platform_keccak::PlatformKeccak256`] wrapper) — Merkle trees,
//! the Fiat-Shamir transcript, the program-id/ELF fold, everything. The Merkle
//! backend additionally splits its share via [`count_merkle`] (every Merkle
//! finalize) and [`count_merkle_node`] (auth-path parent compressions), so a
//! caller can report `nodes`, `leaves = merkle − nodes`, and
//! `transcript+other = total − merkle`.
//!
//! GRINDING IS EXCLUDED at its call site: the verifier wraps `is_valid_nonce`
//! with [`disable`]/re-[`enable`] (guarded by [`is_enabled`]), so the proof-of-
//! work check's finalizes are not counted.
//!
//! Compiled OUT on the riscv64 guest (`#[cfg]`) so the guest — whose cycles we
//! actually care about — pays nothing. On the host each counted site is one
//! relaxed atomic load (negligible vs prove time; disabled by default).

#[cfg(not(target_arch = "riscv64"))]
mod host {
    use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    static ENABLED: AtomicBool = AtomicBool::new(false);
    static TOTAL: AtomicU64 = AtomicU64::new(0);
    static MERKLE: AtomicU64 = AtomicU64::new(0);
    static MERKLE_NODES: AtomicU64 = AtomicU64::new(0);

    /// Every keccak-256 finalize, from any site (called by the host
    /// `PlatformKeccak256` wrapper). The headline "all hashes" number.
    #[inline(always)]
    pub fn count_total() {
        if ENABLED.load(Ordering::Relaxed) {
            TOTAL.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A Merkle finalize (leaf or node). Subset of [`count_total`].
    #[inline(always)]
    pub fn count_merkle() {
        if ENABLED.load(Ordering::Relaxed) {
            MERKLE.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A Merkle parent (auth-path) compression. Subset of [`count_merkle`].
    #[inline(always)]
    pub fn count_merkle_node() {
        if ENABLED.load(Ordering::Relaxed) {
            MERKLE_NODES.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Whether counting is currently on — so the grinding exclusion can restore
    /// the prior state instead of blindly re-enabling.
    #[inline(always)]
    pub fn is_enabled() -> bool {
        ENABLED.load(Ordering::Relaxed)
    }

    pub fn enable() {
        ENABLED.store(true, Ordering::Relaxed);
    }

    pub fn disable() {
        ENABLED.store(false, Ordering::Relaxed);
    }

    /// Zero the counters (does not change the enabled state).
    pub fn reset() {
        TOTAL.store(0, Ordering::Relaxed);
        MERKLE.store(0, Ordering::Relaxed);
        MERKLE_NODES.store(0, Ordering::Relaxed);
    }

    /// `(total, merkle, merkle_nodes)`. leaves = merkle − nodes;
    /// transcript+other = total − merkle.
    pub fn snapshot() -> (u64, u64, u64) {
        (
            TOTAL.load(Ordering::Relaxed),
            MERKLE.load(Ordering::Relaxed),
            MERKLE_NODES.load(Ordering::Relaxed),
        )
    }
}

#[cfg(not(target_arch = "riscv64"))]
pub use host::{
    count_merkle, count_merkle_node, count_total, disable, enable, is_enabled, reset, snapshot,
};

// Guest stubs — compiled to nothing; the guest must not pay for measurement.
// `enable`/`disable`/`is_enabled` exist too so the shared verifier code (which
// wraps the grinding check) compiles for the guest without `#[cfg]` noise.
#[cfg(target_arch = "riscv64")]
#[inline(always)]
pub fn count_total() {}

#[cfg(target_arch = "riscv64")]
#[inline(always)]
pub fn count_merkle() {}

#[cfg(target_arch = "riscv64")]
#[inline(always)]
pub fn count_merkle_node() {}

#[cfg(target_arch = "riscv64")]
#[inline(always)]
pub fn enable() {}

#[cfg(target_arch = "riscv64")]
#[inline(always)]
pub fn disable() {}

#[cfg(target_arch = "riscv64")]
#[inline(always)]
pub fn is_enabled() -> bool {
    false
}
