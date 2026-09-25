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
//! * `perms` — keccak-f permutations, THE unit the recursion guest pays (one
//!   `keccak_permute` syscall each): `sum over hashes of (absorbed_bytes / 136 + 1)`.
//!   Guest-faithful regardless of host call shape — a 64-byte Merkle parent is 1 perm
//!   here (2 updates + finalize) and 1 on the guest (`keccak256_pair`) — so it is
//!   self-validatable against a measured `keccak_permute` census. Keccak only: the
//!   algebraic sponge's finalizes arrive through [`count_total`] and bump nothing here;
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
    /// keccak-f permutations — the unit the guest pays (one `keccak_permute` each).
    /// Keccak-only, so it is zero on an all-algebraic configuration.
    pub perms: u64,
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
    /// ★★ Fiat-Shamir absorbs, ALL configurations. Counted in
    /// `DefaultTranscript`'s own append methods, not in a hash.
    pub transcript_absorbs: u64,
    /// Of those, the ones whose sponge is keccak.
    pub transcript_absorbs_keccak: u64,
    /// Of those, the ones whose sponge is RPX256.
    pub transcript_absorbs_rpx: u64,
    /// ★★ Fiat-Shamir SQUEEZES, ALL configurations — `finalize_reset`, which
    /// advances the chain (the output is re-absorbed).
    pub transcript_squeezes: u64,
    /// Of those, the ones whose sponge is keccak.
    pub transcript_squeezes_keccak: u64,
    /// Of those, the ones whose sponge is RPX256.
    pub transcript_squeezes_rpx: u64,
    /// ★★ Fiat-Shamir STATE reads, ALL configurations — `state()`, a finalize
    /// on a CLONE. No reset and no re-absorb, so it does NOT advance the chain.
    ///
    /// ⚠ A DIFFERENT OPERATION from a squeeze, counted separately because
    /// conflating them makes a number unfalsifiable. On a block proof the two
    /// are 182,734 and 2,996 (lane V1's closed form, pinned to a measured
    /// verify with difference 0): a counter hooked only to `finalize_reset`
    /// misses every one of the 2,996, and a counter reporting their sum —
    /// 185,730 — cannot be checked against either.
    ///
    /// ★ The state reads are a CONTROL that costs nothing: there is exactly one
    /// per grind check, so this must equal the grind count the same run prints.
    /// Two independent instruments on one quantity, and a disagreement names
    /// which of them is wrong.
    pub transcript_states: u64,
    /// Of those, the ones whose sponge is keccak.
    pub transcript_states_keccak: u64,
    /// Of those, the ones whose sponge is RPX256.
    pub transcript_states_rpx: u64,
    /// ★★ Fiat-Shamir absorbs AFTER A STATEMENT that began at a byte offset
    /// which is NOT a multiple of 8, measured within the current WINDOW — the
    /// bytes absorbed since the sponge was last reset by a squeeze.
    ///
    /// An in-guest verifier re-slices a window into field elements every 8
    /// bytes, so a value that starts off a boundary straddles two of them and
    /// costs the machine the arithmetic of putting it back together. The
    /// statement padding exists to keep this at zero on a real proof, and this
    /// is what says whether it did.
    ///
    /// NOT split by sponge, unlike the three above: alignment is a property of
    /// the byte stream, and the same stream under two hashes is misaligned in
    /// the same places or in neither.
    ///
    /// # ⛔ WHY "AFTER A STATEMENT" IS IN THE NAME
    ///
    /// The first version of this field counted EVERY misaligned absorb, and a
    /// zero was pre-registered for a real proof. That zero was a wish. A
    /// STATEMENT is variable-length by nature — a tag, one byte per table
    /// count, a three-byte grind trailer — so absorbs inside it start off
    /// boundaries constantly, and the padding never promised otherwise: what it
    /// promises is that whatever FOLLOWS a statement starts aligned. Measured on
    /// the EQ fixture under the old definition, one epoch statement plus its
    /// proof read 25 misaligned absorbs at `public_output = 0` and 24 at two
    /// bytes, of which 22 and 21 were felt-sized — and ZERO fell after the
    /// statement. Filtering by payload size recovered nothing; only the boundary
    /// did.
    ///
    /// So the transcript is TOLD where a statement ends
    /// ([`IsTranscript::mark_statement_end`], called by the prover's statement
    /// padding) and counts from there. A transcript that is never told counts
    /// nothing, which is correct — it has no "after a statement".
    ///
    /// ⚠ A zero is only a measurement if something can make it non-zero, and
    /// two tests hold that. `a_window_that_opens_off_a_boundary_is_counted`
    /// marks a transcript whose window is already at 13 — the WHIR byte gate's
    /// own first window, documented as not aligned — and requires exactly two.
    /// `a_statement_that_ends_off_a_boundary_is_still_counted` is the sharper
    /// one: the mark does NOT reset the window, so a statement that ended badly
    /// is reported rather than forgiven. Without that, the mark would be a way
    /// of switching off the check it was built to sharpen.
    ///
    /// The cross-check on a real prove is
    /// `statement_alignment_tests::the_counter_and_the_recorder_agree_on_one_prove`,
    /// which measures the same prove with an independent recorder and compares
    /// this counter against the recorder's own after-the-statement count.
    pub transcript_misaligned_absorbs_after_statement: u64,
}

impl Counts {
    /// Transcript work this build could not attribute to a known sponge.
    ///
    /// ★ Zero on every configuration that exists, and it is REPORTED rather
    /// than assumed: the failure this whole group of counters exists to catch
    /// is a hash nobody instrumented reading as a zero that looks like
    /// "nothing ran". A third configuration arriving un-instrumented shows up
    /// here instead of being silently folded into one of the two above.
    pub fn transcript_unattributed(&self) -> (u64, u64, u64) {
        (
            self.transcript_absorbs - self.transcript_absorbs_keccak - self.transcript_absorbs_rpx,
            self.transcript_squeezes
                - self.transcript_squeezes_keccak
                - self.transcript_squeezes_rpx,
            self.transcript_states - self.transcript_states_keccak - self.transcript_states_rpx,
        )
    }
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
    static T_ABSORBS: AtomicU64 = AtomicU64::new(0);
    static T_ABSORBS_KECCAK: AtomicU64 = AtomicU64::new(0);
    static T_ABSORBS_RPX: AtomicU64 = AtomicU64::new(0);
    static T_SQUEEZES: AtomicU64 = AtomicU64::new(0);
    static T_SQUEEZES_KECCAK: AtomicU64 = AtomicU64::new(0);
    static T_SQUEEZES_RPX: AtomicU64 = AtomicU64::new(0);
    static T_STATES: AtomicU64 = AtomicU64::new(0);
    static T_STATES_KECCAK: AtomicU64 = AtomicU64::new(0);
    static T_STATES_RPX: AtomicU64 = AtomicU64::new(0);
    static T_MISALIGNED: AtomicU64 = AtomicU64::new(0);

    /// Which known sponge `D` is, if any: `Some(true)` keccak, `Some(false)`
    /// RPX256, `None` a configuration nobody has instrumented.
    fn sponge<D: 'static>() -> Option<bool> {
        let id = core::any::TypeId::of::<D>();
        if id == core::any::TypeId::of::<crate::hash::platform_keccak::PlatformKeccak256>() {
            Some(true)
        } else if id == core::any::TypeId::of::<crate::hash::rpx::Rpx256Digest>() {
            Some(false)
        } else {
            None
        }
    }

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
    /// same keccak-only guard. Every parent must ALSO call [`count_merkle`] so
    /// `merkle_nodes ⊆ merkle` — on the byte path that is the backend's job (see
    /// `merkle_tree::backends::field_element`), on the algebraic path
    /// [`count_merkle_node_direct`] does both itself.
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

    /// A KECCAK finalize of a hash that absorbed `nbytes`. Bumps the finalize
    /// count and the permutation count (`nbytes / RATE + 1` — the full blocks
    /// absorbed plus the padded final block), the unit the guest pays.
    ///
    /// ⚠ Why this and [`count_total`] both exist. Only the byte sponges know how
    /// many bytes went into the hash they are finalizing: the host
    /// `PlatformKeccak256` wrapper accumulates them and calls this. The algebraic
    /// backend sponges FELTS and has no byte count to report, so it calls
    /// `count_total` and leaves `perms` alone — which is correct, since `perms`
    /// counts keccak-f. Collapsing the two would either invent a byte count for
    /// RPX or lose the permutation count for keccak.
    #[inline(always)]
    pub fn count_finalize(nbytes: usize) {
        TOTAL.fetch_add(1, Ordering::Relaxed);
        PERMS.fetch_add((nbytes / RATE + 1) as u64, Ordering::Relaxed);
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

    /// ★★ One Fiat-Shamir ABSORB, tagged by the sponge that will consume it.
    ///
    /// Called from `DefaultTranscript`'s append methods — the transcript, not
    /// the hash. Two reasons, and the second is the one that matters:
    ///
    /// 1. It is hash-agnostic by construction. A counter living inside keccak
    ///    reads ZERO for an algebraic transcript, which is indistinguishable
    ///    from "no transcript ran" — the trap this module's header describes
    ///    for Merkle, which the transcript never got.
    /// 2. It counts TRANSCRIPT absorbs only. [`count_absorb`] is bumped from a
    ///    digest's `update`, so it mixes Merkle leaf bytes with Fiat-Shamir
    ///    bytes and cannot answer "how much did the transcript absorb" for
    ///    either hash.
    #[inline(always)]
    pub fn count_transcript_absorb<D: 'static>() {
        T_ABSORBS.fetch_add(1, Ordering::Relaxed);
        match sponge::<D>() {
            Some(true) => T_ABSORBS_KECCAK.fetch_add(1, Ordering::Relaxed),
            Some(false) => T_ABSORBS_RPX.fetch_add(1, Ordering::Relaxed),
            None => 0,
        };
    }

    /// ★★ One Fiat-Shamir absorb, PAST A STATEMENT, that began off a field
    /// element boundary.
    ///
    /// Untagged by sponge on purpose — see the field's documentation. The
    /// caller decides both halves: `DefaultTranscript` is the only one, because
    /// it is the only place that knows the window AND the only one that has
    /// been told where the statement ended.
    #[inline(always)]
    pub fn count_transcript_misaligned_absorb() {
        T_MISALIGNED.fetch_add(1, Ordering::Relaxed);
    }

    /// ★★ One Fiat-Shamir SQUEEZE, tagged the same way.
    #[inline(always)]
    pub fn count_transcript_squeeze<D: 'static>() {
        T_SQUEEZES.fetch_add(1, Ordering::Relaxed);
        match sponge::<D>() {
            Some(true) => T_SQUEEZES_KECCAK.fetch_add(1, Ordering::Relaxed),
            Some(false) => T_SQUEEZES_RPX.fetch_add(1, Ordering::Relaxed),
            None => 0,
        };
    }

    /// ★★ One `state()` — a finalize on a CLONE of the sponge.
    ///
    /// Not a squeeze: no reset, no re-absorb, the chain does not advance. It is
    /// its own counter because the two are different operations and a sum
    /// cannot be checked against either. One of these per grind check, which is
    /// what makes it a free control against the grind count.
    #[inline(always)]
    pub fn count_transcript_state<D: 'static>() {
        T_STATES.fetch_add(1, Ordering::Relaxed);
        match sponge::<D>() {
            Some(true) => T_STATES_KECCAK.fetch_add(1, Ordering::Relaxed),
            Some(false) => T_STATES_RPX.fetch_add(1, Ordering::Relaxed),
            None => 0,
        };
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
        T_ABSORBS.store(0, Ordering::Relaxed);
        T_ABSORBS_KECCAK.store(0, Ordering::Relaxed);
        T_ABSORBS_RPX.store(0, Ordering::Relaxed);
        T_SQUEEZES.store(0, Ordering::Relaxed);
        T_SQUEEZES_KECCAK.store(0, Ordering::Relaxed);
        T_SQUEEZES_RPX.store(0, Ordering::Relaxed);
        T_STATES.store(0, Ordering::Relaxed);
        T_STATES_KECCAK.store(0, Ordering::Relaxed);
        T_STATES_RPX.store(0, Ordering::Relaxed);
        T_MISALIGNED.store(0, Ordering::Relaxed);
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
            transcript_absorbs: T_ABSORBS.load(Ordering::Relaxed),
            transcript_absorbs_keccak: T_ABSORBS_KECCAK.load(Ordering::Relaxed),
            transcript_absorbs_rpx: T_ABSORBS_RPX.load(Ordering::Relaxed),
            transcript_squeezes: T_SQUEEZES.load(Ordering::Relaxed),
            transcript_squeezes_keccak: T_SQUEEZES_KECCAK.load(Ordering::Relaxed),
            transcript_squeezes_rpx: T_SQUEEZES_RPX.load(Ordering::Relaxed),
            transcript_states: T_STATES.load(Ordering::Relaxed),
            transcript_states_keccak: T_STATES_KECCAK.load(Ordering::Relaxed),
            transcript_states_rpx: T_STATES_RPX.load(Ordering::Relaxed),
            transcript_misaligned_absorbs_after_statement: T_MISALIGNED.load(Ordering::Relaxed),
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
    #[inline(always)]
    pub fn count_transcript_absorb<D: 'static>() {}
    #[inline(always)]
    pub fn count_transcript_squeeze<D: 'static>() {}
    #[inline(always)]
    pub fn count_transcript_misaligned_absorb() {}
    #[inline(always)]
    pub fn count_transcript_state<D: 'static>() {}
    pub fn reset() {}
    pub fn snapshot() -> Counts {
        Counts::default()
    }
}

pub use imp::{
    count_absorb, count_finalize, count_grinding, count_merkle, count_merkle_direct,
    count_merkle_node, count_merkle_node_direct, count_total, count_transcript_absorb,
    count_transcript_misaligned_absorb, count_transcript_squeeze, count_transcript_state, reset,
    snapshot,
};
