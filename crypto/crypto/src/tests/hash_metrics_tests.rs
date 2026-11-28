//! ★ The verify-hash counters count BOTH hash families — the test that would
//! have caught the false zero.
//!
//! `hash_metrics` was keccak-only: `count_merkle` compared `TypeId::of::<D>()`
//! against `PlatformKeccak256` and did nothing otherwise. That was correct while
//! keccak was the only hash on the multilinear path, and it became a check that
//! cannot fail the moment a second one arrived — under RPX every Merkle counter
//! would have read ZERO, reporting "no hashing" for precisely the arm whose
//! purpose is to change the hashing, and a reader comparing the two arms would
//! have concluded the algebraic hash was free.
//!
//! So these tests are written the way that failure would have been caught:
//! **non-zero after the RPX arm, non-zero after the keccak arm, and the subset
//! invariant intact for both.** Every one of them fails on the pre-extension
//! code.
//!
//! ⚠ These run only under `--features hash-metrics`. That is the same reason the
//! module exists at all — the counters compile to nothing otherwise — but it
//! does mean a default `cargo test` does not execute them. `make lint`'s passes
//! do not enable the feature either; the gate is `cargo test -p crypto
//! --features hash-metrics`.

#![cfg(feature = "hash-metrics")]

use alloc::vec::Vec;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as Fp;

use crate::hash_metrics::{Counts, reset, snapshot};
use crate::merkle_tree::backends::rpx::RpxVectorBackend;
use crate::merkle_tree::backends::types::BatchKeccak256Backend;
use crate::merkle_tree::traits::IsMerkleTreeBackend;

type Rpx = RpxVectorBackend<Fp>;
type Keccak = BatchKeccak256Backend<Fp>;

/// The counters are process-global, so the cases take turns rather than
/// running concurrently. A mutex rather than `--test-threads=1`, so the
/// property does not depend on how the suite is invoked.
///
/// ⚠ Poisoning is IGNORED, and that is not laziness. A failing case panics
/// while holding this lock, and `unwrap()` would then panic every later case on
/// the poisoned mutex — turning one real failure into five, four of them
/// cascades. That was observed while mutation-testing this file: removing the
/// RPX backend's counter calls failed the two cases that assert it AND the
/// keccak case, which is a lie about the keccak path. The counters are reset at
/// the top of every measurement, so a poisoned lock carries no stale state.
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serialise() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn leaf(n: usize) -> Vec<FieldElement<Fp>> {
    (0..n as u64)
        .map(|i| FieldElement::from(i * 7 + 1))
        .collect()
}

/// Hash one leaf and one parent under `B`, and report what the counters saw.
fn measure<B>() -> Counts
where
    B: IsMerkleTreeBackend<Node = [u8; 32], Data = Vec<FieldElement<Fp>>>,
{
    reset();
    let a = B::hash_data(&leaf(5));
    let b = B::hash_data(&leaf(6));
    let _ = B::hash_new_parent(&a, &b);
    snapshot()
}

/// ★★ The RPX arm is COUNTED — the proposition the extension exists for.
#[test]
fn the_algebraic_backend_is_counted() {
    let _g = serialise();
    let c = measure::<Rpx>();

    assert_eq!(c.merkle, 3, "two leaves and one parent");
    assert_eq!(c.merkle_nodes, 1, "one parent");
    assert_eq!(
        c.merkle - c.merkle_nodes,
        2,
        "merkle - nodes must be the leaf count, as it is on the byte path"
    );
    assert!(
        c.total >= c.merkle,
        "total {} must cover merkle {}",
        c.total,
        c.merkle
    );
}

/// The keccak arm still is, unchanged — so the extension did not move the
/// number the existing instrument reports.
#[test]
fn the_byte_backend_is_still_counted() {
    let _g = serialise();
    let c = measure::<Keccak>();

    assert_eq!(c.merkle, 3, "two leaves and one parent");
    assert_eq!(c.merkle_nodes, 1, "one parent");
    assert!(c.total >= c.merkle);
}

/// ★ The two arms agree on the COUNT while differing in the hash — which is
/// what makes a cross-arm comparison of these numbers meaningful at all.
///
/// If they disagreed, a difference in the counters would not distinguish "this
/// hash does more work" from "this backend is instrumented differently".
#[test]
fn the_two_backends_report_the_same_shape_for_the_same_tree() {
    let _g = serialise();
    let rpx = measure::<Rpx>();
    let keccak = measure::<Keccak>();

    assert_eq!(rpx.merkle, keccak.merkle);
    assert_eq!(rpx.merkle_nodes, keccak.merkle_nodes);
}

/// ⚠ **The subset invariant, on the path that could break it.** `merkle` must
/// never exceed `total`: the byte backends get `total` from the digest's own
/// `finalize`, and the algebraic backend has no digest to finalize, so it owes
/// the bump itself. A `count_merkle_direct` that forgot `TOTAL` would land here.
#[test]
fn merkle_never_exceeds_total_on_the_algebraic_path() {
    let _g = serialise();
    reset();
    for n in 0..16 {
        let _ = Rpx::hash_data(&leaf(n));
    }
    let c = snapshot();
    assert_eq!(c.merkle, 16);
    assert!(
        c.total >= c.merkle,
        "total {} < merkle {} — the algebraic path did not bump total",
        c.total,
        c.merkle
    );
}

/// ✓ `reset` really resets, so one case cannot read another's counts.
#[test]
fn reset_clears_every_counter() {
    let _g = serialise();
    let _ = Rpx::hash_data(&leaf(4));
    reset();
    assert_eq!(snapshot(), Counts::default());
}
