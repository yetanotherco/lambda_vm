//! ★★ H4 — a commitment's leaf layer is hashed ONCE, and the bytes it keeps
//! are ones the reservation can see.
//!
//! Needs a GPU:
//!
//! ```text
//! cargo test -p math-cuda --release --test whir_tree_cache -- --nocapture
//! ```
//!
//! # What this is about
//!
//! `commit()` built the tree, took the root and dropped the buffer; `paths()`
//! then rebuilt the whole thing, leaves included, to read a kilobyte per query
//! out of it. Every commitment on this path is opened, so every commitment paid
//! for two leaf-hash passes over its codeword — measured at ~25 s of RPX device
//! hashing on a real block, about half of it that second pass.
//!
//! # ⚠ Why the counts are PER CODEWORD
//!
//! The first run of this file failed two tests for a reason that was not the
//! cache: `leaf_hash_calls()` is process-wide, the six tests share one binary,
//! and cargo runs them in parallel — so one test read 5 where it expected 1
//! purely because its neighbours were committing at the same time. An assertion
//! on a global counter is an assertion about every test in the binary.
//!
//! So the counting assertions use `DeviceCodeword::tree_builds()`, which counts
//! only that codeword's own leaf-hash passes. It is immune to the scheduler and
//! it localises a failure to the codeword that caused it. The process-wide
//! counter keeps one test of its own, and that one takes a lock.
//!
//! # Why the assertions are counts and bytes, not seconds
//!
//! A timing test would pass on a cache that returned the wrong tree, and fail
//! on a quiet machine for unrelated reasons. So: the leaf-hash pass is counted,
//! the reservation's promise is asserted as a number, and the paths are checked
//! against a freshly built tree. Seconds are the six-arm run's business.

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use math_cuda::DeviceHash;
use math_cuda::whir::{leaf_hash_calls, reset_leaf_hash_calls};
use multilinear::mle::Mle;
use multilinear::whir::{self, Domain};
use multilinear::whir_commit::{CodewordCommitment, verify_opening};
use multilinear::whir_hash::{DeviceHashKey, KeccakWhir, RpxWhir, WhirHash};
use std::sync::Mutex;

type FE = FieldElement<F>;

/// ★ Taken by EVERY test in this file, because every one of them commits, and a
/// commit moves two process-wide quantities: the leaf-hash counter and the
/// device reservation total.
///
/// The per-codeword counts added earlier make the *counting* assertions immune
/// to the scheduler, but one proposition is irreducibly global — "dropping a
/// codeword gives its promised bytes BACK" is a statement about
/// `Backend::reserved_bytes()`, and there is no per-codeword handle left to ask
/// once the codeword is gone. So these tests take turns.
///
/// That costs nothing real: they share one card and serialise on it regardless.
/// What the lock buys over `--test-threads=1` is that the property holds however
/// the suite is invoked, rather than only when someone remembers the flag.
///
/// Poisoning is ignored so one failure does not cascade into unrelated tests —
/// a lesson this branch learned once already, in `hash_metrics_tests`.
static DEVICE_GLOBALS: Mutex<()> = Mutex::new(());

fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    DEVICE_GLOBALS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Mirror of `DeviceHashKey::into_math_cuda`, which is cuda-gated on
/// `multilinear` and so unreachable from this crate's dev-dependency. The
/// production bridge is asserted bijective at compile time inside
/// `multilinear`; a swapped mirror here would fail
/// [`the_two_hashes_build_different_trees`] below.
fn key<H: WhirHash>() -> DeviceHash {
    match H::DEVICE {
        DeviceHashKey::Keccak256 => DeviceHash::Keccak256,
        DeviceHashKey::Rpx256 => DeviceHash::Rpx256,
    }
}

/// A polynomial with no structure a kernel could accidentally satisfy.
fn poly(num_vars: usize) -> Mle<F> {
    let evals: Vec<FE> = (0..(1u64 << num_vars))
        .map(|i| FE::from(i.wrapping_mul(6364136223846793005).wrapping_add(11) >> 11))
        .collect();
    Mle::new(evals).expect("power of two")
}

/// Commit on the device at `log_blowup = 2`, above `COMMIT_THRESHOLD`, so the
/// device path is the one taken.
fn commit_on_device(
    num_vars: usize,
    log_folding: usize,
    hash: DeviceHash,
) -> (math_cuda::whir::DeviceCodeword, [u8; 32]) {
    let f = poly(num_vars);
    let raw: Vec<u64> = f.evals().iter().map(|v| *v.value()).collect();
    math_cuda::whir::commit_codeword(&raw, 2, log_folding, false, hash)
        .expect("device commit (needs a GPU)")
}

/// ★★ (1) THE COUNT. Commit then open: ONE leaf-hash pass, not two.
///
/// Without the cache this reads 2 and fails — `LAMBDA_VM_NO_WHIR_TREE_CACHE=1`
/// restores the old behaviour and is how the mutation is run.
#[test]
fn a_commitment_hashes_its_leaves_once() {
    let _exclusive = exclusive();
    for (name, hash) in [("keccak", key::<KeccakWhir>()), ("rpx", key::<RpxWhir>())] {
        let (codeword, _root) = commit_on_device(14, 4, hash);
        assert_eq!(
            codeword.tree_builds(),
            1,
            "{name}: the commit itself must hash the leaves exactly once"
        );

        let _ = codeword.paths(4, &[0, 1, 7], hash).expect("paths");
        assert_eq!(
            codeword.tree_builds(),
            1,
            "{name}: opening must read the tree the commit kept, not rebuild it"
        );

        // …and again, because a cache that served once and then evicted would
        // pass the line above.
        let _ = codeword.paths(4, &[2, 3], hash).expect("paths");
        assert_eq!(
            codeword.tree_builds(),
            1,
            "{name}: still one, on a second open"
        );
    }
}

/// ★ (2) THE PATHS ARE RIGHT. What the cache serves equals what a freshly
/// built tree gives.
///
/// The count alone is satisfied by a cache that hands back stale or wrong
/// nodes: the paths would be internally consistent and wrong.
#[test]
fn the_cached_paths_are_the_ones_a_fresh_tree_gives() {
    let _exclusive = exclusive();
    for (name, hash) in [("keccak", key::<KeccakWhir>()), ("rpx", key::<RpxWhir>())] {
        let num_vars = 12;
        let log_folding = 4;
        let f = poly(num_vars);
        let raw: Vec<u64> = f.evals().iter().map(|v| *v.value()).collect();

        let (codeword, root) = math_cuda::whir::commit_codeword(&raw, 2, log_folding, false, hash)
            .expect("device commit (needs a GPU)");
        let leaves = (raw.len() << 2) >> log_folding;
        let positions: Vec<u32> = [0usize, 1, leaves / 3, leaves - 1]
            .iter()
            .map(|p| *p as u32)
            .collect();
        let cached = codeword
            .paths(log_folding, &positions, hash)
            .expect("paths");

        // A second, independent codeword over the same evaluations: same
        // inputs, a tree built from scratch, nothing shared with the one above.
        let (fresh_codeword, fresh_root) =
            math_cuda::whir::commit_codeword(&raw, 2, log_folding, false, hash)
                .expect("device commit");
        let fresh = fresh_codeword
            .paths(log_folding, &positions, hash)
            .expect("paths");

        assert_eq!(
            root, fresh_root,
            "{name}: the two commits disagree on the root"
        );
        assert_eq!(
            cached, fresh,
            "{name}: the cached tree's paths are not a fresh tree's"
        );
    }
}

/// ★ The same, through the production types, so the openings are checked by
/// the verifier rather than only compared to each other.
#[test]
fn the_cached_openings_verify_against_the_commitment() {
    let _exclusive = exclusive();
    fn check<H: WhirHash>(name: &str) {
        let num_vars = 12;
        let log_folding = 4;
        let f = poly(num_vars);
        let domain = Domain::<F>::new(num_vars + 2).expect("domain");
        let host_codeword =
            whir::encode::<F, F>(&whir::lift_coefficients(&f), &domain).expect("encode");
        let host =
            CodewordCommitment::<_, H>::new(&host_codeword, log_folding).expect("host commit");

        let raw: Vec<u64> = f.evals().iter().map(|v| *v.value()).collect();
        let (_device, root) =
            math_cuda::whir::commit_codeword(&raw, 2, log_folding, false, key::<H>())
                .expect("device commit (needs a GPU)");
        assert_eq!(root, host.root(), "{name}: device and host roots differ");

        for index in [0, 1, host.num_leaves() / 3, host.num_leaves() - 1] {
            let opening = host.open(index).expect("open");
            assert!(
                verify_opening::<_, H>(&root, index, &opening),
                "{name}: opening {index} does not verify against the device root"
            );
        }
    }
    check::<KeccakWhir>("keccak");
    check::<RpxWhir>("rpx");
}

/// ★★ (3) THE RESERVATION SEES IT. The cached tree's bytes are promised, and
/// giving the codeword up gives them back.
///
/// This is the condition the whole change turns on: device memory held outside
/// the accounting is an under-count that nothing reports until a second prover
/// shares the card. The number is asserted, not the absence of a crash.
#[test]
fn the_cached_tree_is_inside_the_reservation() {
    let _exclusive = exclusive();
    let be = math_cuda::device::backend().expect("a device");
    let hash = key::<RpxWhir>();
    let num_vars = 14;
    let log_folding = 4;

    // ⚠ Read under the lock, and it is a BASELINE rather than an assumed zero:
    // a sibling's reservation was what failed this test the first time it ran
    // on a card (786,400 B of someone else's tree). What is asserted below is
    // the DELTA this codeword is responsible for.
    let before = be.reserved_bytes();
    let (codeword, _root) = commit_on_device(num_vars, log_folding, hash);

    // `2*L - 1` nodes of 32 bytes, from the shapes alone.
    let leaves = ((1usize << num_vars) << 2) >> log_folding;
    let tree_bytes = (2 * leaves as u64 - 1) * 32;

    // (a) This codeword's OWN promise covers its tree — no global involved, so
    // this half would hold even without the lock.
    let held = codeword.reserved_bytes();
    assert!(
        held >= tree_bytes,
        "the reservation holds {held} B, which does not cover the {tree_bytes} B tree"
    );

    // (b) …and the global grew by at least that much, as a delta.
    let grown = be.reserved_bytes() - before;
    assert!(
        grown >= tree_bytes,
        "the global accounting grew by {grown} B, less than the {tree_bytes} B tree"
    );
    assert_eq!(
        grown, held,
        "this codeword's promise and the global's growth must be the same bytes"
    );

    // (c) Dropping it gives every one of them back. The irreducibly global
    // proposition, and the reason this test holds the lock.
    drop(codeword);
    assert_eq!(
        be.reserved_bytes(),
        before,
        "dropping the codeword must return the accounting to its baseline"
    );
}

/// ★ (4) THE KEY. A tree built for one blocking must not serve another.
///
/// It REBUILDS rather than refusing — both `log_folding` and the hash are
/// legitimate parameters of the call and a rebuild is always correct, whereas
/// refusing would turn an unusual-but-valid call into an error. What the key
/// rules out is the dangerous outcome: serving a tree that answers a different
/// question, whose paths would be internally consistent and wrong.
#[test]
fn a_tree_built_for_one_blocking_does_not_serve_another() {
    let _exclusive = exclusive();
    let hash = key::<RpxWhir>();
    let (codeword, _root) = commit_on_device(14, 4, hash);
    assert_eq!(codeword.tree_builds(), 1);

    // Same codeword, different blocking: a miss, so it rebuilds.
    let at_two = codeword.paths(2, &[0, 1], hash).expect("paths at k=2");
    assert_eq!(
        codeword.tree_builds(),
        2,
        "a different log_folding must rebuild, not serve the cached tree"
    );

    // And the rebuild answered the question that was asked: at k=2 the tree has
    // four times the leaves, so each path is two levels deeper.
    let at_four = codeword.paths(4, &[0, 1], hash).expect("paths at k=4");
    assert_eq!(
        at_two.len(),
        at_four.len() + 2 * 2 * 32,
        "a k=2 tree's paths must be two levels deeper than a k=4 tree's"
    );
}

/// ✓ The cache is per hash too — the same codeword under two keys must build
/// two different trees, or the dispatch key is being ignored one level up.
#[test]
fn the_two_hashes_build_different_trees() {
    let _exclusive = exclusive();
    let f = poly(12);
    let raw: Vec<u64> = f.evals().iter().map(|v| *v.value()).collect();
    let (_k, keccak_root) =
        math_cuda::whir::commit_codeword(&raw, 2, 4, false, key::<KeccakWhir>())
            .expect("device commit");
    let (_r, rpx_root) = math_cuda::whir::commit_codeword(&raw, 2, 4, false, key::<RpxWhir>())
        .expect("device commit");
    assert_ne!(keccak_root, rpx_root, "the two kernel families agreed");
}

/// ✓ The PROCESS-WIDE counter still tracks the same thing — it is what the
/// bench prints, so it needs a test of its own.
///
/// Takes [`GLOBAL_COUNTER`] across the whole window, because every other test
/// in this binary commits too and the counter cannot tell whose work it is
/// counting. That is exactly why the assertions above do not use it.
#[test]
fn the_process_wide_counter_tracks_the_same_passes() {
    let _exclusive = exclusive();
    let hash = key::<KeccakWhir>();

    reset_leaf_hash_calls();
    let (codeword, _root) = commit_on_device(12, 4, hash);
    let after_commit = leaf_hash_calls();
    assert_eq!(after_commit, 1, "one commit, one leaf-hash pass");

    let _ = codeword.paths(4, &[0, 1], hash).expect("paths");
    assert_eq!(
        leaf_hash_calls(),
        after_commit,
        "an opening adds no pass, so the global counter must not move"
    );
    assert_eq!(
        codeword.tree_builds(),
        leaf_hash_calls(),
        "with one codeword in flight the two counters must agree"
    );
}
