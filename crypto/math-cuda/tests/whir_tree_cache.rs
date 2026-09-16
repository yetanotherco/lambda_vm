//! ★★ H4's result of record — a commitment does NOT keep its tree, and the
//! bytes it holds are only its codeword.
//!
//! Needs a GPU:
//!
//! ```text
//! cargo test -p math-cuda --release --test whir_tree_cache -- --nocapture
//! ```
//!
//! # What this is about
//!
//! `commit()` builds the tree, takes the root and drops the buffer; `paths()`
//! rebuilds it to read a kilobyte per query out of it. Every commitment on this
//! path is opened, so every one pays for two leaf-hash passes over its codeword
//! — ~25 s of RPX device hashing on a real block, about half of it that second
//! pass. H4 cached the first tree to remove the second pass. It was measured on
//! the card and it LOST, ~+15 s in both hashes.
//!
//! # Why keeping the tree cannot work here, which is what these tests pin
//!
//! Not because the cache missed — it returned exactly the hashing it promised.
//! Because the retention is one tree per commitment IN THE GROUP, not one tree.
//! `StackedCommitment::commit` builds every chain's commitment before it
//! returns, since all the roots enter the transcript before any query index is
//! drawn, and the openings follow one chain at a time. So the last chain's tree
//! would live from its commit to the end of the proof, and no placement of an
//! eviction call bounds that peak: all N trees exist before the first opening.
//! Ten chains at half a gigabyte put the card at 96%, after which device
//! allocations fail, commits fall back to the host, and the host grows ~1.5 GiB
//! per fallen-back chain.
//!
//! `multilinear::whir_commit`'s `paths` said this in its doc comment before any
//! of it was built, and the reservation in `StackedCommitment::commit` — "nine
//! codewords of room instead of sixteen" — budgets a retained codeword per
//! commitment and no tree.
//!
//! # ⚠ Why the counts are PER CODEWORD
//!
//! An earlier run of this file failed two tests for a reason that was not the
//! code under test: `leaf_hash_calls()` is process-wide, the tests share one
//! binary, and cargo runs them in parallel — so one test read 5 where it
//! expected 1 purely because its neighbours were committing at the same time.
//! An assertion on a global counter is an assertion about every test in the
//! binary. The counting assertions use `DeviceCodeword::tree_builds()`; the
//! process-wide counter keeps one test of its own, and that one takes the lock.
//!
//! # ⚠ Why the memory guard asks the DRIVER
//!
//! `Backend::reserved_bytes()` counts what callers promised, so it is silent
//! about device memory allocated without a reservation — and it reads baseline
//! while the card fills, which is how the H4 arm's retention stayed invisible
//! to a unit test that passed. It is also trivially at baseline now, which is a
//! check that cannot fail. So the guard samples `free_vram_bytes()` across four
//! live, unopened commitments and asserts what they took is their codewords and
//! nothing else. That one fails if a tree is ever held past the call that
//! builds it, wherever the holding is written.

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

/// ★★ (1) THE COUNT. One leaf-hash pass per tree built: the commit's, and
/// one more for each round that opens it.
///
/// This is the cost H4 tried to remove and the number that says whether anyone
/// has quietly re-added a cache. It is asserted as an integer in both
/// directions — a commitment that read 1 after an opening would mean a tree is
/// being kept, which is the state this file exists to forbid.
#[test]
fn a_commitment_hashes_its_leaves_once_per_tree_it_builds() {
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
            2,
            "{name}: an opening builds its own tree — a 1 here means one is kept"
        );

        // …and again, because a cache that served once and then evicted would
        // read 2 on the line above too.
        let _ = codeword.paths(4, &[2, 3], hash).expect("paths");
        assert_eq!(
            codeword.tree_builds(),
            3,
            "{name}: and a second opening builds a third"
        );
    }
}

/// ★ (2) THE PATHS ARE RIGHT. Two independent codewords over the same
/// evaluations give the same root and the same paths.
///
/// The count alone is satisfied by a build that hands back stale or wrong
/// nodes: the paths would be internally consistent and wrong.
#[test]
fn the_paths_are_the_ones_a_fresh_tree_gives() {
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
        let first = codeword
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
            first, fresh,
            "{name}: two builds over the same codeword disagree on the paths"
        );
    }
}

/// ★ The same, through the production types, so the openings are checked by
/// the verifier rather than only compared to each other.
#[test]
fn the_openings_verify_against_the_device_commitment() {
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

/// ★★ (3) THE RESERVATION SEES THE CODEWORD, AND GETS IT BACK.
///
/// Device memory held outside the accounting is an under-count that nothing
/// reports until a second prover shares the card. What a codeword promises is
/// its own bytes and the folds that halve it — NOT a tree, which is never held
/// past the call that builds it.
#[test]
fn the_codeword_is_inside_the_reservation_and_gives_it_back() {
    let _exclusive = exclusive();
    let be = math_cuda::device::backend().expect("a device");
    let hash = key::<RpxWhir>();
    let num_vars = 14;
    let log_folding = 4;

    // ⚠ Read under the lock, and it is a BASELINE rather than an assumed zero:
    // a sibling's reservation was what failed this test the first time it ran
    // on a card (786,400 B of someone else's). What is asserted below is the
    // DELTA this codeword is responsible for.
    let before = be.reserved_bytes();
    let (codeword, _root) = commit_on_device(num_vars, log_folding, hash);

    let codeword_bytes = ((1u64 << num_vars) << 2) * 8;
    // `2*L - 1` nodes of 32 bytes, from the shapes alone.
    let leaves = ((1usize << num_vars) << 2) >> log_folding;
    let tree_bytes = (2 * leaves as u64 - 1) * 32;

    // (a) This codeword's OWN promise covers its codeword — no global involved,
    // so this half would hold even without the lock.
    let held = codeword.reserved_bytes();
    assert!(
        held >= codeword_bytes,
        "the reservation holds {held} B, which does not cover the {codeword_bytes} B codeword"
    );

    // (b) …and the global grew by exactly that, as a delta.
    let grown = be.reserved_bytes() - before;
    assert_eq!(
        grown, held,
        "this codeword's promise and the global's growth must be the same bytes"
    );

    // (c) ★ and a TREE is not in the promise. The codeword has been committed,
    // so a kept tree would be sitting in this number; `paths` below builds a
    // second one, and neither may appear. This is the half that H4's version of
    // this test asserted the other way round.
    let after_open = {
        let _ = codeword.paths(log_folding, &[0, 1], hash).expect("paths");
        codeword.reserved_bytes()
    };
    assert_eq!(
        after_open, held,
        "a tree was added to the reservation: {after_open} B against {held} B, \
         and a tree here is {tree_bytes} B"
    );

    // (d) Dropping it gives every byte back. The irreducibly global
    // proposition, and the reason this test holds the lock.
    drop(codeword);
    assert_eq!(
        be.reserved_bytes(),
        before,
        "dropping the codeword must return the accounting to its baseline"
    );
}

/// ★★★ (4) THE GUARD, AT GROUP SCALE. Four commitments, none opened, hold
/// four codewords and nothing else.
///
/// This is the test H4 needed and did not have. The unit test that shipped
/// dropped ONE bare codeword and asserted the accounting returned to baseline;
/// it passed on the leaking prover, because an O(chains) peak is not a state
/// one codeword can be in and because the accounting it read is blind to bytes
/// nobody promised. Four LIVE, UNOPENED commitments is the state the group
/// actually reaches — `StackedCommitment::commit` builds all of them before the
/// first opening — and the driver's own free-memory count is the instrument
/// that cannot be fooled by where the retention is written.
///
/// Numbers, from the shapes: a codeword is `2^20 << 2` u64 = 32 MiB, its tree
/// `(2*2^18 - 1) * 32` B = 16 MiB. Four codewords = 128 MiB; four codewords and
/// their kept trees = 192 MiB. The bound is 160 MiB, a full codeword of slack
/// above the first and a full codeword below the second.
///
/// Keccak because this is a memory proposition and the two hash families build
/// identically shaped trees; the cheaper kernel keeps the test short.
#[test]
fn a_group_holds_only_its_codewords_before_any_open() {
    let _exclusive = exclusive();
    let be = math_cuda::device::backend().expect("a device");
    let hash = key::<KeccakWhir>();
    let num_vars = 20;
    let log_folding = 4;

    // ⚠ Without this the measurement is the POOL's, not the caller's: the
    // stream-ordered allocator keeps freed blocks, and a sibling test's 32 MiB
    // would silently serve one of the four commits below. Drain, then sample.
    math_cuda::device::drain_and_trim().expect("drain");
    let free_before = be.free_vram_bytes().expect("cuMemGetInfo");

    let held: Vec<_> = (0..4)
        .map(|_| commit_on_device(num_vars, log_folding, hash))
        .collect();

    let free_after = be.free_vram_bytes().expect("cuMemGetInfo");
    let taken = free_before.saturating_sub(free_after);

    let codeword_bytes = ((1u64 << num_vars) << 2) * 8;
    let bound = 5 * codeword_bytes;
    assert!(
        taken < bound,
        "four unopened commitments took {} MiB from the device; four codewords \
         are {} MiB and the bound is {} MiB, so something is being kept per \
         commitment — a tree is {} MiB",
        taken / (1 << 20),
        (4 * codeword_bytes) / (1 << 20),
        bound / (1 << 20),
        ((2 * ((1u64 << num_vars) << 2 >> log_folding) - 1) * 32) / (1 << 20),
    );

    // The commitments are alive up to here, which is the whole point: a `drop`
    // any earlier and the assertion would be about a group that had already
    // been released.
    drop(held);
}

/// ★ (5) THE BLOCKING IS THE ONE THAT WAS ASKED FOR.
///
/// The dangerous outcome a cache made reachable — serving a tree that answers a
/// different question, whose paths are internally consistent and wrong — is
/// unreachable once nothing is kept, and this pins that it stays unreachable:
/// each call builds for the `log_folding` it was given, and the shapes differ.
#[test]
fn a_tree_is_built_for_the_blocking_that_is_asked_for() {
    let _exclusive = exclusive();
    let hash = key::<RpxWhir>();
    let (codeword, _root) = commit_on_device(14, 4, hash);
    assert_eq!(codeword.tree_builds(), 1);

    // Same codeword, different blocking: its own tree, its own pass.
    let at_two = codeword.paths(2, &[0, 1], hash).expect("paths at k=2");
    assert_eq!(
        codeword.tree_builds(),
        2,
        "the opening must build a tree for the blocking it was given"
    );

    // And the rebuild answered the question that was asked: at k=2 the tree has
    // four times the leaves, so each path is two levels deeper.
    let at_four = codeword.paths(4, &[0, 1], hash).expect("paths at k=4");
    assert_eq!(codeword.tree_builds(), 3, "and a third for the k=4 opening");
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

/// ✓ The PROCESS-WIDE counter tracks the same passes — it is what the bench
/// prints, so it needs a test of its own.
///
/// Takes the lock across the whole window, because every other test in this
/// binary commits too and the counter cannot tell whose work it is counting.
/// That is exactly why the assertions above do not use it.
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
        after_commit + 1,
        "an opening builds a tree, so the global counter must move by one"
    );
    assert_eq!(
        codeword.tree_builds(),
        leaf_hash_calls(),
        "with one codeword in flight the two counters must agree"
    );
}
