//! ★★ H4's result of record, and what replaced it — a commitment keeps its tree
//! INSIDE ITS PROMISE: the whole tree by default, its leaf layer under
//! `LFM_WHIR_WHOLE_TREES=0`, and nothing of a tree outside the room it grew.
//!
//! ⛔ H4's finding stands and is not edited away below: keeping the whole node
//! array LOST, measured on the card at about +15 s, because its trees sat
//! OUTSIDE any promise — one per commitment IN THE GROUP, ten of them put the
//! device at 96%, and commits fell back to the host. What changed is WHERE the
//! bytes sit. The leaf layer came first (half the bytes, most of the saving: a
//! leaf absorbs a whole `2^k` coset while an inner node absorbs two digests),
//! promised through the codeword's room and evictable; the whole tree came
//! back on the same terms (job 248, −1.00 s). These tests pin BOTH sides of the
//! invariant: what is kept is held AND promised, and nothing of a tree is held
//! outside the promise.
//!
//! Needs a GPU:
//!
//! ```text
//! cargo test -p math-cuda --release --test whir_tree_cache -- --nocapture
//! ```
//!
//! # What this is about
//!
//! `commit()` builds the tree and takes the root; `paths()` needs the same
//! tree to read a kilobyte per query out of it. Every commitment on this path
//! is opened, so without retention every one pays for two leaf-hash passes over
//! its codeword — ~25 s of RPX device hashing on a real block, about half of it
//! that second pass. H4 cached the first tree to remove the second pass. It was
//! measured on the card and it LOST, ~+15 s in both hashes.
//!
//! # Why H4's tree could not work, and what the tests pin instead
//!
//! Not because the cache missed — it returned exactly the hashing it promised.
//! Because the retention is one tree per commitment IN THE GROUP, not one tree.
//! `StackedCommitment::commit` builds every chain's commitment before it
//! returns, since all the roots enter the transcript before any query index is
//! drawn, and the openings follow one chain at a time. So the last chain's tree
//! lives from its commit to the end of the proof, and all N exist before the
//! first opening. Held outside the ledger, ten chains at half a gigabyte put
//! the card at 96%, after which device allocations fail, commits fall back to
//! the host, and the host grows ~1.5 GiB per fallen-back chain.
//!
//! Held INSIDE the ledger — grown into each codeword's room, and given back to
//! any request that needs the bytes — the same window costs nothing but the
//! re-hash of whatever a request took. That is the trade these tests pin.
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
//! to a unit test that passed. So the guard samples `free_vram_bytes()` across
//! four live, unopened commitments and asserts two things: what they took is
//! their codewords and what each keeps of its tree, and that equals what they
//! PROMISED, give or take the pool. It fails if anything of a tree is ever held
//! outside the promise, wherever the holding is written.

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use math_cuda::DeviceHash;
use math_cuda::whir::{leaf_hash_calls, retention_report, tree_builds};
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
        DeviceHashKey::Poseidon1 => DeviceHash::Poseidon1,
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

/// `LFM_WHIR_WHOLE_TREES` forced for one test — whole trees or leaf layers —
/// and the environment's setting restored when this drops, on a panic too.
/// Every test that asserts a count or a byte figure that depends on what is
/// kept runs both ways under it, rather than at whatever the default is.
struct WholeTrees;

impl WholeTrees {
    fn force(on: bool) -> Self {
        math_cuda::whir::force_whole_trees(Some(on));
        Self
    }
}

impl Drop for WholeTrees {
    fn drop(&mut self) {
        math_cuda::whir::force_whole_trees(None);
    }
}

/// What a codeword over `2^num_vars` values at `log_blowup = 2` keeps of its
/// tree at `log_folding`: the whole node array, or its leaf layer.
fn kept_bytes(num_vars: usize, log_folding: usize, whole: bool) -> u64 {
    let leaves = ((1u64 << num_vars) << 2) >> log_folding;
    if whole {
        (2 * leaves - 1) * 32
    } else {
        leaves * 32
    }
}

/// ★★ (1) THE COUNT. One leaf-hash pass per codeword while what it keeps is
/// held, and a tree built at an opening only when no whole tree is kept.
///
/// This is the cost H4 tried to remove, asserted as integers in both retention
/// modes. With leaf layers (`LFM_WHIR_WHOLE_TREES=0`) each opening builds its
/// own tree on the kept layer: `tree_builds` climbs, `leaf_passes` stays at 1.
/// With whole trees (the default) each opening is served the kept tree: both
/// stay at 1. A 2 in `leaf_passes` is the retention not working in either mode;
/// a build at an opening under whole trees is an opening that stopped reading
/// the tree it kept; and the kept bytes say which object the counts are about.
#[test]
fn a_commitment_hashes_its_leaves_once_per_tree_it_builds() {
    let _exclusive = exclusive();
    for whole in [false, true] {
        let _mode = WholeTrees::force(whole);
        // Trees an opening builds: one with leaf layers, none with whole trees.
        let per_open = u64::from(!whole);
        for (name, hash) in [("keccak", key::<KeccakWhir>()), ("rpx", key::<RpxWhir>())] {
            let (codeword, _root) = commit_on_device(14, 4, hash);
            assert_eq!(
                codeword.tree_builds(),
                1,
                "{name}, whole trees {whole}: the commit itself must hash the leaves exactly once"
            );
            assert_eq!(
                codeword.leaf_passes(),
                1,
                "{name}, whole trees {whole}: the commit hashes the leaves once"
            );

            let _ = codeword.paths(4, &[0, 1, 7], hash).expect("paths");
            assert_eq!(
                codeword.tree_builds(),
                1 + per_open,
                "{name}, whole trees {whole}: an opening builds a tree only when no \
                 whole tree is kept"
            );
            // ★ THE OTHER DIRECTION, and it is the whole point of the retention:
            // the leaves were not re-hashed, whether the opening built on the
            // kept layer or read the kept tree. A 2 here is the retention not
            // working.
            assert_eq!(
                codeword.leaf_passes(),
                1,
                "{name}, whole trees {whole}: the opening must reuse what was kept — \
                 a 2 here means nothing was kept, or it was not matched"
            );

            // …and again, because a cache that served once and then evicted
            // would read 2 on the line above too.
            let _ = codeword.paths(4, &[2, 3], hash).expect("paths");
            assert_eq!(
                codeword.tree_builds(),
                1 + 2 * per_open,
                "{name}, whole trees {whole}: and the second opening likewise"
            );
            assert_eq!(
                codeword.leaf_passes(),
                1,
                "{name}, whole trees {whole}: and still one leaf pass — what served \
                 once and was then evicted would read 2 here"
            );
            assert_eq!(
                codeword.retained_leaf_bytes(),
                kept_bytes(14, 4, whole),
                "{name}, whole trees {whole}: the codeword keeps the wrong object, so \
                 the counts above are agreeing about the wrong thing"
            );
        }
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
                verify_opening::<_, H>(&root, host.depth(), index, &opening),
                "{name}: opening {index} does not verify against the device root"
            );
        }
    }
    check::<KeccakWhir>("keccak");
    check::<RpxWhir>("rpx");
}

/// ★★ (3) THE RESERVATION SEES THE CODEWORD AND WHAT IT KEEPS, AND GETS IT
/// BACK.
///
/// Device memory held outside the accounting is an under-count that nothing
/// reports until a second prover shares the card. What a codeword promises is
/// its own bytes, the folds that halve it, and what it keeps of its tree — the
/// whole tree or its leaf layer, grown into the promise when kept. A tree an
/// opening builds without keeping it is a transient of that call and never
/// enters the promise. Both retention modes, forced.
#[test]
fn the_codeword_is_inside_the_reservation_and_gives_it_back() {
    let _exclusive = exclusive();
    let be = math_cuda::device::backend().expect("a device");
    let hash = key::<RpxWhir>();
    let num_vars = 14;
    let log_folding = 4;
    let codeword_bytes = ((1u64 << num_vars) << 2) * 8;

    for whole in [false, true] {
        let _mode = WholeTrees::force(whole);
        // ⚠ Read under the lock, and it is a BASELINE rather than an assumed
        // zero: a sibling's reservation was what failed this test the first
        // time it ran on a card (786,400 B of someone else's). What is asserted
        // below is the DELTA this codeword is responsible for.
        let before = be.reserved_bytes();
        let (codeword, _root) = commit_on_device(num_vars, log_folding, hash);

        // (a) This codeword's OWN promise covers its codeword — no global
        // involved, so this half would hold even without the lock.
        let held = codeword.reserved_bytes();
        assert!(
            held >= codeword_bytes,
            "whole trees {whole}: the reservation holds {held} B, which does not cover the \
             {codeword_bytes} B codeword"
        );

        // (b) …and the global grew by exactly that, as a delta.
        let grown = be.reserved_bytes() - before;
        assert_eq!(
            grown, held,
            "whole trees {whole}: this codeword's promise and the global's growth must be \
             the same bytes"
        );

        // (c) ★ What the commit kept of its tree is INSIDE the promise — H4's
        // trees were the bytes that were not — and it is the object the mode
        // names: the whole tree, or the leaf layer.
        let kept = codeword.retained_leaf_bytes();
        assert_eq!(
            kept,
            kept_bytes(num_vars, log_folding, whole),
            "whole trees {whole}: the commit kept the wrong object"
        );
        assert!(
            held >= codeword_bytes + kept,
            "whole trees {whole}: the promise ({held} B) does not cover the codeword \
             ({codeword_bytes} B) and what it keeps ({kept} B): a kept tree is outside it"
        );

        // (d) …and an opening adds nothing to it: it reads the kept tree, or
        // builds a transient one on the kept layer and frees it on return.
        let after_open = {
            let _ = codeword.paths(log_folding, &[0, 1], hash).expect("paths");
            codeword.reserved_bytes()
        };
        assert_eq!(
            after_open, held,
            "whole trees {whole}: an opening changed the reservation, {after_open} B against \
             {held} B"
        );

        // (e) Dropping it gives every byte back, kept tree included. The
        // irreducibly global proposition, and the reason this test holds the
        // lock.
        drop(codeword);
        assert_eq!(
            be.reserved_bytes(),
            before,
            "whole trees {whole}: dropping the codeword must return the accounting to its \
             baseline"
        );
    }
}

/// ★★★ (4) THE GUARD, AT GROUP SCALE. Four commitments, none opened, hold four
/// codewords and what each keeps of its tree — exactly what they promised — and
/// nothing outside the promise.
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
/// # The invariant, and its margins
///
/// A codeword here is `2^20 << 2` u64 = 32 MiB. At `log_folding = 2` its tree
/// is `2^20` leaves: a leaf layer is 32 MiB — one codeword — and a whole tree
/// `(2·2^20 − 1)·32` B ≈ 64 MiB — two — which is the whole reason for that
/// blocking. Both retention modes, forced, each asserting three things:
///
/// - **What is kept is the object the mode names, and it is promised.** Each
///   codeword keeps its leaf layer or its whole tree, and the ledger grew by
///   EXACTLY four codewords and those four objects (a commit alone promises its
///   codeword, and a capture grows that promise by what it keeps).
/// - **Nothing is held outside the promise.** The driver's count less the
///   ledger's — the pool's share, after a symmetric drain — stays under one
///   tree and one codeword (96 MiB). H4's shape, a tree kept outside the
///   promise, would put four trees there (256 MiB); a leaf layer kept outside
///   it, four codewords (128). The allowance is the whole-node-array mutation
///   run's pool share after the symmetric drain, 64 MiB — a fragmentation floor
///   of one largest transient, which a best-effort trim cannot release — plus a
///   codeword.
/// - **What is kept is really held.** The driver took more than the next
///   smaller retention would: four codewords alone (128 MiB) under leaf layers,
///   four codewords and their layers (256 MiB) under whole trees. Under leaf
///   layers it also took less than four codewords, their layers and one
///   codeword (288 MiB) — the bound this guard carried before whole trees,
///   kept as it was.
///
/// ⚠ The slack is not decoration. `free_vram_bytes` reports what the PROCESS
/// has taken from the driver, which includes whatever one-time workspace and
/// twiddle caches the first commit of this size sets up, and those are counted
/// identically in both cases. The warm-up commit below pays those costs before
/// the sample, and the margins absorb what it misses.
///
/// Keccak because this is a memory proposition and the two hash families build
/// identically shaped trees; the cheaper kernel keeps the test short.
#[test]
fn a_group_holds_only_its_codewords_before_any_open() {
    let _exclusive = exclusive();
    let be = math_cuda::device::backend().expect("a device");
    let hash = key::<KeccakWhir>();
    let num_vars = 20;
    // ⚠ Not 4. At `log_folding = 2` the tree is TWO codewords rather than half
    // of one, which is what puts a codeword or more between every case below.
    let log_folding = 2;
    let codeword_bytes = ((1u64 << num_vars) << 2) * 8;
    let leaf_bytes = kept_bytes(num_vars, log_folding, false);
    let tree_bytes = kept_bytes(num_vars, log_folding, true);
    let mib = |b: u64| b / (1 << 20);

    for whole in [false, true] {
        let _mode = WholeTrees::force(whole);
        let kept = if whole { tree_bytes } else { leaf_bytes };

        // One commit of this exact shape before the sample, so the one-time
        // costs of the first — twiddles, workspaces, whatever the pool grows to
        // hold them — are paid outside the window and not attributed to
        // retention.
        drop(commit_on_device(num_vars, log_folding, hash));

        // ⚠ And without this the measurement is the POOL's, not the caller's:
        // the stream-ordered allocator keeps freed blocks, and the warm-up's own
        // codeword would silently serve one of the four commits below. Drain,
        // then sample.
        math_cuda::device::drain_and_trim().expect("drain");
        let free_before = be.free_vram_bytes().expect("cuMemGetInfo");
        // ★ THE SECOND INSTRUMENT, and it is the one that can tell the two
        // stories apart. `free_vram_bytes` is the DRIVER's count and includes
        // whatever the pool is sitting on; `reserved_bytes` is what the CODE
        // promised and is blind to the pool by construction. Their difference is
        // the pool's, and it is what "nothing outside the promise" is asserted on.
        let reserved_before = be.reserved_bytes();

        let held: Vec<_> = (0..4)
            .map(|_| commit_on_device(num_vars, log_folding, hash))
            .collect();

        // ⛔ SYMMETRIC SAMPLING. `free_before` is taken AFTER a drain, so this
        // one must be too: a delta between two samples is about the code only if
        // both are taken at the same pool state. The run of 2026-09-20 read
        // EXACTLY nine codewords on a bound of nine before this drain existed —
        // the pool's transients, not the code's holding — and every peak-demand
        // model missed on both sides, the signature of driver suballocation.
        math_cuda::device::drain_and_trim().expect("drain");
        let free_after = be.free_vram_bytes().expect("cuMemGetInfo");
        let taken = free_before.saturating_sub(free_after);
        let promised = be.reserved_bytes().saturating_sub(reserved_before);
        let pool_share = taken.saturating_sub(promised);

        // ★ THE TWO ACCOUNTINGS, SIDE BY SIDE, PRINTED EVERY TIME under
        // `--nocapture` (which is how the box gate runs this suite), so the
        // honest path's numbers are READ rather than inferred from a pass.
        let ledger = {
            let per: Vec<String> = held
                .iter()
                .map(|(c, _)| {
                    format!(
                        "{}/{}",
                        mib(c.reserved_bytes()),
                        mib(c.retained_leaf_bytes())
                    )
                })
                .collect();
            format!(
                "whole trees {whole}: driver {} MiB · promised {} MiB · pool share {} MiB · \
                 per codeword reserved/retained MiB: [{}]",
                mib(taken),
                mib(promised),
                mib(pool_share),
                per.join(", ")
            )
        };
        println!("   group guard: {ledger}");

        // (i) The object the mode names, and its promise.
        for (c, _) in &held {
            assert_eq!(
                c.retained_leaf_bytes(),
                kept,
                "whole trees {whole}: a codeword keeps {} B where the mode keeps {} B\n   {}",
                c.retained_leaf_bytes(),
                kept,
                ledger,
            );
        }
        assert_eq!(
            promised,
            4 * (codeword_bytes + kept),
            "whole trees {whole}: four commits must promise exactly four codewords and \
             what each keeps\n   {ledger}"
        );

        // (ii) ⛔ Nothing held outside the promise.
        assert!(
            pool_share < tree_bytes + codeword_bytes,
            "whole trees {whole}: the device holds {} MiB beyond what the four commits \
             promised. A tree kept outside the promise, per commitment, reads {} MiB \
             here; a leaf layer, {} MiB.\n   {}",
            mib(pool_share),
            mib(4 * tree_bytes),
            mib(4 * leaf_bytes),
            ledger,
        );

        // (iii) What is kept is really held.
        let floor = if whole {
            4 * (codeword_bytes + leaf_bytes)
        } else {
            4 * codeword_bytes
        };
        assert!(
            taken > floor,
            "whole trees {whole}: four unopened commitments took only {} MiB, at or under \
             the {} MiB of the next smaller retention. What the mode keeps ({} MiB for \
             four) is NOT being held — the capture never ran or the budget refused it, \
             and every opening will pay for it again.\n   {}",
            mib(taken),
            mib(floor),
            mib(4 * kept),
            ledger,
        );
        if !whole {
            let bound = 4 * (codeword_bytes + leaf_bytes) + codeword_bytes;
            assert!(
                taken < bound,
                "leaf layers: four unopened commitments took {} MiB from the device, over \
                 the {} MiB bound. Something bigger than a leaf layer is held per \
                 commitment.\n   {}",
                mib(taken),
                mib(bound),
                ledger,
            );
        }

        // The commitments are alive up to here, which is the whole point: a
        // `drop` any earlier and the assertions would be about a group that had
        // already been released.
        drop(held);
    }
}

/// ★ (5) THE BLOCKING IS THE ONE THAT WAS ASKED FOR.
///
/// The dangerous outcome a cache makes reachable — serving a tree that answers a
/// different question, whose paths are internally consistent and wrong — stays
/// unreachable: whatever is kept is keyed by its blocking, a call at another
/// blocking builds its own tree and hashes its own leaves, and the kept object
/// still serves the blocking it was built for. Both retention modes, forced.
#[test]
fn a_tree_is_built_for_the_blocking_that_is_asked_for() {
    let _exclusive = exclusive();
    let hash = key::<RpxWhir>();
    for whole in [false, true] {
        let _mode = WholeTrees::force(whole);
        let (codeword, _root) = commit_on_device(14, 4, hash);
        assert_eq!(codeword.tree_builds(), 1);

        // Same codeword, different blocking: its own tree, its own pass.
        let at_two = codeword.paths(2, &[0, 1], hash).expect("paths at k=2");
        assert_eq!(
            codeword.tree_builds(),
            2,
            "whole trees {whole}: the opening must build a tree for the blocking it was given"
        );
        // ★ THE KEY, ASSERTED WHERE IT CAN FAIL. What was kept was built at
        // k=4; this opening is at k=2 and describes a DIFFERENT tree, so it must
        // not be served and the leaves must be hashed again. A 1 here is a cache
        // ignoring its key, which is the one way retention could hand back paths
        // that are internally consistent and wrong.
        assert_eq!(
            codeword.leaf_passes(),
            2,
            "whole trees {whole}: a k=2 opening must NOT be served what was kept at k=4"
        );

        // And the rebuild answered the question that was asked: at k=2 the tree
        // has four times the leaves, so each path is two levels deeper.
        let at_four = codeword.paths(4, &[0, 1], hash).expect("paths at k=4");
        // …and what was kept at k=4 IS still there and IS served — the whole
        // tree, with no build, or the layer, under a third tree — so the key
        // rejects a mismatch without throwing away a match. Without these lines
        // the test above would also pass on a cache that had simply stopped
        // working.
        assert_eq!(
            codeword.tree_builds(),
            if whole { 2 } else { 3 },
            "whole trees {whole}: the k=4 opening builds a tree only when no whole tree is kept"
        );
        assert_eq!(
            codeword.leaf_passes(),
            2,
            "whole trees {whole}: the k=4 opening matches what was kept and must not re-hash"
        );
        assert_eq!(
            at_two.len(),
            at_four.len() + 2 * 2 * 32,
            "whole trees {whole}: a k=2 tree's paths must be two levels deeper than a k=4 tree's"
        );
    }
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

    // ⛔ DELTAS, NOT ABSOLUTES. These counters are process-wide and not
    // resettable as a set, and the retention makes them diverge on purpose, so
    // the assertions below are about what THIS codeword moved.
    let at = || {
        (
            tree_builds(),
            leaf_hash_calls(),
            retention_report().5,
            math_cuda::whir::retention_whole().1,
        )
    };

    for whole in [false, true] {
        let _mode = WholeTrees::force(whole);
        let (b0, p0, s0, w0) = at();
        let (codeword, _root) = commit_on_device(12, 4, hash);
        let (b1, p1, s1, w1) = at();
        assert_eq!(
            b1 - b0,
            1,
            "whole trees {whole}: one commit, one tree assembled"
        );
        assert_eq!(
            p1 - p0,
            1,
            "whole trees {whole}: and it paid for its own leaf pass"
        );
        assert_eq!(
            s1 - s0,
            0,
            "whole trees {whole}: with nothing yet in hand to reuse"
        );
        assert_eq!(w1 - w0, 0, "whole trees {whole}: and nothing served");

        let _ = codeword.paths(4, &[0, 1], hash).expect("paths");
        let (b2, p2, s2, w2) = at();
        // ★ THE LINES THAT CHANGED WITH H4's REPLACEMENTS. A tree and a leaf
        // pass were once the same event. With the leaf layer kept, the opening
        // assembles a tree and does not re-hash the leaves; with the whole tree
        // kept, it assembles nothing and is served. A leaf pass here, in either
        // mode, is the retention failing to serve.
        assert_eq!(
            b2 - b1,
            u64::from(!whole),
            "whole trees {whole}: the opening assembles a tree only when no whole tree is kept"
        );
        assert_eq!(
            p2 - p1,
            0,
            "whole trees {whole}: and does NOT re-hash the leaves"
        );
        assert_eq!(
            (s2 - s1, w2 - w1),
            if whole { (0, 1) } else { (1, 0) },
            "whole trees {whole}: the saving is counted where it happens (layer, served)"
        );

        // ⭐ THE ACCOUNTING IDENTITIES, which fail in BOTH directions: every
        // tree assembled either paid for its leaf pass or reused a kept layer,
        // and every call that wanted a tree either assembled one or was served
        // a kept one.
        assert_eq!(
            b2 - b0,
            (p2 - p0) + (s2 - s0),
            "whole trees {whole}: every tree either paid for its leaf pass or reused one; \
             trees {}, passes {}, savings {}",
            b2 - b0,
            p2 - p0,
            s2 - s0
        );
        assert_eq!(
            (b2 - b0) + (w2 - w0),
            2,
            "whole trees {whole}: the commit and the opening each assembled a tree or were served one"
        );
    }
}

/// ⛔ THE ARGUE-SURFACE DEVICE-FALLBACK COUNTER FIRES AT A REAL SITE.
///
/// wt16 read as a win at the slot level because argue's per-table device work
/// fell back to the host UNCOUNTED — `multilinear::gpu::host_fallbacks()`
/// counts the COMMIT path only. `math_cuda::device::device_fallbacks()` is the
/// counter that closes that blind spot; this test proves it actually moves when
/// an argue-surface `reserve` is refused, using the cheapest of the five sites,
/// `DeviceColumns::upload`.
///
/// It forces the refusal WITHOUT a budget setter and WITHOUT allocating any
/// device memory: `Backend::reserve` is a pure atomic bump on the reservation
/// total (no `cuMemAlloc`), so reserving the whole remaining budget makes every
/// later `reserve` return `None` at no memory cost. The reservation is dropped
/// at the end, giving the budget back.
///
/// The gate mutation is deleting the `note_device_fallback()` call at
/// `columns.rs`'s `reserve`→`None` site: the count then reads 0 and the final
/// assertion reddens by name. The four undriven sites are covered card-free by
/// `note_device_fallback_is_called_at_exactly_the_five_argue_sites` in
/// `device.rs`.
#[test]
fn an_argue_reservation_refusal_bumps_the_device_fallback_counter() {
    let _exclusive = exclusive();
    let be = math_cuda::device::backend().expect("device fallback test needs a GPU");

    // Take the whole remaining budget as one reservation — an atomic bump, no
    // device memory — so any further `reserve` must be refused. Held to the end
    // of the test, then dropped.
    let remaining = be.vram_budget_bytes().saturating_sub(be.reserved_bytes());
    let _hog = math_cuda::device::reserve(remaining)
        .expect("reserving the remaining budget is an accounting move and cannot fail");
    assert_eq!(
        be.reserved_bytes(),
        be.vram_budget_bytes(),
        "the budget is now fully promised, so the next reserve must be refused"
    );

    math_cuda::device::reset_device_fallbacks();
    assert_eq!(
        math_cuda::device::device_fallbacks(),
        0,
        "the counter starts this measurement at zero"
    );

    // The cheapest argue site: one column of one element wants 8 bytes the
    // budget cannot promise, so `upload` returns `None` at its `reserve` and
    // the site records the fallback.
    let refused = math_cuda::columns::DeviceColumns::upload(&[&[0u64]]);
    assert!(
        refused.is_none(),
        "with the budget fully promised, the device upload must decline"
    );
    assert_eq!(
        math_cuda::device::device_fallbacks(),
        1,
        "an argue-surface reserve was refused, so the device-fallback counter \
         must read exactly one — a 0 here is the counter not wired to the site"
    );

    // `_hog` is dropped here at scope end, returning the reserved budget.
}

/// ⛔ THE LIVE FOOTPRINT AND RESERVED HIGH-WATER TRACK THE RETENTION.
///
/// The two instruments the evictable retention is built on: `retained_bytes_live`
/// (the SIMULTANEOUS footprint, not the cumulative `held`) and `reserved_high_water`
/// (the peak `be.reserved`, the quantity argue's `reserve` is checked against).
/// This proves both move with a real retention and, crucially, that the layer's
/// bytes are GIVEN BACK when its codeword drops — the balance the eviction relies
/// on. A broken `KeptNodes` drop leaves the live count high and reddens the
/// final assertion; a missing `note_reserved` leaves the high-water below the
/// live reserved total and reddens the middle one.
#[test]
fn the_live_footprint_and_reserved_high_water_track_the_retention() {
    let _exclusive = exclusive();
    let be = math_cuda::device::backend().expect("footprint test needs a GPU");
    let live_before = math_cuda::whir::retained_bytes_live();
    {
        let (codeword, _root) = commit_on_device(14, 4, key::<RpxWhir>());
        let _ = codeword
            .paths(4, &[0, 1, 7], key::<RpxWhir>())
            .expect("paths");
        // A leaf layer is captured and held on the codeword.
        let live = math_cuda::whir::retained_bytes_live();
        assert!(
            live > live_before,
            "a held leaf layer must raise the live footprint (live {live}, before {live_before})"
        );
        assert!(
            math_cuda::whir::retained_bytes_peak() >= live,
            "the peak footprint must be at least the current live count"
        );
        assert!(
            math_cuda::device::reserved_high_water() >= be.reserved_bytes(),
            "the reservation high-water must be at least the current reserved total \
             — note_reserved must fire on every rise of be.reserved"
        );
    }
    // The codeword — and, synchronously in its Drop, the retained layer — is gone.
    assert_eq!(
        math_cuda::whir::retained_bytes_live(),
        live_before,
        "dropping the codeword must give the layer's bytes back: KeptNodes's drop \
         balances the admit, or the live footprint would only ever rise"
    );
}

/// ⛔ A BUDGET MISS EVICTS A RETAINED LAYER AND THE RESERVE THEN SUCCEEDS.
///
/// The whole point of the evictable retention: when a real caller (argue, here a
/// bare reserve) cannot get its bytes, the retention gives a layer back rather
/// than the caller falling to the host. Committing captures ONE base layer (no
/// folds), so exactly one layer is in the registry. Then the budget is filled to
/// leave LESS free than the layer's bytes, so a reserve of the layer's size must
/// miss — and it SUCCEEDS only because the evictor reclaims the layer. The
/// mutation that disables the evictor (skip the consult in reserve, or never
/// install it) makes this reserve return None and the test panic on the expect.
#[test]
fn a_budget_miss_evicts_a_retained_layer_and_the_reserve_succeeds() {
    let _exclusive = exclusive();
    let be = math_cuda::device::backend().expect("eviction test needs a GPU");

    // Commit captures the base leaf layer; hold the codeword so the layer stays.
    let (codeword, _root) = commit_on_device(14, 4, key::<RpxWhir>());
    let layer_bytes = codeword.retained_leaf_bytes();
    assert!(
        layer_bytes > 0,
        "precondition: the commit must have retained a layer"
    );
    let (evictions_before, _) = math_cuda::whir::retention_evictions();

    // Fill the budget to leave a gap SMALLER than the layer, so a reserve of the
    // layer's size cannot fit without eviction. reserve is a pure atomic bump,
    // so the hog costs no device memory.
    let gap = layer_bytes / 2;
    let hog_bytes = be
        .vram_budget_bytes()
        .saturating_sub(be.reserved_bytes())
        .saturating_sub(gap);
    let _hog = math_cuda::device::reserve(hog_bytes).expect("the hog reservation cannot fail");
    assert!(
        be.vram_budget_bytes().saturating_sub(be.reserved_bytes()) < layer_bytes,
        "the free budget must now be below the layer size, so the next reserve misses"
    );

    // This would return None without the evictor; with it, the layer is freed
    // and the reserve succeeds.
    let got = math_cuda::device::reserve(layer_bytes).expect(
        "the reserve must SUCCEED by evicting the retained layer — a None here \
                 is the evictor not consulted, or eviction freeing nothing",
    );

    let (evictions_after, bytes_evicted) = math_cuda::whir::retention_evictions();
    assert!(
        evictions_after > evictions_before,
        "an eviction must have been recorded ({evictions_before} -> {evictions_after})"
    );
    assert!(
        bytes_evicted >= layer_bytes,
        "the eviction must have freed at least the layer's bytes"
    );
    assert_eq!(
        codeword.retained_leaf_bytes(),
        0,
        "the evicted codeword's layer slot must read None (the sole registered layer)"
    );

    drop(got);
    drop(_hog);
}

/// ⛔ A MISS THE LAYERS CANNOT COVER KEEPS THEM UNDER `LFM_WHIR_KEEP_FUTILE`, AND
/// MOVES NO PATH EITHER WAY.
///
/// The budget is filled to leave half a layer free, and the request asks for
/// two layers more than that. Evicting the one layer held cannot cover the
/// deficit, so the reserve fails whatever the evictor does. With the switch off
/// the layer goes anyway and the opening hashes its leaves again; on, it stays
/// and the opening is served from it. Both openings give the paths of a fresh
/// codeword over the same evaluations: the switch decides where a leaf layer
/// comes from, never what it is. A switch the evictor ignores, or a walk that
/// no longer asks whether the miss is futile, reddens the kept arm's eviction
/// count.
#[test]
fn a_miss_the_layers_cannot_cover_keeps_them_and_moves_no_path() {
    let _exclusive = exclusive();
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            math_cuda::whir::force_keep_futile(None);
        }
    }
    let _restore = Restore;
    let be = math_cuda::device::backend().expect("futile-miss test needs a GPU");
    let hash = key::<RpxWhir>();
    let (num_vars, log_folding) = (14, 4);
    let raw: Vec<u64> = poly(num_vars).evals().iter().map(|v| *v.value()).collect();
    let leaves = (raw.len() << 2) >> log_folding;
    let positions: Vec<u32> = [0usize, 1, leaves / 3, leaves - 1]
        .iter()
        .map(|p| *p as u32)
        .collect();
    let (reference, reference_root) =
        math_cuda::whir::commit_codeword(&raw, 2, log_folding, false, hash)
            .expect("device commit (needs a GPU)");
    let want = reference
        .paths(log_folding, &positions, hash)
        .expect("paths");
    // Its layer must not be the one a miss below could reclaim.
    drop(reference);

    for keep in [false, true] {
        math_cuda::whir::force_keep_futile(Some(keep));
        let (codeword, root) = math_cuda::whir::commit_codeword(&raw, 2, log_folding, false, hash)
            .expect("device commit");
        assert_eq!(root, reference_root, "keep {keep}: the root moved");
        let layer = codeword.retained_leaf_bytes();
        assert!(
            layer > 0,
            "keep {keep}: precondition: the commit must have retained a layer"
        );
        let (evictions_before, _) = math_cuda::whir::retention_evictions();
        let (futile_before, _, _) = math_cuda::whir::retention_futile();

        // Half a layer free; the request's deficit is then two layers, against
        // the one layer the evictor could give back.
        let gap = layer / 2;
        let hog_bytes = be
            .vram_budget_bytes()
            .saturating_sub(be.reserved_bytes())
            .saturating_sub(gap);
        let hog = math_cuda::device::reserve(hog_bytes).expect("the hog reservation cannot fail");
        let miss = math_cuda::device::reserve(gap + 2 * layer);
        assert!(
            miss.is_none(),
            "keep {keep}: a miss no eviction can cover must fail either way"
        );
        drop(hog);

        let (evictions_after, _) = math_cuda::whir::retention_evictions();
        let (futile_after, _, kept) = math_cuda::whir::retention_futile();
        assert_eq!(kept, keep, "the switch must read back as forced");
        assert_eq!(
            futile_after,
            futile_before + 1,
            "keep {keep}: the miss must be counted as futile exactly once"
        );
        if keep {
            assert_eq!(
                evictions_after, evictions_before,
                "kept: a futile miss must evict nothing"
            );
            assert_eq!(
                codeword.retained_leaf_bytes(),
                layer,
                "kept: the layer must still be held"
            );
        } else {
            assert!(
                evictions_after > evictions_before,
                "off: the layer goes, as it did before the switch"
            );
            assert_eq!(
                codeword.retained_leaf_bytes(),
                0,
                "off: the layer must be gone"
            );
        }

        let passes = codeword.leaf_passes();
        let got = codeword
            .paths(log_folding, &positions, hash)
            .expect("paths");
        assert_eq!(
            codeword.leaf_passes(),
            passes + u64::from(!keep),
            "keep {keep}: the opening is served when the layer was kept and hashes it again when not"
        );
        assert_eq!(got, want, "keep {keep}: the paths moved");
    }
}

/// ⛔ A WHOLE TREE KEPT UNDER `LFM_WHIR_WHOLE_TREES` SERVES ITS OPENINGS AND
/// MOVES NO BYTE; EVICTED, THE NEXT OPENING BUILDS THE SAME TREE AND KEEPS IT
/// WHOLE AGAIN.
///
/// The reference is a codeword opened with the switch off: its root, its paths,
/// and its paths with their Merkle cap. With the switch on the commit keeps the
/// whole node array, and two openings — the paths, then the paths and the cap —
/// build nothing (`tree_builds` stays 1) and give the reference's bytes. Then a
/// reserve the kept tree can cover evicts it and succeeds, as a layer's would,
/// and the next opening hashes the leaves again, gives the same paths and keeps
/// the tree whole again. An opening that stopped reading the kept tree reddens
/// the build count; a capture that kept only the leaves reddens the byte count.
#[test]
fn a_kept_whole_tree_serves_its_openings_and_moves_no_byte() {
    let _exclusive = exclusive();
    // Leaf layers for the reference; switched to whole trees below. Dropped at
    // the end, it restores the environment's setting.
    let _mode = WholeTrees::force(false);
    let be = math_cuda::device::backend().expect("whole-tree test needs a GPU");
    let hash = key::<RpxWhir>();
    let (num_vars, log_folding, cap_height) = (14, 4, 3);
    let raw: Vec<u64> = poly(num_vars).evals().iter().map(|v| *v.value()).collect();
    let leaves = (raw.len() << 2) >> log_folding;
    let positions: Vec<u32> = [0usize, 1, leaves / 3, leaves - 1]
        .iter()
        .map(|p| *p as u32)
        .collect();

    let (reference, reference_root) =
        math_cuda::whir::commit_codeword(&raw, 2, log_folding, false, hash)
            .expect("device commit (needs a GPU)");
    let want = reference
        .paths(log_folding, &positions, hash)
        .expect("paths");
    let want_capped = reference
        .paths_and_cap(log_folding, &positions, cap_height, hash)
        .expect("paths and cap");
    assert_eq!(
        reference.retained_leaf_bytes(),
        leaves as u64 * 32,
        "switch off: the leaf layer alone is kept"
    );
    drop(reference);

    math_cuda::whir::force_whole_trees(Some(true));
    let (codeword, root) =
        math_cuda::whir::commit_codeword(&raw, 2, log_folding, false, hash).expect("device commit");
    assert_eq!(root, reference_root, "the root moved");
    let whole = codeword.retained_leaf_bytes();
    assert_eq!(
        whole,
        (2 * leaves as u64 - 1) * 32,
        "switch on: the commit must keep the whole tree"
    );
    let (_, served_before) = math_cuda::whir::retention_whole();
    let got = codeword
        .paths(log_folding, &positions, hash)
        .expect("paths");
    let got_capped = codeword
        .paths_and_cap(log_folding, &positions, cap_height, hash)
        .expect("paths and cap");
    let (on, served_after) = math_cuda::whir::retention_whole();
    assert!(on, "the switch must read back as forced");
    assert_eq!(
        served_after,
        served_before + 2,
        "both openings must be served the kept tree"
    );
    assert_eq!(
        codeword.tree_builds(),
        1,
        "openings served a kept tree build none"
    );
    assert_eq!(codeword.leaf_passes(), 1, "and hash no leaf");
    assert_eq!(got, want, "the paths moved");
    assert_eq!(got_capped, want_capped, "the paths or the cap moved");

    // A reserve the kept tree can cover evicts it, whole, and succeeds.
    let (evictions_before, _) = math_cuda::whir::retention_evictions();
    let gap = whole / 2;
    let hog_bytes = be
        .vram_budget_bytes()
        .saturating_sub(be.reserved_bytes())
        .saturating_sub(gap);
    let hog = math_cuda::device::reserve(hog_bytes).expect("the hog reservation cannot fail");
    let room = math_cuda::device::reserve(whole)
        .expect("a miss the kept tree can cover must evict it and succeed");
    let (evictions_after, _) = math_cuda::whir::retention_evictions();
    assert!(
        evictions_after > evictions_before,
        "the kept tree must have been evicted"
    );
    assert_eq!(codeword.retained_leaf_bytes(), 0, "the tree must be gone");
    drop(room);
    drop(hog);

    let again = codeword
        .paths(log_folding, &positions, hash)
        .expect("paths");
    assert_eq!(
        codeword.tree_builds(),
        2,
        "evicted, the opening builds the tree again"
    );
    assert_eq!(codeword.leaf_passes(), 2, "and hashes its leaves again");
    assert_eq!(again, want, "the rebuilt tree gives the same paths");
    assert_eq!(
        codeword.retained_leaf_bytes(),
        whole,
        "and it is kept whole again"
    );
}

/// ⛔ A KEPT TREE AN OPENING IS READING STAYS PROMISED UNTIL THE OPENING LETS
/// GO — through an eviction and through its codeword's drop.
///
/// A served opening reads a kept whole tree through its own handle after the
/// slot's lock is let go. When the slot's drop gave the promise back, emptying
/// the slot while that opening read — an eviction, or the codeword dropping —
/// left the ledger counting less than the card held until the read ended. Now
/// the promise goes with the buffer's LAST handle (`KeptNodes`), and:
/// 1. the evictor passes over a tree an opening holds: a miss only that tree
///    could cover fails, the tree stays kept, and the ledger still counts it;
/// 2. once the opening lets go, the same miss evicts the tree and succeeds;
/// 3. a codeword dropped while an opening holds its kept tree gives back
///    nothing until the opening lets go — the tree's promise, and the room it
///    was grown in, go with the last handle — and then every byte.
///
/// An evictor that ignores readers reddens (1); a promise given back by the
/// slot's drop reddens (3).
#[test]
fn a_kept_tree_an_opening_reads_stays_promised_until_the_opening_lets_go() {
    let _exclusive = exclusive();
    let _mode = WholeTrees::force(true);
    let be = math_cuda::device::backend().expect("reader test needs a GPU");
    let hash = key::<RpxWhir>();
    let (num_vars, log_folding) = (14, 4);

    // Baselines, not assumed zeros: what this test is responsible for is the
    // delta.
    let before = be.reserved_bytes();
    let live_before = math_cuda::whir::retained_bytes_live();
    let (codeword, _root) = commit_on_device(num_vars, log_folding, hash);
    let tree = codeword.retained_leaf_bytes();
    assert_eq!(
        tree,
        kept_bytes(num_vars, log_folding, true),
        "precondition: the commit keeps its whole tree"
    );
    let with_codeword = be.reserved_bytes();

    // (1) An opening holds the tree; a miss only it could cover.
    let reader = codeword
        .hold_kept_tree(log_folding, hash)
        .expect("a kept tree to read");
    let passes_before = math_cuda::whir::retention_read_passes();
    let (evictions_before, _) = math_cuda::whir::retention_evictions();
    let gap = tree / 2;
    let hog_bytes = be
        .vram_budget_bytes()
        .saturating_sub(be.reserved_bytes())
        .saturating_sub(gap);
    let hog = math_cuda::device::reserve(hog_bytes).expect("the hog reservation cannot fail");
    let miss = math_cuda::device::reserve(tree);
    assert!(
        miss.is_none(),
        "the only kept tree is being read: nothing can cover this miss"
    );
    assert_eq!(
        codeword.retained_leaf_bytes(),
        tree,
        "the tree an opening reads must stay kept"
    );
    assert_eq!(
        math_cuda::whir::retention_evictions().0,
        evictions_before,
        "nothing may be evicted"
    );
    assert!(
        math_cuda::whir::retention_read_passes() > passes_before,
        "the evictor must count the tree it passed over"
    );
    assert_eq!(
        be.reserved_bytes(),
        with_codeword + hog_bytes,
        "the ledger must still count the tree an opening reads"
    );

    // (2) The opening lets go: the same miss evicts the tree and succeeds.
    drop(reader);
    let got = math_cuda::device::reserve(tree)
        .expect("the opening let go: the miss must evict the tree and succeed");
    assert_eq!(
        codeword.retained_leaf_bytes(),
        0,
        "the tree must have been evicted"
    );
    // The tree's bytes went back and the miss took as many again.
    assert_eq!(
        be.reserved_bytes(),
        with_codeword + hog_bytes,
        "the eviction must give back exactly the tree's bytes"
    );
    drop(got);
    drop(hog);

    // (3) The codeword drops while an opening holds its tree.
    let _ = codeword.paths(log_folding, &[0, 1], hash).expect("paths");
    assert_eq!(
        codeword.retained_leaf_bytes(),
        tree,
        "the rebuild keeps the tree whole again"
    );
    let reader = codeword
        .hold_kept_tree(log_folding, hash)
        .expect("a kept tree to read");
    let held = be.reserved_bytes();
    let live_held = math_cuda::whir::retained_bytes_live();
    drop(codeword);
    assert_eq!(
        be.reserved_bytes(),
        held,
        "the codeword dropped while an opening reads its tree: nothing may be given back yet"
    );
    assert_eq!(
        math_cuda::whir::retained_bytes_live(),
        live_held,
        "and the tree must still count as held"
    );
    drop(reader);
    assert_eq!(
        be.reserved_bytes(),
        before,
        "the opening let go: every byte must go back"
    );
    assert_eq!(
        math_cuda::whir::retained_bytes_live(),
        live_before,
        "and the live footprint with it"
    );
}
