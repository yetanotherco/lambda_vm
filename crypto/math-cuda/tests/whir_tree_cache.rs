//! ★★ H4's result of record, and what replaced it — a commitment does NOT keep
//! its TREE; it keeps its LEAF LAYER, and the bytes it holds are its codeword
//! and that layer.
//!
//! ⛔ H4's finding stands and is not edited away below: keeping the whole node
//! array LOST, measured on the card at about +15 s, because the retention is
//! one object per commitment IN THE GROUP and ten of those put the device at
//! 96%. What changed is WHICH object. A tree is `2·num_leaves − 1` nodes; its
//! leaf layer is `num_leaves` of them — half the bytes — and the leaf pass it
//! saves is two thirds of a base tree's permutations and six sevenths of an
//! extension one, because a leaf absorbs a whole `2^k` coset while an inner
//! node absorbs two digests. Half the memory for most of the saving is a
//! different trade from the one H4 measured, and these tests now pin BOTH
//! sides of it: the layer must be held, and a whole tree must still never be.
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

        assert_eq!(
            codeword.leaf_passes(),
            1,
            "{name}: the commit hashes the leaves once"
        );

        let _ = codeword.paths(4, &[0, 1, 7], hash).expect("paths");
        assert_eq!(
            codeword.tree_builds(),
            2,
            "{name}: an opening builds its own tree — a 1 here means one is kept"
        );
        // ★ THE OTHER DIRECTION, and it is the whole point of the change: the
        // tree was rebuilt, but its LEAF LAYER was not re-hashed. A 2 here is
        // the retention not working.
        assert_eq!(
            codeword.leaf_passes(),
            1,
            "{name}: the opening must serve the retained leaf layer — a 2 here \
             means the layer was not kept, or not matched"
        );

        // …and again, because a cache that served once and then evicted would
        // read 2 on the line above too.
        let _ = codeword.paths(4, &[2, 3], hash).expect("paths");
        assert_eq!(
            codeword.tree_builds(),
            3,
            "{name}: and a second opening builds a third"
        );
        assert_eq!(
            codeword.leaf_passes(),
            1,
            "{name}: and still one leaf pass — a layer that served once and was \
             then evicted would read 2 here"
        );
        assert!(
            codeword.retained_leaf_bytes() > 0,
            "{name}: the codeword reports no retained layer, so the counts above \
             are agreeing about the wrong thing"
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
                verify_opening::<_, H>(&root, host.depth(), index, &opening),
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
/// # The margin, and why it is this wide
///
/// A codeword here is `2^20 << 2` u64 = 32 MiB. At `log_folding = 2` its tree
/// is `2^20` leaves, `(2*2^20 - 1) * 32` B = **64 MiB** — two codewords, not
/// half of one, which is the whole reason for that blocking. So four
/// codewords are 128 MiB and four codewords with their kept trees are 384 MiB,
/// and the bound sits at 256 MiB: 128 MiB of slack above the passing case and
/// 128 MiB below the failing one.
///
/// ⚠ The slack is not decoration. `free_vram_bytes` reports what the PROCESS
/// has taken from the driver, which includes whatever one-time workspace and
/// twiddle caches the first commit of this size sets up, and those are counted
/// identically in both cases. A bound only a codeword above the passing case
/// would turn any such allocation into a false failure — and a wider blocking,
/// where the tree is half a codeword, would leave no room for one. The warm-up
/// commit below pays those costs before the sample, and the margin absorbs what
/// it misses.
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
    // of one, which is what puts 128 MiB between the passing and failing cases
    // instead of 32.
    let log_folding = 2;

    // One commit of this exact shape before the sample, so the one-time costs
    // of the first — twiddles, workspaces, whatever the pool grows to hold them
    // — are paid outside the window and not attributed to retention.
    drop(commit_on_device(num_vars, log_folding, hash));

    // ⚠ And without this the measurement is the POOL's, not the caller's: the
    // stream-ordered allocator keeps freed blocks, and the warm-up's own
    // codeword would silently serve one of the four commits below. Drain, then
    // sample.
    math_cuda::device::drain_and_trim().expect("drain");
    let free_before = be.free_vram_bytes().expect("cuMemGetInfo");
    // ★ THE SECOND INSTRUMENT, and it is the one that can tell the two stories
    // apart. `free_vram_bytes` is the DRIVER's count and includes whatever the
    // pool is sitting on; `reserved_bytes` is what the CODE promised and is
    // blind to the pool by construction. Their difference is the pool's, and
    // printing it turns "either the pool retained a block or the code holds one
    // more" from a question into a read.
    let reserved_before = be.reserved_bytes();

    let held: Vec<_> = (0..4)
        .map(|_| commit_on_device(num_vars, log_folding, hash))
        .collect();

    // ⛔ SYMMETRIC SAMPLING, AND THIS LINE IS THE FIX. `free_before` is taken
    // AFTER a drain and this one was not, so the difference measured the code
    // plus every transient the four commits made — and with the pool set to
    // retain all freed blocks, that is all of them. A delta between two samples
    // is about the code only if both are taken at the same pool state.
    //
    // ★ WHAT IT COST, and why a widened bound was the wrong repair. The run of
    // 2026-09-20 read `taken` = 301,989,888 B — EXACTLY nine codewords, on a
    // bound of nine, where the model says eight are held. The old one-sided
    // bound was `8 × codeword` against four codewords held, so it carried 128
    // MiB of margin the pool had been living in unnoticed; the leaf-layer
    // retention did not add pool retention, it CONSUMED that margin. And the
    // slack was never sized against the transients anyway: the node buffer
    // `build_tree` allocates is `(2L−1)·32` = 64 MiB, twice the bound's 32.
    //
    // The same run's mutation arm settles which it is. Holding a whole node
    // array instead of a layer moved the measurement to 480 MiB where the code
    // then holds 384 — an excess of 96 against the honest run's 32, tripling
    // while the holding grew by half. No "the code holds one more object" form
    // fits both (`4×(cw+tree)+tree` = 448, `+cw` = 416; `5×(cw+tree)` = 480 fits
    // MUT C exactly and dies on the honest run, where `5×(cw+leaf)` = 320 ≠ 288).
    // Every peak-demand model misses on BOTH sides (320 and 448 predicted), which
    // is the signature of driver suballocation and not of anything this tree
    // accounts for. ⇒ `free_vram_bytes()` cannot carry a bound this tight
    // unless both samples are drained.
    math_cuda::device::drain_and_trim().expect("drain");
    let free_after = be.free_vram_bytes().expect("cuMemGetInfo");
    let taken = free_before.saturating_sub(free_after);
    let promised = be.reserved_bytes().saturating_sub(reserved_before);

    let codeword_bytes = ((1u64 << num_vars) << 2) * 8;
    let leaves = ((1u64 << num_vars) << 2) >> log_folding;
    let leaf_bytes = leaves * 32;
    let tree_bytes = (2 * leaves - 1) * 32;
    // ⛔ TWO-SIDED, AND AGAINST THE FORM RATHER THAN A MULTIPLE. The layers must
    // be HELD (so more than the codewords alone) and a whole TREE must still
    // never be (so less than four of those). At this shape — `log_folding = 2`,
    // chosen by the comment above because it makes a tree two codewords — a
    // leaf layer is exactly ONE codeword, so the three cases are 128, 256 and
    // 384 MiB and the bound sits between the last two with one codeword of
    // slack. A one-sided bound passed either way and is what let the old
    // arithmetic sit on its own edge.
    let expect = 4 * (codeword_bytes + leaf_bytes);
    let bound = expect + codeword_bytes;
    let floor = 4 * codeword_bytes;
    let mib = |b: u64| b / (1 << 20);
    // ★ THE TWO ACCOUNTINGS, SIDE BY SIDE, IN WHICHEVER MESSAGE FIRES. A failure
    // here used to say only how many bytes the DRIVER lost, which cannot
    // distinguish "the pool retained a block" from "the code holds one more" —
    // and those call for opposite repairs. `promised` is the code's own number
    // and is blind to the pool; the per-codeword pair says how many layers are
    // actually in hand. `driver − promised` is the pool's share, and it should
    // now be small: the drain above is what makes that true.
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
            "driver {} MiB · promised {} MiB · pool share {} MiB · per codeword \
             reserved/retained MiB: [{}]",
            mib(taken),
            mib(promised),
            mib(taken.saturating_sub(promised)),
            per.join(", ")
        )
    };
    // ⛔ UNCONDITIONAL, AND THAT IS THE WHOLE POINT. Built only inside the
    // assertion messages, this line appears ONLY when the test fails — so on
    // the honest path the numbers were inferred from a pass and never read.
    //
    // What a pass alone establishes is `driver < bound`, i.e. pool share under
    // one codeword, and NOTHING narrower. The mutation run that keeps a whole
    // node array reads `pool share 64 MiB` even after the symmetric drain — a
    // fragmentation floor of one largest transient, because a best-effort
    // `trim` cannot release a chunk still backing a live allocation. If the
    // honest path sits anywhere near that, this guard has a margin of a few MiB
    // and will flake, and nobody would learn it from a green run.
    //
    // ⇒ printed every time, under `--nocapture`, which is how the box gate runs
    // this suite. The number becomes a READ, and a ceiling can be asserted
    // against it once its honest value is known.
    println!("   group guard: {ledger}");
    assert!(
        taken < bound,
        "four unopened commitments took {} MiB from the device. Four codewords \
         and their leaf layers are {} MiB and the bound is {} MiB; a TREE is {} \
         MiB, so four of those kept would read {} MiB. Something bigger than a \
         leaf layer is held per commitment.\n   {}",
        mib(taken),
        mib(expect),
        mib(bound),
        mib(tree_bytes),
        mib(4 * (codeword_bytes + tree_bytes)),
        ledger,
    );
    assert!(
        taken > floor,
        "four unopened commitments took only {} MiB, which is at or under the {} \
         MiB their codewords alone need. The leaf layers ({} MiB for four) are \
         NOT being held — either the capture never ran or the budget refused it, \
         and in both cases every opening will re-hash its leaves.\n   {}",
        mib(taken),
        mib(floor),
        mib(4 * leaf_bytes),
        ledger,
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
    // ★ THE KEY, ASSERTED WHERE IT CAN FAIL. The retained layer was built at
    // k=4; this opening is at k=2 and describes a DIFFERENT tree, so the layer
    // must not be served and the leaves must be hashed again. A 1 here is a
    // cache ignoring its key, which is the one way this change could hand back
    // paths that are internally consistent and wrong.
    assert_eq!(
        codeword.leaf_passes(),
        2,
        "a k=2 opening must NOT be served the k=4 leaf layer"
    );

    // And the rebuild answered the question that was asked: at k=2 the tree has
    // four times the leaves, so each path is two levels deeper.
    let at_four = codeword.paths(4, &[0, 1], hash).expect("paths at k=4");
    assert_eq!(codeword.tree_builds(), 3, "and a third for the k=4 opening");
    // …and the k=4 layer IS still there and IS served, so the key rejects a
    // mismatch without throwing away a match. Without this line the test above
    // would also pass on a cache that had simply stopped working.
    assert_eq!(
        codeword.leaf_passes(),
        2,
        "the k=4 opening matches the retained layer's key and must not re-hash"
    );
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

    // ⛔ DELTAS, NOT ABSOLUTES. These three counters are process-wide and no
    // longer resettable as a set, and the retention makes them diverge on
    // purpose, so the assertions below are about what THIS codeword moved.
    let at = || (tree_builds(), leaf_hash_calls(), retention_report().5);

    let (b0, p0, s0) = at();
    let (codeword, _root) = commit_on_device(12, 4, hash);
    let (b1, p1, s1) = at();
    assert_eq!(b1 - b0, 1, "one commit, one tree assembled");
    assert_eq!(p1 - p0, 1, "and it paid for its own leaf pass");
    assert_eq!(s1 - s0, 0, "with nothing yet in hand to reuse");

    let _ = codeword.paths(4, &[0, 1], hash).expect("paths");
    let (b2, p2, s2) = at();
    assert_eq!(b2 - b1, 1, "the opening assembles its own tree");
    // ★ THE LINE THAT CHANGED WITH H4's REPLACEMENT. This used to assert the
    // global counter moved by one too, because a tree and a leaf pass were the
    // same event. They are not any more: the tree is assembled, the leaves are
    // not re-hashed, and a 1 here is the retention failing to serve.
    assert_eq!(p2 - p1, 0, "and does NOT re-hash the leaves");
    assert_eq!(s2 - s1, 1, "the saving is counted where it happens");

    // ⭐ THE ACCOUNTING IDENTITY, which is what the old equality became and
    // which fails in BOTH directions: a tree that skipped its pass without
    // recording a saving breaks it, and so does a saving recorded for a tree
    // that was never assembled.
    assert_eq!(
        b2 - b0,
        (p2 - p0) + (s2 - s0),
        "every tree either paid for its leaf pass or reused one; trees {}, \
         passes {}, savings {}",
        b2 - b0,
        p2 - p0,
        s2 - s0
    );
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
/// on. A broken `RetainedLeaves::drop` leaves the live count high and reddens the
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
        "dropping the codeword must give the layer's bytes back: RetainedLeaves::drop \
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
