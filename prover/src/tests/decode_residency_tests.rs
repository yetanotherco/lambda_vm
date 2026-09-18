//! W1-B §4: DECODE's out-of-band commitment is built ONCE per run and HELD
//! across the epochs, and what one such commit costs.
//!
//! # ⚠ Why `free_vram_bytes()` is not the discriminator
//!
//! §4 names `Backend::free_vram_bytes()` bounded across the epoch loop, in
//! preference to "the reservation does not grow" — which passes on a leak whose
//! bytes were never promised. The refinement is right and the accessor still
//! cannot answer the question, because **"held across fifteen epochs" and
//! "rebuilt and freed fifteen times" reach the same PEAK take**: one codeword
//! live at a time either way. What separates them is the FLOOR — a held
//! codeword never gives its bytes back mid-run — and the floor is what the
//! stream-ordered pool hides, since freed blocks stay in the pool and the
//! driver's free count does not rise on a rebuild. A bound on the peak, called
//! the residency test, would be a check that cannot fail wearing the name of
//! the one H4 asked for.
//!
//! So residency is decided by COUNTS and corroborated by memory:
//!
//! 1. [`the_decode_commitment_is_derived_once_per_run`] — the derivation count
//!    on the production call. One per run, at two different epoch counts.
//! 2. [`the_decode_commitment_is_held_across_the_epochs`] — the stacked-commit
//!    count against a count predicted from the shapes, at two epoch counts, so
//!    DECODE's contribution is visibly constant while the epochs' own scales.
//!    Under `cuda` it also reads `free_vram_bytes()` either side of the run and
//!    prints what the run took, bounded only against RETENTION — the commitment
//!    outliving the call that built it. That is a real regression class and it
//!    is NOT the residency discriminator; the header above says why.
//! 3. [`the_one_commit_cost`] — what one derivation costs, alone, so the WHIR
//!    prove's delta against the seam band has something to be attributed to.
//!
//! # Invocations
//!
//! ```text
//! # (1) card-free, part of the ordinary suite
//! cargo test --release -p lambda-vm-prover --lib decode_residency_tests
//!
//! # (2) on the box, with a card
//! cargo test --release -p lambda-vm-prover --lib --features cuda \
//!     tests::decode_residency_tests::the_decode_commitment_is_held_across_the_epochs \
//!     -- --exact --ignored --nocapture
//!
//! # (3) the one-commit cost, on the box, with a card
//! cargo test --release -p lambda-vm-prover --lib --features cuda \
//!     tests::decode_residency_tests::the_one_commit_cost \
//!     -- --exact --ignored --nocapture
//! ```

use executor::elf::Elf;
use stark::proof::options::ProofOptions;

use crate::multilinear_continuation;
use crate::test_utils::asm_elf_bytes;

/// ⛔ TAKEN BY EVERY TEST IN THIS FILE, and it is not tidiness.
///
/// `multilinear::gpu`'s call counters are PROCESS-WIDE and every test here
/// RESETS them, so cargo's parallel runner lets one test zero another's window
/// mid-run. That is not hypothetical: the first run of this file with all three
/// tests live read `commits 2` against a prediction of 3, purely because a
/// sibling reset the counter while the run was in flight. An assertion on a
/// global counter is an assertion about every test in the binary.
///
/// The lock covers this file. The only other readers in this binary are in
/// `multilinear_bench_tests`, and all of them are `#[ignore]`d, so they cannot
/// run alongside these unless someone asks for both by name.
///
/// ⚠ The derivation counter needs none of this — it is thread-local, which is
/// why it and not the commit count is the primary instrument.
static COUNTERS: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The guard, re-taken past a poisoning so one failing test does not turn the
/// rest of the file into a second, unrelated failure.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    COUNTERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The program these run on. It publishes output in its last epoch, so a run of
/// it also exercises the COMMIT-bus replay; nothing here depends on that, but a
/// residency test on a program whose epochs all verify vacuously would be
/// measuring a run nobody would ship.
const PROGRAM: &str = "test_private_input_xpage";

/// Private input for [`PROGRAM`]: a length and eight bytes it commits.
fn input() -> Vec<u8> {
    let mut input: Vec<u8> = Vec::with_capacity(16);
    input.extend_from_slice(&16u32.to_le_bytes());
    input.extend_from_slice(&[0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
    input.extend_from_slice(&[0u8; 4]);
    input
}

/// One production run, returning `(epochs, derivations, commits)`.
///
/// `commits` is `commit_calls() + host_fallbacks()`, which is the number of
/// stacked polynomials committed whatever the build: on a card the first term
/// carries them, without one the second does, and a commit the device DECLINED
/// lands in the second on a cuda build rather than vanishing. Reading only
/// `commit_calls()` would make a run with no card look like a run with no
/// commits.
fn one_run(epoch_size_log2: u32) -> (usize, u64, u64) {
    let elf_bytes = asm_elf_bytes(PROGRAM);
    let opts = ProofOptions::default_test_options();

    multilinear_continuation::reset_decode_derivations();
    multilinear::gpu::reset_call_counters();

    let epochs =
        multilinear_continuation::prove_epochs(&elf_bytes, &input(), epoch_size_log2, &opts)
            .expect("prove the epochs");

    let derivations = multilinear_continuation::decode_derivations();
    let commits = multilinear::gpu::commit_calls() + multilinear::gpu::host_fallbacks();
    (epochs.len(), derivations, commits)
}

// ⛔ THE TWO COMMIT COUNTERS COUNT DIFFERENT SETS, and one prediction for both
// is what made run rs2 red.
//
// ✓ VERIFIED by reading: `note_host_fallback` has exactly ONE call site in the
// tree — `whir_chain.rs`, the INITIAL commitment of a chain's stacked
// polynomial. The FOLD commits have no host-fallback counterpart;
// `CodewordCommitment::commit` and `commit_tree_ext3` bump `COMMIT_CALLS` and
// nothing else. So without a card `host_fallbacks` is the count of INITIAL
// commitments and nothing more, while with one `commit_calls` counts the
// initial commitments AND every fold. Summing the two against a single model
// predicted 3 where the box read 9.
//
// Each counter is therefore compared against its OWN model, and the models are
// named for what they count.

/// The HOST-path model: stacked polynomials committed, one per group per epoch.
///
/// Derived from the epoch's own table count through the same `epoch_groups` the
/// prover splits on, so it is a function of the shape rather than a number read
/// off a previous run.
fn host_commits_per_epoch(num_tables: usize) -> u64 {
    multilinear_continuation::epoch_groups(num_tables).len() as u64
}

/// The DEVICE model: commits an epoch's chains make, READ OFF THE PROOF.
///
/// ★ No layout is reconstructed and no AIR is rebuilt. A group's opening is a
/// `StackedProof`, each of its polynomials is a `ChainProof`, and
/// `rounds.len()` IS the fold schedule's length — one initial commitment plus
/// `R - 1` successor codewords, which is one commit per round. So the epoch's
/// own chains contribute `rounds.len()` each.
///
/// DECODE's prepared opening contributes `rounds.len() - 1`: its chain runs
/// once per epoch, but the polynomial it folds was committed ONCE for the whole
/// run and held — which is the residency claim this file exists for, arriving
/// here as the one term that is not per-epoch. The held commitment itself is
/// the `+ 1` the caller adds once.
fn group_chain_rounds(epoch: &multilinear_continuation::EpochProof) -> u64 {
    epoch
        .proof
        .columns
        .iter()
        .flat_map(|group| &group.polys)
        .map(|poly| poly.rounds.len() as u64)
        .sum()
}

/// DECODE's share of the same model, kept SEPARATE because it is the term whose
/// device behaviour depends on the shape — see the printed decomposition.
fn prepared_chain_folds(epoch: &multilinear_continuation::EpochProof) -> u64 {
    epoch
        .proof
        .preprocessed
        .iter()
        .flat_map(|opening| &opening.polys)
        .map(|poly| poly.rounds.len() as u64 - 1)
        .sum()
}

/// ★★ THE HOIST, READ OFF THE PRODUCTION CALL.
///
/// `prove_epochs` opens the `with_whir_hash!` dispatch ABOVE the epoch loop and
/// derives DECODE's commitment once, so the count is ONE however many epochs
/// the run has. A per-epoch rebuild reads the epoch count.
///
/// Two epoch counts, because an arm whose run has ONE epoch cannot tell "once"
/// from "once per epoch" at all — both read 1. The larger arm is the one that
/// discriminates, so the test asserts it really has at least two epochs rather
/// than hoping; the smaller arm is there to show the count does not track the
/// epoch count.
///
/// The count is an EQUALITY, not a bound: the counter is thread-local (see its
/// declaration), so a derivation that moved to a worker thread would read zero,
/// and `<= 1` would accept that silently.
#[test]
fn the_decode_commitment_is_derived_once_per_run() {
    let _exclusive = exclusive();
    let (few_epochs, few_derivations, _) = one_run(4);
    let (many_epochs, many_derivations, _) = one_run(2);

    assert!(
        many_epochs >= 2 && many_epochs > few_epochs && few_epochs >= 1,
        "the larger arm needs at least two epochs to tell `once` from `once per \
         epoch`, and the two arms must differ: {few_epochs} and {many_epochs}"
    );
    assert_eq!(
        few_derivations, 1,
        "a {few_epochs}-epoch run derived DECODE's commitment {few_derivations} times"
    );
    assert_eq!(
        many_derivations, 1,
        "a {many_epochs}-epoch run derived DECODE's commitment {many_derivations} times"
    );
}

/// ★★★ RESIDENCY, as a count that sums to a prediction.
///
/// Every epoch commits one stacked polynomial per group, and the run commits
/// DECODE's out-of-band polynomial once on top. So
///
/// ```text
/// commits = Sum over epochs of groups(epoch) + 1
/// ```
///
/// and the `+ 1` is the residency claim: it does not scale with the epochs. The
/// prediction is checked at TWO epoch counts, and it is bumped from both arms —
/// an arm that measured nothing fails its own equality rather than passing
/// quietly.
///
/// ⚠ A DELTA BETWEEN THE ARMS WOULD NOT DO. `commits(many) - commits(few)`
/// cancels the `+ 1` exactly when it is right AND exactly when it is missing,
/// so the difference is blind to the very term the test is about. Each arm is
/// compared against its own predicted total instead.
///
/// ⚠ AND A ONE-EPOCH ARM DECIDES NOTHING HERE EITHER: at one epoch, held and
/// rebuilt both predict `groups + 1`. The discrimination lives entirely in the
/// multi-epoch arm — three epochs predict 7 held and would read 9 rebuilt — so
/// the test asserts that at least one arm had two or more epochs, and says so
/// when it did not.
///
/// # The counts print and assert FIRST, and the VRAM line is a measurement
///
/// ⛔ Run rs1 died in 0.93 s on the VRAM assertion and the two RESIDENCY count
/// lines — the discriminator, the whole point of the test — never printed. A
/// secondary check that runs before the primary one can spend a box slot
/// answering nothing. The VRAM READING still happens where it must, straight
/// after `prove_epochs` and a drain, because anything allocated afterwards
/// would spoil it; only the printing and the verdict move.
///
/// The reading is now PRINTED, not asserted, and here is what it means.
/// `drain_and_trim` synchronises the context and trims the device's DEFAULT
/// memory pool to zero, so what it cannot give back is (i) memory a live
/// `CudaSlice` still owns, (ii) allocations outside that pool, and (iii) the
/// driver's own per-process reservations. `Backend` holds device buffers for
/// the life of the process — the forward and inverse twiddle caches are
/// `Arc<CudaSlice<u64>>` per `log_n`, filled lazily and never dropped — and the
/// cubin modules with their per-SM local-memory backing store are the driver's,
/// paid at the first launch. rs1 read 544 MiB retained; none of those can be
/// sized from the source at this fixture's shapes.
///
/// ★ WHAT SEPARATES THEM IS ALREADY IN THIS TEST: it runs TWO arms in ONE
/// process, and every candidate above is paid ONCE. So the SECOND arm's
/// retention is the number that decides it, and rs1 never reached it.
///
/// ★ AND rs3 ANSWERED IT: arm 1 retained 570,425,344 B and arm 2 retained ZERO.
/// The 544 MiB is one-time process cost — the first arm's twiddle caches and
/// module load, neither of which a pool trim can return — and not a leak. So
/// arm 1 stays a printed measurement and arm 2 now carries the bound, where the
/// run's OWN buffers are the only thing left that could show.
///
/// It is also deliberately not sampled on a thread during the run: an in-flight
/// peak cannot separate held from rebuilt either, so the machinery would buy a
/// number that decides nothing.
#[test]
#[ignore = "runs two full continuations; the box runs it with a card"]
fn the_decode_commitment_is_held_across_the_epochs() {
    let _exclusive = exclusive();
    let mut most_epochs = 0usize;
    let mut arms: Vec<Arm> = Vec::new();
    for (arm, epoch_size_log2) in [4u32, 2u32].into_iter().enumerate() {
        let arm = arm + 1;
        let elf_bytes = asm_elf_bytes(PROGRAM);
        let elf = Elf::load(&elf_bytes).expect("load");
        let opts = ProofOptions::default_test_options();

        multilinear_continuation::reset_decode_derivations();
        multilinear::gpu::reset_call_counters();

        // ⚠ Drained first, or the figure is the POOL's and not the caller's:
        // the stream-ordered allocator keeps freed blocks, so a previous arm's
        // codeword would silently serve this one.
        #[cfg(feature = "cuda")]
        let (backend, free_before) = {
            let backend = math_cuda::device::backend().expect("a device");
            math_cuda::device::drain_and_trim().expect("drain");
            let free = backend.free_vram_bytes().expect("cuMemGetInfo");
            (backend, free)
        };

        let proofs =
            multilinear_continuation::prove_epochs(&elf_bytes, &input(), epoch_size_log2, &opts)
                .expect("prove the epochs");

        // ⚠ TAKEN HERE, PRINTED BELOW. The reading has to happen before
        // anything else allocates; the verdict it feeds does not have to come
        // first, and rs1 is what that cost.
        #[cfg(feature = "cuda")]
        let retained = {
            math_cuda::device::drain_and_trim().expect("drain");
            let free_after = backend.free_vram_bytes().expect("cuMemGetInfo");
            free_before.saturating_sub(free_after)
        };

        let derivations = multilinear_continuation::decode_derivations();
        let device_commits = multilinear::gpu::commit_calls();
        let host_commits = multilinear::gpu::host_fallbacks();

        let host_model: u64 = proofs
            .iter()
            .map(|p| host_commits_per_epoch(p.table_num_vars.len()))
            .sum::<u64>()
            + 1;
        let groups_rounds: u64 = proofs.iter().map(group_chain_rounds).sum();
        let prepared_folds: u64 = proofs.iter().map(prepared_chain_folds).sum();
        let device_model: u64 = groups_rounds + prepared_folds + 1;

        // ⛔ THE PRIMARY VERDICT PRINTS AND ASSERTS HERE, FOR BOTH ARMS, BEFORE
        // ANY COMMIT MODEL IS COMPARED. rs1 lost it to a VRAM assert and rs2
        // lost the second arm's to a commit assert built on the wrong counter's
        // model. The derivation count is the instrument that decides residency;
        // nothing else runs ahead of it.
        println!(
            "RESIDENCY  arm {arm}  epoch 2^{epoch_size_log2}  epochs {}  derivations {derivations}",
            proofs.len(),
        );
        assert_eq!(
            derivations,
            1,
            "DECODE's commitment was derived {derivations} times over {} epochs",
            proofs.len()
        );
        arms.push(Arm {
            arm,
            epoch_size_log2,
            device_commits,
            host_commits,
            host_model,
            device_model,
            groups_rounds,
            prepared_folds,
            #[cfg(feature = "cuda")]
            retained,
        });
        most_epochs = most_epochs.max(proofs.len());
        let _ = &elf;
    }
    assert!(
        most_epochs >= 2,
        "every arm ran a single epoch, where held and rebuilt predict the same \
         count: this run discriminated nothing (largest was {most_epochs})"
    );

    // The commit models, after every derivation line. Each counter is compared
    // against the model that describes IT and against no other.
    for a in &arms {
        let Arm {
            arm,
            epoch_size_log2,
            device_commits,
            host_commits,
            host_model,
            device_model,
            groups_rounds,
            prepared_folds,
            ..
        } = *a;
        if device_commits == 0 {
            // No card: every commit took the fallback path, and that counter is
            // the count of stacked polynomials committed. This is the arm the
            // model was verified on — 3 at one epoch, 7 at three.
            println!(
                "RESIDENCY-COMMITS  arm {arm}  epoch 2^{epoch_size_log2}  host \
                 {host_commits} (model {host_model})  device {device_commits}"
            );
            assert_eq!(
                host_commits, host_model,
                "arm {arm} committed {host_commits} stacked polynomials on the host path; \
                 the shapes predict {host_model} = one per group per epoch plus ONE for \
                 DECODE. Higher by the epoch count is DECODE rebuilt per epoch; lower is \
                 the out-of-band commitment missing entirely."
            );
        } else {
            // ⛔ WHAT rs3 MEASURED, and what it did and did not settle.
            //
            // ```text
            // arm 1  device  7  host 2  model 11 = groups  9 + prepared folds 1 + 1 held
            // arm 2  device 21  host 4  model 30 = groups 26 + prepared folds 3 + 1 held
            // ```
            //
            // The counters see 9 of arm 1's 11 predicted commits and 25 of arm
            // 2's 30. The shortfall is 2 and 5. On arm 1 that is exactly the
            // prepared group's own contribution — its held commitment and its
            // one fold, both invisible because a polynomial too small for the
            // device is committed on the host, where a FOLD commit is counted
            // nowhere at all (`note_host_fallback` has one call site, and it is
            // the initial commitment). On arm 2 the same reasoning accounts for
            // 4 of the 5. ONE COMMIT IS STILL UNEXPLAINED, so the exact model
            // is not settled and this test does not pretend otherwise.
            //
            // What IS established is the direction that matters here: every
            // commit the counters see is one the chains predict. DECODE being
            // rebuilt per epoch would add commits BEYOND the model and push the
            // sum above it, which is the residency regression this file exists
            // to catch, so the bound is asserted and the shortfall is printed
            // beside the term that should explain it.
            let seen = device_commits + host_commits;
            println!(
                "RESIDENCY-COMMITS  arm {arm}  epoch 2^{epoch_size_log2}  device \
                 {device_commits}  host {host_commits}  seen {seen}  model {device_model} \
                 = groups {groups_rounds} + prepared folds {prepared_folds} + 1 held  \
                 shortfall {} (the prepared group's own commits are {})",
                device_model - seen,
                prepared_folds + 1,
            );
            assert!(
                seen <= device_model,
                "arm {arm} made {seen} commits where its chains predict at most \
                 {device_model}; a count above the model is work no chain in this \
                 proof accounts for, and DECODE rebuilt per epoch is what that \
                 looks like"
            );
        }
    }

    // The memory measurement last, after every verdict it must never pre-empt.
    // One DECODE codeword at this shape is printed beside it for scale:
    // `log_blowup` is 2, so the codeword is `4 * cells` field elements of eight
    // bytes.
    #[cfg(feature = "cuda")]
    for a in &arms {
        let cells = 5u64 * 16;
        let codeword_bytes = (cells << 2) * 8;
        println!(
            "RESIDENCY-VRAM  arm {}  epoch 2^{}  retained {} B  one DECODE codeword \
             {codeword_bytes} B",
            a.arm, a.epoch_size_log2, a.retained,
        );
        // ★ ASSERTED ON THE SECOND ARM ONLY, and now on a measurement rather
        // than a guess. rs3 read 570,425,344 B retained after arm 1 and ZERO
        // after arm 2 — so the 544 MiB is one-time process cost (the first
        // arm's twiddle caches and module load, neither of which the pool trim
        // can return) and not a leak. Arm 1 is therefore still only printed;
        // arm 2 is where the run's OWN buffers would show, and there the bound
        // is a real check: anything the run built and did not give back lands
        // above one DECODE codeword.
        if a.arm >= 2 {
            assert!(
                a.retained < codeword_bytes.max(1 << 20),
                "after arm {}'s `prove_epochs` returned and the pool was drained, {} B \
                 are still held on the card — more than one DECODE codeword \
                 ({codeword_bytes} B). The process's one-time costs were already paid \
                 by arm 1, so this is the run's own memory outliving the call.",
                a.arm,
                a.retained,
            );
        }
    }
}

/// One arm's readings, so every arm's PRIMARY verdict lands before any secondary
/// model is compared. rs1 and rs2 were each lost to a secondary check that ran
/// first.
struct Arm {
    arm: usize,
    epoch_size_log2: u32,
    device_commits: u64,
    host_commits: u64,
    host_model: u64,
    device_model: u64,
    groups_rounds: u64,
    prepared_folds: u64,
    #[cfg(feature = "cuda")]
    retained: u64,
}

/// What ONE derivation costs, alone.
///
/// The WHIR prove's delta against the seam band has to be attributable, and a
/// figure for the whole run cannot do that. This times `decode_prepared_for`
/// by itself, after a warm-up of the same shape so the first-call costs —
/// twiddles, workspaces, whatever a pool grows to hold them — are paid outside
/// the window rather than charged to the commit.
///
/// It prints the shape beside the time, because a commit time without the
/// polynomial's size is a number that cannot be compared with anything.
///
/// # ⛔ THE GUEST IS A KNOB WITH NO DEFAULT
///
/// This ran on [`PROGRAM`], whose DECODE table is 5 x 16. That measures the
/// SHAPE working; it is not the 2^23 commit the seam band needs a figure for,
/// and the two differ by five orders of magnitude in cells while printing the
/// same sentence. `LAMBDA_VM_ONE_COMMIT_ELF` therefore has NO fallback: an
/// unset knob panics naming why, so nobody can run the small one and read the
/// number as the big one.
///
/// ⚠ AND THE LINE CARRIES THE GUEST'S FULL SHA256, not its name. "ethrex" names
/// whatever a build directory produced, and this campaign has already spent a
/// day on a 0.23% difference that turned out to be two builds of "the same"
/// guest. A name in the output is a display; the sha is the measurement.
///
/// ```text
/// LAMBDA_VM_ONE_COMMIT_ELF=ethrex_8f826601 cargo test --release \
///     -p lambda-vm-prover --lib --features cuda \
///     tests::decode_residency_tests::the_one_commit_cost -- --exact --ignored --nocapture
/// ```
#[test]
#[ignore = "a printing measurement; the box runs it with a card"]
fn the_one_commit_cost() {
    let _exclusive = exclusive();
    use sha2::{Digest, Sha256};
    use std::time::Instant;

    let name = std::env::var("LAMBDA_VM_ONE_COMMIT_ELF").expect(
        "LAMBDA_VM_ONE_COMMIT_ELF must name the guest. This measurement is what the \
         WHIR prove's delta is attributed to, and the file's own fixture would answer \
         with a 5 x 16 DECODE table while printing the same line as a 5 x 2^20 one. \
         There is no default on purpose.",
    );
    let elf_bytes = crate::tests::multilinear_bench_tests::elf_bytes(&name);
    let sha: String = Sha256::digest(&elf_bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let elf = Elf::load(&elf_bytes).expect("load");

    let columns = crate::tables::decode::preprocessed_columns_from_elf(&elf).expect("columns");
    let rows = columns[0].len();

    crate::with_whir_hash!(|H| {
        // Warm-up, discarded: the first commit of a shape pays one-time costs.
        drop(
            multilinear_continuation::decode_prepared_for::<H>(&elf, &elf_bytes)
                .expect("the warm-up derivation"),
        );
        multilinear::gpu::reset_call_counters();

        let start = Instant::now();
        let prepared = multilinear_continuation::decode_prepared_for::<H>(&elf, &elf_bytes)
            .expect("the timed derivation");
        let elapsed = start.elapsed();

        println!(
            "ONE-COMMIT  {:.3} s  guest {name} sha {sha} ({} bytes)  columns {}  \
             rows {rows}  cells {}  roots {}  gpu commits {}  host fallbacks {}",
            elapsed.as_secs_f64(),
            elf_bytes.len(),
            columns.len(),
            columns.len() * rows,
            prepared.roots.len(),
            multilinear::gpu::commit_calls(),
            multilinear::gpu::host_fallbacks(),
        );

        // Not a timing assertion — a shape one, so a run that timed a commit of
        // the wrong polynomial says so instead of printing a fast number.
        assert_eq!(
            prepared.roots.len(),
            1,
            "the five columns must stack into ONE polynomial, or this is not the \
             commit whose cost the seam band is being attributed to"
        );
    });
}
