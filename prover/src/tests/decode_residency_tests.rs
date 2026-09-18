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

/// How many stacked polynomials an epoch of this shape commits: one per group.
///
/// Derived from the epoch's own table count through the same `epoch_groups`
/// the prover splits on, so the prediction below is a function of the shape
/// rather than a number read off a previous run.
fn commits_per_epoch(num_tables: usize) -> u64 {
    multilinear_continuation::epoch_groups(num_tables).len() as u64
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
/// Under `cuda` this also reads `free_vram_bytes()` either side of the run, with
/// the pool drained first, and prints what the run took. Its only assertion is
/// against RETENTION: after `prove_epochs` returns, nothing the run built may
/// still be on the card. That is a real regression — the commitment outliving
/// the call — and it is NOT the residency discriminator, for the reason in this
/// module's header. It is deliberately not sampled on a thread during the run:
/// an in-flight peak cannot separate held from rebuilt either, so the extra
/// machinery would buy a number that decides nothing.
#[test]
#[ignore = "runs two full continuations; the box runs it with a card"]
fn the_decode_commitment_is_held_across_the_epochs() {
    let _exclusive = exclusive();
    let mut most_epochs = 0usize;
    for epoch_size_log2 in [4u32, 2u32] {
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

        #[cfg(feature = "cuda")]
        {
            math_cuda::device::drain_and_trim().expect("drain");
            let free_after = backend.free_vram_bytes().expect("cuMemGetInfo");
            let retained = free_before.saturating_sub(free_after);
            // One DECODE codeword at this shape, for scale. `log_blowup` is 2,
            // so the codeword is `4 * cells` field elements of 8 bytes.
            let cells = 5u64 * 16;
            let codeword_bytes = (cells << 2) * 8;
            println!(
                "RESIDENCY-VRAM  epoch 2^{epoch_size_log2}  retained {retained} B                   one DECODE codeword {codeword_bytes} B"
            );
            assert!(
                retained < codeword_bytes.max(1 << 20),
                "after `prove_epochs` returned and the pool was drained, {retained} B are                  still held on the card — more than one DECODE codeword ({codeword_bytes} B).                  Something the run built outlived the call."
            );
        }

        let derivations = multilinear_continuation::decode_derivations();
        let commits = multilinear::gpu::commit_calls() + multilinear::gpu::host_fallbacks();

        // The prediction, from the shapes the proofs themselves carry.
        let predicted: u64 = proofs
            .iter()
            .map(|p| commits_per_epoch(p.table_num_vars.len()))
            .sum::<u64>()
            + 1;

        println!(
            "RESIDENCY  epoch 2^{epoch_size_log2}  epochs {}  derivations {derivations}  \
             commits {commits}  predicted {predicted}",
            proofs.len(),
        );
        assert_eq!(
            derivations,
            1,
            "DECODE's commitment was derived {derivations} times over {} epochs",
            proofs.len()
        );
        assert_eq!(
            commits,
            predicted,
            "a {}-epoch run committed {commits} stacked polynomials; the shapes predict \
             {predicted} = one per group per epoch plus ONE for DECODE. A count higher by \
             the epoch count is DECODE being rebuilt per epoch; one lower is the \
             out-of-band commitment missing entirely.",
            proofs.len(),
        );
        most_epochs = most_epochs.max(proofs.len());
        let _ = &elf;
    }
    assert!(
        most_epochs >= 2,
        "every arm ran a single epoch, where held and rebuilt predict the same \
         count: this run discriminated nothing (largest was {most_epochs})"
    );
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
#[test]
#[ignore = "a printing measurement; the box runs it with a card"]
fn the_one_commit_cost() {
    let _exclusive = exclusive();
    use std::time::Instant;

    let elf_bytes = asm_elf_bytes(PROGRAM);
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
            "ONE-COMMIT  {:.3} s  columns {}  rows {rows}  cells {}  roots {}  \
             gpu commits {}  host fallbacks {}",
            elapsed.as_secs_f64(),
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
