//! WHIR against FRI on the same programs, same machine, CPU only.
//!
//! Ignored: these are measurements, not assertions. Run one with
//!
//! ```text
//! cargo test --release -p lambda-vm-prover --lib \
//!     multilinear_bench_tests::shapes -- --ignored --nocapture
//! ```
//!
//! `RAYON_NUM_THREADS=1` on both sides is the algorithmic comparison — neither
//! prover gets credit for being better parallelised. All cores is the number a
//! user sees. The gap between them is the parallelisation backlog, which for
//! the multilinear path is most of it: only the Merkle build and the NTT are
//! parallel today.

use std::time::Instant;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use multilinear::whir_chain::GrindBits;
use multilinear::whir_hash::KeccakWhir;
use stark::proof::options::GoldilocksCubicProofOptions;

use crate::multilinear_prove;
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;

/// Blowup 4, 128 bits, 20 bits of grinding — the parameters the multilinear
/// path derives its own from, so the two are being asked for the same security.
pub(super) fn options() -> stark::proof::options::ProofOptions {
    GoldilocksCubicProofOptions::with_params(4, 128, 20).expect("valid options")
}

pub(super) fn elf_bytes(name: &str) -> Vec<u8> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("executor/program_artifacts");
    for dir in ["rust", "asm"] {
        let path = root.join(dir).join(format!("{name}.elf"));
        if let Ok(bytes) = std::fs::read(&path) {
            return bytes;
        }
    }
    panic!("no ELF named {name}");
}

/// The programs to sweep, smallest first: `(elf, private input)`.
///
/// `ethrex` with the 10-transfer block is the repo's reference workload. It runs
/// here **monolithic**, because the multilinear path has no continuations yet,
/// so it is bigger than the epochs the continuation bench proves.
const PROGRAMS: &[(&str, &str)] = &[
    ("sub", ""),
    ("all_instructions_64", ""),
    ("keccak", ""),
    ("ethrex", "ethrex_empty_block"),
    ("ethrex", "ethrex_simple_tx"),
    ("ethrex", "ethrex_bench_4"),
    ("ethrex", "ethrex_10_transfers"),
];

/// A private-input fixture from `executor/tests`, empty for a program that
/// takes none.
pub(super) fn input_bytes(name: &str) -> Vec<u8> {
    if name.is_empty() {
        return Vec::new();
    }
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("executor/tests")
        .join(format!("{name}.bin"));
    std::fs::read(&path).unwrap_or_else(|_| panic!("read {}", path.display()))
}

/// What each program costs in trace, without proving anything.
///
/// The multilinear prover's peak is driven by the **stacked** width
/// `num_vars + log2(columns)`: one codeword of `2^(n_stack + log_blowup)` base
/// elements per table, plus the extension-field fold on top of it. Printed here
/// so a program that cannot fit is ruled out before an hour is spent finding
/// out.
#[test]
#[ignore]
fn shapes() {
    println!(
        "\n{:<22} {:>7} {:>8} {:>7} {:>9} {:>12}",
        "program", "tables", "rows", "cols", "n_stack", "cells"
    );
    for &(name, input) in PROGRAMS {
        let label = if input.is_empty() {
            name.to_string()
        } else {
            input.to_string()
        };
        let bytes = elf_bytes(name);
        let inputs = input_bytes(input);
        let elf = match Elf::load(&bytes) {
            Ok(elf) => elf,
            Err(e) => {
                println!("{label:<22} load failed: {e:?}");
                continue;
            }
        };
        let logs = match Executor::new(&elf, inputs.clone()).and_then(Executor::run) {
            Ok(result) => result.logs,
            Err(e) => {
                println!("{label:<22} run failed: {e:?}");
                continue;
            }
        };
        let mut traces = match Traces::from_elf_and_logs(
            &elf,
            &logs,
            &MaxRowsConfig::default(),
            &inputs,
            #[cfg(feature = "disk-spill")]
            stark::storage_mode::StorageMode::Ram,
        ) {
            Ok(traces) => traces,
            Err(e) => {
                println!("{label:<22} trace failed: {e:?}");
                continue;
            }
        };
        let table_counts = traces.table_counts();
        let airs = crate::VmAirs::new(
            &elf,
            &options(),
            false,
            &traces.page_configs,
            &table_counts,
            None,
            true,
            None,
            None,
            None,
        );
        let pairs = airs.air_trace_pairs(&mut traces);

        // The widest stack sets the query count and the biggest single
        // allocation; the **cell total** is what normalises one prover against
        // another running different shapes, which is how the reference system
        // was compared — cells per second.
        let (tables, mut rows, mut cols, mut n_stack) = (pairs.len(), 0usize, 0usize, 0usize);
        let mut cells = 0u64;
        for (_, trace, _) in &pairs {
            let columns = trace.columns_main();
            let height = columns.first().map_or(0, Vec::len);
            let vars = height.trailing_zeros() as usize;
            let stack = vars + columns.len().next_power_of_two().trailing_zeros() as usize;
            cells += (height * columns.len()) as u64;
            if stack > n_stack {
                (rows, cols, n_stack) = (height, columns.len(), stack);
            }
        }
        println!(
            "{label:<22} {tables:>7} {rows:>8} {cols:>7} {n_stack:>9} {:>10.3}e9",
            cells as f64 / 1e9,
        );
    }
}

/// Proves one program and prints wall-clock and proof size.
///
/// `LAMBDA_VM_BENCH_ELF` picks the program, `LAMBDA_VM_BENCH_INPUT` its private
/// input fixture, `LAMBDA_VM_BENCH_BACKEND` which prover runs (`fri`, `whir`,
/// or both). One backend per process is what makes peak RSS attributable, so
/// the memory numbers come from
///
/// ```text
/// LAMBDA_VM_BENCH_BACKEND=whir /usr/bin/time -l cargo test --release ...
/// ```
#[test]
#[ignore]
fn whir_against_fri() {
    let name =
        std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "all_instructions_64".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let backend = std::env::var("LAMBDA_VM_BENCH_BACKEND").unwrap_or_else(|_| "both".into());
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let opts = options();
    let max_rows = MaxRowsConfig::default();
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "all".into());
    let label = if input.is_empty() { &name } else { &input };
    println!("\n{label} — CPU, RAYON_NUM_THREADS={threads}, backend={backend}");

    let mib = |n: usize| n as f64 / (1024.0 * 1024.0);
    let mut fri = None;
    let mut whir = None;

    if backend != "whir" {
        let start = Instant::now();
        let proof = crate::prove_with_options_and_inputs(&bytes, &inputs, &opts, &max_rows)
            .expect("univariate prove");
        let prove = start.elapsed();
        let size = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .expect("serialize")
            .len();
        let start = Instant::now();
        let ok = crate::verify_with_options(&proof, &bytes, &opts, None, None).expect("verify");
        assert!(ok, "the univariate proof must verify");
        fri = Some((prove, start.elapsed(), size));
    }

    if backend != "fri" {
        let start = Instant::now();
        let proof =
            multilinear_prove::prove_with_options_and_inputs(&bytes, &inputs, &opts, &max_rows)
                .expect("multilinear prove");
        let prove = start.elapsed();
        let size = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .expect("serialize")
            .len();
        let start = Instant::now();
        let ok = multilinear_prove::verify_with_options(&proof, &bytes, &opts).expect("verify");
        assert!(ok, "the multilinear proof must verify");
        whir = Some((prove, start.elapsed(), size));
    }

    println!(
        "{:<12} {:>10} {:>10} {:>12}",
        "backend", "prove", "verify", "proof"
    );
    for (tag, run) in [("FRI", &fri), ("WHIR", &whir)] {
        if let Some((prove, verify, size)) = run {
            println!(
                "{tag:<12} {:>9.2}s {:>9.2}s {:>10.2} MiB",
                prove.as_secs_f64(),
                verify.as_secs_f64(),
                mib(*size)
            );
        }
    }
    if let (Some(f), Some(w)) = (fri, whir) {
        println!(
            "{:<12} {:>9.2}x {:>9.2}x {:>10.2}x",
            "WHIR/FRI",
            w.0.as_secs_f64() / f.0.as_secs_f64(),
            w.1.as_secs_f64() / f.1.as_secs_f64(),
            w.2 as f64 / f.2 as f64,
        );
    }
}

/// A whole run proved by epochs: the multilinear continuation against the
/// univariate one, at the same epoch size.
///
/// This is the measurement the multilinear continuation exists for. Wall-clock
/// is the visible number, but the reason is the peak: a monolithic proof holds
/// every table of the run at once and an epoch holds one epoch's. That number
/// comes from the OS, one backend per process:
///
/// ```text
/// LAMBDA_VM_BENCH_BACKEND=whir LAMBDA_VM_BENCH_ELF=ethrex \
///   LAMBDA_VM_BENCH_INPUT=ethrex_10_transfers \
///   /usr/bin/time -v cargo test --release ...
/// ```
///
/// `LAMBDA_VM_BENCH_EPOCH_LOG2` is the epoch length in cycles, the CLI's
/// default (2^20) unless it is set. It is a resource knob, not a property of
/// either prover: both sides get the same one.
/// One transcript-counter line, labelled with the WINDOW it covers.
///
/// ⚠ The window is part of the number. The prover and the verifier each run a
/// transcript, over different work, and only the verifier's is what a recursive
/// verifier replays — so a count quoted without its side cannot be checked
/// against anything.
///
/// ⚠ `states` is NOT part of `squeezes`: a squeeze is a `finalize_reset` that
/// chains its output back in, a state read finalizes a CLONE and advances
/// nothing. There is one state read per grind check, so on the verify line the
/// states column must equal the grind-check count — two independent instruments
/// on one quantity.
#[cfg(feature = "hash-metrics")]
fn print_transcript_counts(window: &str, c: &crypto::hash_metrics::Counts) {
    let (ua, us, ut) = c.transcript_unattributed();
    println!(
        "{:<12} transcript absorbs {}/{} · squeezes {}/{} · states {}/{} (keccak/rpx) · unattributed {}/{}/{}",
        window,
        c.transcript_absorbs_keccak,
        c.transcript_absorbs_rpx,
        c.transcript_squeezes_keccak,
        c.transcript_squeezes_rpx,
        c.transcript_states_keccak,
        c.transcript_states_rpx,
        ua,
        us,
        ut,
    );
}

/// ★★ THE TRANSCRIPT PAIR, PINNED — one configuration, both sides.
///
/// # Why a pair and not just the verify line
///
/// The verify line alone is the number recursion cares about, but pinning both
/// makes their DIFFERENCE mutation-checkable, and that difference is a derived
/// quantity rather than a measurement: the verifier runs `owed`, the prover does
/// not, and `owed` is `Sum roots.len()` absorbs and `2 x epochs` squeezes. So a
/// change that moves one side without the other fails loudly here instead of
/// silently re-opening a question that took two lanes and a wrong candidate to
/// close.
///
/// The 2 squeezes per `owed` call are not "two samples, two squeezes": a cubic
/// element is three 8-byte draws from a 32-byte buffer, so the first sample
/// squeezes once and leaves a group over, and the second spends the leftover and
/// squeezes again. Three not dividing four is the whole reason it is two.
///
/// # WARNING: the guard is the ELF's sha256, not its name
///
/// Two builds of the same guest, from the same sources, in two worktrees, share
/// a name and differ in bytes. That is not hypothetical: a 0.23% difference in
/// these very counts was read as a model error before it was traced to a
/// different build of "the same" guest — the arms' ELF touches genesis pages
/// (`0x280000`, `0x680000`) that another build does not, which moves the
/// GLOBAL_MEMORY set and with it the chain shapes these counts are made of.
///
/// So a rebuild must not fail this assert — it must SKIP it, out loud, naming
/// the sha it saw. A silent pass and an absent assert are the same thing.
///
/// # Configuration is part of the constant
///
/// The counts are a function of the ELF, the input and the epoch size, so all
/// three are in the guard. Measured on FAST under `cuda,hash-metrics`, and
/// independently reproduced by lane V1's shape-derived closed form, which
/// predicts the verify triple exactly from the table shapes — two derivations,
/// one number.
#[cfg(feature = "hash-metrics")]
mod transcript_pin {
    /// sha256 of the guest ELF these counts were measured against.
    ///
    /// MEASURED, and it has to say where: `sha256sum` on the box fixture
    /// `ethrex_8f826601.elf`. The previous value agreed with this one for
    /// exactly 16 hex characters and was invented for the other 48 — every
    /// message that carried the sha carried a 16-char prefix, and the tail was
    /// written to look like a measurement. The guard then skipped on the pinned
    /// guest itself, and the skip line printed `[..16]` of both sides, which is
    /// precisely the width at which a fabricated tail still agrees.
    ///
    /// A guard on a value nobody measured to full width is a guard on a guess.
    /// Anything shortened for a message is a display; the constant is the
    /// measurement.
    pub const ELF_SHA256: &str = "8f826601776d4085cbb6fbf0302fe8d8d5d1be7940ac1aaca24899c6244ec80a";
    pub const ELF_LEN: usize = 3_948_504;
    pub const EPOCH_LOG2: u32 = 21;

    /// (absorbs, squeezes, states) after `prove_continuation`.
    pub const PROVE: (u64, u64, u64) = (583_924, 183_226, 2_996);
    /// ...and after `verify_continuation`. The difference is `owed`, nothing else.
    pub const VERIFY: (u64, u64, u64) = (584_061, 183_256, 2_996);

    /// `owed`'s own cost, stated rather than left as a subtraction: 137 absorbs
    /// is `Sum roots.len()` over the 15 epoch calls, 30 squeezes is `2 x 15`,
    /// and it reads no state.
    pub const OWED: (u64, u64, u64) = (137, 30, 0);
}

/// Whether the pinned counts describe THIS run.
///
/// Pure, and taking the sha as a string so the refusal can be tested without
/// forging an ELF: a sha that agrees on a prefix and differs in the tail is one
/// `format!` away, which is the case that actually occurred.
#[cfg(feature = "hash-metrics")]
fn pin_applies(sha: &str, len: usize, epoch_size_log2: u32) -> bool {
    sha == transcript_pin::ELF_SHA256
        && len == transcript_pin::ELF_LEN
        && epoch_size_log2 == transcript_pin::EPOCH_LOG2
}

/// The line a skipped pin prints.
///
/// FULL 64 hex on BOTH sides, never a prefix. The truncated version of this
/// line is why a wrong constant survived a box run: it showed `[..16]` of each,
/// the two agreed there, and the mismatch it existed to report was invisible in
/// its own output. A diagnostic that can agree while the values differ is not a
/// diagnostic.
#[cfg(feature = "hash-metrics")]
fn pin_skip_line(sha: &str, len: usize, epoch_size_log2: u32) -> String {
    format!(
        "{:<12} transcript pin SKIPPED - elf sha {} ({} bytes, epoch 2^{}); \
         pinned {} ({} bytes, epoch 2^{})",
        "WHIR",
        sha,
        len,
        epoch_size_log2,
        transcript_pin::ELF_SHA256,
        transcript_pin::ELF_LEN,
        transcript_pin::EPOCH_LOG2,
    )
}

/// Asserts the pinned pair, or says out loud why it did not.
#[cfg(feature = "hash-metrics")]
fn check_transcript_pins(
    elf: &[u8],
    epoch_size_log2: u32,
    prove: &crypto::hash_metrics::Counts,
    verify: &crypto::hash_metrics::Counts,
) {
    use sha2::{Digest, Sha256};

    let sha: String = Sha256::digest(elf)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if !pin_applies(&sha, elf.len(), epoch_size_log2) {
        // Never silent. A skipped assert that prints nothing is
        // indistinguishable from one that passed, which is the failure this
        // whole pin exists against.
        println!("{}", pin_skip_line(&sha, elf.len(), epoch_size_log2));
        return;
    }

    let triple = |c: &crypto::hash_metrics::Counts| {
        (
            c.transcript_absorbs,
            c.transcript_squeezes,
            c.transcript_states,
        )
    };
    assert_pinned_pair(triple(prove), triple(verify));
    println!(
        "{:<12} transcript pin OK (both sides, and the owed delta)",
        "WHIR"
    );
}

/// The assertions themselves, split from the guard so they can be reached
/// without the pinned guest.
///
/// ⚠ This split is the point, not tidiness. With the guard and the assertions
/// in one function, the assertions ran on the box and NOWHERE ELSE: a mistyped
/// constant, a swapped pair or a broken delta would have been discovered by a
/// GPU run rather than by `cargo test`. Taking the triples as arguments makes
/// every branch reachable from a laptop, which is why the four tests below
/// exist and why three of them are `should_panic`.
#[cfg(feature = "hash-metrics")]
fn assert_pinned_pair(prove: (u64, u64, u64), verify: (u64, u64, u64)) {
    // Only the state columns are destructured: the two lines are compared whole
    // against their pins, and the delta between them is a property of the
    // CONSTANTS rather than of a measurement — see
    // `the_pinned_constants_differ_by_owed`.
    let (_, _, pt) = prove;
    let (_, _, vt) = verify;

    assert_eq!(
        prove,
        transcript_pin::PROVE,
        "the PROVE-side transcript counts moved"
    );
    assert_eq!(
        verify,
        transcript_pin::VERIFY,
        "the VERIFY-side transcript counts moved"
    );

    // ⚠ NO `owed` ASSERTION HERE, and its absence is deliberate. Once both
    // lines match their pins, their difference is forced — `OWED` is
    // `VERIFY - PROVE` by construction, so a third runtime assertion could
    // never fire. Writing its test is what exposed that: the constructed
    // counter-example was rejected by the VERIFY assertion two lines up,
    // never reaching the delta.
    //
    // The delta is a statement about the CONSTANTS, not about a measurement,
    // so it is checked where it can fail — see
    // [`the_pinned_constants_differ_by_owed`].

    // The control that costs nothing: one `state()` per grind check, so this
    // column and the grind count are two instruments on one quantity.
    assert_eq!(
        pt, vt,
        "the two sides disagree on state reads, which are grind checks on both"
    );
}

/// ★★ The constants are the measurement — asserted against LITERALS.
///
/// Passing `transcript_pin::PROVE` here would be a check that cannot fail: a
/// mutated constant would move the input and the expectation together and the
/// test would pass on any value. The numbers below are written out so that a
/// constant which drifts, is mistyped, or has its two lines swapped fails here,
/// on a laptop, rather than on the box an hour later.
#[cfg(feature = "hash-metrics")]
#[test]
fn the_pinned_pair_is_the_measurement() {
    assert_pinned_pair((583_924, 183_226, 2_996), (584_061, 183_256, 2_996));
}

/// ★ One unit on the prove line fails on the prove assertion.
#[cfg(feature = "hash-metrics")]
#[test]
#[should_panic(expected = "the PROVE-side transcript counts moved")]
fn a_prove_count_off_by_one_is_rejected() {
    assert_pinned_pair((583_925, 183_226, 2_996), (584_061, 183_256, 2_996));
}

/// ★ One unit on the verify line fails on the verify assertion.
#[cfg(feature = "hash-metrics")]
#[test]
#[should_panic(expected = "the VERIFY-side transcript counts moved")]
fn a_verify_count_off_by_one_is_rejected() {
    assert_pinned_pair((583_924, 183_226, 2_996), (584_062, 183_256, 2_996));
}

/// ★★ THE DERIVED DELTA, checked where it can actually fail: on the constants.
///
/// `owed` is the only thing the verifier does that the prover does not, so the
/// two pinned lines must differ by exactly it — `Sum roots.len()` absorbs,
/// `2 x epochs` squeezes, no state reads. That is a claim about the pair of
/// constants, and it is the claim that survives a re-baseline: if someone
/// measures a new run and updates PROVE and VERIFY together, this fires unless
/// `owed` is still `owed`, forcing them to look at why the difference moved
/// rather than carrying a changed protocol into two numbers that agree with
/// each other.
///
/// ⚠ It lives here rather than inside [`assert_pinned_pair`] because there it
/// could never fire: with both lines asserted against their pins, their
/// difference is forced. That was found by writing the test — the constructed
/// counter-example never reached the delta, because the VERIFY assertion
/// rejected it first.
#[cfg(feature = "hash-metrics")]
#[test]
fn the_pinned_constants_differ_by_owed() {
    let (pa, ps, pt) = transcript_pin::PROVE;
    let (va, vs, vt) = transcript_pin::VERIFY;
    assert_eq!(
        (va - pa, vs - ps, vt - pt),
        transcript_pin::OWED,
        "the two pinned lines no longer differ by `owed` — one was re-baselined \
         without the other, or the protocol changed"
    );

    // …and `owed` is itself derived, not observed: 137 absorbs is one per root
    // over the 15 epoch calls, 30 squeezes is two per call. Stating the shape
    // means a future epoch count cannot silently keep the old constant.
    let (oa, os, ot) = transcript_pin::OWED;
    assert_eq!(os, 2 * 15, "`owed` samples twice per epoch call");
    assert_eq!(ot, 0, "`owed` reads no transcript state");
    // ⚠ The absorb count gets no assertion of its own. It is `Sum roots.len()`
    // over the epochs — data from the table shapes, not something derivable
    // here — so any predicate this test could write about it would be either
    // circular (comparing the constant to itself) or vacuous. An earlier draft
    // had `oa % 1 == 0`, which is true of every integer. It is pinned by
    // `VERIFY - PROVE` above and by V1's closed form, which is where it belongs.
    let _ = oa;
}

/// ★★ A sha that agrees on a PREFIX is refused, and the skip line shows why.
///
/// The defect this is written against: the constant's last 48 hex were
/// fabricated, so the guard skipped on the pinned guest itself — and the skip
/// line printed 16 characters of each side, the exact width at which the real
/// sha and the invented one agreed. The failure was invisible in the output of
/// the thing that existed to report it.
///
/// Both halves are needed. The refusal alone would pass with a truncated
/// diagnostic; the diagnostic alone would pass with a comparison that only
/// looked at a prefix.
#[cfg(feature = "hash-metrics")]
#[test]
fn a_sha_agreeing_only_on_the_prefix_is_refused_and_says_so() {
    // Same first 16 hex, different tail — one `format!`, no ELF to forge.
    let near_miss = format!("{}{}", &transcript_pin::ELF_SHA256[..16], "0".repeat(48));
    assert_eq!(near_miss.len(), 64);
    assert_eq!(
        near_miss[..16],
        transcript_pin::ELF_SHA256[..16],
        "the near miss must agree on the prefix, or it tests nothing"
    );
    assert_ne!(near_miss, transcript_pin::ELF_SHA256);

    assert!(
        !pin_applies(
            &near_miss,
            transcript_pin::ELF_LEN,
            transcript_pin::EPOCH_LOG2
        ),
        "a sha differing only after position 16 was accepted: the comparison is \
         looking at a prefix"
    );

    // …and the line it prints must make the difference visible.
    let line = pin_skip_line(
        &near_miss,
        transcript_pin::ELF_LEN,
        transcript_pin::EPOCH_LOG2,
    );
    assert!(
        line.contains(&near_miss),
        "the skip line does not carry the found sha in full: {line}"
    );
    assert!(
        line.contains(transcript_pin::ELF_SHA256),
        "the skip line does not carry the pinned sha in full: {line}"
    );

    // The property in one assertion: whatever the line shows of each side, the
    // two shown values must differ. A prefix display fails here.
    let shown: Vec<&str> = line.split_whitespace().filter(|w| w.len() == 64).collect();
    assert_eq!(
        shown.len(),
        2,
        "expected two 64-hex values in the skip line, found {}: {line}",
        shown.len()
    );
    assert_ne!(
        shown[0], shown[1],
        "the skip line shows the same value twice for a genuine mismatch"
    );
}

/// ★ The pinned constant is a full-width sha, not a truncation.
#[cfg(feature = "hash-metrics")]
#[test]
fn the_pinned_sha_is_full_width() {
    assert_eq!(
        transcript_pin::ELF_SHA256.len(),
        64,
        "a sha256 is 64 hex characters; anything shorter is a display that got \
         pinned"
    );
    assert!(
        transcript_pin::ELF_SHA256
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "the pinned sha is not lowercase hex"
    );
}

/// ★ The guard skips rather than fires on a guest that is not the pinned one.
///
/// Card-free and not `#[ignore]`d, because the skip path is the half that runs
/// on every other invocation of the bench and the half that would fail silently
/// if it were wrong. If the guard were inverted — asserting on the wrong ELF —
/// every run on any other program would panic on counts that were never about
/// it; if it were absent, the pin would be decorative.
///
/// The counts passed in are deliberately absurd. Reaching the assertions with
/// them would panic, so a test that returns at all proves the guard returned
/// first.
#[cfg(feature = "hash-metrics")]
#[test]
fn the_transcript_pin_skips_a_guest_it_does_not_recognise() {
    let nonsense = crypto::hash_metrics::Counts {
        transcript_absorbs: 1,
        transcript_squeezes: 2,
        transcript_states: 3,
        ..Default::default()
    };

    // Wrong bytes, wrong length.
    check_transcript_pins(
        b"not an elf",
        transcript_pin::EPOCH_LOG2,
        &nonsense,
        &nonsense,
    );

    // ⚠ Right length, wrong bytes — the guard must be the sha and not the size,
    // which is the whole point of preferring it to the ELF's name.
    let same_length = vec![0u8; transcript_pin::ELF_LEN];
    check_transcript_pins(
        &same_length,
        transcript_pin::EPOCH_LOG2,
        &nonsense,
        &nonsense,
    );
}

#[test]
#[ignore]
fn continuations() {
    let name = std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "ethrex".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let backend = std::env::var("LAMBDA_VM_BENCH_BACKEND").unwrap_or_else(|_| "both".into());
    let epoch_size_log2: u32 = std::env::var("LAMBDA_VM_BENCH_EPOCH_LOG2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let opts = options();
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "all".into());
    let label = if input.is_empty() { &name } else { &input };
    println!(
        "\n{label} — continuation, epoch_size_log2={epoch_size_log2}, \
RAYON_NUM_THREADS={threads}, backend={backend}"
    );

    let mib = |n: usize| n as f64 / (1024.0 * 1024.0);
    let mut fri = None;
    let mut whir = None;

    if backend != "whir" {
        let start = Instant::now();
        let bundle =
            crate::continuation::prove_continuation(&bytes, &inputs, epoch_size_log2, &opts)
                .expect("univariate continuation");
        let prove = start.elapsed();
        let size = rkyv::to_bytes::<rkyv::rancor::Error>(&bundle)
            .expect("serialize")
            .len();
        let epochs = bundle.num_epochs();
        let start = Instant::now();
        let output = crate::continuation::verify_continuation(&bytes, &bundle, &opts)
            .expect("univariate verify");
        assert!(output.is_some(), "the univariate continuation must verify");
        fri = Some((prove, start.elapsed(), size, epochs));
    }

    if backend != "fri" {
        // Held from the prove window to the verify one so the pair - and the
        // `owed` delta between them - can be asserted together. Uninitialised
        // and assigned exactly once: an `Option` here would carry a `None` the
        // compiler can prove is never read, since the assignment dominates the
        // use and both sit in this one branch.
        #[cfg(feature = "hash-metrics")]
        let prove_counts;
        // ★ Zeroed per arm, so the counts below belong to THIS prove and not to
        // whatever ran before it in the process.
        #[cfg(feature = "cuda")]
        {
            crypto::grinding::reset_gpu_grind_calls();
            multilinear::gpu::reset_call_counters();
        }
        // ★ Same reason, for the transcript counters: the line printed below
        // must be THIS arm's and not the process's running total.
        #[cfg(feature = "hash-metrics")]
        crypto::hash_metrics::reset();
        let start = Instant::now();
        let bundle = crate::multilinear_continuation::prove_continuation(
            &bytes,
            &inputs,
            epoch_size_log2,
            &opts,
        )
        .expect("multilinear continuation");
        let prove = start.elapsed();

        // ★★ READ 0 FOR EVERY WHIR ARM — which dispatches actually reached the
        // card. The `★ WHIR HASH:` banner says which hash was SELECTED; these
        // say which kernels RAN, and the two are not the same claim.
        //
        // A measured RPX arm once came in 14.5x slower than keccak with a
        // correct, KAT-pinned grind kernel sitting unused, because the host-side
        // dispatch had no arm for it. Nothing in this bench's output named the
        // cause: prove time, verify time, proof size and epoch count were all
        // consistent with "the hash is just expensive". A grind count of ZERO
        // beside a commit count of thousands says it in one line.
        #[cfg(feature = "cuda")]
        println!(
            "{:<12} gpu commits {} · host fallbacks {} · keccak grinds {} · rpx grinds {}",
            "WHIR",
            multilinear::gpu::commit_calls(),
            multilinear::gpu::host_fallbacks(),
            crypto::grinding::gpu_grind_calls(),
            crypto::grinding::gpu_grind_calls_rpx(),
        );
        // ★★ WHICH SPONGE THE TRANSCRIPT RAN ON, per arm and on BOTH sides.
        //
        // The line above says which KERNELS ran; this one says which sponge the
        // Fiat-Shamir transcript used, and they are not the same claim. For
        // four measured A/Bs the RPX arm ran an RPX Merkle backend, an RPX
        // grind and a KECCAK transcript, and nothing printed here disagreed.
        //
        // Both sides are printed because one is not evidence. A counter on the
        // RPX side alone reads "rpx > 0, keccak 0" — and that keccak zero is
        // equally true when the keccak transcript ran and nobody instrumented
        // it, which is exactly the state that hid. `unattributed` is printed
        // for the same reason one level out: a third configuration arriving
        // with no counter of its own would otherwise look like silence.
        #[cfg(feature = "hash-metrics")]
        {
            let c = crypto::hash_metrics::snapshot();
            print_transcript_counts("WHIR prove", &c);
            prove_counts = c;
        }
        let size = rkyv::to_bytes::<rkyv::rancor::Error>(&bundle)
            .expect("serialize")
            .len();
        let epochs = bundle.num_epochs();
        // ★★ The VERIFY side gets its own window, and it is the side that
        // matters for recursion: an LFM replays the VERIFIER's transcript, not
        // the prover's. The two are different numbers over different work —
        // differencing one against a closed form derived for the other is the
        // same category error as comparing a prove stopwatch to a verify one.
        #[cfg(feature = "hash-metrics")]
        crypto::hash_metrics::reset();
        let start = Instant::now();
        let ok = crate::multilinear_continuation::verify_continuation(&bytes, &bundle, &opts)
            .expect("multilinear verify");
        assert!(ok, "the multilinear continuation must verify");
        #[cfg(feature = "hash-metrics")]
        {
            let verify_counts = crypto::hash_metrics::snapshot();
            print_transcript_counts("WHIR verify", &verify_counts);
            check_transcript_pins(&bytes, epoch_size_log2, &prove_counts, &verify_counts);
        }
        whir = Some((prove, start.elapsed(), size, epochs));
    }

    println!(
        "{:<12} {:>10} {:>10} {:>12} {:>8}",
        "backend", "prove", "verify", "proof", "epochs"
    );
    for (tag, run) in [("FRI", &fri), ("WHIR", &whir)] {
        if let Some((prove, verify, size, epochs)) = run {
            println!(
                "{tag:<12} {:>9.2}s {:>9.2}s {:>10.2} MiB {epochs:>8}",
                prove.as_secs_f64(),
                verify.as_secs_f64(),
                mib(*size)
            );
        }
    }
    if let (Some(f), Some(w)) = (fri, whir) {
        println!(
            "{:<12} {:>9.2}x {:>9.2}x {:>10.2}x",
            "WHIR/FRI",
            w.0.as_secs_f64() / f.0.as_secs_f64(),
            w.1.as_secs_f64() / f.1.as_secs_f64(),
            w.2 as f64 / f.2 as f64,
        );
    }
}

/// Where a continuation's time goes: preparing an epoch against proving it.
///
/// Replays what [`crate::multilinear_continuation::prove_continuation`] runs,
/// with a clock on each side of the epoch callback. Preparation is the
/// executor and the trace builders, and it is the half a pipeline can hide
/// behind the previous epoch's proof — which is what the univariate driver
/// does and this one does not. The split is what says whether that is worth
/// building.
#[test]
#[ignore]
fn continuation_phases() {
    use crate::multilinear_continuation;
    use crate::tables::trace_builder::DecodeArtifacts;
    use executor::elf::Elf;

    let name = std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "ethrex".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let epoch_size_log2: u32 = std::env::var("LAMBDA_VM_BENCH_EPOCH_LOG2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let opts = options();
    let label = if input.is_empty() { &name } else { &input };
    println!("\n{label} — continuation phases, epoch_size_log2={epoch_size_log2}");

    let whole = Instant::now();
    let elf = Elf::load(&bytes).expect("load");
    let decode_commitment =
        crate::tables::decode::commitment_from_elf(&elf, &opts).expect("decode commitment");
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");

    let mut prepare = Vec::new();
    let mut prove = Vec::new();
    let mut epochs = Vec::new();
    let mut last = Instant::now();
    let boundaries = crate::continuation::for_each_epoch(
        &elf,
        &inputs,
        epoch_size_log2,
        &artifacts,
        |prepared, _| {
            prepare.push(last.elapsed());
            let start = Instant::now();
            epochs.push(multilinear_continuation::prove_epoch(
                &elf,
                &bytes,
                &prepared.register_init,
                prepared.label,
                prepared.traces,
                prepared.is_final,
                &prepared.boundary,
                &opts,
                Some(decode_commitment),
            )?);
            prove.push(start.elapsed());
            last = Instant::now();
            Ok(())
        },
    )
    .expect("the epochs prepare");

    let start = Instant::now();
    let init_page_data = crate::tables::trace_builder::build_init_page_data(
        &crate::tables::trace_builder::build_initial_image_paged(&elf, &inputs),
    );
    let num_private_input_pages = crate::tables::page::private_input_page_count(&inputs);
    let page_bases = crate::continuation::touched_page_bases(&boundaries);
    multilinear_continuation::prove_global(
        &boundaries,
        &bytes,
        &init_page_data,
        &page_bases,
        num_private_input_pages,
        &opts,
    )
    .expect("the cross-epoch proof");
    let global = start.elapsed();
    let total = whole.elapsed();

    let secs = |d: &std::time::Duration| d.as_secs_f64();
    println!("{:<8} {:>10} {:>10}", "epoch", "prepare", "prove");
    for (i, (p, q)) in prepare.iter().zip(&prove).enumerate() {
        println!("{i:<8} {:>9.2}s {:>9.2}s", secs(p), secs(q));
    }
    let prepared: f64 = prepare.iter().map(secs).sum();
    let proved: f64 = prove.iter().map(secs).sum();
    println!(
        "{:<8} {:>9.2}s {:>9.2}s   cross-epoch {:.2}s   total {:.2}s",
        "sum",
        prepared,
        proved,
        secs(&global),
        secs(&total)
    );
    println!(
        "prepare is {:.0}% of the run — what a pipeline could hide behind the previous proof",
        100.0 * prepared / secs(&total)
    );
    // Which pieces ran on device, summed over every epoch. A count far below
    // the number of tables is a phase above that is a CPU number wearing a GPU
    // label — and over a whole continuation that is easy to miss, because the
    // wall clock grows with the epochs either way.
    #[cfg(feature = "cuda")]
    for (tag, count) in [
        ("gpu commits", multilinear::gpu::commit_calls()),
        ("host fallbacks", multilinear::gpu::host_fallbacks()),
        ("gpu sumchecks", multilinear::gpu::sumcheck_calls()),
        ("gpu evals", multilinear::gpu::evaluate_calls()),
        ("gpu trees", multilinear::gpu::tree_calls()),
        ("gpu factors", multilinear::gpu::factor_calls()),
        ("gpu openings", multilinear::gpu::open_calls()),
    ] {
        println!("{tag:<14} {count:>9}");
    }
}

/// Where the multilinear prover's time goes, phase by phase.
///
/// Replays the same pipeline [`multilinear_prove::prove_with_options_and_inputs`]
/// runs, with a clock between the steps. The split that matters is **commit**
/// (NTT + Merkle, already parallel) against **argue** (sumcheck, GKR, LogUp,
/// stacking — all serial today): the second is the parallelisation backlog, and
/// its share is what says whether closing it is worth the work.
#[test]
#[ignore]
fn phases() {
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use math::field::element::FieldElement;
    use stark::multilinear_air::Uniforms;
    use stark::multilinear_table::{self, CommittedTable, CommittedTables, TableLayout};

    use crate::test_utils::{E, F};

    let name =
        std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "all_instructions_64".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let opts = options();
    let label = if input.is_empty() { &name } else { &input };
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "all".into());

    let total = Instant::now();
    let start = Instant::now();
    let elf = Elf::load(&bytes).expect("load");
    let logs = Executor::new(&elf, inputs.clone())
        .and_then(Executor::run)
        .expect("run")
        .logs;
    let execute = start.elapsed();

    let start = Instant::now();
    let mut traces = Traces::from_elf_and_logs(
        &elf,
        &logs,
        &MaxRowsConfig::default(),
        &inputs,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("traces");
    let trace_build = start.elapsed();

    let table_counts = traces.table_counts();
    let airs = crate::VmAirs::new(
        &elf,
        &opts,
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let pairs = airs.air_trace_pairs(&mut traces);
    // The table's own shape, the way `multilinear_prove::shapes_of` reads it:
    // transposing the trace to count it is the whole trace copied for two
    // numbers, and it would land outside every phase below.
    let shapes: Vec<(usize, usize)> = pairs
        .iter()
        .map(|(_, trace, _)| {
            (
                trace.main_table.width,
                trace.main_table.height.trailing_zeros() as usize,
            )
        })
        .collect();
    let mut config = multilinear_prove::chain_config(&shapes);
    // `LAMBDA_VM_BENCH_NO_GRIND` zeroes the proof of work while leaving the query
    // count alone. Not a valid proof — it drops the bits grinding buys — but it
    // is the only way to read the grinding cost off the same run, since asking
    // `with_security` for fewer bits would hand them back as extra queries.
    if std::env::var_os("LAMBDA_VM_BENCH_NO_GRIND").is_some() {
        config.grind = GrindBits::default();
    }

    let start = Instant::now();
    let layouts: Vec<TableLayout<'_, F, E>> = pairs
        .iter()
        .zip(&shapes)
        .map(|((air, _, _), &(width, num_vars))| {
            TableLayout::<F, E>::new(
                air.constraint_program(),
                air.constraints_meta(),
                air.bus_interactions(),
                width,
                num_vars,
                Uniforms::default(),
            )
            .expect("layout")
        })
        .collect();
    let layout = start.elapsed();

    // `commit` is now the one commitment the whole proof shares, so this phase
    // is where the stacking shows up.
    let start = Instant::now();
    let tables: Vec<CommittedTable<'_, F, E>> = layouts
        .into_iter()
        .zip(&pairs)
        .map(|(layout, (_, trace, _))| {
            let mut columns = trace.columns_main();
            CommittedTable::from_layout(layout, |col| core::mem::take(&mut columns[col as usize]))
                .expect("materialize")
        })
        .collect();
    let count = tables.len();
    // ⚠ KECCAK ONLY, and it refuses rather than mislabels.
    //
    // The committed tables escape into the rest of this function, so the hash
    // cannot be a `match` here — both arms would have to return the same type
    // and they do not. Rather than print `★ WHIR HASH: rpx256` over keccak's
    // seconds, this bench asserts the knob agrees with what it actually runs.
    // The hash-arm split lives in `commit_phases`, whose `merkle` pass isolates
    // the hashing anyway, which is the number a hash comparison wants.
    assert_eq!(
        crate::whir_hash_knob::selected(),
        crate::whir_hash_knob::Setting::Keccak,
        "`phases` commits with keccak; run `commit_phases` for the hash arms"
    );
    let committed = CommittedTables::<_, _, KeccakWhir>::commit(tables, &config).expect("commit");
    let commit = start.elapsed();

    // `multi_prove`'s own body, so the tables' arguments and the one opening
    // that settles them can be clocked apart: the transcript makes the loop
    // sequential, so this is the same work in the same order.
    let start = Instant::now();
    let mut transcript = DefaultTranscript::<E>::new(&[]);
    for root in committed.roots() {
        transcript.append_bytes(root);
    }
    let z: FieldElement<E> = transcript.sample_field_element();
    let alpha: FieldElement<E> = transcript.sample_field_element();
    let beta: FieldElement<E> = transcript.sample_field_element();
    let mut points: Vec<Vec<FieldElement<E>>> = Vec::new();
    let mut values: Vec<FieldElement<E>> = Vec::new();
    let mut table_proofs = Vec::with_capacity(committed.tables().len());
    for table in committed.tables() {
        let (proof, point) =
            multilinear_table::prove(table, &z, &alpha, &beta, &mut transcript).expect("table");
        for _ in 0..table.num_committed_columns() {
            points.push(point.clone());
        }
        values.extend(proof.constraint.reduce.column_values.iter().cloned());
        table_proofs.push(proof);
    }
    let tables_argued = start.elapsed();

    let start = Instant::now();
    let group_columns: Vec<&multilinear::mle::Mle<F>> = committed
        .tables()
        .iter()
        .flat_map(|t| t.columns())
        .collect();
    let columns = multilinear::stacked_eval::prove::<F, E, _, KeccakWhir>(
        &committed.groups()[0],
        &group_columns,
        None,
        &multilinear::stacked_eval::Claimed::PerColumn(&points),
        &values,
        &config,
        &mut transcript,
    )
    .expect("the opening");
    let opened = start.elapsed();
    let argue = tables_argued + opened;
    let proof = multilinear_table::MultiProof {
        roots: committed.roots().to_vec(),
        tables: table_proofs,
        columns: vec![columns],
    };
    let total = total.elapsed();

    println!(
        "\n{label} — CPU, RAYON_NUM_THREADS={threads}, {count} tables, {} queries, {} commitment(s)",
        config.num_queries,
        proof.roots.len(),
    );
    println!("{:<14} {:>9} {:>7}", "phase", "seconds", "share");
    for (tag, took) in [
        ("execute", execute),
        ("trace build", trace_build),
        ("layout", layout),
        ("commit", commit),
        ("argue", argue),
        ("  tables", tables_argued),
        ("  opening", opened),
    ] {
        println!(
            "{tag:<14} {:>9.2} {:>6.1}%",
            took.as_secs_f64(),
            100.0 * took.as_secs_f64() / total.as_secs_f64()
        );
    }
    println!("{:<14} {:>9.2}", "total", total.as_secs_f64());
    // Which pieces of the argument ran on device. A zero here with the `cuda`
    // feature on is the signal that a dispatch declined and the phase above is
    // a CPU number wearing a GPU label.
    #[cfg(feature = "cuda")]
    for (tag, count) in [
        ("gpu grinds", stark::gpu_lde::gpu_grind_calls()),
        ("gpu commits", multilinear::gpu::commit_calls()),
        ("host fallbacks", multilinear::gpu::host_fallbacks()),
        ("gpu sumchecks", multilinear::gpu::sumcheck_calls()),
        ("gpu rounds", multilinear::gpu::sumcheck_rounds()),
        ("gpu evals", multilinear::gpu::evaluate_calls()),
        ("gpu trees", multilinear::gpu::tree_calls()),
        ("gpu factors", multilinear::gpu::factor_calls()),
        ("gpu openings", multilinear::gpu::open_calls()),
    ] {
        println!("{tag:<14} {count:>9}");
    }
    assert_eq!(proof.tables.len(), count);
}

/// Where `commit` goes, pass by pass.
///
/// The phase is one Möbius transform, one NTT and one Merkle commit per stacked
/// polynomial, and only the last two have a kernel already. This says which of
/// the three is worth a kernel first, on the real workload rather than on an
/// operation count.
#[test]
#[ignore]
fn commit_phases() {
    use multilinear::whir::{self, Domain};
    use multilinear::whir_commit::CodewordCommitment;
    use stark::multilinear_air::Uniforms;
    use stark::multilinear_table::{self, TableLayout};

    use crate::test_utils::{E, F};

    let name =
        std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "all_instructions_64".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let elf = Elf::load(&bytes).expect("load");
    let logs = Executor::new(&elf, inputs.clone())
        .and_then(Executor::run)
        .expect("run")
        .logs;
    let mut traces = Traces::from_elf_and_logs(
        &elf,
        &logs,
        &MaxRowsConfig::default(),
        &inputs,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("trace");
    let table_counts = traces.table_counts();
    let airs = crate::VmAirs::new(
        &elf,
        &options(),
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let pairs = airs.air_trace_pairs(&mut traces);

    // The columns that actually get committed are the live ones the layout
    // keeps, not every column of the trace.
    let mut columns: Vec<multilinear::mle::Mle<F>> = Vec::new();
    let mut shapes: Vec<(usize, usize)> = Vec::new();
    for (air, trace, _) in &pairs {
        let main = trace.columns_main();
        let num_vars = main[0].len().trailing_zeros() as usize;
        let layout = TableLayout::<F, E>::new(
            air.constraint_program(),
            air.constraints_meta(),
            air.bus_interactions(),
            main.len(),
            num_vars,
            Uniforms::default(),
        )
        .expect("layout");
        let keys = layout.column_keys().to_vec();
        shapes.push((keys.len(), num_vars));
        for key in keys {
            columns
                .push(multilinear::mle::Mle::new(main[key.col as usize].clone()).expect("column"));
        }
    }
    let config = multilinear_prove::chain_config(&shapes);
    let layout = multilinear_table::global_layout(&shapes).expect("global layout");
    let start = Instant::now();
    let polys = layout
        .stack(&multilinear::stacking::borrow(&columns))
        .expect("stack");
    let stack = start.elapsed();
    drop(columns);

    let mut lift = std::time::Duration::ZERO;
    let mut encode = std::time::Duration::ZERO;
    let mut merkle = std::time::Duration::ZERO;
    for poly in &polys {
        let domain = Domain::<F>::new(poly.num_vars() + config.log_blowup).expect("domain");
        let start = Instant::now();
        let coeffs = whir::lift_coefficients(poly);
        lift += start.elapsed();
        let start = Instant::now();
        let codeword = whir::encode::<F, F>(&coeffs, &domain).expect("encode");
        encode += start.elapsed();
        let start = Instant::now();
        // ★ The `merkle` pass is the LEAF AND TREE HASHING, alone — stack,
        // lift and encode are clocked apart above. So this line is the hash
        // term itself, and it has to follow the knob or the two arms are not
        // comparable.
        crate::with_whir_hash!(|H| {
            let commitment = CodewordCommitment::<_, H>::new(
                &codeword,
                config
                    .schedule(poly.num_vars())
                    .first()
                    .copied()
                    .unwrap_or(0),
            )
            .expect("commit");
            // Dropped inside the arm: the commitment's type names `H`, so it
            // cannot leave the block. That is also why this is the bench the
            // hash arms run through — nothing here escapes.
            drop(commitment);
        });
        merkle += start.elapsed();
    }

    let label = if input.is_empty() { &name } else { &input };
    let total = stack + lift + encode + merkle;
    println!(
        "\n{label} — {} stacked polynomials of 2^{} cells, blowup {}",
        polys.len(),
        layout.n_stack(),
        1 << config.log_blowup,
    );
    println!("{:<14} {:>9} {:>7}", "pass", "seconds", "share");
    for (tag, took) in [
        ("stack", stack),
        ("lift", lift),
        ("encode", encode),
        ("merkle", merkle),
    ] {
        println!(
            "{tag:<14} {:>9.2} {:>6.1}%",
            took.as_secs_f64(),
            100.0 * took.as_secs_f64() / total.as_secs_f64()
        );
    }
    println!("{:<14} {:>9.2}", "total", total.as_secs_f64());
}

/// What a proof is made of, part by part.
///
/// The multilinear proof grows with the **number of tables**, and the suspicion
/// is that the per-table WHIR opening is why: every table commits on its own,
/// so every table pays its own queries. This says how much of the bytes that
/// actually is, which is what decides whether stacking the tables together is
/// worth the work.
#[test]
#[ignore]
fn proof_composition() {
    let name =
        std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "all_instructions_64".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let proof = multilinear_prove::prove_with_options_and_inputs(
        &bytes,
        &inputs,
        &options(),
        &MaxRowsConfig::default(),
    )
    .expect("prove");

    let size = |v: &[u8]| v.len() as f64 / (1024.0 * 1024.0);
    let ser = |x: &dyn Fn() -> Vec<u8>| x();
    let total = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
        .expect("whole")
        .len();

    // Each part on its own, summed across tables. The opening is not among
    // them: there is one for the whole proof, not one per table.
    let mut gkr = 0usize;
    let mut sumcheck = 0usize;
    let mut reduce = 0usize;
    let mut factor_values = 0usize;
    for t in &proof.proof.tables {
        gkr += rkyv::to_bytes::<rkyv::rancor::Error>(&t.gkr)
            .expect("gkr")
            .len();
        sumcheck += rkyv::to_bytes::<rkyv::rancor::Error>(&t.constraint.sumcheck)
            .expect("sumcheck")
            .len();
        reduce += rkyv::to_bytes::<rkyv::rancor::Error>(&t.constraint.reduce)
            .expect("reduce")
            .len();
        factor_values += rkyv::to_bytes::<rkyv::rancor::Error>(&t.constraint.factor_values)
            .expect("factor_values")
            .len();
    }
    let columns = rkyv::to_bytes::<rkyv::rancor::Error>(&proof.proof.columns)
        .expect("columns")
        .len();
    let _ = ser;

    let label = if input.is_empty() { &name } else { &input };
    println!(
        "\n{label} — {} tables, {} commitment(s)",
        proof.proof.tables.len(),
        proof.proof.roots.len(),
    );

    println!(
        "{:<18} {:>10} {:>8} {:>12}",
        "part", "MiB", "share", "per table"
    );
    for (tag, n) in [
        ("WHIR opening", columns),
        ("GKR", gkr),
        ("sumcheck", sumcheck),
        ("claim reduce", reduce),
        ("factor values", factor_values),
    ] {
        println!(
            "{tag:<18} {:>10.2} {:>7.1}% {:>11.3}",
            size(&vec![0u8; n]),
            100.0 * n as f64 / total as f64,
            size(&vec![0u8; n]) / proof.proof.tables.len() as f64,
        );
    }
    println!("{:<18} {:>10.2}", "whole proof", size(&vec![0u8; total]));
}

/// How big the constraint DAG is per table — the thing `IrShape::combine`
/// walks once per hypercube index.
#[test]
#[ignore]
fn constraint_program_sizes() {
    use crate::test_utils::{E, F};
    let bytes = elf_bytes("ethrex");
    let inputs = input_bytes("ethrex_10_transfers");
    let elf = Elf::load(&bytes).unwrap();
    let logs = Executor::new(&elf, inputs.clone())
        .and_then(Executor::run)
        .unwrap()
        .logs;
    let mut traces = Traces::from_elf_and_logs(
        &elf,
        &logs,
        &MaxRowsConfig::default(),
        &inputs,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .unwrap();
    let table_counts = traces.table_counts();
    let airs = crate::VmAirs::new(
        &elf,
        &options(),
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let pairs = airs.air_trace_pairs(&mut traces);
    // What each table would need resident on a device: its factor tables, which
    // is what the sumcheck folds and the biggest thing a round touches.
    let mut sizes: Vec<(String, usize, usize, usize, f64, f64)> = pairs
        .iter()
        .map(|(air, trace, _)| {
            let columns = trace.columns_main();
            let rows = columns[0].len();
            let layout = stark::multilinear_table::TableLayout::<F, E>::new(
                air.constraint_program(),
                air.constraints_meta(),
                air.bus_interactions(),
                columns.len(),
                rows.trailing_zeros() as usize,
                stark::multilinear_air::Uniforms::default(),
            )
            .expect("layout");
            let factors = layout.kinds().len();
            // ext3 is three Goldilocks limbs.
            let bytes = factors as f64 * rows as f64 * 24.0;
            // The fraction tree: an input layer indexed by (interaction, row),
            // and every level above it — which is one more of the same again.
            let interactions = air.bus_interactions().len().max(1);
            let layer = (rows * interactions.next_power_of_two()) as f64;
            let tree = layer * 2.0 * 2.0 * 24.0;
            (
                air.name().to_string(),
                air.constraint_program().nodes.len(),
                rows,
                factors,
                bytes / (1u64 << 30) as f64,
                tree / (1u64 << 30) as f64,
            )
        })
        .collect();
    let factors_total: f64 = sizes.iter().map(|s| s.4).sum();
    let tree_max = sizes.iter().map(|s| s.5).fold(0.0f64, f64::max);
    sizes.sort_by(|a, b| (b.4 + b.5).partial_cmp(&(a.4 + a.5)).unwrap());
    sizes.dedup_by(|a, b| a.0 == b.0);
    println!(
        "\n{:<22} {:>10} {:>9} {:>8} {:>10} {:>10}",
        "table", "DAG nodes", "rows", "factors", "ext3 GiB", "GKR GiB"
    );
    for (name, nodes, rows, factors, gib, tree) in sizes.iter().take(10) {
        println!("{name:<22} {nodes:>10} {rows:>9} {factors:>8} {gib:>9.2} {tree:>9.2}");
    }
    println!("\nfactores de todas las tablas juntos: {factors_total:.2} GiB");
    println!("arbol de fracciones mas grande:      {tree_max:.2} GiB");
}

/// ★ M2 (lane V1, uncommitted measurement): the exact WHIR stack shape of a real
/// epoch — per-table `(name, width, rows)`, then `n_stack` / `num_polys` per
/// commitment group, the rounds each chain runs and the query count the config
/// derives. Pure shape logic on top of the executor; no card, no proof.
#[test]
#[ignore]
fn whir_epoch_shapes() {
    use crate::multilinear_continuation::{epoch_groups, global_groups};
    use crate::multilinear_prove::{chain_config, stacks};
    use crate::tables::trace_builder::DecodeArtifacts;
    use executor::elf::Elf;

    let name = std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "ethrex".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let epoch_size_log2: u32 = std::env::var("LAMBDA_VM_BENCH_EPOCH_LOG2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(21);
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let opts = options();
    let elf = Elf::load(&bytes).expect("load");
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    println!("\n== M2 shapes: {input} epoch_size_log2={epoch_size_log2} ==");

    let mut totals: Vec<(u64, usize, usize, usize, usize, usize)> = Vec::new();
    let boundaries = crate::continuation::for_each_epoch(
        &elf,
        &inputs,
        epoch_size_log2,
        &artifacts,
        |prepared, _| {
            let mut traces = prepared.traces;
            crate::tables::bitwise::update_multiplicities(
                &mut traces.bitwise,
                &crate::tables::local_to_global::collect_bitwise_from_l2g(&prepared.boundary),
            );
            let reg_fini = crate::tables::register::fini_from_trace(&traces.register);
            let table_counts = traces.table_counts();
            let airs = crate::continuation::build_epoch_airs(
                &elf,
                &opts,
                &[],
                &table_counts,
                &prepared.register_init,
                &reg_fini,
                prepared.is_final,
                None,
            );
            let l2g_air = crate::continuation::l2g_memory_air(&opts, prepared.label);
            let mut l2g_trace =
                crate::tables::local_to_global::generate_local_to_global_trace(&prepared.boundary);
            let mut pairs = airs.air_trace_pairs(&mut traces);
            pairs.push((&l2g_air, &mut l2g_trace, &()));

            let shapes: Vec<(usize, usize)> = pairs
                .iter()
                .map(|(_, t, _)| {
                    (
                        t.main_table.width,
                        t.main_table.height.trailing_zeros() as usize,
                    )
                })
                .collect();
            if prepared.index == 0 {
                println!("\n-- epoch 0 per-table census --");
                println!("{:<16} {:>7} {:>10} {:>6} {:>14}", "table", "width", "rows", "vars", "cells");
                for ((air, _, _), &(w, v)) in pairs.iter().zip(&shapes) {
                    println!(
                        "{:<16} {w:>7} {:>10} {v:>6} {:>14}",
                        air.name(),
                        1usize << v,
                        (w as u64) << v
                    );
                }
            }
            let config = chain_config(&shapes);
            let sizes = epoch_groups(shapes.len());
            let (layouts, _d) = stacks(&shapes, &sizes, &config).expect("stacks");
            let cells: u64 = shapes.iter().map(|&(w, v)| (w as u64) << v).sum();
            let chains: usize = layouts.iter().map(|l| l.num_polys()).sum();
            let rounds: usize = layouts
                .iter()
                .map(|l| l.num_polys() * l.n_stack().div_ceil(config.log_folding))
                .sum();
            println!(
                "epoch {:>2}: tables {:>2} cells {:>12} | group0 n_stack {:>2} polys {:>2} | bookend n_stack {:>2} polys {:>2} | chains {:>2} rounds {:>3} | Q {}",
                prepared.index,
                shapes.len(),
                cells,
                layouts[0].n_stack(),
                layouts[0].num_polys(),
                layouts[1].n_stack(),
                layouts[1].num_polys(),
                chains,
                rounds,
                config.num_queries,
            );
            totals.push((
                prepared.index,
                shapes.len(),
                cells as usize,
                chains,
                rounds,
                config.num_queries,
            ));
            Ok(())
        },
    )
    .expect("epochs prepare");

    // The cross-epoch proof's groups: one per bookend, then the global-memory tables.
    let init_page_data = crate::tables::trace_builder::build_init_page_data(
        &crate::tables::trace_builder::build_initial_image_paged(&elf, &inputs),
    );
    let num_private_input_pages = crate::tables::page::private_input_page_count(&inputs);
    let page_bases = crate::continuation::touched_page_bases(&boundaries);
    let gm_configs = crate::continuation::global_memory_configs_from_init_page_data(
        &page_bases,
        &init_page_data,
        num_private_input_pages,
        true,
    );
    let l2g_airs: Vec<_> = (0..boundaries.len())
        .map(|i| {
            crate::continuation::l2g_global_air(
                &opts,
                crate::tables::local_to_global::epoch_label(i as u64),
            )
        })
        .collect();
    let gm_airs: Vec<_> = gm_configs
        .iter()
        .map(|c| crate::continuation::global_memory_air(&opts, c, None))
        .collect();
    let mut l2g_traces: Vec<_> = boundaries
        .iter()
        .map(|e| crate::tables::local_to_global::generate_local_to_global_trace(e.as_slice()))
        .collect();
    let mut final_state: crate::tables::global_memory::FiniStateMap =
        std::collections::HashMap::new();
    for epoch in &boundaries {
        for b in epoch.iter() {
            final_state.insert(
                b.address,
                crate::tables::global_memory::FiniState {
                    value: (b.fini.value & 0xFF) as u8,
                    epoch: b.fini.epoch,
                },
            );
        }
    }
    let mut gm_traces: Vec<_> = gm_configs
        .iter()
        .map(|c| crate::tables::global_memory::generate_global_trace(c, &final_state))
        .collect();
    let mut gpairs: Vec<crate::AirTracePair<'_>> = Vec::new();
    for (a, t) in l2g_airs.iter().zip(l2g_traces.iter_mut()) {
        gpairs.push((a, t, &()));
    }
    for (a, t) in gm_airs.iter().zip(gm_traces.iter_mut()) {
        gpairs.push((a, t, &()));
    }
    let gshapes: Vec<(usize, usize)> = gpairs
        .iter()
        .map(|(_, t, _)| {
            (
                t.main_table.width,
                t.main_table.height.trailing_zeros() as usize,
            )
        })
        .collect();
    let gconfig = chain_config(&gshapes);
    let gsizes = global_groups(boundaries.len(), gm_configs.len());
    let (glayouts, _gd) = stacks(&gshapes, &gsizes, &gconfig).expect("global stacks");
    let gchains: usize = glayouts.iter().map(|l| l.num_polys()).sum();
    let grounds: usize = glayouts
        .iter()
        .map(|l| l.num_polys() * l.n_stack().div_ceil(gconfig.log_folding))
        .sum();
    println!(
        "\nGLOBAL: pages {} tables {} groups {} chains {} rounds {} Q {}",
        gm_configs.len(),
        gshapes.len(),
        gsizes.len(),
        gchains,
        grounds,
        gconfig.num_queries
    );
    for (i, l) in glayouts.iter().enumerate().take(3) {
        println!(
            "   group {i}: n_stack {} polys {}",
            l.n_stack(),
            l.num_polys()
        );
    }
    if let Some(l) = glayouts.last() {
        println!(
            "   group {} (global memory): n_stack {} polys {}",
            glayouts.len() - 1,
            l.n_stack(),
            l.num_polys()
        );
    }
    let ec: usize = totals.iter().map(|t| t.3).sum();
    let er: usize = totals.iter().map(|t| t.4).sum();
    println!(
        "\n★ TOTAL over {} epochs + global: chains {} rounds {} | grinds (3R-1 per chain) {}",
        totals.len(),
        ec + gchains,
        er + grounds,
        3 * (er + grounds) - (ec + gchains)
    );
}

/// ★ The per-table AIR properties V1's census needs, which `whir_epoch_shapes`
/// does not print: the interaction count, the factor layout, the frame offsets
/// and the constraint degree.
///
/// All four are EXECUTION-INDEPENDENT — they are properties of the AIR and of
/// the table's width, not of what the guest did — but building the AIR set
/// needs the guest ELF, so this is a box run and not a laptop one. It is a
/// sibling of `whir_epoch_shapes` rather than an edit of it, so that
/// instrument's log stays the one sh1 filed.
///
/// What each column is for:
///
/// - `I` sets the GKR ladder's height: `input_layer_vars(I, num_vars)` rounds,
///   and the bus weights are `eq_evals` over `ceil(log2 I)` variables, which is
///   the one term of a table's cost that is exponential in anything;
/// - `factors` is what `claim_reduce` batches and `offsets` is how many `shift`
///   kernels it spends — one per DISTINCT offset, however many factors read it;
/// - `degree` is the zerocheck rule's, so the main batched sumcheck runs at
///   `max(degree + 1, 2)` (`multilinear_table.rs:765`, `batch.rs:264`);
/// - `roots` is the length `beta_powers` must have.
#[test]
#[ignore = "needs the guest ELF and builds every epoch's AIRs"]
fn whir_table_shapes() {
    use crate::tables::trace_builder::DecodeArtifacts;
    use executor::elf::Elf;
    use multilinear::constraint_argument::FactorKind;
    use stark::multilinear_air::Uniforms;
    use stark::multilinear_table::TableLayout;

    let name = std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "ethrex".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let epoch_size_log2: u32 = std::env::var("LAMBDA_VM_BENCH_EPOCH_LOG2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(21);
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let opts = options();
    let elf = Elf::load(&bytes).expect("load");
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    println!("\n== V1 table shapes: {input} epoch_size_log2={epoch_size_log2} ==");

    crate::continuation::for_each_epoch(
        &elf,
        &inputs,
        epoch_size_log2,
        &artifacts,
        |prepared, _| {
            if prepared.index != 0 {
                return Ok(());
            }
            let mut traces = prepared.traces;
            crate::tables::bitwise::update_multiplicities(
                &mut traces.bitwise,
                &crate::tables::local_to_global::collect_bitwise_from_l2g(&prepared.boundary),
            );
            let reg_fini = crate::tables::register::fini_from_trace(&traces.register);
            let table_counts = traces.table_counts();
            let airs = crate::continuation::build_epoch_airs(
                &elf,
                &opts,
                &[],
                &table_counts,
                &prepared.register_init,
                &reg_fini,
                prepared.is_final,
                None,
            );
            let l2g_air = crate::continuation::l2g_memory_air(&opts, prepared.label);
            let mut l2g_trace =
                crate::tables::local_to_global::generate_local_to_global_trace(&prepared.boundary);
            let mut pairs = airs.air_trace_pairs(&mut traces);
            pairs.push((&l2g_air, &mut l2g_trace, &()));

            println!(
                "{:<16} {:>5} {:>6} {:>4} {:>8} {:>8} {:>7} {:>7} {:>7} offsets",
                "table", "vars", "width", "I", "columns", "factors", "public", "degree", "roots"
            );
            let mut degree_rounds = 0usize;
            for (air, trace, _) in pairs.iter() {
                let width = trace.main_table.width;
                let num_vars = trace.main_table.height.trailing_zeros() as usize;
                let layout = TableLayout::<
                    crate::tables::types::GoldilocksField,
                    crate::tables::types::GoldilocksExtension,
                >::new(
                    air.constraint_program(),
                    air.constraints_meta(),
                    air.bus_interactions(),
                    width,
                    num_vars,
                    Uniforms::default(),
                )
                .expect("the table lays out");
                let kinds = layout.kinds();
                let public = kinds
                    .iter()
                    .filter(|k| matches!(k, FactorKind::Public))
                    .count();
                let mut offsets: Vec<usize> = kinds
                    .iter()
                    .filter_map(|k| k.source().map(|s| s.offset))
                    .collect();
                offsets.sort_unstable();
                offsets.dedup();
                let degree = layout.shape().degree();
                degree_rounds += num_vars;
                println!(
                    "{:<16} {num_vars:>5} {width:>6} {:>4} {:>8} {:>8} {public:>7} {degree:>7} {:>7} {offsets:?}",
                    air.name(),
                    air.bus_interactions().len(),
                    layout.num_columns(),
                    kinds.len(),
                    layout.shape().num_roots(),
                );
            }
            println!("epoch 0: {} tables, {degree_rounds} main-sumcheck rounds over all of them", pairs.len());
            Ok(())
        },
    )
    .expect("epochs prepare");
}
