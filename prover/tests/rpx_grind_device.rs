//! ★★ The test that fails without the RPX device grind arm.
//!
//! Needs a GPU, like every other test in this file's family:
//!
//! ```text
//! cargo test -p lambda-vm-prover --release --features cuda --test rpx_grind_device -- --nocapture
//! ```
//!
//! # What it is for
//!
//! `generate_nonce_maybe_gpu` dispatches the nonce search to a device kernel
//! chosen by the digest. While that dispatch read "is `D` platform keccak", an
//! RPX configuration silently took the HOST search — ~2^20 RPX permutations per
//! grind, thousands of grinds per block proof. A measured WHIR block arm came in
//! at 571 s against keccak's 39 s, ~510 s of it on this line, with a correct,
//! KAT-pinned `rpx_grind_search` sitting unused in the cubin. Nothing failed;
//! the proof was valid; only the clock said so.
//!
//! So the assertions here are about WHICH KERNEL RAN, not about whether a grind
//! succeeded — a grind succeeds either way, which is exactly why the regression
//! was invisible.
//!
//! # Two counters, not one
//!
//! A single "device grinds" counter is satisfied by the keccak kernel firing
//! under an RPX configuration, which is the precise failure the dispatch exists
//! to prevent. So the RPX arm must show `rpx > 0 && keccak == 0` and the keccak
//! arm `keccak > 0 && rpx == 0`; either counter alone would pass on a dispatch
//! that ignored its key.
#![cfg(feature = "cuda")]

use crypto::grinding::{
    generate_nonce_maybe_gpu, gpu_grind_calls, gpu_grind_calls_rpx, is_valid_nonce,
    reset_gpu_grind_calls,
};
use multilinear::whir_hash::{GrindingDigest, KeccakWhir, RpxWhir};

type Rpx = GrindingDigest<RpxWhir>;
type Keccak = GrindingDigest<KeccakWhir>;

/// The production grind depth. Also comfortably above `GRIND_MIN_FACTOR`, below
/// which the dispatch keeps the search on the CPU on purpose.
const FACTOR: u8 = 20;

fn seed_of(byte: u8) -> [u8; 32] {
    [byte; 32]
}

/// Every test here resets and then reads the PROCESS-GLOBAL grind counters,
/// and libtest runs tests on parallel threads by default. Unserialised, one
/// test's grind or reset lands between another's reset and read: the keccak
/// test's grind is counted under the RPX test, or a reset shrinks the counter
/// test's three grinds to one — failures of the harness, with the dispatch
/// correct. The Makefile runs its GPU counter suites with `--test-threads=1`
/// for the same reason; this lock makes the file correct under the default
/// too. Each test takes it for its whole body.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    // A test that failed while holding the lock poisons it. The next test
    // resets the counters before reading them, so the poison carries nothing.
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// ★★ An RPX grind runs on the RPX kernel, and the nonce it returns is one the
/// host accepts.
///
/// Remove the `Arm::Rpx256` branch from `crypto::grinding`'s dispatch and this
/// reads `rpx grinds 0` and fails — the search still finds a nonce, on the host,
/// which is the whole point.
#[test]
fn an_rpx_grind_runs_on_the_rpx_kernel() {
    let _serial = serial();
    let seed = seed_of(0xA7);
    reset_gpu_grind_calls();

    let nonce = generate_nonce_maybe_gpu::<Rpx>(&seed, FACTOR).expect("a nonce exists");

    println!(
        "rpx grinds {} · keccak grinds {} · nonce {nonce}",
        gpu_grind_calls_rpx(),
        gpu_grind_calls()
    );
    assert!(
        gpu_grind_calls_rpx() > 0,
        "the RPX grind did not reach the device: the dispatch has no RPX arm, \
         or the kernel returned a nonce the host rejected and it fell back"
    );
    assert_eq!(
        gpu_grind_calls(),
        0,
        "the KECCAK kernel ran under an RPX configuration"
    );
    assert!(
        is_valid_nonce::<Rpx>(&seed, nonce, FACTOR),
        "the device nonce must satisfy the host predicate"
    );
}

/// The keccak arm still runs on keccak's kernel — so the change above moved a
/// dispatch rather than replacing one.
#[test]
fn a_keccak_grind_still_runs_on_the_keccak_kernel() {
    let _serial = serial();
    let seed = seed_of(0x5C);
    reset_gpu_grind_calls();

    let nonce = generate_nonce_maybe_gpu::<Keccak>(&seed, FACTOR).expect("a nonce exists");

    println!(
        "keccak grinds {} · rpx grinds {} · nonce {nonce}",
        gpu_grind_calls(),
        gpu_grind_calls_rpx()
    );
    assert!(gpu_grind_calls() > 0, "the keccak grind left the device");
    assert_eq!(
        gpu_grind_calls_rpx(),
        0,
        "the RPX kernel ran under a keccak configuration"
    );
    assert!(is_valid_nonce::<Keccak>(&seed, nonce, FACTOR));
}

/// ★ The device reproduces the ORACLE table, nonce for nonce.
///
/// The three rows of `RPX_GRIND_VECTORS` from
/// `crypto/math-cuda/tests/host_kat/rpx_kat_vectors.h` — the per-table branch's
/// host implementation, which this repository did not produce. The kernel
/// `atomicMin`s, so it returns the SMALLEST valid nonce, which is the column
/// recorded there.
///
/// This is what makes the two counter tests more than launch counts: they say a
/// kernel ran, this says it computed the right hash.
#[test]
fn the_device_reproduces_the_oracle_grind_vectors() {
    let _serial = serial();
    for (byte, factor, want) in [(90u8, 12u8, 1342u64), (17, 13, 300), (32, 14, 705)] {
        let seed = seed_of(byte);
        reset_gpu_grind_calls();
        let got = generate_nonce_maybe_gpu::<Rpx>(&seed, factor).expect("a nonce exists");
        assert!(
            gpu_grind_calls_rpx() > 0,
            "seed 0x{byte:02x}: the search did not reach the RPX kernel"
        );
        assert_eq!(
            got, want,
            "seed 0x{byte:02x}, factor {factor}: the device must return the oracle's nonce"
        );
    }
}

/// ✓ The kill switch still works, and its effect is visible in the counter —
/// so a zero counter in a real run can be read as "no device grind" rather than
/// ambiguously.
///
/// `LAMBDA_VM_NO_GPU_GRIND` is read once into a `OnceLock`, so this cannot be
/// toggled mid-process; it is asserted structurally instead, by requiring the
/// counter and the search to agree about whether a device was used.
#[test]
fn a_device_grind_is_exactly_what_the_counter_reports() {
    let _serial = serial();
    let seed = seed_of(0x11);
    reset_gpu_grind_calls();
    assert_eq!(gpu_grind_calls_rpx(), 0, "reset must zero the counter");

    let n = 3;
    for i in 0..n {
        let mut s = seed;
        s[0] = i as u8;
        let nonce = generate_nonce_maybe_gpu::<Rpx>(&s, FACTOR).expect("a nonce exists");
        assert!(is_valid_nonce::<Rpx>(&s, nonce, FACTOR));
    }
    assert_eq!(
        gpu_grind_calls_rpx(),
        n,
        "the counter must be one per successful device grind, not a flag"
    );
}

/// ★ The prove split's `n/airs on device` counts RPX device grinds.
///
/// Each `PROVE SPLIT` line takes its device grinds as a delta of
/// `stark::prove_split::device_grinds_now`. That read the keccak counter alone,
/// so under RPX every line said `0/airs on device` while every table ground on
/// the card. This reads the same function around one RPX device grind; the
/// keccak-only count reads 0 here and fails.
#[test]
fn the_prove_split_counts_an_rpx_device_grind() {
    let _serial = serial();
    let seed = seed_of(0x3D);
    reset_gpu_grind_calls();
    let before = stark::prove_split::device_grinds_now();

    let nonce = generate_nonce_maybe_gpu::<Rpx>(&seed, FACTOR).expect("a nonce exists");

    assert!(is_valid_nonce::<Rpx>(&seed, nonce, FACTOR));
    assert_eq!(gpu_grind_calls_rpx(), 1, "the grind ran on the RPX kernel");
    assert_eq!(
        stark::prove_split::device_grinds_now() - before,
        1,
        "the prove split must count the RPX device grind"
    );
}
