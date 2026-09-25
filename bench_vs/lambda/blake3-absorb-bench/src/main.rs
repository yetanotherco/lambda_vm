//! Guest cycle microbenchmark for the BLAKE3 chained-absorb ecall.
//!
//! One binary per (arm, message length) pair. `N` hashes of a `LEN`-byte
//! message are folded into a 32-byte XOR accumulator, which is committed, so
//! nothing can be dead-code eliminated.
//!
//! # The arms, and why there are six
//!
//! The brief's control was "one binary, alignment the only difference". That
//! is a real control but it is NOT clean on its own: an unaligned message also
//! makes `Blake3Chain`'s `copy_from_slice` into the pending block cost more,
//! so an alignment-only A/B charges the absorb path with a memcpy saving it
//! did not earn. So both factors — message alignment, and whether
//! `crypto/blake3-absorb` is compiled in — are varied independently:
//!
//! | arm            | message  | `crypto/blake3-absorb` | what it is                     |
//! |----------------|----------|------------------------|--------------------------------|
//! | `arm_none`     | aligned  | off                    | loop + startup baseline        |
//! | `arm_keccak`   | aligned  | off                    | `keccak_permute` precompile    |
//! | `arm_b3old`    | aligned  | off                    | pre-existing block-at-a-time   |
//! | `arm_b3oldun`  | 1-offset | off                    | same, unaligned message        |
//! | `arm_b3single` | 1-offset | ON                     | absorb compiled in, declined   |
//! | `arm_b3absorb` | aligned  | ON                     | the bulk-absorb path           |
//!
//! `arm_b3old` vs `arm_b3absorb` is the headline A/B: equal alignment, the
//! ecall the only difference. `arm_b3single` vs `arm_b3absorb` is the brief's
//! same-binary A/B. `arm_b3old` vs `arm_b3oldun` prices the misalignment, and
//! `arm_b3oldun` vs `arm_b3single` prices merely having the absorb arm
//! compiled in (it should be ~0 — the path is declined at the alignment test).
//!
//! The accelerator call counts printed by `cli execute --cycles` are the proof
//! each arm is what it claims.

#![no_main]

use digest::Digest;

// ---------------------------------------------------------------------------
// Shape selection
// ---------------------------------------------------------------------------

#[cfg(not(any(
    feature = "arm_none",
    feature = "arm_keccak",
    feature = "arm_b3old",
    feature = "arm_b3oldun",
    feature = "arm_b3single",
    feature = "arm_b3absorb"
)))]
compile_error!("select exactly one arm feature");

#[cfg(not(any(
    feature = "len64",
    feature = "len256",
    feature = "len1024",
    feature = "len4096"
)))]
compile_error!("select exactly one length feature");

#[cfg(feature = "len64")]
const LEN: usize = 64;
#[cfg(feature = "len256")]
const LEN: usize = 256;
#[cfg(feature = "len1024")]
const LEN: usize = 1024;
#[cfg(feature = "len4096")]
const LEN: usize = 4096;

/// Hashes per run. Two builds per (arm, length): the per-hash cost is the
/// DIFFERENCE between them divided by the difference in `N`, which cancels
/// program startup exactly instead of trusting a separate baseline binary's
/// startup to be the same as this one's (measured: it is not).
#[cfg(feature = "n1000")]
const N: usize = 1000;
#[cfg(feature = "n2000")]
const N: usize = 2000;

#[cfg(not(any(feature = "n1000", feature = "n2000")))]
compile_error!("select exactly one iteration-count feature");

/// Byte offset of the message inside the 8-aligned buffer. `1` on the two
/// deliberately-unaligned arms; that misalignment is what makes
/// `bulk_absorb_blocks` return 0.
#[cfg(any(feature = "arm_b3oldun", feature = "arm_b3single"))]
const OFF: usize = 1;
#[cfg(not(any(feature = "arm_b3oldun", feature = "arm_b3single")))]
const OFF: usize = 0;

/// The message buffer, forced to 8-byte alignment. A plain `[u8; _]` has
/// alignment 1, which would make the "aligned" arms aligned only by luck.
#[repr(align(8))]
struct Aligned([u8; LEN + 8]);

// ---------------------------------------------------------------------------
// The arms
// ---------------------------------------------------------------------------

#[cfg(feature = "arm_none")]
#[inline(never)]
fn hash_it(msg: &[u8]) -> [u8; 32] {
    // No hash. Touch the input at both ends so the loop and the message
    // mutation stay, and nothing else — this is the cost that is subtracted.
    let mut d = [0u8; 32];
    d[0] = msg[0];
    d[1] = msg[msg.len() - 1];
    d
}

#[cfg(feature = "arm_keccak")]
#[inline(never)]
fn hash_it(msg: &[u8]) -> [u8; 32] {
    let mut h = <crypto::hash::platform_keccak::PlatformKeccak256 as Digest>::new();
    Digest::update(&mut h, msg);
    Digest::finalize(h).into()
}

#[cfg(any(
    feature = "arm_b3old",
    feature = "arm_b3oldun",
    feature = "arm_b3single",
    feature = "arm_b3absorb"
))]
#[inline(never)]
fn hash_it(msg: &[u8]) -> [u8; 32] {
    let mut h = <crypto::hash::platform_blake3::PlatformBlake3 as Digest>::new();
    Digest::update(&mut h, msg);
    Digest::finalize(h).into()
}

#[unsafe(export_name = "main")]
pub fn main() -> ! {
    lambda_vm_syscalls::allocator::init_allocator();

    // Panic -> sys_panic; unwinding is very expensive in-guest.
    const PANIC_MSG: &str = "PANICKED";
    std::panic::set_hook(Box::new(|_| unsafe {
        lambda_vm_syscalls::syscalls::sys_panic(PANIC_MSG.as_ptr(), PANIC_MSG.len())
    }));

    let mut buf = Aligned([0u8; LEN + 8]);
    for (i, b) in buf.0.iter_mut().enumerate() {
        // The same generator `crypto`'s chain KATs use.
        *b = (i as u8).wrapping_mul(37).wrapping_add(11);
    }

    let mut acc = [0u8; 32];
    // `black_box` on the bound so the two `N` builds cannot compile the loop
    // body differently (no unrolling, no specialization). The measured
    // difference is then exactly `(N2 - N1)` executions of one identical body.
    let n = core::hint::black_box(N);
    for i in 0..n {
        // Vary the message every iteration so no hash can be hoisted or CSE'd.
        buf.0[OFF] = i as u8;
        buf.0[OFF + 1] = (i >> 8) as u8;
        let digest = hash_it(&buf.0[OFF..OFF + LEN]);
        for k in 0..32 {
            acc[k] ^= digest[k];
        }
    }

    lambda_vm_syscalls::syscalls::commit(&acc);
    lambda_vm_syscalls::syscalls::sys_halt();
}
