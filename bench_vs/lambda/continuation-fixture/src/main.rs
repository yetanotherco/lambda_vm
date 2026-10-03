//! ★ The continuation fixture's inner guest: it COMMITS, and then keeps going.
//!
//! `bench_vs/lambda/fibonacci` cannot be this guest, for two separate reasons.
//!
//! The first is what the fixture needs. `epoch_tests::EpochFront::build` asserts
//! `!is_final` — it deliberately harvests an INTERMEDIATE epoch — and a guest
//! that commits immediately before `halt()` puts its public output in the FINAL
//! one, always. So `e.public_output` is empty for every epoch that harness can
//! reach, and `the_closure_rejects_a_moved_index_or_output`'s second half (move
//! a half of the public output, watch the COMMIT bus refuse to close) has no
//! subject. Its anti-vacuity guard says so rather than passing quietly. This
//! guest commits and then does bounded work, so the committing epoch is an
//! intermediate one by CONSTRUCTION rather than by arithmetic luck.
//!
//! The second is that `fibonacci` is not ours to change: `bench_vs/run.sh` builds
//! it as the Lambda VM arm of a cross-prover benchmark against `bench_vs/sp1/
//! fibonacci`, and a tail loop that one arm runs and the other does not stops the
//! two measuring the same program.
//!
//! ⚠ **The cycle count is a property of the compiled ELF**, as
//! `proof_fixture::FIXTURE_EPOCH_LOG2` has recorded since the toolchain last
//! moved it. What keeps that from biting here is MARGIN, not luck — see
//! [`TAIL_STEPS`]. Re-measure rather than guess: run this ELF to completion under
//! `Executor::resume_with_limit` and count the logs, one per cycle.

#![no_std]
#![no_main]

use core::arch::asm;
use core::panic::PanicInfo;

const PRIVATE_INPUT_START: usize = 0xFF000000;
const SYSCALL_COMMIT: u64 = 64;
const SYSCALL_HALT: u64 = 93;

/// ★ Iterations of post-commit work, and the number the fixture's whole shape
/// rests on.
///
/// Measured on the ELF this source builds, by counting logs: the commit lands at
/// cycle **13** and the run is **48** cycles. One step costs TWO cycles — 10
/// steps gave 36 and 16 gave 48 — so this is a dial, not a guess.
/// `FIXTURE_EPOCH_LOG2 = 5` puts the epoch boundary at 32, which leaves
///
/// - **19 cycles** between the commit and the boundary — epoch 0 CARRIES the
///   public output, which is what the whole guest is for;
/// - **16 cycles** between the boundary and the halt — epoch 0 is INTERMEDIATE,
///   so `EpochFront::build`'s `!is_final` holds;
/// - **16 cycles** between the halt and 64 — there are exactly TWO epochs, which
///   is what `continuation_fixture_generates_two_epochs` is the canary for.
///
/// Three margins of sixteen cycles or more, against the two-instruction codegen
/// drift that crossed a boundary the last time this was a four-cycle question. A
/// tail sized to land just past the boundary would have reproduced that
/// fragility rather than fixed it, which is why this is not the smallest number
/// that works — 10 steps also "works", with four cycles to spare, and that is
/// the state this guest exists to get out of. Raising it eats the third margin
/// and runs out at 24 steps (total 64); lowering it eats the second.
///
/// ★ `proof_fixture::tests::the_fixture_guest_commits_in_an_intermediate_epoch`
/// asserts all three, so none of this has to be believed on the strength of a
/// comment.
const TAIL_STEPS: u64 = 16;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

fn read_n() -> u64 {
    // Layout matches `syscalls::get_private_input`: 4-byte LE length prefix at
    // PRIVATE_INPUT_START, payload at +4. We only need the first 8 bytes (u64).
    let mut n_bytes = [0u8; 8];

    debug_assert!(
        unsafe { core::ptr::read_volatile(PRIVATE_INPUT_START as *const u32) } >= 8,
        "private input too short to contain a u64"
    );

    let input_data = (PRIVATE_INPUT_START + 4) as *const u8;
    n_bytes.copy_from_slice(unsafe { core::slice::from_raw_parts(input_data, 8) });

    u64::from_le_bytes(n_bytes)
}

fn commit(bytes: &[u8]) {
    unsafe {
        asm!(
            "ecall",
            in("a0") 1u64,
            in("a1") bytes.as_ptr(),
            in("a2") bytes.len(),
            in("a7") SYSCALL_COMMIT,
        );
    }
}

fn halt() -> ! {
    unsafe {
        asm!(
            "ecall",
            in("a0") 0u64,
            in("a7") SYSCALL_HALT,
            options(noreturn),
        );
    }
}

#[unsafe(no_mangle)]
pub fn main() -> ! {
    let n = read_n();
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..n {
        let c = a.wrapping_add(b);
        a = b;
        b = c;
    }

    commit(&b.to_le_bytes());

    // ★ The tail. Two constraints shape it, and neither is obvious.
    //
    // A VOLATILE store is what makes it survive: the value is never read back,
    // so anything weaker is dead code the optimiser is entitled to delete, and
    // deleting it would silently restore the very shape this guest exists to
    // avoid. The sink is a stack local rather than a `static mut` because the
    // stack is mapped under every image builder the harnesses use.
    //
    // ⚠ And the arithmetic is MORE FIBONACCI rather than anything with a
    // constant in it, because `recursion::precomputed_commitments` lists every
    // ELF page whose `init_values.is_some()` — a pure function of the BINARY,
    // not of the run. A 64-bit LCG multiplier is not materialisable inline on
    // rv64, so LLVM spills it to a constant pool, the guest grows a `.rodata`
    // section fibonacci does not have, and the fixture starts carrying a page
    // commitment. `program_id_matches_production_on_the_real_fixture` asserts
    // it carries none. Wrapping adds of values already in registers keep the
    // section table identical to `fibonacci`'s: `.text` and nothing else.
    let mut sink: u64 = 0;
    let mut step: u64 = 0;
    while step < TAIL_STEPS {
        let c = a.wrapping_add(b);
        a = b;
        b = c;
        unsafe { core::ptr::write_volatile(&raw mut sink, b) };
        step += 1;
    }

    halt()
}
