//! Inlining-immune markers for guest-side step profiling.
//!
//! Each marker emits `addi x0, x0, N` on the RISC-V guest: a real instruction
//! (so it survives inlining and optimization, unlike a removed symbol) that
//! writes to the zero register and is otherwise a no-op. Real generated code
//! never emits `addi x0, x0, N` for any nonzero `N` spontaneously (`x0` is
//! hardwired to zero and writes to it are always discarded), so these values
//! can't collide with organic instructions. Do not reuse this immediate
//! encoding space for anything other than step markers.
//!
//! Kept separate from the `instruments` feature: `instruments` uses
//! `std::time::Instant::now()`, which panics on the guest target.

pub const STEP_DECODE_DONE: u32 = 1;
pub const STEP_AIRS_AND_BUS_BALANCE_DONE: u32 = 2;
pub const STEP_REPLAY_ROUNDS_AFTER_ROUND_1: u32 = 3;
pub const STEP_VERIFY_CLAIMED_COMPOSITION_POLYNOMIAL: u32 = 4;
pub const STEP_VERIFY_FRI: u32 = 5;
pub const STEP_VERIFY_TRACE_AND_COMPOSITION_OPENINGS: u32 = 6;

#[inline(always)]
pub fn step_marker<const N: u32>() {
    #[cfg(all(feature = "profile-markers", target_arch = "riscv64"))]
    unsafe {
        core::arch::asm!("addi x0, x0, {n}", n = const N);
    }
}
