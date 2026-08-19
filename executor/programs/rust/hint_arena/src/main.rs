//! Hint-arena guest: reads THREE 32-byte hints from the private-input hint
//! arena (no ecall — ordinary aligned loads from the memory-mapped region),
//! XOR-accumulates them, and commits the accumulator.
//!
//! Arena counterpart to `hint_multi`: same three logical hints (one per
//! selector's worth of values), but the bytes arrive as prover-chosen
//! private-input page data instead of executor-written ecall outputs, so the
//! proof needs no HINT table rows at all — the guest's reads are ordinary
//! MEMR loads chained to the private-input pages.

use lambda_vm_syscalls as syscalls;

pub fn main() {
    let mut acc = [0u8; 32];

    assert_eq!(
        syscalls::syscalls::hint_count(),
        3,
        "the host must supply exactly three hint slots"
    );

    for _ in 0..3 {
        let hint = syscalls::syscalls::next_hint().expect("arena exhausted");
        for i in 0..32 {
            acc[i] ^= hint[i];
        }
    }

    // Positional consumption is one slot per request: a fourth request runs
    // past the end and must yield None (the caller's cue to fall back to
    // software), never a desynced stream.
    assert!(syscalls::syscalls::next_hint().is_none());

    syscalls::syscalls::commit(&acc);
}
