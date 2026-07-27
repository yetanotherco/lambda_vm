//! Multi-hint P0/P2 guest for the Hint prover table: THREE `hint` ecalls
//! (field inverse of three different small values), each result read back with
//! ordinary `LOAD`s (XOR-accumulated) and the accumulator committed.
//!
//! Complements `hint_min` (one hint, read back via `commit`): this exercises the
//! parts the ethrex consumer relies on that a single-call guest does not —
//! **multiple real HINT rows** (padded to a power of two) and **read-back of the
//! hinted output via normal `LOAD` instructions** (whose MEMW reads must chain to
//! the HINT table's writes). Buffers are 8-byte aligned so the writes land in the
//! aligned MEMW table.

use lambda_vm_syscalls as syscalls;

#[repr(align(8))]
struct Aligned32([u8; 32]);

pub fn main() {
    let mut acc = Aligned32([0u8; 32]);

    for seed in [3u8, 5u8, 7u8] {
        let mut x = Aligned32([0u8; 32]);
        x.0[31] = seed;
        let mut inv = Aligned32([0u8; 32]);

        syscalls::syscalls::hint(syscalls::syscalls::HINT_FIELD_INV, &mut inv.0, &x.0);

        // Read the hinted output back via ordinary loads and fold it in, so the
        // MEMW reads of `inv` must chain to the HINT table's writes.
        for i in 0..32 {
            acc.0[i] ^= inv.0[i];
        }
    }

    syscalls::syscalls::commit(&acc.0);
}
