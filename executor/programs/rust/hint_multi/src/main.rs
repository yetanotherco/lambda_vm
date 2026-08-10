//! Multi-hint P0/P2 guest for the Hint prover table: THREE `hint` ecalls, one per
//! selector, each result read back with ordinary `LOAD`s (XOR-accumulated) and the
//! accumulator committed.
//!
//! Complements `hint_min` (one hint, read back via `commit`): this exercises the
//! parts the ethrex consumer relies on that a single-call guest does not —
//! **multiple real HINT rows** (padded to a power of two), **all three selectors**
//! (`HINT_FIELD_INV` / `HINT_SCALAR_INV` / `HINT_FIELD_SQRT`, so the AIR's
//! `selector < 3` range-check is exercised at every accepted value rather than only
//! at 0) and **read-back of the hinted output via normal `LOAD` instructions**
//! (whose MEMW reads must chain to the HINT table's writes). Buffers are 8-byte
//! aligned so the writes land in the aligned MEMW table.

use lambda_vm_syscalls as syscalls;

#[repr(align(8))]
struct Aligned32([u8; 32]);

pub fn main() {
    let mut acc = Aligned32([0u8; 32]);

    // One call per selector. 4 is a quadratic residue mod p, so the sqrt hint has a
    // real root rather than the zeros `compute_hint` returns on a numeric failure.
    for (hint_id, seed) in [
        (syscalls::syscalls::HINT_FIELD_INV, 3u8),
        (syscalls::syscalls::HINT_SCALAR_INV, 5u8),
        (syscalls::syscalls::HINT_FIELD_SQRT, 4u8),
    ] {
        let mut x = Aligned32([0u8; 32]);
        x.0[31] = seed;
        let mut out = Aligned32([0u8; 32]);

        syscalls::syscalls::hint(hint_id, &mut out.0, &x.0);

        // Read the hinted output back via ordinary loads and fold it in, so the
        // MEMW reads of `out` must chain to the HINT table's writes.
        for i in 0..32 {
            acc.0[i] ^= out.0[i];
        }
    }

    syscalls::syscalls::commit(&acc.0);
}
