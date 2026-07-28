//! Minimal P0 guest for the Hint prover table: one `hint` ecall (field inverse of
//! a small value) + commit the result. No in-guest verify — this exercises exactly
//! the Hint table's bus surface (Ecall receive + one 32-byte MEMW read + one 32-byte
//! MEMW write) so we can get prove→verify to balance before scaling to ethrex.
//!
//! Buffers are 8-byte aligned so the MEMW accesses land in the aligned MEMW table.

use lambda_vm_syscalls as syscalls;

#[repr(align(8))]
struct Aligned32([u8; 32]);

pub fn main() {
    // input = 3 (little-endian), a valid invertible field element.
    let mut x = Aligned32([0u8; 32]);
    x.0[31] = 3; // big-endian hint input (ABI is BE)
    let mut inv = Aligned32([0u8; 32]);

    syscalls::syscalls::hint(
        syscalls::syscalls::HINT_FIELD_INV,
        &mut inv.0,
        &x.0,
    );

    syscalls::syscalls::commit(&inv.0);
}
