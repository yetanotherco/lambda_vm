//! Minimal P0 guest for the Hint prover table: one `hint` ecall (field inverse of
//! a small value) + commit the result. No in-guest verify — this exercises exactly
//! the Hint table's bus surface (Ecall receive, the register read binding `out_addr`
//! to `a2`, four 8-byte MEMW writes and the output range checks; the input read is
//! deliberately not modelled) so we can get prove→verify to balance before scaling
//! to ethrex.
//!
//! Buffers are 8-byte aligned so the writes land in the aligned MEMW table, which is
//! a preference rather than a requirement — `classify_memw` routes unaligned accesses
//! to the general MEMW table, and the ethrex call site is in fact unaligned.

use lambda_vm_syscalls as syscalls;

#[repr(align(8))]
struct Aligned32([u8; 32]);

pub fn main() {
    // input = 3 (big-endian), a valid invertible field element.
    let mut x = Aligned32([0u8; 32]);
    x.0[31] = 3;
    let mut inv = Aligned32([0u8; 32]);

    syscalls::syscalls::hint(
        syscalls::syscalls::HINT_FIELD_INV,
        &mut inv.0,
        &x.0,
    );

    syscalls::syscalls::commit(&inv.0);
}
