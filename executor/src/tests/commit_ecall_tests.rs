//! The `write(2)` ecall's register ABI, which the COPY chip proves.
//!
//! `copy:c:read_fd` pins the old value of `x10` to the file descriptor `1` and
//! returns `count` there, so the executor has to agree on both halves or the
//! register token the chip emits has no counterpart and an honest program becomes
//! unprovable. Neither half had a test before.

use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{ExecutionError, SyscallNumbers};
use crate::vm::memory::Memory;
use crate::vm::registers::Registers;

/// Drive one `write` ecall and hand back the register file it left behind.
fn run_commit(
    memory: &mut Memory,
    fd: u64,
    buf_addr: u64,
    count: u64,
) -> Result<Registers, ExecutionError> {
    let mut registers = Registers::default();
    let mut pc = 0;
    registers.write(17, SyscallNumbers::Commit as u64)?;
    registers.write(10, fd)?;
    registers.write(11, buf_addr)?;
    registers.write(12, count)?;
    Instruction::EcallEbreak.run(&mut pc, &mut registers, memory)?;
    Ok(registers)
}

#[test]
fn commit_returns_the_byte_count_in_a0() {
    // The AIR writes `count` back to x10 on every `write`. If the executor leaves
    // `fd` there instead, the two register mirrors diverge at this ecall and the
    // next instruction that *reads* a0 without writing it first emits a token the
    // COPY chip never produced -- a bus imbalance on an honest program.
    let mut memory = Memory::default();
    memory.store_byte(0x100, b'a');
    memory.store_byte(0x101, b'b');

    let registers = run_commit(&mut memory, 1, 0x100, 2).expect("commit should succeed");
    assert_eq!(
        registers.read(10).expect("a0 readable"),
        2,
        "a0 must hold the number of bytes written, not the file descriptor"
    );
    // The operand registers are untouched.
    assert_eq!(registers.read(11).expect("a1"), 0x100);
    assert_eq!(registers.read(12).expect("a2"), 2);
    assert_eq!(memory.read_return_value().expect("output"), b"ab".to_vec());
}

#[test]
fn commit_returns_zero_for_an_empty_slice() {
    // `syscalls::commit` passes `slice.len()` straight through, so `commit(b"")` is
    // reachable from any guest. It is the one shape that produces the degenerate
    // COPY row (`count = 0`, `first = end = 1`, no memory access at all).
    let mut memory = Memory::default();
    let registers = run_commit(&mut memory, 1, 0x100, 0).expect("empty commit should succeed");
    assert_eq!(registers.read(10).expect("a0"), 0);
    assert!(
        memory.read_return_value().expect("output").is_empty(),
        "an empty commit must contribute no output"
    );
}

#[test]
fn commit_rejects_every_file_descriptor_but_stdout() {
    // Only stdout is supported, and the AIR says so by pinning the old value of x10
    // to the constant 1 in `copy:c:read_fd`. The executor must accept exactly that
    // set, or it admits an execution no prover can build.
    for fd in [0u64, 2, 3, u64::MAX] {
        let mut memory = Memory::default();
        assert!(
            matches!(
                run_commit(&mut memory, fd, 0x100, 2),
                Err(ExecutionError::InvalidCommitFd(got)) if got == fd
            ),
            "fd {fd} must be rejected"
        );
    }
}
