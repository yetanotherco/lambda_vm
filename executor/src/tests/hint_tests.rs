//! Tests for the non-constraining `Hint` syscall (BENCH ONLY).

use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{
    ExecutionError, HINT_FIELD_INV, HINT_SYSCALL_NUMBER, compute_hint,
};
use crate::vm::memory::Memory;
use crate::vm::registers::Registers;

fn write_u256(memory: &mut Memory, addr: u64, bytes: &[u8; 32]) {
    for i in 0..4 {
        let mut dw = [0u8; 8];
        dw.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
        memory
            .store_doubleword(addr + (i as u64) * 8, u64::from_le_bytes(dw))
            .unwrap();
    }
}

fn read_u256(memory: &Memory, addr: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..4 {
        let dw = memory.load_doubleword(addr + (i as u64) * 8).unwrap();
        out[i * 8..i * 8 + 8].copy_from_slice(&dw.to_le_bytes());
    }
    out
}

/// Runs one `Hint` ecall with the given operand addresses, returning the 32 bytes
/// written at `out_addr`.
fn run_hint_at(
    hint_id: u64,
    in_addr: u64,
    out_addr: u64,
    input: &[u8; 32],
) -> Result<[u8; 32], ExecutionError> {
    let mut memory = Memory::default();
    let mut registers = Registers::default();
    let mut pc = 0u64;

    write_u256(&mut memory, in_addr, input);
    registers.write(17, HINT_SYSCALL_NUMBER).unwrap();
    registers.write(10, hint_id).unwrap();
    registers.write(11, in_addr).unwrap();
    registers.write(12, out_addr).unwrap();
    Instruction::EcallEbreak.run(&mut pc, &mut registers, &mut memory)?;
    Ok(read_u256(&memory, out_addr))
}

/// The base-field inverse hint round-trips through guest memory, big-endian in and
/// out, and matches `compute_hint` (the value the prover recomputes).
#[test]
fn hint_syscall_writes_the_field_inverse() {
    let mut input = [0u8; 32];
    input[31] = 3; // 3, big-endian

    let out = run_hint_at(HINT_FIELD_INV, 0x1000, 0x2000, &input).expect("hint must run");
    assert_eq!(out, compute_hint(HINT_FIELD_INV, &input));

    // 3 · 3⁻¹ ≡ 1 (mod p) — the same check the guest performs on the untrusted value.
    let three: k256::FieldElement =
        Option::from(k256::FieldElement::from_bytes(&input.into())).unwrap();
    let inv: k256::FieldElement =
        Option::from(k256::FieldElement::from_bytes(&out.into())).unwrap();
    assert_eq!(
        (three * inv).to_bytes(),
        k256::FieldElement::ONE.to_bytes(),
        "hinted inverse must satisfy x·inv == 1"
    );
}

/// Both operands must keep their 32-byte range inside the lower address limb: the
/// HINT table sends the output writes as `[out_addr_lo + 8i, out_addr_hi]`, which
/// cannot represent a carry into the high limb, so a straddling operand would make
/// the trace unprovable. The executor rejects it upfront instead.
#[test]
fn hint_syscall_rejects_address_overflow() {
    let input = [0u8; 32];
    // Last accessed byte is at +31, so the first rejected base is 2^32 - 31.
    for (in_addr, out_addr) in [
        (0x1000, 0xFFFF_FFE8),
        (0xFFFF_FFE8, 0x2000),
        (0x1000, 0xFFFF_FFE1),
        (0xFFFF_FFE1, 0x2000),
        (0x1000, 0xFFFF_FFFF),
    ] {
        let err = run_hint_at(HINT_FIELD_INV, in_addr, out_addr, &input)
            .expect_err("straddling operand must be rejected");
        assert!(
            matches!(err, ExecutionError::HintAddressOverflow),
            "expected address overflow for in={in_addr:#x}, out={out_addr:#x}, got {err:?}"
        );
    }
}

/// The boundary case: an operand ending exactly on the last byte of the limb is
/// still representable and must be accepted.
#[test]
fn hint_syscall_accepts_operand_ending_at_the_limb_boundary() {
    let input = [0u8; 32];
    // 2^32 - 32: last byte lands at 2^32 - 1, the largest in-limb address.
    run_hint_at(HINT_FIELD_INV, 0x1000, 0xFFFF_FFE0, &input)
        .expect("operand ending at the limb boundary must run");
    run_hint_at(HINT_FIELD_INV, 0xFFFF_FFE0, 0x2000, &input)
        .expect("operand ending at the limb boundary must run");
}

/// An unknown `hint_id` is not an error — the ecall writes zeros and the guest's
/// verify is what rejects the value. Pins that contract so a future selector can't
/// silently start trapping instead.
#[test]
fn hint_syscall_writes_zeros_for_an_unknown_selector() {
    let mut input = [0u8; 32];
    input[31] = 3;
    let out =
        run_hint_at(u64::MAX, 0x1000, 0x2000, &input).expect("unknown selector must not trap");
    assert_eq!(out, [0u8; 32]);
}
