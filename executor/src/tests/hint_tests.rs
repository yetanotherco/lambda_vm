//! Tests for the non-constraining `Hint` syscall.

use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{
    ExecutionError, HINT_FIELD_INV, HINT_FIELD_SQRT, HINT_SCALAR_INV, HINT_SYSCALL_NUMBER,
    compute_hint,
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

/// The scalar-field inverse hint (mod n) round-trips through guest memory and
/// satisfies `x·inv == 1 (mod n)` — the check the guest performs on the untrusted
/// value. Used by production ecrecover (`r⁻¹`).
#[test]
fn hint_syscall_writes_the_scalar_inverse() {
    use k256::elliptic_curve::PrimeField;

    let mut input = [0u8; 32];
    input[31] = 3; // 3, big-endian

    let out = run_hint_at(HINT_SCALAR_INV, 0x1000, 0x2000, &input).expect("hint must run");
    assert_eq!(out, compute_hint(HINT_SCALAR_INV, &input));

    let three: k256::Scalar = Option::from(k256::Scalar::from_repr(input.into())).unwrap();
    let inv: k256::Scalar = Option::from(k256::Scalar::from_repr(out.into())).unwrap();
    assert_eq!(
        (three * inv).to_bytes(),
        k256::Scalar::ONE.to_bytes(),
        "hinted scalar inverse must satisfy x·inv == 1 (mod n)"
    );
}

/// The base-field sqrt hint (mod p) round-trips and satisfies `y² == rhs (mod p)`.
/// Used by production ecrecover (decompressing R). `4 = 2²` is a residue.
#[test]
fn hint_syscall_writes_the_field_sqrt() {
    let mut input = [0u8; 32];
    input[31] = 4; // rhs = 4, big-endian

    let out = run_hint_at(HINT_FIELD_SQRT, 0x1000, 0x2000, &input).expect("hint must run");
    assert_eq!(out, compute_hint(HINT_FIELD_SQRT, &input));

    let rhs: k256::FieldElement =
        Option::from(k256::FieldElement::from_bytes(&input.into())).unwrap();
    let y: k256::FieldElement = Option::from(k256::FieldElement::from_bytes(&out.into())).unwrap();
    assert_eq!(
        y.square().to_bytes(),
        rhs.to_bytes(),
        "hinted sqrt must satisfy y² == rhs (mod p)"
    );
}

/// An unknown `hint_id` is rejected up front. Silently writing zeros would be
/// indistinguishable from a legitimate numeric failure and — because the guest reads
/// the value back — could let a prover-chosen selector steer a caller's accept/reject
/// outcome. The executor traps so a guest bug surfaces loudly. `HINT_FIELD_SQRT = 2`
/// is the last known selector, so 3 is the first unknown one.
#[test]
fn hint_syscall_rejects_an_unknown_selector() {
    let mut input = [0u8; 32];
    input[31] = 3;
    for bad in [3u64, 100, u64::MAX] {
        let err = run_hint_at(bad, 0x1000, 0x2000, &input).expect_err("unknown selector must trap");
        assert!(
            matches!(err, ExecutionError::HintUnknownSelector(id) if id == bad),
            "expected HintUnknownSelector({bad}), got {err:?}"
        );
    }
}

/// The guest's `lambda-vm-syscalls` crate re-declares the selectors as `usize`,
/// linked to the `u64` copies here only by a comment. A divergence is **silent**:
/// the ecall would trap on an unknown selector, or — worse for the selectors that
/// stay in range — hand back the wrong function's answer, which the guest's
/// verify-then-fallback swallows as "the host lied" and quietly recomputes in
/// software. Nothing fails; the guest just runs ~2000× slower for the right result.
/// This test is the only thing that would notice.
///
/// `is_valid_hint_selector`'s const-assert pins the AIR's range-check to this crate's
/// accepted set, but nothing ties the *guest's* copy of the selectors to it — that is
/// a third declaration, in a crate the workspace excludes, and this is what binds it.
///
/// The syscall number itself is not asserted here: the guest's copy is
/// `#[cfg(target_arch = "riscv64")]` and private, so it does not exist in a host
/// build. It is covered indirectly — a wrong number makes every `hint` guest fail
/// to prove, which `test_prove_hint_min_rust_guest` catches loudly.
#[cfg(test)]
mod guest_constant_sync {
    use super::{HINT_FIELD_INV, HINT_FIELD_SQRT, HINT_SCALAR_INV};
    use lambda_vm_syscalls::syscalls as guest;

    #[test]
    fn hint_selectors_match_the_guest() {
        assert_eq!(guest::HINT_FIELD_INV as u64, HINT_FIELD_INV);
        assert_eq!(guest::HINT_SCALAR_INV as u64, HINT_SCALAR_INV);
        assert_eq!(guest::HINT_FIELD_SQRT as u64, HINT_FIELD_SQRT);
    }
}
