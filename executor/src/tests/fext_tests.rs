//! Tests for the FEXT (extension-field) accelerator syscalls: FEXT_LOAD and
//! FEXT_FMA over the native degree-3 Goldilocks extension `Fp[x]/(x^3 - 2)`.

use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{
    ExecutionError, FEXT_FMA_SYSCALL_NUMBER, FEXT_LOAD_SYSCALL_NUMBER,
};
use crate::vm::memory::Memory;
use crate::vm::registers::Registers;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::{GOLDILOCKS_PRIME, GoldilocksElement};

type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

/// Independent reference for `a*b + c` over Fp3, built directly from the `math`
/// crate (cross-checks the executor's own computation).
fn reference_fma(a: [u64; 3], b: [u64; 3], c: [u64; 3]) -> [u64; 3] {
    let to_fp3 = |x: [u64; 3]| {
        Fp3::from_raw([
            GoldilocksElement::from(x[0]),
            GoldilocksElement::from(x[1]),
            GoldilocksElement::from(x[2]),
        ])
    };
    let r = to_fp3(a) * to_fp3(b) + to_fp3(c);
    let v = r.value();
    [
        v[0].canonical_u64(),
        v[1].canonical_u64(),
        v[2].canonical_u64(),
    ]
}

fn run_load(memory: &mut Memory, addr: u64, coeffs: [u64; 3]) -> Result<(), ExecutionError> {
    let mut pc = 0;
    let mut registers = Registers::default();
    registers.write(17, FEXT_LOAD_SYSCALL_NUMBER).unwrap();
    registers.write(10, addr).unwrap();
    registers.write(11, coeffs[0]).unwrap();
    registers.write(12, coeffs[1]).unwrap();
    registers.write(13, coeffs[2]).unwrap();
    Instruction::EcallEbreak.run(&mut pc, &mut registers, memory)?;
    Ok(())
}

fn run_fma(memory: &mut Memory, out: u64, a: u64, b: u64, c: u64) {
    let mut pc = 0;
    let mut registers = Registers::default();
    registers.write(17, FEXT_FMA_SYSCALL_NUMBER).unwrap();
    registers.write(10, a).unwrap();
    registers.write(11, b).unwrap();
    registers.write(12, c).unwrap();
    registers.write(13, out).unwrap();
    Instruction::EcallEbreak
        .run(&mut pc, &mut registers, memory)
        .unwrap();
}

fn run_fma_result(out: u64, a: u64, b: u64, c: u64) -> Result<(), ExecutionError> {
    let mut pc = 0;
    let mut memory = Memory::default();
    let mut registers = Registers::default();
    registers.write(17, FEXT_FMA_SYSCALL_NUMBER).unwrap();
    registers.write(10, a).unwrap();
    registers.write(11, b).unwrap();
    registers.write(12, c).unwrap();
    registers.write(13, out).unwrap();
    Instruction::EcallEbreak.run(&mut pc, &mut registers, &mut memory)?;
    Ok(())
}

#[test]
fn fext_fma_rejects_overlapping_addresses() {
    // The single-timestamp design requires out/a/b/c pairwise distinct.
    for (out, a, b, c) in [
        (0x10, 0x10, 0x20, 0x30), // out == a
        (0x40, 0x20, 0x20, 0x30), // a == b (squaring)
        (0x40, 0x10, 0x20, 0x40), // out == c
        (0x40, 0x10, 0x30, 0x30), // b == c
    ] {
        let err = run_fma_result(out, a, b, c).unwrap_err();
        assert!(
            matches!(err, ExecutionError::FextOperandOverlap),
            "out={out:#x} a={a:#x} b={b:#x} c={c:#x} must be rejected"
        );
    }
    // Pairwise-distinct addresses run fine.
    run_fma_result(0x40, 0x10, 0x20, 0x30).expect("distinct addresses must run");
}

#[test]
fn fext_load_then_fma_matches_reference() {
    let mut memory = Memory::default();
    let (a_addr, b_addr, c_addr, out_addr) = (0x10, 0x20, 0x30, 0x40);

    let cases = [
        ([1, 0, 0], [1, 0, 0], [0, 0, 0]),                    // 1 * 1 + 0
        ([0, 1, 0], [0, 1, 0], [0, 0, 0]),                    // w * w = w^2
        ([0, 0, 1], [0, 0, 1], [0, 0, 0]),                    // w^2 * w^2 = w^4 = 2w
        ([1, 2, 3], [4, 5, 6], [7, 8, 9]),                    // generic
        ([GOLDILOCKS_PRIME - 1, 0, 0], [2, 0, 0], [1, 0, 0]), // wrap-around
    ];

    for (a, b, c) in cases {
        run_load(&mut memory, a_addr, a).unwrap();
        run_load(&mut memory, b_addr, b).unwrap();
        run_load(&mut memory, c_addr, c).unwrap();
        run_fma(&mut memory, out_addr, a_addr, b_addr, c_addr);
        assert_eq!(
            memory.field_load(out_addr),
            reference_fma(a, b, c),
            "a={a:?} b={b:?} c={c:?}"
        );
    }
}

#[test]
fn fext_load_stores_all_three_coefficients() {
    let mut memory = Memory::default();
    run_load(&mut memory, 0x100, [11, 22, 33]).unwrap();
    assert_eq!(memory.field_load(0x100), [11, 22, 33]);
}

#[test]
fn fext_fma_reads_uninitialized_storage_as_zero() {
    // Never-loaded field-storage addresses read as the extension-field zero, so
    // 0 * 0 + 0 = 0.
    let mut memory = Memory::default();
    run_fma(&mut memory, 0x40, 0x10, 0x20, 0x30);
    assert_eq!(memory.field_load(0x40), [0, 0, 0]);
}

#[test]
fn fext_fma_c_only_when_a_is_zero() {
    // a = 0 ⇒ out = c.
    let mut memory = Memory::default();
    run_load(&mut memory, 0x20, [9, 9, 9]).unwrap(); // b (irrelevant)
    run_load(&mut memory, 0x30, [4, 5, 6]).unwrap(); // c
    run_fma(&mut memory, 0x40, 0x10, 0x20, 0x30); // a untouched = 0
    assert_eq!(memory.field_load(0x40), [4, 5, 6]);
}

#[test]
fn fext_load_rejects_non_canonical_coefficient() {
    let mut memory = Memory::default();
    // p itself and p+1 are non-canonical (>= p) and must be rejected.
    for bad in [GOLDILOCKS_PRIME, GOLDILOCKS_PRIME + 1, u64::MAX] {
        let err = run_load(&mut memory, 0x10, [1, bad, 2]).unwrap_err();
        assert!(
            matches!(err, ExecutionError::FextCoefficientNotCanonical(v) if v == bad),
            "coefficient {bad:#x} must be rejected"
        );
    }
}
