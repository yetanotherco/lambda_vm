//! Tests for the ECSM (elliptic-curve scalar multiplication) syscall.

use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{ECSM_SYSCALL_NUMBER, ExecutionError};
use crate::vm::memory::{Memory, MemoryError};
use crate::vm::registers::Registers;

/// secp256k1 generator x-coordinate, little-endian.
fn gx_le() -> [u8; 32] {
    let mut be = [
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
        0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8,
        0x17, 0x98,
    ];
    be.reverse();
    be
}

fn write_u256_le(memory: &mut Memory, addr: u64, bytes: &[u8; 32]) {
    for i in 0..4 {
        let mut dw = [0u8; 8];
        dw.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
        memory
            .store_doubleword(addr + (i as u64) * 8, u64::from_le_bytes(dw))
            .unwrap();
    }
}

fn read_u256_le(memory: &Memory, addr: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..4 {
        let dw = memory.load_doubleword(addr + (i as u64) * 8).unwrap();
        out[i * 8..i * 8 + 8].copy_from_slice(&dw.to_le_bytes());
    }
    out
}

/// Runs the ECSM syscall with the given scalar (as little-endian bytes) and `xG`,
/// returning the `xR` written back to memory.
fn run_ecsm(k_le: &[u8; 32], xg_le: &[u8; 32]) -> Result<[u8; 32], ExecutionError> {
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();

    let addr_xr = 0x1000u64;
    let addr_xg = 0x2000u64;
    let addr_k = 0x3000u64;
    write_u256_le(&mut memory, addr_xg, xg_le);
    write_u256_le(&mut memory, addr_k, k_le);

    registers.write(17, ECSM_SYSCALL_NUMBER).unwrap();
    registers.write(10, addr_xr).unwrap();
    registers.write(11, addr_xg).unwrap();
    registers.write(12, addr_k).unwrap();

    Instruction::EcallEbreak.run(&mut pc, &mut registers, &mut memory)?;
    Ok(read_u256_le(&memory, addr_xr))
}

fn k_le(v: u64) -> [u8; 32] {
    let mut k = [0u8; 32];
    k[..8].copy_from_slice(&v.to_le_bytes());
    k
}

#[test]
fn ecsm_syscall_writes_correct_result() {
    let xg = gx_le();
    // 1·G = G
    assert_eq!(run_ecsm(&k_le(1), &xg).unwrap(), xg);
    // Matches the reference scalar multiplication for several scalars.
    for v in [2u64, 3, 5, 0xFFFF, 1_000_003] {
        assert_eq!(
            run_ecsm(&k_le(v), &xg).unwrap(),
            ecsm::scalar_mul_x(&k_le(v), &xg).unwrap(),
            "k = {v}"
        );
    }
}

#[test]
fn ecsm_syscall_rejects_zero_scalar() {
    let err = run_ecsm(&k_le(0), &gx_le()).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::Ecsm(ecsm::EcsmError::ScalarIsZero)
    ));
}

#[test]
fn ecsm_syscall_rejects_out_of_range_scalar() {
    let err = run_ecsm(&ecsm::N_BYTES, &gx_le()).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::Ecsm(ecsm::EcsmError::ScalarOutOfRange)
    ));
}

#[test]
fn ecsm_syscall_rejects_non_canonical_xg() {
    // xG = p + 1 (the alias of x = 1) must error, not silently reduce: with
    // k = 1 the executor would echo the non-canonical bytes back as xR, which
    // the prover's xR < p range check cannot prove.
    let mut xg = ecsm::P_BYTES;
    xg[0] += 1; // p ends in 0x2F little-endian, so no carry
    let err = run_ecsm(&k_le(1), &xg).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::Ecsm(ecsm::EcsmError::CoordinateOutOfRange)
    ));
}

#[test]
fn ecsm_syscall_rejects_xg_not_on_curve() {
    // p - 1 is canonical, but not a valid secp256k1 x-coordinate.
    let mut xg = ecsm::P_BYTES;
    xg[0] -= 1;
    let err = run_ecsm(&k_le(1), &xg).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::Ecsm(ecsm::EcsmError::NotOnCurve)
    ));
}

/// Runs the ECSM syscall with caller-chosen operand addresses, `xG = Gx` and `k = 5`.
fn run_ecsm_at(addr_xr: u64, addr_xg: u64, addr_k: u64) -> Result<(), ExecutionError> {
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();
    write_u256_le(&mut memory, addr_xg, &gx_le());
    write_u256_le(&mut memory, addr_k, &k_le(5));
    registers.write(17, ECSM_SYSCALL_NUMBER).unwrap();
    registers.write(10, addr_xr).unwrap();
    registers.write(11, addr_xg).unwrap();
    registers.write(12, addr_k).unwrap();
    Instruction::EcallEbreak.run(&mut pc, &mut registers, &mut memory)?;
    Ok(())
}

#[test]
fn ecsm_syscall_rejects_overlapping_xg_k() {
    // A conservative precondition, not a provability requirement: xG is read at T and k
    // at T+1, so an overlapping cell chains through MEMW like any other pair of accesses
    // at increasing timestamps. No caller needs it — two live 32-byte objects are disjoint.
    for addr_k in [0x2000u64, 0x2008, 0x2018, 0x1FE8] {
        let err = run_ecsm_at(0x1000, 0x2000, addr_k).unwrap_err();
        assert!(
            matches!(err, ExecutionError::EcsmOperandOverlap),
            "addr_k = {addr_k:#x} overlaps addr_xg and must be rejected"
        );
    }
    // Touching-but-disjoint ranges are fine (boundary: |diff| = 32)...
    run_ecsm_at(0x1000, 0x2000, 0x2020).expect("disjoint k above xG must run");
    run_ecsm_at(0x1000, 0x2000, 0x1FE0).expect("disjoint k below xG must run");
    // ...and xR may alias xG (its accesses are offset to later timestamps).
    run_ecsm_at(0x2000, 0x2000, 0x3000).expect("xR aliasing xG is allowed");
    run_ecsm_at(0x3000, 0x2000, 0x3000).expect("xR aliasing k is allowed");
}

#[test]
fn ecsm_syscall_rejects_address_overflow() {
    // The bound is the last doubleword BASE (+24), for every operand: at 0xFFFF_FFE8 that
    // base is exactly 2^32, so the AIR emits a non-canonical low limb and cannot prove it.
    for (addr_xr, addr_xg, addr_k) in [
        (0xFFFF_FFE8, 0x2000, 0x3000),
        (0x1000, 0xFFFF_FFE8, 0x3000),
        (0x1000, 0x2000, 0xFFFF_FFE8),
        (0xFFFF_FFF0, 0x1000, 0x2000),
        (0x1000, 0xFFFF_FFF8, 0x2000),
        (0x1000, 0x2000, 0xFFFF_FFFF),
    ] {
        let err = run_ecsm_at(addr_xr, addr_xg, addr_k).unwrap_err();
        assert!(
            matches!(err, ExecutionError::EcsmAddressOverflow),
            "expected address overflow for xR={addr_xr:#x}, xG={addr_xg:#x}, k={addr_k:#x}"
        );
    }
}

#[test]
fn ecsm_syscall_accepts_operands_crossing_the_limb() {
    // The seven low limbs 0xFFFF_FFE1..=0xFFFF_FFE7 keep every doubleword base inside the
    // limb while the last doubleword's trailing bytes cross 2^32, which MEMW's carry columns
    // handle. The AIR proves these (see the prover-side
    // `test_prove_ecsm_operand_crossing_limb_boundary`), so the executor must not reject
    // them: a +31 bound here would accept a different set than the circuit.
    for lo32 in 0xFFFF_FFE1u64..=0xFFFF_FFE7 {
        run_ecsm_at(lo32, 0x1000, 0x2000)
            .unwrap_or_else(|e| panic!("xR at {lo32:#x} must run, got {e:?}"));
        run_ecsm_at(0x1000, lo32, 0x2000)
            .unwrap_or_else(|e| panic!("xG at {lo32:#x} must run, got {e:?}"));
        run_ecsm_at(0x1000, 0x2000, lo32)
            .unwrap_or_else(|e| panic!("k at {lo32:#x} must run, got {e:?}"));
    }
    // The largest low limb with no crossing at all is still fine.
    run_ecsm_at(0xFFFF_FFE0, 0x1000, 0x2000).expect("xR at 0xFFFF_FFE0 must run");
}

#[test]
fn ecsm_syscall_leaves_the_top_of_the_address_space_to_the_memory_path() {
    // With high limb 0xFFFF_FFFF a carry out of the low limb would need hi + 1 = 2^32, which
    // no page can hold, so the AIR cannot prove it. `ecsm_addr_ok` does not look at the high
    // limb; the general memory path rejects these because the byte address passes u64::MAX.
    // This test pins that reliance: if `Memory::store_doubleword` ever stopped checking,
    // the ECSM guard would have to grow a high-limb condition of its own.
    for lo32 in [0xFFFF_FFE1u64, 0xFFFF_FFE7] {
        let addr = 0xFFFF_FFFF_0000_0000 | lo32;
        let err = run_ecsm_at(addr, 0x1000, 0x2000).unwrap_err();
        assert!(
            matches!(
                err,
                ExecutionError::MemoryError(MemoryError::AddressOverflow)
            ),
            "xR at {addr:#x} must be rejected by the memory path, got {err:?}"
        );
    }
    // An 8-aligned base can never carry, so the same high limb is fine there.
    run_ecsm_at(0xFFFF_FFFF_FFFF_FFE0, 0x1000, 0x2000)
        .expect("aligned operand at the top of the address space must run");
}
