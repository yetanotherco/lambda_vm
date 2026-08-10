//! Tests for the ECSM (elliptic-curve scalar multiplication) syscall.

use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{
    ECSM_AFFINE_SYSCALL_NUMBER, ECSM_SYSCALL_NUMBER, ExecutionError,
};
use crate::vm::memory::Memory;
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

/// secp256k1 generator y-coordinate, little-endian.
fn gy_le() -> [u8; 32] {
    let mut be = [
        0x48, 0x3A, 0xDA, 0x77, 0x26, 0xA3, 0xC4, 0x65, 0x5D, 0xA4, 0xFB, 0xFC, 0x0E, 0x11, 0x08,
        0xA8, 0xFD, 0x17, 0xB4, 0x48, 0xA6, 0x85, 0x54, 0x19, 0x9C, 0x47, 0xD0, 0x8F, 0xFB, 0x10,
        0xD4, 0xB8,
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

/// Runs the AFFINE ECSM syscall (full point in/out) with the given scalar, `xG` and `yG`,
/// returning the `(xR, yR)` written back to the contiguous 64-byte output buffer.
fn run_ecsm_affine(
    k_le: &[u8; 32],
    xg_le: &[u8; 32],
    yg_le: &[u8; 32],
) -> Result<([u8; 32], [u8; 32]), ExecutionError> {
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();

    let addr_xr = 0x1000u64; // output xR‖yR (64 bytes)
    let addr_xg = 0x2000u64; // input xG‖yG (64 bytes)
    let addr_k = 0x3000u64;
    write_u256_le(&mut memory, addr_xg, xg_le);
    write_u256_le(&mut memory, addr_xg + 32, yg_le);
    write_u256_le(&mut memory, addr_k, k_le);

    registers.write(17, ECSM_AFFINE_SYSCALL_NUMBER).unwrap();
    registers.write(10, addr_xr).unwrap();
    registers.write(11, addr_xg).unwrap();
    registers.write(12, addr_k).unwrap();

    Instruction::EcallEbreak.run(&mut pc, &mut registers, &mut memory)?;
    Ok((
        read_u256_le(&memory, addr_xr),
        read_u256_le(&memory, addr_xr + 32),
    ))
}

#[test]
fn ecsm_affine_syscall_writes_both_coords() {
    let (xg, yg) = (gx_le(), gy_le());
    for v in [1u64, 2, 3, 5, 0xFFFF, 1_000_003] {
        let (xr, yr) = run_ecsm_affine(&k_le(v), &xg, &yg).unwrap();
        let (exr, eyr) = ecsm::scalar_mul_xy_with_y(&k_le(v), &xg, &yg).unwrap();
        assert_eq!(xr, exr, "xR mismatch for k = {v}");
        assert_eq!(yr, eyr, "yR mismatch for k = {v}");
    }
}

#[test]
fn ecsm_affine_syscall_rejects_point_not_on_curve() {
    // A valid xG paired with the wrong yG (here yG = Gy of a different point) must be
    // rejected on-curve, unlike the x-only variant which lifts its own canonical y.
    let mut yg = gy_le();
    yg[0] ^= 1; // perturb so (xG, yG) is no longer on the curve
    let err = run_ecsm_affine(&k_le(5), &gx_le(), &yg).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::Ecsm(ecsm::EcsmError::NotOnCurve)
    ));
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

#[test]
fn ecsm_affine_syscall_rejects_non_canonical_yg() {
    // yG = p is a non-canonical zero. The prover reads yG straight out of the caller's
    // buffer, so nothing downstream would reduce it — the executor must reject it.
    let err = run_ecsm_affine(&k_le(5), &gx_le(), &ecsm::P_BYTES).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::Ecsm(ecsm::EcsmError::CoordinateOutOfRange)
    ));
}

#[test]
fn ecsm_affine_syscall_rejects_zero_scalar() {
    let err = run_ecsm_affine(&k_le(0), &gx_le(), &gy_le()).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::Ecsm(ecsm::EcsmError::ScalarIsZero)
    ));
}

/// Runs the affine ECSM syscall with caller-chosen operand addresses, input point `G`
/// and `k = 5`.
fn run_ecsm_affine_at(addr_xr: u64, addr_xg: u64, addr_k: u64) -> Result<(), ExecutionError> {
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();
    write_u256_le(&mut memory, addr_xg, &gx_le());
    write_u256_le(&mut memory, addr_xg.wrapping_add(32), &gy_le());
    write_u256_le(&mut memory, addr_k, &k_le(5));
    registers.write(17, ECSM_AFFINE_SYSCALL_NUMBER).unwrap();
    registers.write(10, addr_xr).unwrap();
    registers.write(11, addr_xg).unwrap();
    registers.write(12, addr_k).unwrap();
    Instruction::EcallEbreak.run(&mut pc, &mut registers, &mut memory)?;
    Ok(())
}

#[test]
fn ecsm_affine_syscall_rejects_overlapping_point_k() {
    // The input point spans 64 bytes, so any k landing inside [xG, xG + 64) is read at
    // both T and T+1 and makes the trace unprovable.
    for addr_k in [0x2000u64, 0x2008, 0x2020, 0x2038] {
        let err = run_ecsm_affine_at(0x1000, 0x2000, addr_k).unwrap_err();
        assert!(
            matches!(err, ExecutionError::EcsmOperandOverlap),
            "addr_k = {addr_k:#x} overlaps the 64-byte input point and must be rejected"
        );
    }
    // Disjoint is disjoint regardless of distance: k directly below the point is 32 bytes
    // away, which a `|diff| >= 64` bound would have rejected. Guest stack layouts produce
    // exactly this case, so it must run.
    run_ecsm_affine_at(0x1000, 0x2000, 0x1FE0).expect("k immediately below the point must run");
    run_ecsm_affine_at(0x1000, 0x2000, 0x2040).expect("k immediately above the point must run");
    // The output may alias either operand: its accesses are at later timestamps.
    run_ecsm_affine_at(0x2000, 0x2000, 0x3000).expect("xR aliasing the input point is allowed");
    run_ecsm_affine_at(0x3000, 0x2000, 0x3000).expect("xR aliasing k is allowed");
}

/// The overlap guard must not be defeated by 64-bit wraparound.
///
/// `addr_limb_ok` bounds only the LOW limb, so `u64::MAX - 63` passes it while
/// `addr_xg + 64` wraps to 0. With a plain `u64` `wrapping_add` the first clause becomes
/// `addr_k < 0` — always false — and the whole check is skipped, accepting exactly the
/// overlap it exists to reject. Reachable: `STACK_TOP` is `0xFFFF_FFFF_FFFF_FFF0`, so
/// these addresses sit inside `main`'s first frame.
#[test]
fn ecsm_affine_overlap_guard_survives_address_wraparound() {
    // 0xFFFF_FFFF_FFFF_FFC0 — the largest address passing addr_limb_ok(_, 63), and the one
    // for which addr_xg + 64 wraps to exactly 0.
    let top_point = u64::MAX - 63;
    // addr_k = top_point exercises the first clause (addr_xg + 64 wraps); top_point + 32
    // additionally makes addr_k + 32 wrap, exercising the second. Both are genuine overlaps
    // (k inside the point's 64 bytes) that the pre-u128 guard accepted.
    for addr_k in [top_point, top_point + 32] {
        let err = run_ecsm_affine_at(0x1000, top_point, addr_k).unwrap_err();
        assert!(
            matches!(err, ExecutionError::EcsmOperandOverlap),
            "addr_k = {addr_k:#x} overlaps the point at {top_point:#x} and must be rejected \
             even though the guard's + 64 / + 32 wrap"
        );
    }
}

#[test]
fn ecsm_affine_syscall_rejects_address_overflow() {
    // Point and output span offset 63 (not 31), so their last accessed byte must stay in
    // the limb: 0xFFFF_FFE8 fits a 32-byte operand but not a 64-byte one.
    for (addr_xr, addr_xg, addr_k) in [
        (0xFFFF_FFE8, 0x2000, 0x3000),
        (0x1000, 0xFFFF_FFE8, 0x3000),
        (0xFFFF_FFC8, 0x2000, 0x3000),
        (0x1000, 0xFFFF_FFC8, 0x3000),
        (0x1000, 0x2000, 0xFFFF_FFF0),
    ] {
        let err = run_ecsm_affine_at(addr_xr, addr_xg, addr_k).unwrap_err();
        assert!(
            matches!(err, ExecutionError::EcsmAddressOverflow),
            "expected address overflow for xR={addr_xr:#x}, point={addr_xg:#x}, k={addr_k:#x}"
        );
    }
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
    // xG and k are read at the same proof timestamp, so overlapping ranges
    // would make the trace unprovable — the executor must reject them upfront.
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
    // Every operand's last accessed byte must stay in the limb (+31); the 0xFFFF_FFE1
    // cases are the off-by-7 window the old +24 bound for xR/xG let through.
    for (addr_xr, addr_xg, addr_k) in [
        (0xFFFF_FFE8, 0x2000, 0x3000),
        (0x1000, 0xFFFF_FFE8, 0x3000),
        (0x1000, 0x2000, 0xFFFF_FFF0),
        (0xFFFF_FFE1, 0x1000, 0x2000),
        (0x1000, 0xFFFF_FFE1, 0x2000),
    ] {
        let err = run_ecsm_at(addr_xr, addr_xg, addr_k).unwrap_err();
        assert!(
            matches!(err, ExecutionError::EcsmAddressOverflow),
            "expected address overflow for xR={addr_xr:#x}, xG={addr_xg:#x}, k={addr_k:#x}"
        );
    }
}
