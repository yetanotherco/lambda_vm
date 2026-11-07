//! Tests for the ECREC (ECDSA public-key recovery) syscall.

use k256::elliptic_curve::PrimeField;
use k256::elliptic_curve::ops::Reduce;
use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::{ProjectivePoint, Scalar, U256};

use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{ECREC_SYSCALL_NUMBER, ExecutionError};
use crate::vm::memory::Memory;
use crate::vm::registers::Registers;

fn store(memory: &mut Memory, addr: u64, bytes: &[u8]) {
    for (i, chunk) in bytes.chunks_exact(8).enumerate() {
        let dw = u64::from_le_bytes(chunk.try_into().unwrap());
        memory.store_doubleword(addr + 8 * i as u64, dw).unwrap();
    }
}

fn load(memory: &Memory, addr: u64, len: usize) -> Vec<u8> {
    (0..len / 8)
        .flat_map(|i| {
            memory
                .load_doubleword(addr + 8 * i as u64)
                .unwrap()
                .to_le_bytes()
        })
        .collect()
}

/// Runs the call with `sig` at `addr_sig`, `msg` at `addr_msg`, the output at
/// `addr_out` and `parity` in x13; returns the 72 bytes written.
fn run_at(
    sig: &[u8; 64],
    msg: &[u8; 32],
    parity: u64,
    addr_out: u64,
    addr_sig: u64,
    addr_msg: u64,
) -> Result<Vec<u8>, ExecutionError> {
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();
    store(&mut memory, addr_sig, sig);
    store(&mut memory, addr_msg, msg);
    registers.write(17, ECREC_SYSCALL_NUMBER).unwrap();
    registers.write(10, addr_out).unwrap();
    registers.write(11, addr_sig).unwrap();
    registers.write(12, addr_msg).unwrap();
    registers.write(13, parity).unwrap();
    Instruction::EcallEbreak.run(&mut pc, &mut registers, &mut memory)?;
    Ok(load(&memory, addr_out, 72))
}

fn run(sig: &[u8; 64], msg: &[u8; 32], parity: u64) -> Result<Vec<u8>, ExecutionError> {
    run_at(sig, msg, parity, 0x1000, 0x2000, 0x3000)
}

/// A k256 signature by `d` with nonce `k` on hash `msg`, its R parity and the key.
fn sign(d: u64, k: u64, msg: &[u8; 32]) -> ([u8; 64], u64, Vec<u8>) {
    let (d, k) = (
        Scalar::from(d) * Scalar::from(d),
        Scalar::from(k) * Scalar::from(k),
    );
    let r_pt = (ProjectivePoint::GENERATOR * k)
        .to_affine()
        .to_encoded_point(false);
    let r = Option::<Scalar>::from(Scalar::from_repr(*r_pt.x().unwrap())).unwrap();
    let z = <Scalar as Reduce<U256>>::reduce_bytes(&(*msg).into());
    let s = k.invert().unwrap() * (z + r * d);
    let mut sig = [0u8; 64];
    sig[..32].copy_from_slice(&r.to_bytes());
    sig[32..].copy_from_slice(&s.to_bytes());
    let q = (ProjectivePoint::GENERATOR * d)
        .to_affine()
        .to_encoded_point(false);
    (
        sig,
        u64::from(r_pt.y().unwrap()[31] & 1),
        q.as_bytes()[1..65].to_vec(),
    )
}

#[test]
fn ecrec_syscall_recovers_the_key() {
    for (d, k) in [(3u64, 5u64), (0xDEAD_BEEF, 0x1234_5678), (u64::MAX, 77)] {
        let msg = [0xA7u8; 32];
        let (sig, parity, key) = sign(d, k, &msg);
        let out = run(&sig, &msg, parity).unwrap();
        assert_eq!(&out[..64], &key[..], "d = {d}");
        assert_eq!(&out[64..], &[0u8; 8], "status 0");
        // The other parity is the other R: some other key, still status 0.
        let other = run(&sig, &msg, parity ^ 1).unwrap();
        assert_ne!(&other[..64], &key[..]);
        assert_eq!(other[64], 0);
    }
}

#[test]
fn ecrec_syscall_bails_on_what_it_cannot_compute() {
    let msg = [0x11u8; 32];
    let (sig, parity, _) = sign(9, 13, &msg);
    let mut n_be = ecsm::N_BYTES;
    n_be.reverse();
    let mut cases = vec![];
    for (lo, hi, value) in [
        (0, 32, [0u8; 32]),
        (0, 32, n_be),
        (32, 64, [0u8; 32]),
        (32, 64, n_be),
    ] {
        let mut s = sig;
        s[lo..hi].copy_from_slice(&value);
        cases.push((s, msg));
    }
    cases.push((sig, [0u8; 32])); // z = 0: u1 = 0
    for (sig, msg) in cases {
        let out = run(&sig, &msg, parity).unwrap();
        assert_eq!(&out[..64], &[0u8; 64]);
        assert_eq!(out[64], 1, "status 1");
        assert_eq!(&out[65..], &[0u8; 7]);
    }
}

#[test]
fn ecrec_syscall_rejects_a_parity_above_one() {
    let msg = [0x22u8; 32];
    let (sig, _, _) = sign(5, 7, &msg);
    for parity in [2u64, 3, u64::MAX] {
        assert!(matches!(
            run(&sig, &msg, parity),
            Err(ExecutionError::EcrecParityOutOfRange(p)) if p == parity
        ));
    }
}

#[test]
fn ecrec_syscall_rejects_overlapping_operands_and_address_overflow() {
    let msg = [0x33u8; 32];
    let (sig, parity, _) = sign(5, 7, &msg);
    // The hash overlapping the signature (64 bytes at 0x2000).
    for addr_msg in [0x2000u64, 0x2020, 0x2038, 0x1FE8] {
        assert!(matches!(
            run_at(&sig, &msg, parity, 0x1000, 0x2000, addr_msg),
            Err(ExecutionError::EcrecOperandOverlap)
        ));
    }
    run_at(&sig, &msg, parity, 0x1000, 0x2000, 0x2040).expect("hash right after the signature");
    run_at(&sig, &msg, parity, 0x1000, 0x2000, 0x1FE0).expect("hash right before it");
    run_at(&sig, &msg, parity, 0x2000, 0x2000, 0x3000).expect("output aliasing the signature");
    // Each operand's last byte stays in the low limb: +71 out, +63 sig, +31 msg.
    for (out, s, m) in [
        (0xFFFF_FFB9, 0x2000, 0x3000),
        (0x1000, 0xFFFF_FFC1, 0x3000),
        (0x1000, 0x2000, 0xFFFF_FFE1),
    ] {
        assert!(matches!(
            run_at(&sig, &msg, parity, out, s, m),
            Err(ExecutionError::EcrecAddressOverflow)
        ));
    }
    run_at(&sig, &msg, parity, 0xFFFF_FFB8, 0x2000, 0x3000).expect("output ending at the limb top");
}
