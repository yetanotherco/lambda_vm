//! Tests for the ECSM `lincomb2` syscall — `Q = u1·P1 + u2·P2` on secp256k1.
//!
//! Two properties are load-bearing and get separate coverage:
//!
//!  1. **Correctness.** The `Q` the syscall writes equals both `lincomb2_witness`'s `Q`
//!     (the chip spec) and `k256`'s `ProjectivePoint::lincomb` (the software fallback the
//!     guest uses whenever the accelerator declines), so accelerating a call can never
//!     change the recovered public key.
//!  2. **The status contract.** Degenerate *values* return a non-zero status and leave the
//!     result buffer untouched — they never trap, because they come from transaction data
//!     and a trap would let one crafted transaction abort a whole block's proof. Degenerate
//!     *addresses* are guest-program bugs and stay hard errors.

use k256::elliptic_curve::ff::PrimeField as _;
use k256::elliptic_curve::ops::LinearCombination as _;
use k256::elliptic_curve::sec1::ToEncodedPoint as _;
use k256::{ProjectivePoint, Scalar};
use num_bigint::BigUint;

use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{
    ECSM_LINCOMB2_SYSCALL_NUMBER, ExecutionError, LINCOMB2_STATUS_OK,
    LINCOMB2_STATUS_P1_NOT_GENERATOR, LINCOMB2_STATUS_POINT_NOT_CANONICAL,
    LINCOMB2_STATUS_POINT_NOT_ON_CURVE, LINCOMB2_STATUS_RESULT_INFINITY,
    LINCOMB2_STATUS_SCALAR_IS_ZERO, LINCOMB2_STATUS_SCALAR_OUT_OF_RANGE,
    LINCOMB2_STATUS_SUM_DEGENERATE,
};
use crate::vm::memory::Memory;
use crate::vm::registers::Registers;

// ---------------------------------------------------------------------------
// Operand helpers. Every operand is 64 bytes: two 32-byte little-endian values.
// ---------------------------------------------------------------------------

const ADDR_Q: u64 = 0x1000;
const ADDR_P1: u64 = 0x2000;
const ADDR_P2: u64 = 0x3000;
const ADDR_U: u64 = 0x4000;

/// Byte pattern pre-written to the result region so "the buffer is untouched" is a
/// positive assertion rather than "it still reads as zero".
const SENTINEL: u8 = 0xA5;

fn write_bytes(memory: &mut Memory, addr: u64, bytes: &[u8]) {
    for (i, b) in bytes.iter().enumerate() {
        memory.store_byte(addr + i as u64, *b);
    }
}

fn read_64(memory: &Memory, addr: u64) -> [u8; 64] {
    let mut out = [0u8; 64];
    for (i, b) in out.iter_mut().enumerate() {
        *b = memory.load_byte(addr + i as u64);
    }
    out
}

/// Joins two 32-byte little-endian values into one 64-byte operand.
fn operand(lo: &[u8; 32], hi: &[u8; 32]) -> [u8; 64] {
    let mut out = [0u8; 64];
    out[..32].copy_from_slice(lo);
    out[32..].copy_from_slice(hi);
    out
}

/// `x‖y` of a k256 point, little-endian, in the syscall's operand layout.
fn point_le(p: &ProjectivePoint) -> [u8; 64] {
    let affine = p.to_affine();
    let encoded = affine.to_encoded_point(false);
    let mut out = [0u8; 64];
    for (i, b) in encoded.x().unwrap().iter().rev().enumerate() {
        out[i] = *b;
    }
    for (i, b) in encoded.y().unwrap().iter().rev().enumerate() {
        out[32 + i] = *b;
    }
    out
}

fn scalar_le(v: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&v.to_le_bytes());
    out
}

fn k256_scalar(le: &[u8; 32]) -> Scalar {
    let mut be = *le;
    be.reverse();
    Scalar::from_repr(be.into()).unwrap()
}

/// `k256`'s `ProjectivePoint::lincomb` — the guest's software fallback — as the
/// syscall's 64-byte result layout.
fn k256_lincomb(p1: &[u8; 64], p2: &[u8; 64], u1: &[u8; 32], u2: &[u8; 32]) -> [u8; 64] {
    point_le(&ProjectivePoint::lincomb(
        &le_to_k256(p1),
        &k256_scalar(u1),
        &le_to_k256(p2),
        &k256_scalar(u2),
    ))
}

fn le_to_k256(p: &[u8; 64]) -> ProjectivePoint {
    use k256::elliptic_curve::sec1::FromEncodedPoint as _;
    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    x.copy_from_slice(&p[..32]);
    y.copy_from_slice(&p[32..]);
    x.reverse();
    y.reverse();
    let encoded = k256::EncodedPoint::from_affine_coordinates(&x.into(), &y.into(), false);
    ProjectivePoint::from(k256::AffinePoint::from_encoded_point(&encoded).unwrap())
}

/// `lincomb2_witness`'s `Q` (the chip spec) in the syscall's result layout.
fn witness_q(p1: &[u8; 64], p2: &[u8; 64], u1: &[u8; 32], u2: &[u8; 32]) -> [u8; 64] {
    let point = |b: &[u8; 64]| ecsm::AffinePoint {
        x: BigUint::from_bytes_le(&b[..32]),
        y: BigUint::from_bytes_le(&b[32..]),
    };
    let w = ecsm::witness::lincomb2_witness(u1, u2, &point(p1), &point(p2)).expect("witness");
    operand(&w.x_q, &w.y_q)
}

/// One syscall invocation at caller-chosen addresses. Returns the status word left in
/// `a0` and the 64 bytes at the result address.
fn run_lincomb2_at(
    addrs: [u64; 4],
    p1: &[u8; 64],
    p2: &[u8; 64],
    u: &[u8; 64],
) -> Result<(u64, [u8; 64]), ExecutionError> {
    let [addr_q, addr_p1, addr_p2, addr_u] = addrs;
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();

    // The inputs are written before the result sentinel so that an overlapping result
    // region visibly clobbers them rather than the other way round.
    write_bytes(&mut memory, addr_p1, p1);
    write_bytes(&mut memory, addr_p2, p2);
    write_bytes(&mut memory, addr_u, u);
    write_bytes(&mut memory, addr_q, &[SENTINEL; 64]);

    registers.write(17, ECSM_LINCOMB2_SYSCALL_NUMBER).unwrap();
    registers.write(10, addr_q).unwrap();
    registers.write(11, addr_p1).unwrap();
    registers.write(12, addr_p2).unwrap();
    registers.write(13, addr_u).unwrap();

    Instruction::EcallEbreak.run(&mut pc, &mut registers, &mut memory)?;

    // The status is returned in a0; the three input pointers must survive the call.
    let status = registers.read(10).unwrap();
    assert_eq!(registers.read(11).unwrap(), addr_p1, "a1 clobbered");
    assert_eq!(registers.read(12).unwrap(), addr_p2, "a2 clobbered");
    assert_eq!(registers.read(13).unwrap(), addr_u, "a3 clobbered");

    // Both paths must leave the inputs exactly as they were: the chip re-reads these
    // bytes at the ecall's timestamp and proves what it consumed.
    assert_eq!(&read_64(&memory, addr_p1), p1, "P1 operand modified");
    assert_eq!(&read_64(&memory, addr_p2), p2, "P2 operand modified");
    assert_eq!(&read_64(&memory, addr_u), u, "scalar operand modified");

    Ok((status, read_64(&memory, addr_q)))
}

/// One syscall invocation at the standard disjoint addresses.
fn run_lincomb2(p1: &[u8; 64], p2: &[u8; 64], u: &[u8; 64]) -> (u64, [u8; 64]) {
    run_lincomb2_at([ADDR_Q, ADDR_P1, ADDR_P2, ADDR_U], p1, p2, u)
        .expect("valid addresses must not error")
}

fn g() -> [u8; 64] {
    point_le(&ProjectivePoint::GENERATOR)
}

/// `m·G` as an operand — a convenient supply of distinct valid points.
fn mul_g(m: u64) -> [u8; 64] {
    point_le(&(ProjectivePoint::GENERATOR * k256_scalar(&scalar_le(m))))
}

/// Asserts a rejected call left the result buffer exactly as the sentinel found it.
fn assert_untouched(out: &[u8; 64]) {
    assert_eq!(
        out, &[SENTINEL; 64],
        "a non-zero status must leave the result buffer untouched"
    );
}

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

/// One happy-path input tuple: `(P2, u1, u2)`. `P1` is always `G` — the syscall accepts
/// no other `P1` (see `lincomb2_syscall_rejects_non_generator_p1`).
type Case = ([u8; 64], [u8; 32], [u8; 32]);

#[test]
fn lincomb2_syscall_matches_witness_and_k256() {
    // P2 = m·G for a few m, over a spread of scalar sizes (including the 256-bit range
    // where the joint schedule runs its full 256 doublings).
    let mut n_minus_3 = ecsm::N_BYTES;
    n_minus_3[0] -= 3; // N ends in 0x41 little-endian, so no borrow

    let cases: [Case; 6] = [
        (mul_g(2), scalar_le(1), scalar_le(1)),
        (mul_g(2), scalar_le(3), scalar_le(5)),
        (mul_g(7), scalar_le(0xFFFF), scalar_le(1_000_003)),
        (mul_g(11), scalar_le(u64::MAX), scalar_le(2)),
        (mul_g(2), n_minus_3, scalar_le(1)),
        (mul_g(9), n_minus_3, n_minus_3),
    ];

    let p1 = g();
    for (i, (p2, u1, u2)) in cases.iter().enumerate() {
        let (status, out) = run_lincomb2(&p1, p2, &operand(u1, u2));
        assert_eq!(status, LINCOMB2_STATUS_OK, "case {i}");
        assert_eq!(out, witness_q(&p1, p2, u1, u2), "case {i}: Q != witness Q");
        assert_eq!(out, k256_lincomb(&p1, p2, u1, u2), "case {i}: Q != k256 Q");
    }
}

#[test]
fn lincomb2_syscall_writes_exactly_the_result_region() {
    // The write must be the 64 bytes at a0 and nothing else: a wider store would
    // silently corrupt whatever the guest put next to its buffer.
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();

    let (p1, p2) = (g(), mul_g(2));
    let u = operand(&scalar_le(3), &scalar_le(5));
    write_bytes(&mut memory, ADDR_P1, &p1);
    write_bytes(&mut memory, ADDR_P2, &p2);
    write_bytes(&mut memory, ADDR_U, &u);
    write_bytes(&mut memory, ADDR_Q - 8, &[SENTINEL; 80]); // 8 before .. 8 after

    registers.write(17, ECSM_LINCOMB2_SYSCALL_NUMBER).unwrap();
    registers.write(10, ADDR_Q).unwrap();
    registers.write(11, ADDR_P1).unwrap();
    registers.write(12, ADDR_P2).unwrap();
    registers.write(13, ADDR_U).unwrap();
    Instruction::EcallEbreak
        .run(&mut pc, &mut registers, &mut memory)
        .unwrap();

    assert_eq!(registers.read(10).unwrap(), LINCOMB2_STATUS_OK);
    for offset in 0..8u64 {
        assert_eq!(
            memory.load_byte(ADDR_Q - 8 + offset),
            SENTINEL,
            "wrote below"
        );
        assert_eq!(
            memory.load_byte(ADDR_Q + 64 + offset),
            SENTINEL,
            "wrote above"
        );
    }
}

// ---------------------------------------------------------------------------
// Status contract: one test per `Lincomb2Error` variant
// ---------------------------------------------------------------------------

#[test]
fn lincomb2_syscall_reports_zero_scalar() {
    let (p1, p2) = (g(), mul_g(2));
    for u in [
        operand(&scalar_le(0), &scalar_le(5)),
        operand(&scalar_le(5), &scalar_le(0)),
    ] {
        let (status, out) = run_lincomb2(&p1, &p2, &u);
        assert_eq!(status, LINCOMB2_STATUS_SCALAR_IS_ZERO);
        assert_untouched(&out);
    }
}

#[test]
fn lincomb2_syscall_reports_out_of_range_scalar() {
    let (p1, p2) = (g(), mul_g(2));
    for u in [
        operand(&ecsm::N_BYTES, &scalar_le(5)),
        operand(&scalar_le(5), &ecsm::N_BYTES),
    ] {
        let (status, out) = run_lincomb2(&p1, &p2, &u);
        assert_eq!(status, LINCOMB2_STATUS_SCALAR_OUT_OF_RANGE);
        assert_untouched(&out);
    }
}

#[test]
fn lincomb2_syscall_reports_point_not_on_curve() {
    // Only P2 can reach this: P1 is pinned to G, so a malformed P1 is caught earlier by
    // the generator check.
    // y + 1 is still canonical (G's y is nowhere near p) but off the curve.
    let mut bad = g();
    bad[32] += 1;
    let u = operand(&scalar_le(3), &scalar_le(5));
    let (status, out) = run_lincomb2(&g(), &bad, &u);
    assert_eq!(status, LINCOMB2_STATUS_POINT_NOT_ON_CURVE);
    assert_untouched(&out);
}

#[test]
fn lincomb2_syscall_reports_non_canonical_point() {
    // y = p is the non-canonical alias of y = 0. Dropping this check is a real forgery:
    // yP2 + p is the same point mod p but has the OPPOSITE parity as bytes, which flips
    // the sign of Q. The executor must never hand such a point to the chip.
    let mut bad_y = g();
    bad_y[32..].copy_from_slice(&ecsm::P_BYTES);
    let mut bad_x = g();
    bad_x[..32].copy_from_slice(&ecsm::P_BYTES);
    let u = operand(&scalar_le(3), &scalar_le(5));
    for p2 in [bad_y, bad_x] {
        let (status, out) = run_lincomb2(&g(), &p2, &u);
        assert_eq!(status, LINCOMB2_STATUS_POINT_NOT_CANONICAL);
        assert_untouched(&out);
    }
}

#[test]
fn lincomb2_syscall_rejects_non_generator_p1() {
    // ECSM′ binds a1's bytes to G by construction — it has no `mem_p1` membership
    // witness — so a P1 the chip cannot represent must NOT come back as status 0. If it
    // did, the trace builder would emit a row asserting bytes that are not in memory, the
    // constraint would fail, and the block would be unprovable. Returning a status keeps
    // executor and chip in agreement and degrades such a caller to software.
    let u = operand(&scalar_le(3), &scalar_le(5));

    // Perfectly valid curve points that simply are not G.
    for p1 in [mul_g(2), mul_g(3), mul_g(0xDEAD_BEEF)] {
        let (status, out) = run_lincomb2(&p1, &mul_g(7), &u);
        assert_eq!(status, LINCOMB2_STATUS_P1_NOT_GENERATOR);
        assert_untouched(&out);
    }

    // −G: same x as G, opposite y. The check is on all 64 bytes, not just x, because the
    // chip binds both coordinates.
    let mut neg_g = g();
    neg_g[32..].copy_from_slice(&ecsm::to_le_32(
        &(ecsm::p() - BigUint::from_bytes_le(&g()[32..])),
    ));
    let (status, out) = run_lincomb2(&neg_g, &mul_g(7), &u);
    assert_eq!(status, LINCOMB2_STATUS_P1_NOT_GENERATOR);
    assert_untouched(&out);

    // A single flipped byte in either coordinate is caught too.
    for i in [0usize, 31, 32, 63] {
        let mut p1 = g();
        p1[i] ^= 1;
        let (status, out) = run_lincomb2(&p1, &mul_g(7), &u);
        assert_eq!(status, LINCOMB2_STATUS_P1_NOT_GENERATOR, "byte {i}");
        assert_untouched(&out);
    }

    // The real G is accepted, so the check is not vacuous.
    let (status, _) = run_lincomb2(&g(), &mul_g(7), &u);
    assert_eq!(status, LINCOMB2_STATUS_OK);
}

#[test]
fn generator_le_is_the_secp256k1_generator() {
    // The pinned constant must be the curve's actual generator; k256 re-derives it.
    use crate::vm::instruction::execution::GENERATOR_LE;
    assert_eq!(GENERATOR_LE, g());
}

#[test]
fn lincomb2_syscall_reports_degenerate_sum() {
    // P1 = ±P2 makes the P1 + P2 precompute a doubling (or infinity), so the addend
    // table the joint chain indexes does not exist.
    let g = g();
    let mut neg_g = g;
    neg_g[32..].copy_from_slice(&ecsm::to_le_32(
        &(ecsm::p() - BigUint::from_bytes_le(&g[32..])),
    ));
    let u = operand(&scalar_le(3), &scalar_le(5));
    for p2 in [g, neg_g] {
        let (status, out) = run_lincomb2(&g, &p2, &u);
        assert_eq!(status, LINCOMB2_STATUS_SUM_DEGENERATE);
        assert_untouched(&out);
    }
}

#[test]
fn lincomb2_syscall_reports_infinite_result() {
    // (N-2)·G + 1·(2G) = N·G = ∞, which has no affine witness.
    let mut u1 = ecsm::N_BYTES;
    u1[0] -= 2; // N ends in 0x41 little-endian, so no borrow
    let (status, out) = run_lincomb2(&g(), &mul_g(2), &operand(&u1, &scalar_le(1)));
    assert_eq!(status, LINCOMB2_STATUS_RESULT_INFINITY);
    assert_untouched(&out);
}

#[test]
fn lincomb2_syscall_always_writes_the_status_register() {
    // The error path must be expressible as a proof row: the chip gives every lincomb2
    // ecall one row that receives the Ecall bus and performs the same fixed MEMW
    // accesses, so `a0` must be overwritten with the status whether or not a witness
    // exists. If a future refactor early-returns before `registers.write(10, ...)`, `a0`
    // would still hold the result address and the ecall would have no receiver.
    let (p1, p2) = (g(), mul_g(2));
    let good = operand(&scalar_le(3), &scalar_le(5));
    let bad = operand(&scalar_le(0), &scalar_le(5));

    for (u, expected) in [
        (good, LINCOMB2_STATUS_OK),
        (bad, LINCOMB2_STATUS_SCALAR_IS_ZERO),
    ] {
        let mut pc = 0;
        let mut registers = Registers::default();
        let mut memory = Memory::default();
        write_bytes(&mut memory, ADDR_P1, &p1);
        write_bytes(&mut memory, ADDR_P2, &p2);
        write_bytes(&mut memory, ADDR_U, &u);
        registers.write(17, ECSM_LINCOMB2_SYSCALL_NUMBER).unwrap();
        registers.write(10, ADDR_Q).unwrap();
        registers.write(11, ADDR_P1).unwrap();
        registers.write(12, ADDR_P2).unwrap();
        registers.write(13, ADDR_U).unwrap();
        Instruction::EcallEbreak
            .run(&mut pc, &mut registers, &mut memory)
            .unwrap();

        let a0 = registers.read(10).unwrap();
        assert_eq!(a0, expected);
        assert_ne!(
            a0, ADDR_Q,
            "a0 must be overwritten, not left as the address"
        );
    }
}

#[test]
fn lincomb2_status_codes_are_distinct() {
    // The guest only tests `!= 0`, but the codes must stay distinguishable for
    // debugging — a collision would silently merge two failure modes.
    let codes = [
        LINCOMB2_STATUS_OK,
        LINCOMB2_STATUS_SCALAR_IS_ZERO,
        LINCOMB2_STATUS_SCALAR_OUT_OF_RANGE,
        LINCOMB2_STATUS_POINT_NOT_ON_CURVE,
        LINCOMB2_STATUS_POINT_NOT_CANONICAL,
        LINCOMB2_STATUS_SUM_DEGENERATE,
        LINCOMB2_STATUS_RESULT_INFINITY,
        LINCOMB2_STATUS_P1_NOT_GENERATOR,
    ];
    let mut sorted = codes.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), codes.len(), "status codes collide");
    assert_eq!(LINCOMB2_STATUS_OK, 0, "only success may be zero");
}

// ---------------------------------------------------------------------------
// Address guards
// ---------------------------------------------------------------------------

/// Runs the syscall with valid operands at caller-chosen addresses.
fn run_lincomb2_addrs(addrs: [u64; 4]) -> Result<(u64, [u8; 64]), ExecutionError> {
    run_lincomb2_at(
        addrs,
        &g(),
        &mul_g(2),
        &operand(&scalar_le(3), &scalar_le(5)),
    )
}

#[test]
fn lincomb2_syscall_rejects_unaligned_addresses() {
    // Each region is read/written as eight ALIGNED doubleword MEMW accesses.
    for slot in 0..4 {
        for misalignment in [1u64, 2, 4, 7] {
            let mut addrs = [ADDR_Q, ADDR_P1, ADDR_P2, ADDR_U];
            addrs[slot] += misalignment;
            let err = run_lincomb2_addrs(addrs).unwrap_err();
            assert!(
                matches!(err, ExecutionError::Lincomb2UnalignedAddress(a) if a == addrs[slot]),
                "slot {slot} at +{misalignment} must be rejected, got {err:?}"
            );
        }
    }
}

#[test]
fn lincomb2_syscall_rejects_address_overflow() {
    // The last byte of a 64-byte operand sits at +63; crossing 2^32 splits the operand
    // across the MEMW address limbs.
    for slot in 0..4 {
        let mut addrs = [ADDR_Q, ADDR_P1, ADDR_P2, ADDR_U];
        addrs[slot] = 0xFFFF_FFC8; // 8-aligned, but +63 lands at 2^32 + 7
        let err = run_lincomb2_addrs(addrs).unwrap_err();
        assert!(
            matches!(err, ExecutionError::Lincomb2AddressOverflow(a) if a == addrs[slot]),
            "slot {slot} must overflow, got {err:?}"
        );

        // The exact boundary is fine: 0xFFFF_FFC0 + 63 = 0xFFFF_FFFF.
        let mut addrs = [ADDR_Q, ADDR_P1, ADDR_P2, ADDR_U];
        addrs[slot] = 0xFFFF_FFC0;
        run_lincomb2_addrs(addrs)
            .unwrap_or_else(|e| panic!("slot {slot} at the limb boundary must run, got {e:?}"));
    }
}

#[test]
fn lincomb2_syscall_rejects_overlapping_operands() {
    // All six pairs, including the result against each input: unlike `ecsm_mul`'s xR,
    // the result region may not alias an input, since it is written after the reads.
    let base = [ADDR_Q, ADDR_P1, ADDR_P2, ADDR_U];
    for i in 0..4 {
        for j in 0..4 {
            if i == j {
                continue;
            }
            for delta in [0i64, 8, 56, -8, -56] {
                let mut addrs = base;
                addrs[j] = (base[i] as i64 + delta) as u64;
                let err = run_lincomb2_addrs(addrs)
                    .err()
                    .unwrap_or_else(|| panic!("slots {i}/{j} at delta {delta} must be rejected"));
                assert!(
                    matches!(err, ExecutionError::Lincomb2OperandOverlap(_, _)),
                    "slots {i}/{j} at delta {delta}: expected overlap, got {err:?}"
                );
            }
            // Touching but disjoint (|diff| = 64) is allowed.
            for delta in [64i64, -64] {
                let mut addrs = base;
                addrs[j] = (base[i] as i64 + delta) as u64;
                run_lincomb2_addrs(addrs).unwrap_or_else(|e| {
                    panic!("slots {i}/{j} at delta {delta} are disjoint and must run, got {e:?}")
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Guest → executor round trip
// ---------------------------------------------------------------------------

#[test]
fn lincomb2_syscall_round_trips_the_ecrecover_shape() {
    // The shape the guest actually calls: pk = u1·G + u2·R, with R a decompressed
    // signature point and u1/u2 the ECDSA recovery scalars. Both are full-width
    // 256-bit scalars, so this exercises the longest joint schedule.
    let u1 = {
        let mut b = ecsm::N_BYTES;
        b[0] -= 0x37;
        b[31] -= 0x11;
        b
    };
    let u2 = {
        let mut b = [0u8; 32];
        for (i, x) in b.iter_mut().enumerate() {
            *x = (i as u8).wrapping_mul(37).wrapping_add(3);
        }
        b[31] &= 0x7F; // keep it below N
        b
    };
    let r = mul_g(0xDEAD_BEEF);

    let (status, out) = run_lincomb2(&g(), &r, &operand(&u1, &u2));
    assert_eq!(status, LINCOMB2_STATUS_OK);
    assert_eq!(out, k256_lincomb(&g(), &r, &u1, &u2));

    // A guest that trusts the status word gets exactly what the fallback would produce.
    let pk = if status == LINCOMB2_STATUS_OK {
        out
    } else {
        k256_lincomb(&g(), &r, &u1, &u2)
    };
    assert_eq!(pk, witness_q(&g(), &r, &u1, &u2));
}

/// The full guest → executor path: a real ELF, decoded and executed, whose `main`
/// materializes the operands on its stack, issues `ecall` with `a7 = -12`, and commits
/// `status‖xQ‖yQ`. This is what the register-level tests above cannot cover — that the
/// ABI's register assignment survives instruction decode and that the guest observes the
/// status in `a0`.
///
/// Reads a prebuilt artifact, like every other guest-ELF test here; run
/// `make compile-programs-asm` first.
#[test]
fn lincomb2_asm_guest_round_trip() {
    use crate::elf::Elf;
    use crate::vm::execution::Executor;

    let path = "./program_artifacts/asm/test_ecsm_lincomb2.elf";
    let elf_data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let program = Elf::load(&elf_data).unwrap();
    let result = Executor::new(&program, vec![]).unwrap().run().unwrap();

    // The guest commits status(8, little-endian) ‖ xQ(32) ‖ yQ(32).
    let committed = result.return_values.memory_values;
    assert_eq!(committed.len(), 72, "guest must commit 72 bytes");
    let status = u64::from_le_bytes(committed[..8].try_into().unwrap());
    assert_eq!(status, LINCOMB2_STATUS_OK, "guest saw a non-zero status");

    let mut q = [0u8; 64];
    q.copy_from_slice(&committed[8..]);

    // The guest computes 3·G + 5·(2G) = 13·G.
    let (p1, p2) = (g(), mul_g(2));
    let (u1, u2) = (scalar_le(3), scalar_le(5));
    assert_eq!(q, witness_q(&p1, &p2, &u1, &u2));
    assert_eq!(q, k256_lincomb(&p1, &p2, &u1, &u2));
    assert_eq!(q, mul_g(13), "3G + 5·(2G) must be 13G");

    // The program halts with exit code 0.
    assert_eq!(result.return_values.register_values.0, 0);
}

#[test]
fn lincomb2_syscall_number_is_free() {
    use crate::vm::instruction::execution::{
        ECSM_SYSCALL_NUMBER, KECCAK_SYSCALL_NUMBER, SyscallNumbers,
    };

    // -12, one below ECSM's -11, and distinct from every other syscall this VM decodes.
    assert_eq!(ECSM_LINCOMB2_SYSCALL_NUMBER, u64::MAX - 11);
    for taken in [
        KECCAK_SYSCALL_NUMBER,
        ECSM_SYSCALL_NUMBER,
        SyscallNumbers::Print as u64,
        SyscallNumbers::Panic as u64,
        SyscallNumbers::Commit as u64,
        SyscallNumbers::Halt as u64,
    ] {
        assert_ne!(ECSM_LINCOMB2_SYSCALL_NUMBER, taken);
    }
    assert!(matches!(
        SyscallNumbers::try_from(ECSM_LINCOMB2_SYSCALL_NUMBER),
        Ok(SyscallNumbers::EcsmLincomb2)
    ));
}
