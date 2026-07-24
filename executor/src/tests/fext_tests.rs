//! Tests for the FEXT (extension-field) accelerator syscalls: FEXT_LOAD and
//! FEXT_FMA over the native degree-3 Goldilocks extension `Fp[x]/(x^3 - 2)`.

use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{
    ExecutionError, FEXT_FMA_SYSCALL_NUMBER, FEXT_LOAD_SYSCALL_NUMBER, FEXT_STORE_SYSCALL_NUMBER,
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

fn run_store(memory: &mut Memory, src_addr: u64) -> [u64; 3] {
    let mut pc = 0;
    let mut registers = Registers::default();
    registers.write(17, FEXT_STORE_SYSCALL_NUMBER).unwrap();
    registers.write(10, src_addr).unwrap();
    Instruction::EcallEbreak
        .run(&mut pc, &mut registers, memory)
        .unwrap();
    [
        registers.read(11).unwrap(),
        registers.read(12).unwrap(),
        registers.read(13).unwrap(),
    ]
}

#[test]
fn fext_store_reads_back_loaded_value() {
    let mut memory = Memory::default();
    run_load(&mut memory, 0x100, [11, 22, 33]).unwrap();
    assert_eq!(run_store(&mut memory, 0x100), [11, 22, 33]);
}

#[test]
fn fext_store_then_reload_roundtrips_fma() {
    // LOAD a,b,c → FMA → STORE result back to registers → equals reference.
    let mut memory = Memory::default();
    let (a, b, c) = ([1, 2, 3], [4, 5, 6], [7, 8, 9]);
    run_load(&mut memory, 0x10, a).unwrap();
    run_load(&mut memory, 0x20, b).unwrap();
    run_load(&mut memory, 0x30, c).unwrap();
    run_fma(&mut memory, 0x40, 0x10, 0x20, 0x30);
    assert_eq!(run_store(&mut memory, 0x40), reference_fma(a, b, c));
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

// Differential tests for the exact syscall sequences the in-guest STARK verifier
// emits through `crypto::field_ext::Fp3Fma` (`ext_mul` and the `prod_acc`
// resident accumulator). Those guest impls call `fext_load`/`fext_fma`/
// `fext_store` and are `#[cfg(target_arch = "riscv64")]`, so they cannot run on
// host; here we replay the same handle choreography against the real executor
// and compare to a math-crate oracle. Keep these in sync with the sequences in
// `crypto/crypto/src/field_ext.rs`.

// Handles mirror `field_ext.rs`; only pairwise distinctness and "H_ZERO is never
// loaded" (so it reads as the extension zero) actually matter here.
const H_A: u64 = 0;
const H_B: u64 = 1;
const H_C: u64 = 2;
const H_OUT: u64 = 3;
const H_ZERO: u64 = 4;
const H_T: u64 = 5;
const H_ACC0: u64 = 6;
const H_ACC1: u64 = 7;

/// One `a*b*c` term of a `prod_acc` chain.
type ProdTerm = ([u64; 3], [u64; 3], [u64; 3]);

/// Dependency-free deterministic SplitMix64, used to draw random canonical Fp3
/// coefficients without pulling `rand` into dev-dependencies.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Canonical (`< p`) coefficients, the only form `fext_load` accepts.
    fn coeffs(&mut self) -> [u64; 3] {
        [
            self.next_u64() % GOLDILOCKS_PRIME,
            self.next_u64() % GOLDILOCKS_PRIME,
            self.next_u64() % GOLDILOCKS_PRIME,
        ]
    }
}

/// Independent reference for `sum_i a_i * b_i * c_i` over Fp3 (the value the
/// `prod_acc` chain computes).
fn reference_prod_sum(terms: &[ProdTerm]) -> [u64; 3] {
    let to_fp3 = |x: [u64; 3]| {
        Fp3::from_raw([
            GoldilocksElement::from(x[0]),
            GoldilocksElement::from(x[1]),
            GoldilocksElement::from(x[2]),
        ])
    };
    let mut acc = Fp3::zero();
    for (a, b, c) in terms {
        acc += to_fp3(*a) * to_fp3(*b) * to_fp3(*c);
    }
    let v = acc.value();
    [
        v[0].canonical_u64(),
        v[1].canonical_u64(),
        v[2].canonical_u64(),
    ]
}

/// Replays `Fp3Fma::prod_acc_{new,add,finish}` against the real executor and
/// returns the stored accumulator coefficients.
fn run_prod_acc_chain(terms: &[ProdTerm]) -> [u64; 3] {
    let mut memory = Memory::default();
    // prod_acc_new: zero the starting buffer (it is written across chains).
    run_load(&mut memory, H_ACC0, [0, 0, 0]).unwrap();
    let mut buf = 0u8;
    for (a, b, c) in terms {
        // prod_acc_add
        run_load(&mut memory, H_A, *a).unwrap();
        run_load(&mut memory, H_B, *b).unwrap();
        run_load(&mut memory, H_C, *c).unwrap();
        run_fma(&mut memory, H_T, H_A, H_B, H_ZERO); // tmp = a * b
        let (cur, alt) = if buf == 0 {
            (H_ACC0, H_ACC1)
        } else {
            (H_ACC1, H_ACC0)
        };
        run_fma(&mut memory, alt, H_T, H_C, cur); // alt = tmp * c + cur
        buf ^= 1;
    }
    // prod_acc_finish
    let cur = if buf == 0 { H_ACC0 } else { H_ACC1 };
    memory.field_load(cur)
}

#[test]
fn fext_ext_mul_via_unwritten_zero_matches_reference() {
    // Replays `Fp3Fma::ext_mul`: fma(a, b, H_ZERO) with H_ZERO never loaded, so
    // the accumulator input reads as zero and the result is a*b.
    let mut rng = SplitMix64(0xF00D_BEEF);
    for _ in 0..100 {
        let (a, b) = (rng.coeffs(), rng.coeffs());
        let mut memory = Memory::default();
        run_load(&mut memory, H_A, a).unwrap();
        run_load(&mut memory, H_B, b).unwrap();
        run_fma(&mut memory, H_OUT, H_A, H_B, H_ZERO);
        assert_eq!(
            memory.field_load(H_OUT),
            reference_fma(a, b, [0, 0, 0]),
            "a={a:?} b={b:?}"
        );
    }
}

#[test]
fn fext_prod_acc_chain_matches_reference() {
    // Fixed chains covering the buffer-toggle boundaries: empty (finish reads the
    // freshly zeroed H_ACC0), length 1 (ends on H_ACC1), length 2 (ends back on
    // H_ACC0), length 3 (ends on H_ACC1), including a wrap-around coefficient.
    let fixed: &[&[ProdTerm]] = &[
        &[],
        &[([1, 2, 3], [4, 5, 6], [7, 8, 9])],
        &[
            ([1, 2, 3], [4, 5, 6], [7, 8, 9]),
            ([10, 0, 1], [0, 2, 0], [3, 3, 3]),
        ],
        &[
            ([1, 0, 0], [0, 1, 0], [0, 0, 1]),
            ([2, 2, 2], [1, 1, 1], [9, 0, 9]),
            ([GOLDILOCKS_PRIME - 1, 0, 0], [2, 0, 0], [1, 1, 1]),
        ],
    ];
    for terms in fixed {
        assert_eq!(
            run_prod_acc_chain(terms),
            reference_prod_sum(terms),
            "fixed chain len {}",
            terms.len()
        );
    }

    // Random chains of varying length exercise both toggle directions broadly.
    let mut rng = SplitMix64(0x1234_5678);
    for len in 0..=6usize {
        for _ in 0..20 {
            let terms: Vec<_> = (0..len)
                .map(|_| (rng.coeffs(), rng.coeffs(), rng.coeffs()))
                .collect();
            assert_eq!(
                run_prod_acc_chain(&terms),
                reference_prod_sum(&terms),
                "random chain len {len}"
            );
        }
    }
}
