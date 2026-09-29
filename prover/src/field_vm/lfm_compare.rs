//! Field VM against the LFM on the same kernel: Horner over the cubic
//! extension, coefficients supplied as inputs.
//!
//! `cargo test --release -p lambda-vm-prover --lib field_vm::lfm_compare -- --ignored --nocapture --test-threads=1`

use std::time::Instant;

use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::proof::stark::MultiProof;

use super::asm::Asm;
use super::executor::{Memory, NoHints, execute};
use super::isa::{Arg, Program, gpr};
use super::prove::{generate_traces, program_id, prove_traces, verify};
use crate::lfm::builder::LfmBuilder;
use crate::lfm::compiler::compile;
use crate::lfm::edsl::horner_ext;
use crate::lfm::proof::lfm_prove;
use crate::lfm::registry::build_artifacts;
use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

const RUNS: usize = 5;

fn options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(4).expect("blowup 4 is valid")
}

fn inputs(len: usize) -> Vec<FEE> {
    (0..=len as u64)
        .map(|k| FEE::new([FE::from(k + 3), FE::from(2 * k + 1), FE::from(k * k + 7)]))
        .collect()
}

/// `(Σ rows, Σ rows × committed columns)`, an extension column counting as one.
fn census<PI>(proof: &MultiProof<GoldilocksField, GoldilocksExtension, PI>) -> (usize, usize) {
    proof.proofs.iter().fold((0, 0), |(r, c), p| {
        (
            r + p.trace_length,
            c + p.trace_length * p.trace_ood_evaluations.width,
        )
    })
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn fvm_loop(len: usize) -> Program {
    let (acc, i, x) = (gpr(0), gpr(1), gpr(2));
    let mut asm = Asm::new();
    asm.mov(Arg::reg(x), Arg::mem_abs(0))
        .mov(Arg::reg(acc), Arg::imm(0))
        .mov(Arg::reg(i), Arg::imm(len as i64));
    let top = asm.label();
    asm.bind(top);
    asm.fma_out(Arg::reg(acc), Arg::reg(acc), Arg::reg(x), Arg::mem(i, 0))
        .sub(Arg::reg(i), Arg::reg(i), Arg::imm(1))
        .jump_not_zero(top)
        .store(Arg::mem_abs(len as u64 + 1), Arg::reg(acc))
        .halt();
    asm.finish().unwrap()
}

/// Straight-line, one row per coefficient, like the LFM's `horner_ext`.
fn fvm_unrolled(len: usize) -> Program {
    let (acc, x) = (gpr(0), gpr(2));
    let mut asm = Asm::new();
    asm.mov(Arg::reg(x), Arg::mem_abs(0))
        .mov(Arg::reg(acc), Arg::mem_abs(len as u64));
    for k in (1..len as u64).rev() {
        asm.fma_out(Arg::reg(acc), Arg::reg(acc), Arg::reg(x), Arg::mem_abs(k));
    }
    asm.store(Arg::mem_abs(len as u64 + 1), Arg::reg(acc))
        .halt();
    asm.finish().unwrap()
}

fn report_fvm(label: &str, program: &Program, len: usize) {
    let opts = options();
    let exec = execute(
        program,
        Memory::with_init(len + 2, &inputs(len)),
        &mut NoHints,
        1 << 24,
    )
    .unwrap();
    let id = program_id(program, &opts);
    let (mut prove_ms, mut verify_ms, mut shape) = (Vec::new(), Vec::new(), (0, 0));
    for _ in 0..RUNS {
        let mut traces = generate_traces(program, &exec);
        let t = Instant::now();
        let proof = prove_traces(&id, &mut traces, &opts).unwrap();
        prove_ms.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        assert!(verify(&id, &proof, &opts));
        verify_ms.push(t.elapsed().as_secs_f64() * 1e3);
        shape = census(&proof);
    }
    eprintln!(
        "{label:>12} len={len:>6} program={:>6} rows={:>7} cells={:>9} prove_ms={:>8.1} verify_ms={:>6.1}",
        program.len(),
        shape.0,
        shape.1,
        median(prove_ms),
        median(verify_ms)
    );
}

fn report_lfm(len: usize) {
    let opts = options();
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(len as u32 + 1);
    let x = b.hint_word(arena, 0).as_ext();
    let coeffs: Vec<_> = (1..=len as u32)
        .map(|k| b.hint_word(arena, k).as_ext())
        .collect();
    let r = horner_ext(&mut b, x, &coeffs);
    b.public(r.as_cell());
    let program = compile(b.finish());
    let artifacts = build_artifacts(&program, &opts);
    let words = inputs(len)
        .iter()
        .map(|v| {
            let [a, b, c] = *v.value();
            [a, b, c, FE::zero()]
        })
        .collect();
    let arenas = vec![words];
    let (mut prove_ms, mut shape) = (Vec::new(), (0, 0));
    for _ in 0..RUNS {
        let t = Instant::now();
        let proof = lfm_prove(&program, &artifacts, &arenas, &opts).unwrap();
        prove_ms.push(t.elapsed().as_secs_f64() * 1e3);
        shape = census(&proof.proof);
        if std::env::var_os("FVM_COMPARE_TABLES").is_some() && prove_ms.len() == 1 {
            let tables: Vec<_> = proof
                .proof
                .proofs
                .iter()
                .map(|p| (p.trace_length, p.trace_ood_evaluations.width))
                .collect();
            eprintln!("lfm tables (rows, cols): {tables:?}");
        }
    }
    eprintln!(
        "{:>12} len={len:>6} program={:>6} rows={:>7} cells={:>9} prove_ms={:>8.1}",
        "lfm",
        program.instrs.len(),
        shape.0,
        shape.1,
        median(prove_ms)
    );
}

/// Cost of building each machine's program identity (the preprocessed roots).
#[test]
#[ignore]
fn program_build_times() {
    let opts = options();
    for len in [1usize << 10, 1 << 14, 1 << 16] {
        let (loop_p, unrolled) = (fvm_loop(len), fvm_unrolled(len));
        let t = Instant::now();
        program_id(&loop_p, &opts);
        let fvm_loop_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        program_id(&unrolled, &opts);
        let fvm_unrolled_ms = t.elapsed().as_secs_f64() * 1e3;
        let mut b = LfmBuilder::new();
        let arena = b.declare_arena(len as u32 + 1);
        let x = b.hint_word(arena, 0).as_ext();
        let coeffs: Vec<_> = (1..=len as u32)
            .map(|k| b.hint_word(arena, k).as_ext())
            .collect();
        let r = horner_ext(&mut b, x, &coeffs);
        b.public(r.as_cell());
        let program = compile(b.finish());
        let t = Instant::now();
        build_artifacts(&program, &opts);
        let lfm_ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "build len={len:>6} fvm-loop={fvm_loop_ms:>8.1}ms fvm-unrolled={fvm_unrolled_ms:>8.1}ms lfm={lfm_ms:>8.1}ms"
        );
    }
}

#[test]
#[ignore]
fn lfm_tables() {
    for len in [1usize << 10, 1 << 16] {
        report_lfm(len);
    }
}

#[test]
#[ignore]
fn horner_field_vm_vs_lfm() {
    for len in [1usize << 10, 1 << 14, 1 << 16] {
        report_fvm("fvm-loop", &fvm_loop(len), len);
        report_fvm("fvm-unrolled", &fvm_unrolled(len), len);
        report_lfm(len);
    }
}
