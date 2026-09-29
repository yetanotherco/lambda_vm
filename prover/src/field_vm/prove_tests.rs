use super::air::cols;
use super::asm::Asm;
use super::executor::{Execution, HintRequest, Memory, NoHints, execute};
use super::isa::{Arg, Program, gpr};
use super::mem::cols as mem_cols;
use super::prove::{
    FvmTraces, default_options, generate_traces, program_id, prove, prove_traces, verify,
};
use crate::tables::types::{FE, FEE};

fn fee(a: u64, b: u64, c: u64) -> FEE {
    FEE::new([FE::from(a), FE::from(b), FE::from(c)])
}

fn power_program(n: i64) -> Program {
    let (acc, cnt, x) = (gpr(0), gpr(1), gpr(2));
    let mut asm = Asm::new();
    asm.mov(Arg::reg(acc), Arg::imm(1))
        .mov(Arg::reg(x), Arg::mem_abs(0))
        .mov(Arg::reg(cnt), Arg::imm(n));
    let top = asm.label();
    asm.bind(top);
    asm.mul(Arg::reg(acc), Arg::reg(acc), Arg::reg(x))
        .sub(Arg::reg(cnt), Arg::reg(cnt), Arg::imm(1))
        .jump_not_zero(top)
        .store(Arg::mem_abs(1), Arg::reg(acc))
        .halt();
    asm.finish().unwrap()
}

fn prove_and_verify(program: &Program, exec: &Execution) -> bool {
    let options = default_options();
    let proof = prove(program, exec, &options).expect("proving");
    verify(&program_id(program, &options), &proof, &options)
}

#[test]
fn power_loop_proves_and_verifies() {
    let program = power_program(13);
    let exec = execute(
        &program,
        Memory::with_init(2, &[fee(2, 1, 9)]),
        &mut NoHints,
        1 << 12,
    )
    .unwrap();
    assert!(prove_and_verify(&program, &exec));
}

#[test]
fn inv_and_straight_line_prove_and_verify() {
    let (r, a) = (gpr(0), gpr(1));
    let mut asm = Asm::new();
    asm.mov(Arg::reg(a), Arg::mem_abs(0))
        .inv(r, Arg::reg(a))
        .fma(Arg::mem_abs(1), Arg::reg(r), Arg::mem_abs(0), Arg::imm(0))
        .halt();
    let program = asm.finish().unwrap();
    let exec = execute(
        &program,
        Memory::with_init(2, &[fee(5, 6, 7), FEE::one()]),
        &mut NoHints,
        64,
    )
    .unwrap();
    assert!(prove_and_verify(&program, &exec));
}

#[test]
fn recursive_calls_prove_and_verify() {
    let (fp, n, t, res) = (gpr(0), gpr(1), gpr(2), gpr(3));
    let mut asm = Asm::new();
    let fact = asm.label();
    asm.mov(Arg::reg(fp), Arg::imm(8))
        .call(fact, fp, &[], &[Arg::imm(4)], Some(res))
        .store(Arg::mem_abs(0), Arg::reg(res))
        .halt();
    asm.bind(fact);
    let base = asm.label();
    asm.mov(Arg::reg(n), Arg::mem(fp, 2)).jump_zero(base);
    asm.sub(Arg::reg(t), Arg::reg(n), Arg::imm(1))
        .call(fact, fp, &[n], &[Arg::reg(t)], Some(res))
        .mul(Arg::reg(res), Arg::reg(res), Arg::reg(n))
        .store(Arg::mem(fp, 3), Arg::reg(res))
        .ret(fp);
    asm.bind(base);
    asm.store(Arg::mem(fp, 3), Arg::imm(1)).ret(fp);
    let program = asm.finish().unwrap();
    let mut next = 16u64;
    let mut frames = |_: &HintRequest| {
        next += 16;
        Some(FEE::from(next))
    };
    let exec = execute(&program, Memory::new(128), &mut frames, 1 << 12).unwrap();
    assert_eq!(exec.mem[0], FEE::from(24u64));
    assert!(prove_and_verify(&program, &exec));
}

fn power_exec() -> (Program, Execution) {
    let program = power_program(3);
    let exec = execute(
        &program,
        Memory::with_init(2, &[fee(2, 1, 9)]),
        &mut NoHints,
        1 << 12,
    )
    .unwrap();
    (program, exec)
}

/// Proves `exec` after `tamper` edits its traces; `true` iff the proof verifies.
fn verifies_after(
    program: &Program,
    exec: &Execution,
    tamper: impl FnOnce(&mut FvmTraces),
) -> bool {
    let options = default_options();
    let id = program_id(program, &options);
    let mut traces = generate_traces(program, exec);
    tamper(&mut traces);
    match prove_traces(&id, &mut traces, &options) {
        Ok(proof) => verify(&id, &proof, &options),
        Err(_) => false,
    }
}

fn bump(
    t: &mut stark::table::Table<crate::tables::types::GoldilocksField>,
    row: usize,
    col: usize,
) {
    let v = *t.get(row, col);
    t.set(row, col, v + FE::one());
}

#[test]
fn untampered_traces_verify() {
    let (program, exec) = power_exec();
    assert!(verifies_after(&program, &exec, |_| {}));
}

#[test]
fn register_changed_without_a_hint_is_rejected() {
    let (program, exec) = power_exec();
    // General-purpose register 4 is never hinted by this program.
    assert!(!verifies_after(&program, &exec, |t| bump(
        &mut t.fvm.main_table,
        3,
        cols::reg(4)
    )));
}

#[test]
fn pc_skipping_without_a_hint_is_rejected() {
    let (program, exec) = power_exec();
    assert!(!verifies_after(&program, &exec, |t| bump(
        &mut t.fvm.main_table,
        2,
        cols::PC
    )));
}

#[test]
fn lying_zero_flag_is_rejected() {
    let (program, exec) = power_exec();
    assert!(!verifies_after(&program, &exec, |t| {
        let z = *t.fvm.main_table.get(4, cols::ZERO);
        t.fvm.main_table.set(4, cols::ZERO, FE::one() - z);
    }));
}

#[test]
fn wrong_fma_output_is_rejected() {
    let (program, exec) = power_exec();
    assert!(!verifies_after(&program, &exec, |t| bump(
        &mut t.fvm.main_table,
        1,
        cols::arg(0)
    )));
}

#[test]
fn out_of_range_register_index_is_rejected() {
    let (program, exec) = power_exec();
    assert!(!verifies_after(&program, &exec, |t| {
        t.fvm
            .main_table
            .set(1, cols::ARG_REG + 1, FE::from(super::isa::NUM_REGS as u64));
    }));
}

#[test]
fn forged_memory_value_is_rejected() {
    let (program, exec) = power_exec();
    assert!(!verifies_after(&program, &exec, |t| bump(
        &mut t.mem.main_table,
        0,
        mem_cols::VALUE
    )));
}

#[test]
fn duplicate_memory_address_is_rejected() {
    let (program, exec) = power_exec();
    assert!(!verifies_after(&program, &exec, |t| t.mem.main_table.set(
        1,
        mem_cols::ADDR,
        FE::zero()
    )));
}

#[test]
fn tampered_program_column_is_rejected() {
    let (program, exec) = power_exec();
    assert!(!verifies_after(&program, &exec, |t| bump(
        &mut t.decode.main_table,
        5,
        1
    )));
}

#[test]
fn execution_that_never_halts_is_rejected() {
    let (program, mut exec) = power_exec();
    exec.steps.truncate(exec.steps.len() - 2);
    assert!(!verifies_after(&program, &exec, |_| {}));
}

#[test]
fn proof_does_not_verify_for_another_program() {
    let (program, exec) = power_exec();
    let options = default_options();
    let proof = prove(&program, &exec, &options).unwrap();
    assert!(!verify(
        &program_id(&power_program(4), &options),
        &proof,
        &options
    ));
}

#[test]
fn lying_last_row_is_rejected() {
    let (program, exec) = power_exec();
    let options = default_options();
    let mut proof = prove(&program, &exec, &options).unwrap();
    proof.proofs[0].public_inputs.last_row -= 1;
    assert!(!verify(&program_id(&program, &options), &proof, &options));
}

fn horner_program(len: usize) -> Program {
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

/// Sizes and timings of a Horner evaluation (3 rows per coefficient).
/// `cargo test --release -p lambda-vm-prover --lib field_vm::prove_tests::report_horner -- --ignored --nocapture`
#[test]
#[ignore]
fn report_horner() {
    use std::time::Instant;
    let options = default_options();
    for len in [1usize << 10, 1 << 14, 1 << 16] {
        let program = horner_program(len);
        let init: Vec<FEE> = (0..=len as u64)
            .map(|k| fee(k + 3, 2 * k + 1, k * k + 7))
            .collect();
        let exec = execute(
            &program,
            Memory::with_init(len + 2, &init),
            &mut NoHints,
            1 << 22,
        )
        .unwrap();
        let id = program_id(&program, &options);
        let traces = generate_traces(&program, &exec);
        let rows = [
            traces.fvm.num_rows(),
            traces.decode.num_rows(),
            traces.mem.num_rows(),
        ];
        // Main width + 3 base columns per LogUp aux column (⌈interactions / 2⌉).
        let widths = [
            cols::NUM_COLUMNS + 3 * 3,
            super::decode::NUM_COLUMNS + 3,
            mem_cols::NUM_COLUMNS + 3,
        ];
        let cells: usize = rows.iter().zip(widths).map(|(r, w)| r * w).sum();
        let mut prove_ms = Vec::new();
        let mut verify_ms = Vec::new();
        let mut size = 0;
        for _ in 0..5 {
            let mut t = generate_traces(&program, &exec);
            let start = Instant::now();
            let proof = prove_traces(&id, &mut t, &options).unwrap();
            prove_ms.push(start.elapsed().as_secs_f64() * 1e3);
            let start = Instant::now();
            assert!(verify(&id, &proof, &options));
            verify_ms.push(start.elapsed().as_secs_f64() * 1e3);
            size = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).unwrap().len();
        }
        prove_ms.sort_by(f64::total_cmp);
        verify_ms.sort_by(f64::total_cmp);
        eprintln!(
            "len={len} steps={} rows(fvm,dec,mem)={rows:?} cells={cells} prove_ms(min/med/max)={:.1}/{:.1}/{:.1} verify_ms(med)={:.1} proof_bytes={size}",
            exec.steps.len(),
            prove_ms[0],
            prove_ms[2],
            prove_ms[4],
            verify_ms[2]
        );
    }
}
