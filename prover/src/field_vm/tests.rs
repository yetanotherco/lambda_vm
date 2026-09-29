use super::asm::{Asm, AsmError};
use super::executor::{ExecError, Execution, HintKind, HintRequest, Memory, NoHints, execute};
use super::isa::{Arg, HALT_PC, Instr, N, Program, REG_PC, REG_ZERO, gpr};
use crate::tables::types::{FE, FEE};

const MAX_STEPS: usize = 1 << 16;

fn fee(a: u64, b: u64, c: u64) -> FEE {
    FEE::new([FE::from(a), FE::from(b), FE::from(c)])
}

fn run(program: &Program, mem: Memory) -> Result<Execution, ExecError> {
    execute(program, mem, &mut NoHints, MAX_STEPS)
}

fn assert_bookkeeping(program: &Program, exec: &Execution) {
    assert_eq!(
        exec.decode_mult.iter().sum::<u64>(),
        exec.steps.len() as u64
    );
    let mem_accesses: u64 = exec
        .steps
        .iter()
        .map(|s| {
            program.instrs[s.state.pc as usize]
                .args()
                .iter()
                .filter(|a| a.mem)
                .count() as u64
        })
        .sum();
    assert_eq!(exec.mem_mult.iter().sum::<u64>(), mem_accesses);

    let last = exec.steps.last().unwrap();
    assert_eq!(last.state.pc, HALT_PC);
    assert!(last.state.zero);
    assert_eq!(last.state.regs, [FEE::zero(); N]);
    assert_eq!(last.args, [FEE::zero(); 4]);
}

#[test]
fn straight_line_fma_matches_native() {
    let (x, y, z) = (fee(3, 5, 7), fee(11, 13, 17), fee(19, 23, 29));
    let mut asm = Asm::new();
    asm.fma(
        Arg::mem_abs(3),
        Arg::mem_abs(0),
        Arg::mem_abs(1),
        Arg::mem_abs(2),
    )
    .halt();
    let program = asm.finish().unwrap();
    let exec = run(&program, Memory::with_init(8, &[x, y, z])).unwrap();
    assert_eq!(exec.mem[3], x * y + z);
    assert_eq!(exec.mem_mult[..4], [1, 1, 1, 1]);
    assert_bookkeeping(&program, &exec);
}

#[test]
fn loop_computes_power() {
    let (acc, cnt, x) = (gpr(0), gpr(1), gpr(2));
    let base = fee(2, 1, 9);
    let n = 37;
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
    let program = asm.finish().unwrap();
    let exec = run(&program, Memory::with_init(2, &[base])).unwrap();
    assert_eq!(exec.mem[1], base.pow(n as u64));
    assert_eq!(exec.steps.len(), 2 + 3 + 3 * n as usize + 2);
    assert_bookkeeping(&program, &exec);
}

#[test]
fn jump_zero_takes_and_skips() {
    for (v, expected) in [(0, 1), (5, 2)] {
        let r = gpr(0);
        let mut asm = Asm::new();
        let taken = asm.label();
        asm.mov(Arg::reg(r), Arg::imm(v)).jump_zero(taken);
        asm.store(Arg::mem_abs(0), Arg::imm(2)).halt();
        asm.bind(taken);
        asm.store(Arg::mem_abs(0), Arg::imm(1)).halt();
        let program = asm.finish().unwrap();
        let exec = run(&program, Memory::new(1)).unwrap();
        assert_eq!(exec.mem[0], FEE::from(expected as u64));
        assert_bookkeeping(&program, &exec);
    }
}

#[test]
fn horner_over_indexed_memory() {
    let (acc, i, x) = (gpr(0), gpr(1), gpr(2));
    let coeffs: Vec<FEE> = (0..20u64).map(|k| fee(k + 1, 3 * k, k * k)).collect();
    let point = fee(7, 0, 1);
    // MEM[0] = x, MEM[1..=20] = coeffs (low to high), result at MEM[21].
    let mut init = vec![point];
    init.extend(&coeffs);
    let mut asm = Asm::new();
    asm.mov(Arg::reg(x), Arg::mem_abs(0))
        .mov(Arg::reg(acc), Arg::imm(0))
        .mov(Arg::reg(i), Arg::imm(coeffs.len() as i64));
    let top = asm.label();
    asm.bind(top);
    asm.fma_out(Arg::reg(acc), Arg::reg(acc), Arg::reg(x), Arg::mem(i, 0))
        .sub(Arg::reg(i), Arg::reg(i), Arg::imm(1))
        .jump_not_zero(top)
        .store(Arg::mem_abs(21), Arg::reg(acc))
        .halt();
    let program = asm.finish().unwrap();
    let exec = run(&program, Memory::with_init(22, &init)).unwrap();
    let expected = coeffs
        .iter()
        .rev()
        .fold(FEE::zero(), |acc, c| acc * point + c);
    assert_eq!(exec.mem[21], expected);
    assert_bookkeeping(&program, &exec);
}

#[test]
fn inv_is_solved_from_the_constraint() {
    let (r, a) = (gpr(0), gpr(1));
    let v = fee(123, 456, 789);
    let mut asm = Asm::new();
    asm.mov(Arg::reg(a), Arg::mem_abs(0))
        .inv(r, Arg::reg(a))
        .store(Arg::mem_abs(1), Arg::reg(r))
        .halt();
    let program = asm.finish().unwrap();
    let exec = run(&program, Memory::with_init(2, &[v])).unwrap();
    assert_eq!(exec.mem[1] * v, FEE::one());
    assert_bookkeeping(&program, &exec);
}

#[test]
fn inv_of_zero_is_unresolved() {
    let (r, a) = (gpr(0), gpr(1));
    let mut asm = Asm::new();
    asm.mov(Arg::reg(a), Arg::imm(0)).inv(r, Arg::reg(a)).halt();
    let program = asm.finish().unwrap();
    let pc = 3;
    assert_eq!(
        run(&program, Memory::new(0)).unwrap_err(),
        ExecError::UnresolvedHint { pc, reg: r }
    );
}

/// Recursive factorial through `CALL`/`RET`, with fresh frames from the oracle.
#[test]
fn call_ret_recursive_factorial() {
    let (fp, n, t, res) = (gpr(0), gpr(1), gpr(2), gpr(3));
    let mut asm = Asm::new();
    let fact = asm.label();
    asm.mov(Arg::reg(fp), Arg::imm(8))
        .call(fact, fp, &[], &[Arg::imm(6)], Some(res))
        .store(Arg::mem_abs(0), Arg::reg(res))
        .halt();

    // fact: n = MEM[fp + 2]; MEM[fp + 3] = n == 0 ? 1 : n * fact(n - 1).
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

    let mut next_frame = 16u64;
    let mut frames = |req: &HintRequest| {
        assert_eq!((req.kind, req.reg), (HintKind::Output, fp));
        next_frame += 16;
        Some(FEE::from(next_frame))
    };
    let exec = execute(&program, Memory::new(256), &mut frames, MAX_STEPS).unwrap();
    assert_eq!(exec.mem[0], FEE::from(720u64));
    assert_bookkeeping(&program, &exec);
}

#[test]
fn false_assertion_is_an_fma_violation() {
    let mut asm = Asm::new();
    asm.fma(Arg::imm(1), Arg::imm(0), Arg::imm(0), Arg::imm(0))
        .halt();
    let program = asm.finish().unwrap();
    assert_eq!(
        run(&program, Memory::new(0)).unwrap_err(),
        ExecError::FmaViolation { pc: 2 }
    );
}

#[test]
fn storing_a_different_value_twice_is_an_fma_violation() {
    let mut asm = Asm::new();
    asm.store(Arg::mem_abs(0), Arg::imm(1))
        .store(Arg::mem_abs(0), Arg::imm(1))
        .store(Arg::mem_abs(0), Arg::imm(2))
        .halt();
    let program = asm.finish().unwrap();
    assert_eq!(
        run(&program, Memory::new(1)).unwrap_err(),
        ExecError::FmaViolation { pc: 4 }
    );
}

#[test]
fn reading_a_free_cell_fails() {
    let mut asm = Asm::new();
    asm.mov(Arg::reg(gpr(0)), Arg::mem_abs(3)).halt();
    let program = asm.finish().unwrap();
    assert_eq!(
        run(&program, Memory::new(4)).unwrap_err(),
        ExecError::UninitializedRead { pc: 2, addr: 3 }
    );
}

#[test]
fn bad_addresses_fail() {
    let mut asm = Asm::new();
    asm.mov(Arg::reg(gpr(0)), Arg::mem_abs(4)).halt();
    let program = asm.finish().unwrap();
    assert!(matches!(
        run(&program, Memory::new(4)).unwrap_err(),
        ExecError::BadAddress { pc: 2, .. }
    ));

    // An extension-field address.
    let r = gpr(0);
    let mut asm = Asm::new();
    asm.mov(Arg::reg(r), Arg::mem_abs(0))
        .mov(Arg::reg(gpr(1)), Arg::mem(r, 0))
        .halt();
    let program = asm.finish().unwrap();
    assert!(matches!(
        run(&program, Memory::with_init(4, &[fee(1, 1, 0)])).unwrap_err(),
        ExecError::BadAddress { pc: 3, .. }
    ));
}

#[test]
fn output_hint_followed_by_input_hint_is_a_collision() {
    let (r, a) = (gpr(0), gpr(1));
    let mut asm = Asm::new();
    asm.mov(Arg::reg(a), Arg::imm(3))
        .mov(Arg::reg(r), Arg::imm(5))
        .inv(r, Arg::reg(a))
        .halt();
    let program = asm.finish().unwrap();
    assert_eq!(
        run(&program, Memory::new(0)).unwrap_err(),
        ExecError::HintCollision { pc: 3, reg: r }
    );
}

#[test]
fn jump_out_of_the_program_fails() {
    let mut asm = Asm::new();
    asm.fma_out(Arg::reg(REG_PC), Arg::imm(0), Arg::imm(0), Arg::imm(1000));
    let program = asm.finish().unwrap();
    assert!(matches!(
        run(&program, Memory::new(0)).unwrap_err(),
        ExecError::PcOutOfRange { pc: 1000, .. }
    ));
}

#[test]
fn infinite_loop_hits_the_step_limit() {
    let mut asm = Asm::new();
    let top = asm.label();
    asm.bind(top);
    asm.jump(top);
    let program = asm.finish().unwrap();
    assert_eq!(
        execute(&program, Memory::new(0), &mut NoHints, 100).unwrap_err(),
        ExecError::StepLimit
    );
}

#[test]
fn invalid_programs_are_rejected() {
    let mut asm = Asm::new();
    asm.fma_out(Arg::reg(REG_ZERO), Arg::imm(0), Arg::imm(0), Arg::imm(0));
    assert!(matches!(asm.finish(), Err(AsmError::Invalid { pc: 2, .. })));

    let mut asm = Asm::new();
    asm.emit(Instr::new(
        Arg::reg(9),
        Arg::imm(0),
        Arg::imm(0),
        Arg::imm(0),
        false,
    ));
    assert!(matches!(asm.finish(), Err(AsmError::Invalid { pc: 2, .. })));

    let mut asm = Asm::new();
    let l = asm.label();
    asm.jump(l);
    assert_eq!(asm.finish().unwrap_err(), AsmError::UnboundLabel(l));
}
