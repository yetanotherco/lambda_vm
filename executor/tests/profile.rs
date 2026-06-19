use executor::{
    profile::{InstrClass, InstrHistogram},
    vm::{
        execution::InstructionCache,
        instruction::{
            decoding::{ArithOp, Instruction},
            execution::KECCAK_SYSCALL_NUMBER,
        },
        logs::Log,
        memory::U64HashMap,
    },
};

fn make_instructions(instructions: Vec<(u64, Instruction)>) -> InstructionCache {
    let map: U64HashMap<Instruction> = instructions.into_iter().collect();
    InstructionCache::from_map(&map)
}

fn log_at(pc: u64) -> Log {
    Log {
        current_pc: pc,
        next_pc: pc + 4,
        src1_val: 0,
        src2_val: 0,
        dst_val: 0,
    }
}

#[test]
fn classifies_arith_chips_distinctly() {
    let instructions = make_instructions(vec![
        (
            0x0,
            Instruction::Arith {
                dst: 1,
                src1: 2,
                src2: 3,
                op: ArithOp::Add,
            },
        ),
        (
            0x4,
            Instruction::Arith {
                dst: 1,
                src1: 2,
                src2: 3,
                op: ArithOp::Mul,
            },
        ),
        (
            0x8,
            Instruction::Arith {
                dst: 1,
                src1: 2,
                src2: 3,
                op: ArithOp::DivUnsigned,
            },
        ),
        (
            0xc,
            Instruction::Arith {
                dst: 1,
                src1: 2,
                src2: 3,
                op: ArithOp::ShiftLeftLogical,
            },
        ),
        (
            0x10,
            Instruction::Arith {
                dst: 1,
                src1: 2,
                src2: 3,
                op: ArithOp::SetLessThan,
            },
        ),
    ]);
    let logs: Vec<Log> = (0..5).map(|i| log_at(i * 4)).collect();

    let mut h = InstrHistogram::new();
    h.process_logs(&logs, &instructions).unwrap();

    assert_eq!(h.total(), 5);
    let counts: std::collections::BTreeMap<_, _> = h.sorted().into_iter().collect();
    assert_eq!(counts.get(&InstrClass::AluBasic), Some(&1));
    assert_eq!(counts.get(&InstrClass::Mul), Some(&1));
    assert_eq!(counts.get(&InstrClass::DivRem), Some(&1));
    assert_eq!(counts.get(&InstrClass::Shift), Some(&1));
    assert_eq!(counts.get(&InstrClass::Compare), Some(&1));
}

#[test]
fn classifies_memory_control_and_syscalls() {
    use executor::vm::instruction::decoding::LoadStoreWidth;

    let instructions = make_instructions(vec![
        (
            0x0,
            Instruction::Load {
                dst: 1,
                offset: 0,
                base: 2,
                width: LoadStoreWidth::DoubleWord,
            },
        ),
        (
            0x4,
            Instruction::Store {
                src: 1,
                offset: 0,
                base: 2,
                width: LoadStoreWidth::DoubleWord,
            },
        ),
        (0x8, Instruction::JumpAndLink { dst: 1, offset: 16 }),
        (0xc, Instruction::EcallEbreak),
        (0x10, Instruction::EcallEbreak),
    ]);

    // ecall classification is keyed on src1_val (the syscall number in x17).
    let logs = vec![
        log_at(0x0),
        log_at(0x4),
        log_at(0x8),
        Log {
            current_pc: 0xc,
            next_pc: 0x10,
            src1_val: KECCAK_SYSCALL_NUMBER,
            src2_val: 0,
            dst_val: 0,
        },
        Log {
            current_pc: 0x10,
            next_pc: 0x14,
            src1_val: 93, // halt
            src2_val: 0,
            dst_val: 0,
        },
    ];

    let mut h = InstrHistogram::new();
    h.process_logs(&logs, &instructions).unwrap();

    let counts: std::collections::BTreeMap<_, _> = h.sorted().into_iter().collect();
    assert_eq!(counts.get(&InstrClass::Load), Some(&1));
    assert_eq!(counts.get(&InstrClass::Store), Some(&1));
    assert_eq!(counts.get(&InstrClass::Jump), Some(&1));
    assert_eq!(counts.get(&InstrClass::EcallKeccak), Some(&1));
    assert_eq!(counts.get(&InstrClass::EcallHalt), Some(&1));
}
