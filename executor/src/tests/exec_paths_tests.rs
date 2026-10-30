//! The executor's access paths ([`ExecPaths`]): every fast path reads and
//! writes the bytes, and marks the written words, the general path does, and an
//! executor on the fast paths writes the logs the general one writes.
use crate::elf::{Elf, Segment};
use crate::vm::execution::Executor;
use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{
    DMA_MEMCPY_SYSCALL_NUMBER, DMA_MEMSET_GAP, DMA_MEMSET_SYSCALL_NUMBER, ExecutionError,
};
use crate::vm::logs::Log;
use crate::vm::memory::{ExecPaths, Memory, StoreKind};
use crate::vm::registers::Registers;
use proptest::prelude::*;

/// Every path setting: the general one, each fast path alone, both.
const SETTINGS: [ExecPaths; 4] = [
    ExecPaths::GENERAL,
    ExecPaths {
        fixed_access: true,
        dma_bulk: false,
    },
    ExecPaths {
        fixed_access: false,
        dma_bulk: true,
    },
    ExecPaths::FAST,
];

/// Every written word's bytes, in address order: what a snapshot, the prover's
/// image and `iter_bytes` read.
fn written(memory: &Memory) -> Vec<(u64, u8)> {
    let mut bytes: Vec<_> = memory.iter_bytes().collect();
    bytes.sort_unstable();
    bytes
}

/// Bases near the places an access can go wrong: a page's start and end, the
/// low directory's last page, the stack's end of the space and a page in
/// neither directory.
const BASES: [u64; 6] = [
    0x1000,
    (1 << 16) - 16,
    (1 << 33) - 24,
    u64::MAX - 0x40,
    1 << 40,
    0xFF00_0000,
];

#[derive(Clone, Debug)]
enum Access {
    Load { width: u8, at: u64 },
    Store { width: u8, at: u64, value: u64 },
}

fn access() -> impl Strategy<Value = Access> {
    let at = (0..BASES.len(), 0u64..32).prop_map(|(b, off)| BASES[b].wrapping_add(off));
    let width = prop_oneof![Just(1u8), Just(2u8), Just(4u8), Just(8u8)];
    prop_oneof![
        (width.clone(), at.clone()).prop_map(|(width, at)| Access::Load { width, at }),
        (width, at, any::<u64>()).prop_map(|(width, at, value)| Access::Store { width, at, value }),
    ]
}

/// One access on `memory`: a load's value, or `None` for a store; an access
/// past the end of the space is the error either path gives.
fn apply(memory: &mut Memory, access: &Access) -> Result<Option<u64>, String> {
    let e = |e: crate::vm::memory::MemoryError| e.to_string();
    Ok(match *access {
        Access::Load { width, at } => Some(match width {
            1 => memory.load_byte(at) as u64,
            2 => memory.load_half(at).map_err(e)? as u64,
            4 => memory.load_word(at).map_err(e)? as u64,
            _ => memory.load_doubleword(at).map_err(e)?,
        }),
        Access::Store { width, at, value } => {
            match width {
                1 => memory.store_byte(at, value as u8),
                2 => memory.store_half(at, value as u16).map_err(e)?,
                4 => memory.store_word(at, value as u32).map_err(e)?,
                _ => memory.store_doubleword(at, value).map_err(e)?,
            }
            None
        }
    })
}

proptest! {
    // Any run of loads and stores, inside a page, across a page boundary, in
    // either directory and in neither: every setting loads the same values and
    // leaves the same written bytes, on both stores.
    #[test]
    fn every_access_path_reads_and_writes_the_same_bytes(
        accesses in prop::collection::vec(access(), 1..200),
    ) {
        for kind in [StoreKind::Pages, StoreKind::Words] {
            let mut reference = Memory::with_store_and_paths(kind, ExecPaths::GENERAL);
            let mut fast = Memory::with_store_and_paths(kind, ExecPaths::FAST);
            for a in &accesses {
                prop_assert_eq!(apply(&mut reference, a), apply(&mut fast, a), "{:?} on {:?}", a, kind);
            }
            prop_assert_eq!(written(&reference), written(&fast), "{:?}", kind);
        }
    }

    // A DMA memcpy or memset of any length, overlapping or not, crossing a page
    // or not: every setting writes the same log, the same bytes and the same
    // written words, and refuses the same calls.
    #[test]
    fn every_dma_path_copies_the_same_bytes(
        base in 0usize..BASES.len(),
        src_off in 0u64..64,
        gap in -40i64..300,
        count in 0u64..=256,
        memset in any::<bool>(),
        seed in prop::collection::vec(any::<u8>(), 600),
    ) {
        let src = BASES[base].wrapping_sub(256).wrapping_add(src_off);
        let dst = if memset { src.wrapping_add(DMA_MEMSET_GAP) } else { src.wrapping_add_signed(gap) };
        let number = if memset { DMA_MEMSET_SYSCALL_NUMBER } else { DMA_MEMCPY_SYSCALL_NUMBER };
        let mut outcomes = Vec::new();
        for paths in SETTINGS {
            let mut memory = Memory::with_store_and_paths(StoreKind::Pages, paths);
            // Some of the bytes around the call written, some never.
            for (i, &b) in seed.iter().enumerate() {
                if b % 3 != 0 {
                    memory.store_byte(src.wrapping_sub(300).wrapping_add(i as u64), b);
                }
            }
            let log = run_ecall(&mut memory, number, dst, src, count)
                .map(|log| (log.next_pc, log.src1_val, log.src2_val, log.dst_val))
                .map_err(|e| e.to_string());
            outcomes.push((log, written(&memory)));
        }
        for (paths, outcome) in SETTINGS.iter().zip(&outcomes).skip(1) {
            prop_assert_eq!(outcome, &outcomes[0], "{:?}", paths);
        }
    }
}

fn run_ecall(
    memory: &mut Memory,
    number: u64,
    dst: u64,
    src: u64,
    count: u64,
) -> Result<Log, ExecutionError> {
    let mut registers = Registers::default();
    let mut pc = 0x1000;
    registers.write(17, number)?;
    registers.write(10, dst)?;
    registers.write(11, src)?;
    registers.write(12, count)?;
    Instruction::EcallEbreak.run(&mut pc, &mut registers, memory)
}

// RV64 encodings for the little program below.
fn i_type(opcode: u32, funct3: u32, rd: u32, rs1: u32, imm: i32) -> u32 {
    ((imm as u32 & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}
fn s_type(funct3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
    let imm = imm as u32 & 0xfff;
    ((imm >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | ((imm & 0x1f) << 7) | 0x23
}
fn b_type(funct3: u32, rs1: u32, rs2: u32, imm: i32) -> u32 {
    let imm = imm as u32 & 0x1fff;
    (((imm >> 12) & 1) << 31)
        | (((imm >> 5) & 0x3f) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (((imm >> 1) & 0xf) << 8)
        | (((imm >> 11) & 1) << 7)
        | 0x63
}
fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
    i_type(0x13, 0, rd, rs1, imm)
}
const ECALL: u32 = 0x73;

/// A program that stores and loads bytes, halves, words and doublewords at a
/// moving, misaligned address across a page boundary, and copies with both DMA
/// calls, `rounds` times, then halts.
fn program(rounds: i32) -> Elf {
    let base = 0x1_0000u64;
    let mut code = vec![
        // x5 = rounds, x6 = a moving address below the page boundary at
        // 0x30000 (it passes it on the late rounds), x7 = a value.
        addi(5, 0, rounds),
        (0x30 << 12) | (6 << 7) | 0x37, // lui x6, 0x30 → 0x30000
        addi(6, 6, -0x200),             // x6 = 0x2fe00
        addi(7, 0, -0x123),
    ];
    let top = code.len();
    code.extend([
        addi(7, 7, 0x2f1),
        s_type(0, 6, 7, 3),            // sb x7, 3(x6)
        s_type(1, 6, 7, 5),            // sh x7, 5(x6)
        s_type(2, 6, 7, 9),            // sw x7, 9(x6)
        s_type(3, 6, 7, 0x1fb),        // sd x7, 0x1fb(x6): crosses 0x30000 on some rounds
        i_type(0x03, 3, 28, 6, 0x1f9), // ld x28, 0x1f9(x6)
        i_type(0x03, 4, 29, 6, 3),     // lbu x29, 3(x6)
        i_type(0x03, 1, 30, 6, 5),     // lh x30, 5(x6)
        i_type(0x03, 6, 31, 6, 9),     // lwu x31, 9(x6)
        // memcpy(dst = x6 + 64, src = x6, n = 37)
        addi(10, 6, 64),
        addi(11, 6, 0),
        addi(12, 0, 37),
        addi(17, 0, -30), // DMA_MEMCPY_SYSCALL_NUMBER = -30
        ECALL,
        // memset(dst = x6 + 8 + 128, src = x6 + 128, n = 29)
        addi(11, 6, 128),
        addi(10, 11, 8),
        addi(12, 0, 29),
        addi(17, 0, -31), // DMA_MEMSET_SYSCALL_NUMBER = -31
        ECALL,
        addi(6, 6, 7),
        addi(5, 5, -1),
    ]);
    let back = -(((code.len() - top) * 4) as i32);
    code.push(b_type(1, 5, 0, back)); // bne x5, x0, top
    code.extend([addi(10, 0, 0), addi(17, 0, 93), ECALL]);
    Elf {
        entry_point: base,
        data: vec![Segment {
            base_addr: base,
            values: code,
            is_executable: true,
        }],
    }
}

/// Every log of a run, a window of `window` cycles at a time.
fn logs_of(mut executor: Executor, window: usize, owned: bool) -> Vec<Log> {
    let mut all = Vec::new();
    if owned {
        while let Some(logs) = executor.resume_owned(window).expect("the run") {
            all.extend(logs);
        }
    } else {
        while let Some(logs) = executor.resume_with_limit(window).expect("the run") {
            all.extend_from_slice(logs);
        }
    }
    all
}

fn fields(logs: &[Log]) -> Vec<[u64; 5]> {
    logs.iter()
        .map(|l| [l.current_pc, l.next_pc, l.src1_val, l.src2_val, l.dst_val])
        .collect()
}

// The program above writes the same logs on every path setting, and the
// fast paths are not idle in it (the copies and the page-crossing doubleword).
#[test]
fn every_path_setting_writes_the_same_logs() {
    let elf = program(80);
    let reference = fields(&logs_of(
        Executor::new_with_paths(&elf, vec![], ExecPaths::GENERAL).unwrap(),
        1 << 10,
        false,
    ));
    assert_eq!(reference.len(), 4 + 80 * 22 + 3, "the program's cycles");
    for paths in SETTINGS {
        let run = Executor::new_with_paths(&elf, vec![], paths).unwrap();
        assert_eq!(
            fields(&logs_of(run, 1 << 10, false)),
            reference,
            "{paths:?}"
        );
    }
}

// `resume_owned` hands out, window by window, the logs `resume_with_limit`
// lends, including a last window shorter than the rest.
#[test]
fn resume_owned_hands_out_the_logs_resume_lends() {
    let elf = program(30);
    for window in [1, 7, 64, 1 << 20] {
        let lent = logs_of(Executor::new(&elf, vec![]).unwrap(), window, false);
        let owned = logs_of(Executor::new(&elf, vec![]).unwrap(), window, true);
        assert_eq!(fields(&owned), fields(&lent), "window {window}");
    }
    let mut executor = Executor::new(&elf, vec![]).unwrap();
    let first = executor.resume_owned(7).unwrap().unwrap();
    assert_eq!(first.len(), 7);
    let second = executor.resume_owned(7).unwrap().unwrap();
    assert_eq!(
        second[0].current_pc, first[6].next_pc,
        "the next window goes on"
    );
}

// The knobs: each path on unless its knob is `0`.
#[test]
fn each_path_is_on_unless_its_knob_is_zero() {
    assert_eq!(ExecPaths::from_values(None, None), ExecPaths::FAST);
    assert_eq!(ExecPaths::from_values(Some("1"), Some("")), ExecPaths::FAST);
    assert_eq!(
        ExecPaths::from_values(Some("0"), Some(" 0 ")),
        ExecPaths::GENERAL
    );
    assert_eq!(
        ExecPaths::from_values(Some("0"), None),
        ExecPaths {
            fixed_access: false,
            dma_bulk: true
        }
    );
    assert_eq!(
        ExecPaths::from_values(None, Some("0")),
        ExecPaths {
            fixed_access: true,
            dma_bulk: false
        }
    );
}
