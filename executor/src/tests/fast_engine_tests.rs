//! The fast engine ([`FastExecutor`]) against the reference [`Executor`]:
//! random programs over every op and register/immediate edge, every ecall
//! and its refusals, accesses on and across the flat regions' edges. Each
//! window's logs, the state after it (pc, registers, the published memory)
//! and every error must be the reference's.
use crate::elf::{Elf, Segment};
use crate::vm::engine::{Engine, EngineKind};
use crate::vm::execution::{Executor, ExecutorError};
use crate::vm::fast::FastExecutor;
use crate::vm::flat_memory::{FlatLayout, FlatMemory};
use crate::vm::instruction::execution::{
    BLAKE3_ABSORB_SYSCALL_NUMBER, BLAKE3_SYSCALL_NUMBER, DMA_MEMCPY_SYSCALL_NUMBER,
    DMA_MEMSET_SYSCALL_NUMBER, ECREC_SYSCALL_NUMBER, ECSM_SYSCALL_NUMBER, HINT_SYSCALL_NUMBER,
    KECCAK_SYSCALL_NUMBER, KECCAK256_SYSCALL_NUMBER,
};
use crate::vm::logs::Log;
use crate::vm::memory::{GuestMemory, Memory, MemoryError, StoreKind};
use proptest::prelude::*;

/// A small layout: the low region's end at 1 MiB and the high region the top
/// 1 MiB, so accesses on and across both edges are cheap to reach.
const SMALL: FlatLayout = FlatLayout {
    low_end: 1 << 20,
    high_base: 0u64.wrapping_sub(1 << 20),
};

const LAYOUTS: [FlatLayout; 2] = [SMALL, FlatLayout::GUEST];

/// Where the registers' initial values (and some data) sit: reachable by a
/// load at a 12-bit offset from `x0`.
const DATA_BASE: u64 = 0x100;

/// Addresses near where an access can go wrong: page and chunk edges, both
/// layouts' region edges, the private input, the paged store's directories,
/// the top of the space.
fn edges() -> Vec<u64> {
    let mut edges = vec![
        0,
        DATA_BASE,
        0x1000,
        0xFFFF,
        0x1_0000,
        0xFF00_0000,
        1 << 33,
        1 << 40,
        u64::MAX,
        u64::MAX - 7,
        u64::MAX - 0x800,
        0u64.wrapping_sub(0x1000),
        0u64.wrapping_sub(16),
    ];
    for layout in LAYOUTS {
        edges.extend([
            layout.low_end,
            layout.low_end - 4,
            layout.low_end - 0x800,
            layout.high_base,
            layout.high_base - 3,
            layout.high_base + 0x7FC,
            // A chunk's and a page's edge inside the high region.
            layout.high_base + 0x1000,
            layout.high_base + 0x1_0000,
        ]);
    }
    edges
}

/// A register value: an edge plus a small offset, a small number, a syscall
/// number, or anything.
fn value() -> impl Strategy<Value = u64> {
    let edges = edges();
    let n = edges.len();
    prop_oneof![
        4 => (0..n, -24i64..24).prop_map(move |(i, d)| edges[i].wrapping_add_signed(d)),
        2 => 0u64..300,
        1 => prop::sample::select(SYSCALLS.to_vec()),
        1 => Just(i64::MIN as u64),
        1 => Just(u64::MAX),
        2 => any::<u64>(),
    ]
}

/// Every syscall number the executor knows, and some it does not.
const SYSCALLS: [u64; 16] = [
    1,
    2,
    64,
    93,
    KECCAK_SYSCALL_NUMBER,
    KECCAK256_SYSCALL_NUMBER,
    BLAKE3_SYSCALL_NUMBER,
    BLAKE3_ABSORB_SYSCALL_NUMBER,
    ECSM_SYSCALL_NUMBER,
    ECREC_SYSCALL_NUMBER,
    DMA_MEMCPY_SYSCALL_NUMBER,
    DMA_MEMSET_SYSCALL_NUMBER,
    HINT_SYSCALL_NUMBER,
    0,
    77,
    u64::MAX - 40,
];

// --- RV64IM encoders ---------------------------------------------------------

fn r(opcode: u32, f7: u32, rs2: u32, rs1: u32, f3: u32, rd: u32) -> u32 {
    f7 << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | rd << 7 | opcode
}

fn i(opcode: u32, imm: i32, rs1: u32, f3: u32, rd: u32) -> u32 {
    ((imm as u32) & 0xFFF) << 20 | rs1 << 15 | f3 << 12 | rd << 7 | opcode
}

fn s(imm: i32, rs2: u32, rs1: u32, f3: u32) -> u32 {
    let imm = imm as u32;
    ((imm >> 5) & 0x7F) << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | (imm & 0x1F) << 7 | 0x23
}

fn b(imm: i32, rs2: u32, rs1: u32, f3: u32) -> u32 {
    let imm = imm as u32;
    ((imm >> 12) & 1) << 31
        | ((imm >> 5) & 0x3F) << 25
        | rs2 << 20
        | rs1 << 15
        | f3 << 12
        | ((imm >> 1) & 0xF) << 8
        | ((imm >> 11) & 1) << 7
        | 0x63
}

fn j(imm: i32, rd: u32) -> u32 {
    let imm = imm as u32;
    ((imm >> 20) & 1) << 31
        | ((imm >> 1) & 0x3FF) << 21
        | ((imm >> 11) & 1) << 20
        | ((imm >> 12) & 0xFF) << 12
        | rd << 7
        | 0x6F
}

/// `(f3, f7)` of the 64-bit register-register ops.
const R_OPS: [(u32, u32); 18] = [
    (0, 0x00),
    (0, 0x20),
    (4, 0),
    (6, 0),
    (7, 0),
    (1, 0),
    (5, 0),
    (5, 0x20),
    (2, 0),
    (3, 0),
    (0, 1),
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (6, 1),
    (7, 1),
];

/// `(f3, f7)` of the 32-bit register-register ops.
const RW_OPS: [(u32, u32); 10] = [
    (0, 0x00),
    (0, 0x20),
    (1, 0),
    (5, 0),
    (5, 0x20),
    (0, 1),
    (4, 1),
    (5, 1),
    (6, 1),
    (7, 1),
];

/// `ld x, (DATA_BASE + 8k)(x0)`.
fn ld_data(rd: u32, k: u64) -> u32 {
    i(0x03, (DATA_BASE + 8 * k) as i32, 0, 3, rd)
}

fn imm12() -> impl Strategy<Value = i32> {
    prop_oneof![
        3 => -2048i32..2048,
        1 => prop::sample::select(vec![0, -1, 1, 2047, -2048, 7, -8, 4, -4]),
    ]
}

fn reg() -> impl Strategy<Value = u32> {
    prop_oneof![5 => 0u32..32, 1 => Just(0u32)]
}

/// The ecalls a random program makes, by number: every one whose operands
/// cannot ask the host for a buffer of an arbitrary length (print, panic and
/// Keccak-256 can: `ecalls_run_alike` bounds their lengths).
const RANDOM_ECALLS: [i32; 10] = [64, 93, -2, -3, -4, -11, -12, -30, -31, -32];

/// Any instruction word the decoder accepts, its fields at their edges; an
/// ecall comes with the `addi a7, x0, n` that picks it.
fn word() -> impl Strategy<Value = Vec<u32>> {
    prop_oneof![
        40 => one_word().prop_map(|w| vec![w]),
        1 => prop::sample::select(RANDOM_ECALLS.to_vec())
            .prop_map(|n| vec![i(0x13, n, 0, 0, 17), 0x73]),
    ]
}

fn one_word() -> impl Strategy<Value = u32> {
    // Jump and branch offsets: mostly a few instructions away, rarely 2
    // bytes off (a misaligned pc) or far (out of the code).
    let offset = prop_oneof![
        20 => (-12i32..12).prop_map(|k| 4 * k),
        1 => (-12i32..12).prop_map(|k| 4 * k + 2),
        1 => (-2048i32..2048).prop_map(|k| 2 * k),
    ];
    prop_oneof![
        16 => (0..R_OPS.len(), reg(), reg(), reg()).prop_map(|(k, rd, rs1, rs2)| {
            let (f3, f7) = R_OPS[k];
            r(0x33, f7, rs2, rs1, f3, rd)
        }),
        8 => (0..RW_OPS.len(), reg(), reg(), reg()).prop_map(|(k, rd, rs1, rs2)| {
            let (f3, f7) = RW_OPS[k];
            r(0x3B, f7, rs2, rs1, f3, rd)
        }),
        // addi, slti, sltiu, xori, ori, andi
        12 => (prop::sample::select(vec![0u32, 2, 3, 4, 6, 7]), reg(), reg(), imm12())
            .prop_map(|(f3, rd, rs1, imm)| i(0x13, imm, rs1, f3, rd)),
        // slli, srli, srai
        4 => (prop::sample::select(vec![(1u32, 0i32), (5, 0), (5, 0x400)]), reg(), reg(), 0i32..64)
            .prop_map(|((f3, tag), rd, rs1, sh)| i(0x13, tag | sh, rs1, f3, rd)),
        4 => (reg(), reg(), imm12()).prop_map(|(rd, rs1, imm)| i(0x1B, imm, rs1, 0, rd)),
        // slliw, srliw, sraiw
        4 => (prop::sample::select(vec![(1u32, 0i32), (5, 0), (5, 0x400)]), reg(), reg(), 0i32..32)
            .prop_map(|((f3, tag), rd, rs1, sh)| i(0x1B, tag | sh, rs1, f3, rd)),
        16 => (0u32..7, reg(), reg(), imm12()).prop_map(|(f3, rd, rs1, imm)| i(0x03, imm, rs1, f3, rd)),
        12 => (0u32..4, reg(), reg(), imm12()).prop_map(|(f3, rs2, rs1, imm)| s(imm, rs2, rs1, f3)),
        8 => (prop::sample::select(vec![0u32, 1, 4, 5, 6, 7]), reg(), reg(), offset.clone())
            .prop_map(|(f3, rs1, rs2, off)| b(off, rs2, rs1, f3)),
        2 => (reg(), offset).prop_map(|(rd, off)| j(off, rd)),
        1 => (reg(), reg(), imm12()).prop_map(|(rd, rs1, imm)| i(0x67, imm, rs1, 0, rd)),
        3 => (reg(), any::<u32>()).prop_map(|(rd, imm)| (imm & 0xFFFF_F000) | rd << 7 | 0x37),
        3 => (reg(), any::<u32>()).prop_map(|(rd, imm)| (imm & 0xFFFF_F000) | rd << 7 | 0x17),
        1 => Just(0x0000_000Fu32),
        // csrrw / csrrs
        1 => (prop::sample::select(vec![1u32, 2]), reg(), reg(), 0i32..4096)
            .prop_map(|(f3, rd, rs1, csr)| i(0x73, csr, rs1, f3, rd)),
    ]
}

/// A program: its registers loaded from the data segment, then `body`, then
/// a halt; the code at `code_base`.
fn program(code_base: u64, regs: &[u64], extra: &[u8], body: &[u32]) -> Elf {
    let mut code: Vec<u32> = (1..32).map(|rd| ld_data(rd, rd as u64 - 1)).collect();
    code.extend_from_slice(body);
    code.push(i(0x13, 93, 0, 0, 17)); // addi a7, x0, 93
    code.push(0x73); // ecall: halt
    let mut data: Vec<u32> = regs
        .iter()
        .flat_map(|&v| [v as u32, (v >> 32) as u32])
        .collect();
    data.extend(
        extra
            .chunks(4)
            .map(|c| c.iter().rev().fold(0u32, |w, &b| w << 8 | b as u32)),
    );
    Elf {
        entry_point: code_base,
        data: vec![
            Segment {
                base_addr: DATA_BASE,
                values: data,
                is_executable: false,
            },
            Segment {
                base_addr: code_base,
                values: code,
                is_executable: true,
            },
        ],
    }
}

fn record(log: &Log) -> [u64; 5] {
    [
        log.current_pc,
        log.next_pc,
        log.src1_val,
        log.src2_val,
        log.dst_val,
    ]
}

fn error(e: &ExecutorError) -> String {
    format!("{e:?}")
}

/// What a window of either engine came to.
type Window = Result<Option<Vec<[u64; 5]>>, String>;

/// Runs `program` on the reference and on the fast engine with `layout`,
/// window by window (sizes cycling through `windows`, at most `cycles` in
/// all), and checks every window's logs and error, and the pc, registers
/// and published memory after it, are the reference's. Every other window
/// is taken with `resume_owned`.
pub(crate) fn same_run(
    program: &Elf,
    input: Vec<u8>,
    layout: FlatLayout,
    windows: &[usize],
    cycles: usize,
) -> Result<(), TestCaseError> {
    let reference = Executor::new(program, input.clone());
    let fast = FastExecutor::with_layout(program, input, layout);
    let (mut reference, mut fast) = match (reference, fast) {
        (Ok(r), Ok(f)) => (r, f),
        (Err(r), Err(f)) => {
            prop_assert_eq!(error(&r), error(&f));
            return Ok(());
        }
        (r, f) => {
            return Err(TestCaseError::fail(format!(
                "setup: reference {:?}, fast {:?}",
                r.err().map(|e| error(&e)),
                f.err().map(|e| error(&e))
            )));
        }
    };
    let first = fast.snapshot();
    let first_digest = first.content_digest();
    prop_assert_eq!(reference.memory().snapshot().content_digest(), first_digest);
    let mut done = 0;
    for (n, &window) in windows.iter().cycle().enumerate() {
        if done >= cycles {
            break;
        }
        let (want, got): (Window, Window) = if n % 2 == 0 {
            (
                reference
                    .resume_with_limit(window)
                    .map(|l| l.map(|l| l.iter().map(record).collect()))
                    .map_err(|e| error(&e)),
                fast.resume_with_limit(window)
                    .map(|l| l.map(|l| l.iter().map(record).collect()))
                    .map_err(|e| error(&e)),
            )
        } else {
            (
                reference
                    .resume_owned(window)
                    .map(|l| l.map(|l| l.iter().map(record).collect()))
                    .map_err(|e| error(&e)),
                fast.resume_owned(window)
                    .map(|l| l.map(|l| l.iter().map(record).collect()))
                    .map_err(|e| error(&e)),
            )
        };
        prop_assert_eq!(&want, &got, "window {} at cycle {}", n, done);
        prop_assert_eq!(reference.pc(), fast.pc());
        prop_assert_eq!(
            reference.registers().to_array(),
            fast.registers().to_array()
        );
        prop_assert_eq!(
            reference.memory().snapshot().content_digest(),
            fast.snapshot().content_digest(),
            "memory after window {}",
            n
        );
        match want {
            Ok(Some(logs)) => done += logs.len(),
            _ => break,
        }
    }
    // A published snapshot keeps reading its moment.
    prop_assert_eq!(first.content_digest(), first_digest);
    prop_assert_eq!(
        GuestMemory::public_output(reference.memory()),
        fast.memory().public_output_bytes()
    );
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    // Random programs over every op, register and immediate edge: x0 as a
    // source and a destination, the W ops, division by zero and overflow,
    // loads and stores on and across the flat regions' edges, jumps and
    // branches to misaligned or undecoded addresses, ecalls with whatever
    // the registers hold.
    #[test]
    fn random_programs_run_alike(
        regs in prop::collection::vec(value(), 31),
        // x17 never starts as print, panic or Keccak-256 (see `RANDOM_ECALLS`).
        a7 in prop::sample::select(RANDOM_ECALLS.to_vec()),
        extra in prop::collection::vec(any::<u8>(), 0..256),
        body in prop::collection::vec(word(), 1..80).prop_map(|w| w.concat()),
        code in prop::sample::select(vec![0x1000u64, 0x4_0000, (1 << 20) - 0x100, 0xFF00_0000]),
        windows in prop::collection::vec(1usize..64, 1..4),
    ) {
        let mut regs = regs;
        regs[16] = a7 as i64 as u64;
        for layout in LAYOUTS {
            let elf = program(code, &regs, &extra, &body);
            same_run(&elf, vec![], layout, &windows, 4_000)?;
        }
    }

    // The ecalls, each with operands drawn from where they are accepted and
    // where they are refused: unaligned, overlapping, past a 2^32 limb or the
    // end of the space, across the flat regions' edges, over-long.
    #[test]
    fn ecalls_run_alike(
        calls in prop::collection::vec(
            (
                prop::sample::select(SYSCALLS.to_vec()),
                prop::collection::vec(ecall_operand(), 4),
            ),
            1..5,
        ),
        extra in prop::collection::vec(any::<u8>(), 0..512),
        input in prop::collection::vec(any::<u8>(), 0..64),
    ) {
        // Each call: x10..x13 and x17 loaded from its data, then ecall.
        let mut regs = Vec::new();
        let mut body = Vec::new();
        for (number, operands) in &calls {
            let mut operands = operands.clone();
            // Print, panic and Keccak-256 read a buffer of x11 bytes into a
            // host allocation: at most 4 KiB here.
            if [1, 2, KECCAK256_SYSCALL_NUMBER].contains(number) {
                operands[1] %= 4097;
            }
            let operands = &operands;
            let k = 31 + regs.len() as u64;
            for (j, rd) in [10u32, 11, 12, 13].into_iter().enumerate() {
                body.push(ld_data(rd, k + j as u64));
            }
            body.push(ld_data(17, k + 4));
            body.push(0x73);
            regs.extend_from_slice(operands);
            regs.push(*number);
        }
        // The first 31 values are the prologue's registers: x1 = 1 for the
        // commit fd, the rest small numbers.
        let mut all: Vec<u64> = (1..32).map(|r| if r == 1 { 1 } else { r }).collect();
        all.extend(regs);
        for layout in LAYOUTS {
            let elf = program(0x1000, &all, &extra, &body);
            same_run(&elf, input.clone(), layout, &[1, 2, 3], 1_000)?;
        }
    }
}

/// An ecall operand: an address near an edge (aligned or not), a buffer in
/// the data segment, a length or a count, a selector, a flag word.
fn ecall_operand() -> impl Strategy<Value = u64> {
    let data_end = DATA_BASE + 8 * 64;
    prop_oneof![
        3 => value(),
        3 => (DATA_BASE..data_end + 512).prop_map(|a| a & !7),
        2 => DATA_BASE..data_end + 512,
        2 => 0u64..300,
        1 => 0u64..4,
        1 => Just(1u64),
        1 => (0..LAYOUTS.len(), 0u64..400).prop_map(|(l, d)| LAYOUTS[l].low_end - d),
        1 => (0..LAYOUTS.len(), 0u64..400).prop_map(|(l, d)| LAYOUTS[l].high_base.wrapping_sub(200) + d),
    ]
}

/// One access on any guest memory: a load's value, or `None` for a store;
/// an access past the end of the space is its error.
fn access(
    memory: &mut impl GuestMemory,
    store: bool,
    width: u8,
    at: u64,
    value: u64,
) -> Result<Option<u64>, String> {
    let e = |e: MemoryError| e.to_string();
    Ok(if store {
        match width {
            1 => memory.store_byte(at, value as u8),
            2 => memory.store_half(at, value as u16).map_err(e)?,
            4 => memory.store_word(at, value as u32).map_err(e)?,
            _ => memory.store_doubleword(at, value).map_err(e)?,
        }
        None
    } else {
        Some(match width {
            1 => memory.load_byte(at) as u64,
            2 => memory.load_half(at).map_err(e)? as u64,
            4 => memory.load_word(at).map_err(e)? as u64,
            _ => memory.load_doubleword(at).map_err(e)?,
        })
    })
}

proptest! {
    // The flat memory answers every access as the paged memory does, and
    // publishes the same bytes at every snapshot, while earlier snapshots
    // keep their moment: on and across both regions' edges, chunk and page
    // edges, and the top of the space; ranges too (as the DMA ecalls and
    // commits move them).
    #[test]
    fn flat_memory_reads_and_writes_as_the_paged_one(
        steps in prop::collection::vec(
            (0u8..6, prop::sample::select(vec![1u8, 2, 4, 8]), value(), any::<u64>(), 0usize..300),
            1..300,
        ),
    ) {
        let mut reference = Memory::with_store(StoreKind::Pages);
        let mut flat = FlatMemory::new(SMALL).unwrap();
        let mut kept = Vec::new();
        for (kind, width, at, value, len) in steps {
            match kind {
                0 | 1 => prop_assert_eq!(
                    access(&mut reference, kind == 1, width, at, value),
                    access(&mut flat, kind == 1, width, at, value),
                    "{} {} at {:#x}", kind, width, at
                ),
                2 => {
                    // A range that does not wrap.
                    let at = at.min(u64::MAX - len as u64);
                    let bytes: Vec<u8> = (0..len).map(|i| (value >> (i % 8 * 8)) as u8 ^ i as u8).collect();
                    GuestMemory::write_range(&mut reference, at, &bytes);
                    GuestMemory::write_range(&mut flat, at, &bytes);
                }
                3 => {
                    let at = at.min(u64::MAX - len as u64);
                    let (mut a, mut b) = (vec![0u8; len], vec![0u8; len]);
                    GuestMemory::read_range(&reference, at, &mut a);
                    GuestMemory::read_range(&flat, at, &mut b);
                    prop_assert_eq!(a, b, "range at {:#x}", at);
                }
                _ => {
                    let (r, f) = (reference.snapshot(), flat.snapshot());
                    prop_assert_eq!(r.content_digest(), f.content_digest());
                    let digest = f.content_digest();
                    kept.push((f, digest));
                }
            }
        }
        for (snapshot, digest) in &kept {
            prop_assert_eq!(snapshot.content_digest(), *digest);
        }
        prop_assert_eq!(reference.snapshot().content_digest(), flat.snapshot().content_digest());
    }
}

// A program the decoder refuses is refused by both engines alike, before any
// cycle runs.
#[test]
fn an_undecodable_program_is_refused_alike() {
    let elf = program(0x1000, &[0; 31], &[], &[0xFFFF_FFFF]);
    let reference = Executor::new(&elf, vec![]).err().map(|e| error(&e));
    let fast = FastExecutor::new(&elf, vec![]).err().map(|e| error(&e));
    assert!(reference.is_some());
    assert_eq!(reference, fast);
}

// The engine knob: unset or empty is the reference, `fast` the fast engine,
// anything else refused.
#[test]
fn the_engine_knob_names_an_engine() {
    assert_eq!(EngineKind::from_value(None), Ok(EngineKind::Reference));
    assert_eq!(EngineKind::from_value(Some("")), Ok(EngineKind::Reference));
    assert_eq!(
        EngineKind::from_value(Some("reference")),
        Ok(EngineKind::Reference)
    );
    assert_eq!(EngineKind::from_value(Some(" fast ")), Ok(EngineKind::Fast));
    assert!(EngineKind::from_value(Some("tight")).is_err());
    for kind in [EngineKind::Reference, EngineKind::Fast] {
        assert_eq!(EngineKind::from_value(Some(kind.name())), Ok(kind));
    }
}

// Either engine through [`Engine`]: the same logs, state and memory.
#[test]
fn the_engine_wrapper_runs_either_engine_alike() {
    // A loop that stores 0..40 in an array and loads the last one back.
    let body = [
        i(0x13, 0, 0, 0, 5),   // addi t0, x0, 0
        i(0x13, 40, 0, 0, 6),  // addi t1, x0, 40
        s(0, 5, 3, 3),         // sd t0, 0(gp)
        i(0x13, 8, 3, 0, 3),   // addi gp, gp, 8
        i(0x13, 1, 5, 0, 5),   // addi t0, t0, 1
        b(-12, 6, 5, 4),       // blt t0, t1, -12
        i(0x03, -8, 3, 3, 10), // ld a0, -8(gp)
    ];
    let mut regs = vec![0u64; 31];
    regs[2] = 0x2_0000; // gp
    let elf = program(0x1000, &regs, &[], &body);
    let mut runs: Vec<_> = [EngineKind::Reference, EngineKind::Fast]
        .into_iter()
        .map(|kind| Engine::new(kind, &elf, vec![]).unwrap())
        .collect();
    loop {
        let windows: Vec<_> = runs
            .iter_mut()
            .map(|e| {
                e.resume_owned(17)
                    .unwrap()
                    .map(|l| l.iter().map(record).collect::<Vec<_>>())
            })
            .collect();
        assert_eq!(windows[0], windows[1]);
        assert_eq!(runs[0].pc(), runs[1].pc());
        assert_eq!(
            runs[0].registers().to_array(),
            runs[1].registers().to_array()
        );
        let digests: Vec<_> = runs
            .iter_mut()
            .map(|e| e.snapshot().content_digest())
            .collect();
        assert_eq!(digests[0], digests[1]);
        if windows[0].is_none() {
            break;
        }
    }
    assert_eq!(runs[0].registers().to_array()[10], 39);
    let outputs: Vec<_> = runs
        .into_iter()
        .map(|e| e.finish().unwrap().register_values)
        .collect();
    assert_eq!(outputs[0], outputs[1]);
}

/// A program long enough for a few windows: a store-and-load loop.
fn looping_program() -> Elf {
    let body = [
        i(0x13, 0, 0, 0, 5),   // addi t0, x0, 0
        i(0x13, 500, 0, 0, 6), // addi t1, x0, 500
        s(0, 5, 3, 3),         // sd t0, 0(gp)
        i(0x03, 0, 3, 3, 7),   // ld t2, 0(gp)
        i(0x13, 8, 3, 0, 3),   // addi gp, gp, 8
        i(0x13, 1, 5, 0, 5),   // addi t0, t0, 1
        b(-16, 6, 5, 4),       // blt t0, t1, -16
    ];
    let mut regs = vec![0u64; 31];
    regs[2] = 0x2_0000; // gp
    program(0x1000, &regs, &[], &body)
}

/// Runs `elf` on the fast engine in windows of `window`, handing every
/// sampled window to `shadow` with its logs passed through `tamper`.
fn shadowed_run(
    elf: &Elf,
    window: usize,
    shadow: &mut crate::vm::shadow::ShadowCheck,
    mut tamper: impl FnMut(usize, &mut Vec<Log>, &mut [u64; 32]),
) -> usize {
    let mut fast = FastExecutor::new(elf, vec![]).unwrap();
    let (mut n, mut cycles) = (0, 0u64);
    loop {
        let start = crate::vm::shadow::WindowStart {
            window: n,
            first_cycle: cycles,
            pc: fast.pc(),
            registers: fast.registers(),
            memory: fast.snapshot(),
            public_output: fast.memory().public_output_bytes().to_vec(),
        };
        let Some(mut logs) = fast.resume_owned(window).unwrap() else {
            break;
        };
        cycles += logs.len() as u64;
        if shadow.samples(n) {
            let mut regs = fast.registers().to_array();
            tamper(n, &mut logs, &mut regs);
            shadow.submit(
                start,
                &logs,
                fast.pc(),
                crate::vm::registers::Registers::from_array(&regs),
            );
        }
        n += 1;
    }
    n
}

// The shadow check finds an honest run equal, window after window.
#[test]
fn the_shadow_check_passes_an_honest_run() {
    use crate::vm::shadow::ShadowCheck;
    let elf = looping_program();
    let mut shadow = ShadowCheck::new(&elf, 2).unwrap();
    let windows = shadowed_run(&elf, 300, &mut shadow, |_, _, _| {});
    assert!(windows >= 8, "{windows} windows");
    let counts = shadow.finish().unwrap();
    assert!(counts.checked >= 1);
    assert_eq!(counts.checked + counts.skipped, windows.div_ceil(2));
}

// The shadow check catches one value changed in one log, and a register
// changed after the window, naming the cycle.
#[test]
fn the_shadow_check_catches_a_changed_log_or_register() {
    use crate::vm::shadow::ShadowCheck;
    let elf = looping_program();
    for field in 0..6 {
        let mut shadow = ShadowCheck::new(&elf, 1_000).unwrap();
        // Sample window 0 only: the tampered one.
        shadowed_run(&elf, 300, &mut shadow, |n, logs, regs| {
            if n != 0 {
                return;
            }
            match field {
                0 => logs[17].current_pc ^= 4,
                1 => logs[17].next_pc ^= 1,
                2 => logs[17].src1_val ^= 1 << 40,
                3 => logs[17].src2_val ^= 1,
                4 => logs[17].dst_val ^= 1,
                _ => regs[7] ^= 1,
            }
        });
        let failure = shadow
            .finish()
            .expect_err("the check passed a changed window");
        let cycle = if field < 5 { "cycle 17)" } else { "cycle 300)" };
        assert!(failure.contains(cycle), "field {field}: {failure}");
    }
}

// The shadow knob: unset, empty or 0 is off, a count is a period, anything
// else refused.
#[test]
fn the_shadow_knob_names_a_period() {
    use crate::vm::shadow::ShadowCheck;
    assert_eq!(ShadowCheck::every_from_value(None), Ok(None));
    assert_eq!(ShadowCheck::every_from_value(Some("")), Ok(None));
    assert_eq!(ShadowCheck::every_from_value(Some("0")), Ok(None));
    assert_eq!(ShadowCheck::every_from_value(Some("16")), Ok(Some(16)));
    assert!(ShadowCheck::every_from_value(Some("often")).is_err());
}

// A window's buffer given back is written into again two windows later
// (the next window's buffer is set aside when a window is handed out), on
// both engines, and the logs are those of a run that never recycles.
#[test]
fn a_recycled_buffer_takes_the_next_window() {
    let elf = looping_program();
    let window = 300;
    for kind in [EngineKind::Reference, EngineKind::Fast] {
        let mut plain = Engine::new(kind, &elf, vec![]).unwrap();
        let mut recycling = Engine::new(kind, &elf, vec![]).unwrap();
        let (mut windows, mut reused) = (0, 0);
        let mut given_back: std::collections::VecDeque<*const Log> = Default::default();
        loop {
            let want = plain.resume_owned(window).unwrap();
            let got = recycling.resume_owned(window).unwrap();
            let (Some(want), Some(got)) = (want, got) else {
                break;
            };
            windows += 1;
            assert_eq!(
                want.iter().map(record).collect::<Vec<_>>(),
                got.iter().map(record).collect::<Vec<_>>(),
                "{kind:?} window {windows}"
            );
            if given_back.len() == 2 && given_back.pop_front() == Some(got.as_ptr()) {
                reused += 1;
            }
            if got.len() == window {
                given_back.push_back(got.as_ptr());
                recycling.recycle(got);
            }
        }
        assert!(windows >= 8, "{kind:?}: {windows} windows");
        // Every window from the third on but the last (shorter, its buffer
        // new) is written into a buffer given back.
        assert!(
            reused >= windows - 3,
            "{kind:?}: {reused} of {windows} windows reused"
        );
    }
}
