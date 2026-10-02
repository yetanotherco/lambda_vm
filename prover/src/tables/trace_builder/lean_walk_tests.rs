//! The lean walk's parts against the walk they replace (D-EXEC E1). The whole
//! builds are compared in `tests::windowed_builder_tests`; these pin each part.

use std::collections::HashMap;

use executor::vm::execution::Executor;

use super::*;
use crate::test_utils::asm_elf_bytes;

const PAGE: u64 = page::DEFAULT_PAGE_SIZE as u64;

/// A fixed xorshift stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// The decode table answers exactly the pcs the instruction map holds, each
/// with the decode `from_log_and_instruction` makes: across runs with gaps,
/// misaligned pcs and a run that ends at the top of the address space.
#[test]
fn the_decode_table_holds_exactly_the_maps_pcs() {
    // nop · addi x1, x0, 10 · add x3, x1, x2 · ecall
    let words = [0x0000_0013u32, 0x00a0_0093, 0x0020_81b3, 0x0000_0073];
    let mut map = U64HashMap::default();
    for (first, n) in [(0x1000u64, 10usize), (0x2000, 5), (u64::MAX - 7, 2)] {
        for k in 0..n {
            let instruction = Instruction::parse(words[k % words.len()]).expect("a valid word");
            map.insert(first + 4 * k as u64, instruction);
        }
    }
    let table = DecodeTable::from_instructions(&map);
    assert_eq!(table.runs.len(), 3, "three runs of consecutive pcs");
    for (&pc, &instruction) in map.iter() {
        assert_eq!(
            table.get(pc),
            Some(&DecodeEntry::from_instruction(pc, instruction, 4)),
            "pc {pc:#x}"
        );
    }
    for pc in [
        0,
        0xffc,
        0x1002,
        0x1028,
        0x1ffc,
        0x2014,
        u64::MAX - 11,
        u64::MAX - 1,
        u64::MAX,
    ] {
        assert!(table.get(pc).is_none(), "pc {pc:#x} is not in the map");
    }
}

/// A plain and a lean memory state fed the same accesses.
struct Twin {
    plain: MemoryState,
    lean: MemoryState,
}

impl Twin {
    fn new() -> Self {
        let mut image: HashMap<u64, u8> = HashMap::new();
        for (k, addr) in [0, 7, PAGE - 1, PAGE, 3 * PAGE - 2, 12 * PAGE + 5, u64::MAX]
            .into_iter()
            .enumerate()
        {
            image.insert(addr, 0x10 + k as u8);
        }
        let plain = MemoryState::from_image(&image);
        let mut lean = MemoryState::from_image(&image);
        lean.lean = true;
        Self { plain, lean }
    }

    fn read(&mut self, addr: u64, count: usize) {
        assert_eq!(
            self.plain.read_bytes(addr, count),
            self.lean.read_bytes(addr, count),
            "read of {count} at {addr:#x}"
        );
    }

    fn write(&mut self, addr: u64, value: u64, count: usize, ts: u64) {
        self.plain.write_bytes(addr, value, count, ts);
        self.lean.write_bytes(addr, value, count, ts);
    }

    fn write_byte(&mut self, addr: u64, value: u8, ts: u64) {
        self.plain.write_byte(addr, value, ts);
        self.lean.write_byte(addr, value, ts);
    }

    /// The same cells set, the same pages allocated.
    fn same_cells(&self) {
        let cells = |m: &MemoryState| m.cells.iter().collect::<Vec<_>>();
        assert!(cells(&self.plain) == cells(&self.lean), "cells differ");
        assert_eq!(
            self.plain.cells.page_bases().collect::<Vec<_>>(),
            self.lean.cells.page_bases().collect::<Vec<_>>()
        );
    }
}

/// Lean memory accesses read and write what the plain ones do: an access
/// straddling a page, a first touch between two allocated pages (which moves
/// the upper page's index), accesses at the top of the address space and
/// wrapping past it, a store and a load of one byte with another page found in
/// between, single-byte writes among them, and a long random mix.
#[test]
fn lean_memory_accesses_are_the_plain_ones() {
    let mut t = Twin::new();
    t.read(3 * PAGE - 3, 8);
    t.write(3 * PAGE - 3, 0x0102_0304_0506_0708, 8, 10);
    t.read(3 * PAGE - 3, 8);
    t.read(3 * PAGE - 8, 8);

    t.write(10 * PAGE + 8, 0xaa, 1, 11);
    t.write(12 * PAGE, 0xbb, 2, 12);
    t.read(12 * PAGE, 8);
    t.write(11 * PAGE + 100, 0xcc, 4, 13);
    t.read(12 * PAGE, 8);
    t.read(10 * PAGE + 8, 8);
    t.read(11 * PAGE + 96, 8);

    t.write(u64::MAX - 7, u64::MAX, 8, 14);
    t.read(u64::MAX - 7, 8);
    t.write(u64::MAX - 3, 0x1122_3344_5566_7788, 8, 15);
    t.read(u64::MAX - 3, 8);
    t.read(0, 8);

    t.write(5 * PAGE + 1, 0x77, 1, 16);
    t.read(9 * PAGE, 4);
    t.read(5 * PAGE + 1, 1);
    t.read(5 * PAGE, 8);

    t.write_byte(20 * PAGE - 1, 0x55, 17);
    t.read(20 * PAGE - 4, 8);
    t.write_byte(4 * PAGE + 3, 0x66, 18);
    t.read(4 * PAGE, 8);
    t.write(4 * PAGE, 0, 0, 19);
    t.read(4 * PAGE, 8);
    t.same_cells();

    let spots = [
        0,
        PAGE - 4,
        PAGE,
        2 * PAGE - 1,
        7 * PAGE + 12,
        30 * PAGE,
        31 * PAGE - 6,
        0x8000_0000,
        u64::MAX - 5,
    ];
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for ts in 100..40_000u64 {
        let r = rng.next();
        let addr = spots[(r % spots.len() as u64) as usize]
            .wrapping_add((r >> 8) % 16)
            .wrapping_sub(8);
        let count = [0, 1, 2, 4, 8][((r >> 16) % 5) as usize];
        match (r >> 20) % 3 {
            0 => t.read(addr, count),
            1 => t.write(addr, r >> 24, count, ts),
            _ => t.write_byte(addr, (r >> 24) as u8, ts),
        }
    }
    t.same_cells();
}

/// `LAMBDA_VM_WALK_LEAN`'s values: every part, none, a list of parts; anything
/// else is refused.
#[test]
fn the_walk_lean_knob_names_its_parts() {
    assert_eq!(WalkLean::parse("1"), Some(WalkLean::ALL));
    assert_eq!(WalkLean::parse(" 0 "), Some(WalkLean::NONE));
    assert_eq!(
        WalkLean::parse("decode,memory,lookups,route"),
        Some(WalkLean::ALL)
    );
    assert_eq!(
        WalkLean::parse("memory, route"),
        Some(WalkLean {
            memory: true,
            route: true,
            ..WalkLean::NONE
        })
    );
    for bad in ["", "yes", "2", "decode,", "decode;memory", "Decode"] {
        assert_eq!(WalkLean::parse(bad), None, "{bad:?}");
    }
}

/// BITWISE's multiplicities from `histogram`, row by row.
fn multiplicities(histogram: &bitwise::BitwiseHistogram) -> Vec<Vec<u64>> {
    let mut trace = bitwise::generate_bitwise_trace();
    histogram.fill_multiplicities(&mut trace);
    (0..trace.main_table.height)
        .map(|r| {
            trace
                .main_table
                .get_row(r)
                .iter()
                .map(|v| v.canonical())
                .collect()
        })
        .collect()
}

/// The lookups a lean walk leaves out, counted from its CPU and LOAD ops, are
/// the lookups the plain walk lists; and the LOAD ops' part carries weight:
/// without it the counts differ on programs with narrow loads.
#[test]
fn the_counted_lookups_are_the_listed_ones() {
    let mut narrow = 0;
    for name in ["lw_sw_offset_odd", "test_memw_split_ts", "test_keccak"] {
        let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
        let logs = Executor::new(&program, Vec::new())
            .expect("the executor starts")
            .run()
            .expect("the program runs")
            .logs;
        let artifacts = DecodeArtifacts::from_elf(&program).expect("the decode artifacts");
        let cpu_ops = collect_cpu_ops(&logs, &artifacts.instructions).expect("the CPU ops");
        let image = build_initial_image(&program, &[]);
        let register_init = register::register_init_from_entry_point(program.entry_point);
        let walk = |lookups: bool| {
            let mut memory_state = MemoryState::from_image(&image);
            memory_state.lean = !lookups;
            let mut register_state = RegisterState::from_init(&register_init);
            let mut out = WalkOutputs::for_walk(cpu_ops.len(), lookups);
            collect_ops_from_cpu_into(
                &cpu_ops,
                &mut memory_state,
                &mut register_state,
                &mut out,
                lookups,
            );
            out
        };
        let (listed, lean) = (walk(true), walk(false));
        assert!(
            lean.bitwise_ops.is_empty(),
            "{name}: the lean walk lists no lookup"
        );
        assert!(
            !listed.bitwise_ops.is_empty(),
            "{name}: the plain walk lists them"
        );

        let mut from_list = bitwise::BitwiseHistogram::new();
        from_list.add_ops(&listed.bitwise_ops);
        let mut counted = bitwise::BitwiseHistogram::new();
        cpu_ops
            .iter()
            .for_each(|op| op.count_bitwise_into(&mut counted));
        let cpu_only = multiplicities(&counted);
        lean.load_ops
            .iter()
            .for_each(|op| op.count_bitwise_into(&mut counted));
        let expected = multiplicities(&from_list);
        assert!(
            multiplicities(&counted) == expected,
            "{name}: counted ≠ listed"
        );
        if lean.load_ops.iter().any(|op| op.width < 8) {
            narrow += 1;
            assert!(
                cpu_only != expected,
                "{name}: the LOAD lookups made no difference"
            );
        }
    }
    assert!(narrow > 0, "no program with a narrow load");
}

/// The one-pass routing builds every segment the per-segment filters build.
#[test]
fn the_one_pass_routing_is_the_routing() {
    for name in [
        "all_instructions_64",
        "lw_sw_offset_odd",
        "test_keccak_multi",
        "mulh_max",
    ] {
        let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
        let logs = Executor::new(&program, Vec::new())
            .expect("the executor starts")
            .run()
            .expect("the program runs")
            .logs;
        let artifacts = DecodeArtifacts::from_elf(&program).expect("the decode artifacts");
        let cpu_ops = collect_cpu_ops(&logs, &artifacts.instructions).expect("the CPU ops");
        let image = build_initial_image(&program, &[]);
        let register_init = register::register_init_from_entry_point(program.entry_point);
        let mut memory_state = MemoryState::from_image(&image);
        let mut register_state = RegisterState::from_init(&register_init);
        let mut walk = WalkOutputs::with_capacity(cpu_ops.len());
        collect_ops_from_cpu_into(
            &cpu_ops,
            &mut memory_state,
            &mut register_state,
            &mut walk,
            true,
        );
        let show = |s: RoutedSegments| {
            let RoutedSegments {
                branch_ops,
                mul_filter,
                dvrm_filter,
                eq_ops,
                bytewise_ops,
                store_ops,
                shift_cpu32,
                mul_cpu32,
                dvrm_cpu32,
                bitwise_cpu32,
                lt_dvrm_filter,
                lt_dvrm_cpu32,
                mul_dvrm_filter,
                mul_dvrm_cpu32,
            } = s;
            format!(
                "{branch_ops:?}{mul_filter:?}{dvrm_filter:?}{eq_ops:?}{bytewise_ops:?}{store_ops:?}\
                 {shift_cpu32:?}{mul_cpu32:?}{dvrm_cpu32:?}{bitwise_cpu32:?}{lt_dvrm_filter:?}\
                 {lt_dvrm_cpu32:?}{mul_dvrm_filter:?}{mul_dvrm_cpu32:?}"
            )
        };
        assert_eq!(
            show(route_ops(&cpu_ops, &walk.cpu32_ops)),
            show(route_ops_one_pass(&cpu_ops, &walk.cpu32_ops)),
            "{name}"
        );
    }
}

/// The decode table's row of each pc is the DECODE trace's
/// ([`decode::generate_decode_trace`]'s `pc_to_row`), so counting DECODE's
/// multiplicities through it counts the same rows; a pc outside the map has
/// none.
#[test]
fn the_decode_tables_rows_are_the_decode_traces() {
    for name in [
        "all_instructions_64",
        "test_keccak_multi",
        "lw_sw_offset_odd",
    ] {
        let program = Elf::load(&asm_elf_bytes(name)).expect("the ELF loads");
        let instructions = decode::instructions_from_elf(&program).expect("the instructions");
        let (_, pc_to_row) = decode::generate_decode_trace(&instructions);
        let table = DecodeTable::from_instructions(&instructions);
        for &pc in instructions.keys() {
            assert_eq!(
                table.row(pc),
                pc_to_row.get(&pc).copied(),
                "{name}: pc {pc:#x}"
            );
        }
        assert_eq!(table.row(super::cpu::CPU_PADDING_PC), None);
        assert_eq!(table.row(u64::MAX), None);
    }
}
