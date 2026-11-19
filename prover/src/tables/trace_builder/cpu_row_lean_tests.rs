//! The lean CPU row ([`CpuRowForm::Lean`]) on the windowed builder's
//! traces, with no proof: every bus the tables close among themselves balances
//! under it, DECODE's included, and each check it rests on refuses the forgery
//! it is there for, beside a control mutant without that check that lets the
//! forgery through.
//!
//! The forgeries are the decode-byte malleability of the packed DECODE element
//! (a byte moved into another field's offset keeps the packed sum), the
//! ECALL's tag on the merged memory dispatch, the syscall number its ADD flag
//! pins, and the constraint the merge adds.

use std::collections::{BTreeMap, HashMap};

use executor::elf::Elf;
use executor::vm::logs::Log;
use stark::constraints::builder::ConstraintSet;
use stark::lookup::{BusInteraction, BusValue, LinearTerm};
use stark::proof::options::ProofOptions;
use stark::trace::TraceTable;
use stark::traits::AIR;

use super::Traces;
use super::bus_net_tests::multiplicity;
use super::finish_tests::{run, windowed};
use super::in_row_forgery_tests::with_verifier_files;
use crate::VmAirs;
use crate::tables::MaxRowsConfig;
use crate::tables::cpu::{self, CpuRowForm, ECALL_MEMORY_OP_TAG, cols as cpu_cols};
use crate::tables::cpu32::cols as c32;
use crate::tables::fused_memory::MemoryFormat;
use crate::tables::memw_aligned::MemwAlignedOrder;
use crate::tables::register::RegisterFormat;
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

type Table = TraceTable<GoldilocksField, GoldilocksExtension>;

/// The programs: every instruction class (word instructions among them), the
/// loads and stores, and a call into each syscall chip the word memory format
/// proves (the halt, COPY's, keccak256's, ECREC's).
const PROGRAMS: [&str; 9] = [
    "all_instructions_64",
    "all_loadstore_32",
    "test_wsuffix_64bit",
    "test_addw_8",
    "test_copy_ecalls",
    "test_commit_4",
    "test_keccak256",
    "test_ecrec",
    "fused_mem_cases",
];

/// `program`'s run built under the word memory format and in-row registers
/// (the block's), its CPU row in `form`.
fn build(program: &Elf, logs: &[Log], form: CpuRowForm) -> Traces {
    build_in(program, logs, form, MemoryFormat::Words)
}

/// [`build`] under the memory format `memory`.
fn build_in(program: &Elf, logs: &[Log], form: CpuRowForm, memory: MemoryFormat) -> Traces {
    let keccak_rnd = crate::block_whir::BlockFormat::production().keccak_rnd;
    let mut traces = windowed(program, logs, &MaxRowsConfig::small(), 1000, &|b| {
        b.register_format(RegisterFormat::InRow)
            .expect("before any window")
            .memw_aligned_order(MemwAlignedOrder::Limbs)
            .expect("before any window")
            .keccak_rnd_form(keccak_rnd)
            .memory_format(memory)
            .expect("before any window")
            .cpu_row_form(form)
            .expect("before any window")
    });
    with_verifier_files(program, &mut traces);
    traces
}

/// The block's AIR set for `traces` under `form`, as the verifier builds it.
fn airs(program: &Elf, traces: &Traces, form: CpuRowForm) -> VmAirs {
    airs_in(program, traces, form, MemoryFormat::Words)
}

/// [`airs`] under the memory format `memory`.
fn airs_in(program: &Elf, traces: &Traces, form: CpuRowForm, memory: MemoryFormat) -> VmAirs {
    let options = ProofOptions::default_test_options();
    let inputs = crate::BlockAirInputs::new(
        &options,
        program,
        crate::cpu_heights_from_traces(traces),
        &traces.public_output_bytes,
    )
    .expect("the block's inputs");
    let format = crate::block_whir::BlockFormat::production()
        .with_memory(memory)
        .with_cpu_row(form);
    let mut airs = crate::block_airs(
        program,
        &options,
        &traces.page_configs,
        &traces.table_counts(),
        &inputs,
        format.keccak_rnd,
        format.memw_aligned,
        format.registers,
    );
    airs.with_fused_memory(&options, traces.fused_counts());
    if memory == MemoryFormat::Words {
        airs.with_word_memory(&options, traces.word_counts(), &traces.page_configs)
            .expect("the word format's set");
    }
    airs.with_cpu_row(&options, form, format.registers, format.memory, &inputs)
        .expect("the lean row's set");
    airs
}

/// The buses the statement closes from outside the tables: memory's first and
/// last values. The committed output is closed here as the verifier closes
/// it; DECODE's bus is an inner one (its multiplicities are the prover's).
const OUTSIDE: [BusId; 2] = [BusId::Memory, BusId::GlobalMemory];

/// Each bus's tuples left over, a table's interactions taken from `over` when
/// it names them (by its AIR name).
type Left = BTreeMap<u64, Vec<(Vec<u64>, i128)>>;

/// A test's interactions for the table it names, `None` for its AIR's.
type Over<'a> = &'a dyn Fn(&str, &[BusInteraction]) -> Option<Vec<BusInteraction>>;

/// One forgery: what it is, the program, and the edit.
type Case<'a> = (&'a str, &'a str, fn(&mut Traces));

fn left_with(airs: &VmAirs, traces: &mut Traces, over: Over<'_>) -> Left {
    let outside: Vec<u64> = OUTSIDE.iter().map(|b| u64::from(*b)).collect();
    let public = traces.public_output_bytes.clone();
    let mut net: HashMap<(u64, Vec<u64>), FE> = HashMap::new();
    for (air, trace, _) in airs.air_trace_pairs(traces) {
        let owned = over(air.name(), air.bus_interactions());
        let interactions: &[BusInteraction] = match &owned {
            Some(v) => v,
            None => air.bus_interactions(),
        };
        for row in 0..trace.num_rows() {
            let g = |c: usize| *trace.get_main(row, c);
            for bi in interactions
                .iter()
                .filter(|bi| !outside.contains(&bi.bus_id))
            {
                let m = multiplicity(&bi.multiplicity, &g);
                if m == FE::zero() {
                    continue;
                }
                let tuple: Vec<u64> = bi
                    .values
                    .iter()
                    .flat_map(|v| v.combine_from(g))
                    .map(|x: FE| x.canonical_u64())
                    .collect();
                let e = net.entry((bi.bus_id, tuple)).or_insert(FE::zero());
                *e = if bi.is_sender { *e + m } else { *e - m };
            }
        }
    }
    for (i, &byte) in public.iter().enumerate() {
        let e = net
            .entry((u64::from(BusId::Commit), vec![i as u64, u64::from(byte)]))
            .or_insert(FE::zero());
        *e = *e - FE::one();
    }
    const P: u64 = 0xFFFF_FFFF_0000_0001;
    let mut out: Left = BTreeMap::new();
    for ((bus, tuple), v) in net {
        if v == FE::zero() {
            continue;
        }
        let v = v.canonical_u64();
        let signed = if v > u64::MAX / 2 {
            -((P - v) as i128)
        } else {
            v as i128
        };
        out.entry(bus).or_default().push((tuple, signed));
    }
    for tuples in out.values_mut() {
        tuples.sort();
    }
    out
}

fn left(airs: &VmAirs, traces: &mut Traces) -> Left {
    left_with(airs, traces, &|_, _| None)
}

fn buses(left: &Left) -> Vec<&'static str> {
    left.keys()
        .map(|&bus| BusId::try_from(bus).map_or("?", |b| b.name()))
        .collect()
}

/// An interaction as text, to compare two.
fn show(i: &BusInteraction) -> String {
    format!(
        "{} {} {:?} {:?}",
        i.bus_id, i.is_sender, i.multiplicity, i.values
    )
}

fn get(t: &Table, row: usize, col: usize) -> u64 {
    t.get_main(row, col).canonical_u64()
}

fn set(t: &mut Table, row: usize, col: usize, value: FE) {
    assert!(t.narrow_main().is_none(), "the forgeries edit wide tables");
    t.main_table.set(row, col, value);
}

/// Completeness, and the control for every forgery below: under each form
/// every inner bus balances on each program's build — the lean row's DECODE
/// with its byte elements, BITWISE without the byte checks the lean rows no
/// longer send, `MemoryOp` with the ECALLs on it — and under the lean row
/// nothing is left on the ECALL bus.
#[test]
fn every_inner_bus_balances_under_both_rows() {
    // The word memory format proves the halt, COPY's, keccak256's and ECREC's
    // calls; the fused (the production format a base-only run takes without
    // a memory word) and the split formats every syscall chip's.
    const CHIPS: [&str; 5] = [
        "test_keccak",
        "test_ecsm",
        "test_blake3",
        "test_blake3_absorb",
        "test_private_input_spin",
    ];
    for memory in [
        MemoryFormat::Words,
        MemoryFormat::Fused,
        MemoryFormat::Split,
    ] {
        let programs = PROGRAMS.iter().chain(if memory == MemoryFormat::Words {
            &[][..]
        } else {
            &CHIPS[..]
        });
        for &name in programs {
            let (program, logs) = run(name);
            for form in [CpuRowForm::Full, CpuRowForm::Lean] {
                let mut traces = build_in(&program, &logs, form, memory);
                let airs = airs_in(&program, &traces, form, memory);
                let left = left(&airs, &mut traces);
                assert!(
                    left.is_empty(),
                    "{name} under {form:?}, {memory:?}: buses {:?} left over: {:?}",
                    buses(&left),
                    left.values().flatten().take(6).collect::<Vec<_>>()
                );
                if form == CpuRowForm::Lean {
                    for air in airs.air_refs() {
                        assert!(
                            air.bus_interactions()
                                .iter()
                                .all(|i| i.bus_id != BusId::Ecall as u64),
                            "{name}, {memory:?}: {} still uses the ECALL bus under the lean row",
                            air.name()
                        );
                    }
                }
            }
        }
    }
}

/// The lean DECODE trace is today's with the six bytes inserted before `MU`,
/// each the entry's, and the lean receiver's flags element is the ten flag
/// bits on every row.
#[test]
fn the_lean_decode_carries_each_entry_byte() {
    let (program, logs) = run("all_instructions_64");
    let full = build(&program, &logs, CpuRowForm::Full);
    let lean = build(&program, &logs, CpuRowForm::Lean);
    use crate::tables::decode::{cols, lean_cols};
    assert_eq!(full.decode.main_table.width, cols::NUM_COLUMNS);
    assert_eq!(lean.decode.main_table.width, lean_cols::NUM_COLUMNS);
    let receiver = &crate::tables::decode::bus_interactions_for(CpuRowForm::Lean)[0];
    use crate::tables::types::packed_decode_shrunk as pd;
    let mut ecalls = 0;
    for row in 0..full.decode.num_rows() {
        for col in 0..crate::tables::decode::NUM_PRECOMPUTED_COLS {
            if col == cols::PACKED_DECODE {
                continue;
            }
            assert_eq!(get(&full.decode, row, col), get(&lean.decode, row, col));
        }
        // An ECALL's entry sets ADD under the lean row, and no other moves.
        let today = get(&full.decode, row, cols::PACKED_DECODE);
        let ecall = (today >> pd::ECALL) & 1 == 1;
        ecalls += usize::from(ecall);
        let expect = if ecall { today | (1 << pd::ADD) } else { today };
        assert_eq!(
            get(&lean.decode, row, cols::PACKED_DECODE),
            expect,
            "row {row}"
        );
        assert_eq!(
            get(&full.decode, row, cols::MU),
            get(&lean.decode, row, lean_cols::MU)
        );
        let packed = expect;
        let fields = crate::tables::types::ShrunkDecode::unpack(packed);
        let bytes = [
            fields.rs1,
            fields.rs2,
            fields.rd,
            fields.half_instruction_length,
            fields.alu_flags,
            fields.mem_flags,
        ];
        for (k, byte) in bytes.into_iter().enumerate() {
            assert_eq!(get(&lean.decode, row, lean_cols::RS1 + k), u64::from(byte));
        }
        let g = |c: usize| *lean.decode.get_main(row, c);
        let tuple: Vec<u64> = receiver
            .values
            .iter()
            .flat_map(|v| v.combine_from(g))
            .map(|x: FE| x.canonical_u64())
            .collect();
        assert_eq!(tuple[4], packed & 0x3FF, "row {row}: the flags element");
        assert_eq!(&tuple[5..], &bytes.map(u64::from)[..]);
    }
    assert!(ecalls > 0, "the program makes an ECALL");
}

/// The interactions: 24 → 20 in-row, DECODE one tuple of 2 + 2 + 1 + 6
/// elements, CPU32 without its five byte checks, and every ECALL receiver of
/// every syscall chip (eleven sites in all) on `MemoryOp` with the tag.
#[test]
fn the_lean_row_sends_twenty_and_every_syscall_chip_takes_the_tag() {
    use crate::tables::cpu::{bus_interactions_inrow_for, receivers_under};
    assert_eq!(bus_interactions_inrow_for(0, CpuRowForm::Full).len(), 24);
    assert_eq!(bus_interactions_inrow_for(0, CpuRowForm::Lean).len(), 20);
    let lean = bus_interactions_inrow_for(0, CpuRowForm::Lean);
    let count = |bus: BusId| lean.iter().filter(|i| i.bus_id == bus as u64).count();
    assert_eq!(count(BusId::AreBytes), 0);
    assert_eq!(count(BusId::Ecall), 0);
    assert_eq!(count(BusId::MemoryOp), 1);
    assert_eq!(count(BusId::IsHalfword), 7, "four on res, three steps");
    let decode = |list: &[BusInteraction]| {
        list.iter()
            .find(|i| i.bus_id == BusId::Decode as u64)
            .map(|i| {
                i.values
                    .iter()
                    .map(BusValue::num_bus_elements)
                    .sum::<usize>()
            })
    };
    assert_eq!(decode(&lean), Some(11));
    assert_eq!(
        decode(&bus_interactions_inrow_for(0, CpuRowForm::Full)),
        Some(5)
    );
    let cpu32 = crate::tables::cpu32::bus_interactions_for(CpuRowForm::Lean);
    assert_eq!(
        crate::tables::cpu32::bus_interactions().len() - cpu32.len(),
        5,
        "CPU32's five byte checks"
    );
    assert_eq!(decode(&cpu32), Some(11));

    // Every syscall chip's ECALL receivers, in both forms.
    let families: [(&str, Vec<BusInteraction>); 11] = [
        ("HALT", crate::tables::halt::bus_interactions()),
        ("HALT~inrow", crate::tables::halt::bus_interactions_inrow()),
        ("COPY", crate::tables::copy::bus_interactions()),
        ("COPY_W", crate::tables::copy_w::bus_interactions()),
        ("KECCAK", crate::tables::keccak::bus_interactions()),
        ("KECCAK_W", crate::tables::keccak_w::bus_interactions()),
        ("ECREC", crate::tables::ecrec::bus_interactions()),
        ("ECREC_W", crate::tables::ecrec_w::bus_interactions()),
        ("ECSM", crate::tables::ecsm::bus_interactions()),
        ("HINT", crate::tables::hint::bus_interactions()),
        ("BLAKE3", crate::tables::blake3::bus_interactions()),
    ];
    let mut receivers = 0;
    for (name, full) in families {
        let ecall: Vec<&BusInteraction> = full
            .iter()
            .filter(|i| i.bus_id == BusId::Ecall as u64)
            .collect();
        assert!(!ecall.is_empty(), "{name} receives an ECALL");
        assert!(ecall.iter().all(|i| !i.is_sender), "{name}: receivers only");
        receivers += ecall.len();
        let moved = receivers_under(CpuRowForm::Lean, full.clone());
        assert_eq!(
            moved.len(),
            full.len(),
            "{name}: no interaction gained or lost"
        );
        for (before, after) in full.iter().zip(&moved) {
            if before.bus_id != BusId::Ecall as u64 {
                assert_eq!(show(before), show(after), "{name}");
                continue;
            }
            assert_eq!(after.bus_id, BusId::MemoryOp as u64, "{name}");
            assert!(!after.is_sender, "{name}");
            assert_eq!(
                format!("{:?}", before.multiplicity),
                format!("{:?}", after.multiplicity),
                "{name}: the multiplicity"
            );
            let v = |i: &BusInteraction, k: usize| format!("{:?}", i.values[k]);
            assert_eq!(after.values.len(), 9, "{name}");
            for (k, from) in [(0, 0), (1, 1), (2, 2), (3, 3), (7, 2), (8, 3)] {
                assert_eq!(v(after, k), v(before, from), "{name}: element {k}");
            }
            assert_eq!(v(after, 4), format!("{:?}", BusValue::constant(0)));
            assert_eq!(v(after, 5), format!("{:?}", BusValue::constant(0)));
            assert_eq!(
                v(after, 6),
                format!("{:?}", BusValue::constant(ECALL_MEMORY_OP_TAG))
            );
        }
        assert_eq!(
            receivers_under(CpuRowForm::Full, full.clone())
                .iter()
                .map(show)
                .collect::<Vec<_>>(),
            full.iter().map(show).collect::<Vec<_>>(),
            "{name}: today's row moves nothing"
        );
    }
    assert!(
        receivers >= 11,
        "the eleven receiver sites, {receivers} found"
    );
}

/// The tag: no memory receiver's flags element reaches 32 under any
/// assignment of its flag bits (LOAD, STORE, MEM_LOAD, MEM_STORE and the six
/// word families, so every memory format's), so a syscall chip's tagged
/// receive never matches a load or a store, nor a dispatch of a memory row a
/// syscall chip's. The control: without the tag a memory receiver's flags do
/// reach the ECALL's untagged 0 (an LBU), the collision the tag rules out.
#[test]
fn no_memory_receiver_reaches_the_ecall_tag() {
    let options = ProofOptions::default_test_options();
    use crate::test_utils as tu;
    let receivers: Vec<(&str, Vec<BusInteraction>)> = vec![
        (
            "LOAD",
            tu::create_load_air(&options).bus_interactions().to_vec(),
        ),
        (
            "STORE",
            tu::create_store_air(&options).bus_interactions().to_vec(),
        ),
        (
            "MEM_LOAD",
            tu::create_mem_load_air(&options)
                .bus_interactions()
                .to_vec(),
        ),
        (
            "MEM_STORE",
            tu::create_mem_store_air(&options)
                .bus_interactions()
                .to_vec(),
        ),
        (
            "MEM_LOAD8",
            tu::create_mem_load8_air(&options)
                .bus_interactions()
                .to_vec(),
        ),
        (
            "MEM_STORE8",
            tu::create_mem_store8_air(&options)
                .bus_interactions()
                .to_vec(),
        ),
        (
            "MEM_LOADW",
            tu::create_mem_loadw_air(&options)
                .bus_interactions()
                .to_vec(),
        ),
        (
            "MEM_STOREW",
            tu::create_mem_storew_air(&options)
                .bus_interactions()
                .to_vec(),
        ),
        (
            "MEM_LOAD2W",
            tu::create_mem_load2w_air(&options)
                .bus_interactions()
                .to_vec(),
        ),
        (
            "MEM_STORE2W",
            tu::create_mem_store2w_air(&options)
                .bus_interactions()
                .to_vec(),
        ),
    ];
    let mut zero_reachable = false;
    let mut seen = 0;
    for (name, interactions) in &receivers {
        for i in interactions
            .iter()
            .filter(|i| i.bus_id == BusId::MemoryOp as u64 && !i.is_sender)
        {
            seen += 1;
            // The flags element is the tuple's seventh: ts, ts_hi, the
            // address's two words, the value's two.
            let mut at = 0;
            let flags = i
                .values
                .iter()
                .find(|v| {
                    let here = at;
                    at += v.num_bus_elements();
                    here == 6
                })
                .unwrap_or_else(|| panic!("{name}: a seventh element"));
            let BusValue::Linear(terms) = flags else {
                panic!("{name}: a flags element of flag bits, not {flags:?}");
            };
            let (mut low, mut high) = (0i64, 0i64);
            for t in terms {
                match *t {
                    LinearTerm::Constant(c) => {
                        low += c;
                        high += c;
                    }
                    LinearTerm::Column { coefficient, .. } if coefficient > 0 => {
                        high += coefficient
                    }
                    LinearTerm::Column { coefficient, .. } => low += coefficient,
                    ref other => panic!("{name}: {other:?} in the flags"),
                }
            }
            assert!(
                0 <= low && high < ECALL_MEMORY_OP_TAG as i64,
                "{name}: flags in [{low}, {high}] may reach the tag"
            );
            zero_reachable |= low == 0;
        }
    }
    assert_eq!(seen, receivers.len(), "one dispatch receiver a table");
    assert!(
        zero_reachable,
        "the control: some receiver takes flags 0, the untagged ECALL's"
    );
    // The CPU's side: no DECODE entry is both MEMORY and ECALL, an ECALL's
    // mem_flags is 0, and every mem_flags byte is below the tag.
    for name in PROGRAMS {
        let (program, _) = run(name);
        let instructions =
            crate::tables::decode::instructions_from_elf(&program).expect("it decodes");
        let table = crate::tables::decode::DecodeTable::from_instructions(&instructions);
        for row in 0..table.padding_row() {
            let f = &table.entry(row as u32).fields;
            assert!(
                !(f.memory && f.ecall),
                "{name}: an entry both MEMORY and ECALL"
            );
            assert!(
                !f.ecall || f.mem_flags == 0,
                "{name}: an ECALL with mem_flags"
            );
            assert!(u64::from(f.mem_flags) < ECALL_MEMORY_OP_TAG, "{name}");
        }
    }
}

/// A CPU row of `traces` whose entry satisfies `pick`, as (table, row).
fn cpu_row(traces: &Traces, pick: &dyn Fn(&Table, usize) -> bool) -> (usize, usize) {
    for (t, table) in traces.cpus.iter().enumerate() {
        for row in 0..table.num_rows() {
            if pick(table, row) {
                return (t, row);
            }
        }
    }
    panic!("no CPU row of that kind");
}

/// Row `row` of `table` with `col_up` raised by one and `col_down` lowered by
/// `2^(off_up − off_down)` (a field element, a fraction when `off_up` is the
/// lower offset): the packed DECODE element, `Σ 2^off · field`, is unchanged.
fn shift(
    table: &mut Table,
    row: usize,
    (col_up, off_up): (usize, u32),
    (col_down, off_down): (usize, u32),
) {
    let two = |k: u32| FE::from(1u64 << k);
    let down = two(off_up) * two(off_down).inv().expect("a power of two");
    let a = *table.get_main(row, col_up) + FE::one();
    let b = *table.get_main(row, col_down) - down;
    set(table, row, col_up, a);
    set(table, row, col_down, b);
}

/// The control mutant of the decode-byte forgeries: the lean row with today's packed DECODE
/// element in place of its split one, on DECODE's receiver and on the CPU's
/// and CPU32's sends — the lean row without the change that replaced the byte
/// checks. Every other interaction is the lean row's.
fn packed_decode_mutant(name: &str, list: &[BusInteraction]) -> Option<Vec<BusInteraction>> {
    let decode = BusId::Decode as u64;
    let today = if name == "DECODE" {
        crate::tables::decode::bus_interactions_for(CpuRowForm::Full)
    } else if name.starts_with("CPU32[") {
        crate::tables::cpu32::bus_interactions_for(CpuRowForm::Full)
    } else if name.starts_with("CPU[") {
        crate::tables::cpu::bus_interactions_inrow_for(0, CpuRowForm::Full)
    } else {
        return None;
    };
    let mut packed = today
        .into_iter()
        .find(|i| i.bus_id == decode)
        .expect("today's DECODE interaction");
    if name == "DECODE" {
        // The lean trace holds `MU` after the bytes.
        packed.multiplicity =
            stark::lookup::Multiplicity::Column(crate::tables::decode::lean_cols::MU);
    }
    Some(
        list.iter()
            .map(|i| {
                if i.bus_id == decode {
                    packed.clone()
                } else {
                    i.clone()
                }
            })
            .collect(),
    )
}

/// The decode bytes: a byte raised by one and paid for by another field (the
/// packed element unchanged), on a row where the payer is unused — rd paid by
/// rs2 on an ADDI, rs1 paid by rd on a store, rd paid by alu_flags on an ADDI,
/// and a word row's half length paid by rd on CPU32. The lean row's DECODE refuses each; its control mutant with
/// today's packed element takes each on DECODE's bus, and today's row refuses
/// each on its byte checks alone.
#[test]
fn a_byte_moved_between_decode_fields_is_refused() {
    use crate::tables::types::packed_decode_shrunk as pd;
    fn addi(t: &Table, r: usize) -> bool {
        get(t, r, cpu_cols::ADD) == 1
            && get(t, r, cpu_cols::READ_REGISTER2) == 0
            && get(t, r, cpu_cols::MEMORY) == 0
            && get(t, r, cpu_cols::ECALL) == 0
            && get(t, r, cpu_cols::WORD_INSTR) == 0
            && get(t, r, cpu_cols::PC_0) != crate::tables::cpu::CPU_PADDING_PC
    }
    fn store(t: &Table, r: usize) -> bool {
        get(t, r, cpu_cols::MEMORY) == 1 && get(t, r, cpu_cols::MEM_FLAGS) & 1 == 1
    }
    fn on_cpu(
        traces: &mut Traces,
        pick: &dyn Fn(&Table, usize) -> bool,
        up: (usize, u32),
        down: (usize, u32),
    ) {
        let (t, row) = cpu_row(traces, pick);
        shift(&mut traces.cpus[t], row, up, down);
    }
    fn rd_paid_by_rs2(tr: &mut Traces) {
        on_cpu(tr, &addi, (cpu_cols::RD, pd::RD), (cpu_cols::RS2, pd::RS2));
    }
    fn rs1_paid_by_rd(tr: &mut Traces) {
        on_cpu(tr, &store, (cpu_cols::RS1, pd::RS1), (cpu_cols::RD, pd::RD));
    }
    fn rd_paid_by_alu_flags(tr: &mut Traces) {
        on_cpu(
            tr,
            &addi,
            (cpu_cols::RD, pd::RD),
            (cpu_cols::ALU_FLAGS, pd::ALU_FLAGS),
        );
    }
    fn half_length_paid_by_rd(tr: &mut Traces) {
        let (t, row) = tr
            .cpu32s
            .iter()
            .enumerate()
            .find_map(|(t, table)| {
                (0..table.num_rows())
                    .find(|&r| get(table, r, c32::MU) == 1)
                    .map(|r| (t, r))
            })
            .expect("a word row");
        shift(
            &mut tr.cpu32s[t],
            row,
            (c32::HALF_INSTRUCTION_LENGTH, pd::HALF_INSTRUCTION_LENGTH),
            (c32::RD, pd::RD),
        );
    }
    let cases: [Case<'_>; 4] = [
        ("rd paid by rs2", "all_instructions_64", rd_paid_by_rs2),
        ("rs1 paid by rd", "all_loadstore_32", rs1_paid_by_rd),
        (
            "rd paid by alu_flags",
            "all_instructions_64",
            rd_paid_by_alu_flags,
        ),
        (
            "half length paid by rd",
            "test_wsuffix_64bit",
            half_length_paid_by_rd,
        ),
    ];
    let decode = u64::from(BusId::Decode);
    let bytes = u64::from(BusId::AreBytes);
    for (what, name, forge) in cases {
        let (program, logs) = run(name);
        let mut lean = build(&program, &logs, CpuRowForm::Lean);
        forge(&mut lean);
        let lean_airs = airs(&program, &lean, CpuRowForm::Lean);
        assert!(
            left(&lean_airs, &mut lean).contains_key(&decode),
            "{what}: the lean DECODE took it"
        );
        assert!(
            !left_with(&lean_airs, &mut lean, &packed_decode_mutant).contains_key(&decode),
            "{what}: the control mutant (today's packed element) must take it on DECODE's bus"
        );

        let mut full = build(&program, &logs, CpuRowForm::Full);
        forge(&mut full);
        let full_airs = airs(&program, &full, CpuRowForm::Full);
        let left = left(&full_airs, &mut full);
        assert!(
            !left.contains_key(&decode),
            "{what}: the packed element moved"
        );
        assert!(
            left.contains_key(&bytes),
            "{what}: today's byte checks refuse it"
        );
    }
}

/// A constraint set's values on one row (as `word_walk_tests` evaluates them).
fn constraint_values<CS: ConstraintSet<GoldilocksField, GoldilocksExtension>>(
    set: &CS,
    row: &[FE],
) -> Vec<FE> {
    use math::field::element::FieldElement;
    use stark::constraints::builder::ProverEvalFolder;
    use stark::frame::Frame;
    use stark::table::TableView;
    use stark::traits::TransitionEvaluationContext;
    let n = set.meta().len();
    let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
        vec![row.to_vec()],
        vec![vec![]],
    )]);
    let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
    let offset_e = FieldElement::<GoldilocksExtension>::zero();
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
    let mut base = vec![FE::zero(); n];
    let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
    set.eval(&mut folder);
    base
}

fn nonzero(values: &[FE]) -> Vec<usize> {
    (0..values.len())
        .filter(|&k| values[k] != FE::zero())
        .collect()
}

/// The first ECALL row of the lean build of `name`, its cells.
fn ecall_row(name: &str) -> Vec<FE> {
    let (program, logs) = run(name);
    let traces = build(&program, &logs, CpuRowForm::Lean);
    let (t, row) = cpu_row(&traces, &|t, r| get(t, r, cpu_cols::ECALL) == 1);
    let table = &traces.cpus[t];
    (0..table.main_table.width)
        .map(|c| *table.get_main(row, c))
        .collect()
}

/// The syscall number: an ECALL row whose `res` (and `rvd = res`) is
/// another number than `a7` (`rv1`), so another syscall chip would take its
/// dispatch. Under the lean row the ECALL's entry sets ADD, and the ADD fast
/// path's pair refuses it; the control mutant, the same row without the ADD
/// flag (today's ECALL entry), satisfies every lean constraint.
#[test]
fn an_ecall_rows_dispatch_carries_a7() {
    use crate::constraints::cpu::CpuLeanConstraints;
    let honest = ecall_row("test_commit_4");
    assert_eq!(
        honest[cpu_cols::ADD],
        FE::one(),
        "the lean ECALL entry sets ADD"
    );
    assert!(nonzero(&constraint_values(&CpuLeanConstraints, &honest)).is_empty());
    let a7 = honest[cpu_cols::RV1_0].canonical_u64()
        + (honest[cpu_cols::RV1_0 + 1].canonical_u64() << 32);
    for k in 0..4 {
        assert_eq!(
            honest[cpu_cols::RES_0 + k].canonical_u64(),
            (a7 >> (16 * k)) & 0xFFFF,
            "an honest ECALL row's res is a7"
        );
    }
    let mut forged = honest.clone();
    let other = a7 + 1;
    for k in 0..4 {
        forged[cpu_cols::RES_0 + k] = FE::from((other >> (16 * k)) & 0xFFFF);
    }
    forged[cpu_cols::RVD_0] = FE::from(other & 0xFFFF_FFFF);
    forged[cpu_cols::RVD_0 + 1] = FE::from(other >> 32);
    let refused = nonzero(&constraint_values(&CpuLeanConstraints, &forged));
    assert!(!refused.is_empty(), "the lean row refuses it");
    let mut control = forged.clone();
    control[cpu_cols::ADD] = FE::zero();
    assert!(
        nonzero(&constraint_values(&CpuLeanConstraints, &control)).is_empty(),
        "the control: without the ADD flag every constraint holds on it"
    );
}

/// The dispatch's multiplicity: MEMORY and ECALL both set on one row
/// (multiplicity 2). The lean constraints refuse it at `MEMORY·ECALL`.
#[test]
fn memory_and_ecall_on_one_row_is_refused() {
    use crate::constraints::cpu::{CpuLeanConstraints, NUM_CPU_CONSTRAINTS};
    let mut forged = ecall_row("test_commit_4");
    forged[cpu_cols::MEMORY] = FE::one();
    assert_eq!(
        nonzero(&constraint_values(&CpuLeanConstraints, &forged)),
        vec![NUM_CPU_CONSTRAINTS],
        "MEMORY·ECALL = 0 refuses it, and nothing else does"
    );
    assert!(
        nonzero(&constraint_values(
            &crate::constraints::cpu::CpuConstraints,
            &forged
        ))
        .is_empty(),
        "the control: without MEMORY·ECALL every constraint holds on it"
    );
}

/// The lean constraint set is today's and one, each emitted once.
#[test]
fn the_lean_constraints_are_todays_and_one() {
    use crate::constraints::cpu::{
        CpuConstraints, CpuLeanConstraints, NUM_CPU_CONSTRAINTS, NUM_CPU_LEAN_CONSTRAINTS,
    };
    assert_eq!(CpuConstraints.meta().len(), NUM_CPU_CONSTRAINTS);
    assert_eq!(CpuLeanConstraints.meta().len(), NUM_CPU_LEAN_CONSTRAINTS);
    assert_eq!(NUM_CPU_LEAN_CONSTRAINTS, NUM_CPU_CONSTRAINTS + 1);
    stark::constraints::builder::check_dense_index_set(
        &CpuLeanConstraints.meta(),
        NUM_CPU_LEAN_CONSTRAINTS,
    )
    .expect("each index once");
}

/// The statement tag names the lean row and leaves today's alone.
#[test]
fn the_tag_names_the_lean_row_only() {
    use crate::block_whir::{BlockFormat, block_format_tag};
    let today = BlockFormat::production();
    let lean = today.with_cpu_row(CpuRowForm::Lean);
    let (a, b) = (block_format_tag(&today), block_format_tag(&lean));
    assert!(b.starts_with(&a) && b[a.len()..] == *b"/Rl", "{a:?} {b:?}");
    assert_eq!(BlockFormat::pinned().cpu_row, CpuRowForm::Full);
    assert_eq!(BlockFormat::production().cpu_row, CpuRowForm::Full);
    let _ = cpu::DECODE_BYTE_COLUMNS;
}

/// Every syscall chip, every memory receiver: a block's AIR set with one
/// table of every family (the pinned LogUp census's counts), under each
/// memory and register format. Under the lean row no AIR uses the ECALL bus;
/// under both rows the CPU tables are `MemoryOp`'s only senders (the sole
/// sender argument a syscall chip's presence rests on); and under the lean
/// row every CPU interaction that reads a decode byte column has a
/// multiplicity that a word row zeroes — `1 − word_instr`, a register flag,
/// `MEMORY + ECALL`, `branch_cond`, `ALU` — except the CPU32 send, the one
/// reader of `half_instruction_length` on a word row, whose CPU32 lookup binds
/// it to CPU32's DECODE.
#[test]
fn every_lean_set_moves_the_ecall_off_its_bus_and_keeps_the_sole_sender() {
    use crate::tables::register::RegisterFormat;
    use stark::lookup::Multiplicity;
    let elf = crate::test_utils::asm_elf_bytes("all_instructions_64");
    let program = Elf::load(&elf).expect("the ELF loads");
    let opts = ProofOptions::default_test_options();
    let counts = crate::TableCounts {
        cpu: 1,
        lt: 1,
        memw: 1,
        memw_aligned: 1,
        load: 1,
        mul: 1,
        dvrm: 1,
        shift: 1,
        branch: 1,
        memw_register: 1,
        eq: 1,
        bytewise: 1,
        store: 1,
        cpu32: 1,
        keccak: 1,
        keccak_rnd: 1,
        ecsm: 1,
        ecdas: 1,
        hint: 1,
        copy: 1,
        blake3: 1,
        is_wh: 1,
        ecrec: 1,
    };
    let pages =
        Traces::page_configs_from_elf_and_runtime(&program, &[], 0, 1 << 20).expect("pages");
    let inputs = crate::BlockAirInputs::new(&opts, &program, vec![1 << 10], &[]).expect("inputs");
    let pinned = crate::block_whir::BlockFormat::pinned();
    let byte_columns = cpu::DECODE_BYTE_COLUMNS;
    let gated = |m: &Multiplicity| match m {
        Multiplicity::Negated(c) => *c == cpu_cols::WORD_INSTR,
        Multiplicity::Column(c) => [
            cpu_cols::READ_REGISTER1,
            cpu_cols::READ_REGISTER2,
            cpu_cols::WRITE_REGISTER,
            cpu_cols::BRANCH_COND,
            cpu_cols::ALU,
            cpu_cols::WORD_INSTR,
        ]
        .contains(c),
        Multiplicity::Sum(a, b) => (*a, *b) == (cpu_cols::MEMORY, cpu_cols::ECALL),
        _ => false,
    };
    for registers in [RegisterFormat::InRow, RegisterFormat::Tokens32] {
        for memory in [
            MemoryFormat::Split,
            MemoryFormat::Fused,
            MemoryFormat::Words,
        ] {
            for form in [CpuRowForm::Full, CpuRowForm::Lean] {
                let mut airs = crate::block_airs(
                    &program,
                    &opts,
                    &pages,
                    &counts,
                    &inputs,
                    pinned.keccak_rnd,
                    pinned.memw_aligned,
                    registers,
                );
                match memory {
                    MemoryFormat::Split => {}
                    MemoryFormat::Fused => airs.with_fused_memory(&opts, [1, 1]),
                    MemoryFormat::Words => airs
                        .with_word_memory(&opts, [1; 7], &pages)
                        .expect("the word tables"),
                }
                airs.with_cpu_row(&opts, form, registers, memory, &inputs)
                    .expect("the set");
                let what = format!("{form:?}, {memory:?}, {registers:?}");
                let mut syscall_receivers = 0;
                for air in airs.air_refs() {
                    let name = air.name();
                    for i in air.bus_interactions() {
                        if i.bus_id == BusId::Ecall as u64 {
                            assert_eq!(form, CpuRowForm::Full, "{what}: {name} on the ECALL bus");
                        }
                        if i.bus_id == BusId::MemoryOp as u64 && i.is_sender {
                            assert!(name.starts_with("CPU["), "{what}: {name} sends on MemoryOp");
                        }
                        let tagged = i.values.len() == 9
                            && format!("{:?}", i.values[6])
                                == format!("{:?}", BusValue::constant(ECALL_MEMORY_OP_TAG));
                        if i.bus_id == BusId::MemoryOp as u64 && !i.is_sender && tagged {
                            syscall_receivers += 1;
                        }
                    }
                    if form == CpuRowForm::Lean && name.starts_with("CPU[") {
                        for i in air.bus_interactions() {
                            let reads = i.columns_read();
                            let byte = byte_columns.iter().any(|c| reads.contains(c));
                            if !byte {
                                continue;
                            }
                            let cpu32_half = i.bus_id == BusId::Cpu32 as u64;
                            assert!(
                                gated(&i.multiplicity) || cpu32_half,
                                "{what}: a CPU interaction on bus {} reads a decode byte at \
                                 multiplicity {:?}",
                                i.bus_id,
                                i.multiplicity
                            );
                        }
                    }
                }
                if form == CpuRowForm::Lean {
                    assert!(
                        syscall_receivers >= 6,
                        "{what}: {syscall_receivers} tagged receivers"
                    );
                }
            }
        }
    }
}

/// The ECALL's DECODE entry the lean row's ADD argument rests on: it reads
/// `rs1 = 17` and nothing else, has no immediate and sets no other flag. The
/// lean table refuses an ECALL entry outside that (an immediate, an rs2 read,
/// another class flag), and takes the honest one.
#[test]
fn the_ecall_entry_is_the_one_the_lean_row_covers() {
    use crate::tables::decode::check_lean_ecall_entries;
    use crate::tables::types::{DecodeEntry, ShrunkDecode};
    use executor::vm::instruction::decoding::Instruction;
    let f = ShrunkDecode::from_instruction(Instruction::EcallEbreak, 4);
    let expected = ShrunkDecode {
        rs1: 17,
        read_register1: true,
        ecall: true,
        half_instruction_length: 2,
        ..Default::default()
    };
    assert_eq!(f, expected, "the ECALL's fields");
    let honest = DecodeEntry::from_instruction(0x1000, Instruction::EcallEbreak, 4);
    assert_eq!(honest.imm, 0, "the ECALL's immediate");
    assert!(check_lean_ecall_entries([&honest], CpuRowForm::Lean).is_ok());
    let mut with_imm = honest.clone();
    with_imm.imm = 4;
    let mut reads_rs2 = honest.clone();
    reads_rs2.fields.read_register2 = true;
    let mut writes = honest.clone();
    writes.fields.write_register = true;
    let mut also_memory = honest.clone();
    also_memory.fields.memory = true;
    for (what, e) in [
        ("an immediate", &with_imm),
        ("an rs2 read", &reads_rs2),
        ("a register write", &writes),
        ("another class flag", &also_memory),
    ] {
        assert!(
            check_lean_ecall_entries([e], CpuRowForm::Lean).is_err(),
            "{what}: refused"
        );
        assert!(
            check_lean_ecall_entries([e], CpuRowForm::Full).is_ok(),
            "{what}: today's row"
        );
    }
}

/// The prover's refusals: the CPU row's form set after a window, or after the
/// card took the CPU chunks' decode, and a BITWISE take-back below zero.
#[test]
fn the_prover_refuses_the_lean_row_where_it_cannot_hold() {
    use crate::tables::trace_builder::WindowedTraceBuilder;
    let (program, logs) = run("sub");
    let mut walked =
        WindowedTraceBuilder::new(&program, &[], &MaxRowsConfig::small()).expect("the builder");
    assert!(logs.len() > 2, "a run past its first cycle");
    let _ = walked.push(&logs[..1]).expect("a window");
    assert!(
        walked.cpu_row_form(CpuRowForm::Lean).is_err(),
        "after a window"
    );
    let carded = WindowedTraceBuilder::new(&program, &[], &MaxRowsConfig::small())
        .and_then(|b| b.drop_streamed_ops())
        .and_then(|b| b.card_fill_streamed(std::sync::Arc::new(|_| true), true, false))
        .expect("the builder");
    assert!(
        carded.cpu_row_form(CpuRowForm::Lean).is_err(),
        "after the card took the decode"
    );
    let removed = vec![(
        crate::tables::bitwise::BitwiseOperation::byte_op(
            crate::tables::bitwise::BitwiseOperationType::AreBytes,
            1,
            2,
        ),
        1,
    )];
    assert!(
        super::finish::without_removed(
            crate::tables::bitwise::BitwiseHistogram::new(),
            Some(&removed)
        )
        .is_err(),
        "a take-back of a lookup never counted"
    );
}
