//! Parity tests for the register-walk decomposition (`emit_register_accesses` +
//! `walk_register_accesses` + `collect_register_ops_parallel`), the CPU reference /
//! swap point for the future device walk. These pin it bit-for-bit against the
//! sequential `collect_register_ops_from_cpu`.
//!
//! Load-bearing semantic: read-old + write-new where **every** access advances the
//! cell timeline (a read writes the cell back at its own ts), so `old_ts` is the
//! *previous access's* ts, not the previous write's. `read_between_writes` guards
//! exactly this — a "chain on writes only" walk gets it wrong.
//!
//! Consistency note: the sequential path sets a READ row's `old_value` to the read
//! value itself, while the walk recovers it from the previous access. They agree
//! only on a *consistent* register trace (read value == last written) — always true
//! in real execution; `parallel_matches_sequential` threads a shadow register file
//! to honor it.

use crate::tables::cpu::CpuOperation;
use crate::tables::memw_register::RegRow;
use crate::tables::trace_builder::{
    MemwBuckets, PC_WORD_ADDR, RegAccess, RegisterState, collect_register_ops_from_cpu,
    collect_register_ops_parallel, walk_register_accesses,
};
use crate::tables::types::{DecodeEntry, ShrunkDecode};

/// Walk a hand-built access list into a fresh `MemwBuckets` seeded from `init`.
fn walk(accesses: &[RegAccess], init: &RegisterState) -> MemwBuckets {
    let mut buckets = MemwBuckets::with_register_capacity(accesses.len());
    walk_register_accesses(accesses, init, &mut buckets);
    buckets
}

/// A row-emitting register access (M1/M3/M5).
fn row(reg_addr: u64, timestamp: u64, value: u64, is_read: bool) -> RegAccess {
    RegAccess {
        reg_addr,
        timestamp,
        value,
        is_read,
        emits_row: true,
    }
}

/// A carry-only PC write (emits no row; advances x255's timeline).
fn pc_write(timestamp: u64, value: u64) -> RegAccess {
    RegAccess {
        reg_addr: PC_WORD_ADDR,
        timestamp,
        value,
        is_read: false,
        emits_row: false,
    }
}

/// THE load-bearing test: a read between two writes to the same register. The
/// second write's `old_ts` must be the intervening READ's ts (the previous
/// *access*), not the first write's ts. A last-write-only walk would yield 6.
#[test]
fn read_between_writes_uses_previous_access_ts() {
    // Register x5 → reg_addr 10. Consistent values (the read returns the last write).
    let accesses = [
        row(10, 6, 100, false),  // write 100 @ 6; old = seed (0, ts 1)
        row(10, 9, 100, true),   // read  100 @ 9; old = (100, ts 6)
        row(10, 14, 200, false), // write 200 @ 14; old = (100, ts *9* — the read)
    ];
    let buckets = walk(&accesses, &RegisterState::new(0));
    let expected = vec![
        RegRow::new(10, 6, 100, 0, 0, 0, 1, false),
        RegRow::new(10, 9, 100, 0, 100, 0, 6, true),
        RegRow::new(10, 14, 200, 0, 100, 0, 9, false), // old_ts = 9, not 6
    ];
    assert_eq!(buckets.register_rows, expected);
}

/// An `rs1 == 255` PC read chains its `old_ts` through the carry-only per-instruction
/// PC writes (which emit no row themselves).
#[test]
fn pc_read_chains_through_implicit_writes() {
    let accesses = [
        pc_write(5, 1000),                 // no row; PC cell ← (1000, 5)
        row(PC_WORD_ADDR, 8, 1000, true),  // read PC; old = (1000, ts 5)
        pc_write(9, 2000),                 // no row; PC cell ← (2000, 9)
        row(PC_WORD_ADDR, 12, 2000, true), // read PC; old = (2000, ts 9)
    ];
    let buckets = walk(&accesses, &RegisterState::new(0));
    let expected = vec![
        RegRow::new(PC_WORD_ADDR, 8, 1000, 0, 1000, 0, 5, true),
        RegRow::new(PC_WORD_ADDR, 12, 2000, 0, 2000, 0, 9, true),
    ];
    assert_eq!(buckets.register_rows, expected);
}

/// A continuation epoch seeds each register's first `old` from the boundary init
/// vector (all init timestamps 1).
#[test]
fn continuation_seed_provides_first_old() {
    // Boundary vector in `register_word_address_list` order: x3 (reg_addr 6) at
    // positions 6 (lo) / 7 (hi). Length ≥ 67 to cover x254/PC index slots.
    let mut init_vec = vec![0u32; 67];
    init_vec[6] = 0xABCD; // x3 low word
    let accesses = [row(6, 5, 0xABCD, true)]; // read x3; consistent with seed
    let buckets = walk(&accesses, &RegisterState::from_init(&init_vec));
    assert_eq!(
        buckets.register_rows,
        vec![RegRow::new(6, 5, 0xABCD, 0, 0xABCD, 0, 1, true)],
    );
}

/// `collect_register_ops_parallel` (emit → walk) must produce the identical MEMW_R
/// rows as the sequential `collect_register_ops_from_cpu` over a whole op stream, on
/// a consistent trace (shadow register file so every read returns the last write).
#[test]
fn parallel_matches_sequential_consistent() {
    let ops = build_consistent_ops(200);

    // Sequential: thread one RegisterState op-by-op.
    let mut seq = MemwBuckets::with_register_capacity(ops.len() * 3);
    let mut state = RegisterState::new(0x1000);
    for op in &ops {
        collect_register_ops_from_cpu(op, &mut state, &mut seq);
    }

    // Parallel: emit all accesses, then walk.
    let mut par = MemwBuckets::with_register_capacity(ops.len() * 3);
    collect_register_ops_parallel(&ops, &RegisterState::new(0x1000), &mut par);

    assert_eq!(
        par.register_rows.len(),
        seq.register_rows.len(),
        "row count differs"
    );
    for (i, (p, s)) in par
        .register_rows
        .iter()
        .zip(seq.register_rows.iter())
        .enumerate()
    {
        assert_eq!(p, s, "first mismatch at row {i}: par={p:?} seq={s:?}");
    }
    assert!(
        !par.register_rows.is_empty(),
        "test must exercise real rows"
    );
}

/// x0 reads/writes emit no access (hardwired zero).
#[test]
fn x0_is_never_accessed() {
    let op = mk_op(MkOp {
        timestamp: 8,
        rs1: 0,
        read_register1: true, // x0 → suppressed
        rd: 0,
        write_register: true, // x0 → suppressed
        next_pc: 0x2000,
        ..MkOp::default()
    });
    let mut accesses = Vec::new();
    crate::tables::trace_builder::emit_register_accesses(&op, &mut accesses);
    // Only the implicit (carry-only) PC write is emitted.
    assert_eq!(accesses.len(), 1);
    assert_eq!(accesses[0].reg_addr, PC_WORD_ADDR);
    assert!(!accesses[0].emits_row);
}

// ---- op-builder helpers (consistent synthetic instruction streams) ----

#[derive(Default, Clone, Copy)]
struct MkOp {
    timestamp: u64,
    rs1: u8,
    read_register1: bool,
    rv1: u64,
    rs2: u8,
    read_register2: bool,
    rv2: u64,
    rd: u8,
    write_register: bool,
    rvd: u64,
    next_pc: u64,
}

fn mk_op(m: MkOp) -> CpuOperation {
    let fields = ShrunkDecode {
        read_register1: m.read_register1,
        read_register2: m.read_register2,
        write_register: m.write_register,
        rs1: m.rs1,
        rs2: m.rs2,
        rd: m.rd,
        ..ShrunkDecode::default()
    };
    CpuOperation {
        decode: DecodeEntry {
            fields,
            ..DecodeEntry::default()
        },
        timestamp: m.timestamp,
        rv1: m.rv1,
        rv2: m.rv2,
        rvd: m.rvd,
        next_pc: m.next_pc,
        ..CpuOperation::default()
    }
}

/// Build `n` instructions with distinct rs1/rs2/rd (no intra-op collision) and a
/// shadow register file, so every read returns the register's last-written value —
/// the consistency the walk's `old_value` recovery requires. Every 7th op reads the
/// PC (rs1 == 255) to exercise the PC timeline.
fn build_consistent_ops(n: usize) -> Vec<CpuOperation> {
    // Seed the shadow file from the same register state the walk uses, so reads of
    // specially-seeded registers (x2 = SP = STACK_TOP) stay consistent.
    let seed = RegisterState::new(0x1000);
    let mut shadow: [u64; 32] = core::array::from_fn(|r| seed.read(r as u8).0);
    let mut shadow_pc = 0x1000u64;
    let mut ops = Vec::with_capacity(n);
    for i in 0..n as u64 {
        let ts = 4 * i + 4;
        // Distinct register bands: rs1∈1..=10, rs2∈11..=20, rd∈21..=30.
        let rs2 = 11 + (i % 10) as u8;
        let rd = 21 + (i % 10) as u8;
        let read_pc = i % 7 == 0;
        let (rs1, rv1) = if read_pc {
            (255u8, shadow_pc)
        } else {
            let r = 1 + (i % 10) as u8;
            (r, shadow[r as usize])
        };
        let rvd = i.wrapping_mul(0x1000).wrapping_add(7);
        let next_pc = 0x1000 + 4 * i;
        let op = mk_op(MkOp {
            timestamp: ts,
            rs1,
            read_register1: true,
            rv1,
            rs2,
            read_register2: i % 2 == 0,
            rv2: shadow[rs2 as usize],
            rd,
            write_register: i % 3 != 0,
            rvd,
            next_pc,
        });
        // Update the shadow file exactly as the walk will (M5 write, then PC write).
        if op.decode.fields.write_register {
            shadow[rd as usize] = rvd;
        }
        shadow_pc = next_pc;
        ops.push(op);
    }
    ops
}
