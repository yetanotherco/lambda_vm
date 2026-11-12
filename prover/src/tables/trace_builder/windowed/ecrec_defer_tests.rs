//! ECREC witnesses off the walk (`WindowedTraceBuilder::defer_ecrec_witnesses`):
//! a deferred call writes what a computed one writes, the finish puts its
//! witness and ECDAS rows where a computed call's are, each check of the join
//! refuses what it guards, and a windowed build that defers builds the
//! whole-run tables.

use crate::tables::MaxRowsConfig;
use crate::tables::cpu::{CpuOp, CpuOperation};
use crate::tables::ecdas::{CompactEcdasOp, EcdasOperation};
use crate::tables::ecrec::EcrecOperation;
use crate::tables::memw::MemwOperation;
use crate::tables::trace_builder::Traces;
use crate::tests::windowed_builder_tests::{run, same_traces, whole};

use super::super::{
    BlockVec, MemoryState, RegisterState, collect_ecrec_ops, resolve_deferred_ecrec,
};
use super::WindowedTraceBuilder;

fn hex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

/// `test_ecrec`'s first signature and a hash it recovers a key for; with
/// `bail`, `s` zero (the call bails).
fn inputs(bail: bool) -> (Vec<u8>, Vec<u8>) {
    let mut sig = hex(concat!(
        "3284bae4a1b693dfbe8ac67400e4891f3b53681c5159a315561051f3d2c57139",
        "97d4b2f2f728a54ed562a12237f6b11d32d626e7fcc24c3470b0729e095bd057"
    ));
    if bail {
        sig[32..].fill(0);
    }
    let msg = hex("c0ffee0011223344556677889900aabbccddeeff00112233445566778899aabb");
    (sig, msg)
}

/// What one ECREC call at `t` collects, walked with `defer` or not: its MEMW
/// ops, its ECREC op, its ECDAS ops, and the 72 bytes it wrote.
fn call(
    t: u64,
    bail: bool,
    defer: bool,
) -> (
    Vec<MemwOperation>,
    EcrecOperation,
    Vec<EcdasOperation>,
    Vec<u32>,
) {
    let (sig, msg) = inputs(bail);
    let (addr_out, addr_sig, addr_msg) = (0x1000u64, 0x2000u64, 0x3000u64);
    let mut memory_state = MemoryState::new();
    let mut register_state = RegisterState::new(0);
    register_state.write(10, addr_out, 1);
    register_state.write(11, addr_sig, 1);
    register_state.write(12, addr_msg, 1);
    register_state.write(13, 0, 1);
    for (base, bytes) in [(addr_sig, &sig), (addr_msg, &msg)] {
        for (i, chunk) in bytes.chunks_exact(8).enumerate() {
            let dword = u64::from_le_bytes(chunk.try_into().unwrap());
            memory_state.write_bytes(base + 8 * i as u64, dword, 8, 1);
        }
    }
    let decode = crate::tables::types::DecodeEntry::default();
    let op = CpuOperation {
        timestamp: t,
        ..Default::default()
    };
    let (memw, ecrec_op, ecdas_ops) = collect_ecrec_ops(
        CpuOp::new(&decode, &op),
        &mut memory_state,
        &mut register_state,
        defer,
    )
    .expect("the operands are held");
    let written = (0..9)
        .flat_map(|i| memory_state.read_bytes(addr_out + 8 * i, 8).0)
        .collect();
    (memw, ecrec_op, ecdas_ops, written)
}

fn dbg<T: std::fmt::Debug>(x: &T) -> String {
    format!("{x:?}")
}

fn compact(ops: &[EcdasOperation]) -> Vec<CompactEcdasOp> {
    ops.iter().map(CompactEcdasOp::from_op).collect()
}

/// A real ECDAS row at `timestamp` (another call's), to stand between calls.
fn other_row(timestamp: u64) -> CompactEcdasOp {
    let (_, _, rows, _) = call(4, false, false);
    CompactEcdasOp::from_op(&EcdasOperation {
        timestamp,
        step: rows[3].step.clone(),
    })
}

/// The run's two lists of three calls (at 400, 800 and 1200, the middle one a
/// bail) between other ECDAS rows, each call walked with `defer`, laid out as
/// a list built whole (`block` 0) or pushed in blocks of `block` rows.
fn lists(defer: bool, block: usize) -> (BlockVec<CompactEcdasOp>, BlockVec<EcrecOperation>) {
    let mut ecdas = Vec::new();
    let mut ecrec = Vec::new();
    ecdas.push(other_row(100));
    for (t, bail) in [(400, false), (800, true), (1200, false)] {
        let (_, op, rows, _) = call(t, bail, defer);
        ecdas.extend(compact(&rows));
        ecdas.push(other_row(t + 100));
        ecrec.push(op);
    }
    if block == 0 {
        return (BlockVec::from_vec(ecdas), BlockVec::from_vec(ecrec));
    }
    let mut e = BlockVec::new(block);
    e.extend(ecdas);
    let mut r = BlockVec::new(block);
    r.extend(ecrec);
    (e, r)
}

fn layout<T>(list: &BlockVec<T>) -> Vec<usize> {
    list.parts().map(<[T]>::len).collect()
}

/// ★ A deferred call writes the bytes, and makes the MEMW ops, a computed call
/// makes; its ECDAS rows are one row standing for them, its witness pending.
/// A bail is computed in the walk either way.
#[test]
fn a_deferred_call_writes_what_a_computed_call_writes() {
    for bail in [false, true] {
        let (memw, op, rows, written) = call(400, bail, false);
        let (d_memw, d_op, d_rows, d_written) = call(400, bail, true);
        assert_eq!(dbg(&memw), dbg(&d_memw), "bail {bail}: MEMW ops");
        assert_eq!(written, d_written, "bail {bail}: the written bytes");
        assert!(op.pending.is_none());
        if bail {
            assert!(d_op.pending.is_none(), "a bail is never deferred");
            assert_eq!(dbg(&op), dbg(&d_op));
            assert_eq!(dbg(&rows), dbg(&d_rows));
            assert!(rows.is_empty());
        } else {
            assert!(d_op.pending.is_some(), "the call is deferred");
            assert_eq!(d_rows.len(), 1);
            assert_eq!(
                CompactEcdasOp::from_op(&d_rows[0]).deferred_ecrec(),
                Some(400)
            );
            assert!(rows.len() > 2);
            assert!(
                rows.iter()
                    .all(|r| CompactEcdasOp::from_op(r).deferred_ecrec().is_none())
            );
        }
    }
}

/// ★ Joined, deferred calls' lists are the computed calls' lists, element for
/// element (witness, ECDAS rows in place between other rows) and block for
/// block, built whole or pushed in blocks of any length; lists with nothing
/// deferred are left as they are.
#[test]
fn joined_deferred_calls_are_the_computed_calls() {
    for block in [0, 1, 3, 64, 1 << 17] {
        let (want_ecdas, want_ecrec) = lists(false, block);
        let (mut ecdas, mut ecrec) = lists(true, block);
        assert!(ecdas.len() < want_ecdas.len());
        resolve_deferred_ecrec(&mut ecdas, &mut ecrec).expect("the join");
        assert!(ecrec.iter().all(|op| op.pending.is_none()), "block {block}");
        assert_eq!(
            dbg(&ecrec.to_vec()),
            dbg(&want_ecrec.to_vec()),
            "block {block}: ECREC"
        );
        assert_eq!(
            dbg(&ecdas.to_vec()),
            dbg(&want_ecdas.to_vec()),
            "block {block}: ECDAS"
        );
        assert_eq!(
            layout(&ecdas),
            layout(&want_ecdas),
            "block {block}: ECDAS blocks"
        );
        // Nothing deferred: left as it is.
        let (mut ecdas, mut ecrec) = lists(false, block);
        resolve_deferred_ecrec(&mut ecdas, &mut ecrec).expect("nothing to join");
        assert_eq!(dbg(&ecdas.to_vec()), dbg(&want_ecdas.to_vec()));
        assert_eq!(layout(&ecdas), layout(&want_ecdas));
    }
}

/// The deferred lists with `tamper` applied, joined: the join's error.
fn joined_after(
    tamper: impl FnOnce(&mut Vec<CompactEcdasOp>, &mut Vec<EcrecOperation>),
) -> Result<(), crate::Error> {
    let (ecdas, ecrec) = lists(true, 0);
    let (mut ecdas, mut ecrec) = (ecdas.to_vec(), ecrec.to_vec());
    tamper(&mut ecdas, &mut ecrec);
    let (mut ecdas, mut ecrec) = (BlockVec::from_vec(ecdas), BlockVec::from_vec(ecrec));
    resolve_deferred_ecrec(&mut ecdas, &mut ecrec)
}

fn refused(what: &str, got: Result<(), crate::Error>, says: &str) {
    match got {
        Err(crate::Error::Prover(e)) => assert!(e.contains(says), "{what}: {e}"),
        other => panic!("{what}: not refused ({other:?})"),
    }
}

/// ★ The join refuses lists whose standing rows and deferred calls do not pair
/// up one for one, in order: a standing row lost, one too many, one moved to
/// another call's timestamp, two calls' rows swapped; a call whose deferred
/// witness is another's; and a witness whose key is not the one written.
#[test]
fn the_join_refuses_what_does_not_pair_up() {
    joined_after(|_, _| {}).expect("untampered, it joins");
    let standing = |ecdas: &Vec<CompactEcdasOp>| -> Vec<usize> {
        (0..ecdas.len())
            .filter(|&i| ecdas[i].deferred_ecrec().is_some())
            .collect()
    };
    refused(
        "a standing row lost",
        joined_after(|ecdas, _| {
            let at = standing(ecdas)[1];
            ecdas.remove(at);
        }),
        "has no ECDAS row standing for it",
    );
    refused(
        "a standing row too many",
        joined_after(|ecdas, _| {
            let at = standing(ecdas)[1];
            let extra = ecdas[at].clone();
            ecdas.push(extra);
        }),
        "meets no deferred call",
    );
    refused(
        "a standing row at another timestamp",
        joined_after(|ecdas, _| {
            let at = standing(ecdas)[0];
            ecdas[at] = CompactEcdasOp::from_op(&EcdasOperation::deferred_ecrec(404));
        }),
        "meets the deferred call at 400",
    );
    refused(
        "two calls swapped",
        joined_after(|_, ecrec| ecrec.swap(0, 2)),
        "meets the deferred call at 1200",
    );
    refused(
        "a call with another's deferred witness",
        joined_after(|_, ecrec| ecrec[0].timestamp = 404),
        "its deferred witness is the call at 400's",
    );
    refused(
        "a key not the one written",
        joined_after(|_, ecrec| ecrec[2].witness.x_q[5] ^= 1),
        "is not the key the walk wrote",
    );
    refused(
        "a status not the one written",
        joined_after(|_, ecrec| ecrec[0].witness.ok = false),
        "is not the key the walk wrote",
    );
}

/// The windowed build of `program`'s run over windows of `window` cycles,
/// the walker and the accumulator split (the block drivers' path), deferring
/// the ECREC witnesses, the handed-out ops kept or dropped; and how many calls
/// the walk deferred.
fn deferred_build(
    program: &executor::elf::Elf,
    logs: &[executor::vm::logs::Log],
    max_rows: &MaxRowsConfig,
    window: usize,
    drop: bool,
) -> (Traces, usize) {
    let mut builder = WindowedTraceBuilder::new(program, &[], max_rows)
        .expect("the builder")
        .defer_ecrec_witnesses(true);
    if drop {
        builder = builder
            .drop_streamed_ops()
            .expect("before the first window");
    }
    let body = logs.len() - 1;
    let cut = body - body % window;
    let (mut chunks, mut deferred) = (Vec::new(), 0);
    for w in logs[..cut].chunks(window) {
        let (mut walker, mut accumulator) = builder.split();
        let walked = walker.walk(w).expect("a window");
        deferred += walked
            .walk
            .ecrec_ops
            .iter()
            .filter(|op| op.pending.is_some())
            .count();
        chunks.extend(
            accumulator
                .absorb(walked)
                .into_iter()
                .map(|job| job.generate()),
        );
    }
    let mut traces = builder.finish(&logs[cut..]).expect("the last window");
    traces
        .insert_streamed(chunks)
        .expect("every chunk has a placeholder");
    (traces, deferred)
}

/// ★ A windowed build that defers the ECREC witnesses builds the whole-run
/// tables at every window length, the handed-out ops kept or dropped; on
/// `test_ecrec`, walked a cycle at a time, it defers both computed recoveries
/// (the bail is computed in the walk). Programs without ECREC calls build the
/// same with it on.
#[test]
fn a_windowed_build_deferring_the_ecrec_witnesses_builds_the_whole_run_tables() {
    let _ecrec = crate::tables::trace_builder::ecrec_forge_free();
    let max_rows = MaxRowsConfig::small();
    for name in ["test_ecrec", "test_ecsm", "all_instructions_64"] {
        let (program, logs) = run(name);
        let reference = whole(&program, &logs, &max_rows);
        for window in [1, 7, 33, 1000] {
            for drop in [false, true] {
                let (traces, deferred) = deferred_build(&program, &logs, &max_rows, window, drop);
                same_traces(&reference, &traces);
                if name == "test_ecrec" && window == 1 {
                    assert_eq!(deferred, 2, "{name}: the calls deferred");
                } else if name != "test_ecrec" {
                    assert_eq!(deferred, 0, "{name}: no ECREC call");
                }
            }
        }
    }
}
