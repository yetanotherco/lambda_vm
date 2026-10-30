//! ★ Keccak-256 of a buffer (`keccak256` ecall): the KECCAK table's sponge mode
//! on `test_keccak256`, an assembly program that hashes eleven buffers (every
//! padding case, unaligned inputs, a digest written over its own input) and
//! commits the digests.
//!
//! - the committed digests are the reference Keccak-256 of each buffer;
//! - every KECCAK row satisfies the table's constraints, and every bus the
//!   tables close among themselves balances, whole-run and windowed;
//! - a KECCAK trace forged one way at a time (each of the attacks the sponge
//!   mode's constraints are there to stop) breaks a constraint or a bus.

use executor::elf::Elf;
use tiny_keccak::{Hasher, Keccak};

use super::bus_net_tests::{named, unbalanced_with};
use super::finish_tests::run;
use super::*;
use crate::tables::MaxRowsConfig;
use crate::tables::keccak::{KeccakConstraints, cols};
use crate::tables::types::FE;
use crate::test_utils::{busless_air, validate_busless};

const PROGRAM: &str = "test_keccak256";

/// `(input offset, length)` of each call, in commit order (the program's table).
const CALLS: [(usize, usize); 12] = [
    (0, 0),
    (1, 1),
    (3, 135),
    (0, 136),
    (5, 137),
    (2, 271),
    (0, 272),
    (7, 600),
    (8, 20),
    (6, 32),
    (4, 140),
    (9, 136),
];

fn reference(input: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak::v256();
    hasher.update(input);
    let mut out = [0u8; 32];
    hasher.finalize(&mut out);
    out
}

fn buffer() -> Vec<u8> {
    (0..1024).map(|i| ((13 * i + 5) % 256) as u8).collect()
}

fn traces() -> (Elf, Traces) {
    let (program, logs) = run(PROGRAM);
    let traces = Traces::from_elf_and_logs(&program, &logs, &MaxRowsConfig::default(), &[])
        .expect("the build");
    (program, traces)
}

/// The KECCAK rows of the trace that are sponge rows, `(table, row)`.
fn sponge_rows(traces: &Traces) -> Vec<(usize, usize)> {
    let one = FE::one();
    let mut rows = Vec::new();
    for (k, t) in traces.keccaks.iter().enumerate() {
        for row in 0..t.num_rows() {
            if *t.get_main(row, cols::MU_S) == one {
                rows.push((k, row));
            }
        }
    }
    rows
}

fn cell(traces: &Traces, (k, row): (usize, usize), col: usize) -> FE {
    *traces.keccaks[k].get_main(row, col)
}

fn set(traces: &mut Traces, (k, row): (usize, usize), col: usize, value: FE) {
    traces.keccaks[k].main_table.set(row, col, value);
}

/// Whether every KECCAK table satisfies the constraints and every inner bus
/// balances; `Err` names what broke.
fn check(program: &Elf, traces: &mut Traces) -> Result<(), String> {
    let air = busless_air(cols::NUM_COLUMNS, KeccakConstraints);
    for (k, t) in traces.keccaks.iter().enumerate() {
        if !validate_busless(&air, t) {
            return Err(format!("KECCAK[{k}] constraints"));
        }
    }
    // KECCAK's interactions built here, on the test's thread, so a check a test
    // leaves out (`keccak::DROPPED`, thread-local) is left out of them too.
    let left = unbalanced_with(program, traces, &|name| {
        name.starts_with("KECCAK[")
            .then(crate::tables::keccak::bus_interactions)
    });
    if left.is_empty() {
        Ok(())
    } else {
        Err(named(&left))
    }
}

/// The row of call `call`'s block `block` (calls in program order).
fn row_of(traces: &Traces, call: usize, block: usize) -> (usize, usize) {
    let rows = sponge_rows(traces);
    let firsts: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter(|(_, at)| cell(traces, **at, cols::FIRST) == FE::one())
        .map(|(i, _)| i)
        .collect();
    let at = rows[firsts[call] + block];
    assert_eq!(
        cell(traces, at, cols::TIMESTAMP_1),
        FE::from(block as u64),
        "rows of a call are in block order"
    );
    at
}

#[test]
fn the_committed_digests_are_the_reference_hashes() {
    let (program, logs) = run(PROGRAM);
    let traces = Traces::from_elf_and_logs(&program, &logs, &MaxRowsConfig::default(), &[])
        .expect("the build");
    let output = &traces.public_output_bytes;
    let data = buffer();
    for (k, &(offset, len)) in CALLS.iter().enumerate() {
        assert_eq!(
            &output[32 * k..32 * k + 32],
            &reference(&data[offset..offset + len]),
            "call {k}: offset {offset}, length {len}"
        );
    }
    let calls = logs
        .iter()
        .filter(|l| l.src1_val == executor::vm::instruction::execution::KECCAK256_SYSCALL_NUMBER)
        .count();
    assert!(calls >= CALLS.len());
}

#[test]
fn the_sponge_rows_satisfy_the_constraints_and_every_bus_balances() {
    let (program, mut traces) = traces();
    let rows = sponge_rows(&traces);
    let blocks: usize = CALLS.iter().map(|&(_, len)| len / 136 + 1).sum();
    assert_eq!(rows.len(), blocks, "one KECCAK row per block");
    check(&program, &mut traces).expect("the honest trace");
}

/// Applies `forge` to a fresh honest trace and requires the check to fail.
fn refused(what: &str, forge: impl FnOnce(&mut Traces)) -> String {
    let (program, mut traces) = traces();
    forge(&mut traces);
    match check(&program, &mut traces) {
        Ok(()) => panic!("{what}: the forged trace passed"),
        Err(why) => {
            eprintln!("{what}: refused by {why}");
            why
        }
    }
}

/// The `0x01` moved one byte later on a last row: the mask claims one more
/// input byte (a zero), and the count is raised to match.
#[test]
fn a_padding_byte_moved_is_refused() {
    refused("0x01 one byte late", |t| {
        let at = row_of(t, 4, 1); // 137 bytes: a 1-byte tail
        // byte 1 of lane 0 joins the input as a zero.
        set(t, at, cols::part(0, 1), FE::one());
        let count = cell(t, at, cols::COUNT);
        set(t, at, cols::COUNT, count + FE::one());
    });
}

/// The 0x80 dropped: byte 135 of the last block is claimed input (FULL_16 on
/// the last row), which the last-row bound refuses.
#[test]
fn a_last_block_of_136_bytes_is_refused() {
    refused("last block full", |t| {
        let at = row_of(t, 3, 1); // 136 bytes: the padding-alone block
        for lane in 0..cols::RATE_LANES {
            set(t, at, cols::full(lane), FE::one());
        }
        set(t, at, cols::COUNT, FE::from(136u64));
    });
}

/// A middle block skipped: its row turned into a padding row.
#[test]
fn a_skipped_block_is_refused() {
    refused("middle block skipped", |t| {
        let at = row_of(t, 7, 2); // 600 bytes: five blocks
        for col in 0..cols::NUM_COLUMNS {
            set(t, at, col, FE::zero());
        }
        for lane in 0..25 {
            set(t, at, cols::state_ptr(lane, 0), FE::from(8 * lane as u64));
        }
    });
}

/// Two blocks of one call swapped: row 1 absorbs block 2's bytes and the
/// reverse.
#[test]
fn swapped_blocks_are_refused() {
    refused("blocks 1 and 2 swapped", |t| {
        let (a, b) = (row_of(t, 7, 1), row_of(t, 7, 2));
        for i in 0..cols::RATE {
            let (x, y) = (cell(t, a, cols::msg(i)), cell(t, b, cols::msg(i)));
            set(t, a, cols::msg(i), y);
            set(t, b, cols::msg(i), x);
        }
    });
}

/// The digest written from the state before the last permutation: the last
/// row's first four output lanes replaced by the previous row's.
#[test]
fn a_digest_from_a_non_final_state_is_refused() {
    refused("digest from the previous state", |t| {
        let (prev, last) = (row_of(t, 6, 1), row_of(t, 6, 2));
        for i in 0..32 {
            let v = cell(t, prev, cols::output_byte(i));
            set(t, last, cols::output_byte(i), v);
        }
    });
}

/// A block read one byte off: the row's address moved by one, its range
/// check kept consistent.
#[test]
fn a_block_read_from_the_wrong_address_is_refused() {
    refused("block read one byte off", |t| {
        let at = row_of(t, 6, 1); // 272 bytes, aligned
        let src = cell(t, at, cols::SRC_LO);
        set(t, at, cols::SRC_LO, src + FE::one());
        let end = cell(t, at, cols::SRC_END);
        set(t, at, cols::SRC_END, end + FE::one());
    });
}

/// One input byte changed in the row (the hash of other bytes than memory's).
#[test]
fn a_message_byte_not_in_memory_is_refused() {
    refused("a message byte changed", |t| {
        let at = row_of(t, 2, 0);
        let v = cell(t, at, cols::msg(17));
        set(t, at, cols::msg(17), v + FE::one());
    });
}

/// A byte past the input claimed nonzero (it is never read).
#[test]
fn a_nonzero_byte_past_the_input_is_refused() {
    refused("byte past the input", |t| {
        let at = row_of(t, 1, 0); // one input byte
        set(t, at, cols::msg(40), FE::from(7u64));
    });
}

/// A call's first row restarting from a nonzero state.
#[test]
fn a_first_row_with_a_nonzero_state_is_refused() {
    refused("nonzero starting state", |t| {
        let at = row_of(t, 6, 0);
        set(t, at, cols::input_byte(150), FE::from(1u64));
    });
}

/// ★ A row in both modes, or with the other mode's flags.
#[test]
fn a_row_mixing_the_two_modes_is_refused() {
    let both = refused("permute and sponge at once", |t| {
        let at = row_of(t, 0, 0);
        set(t, at, cols::MU_P, FE::one());
    });
    assert!(both.contains("constraints"), "{both}");
    let flag = refused("FIRST on a permute row", |t| {
        let at = row_of(t, 0, 0);
        set(t, at, cols::MU_S, FE::zero());
        set(t, at, cols::MU_P, FE::one());
    });
    assert!(flag.contains("constraints"), "{flag}");
    let mu = refused("mu 2 over both modes", |t| {
        let at = row_of(t, 0, 0);
        set(t, at, cols::MU, FE::from(2u64));
        set(t, at, cols::MU_P, FE::one());
    });
    assert!(mu.contains("constraints"), "{mu}");
}

/// ★ Multiplicity columns that are not bits (a spread `+1, +1, −1` witness).
#[test]
fn non_bit_flags_are_refused() {
    for (name, col) in [
        ("FIRST", cols::FIRST),
        ("LAST", cols::LAST),
        ("FULL", cols::full(3)),
        ("PART", cols::part(0, 0)),
    ] {
        let why = refused(name, |t| {
            let at = row_of(t, 6, 1);
            set(t, at, col, -FE::one());
        });
        assert!(why.contains("constraints"), "{name}: {why}");
    }
}

// =============================================================================
// ★ Each sponge check is load-bearing: left out, the forgery it stops passes
// =============================================================================
//
// A forgery here is consistent everywhere but at the one check: the walk builds
// it (`keccak::FORGE` rewrites one call's rows before the walk emits their memory
// traffic, and `keccak::rechain` recomputes their permutations), so the round
// chip, the BITWISE counts, the memory reads and the digest writes all follow it,
// and the committed digest is not the input's hash. With the check in, the
// forged trace is refused; with it left out (`keccak::DROPPED`), the trace passes
// every KECCAK constraint and every bus the tables close among themselves.

use crate::tables::keccak::{self, KeccakOperation, SpongeCheck, SpongeRow};

/// Whether `rows` are the call of `len` bytes at `offset` into the buffer.
fn is_call(rows: &[KeccakOperation], (offset, len): (usize, usize), data: &[u8]) -> bool {
    rows.first()
        .and_then(|r| r.sponge.as_ref())
        .is_some_and(|s| s.count == len as u64 && (len == 0 || s.msg[0] == data[offset]))
}

fn sponge(row: &mut KeccakOperation) -> &mut SpongeRow {
    row.sponge.as_mut().expect("a sponge row")
}

/// The traces with the call `call` (`(offset, length)`) forged by `forge`
/// (which gets the call's rows and the input buffer's bytes).
fn forged_traces(
    call: (usize, usize),
    forge: impl Fn(&mut Vec<KeccakOperation>, &[u8]) + 'static,
) -> (Elf, Traces) {
    let ran = std::rc::Rc::new(std::cell::Cell::new(false));
    let seen = std::rc::Rc::clone(&ran);
    keccak::FORGE.with(|f| {
        *f.borrow_mut() = Some(Box::new(move |rows: &mut Vec<KeccakOperation>| {
            let data = buffer();
            if is_call(rows, call, &data) {
                forge(rows, &data);
                seen.set(true);
            }
        }));
    });
    let built = traces();
    keccak::FORGE.with(|f| *f.borrow_mut() = None);
    assert!(ran.get(), "the forger ran on the call {call:?}");
    built
}

/// ★ `which` stops `forge` (on the call `call`): refused with the check,
/// passing without it, and the committed digest moved off the reference.
fn load_bearing(
    which: SpongeCheck,
    call: (usize, usize),
    forge: impl Fn(&mut Vec<KeccakOperation>, &[u8]) + Clone + 'static,
) {
    let (_, honest) = traces();
    let (program, mut with) = forged_traces(call, forge.clone());
    assert_ne!(
        with.public_output_bytes, honest.public_output_bytes,
        "{which:?}: the forgery commits another digest"
    );
    let refused = check(&program, &mut with);
    assert!(
        refused.is_err(),
        "{which:?}: the forgery passed with the check in"
    );
    keccak::DROPPED.with(|d| d.set(Some(which)));
    let (program, mut without) = forged_traces(call, forge);
    let passed = check(&program, &mut without);
    keccak::DROPPED.with(|d| d.set(None));
    eprintln!(
        "{which:?}: refused by {} with the check; without it: {passed:?}",
        refused.unwrap_err()
    );
    assert!(
        passed.is_ok(),
        "{which:?}: still refused without the check: {passed:?}"
    );
}

/// Bytes past the input are 0: a byte past a 1-byte input, never read, set
/// nonzero, so the hash is of a block memory never held.
#[test]
fn bytes_past_the_input_are_load_bearing() {
    load_bearing(SpongeCheck::MsgZero, (1, 1), |rows, _| {
        sponge(&mut rows[0]).msg[40] = 7;
        keccak::rechain(rows, [0; 25]);
    });
}

/// The mask is a prefix: a 20-byte input read with a hole at byte 17 and the
/// byte past its end read instead; the count still matches, the `0x01` lands in
/// the hole, and byte 18 drops by one.
#[test]
fn the_prefix_mask_is_load_bearing() {
    load_bearing(SpongeCheck::Prefix, (8, 20), |rows, data| {
        let s = sponge(&mut rows[0]);
        s.mask[17] = false;
        s.msg[17] = 0;
        s.mask[20] = true;
        s.msg[20] = data[8 + 20];
        keccak::rechain(rows, [0; 25]);
    });
}

/// The last row reads exactly COUNT bytes: a 137-byte input whose last block
/// reads none of its one byte, so the hash is of the first 136 bytes.
#[test]
fn the_count_on_the_last_row_is_load_bearing() {
    load_bearing(SpongeCheck::CountOnLast, (5, 137), |rows, _| {
        let s = sponge(&mut rows[1]);
        s.mask = SpongeRow::prefix(0);
        s.msg = [0; 136];
        keccak::rechain(rows, [0; 25]);
    });
}

/// The last block holds at most 135 bytes: a 136-byte input as one full last
/// block (its padding folded into it), one permutation where Keccak takes two.
#[test]
fn the_last_block_bound_is_load_bearing() {
    load_bearing(SpongeCheck::LastNotFull, (9, 136), |rows, data| {
        rows.truncate(1);
        let s = sponge(&mut rows[0]);
        assert!(
            data[9 + 135] < 0x80,
            "the block's last byte leaves room for 0x80"
        );
        s.last = true;
        keccak::rechain(rows, [0; 25]);
    });
}

/// A row that is not last reads a whole block: block 1 of a 600-byte input
/// reads 135 bytes, so a `0x01` lands in the middle of the input.
#[test]
fn whole_blocks_before_the_last_are_load_bearing() {
    load_bearing(SpongeCheck::FullUnlessLast, (7, 600), |rows, _| {
        let s = sponge(&mut rows[1]);
        s.mask = SpongeRow::prefix(135);
        s.msg[135] = 0;
        keccak::rechain(rows, [0; 25]);
    });
}

/// The first row starts from the zero state: a 32-byte input hashed from a
/// state of the prover's choosing.
#[test]
fn the_zero_starting_state_is_load_bearing() {
    load_bearing(SpongeCheck::FirstState, (6, 32), |rows, _| {
        let mut start = [0u64; 25];
        start[3] = 0x0123_4567_89ab_cdef;
        start[20] = 1;
        keccak::rechain(rows, start);
    });
}

/// The digest address rides the link: the last row of a 271-byte input writes
/// its digest 64 bytes past where x12 named.
#[test]
fn the_digest_address_on_the_link_is_load_bearing() {
    load_bearing(SpongeCheck::LinkOut, (2, 271), |rows, _| {
        let s = sponge(&mut rows[1]);
        s.out += 64;
    });
}

/// Lane 0's memory read: block 0 of a 272-byte input claims other bytes in
/// lane 0 than memory holds.
#[test]
fn the_lane_read_is_load_bearing() {
    load_bearing(SpongeCheck::LaneRead, (0, 272), |rows, _| {
        sponge(&mut rows[0]).msg[0] ^= 0x5a;
        keccak::rechain(rows, [0; 25]);
    });
}
