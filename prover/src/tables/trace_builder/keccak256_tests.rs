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

use super::bus_net_tests::{named, unbalanced};
use super::finish_tests::run;
use super::*;
use crate::tables::MaxRowsConfig;
use crate::tables::keccak::{KeccakConstraints, cols};
use crate::tables::types::FE;
use crate::test_utils::{busless_air, validate_busless};

const PROGRAM: &str = "test_keccak256";

/// `(input offset, length)` of each call, in commit order (the program's table).
const CALLS: [(usize, usize); 11] = [
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
    let left = unbalanced(program, traces);
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
