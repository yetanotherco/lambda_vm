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
    // leaves out (`keccak::leave_out`, thread-local) is left out of them too.
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
// forged trace is refused; with it left out (`keccak::leave_out`), the trace passes
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

/// A forgery and the checks that stop it: `build` makes its traces (a forged
/// walk, and for some a post-build edit of the KECCAK cells) with whatever
/// checks are left out on this thread at the time; without `left_out`, the
/// traces pass.
struct Forgery {
    name: &'static str,
    left_out: &'static [SpongeCheck],
    build: fn() -> (Elf, Traces),
}

/// Every forgery a sponge check is load-bearing for, one per check.
const FORGERIES: [Forgery; 10] = [
    Forgery {
        name: "a byte past the input set nonzero",
        left_out: &[SpongeCheck::MsgZero],
        build: past_the_input,
    },
    Forgery {
        name: "a hole in the read mask",
        left_out: &[SpongeCheck::Prefix],
        build: hole_in_the_mask,
    },
    Forgery {
        name: "the last block reads none of its byte",
        left_out: &[SpongeCheck::CountOnLast],
        build: last_block_reads_nothing,
    },
    Forgery {
        name: "a full last block",
        left_out: &[SpongeCheck::LastNotFull],
        build: full_last_block,
    },
    Forgery {
        name: "a middle block of 135 bytes",
        left_out: &[SpongeCheck::FullUnlessLast],
        build: short_middle_block,
    },
    Forgery {
        name: "a nonzero starting state",
        left_out: &[SpongeCheck::FirstState],
        build: nonzero_start,
    },
    Forgery {
        name: "the digest written 64 bytes past x12",
        left_out: &[SpongeCheck::LinkOut],
        build: digest_moved,
    },
    Forgery {
        name: "lane 0 not memory's bytes",
        left_out: &[SpongeCheck::LaneRead],
        build: lane_zero_not_memorys,
    },
    Forgery {
        name: "blocks 1 and 2 swapped, each block end at the other's address",
        left_out: &[SpongeCheck::AddressHalves],
        build: blocks_swapped_by_their_ends,
    },
    Forgery {
        name: "a row off the round chip by mu, a padding row answering for it",
        left_out: &[SpongeCheck::ModeSum, SpongeCheck::TokensOnModes],
        build: row_off_the_round_chip,
    },
];

/// ★ `forgery` is refused with every check in, passes with its checks left
/// out, and commits another digest than the reference.
fn load_bearing(forgery: &Forgery) {
    let (_, honest) = traces();
    let (program, mut with) = (forgery.build)();
    assert_ne!(
        with.public_output_bytes, honest.public_output_bytes,
        "{}: the forgery commits another digest",
        forgery.name
    );
    let refused = check(&program, &mut with);
    assert!(
        refused.is_err(),
        "{}: the forgery passed with every check in",
        forgery.name
    );
    keccak::leave_out(forgery.left_out);
    let (program, mut without) = (forgery.build)();
    let passed = check(&program, &mut without);
    keccak::leave_out(&[]);
    eprintln!(
        "{} ({:?}): refused by {} with the checks; without them: {passed:?}",
        forgery.name,
        forgery.left_out,
        refused.unwrap_err()
    );
    assert!(
        passed.is_ok(),
        "{}: still refused without {:?}: {passed:?}",
        forgery.name,
        forgery.left_out
    );
}

/// Bytes past the input are 0: a byte past a 1-byte input, never read, set
/// nonzero, so the hash is of a block memory never held.
fn past_the_input() -> (Elf, Traces) {
    forged_traces((1, 1), |rows, _| {
        sponge(&mut rows[0]).msg[40] = 7;
        keccak::rechain(rows, [0; 25]);
    })
}

#[test]
fn bytes_past_the_input_are_load_bearing() {
    load_bearing(&FORGERIES[0]);
}

/// The mask is a prefix: a 20-byte input read with a hole at byte 17 and the
/// byte past its end read instead; the count still matches, the `0x01` lands in
/// the hole, and byte 18 drops by one.
fn hole_in_the_mask() -> (Elf, Traces) {
    forged_traces((8, 20), |rows, data| {
        let s = sponge(&mut rows[0]);
        s.mask[17] = false;
        s.msg[17] = 0;
        s.mask[20] = true;
        s.msg[20] = data[8 + 20];
        keccak::rechain(rows, [0; 25]);
    })
}

#[test]
fn the_prefix_mask_is_load_bearing() {
    load_bearing(&FORGERIES[1]);
}

/// The last row reads exactly COUNT bytes: a 137-byte input whose last block
/// reads none of its one byte, so the hash is of the first 136 bytes.
fn last_block_reads_nothing() -> (Elf, Traces) {
    forged_traces((5, 137), |rows, _| {
        let s = sponge(&mut rows[1]);
        s.mask = SpongeRow::prefix(0);
        s.msg = [0; 136];
        keccak::rechain(rows, [0; 25]);
    })
}

#[test]
fn the_count_on_the_last_row_is_load_bearing() {
    load_bearing(&FORGERIES[2]);
}

/// The last block holds at most 135 bytes: a 136-byte input as one full last
/// block (its padding folded into it), one permutation where Keccak takes two.
fn full_last_block() -> (Elf, Traces) {
    forged_traces((9, 136), |rows, data| {
        rows.truncate(1);
        let s = sponge(&mut rows[0]);
        assert!(
            data[9 + 135] < 0x80,
            "the block's last byte leaves room for 0x80"
        );
        s.last = true;
        keccak::rechain(rows, [0; 25]);
    })
}

#[test]
fn the_last_block_bound_is_load_bearing() {
    load_bearing(&FORGERIES[3]);
}

/// A row that is not last reads a whole block: block 1 of a 600-byte input
/// reads 135 bytes, so a `0x01` lands in the middle of the input.
fn short_middle_block() -> (Elf, Traces) {
    forged_traces((7, 600), |rows, _| {
        let s = sponge(&mut rows[1]);
        s.mask = SpongeRow::prefix(135);
        s.msg[135] = 0;
        keccak::rechain(rows, [0; 25]);
    })
}

#[test]
fn whole_blocks_before_the_last_are_load_bearing() {
    load_bearing(&FORGERIES[4]);
}

/// The first row starts from the zero state: a 32-byte input hashed from a
/// state of the prover's choosing.
fn nonzero_start() -> (Elf, Traces) {
    forged_traces((6, 32), |rows, _| {
        let mut start = [0u64; 25];
        start[3] = 0x0123_4567_89ab_cdef;
        start[20] = 1;
        keccak::rechain(rows, start);
    })
}

#[test]
fn the_zero_starting_state_is_load_bearing() {
    load_bearing(&FORGERIES[5]);
}

/// The digest address rides the link: the last row of a 271-byte input writes
/// its digest 64 bytes past where x12 named.
fn digest_moved() -> (Elf, Traces) {
    forged_traces((2, 271), |rows, _| {
        let s = sponge(&mut rows[1]);
        s.out += 64;
    })
}

#[test]
fn the_digest_address_on_the_link_is_load_bearing() {
    load_bearing(&FORGERIES[6]);
}

/// Lane 0's memory read: block 0 of a 272-byte input claims other bytes in
/// lane 0 than memory holds.
fn lane_zero_not_memorys() -> (Elf, Traces) {
    forged_traces((0, 272), |rows, _| {
        sponge(&mut rows[0]).msg[0] ^= 0x5a;
        keccak::rechain(rows, [0; 25]);
    })
}

#[test]
fn the_lane_read_is_load_bearing() {
    load_bearing(&FORGERIES[7]);
}

/// The sponge rows of the call whose first row has `COUNT = len` and reads
/// `data[offset]` first: `(table, row)` in block order.
fn call_rows(t: &Traces, (offset, len): (usize, usize), data: &[u8]) -> Vec<(usize, usize)> {
    let rows = sponge_rows(t);
    let start = rows
        .iter()
        .position(|&at| {
            cell(t, at, cols::FIRST) == FE::one()
                && cell(t, at, cols::COUNT) == FE::from(len as u64)
                && (len == 0 || cell(t, at, cols::msg(0)) == FE::from(u64::from(data[offset])))
        })
        .expect("the call's first row");
    rows[start..start + len / 136 + 1].to_vec()
}

/// A padding row of KECCAK table `k` (`μ = 0`).
fn padding_row(t: &Traces, k: usize) -> (usize, usize) {
    let n = t.keccaks[k].num_rows();
    let row = (0..n)
        .rev()
        .find(|&r| *t.keccaks[k].get_main(r, cols::MU) == FE::zero())
        .expect("a padding row");
    (k, row)
}

/// The block end `SRC_END = SRC_LO + 135` is also the next block's address
/// (the link sends `SRC_END + 1`), at any address, not only near 2^32: blocks
/// 1 and 2 of the 600-byte call swapped (row 1 reads block 2's bytes and row 2
/// block 1's), each row's `SRC_END` set to the next row's address − 1. The
/// IsHalfword values are the honest ends permuted, so only the equation
/// `MU_S·(SRC_LO + 135 − SRC_END) = 0` refuses it.
fn blocks_swapped_by_their_ends() -> (Elf, Traces) {
    let call = (7, 600);
    let (program, mut t) = forged_traces(call, |rows, _| {
        let (s1, s2) = (sponge(&mut rows[1]).src, sponge(&mut rows[2]).src);
        let (m1, m2) = (sponge(&mut rows[1]).msg, sponge(&mut rows[2]).msg);
        sponge(&mut rows[1]).src = s2;
        sponge(&mut rows[1]).msg = m2;
        sponge(&mut rows[2]).src = s1;
        sponge(&mut rows[2]).msg = m1;
        keccak::rechain(rows, [0; 25]);
    });
    let r = call_rows(&t, call, &buffer());
    for j in 0..3 {
        let end = cell(&t, r[j + 1], cols::SRC_LO).canonical_u64() - 1;
        set(&mut t, r[j], cols::SRC_END, FE::from(end & 0xFFFF));
        set(&mut t, r[j], cols::SRC_END + 1, FE::from(end >> 16));
    }
    (program, t)
}

#[test]
fn the_block_end_defines_the_next_block_address() {
    load_bearing(&FORGERIES[8]);
}

/// A 32-byte call's only row writes a digest that is not the permutation of
/// its input, with `μ = 0` on that row (its mode bit `MU_S` still 1) and a
/// padding row of the same table carrying the real request and reply (`μ = 1`,
/// no mode) so the round chip's 24 rows still balance.
fn row_off_the_round_chip() -> (Elf, Traces) {
    use executor::vm::instruction::execution::keccak_f1600;
    let call = (6, 32);
    let (program, mut t) = forged_traces(call, |rows, _| {
        rows[0].output[0] ^= 0x0123_4567_89ab_cdef;
        rows[0].output[3] ^= 1;
    });
    let at = call_rows(&t, call, &buffer())[0];
    let mut lanes = [0u64; 25];
    for (l, lane) in lanes.iter_mut().enumerate() {
        for b in 0..8 {
            let v = cell(&t, at, cols::input_byte(8 * l + b)).canonical_u64();
            *lane |= v << (8 * b);
        }
    }
    let input = lanes;
    keccak_f1600(&mut lanes);
    set(&mut t, at, cols::MU, FE::zero());
    let pad = padding_row(&t, at.0);
    set(&mut t, pad, cols::MU, FE::one());
    let (ts0, ts1) = (
        cell(&t, at, cols::TIMESTAMP_0),
        cell(&t, at, cols::TIMESTAMP_1),
    );
    set(&mut t, pad, cols::TIMESTAMP_0, ts0);
    set(&mut t, pad, cols::TIMESTAMP_1, ts1);
    for l in 0..25 {
        for b in 0..8 {
            let byte = |w: u64| FE::from((w >> (8 * b)) & 0xFF);
            set(&mut t, pad, cols::input_byte(8 * l + b), byte(input[l]));
            set(&mut t, pad, cols::output_byte(8 * l + b), byte(lanes[l]));
        }
    }
    (program, t)
}

/// ★ A row's mode bit is what puts it on the round chip. The two Keccak-bus
/// tokens are weighted by `MU_P + MU_S`, so a sponge row with `μ = 0` still asks
/// the round chip and gets the real permutation back: the forged digest is
/// refused even with idx 56 (`μ = MU_P + MU_S`) left out. Only with the tokens
/// put back on `μ` as well does the forgery pass; with the tokens on `μ` and
/// idx 56 in, idx 56 refuses it.
#[test]
fn the_mode_bits_put_a_row_on_the_round_chip() {
    load_bearing(&FORGERIES[9]);
    for one in [SpongeCheck::ModeSum, SpongeCheck::TokensOnModes] {
        keccak::leave_out(&[one]);
        let (program, mut traces) = row_off_the_round_chip();
        let refused = check(&program, &mut traces);
        keccak::leave_out(&[]);
        eprintln!("{one:?} alone left out: {refused:?}");
        assert!(
            refused.is_err(),
            "{one:?} alone left out: the forgery passed"
        );
    }
}

// =============================================================================
// ★ Box: the same forgeries through a real prove and verify
// =============================================================================
//
// The checker above closes every bus the tables close among themselves, but
// not the Memory bus (the statement closes it), so on its own it shows that a
// forgery passes the chips, not that it passes the verifier. Here each forgery
// is proved and verified: refused with every check in, and accepted (Memory bus
// included) with its checks left out, so each mutation result above holds for
// the verifier. The prover's table workers are plain scoped threads, so the
// checks are left out process-wide (`keccak::LeftOutEverywhere`): the test must
// be the only one in its process, which `BOX_ALONE` makes its runner say.

/// The environment word the box test's runner sets to say the test runs alone
/// (`--exact --test-threads=1`); without it the test refuses to run.
const BOX_ALONE: &str = "LAMBDA_VM_KECCAK256_BOX_ALONE";

/// Proves and verifies `build`'s traces with `left_out` left out on every
/// thread: `(accepted, the bus balance alone refused it)`.
///
/// The AIRs are built fresh for each call. `test_utils` keeps one prototype of
/// each AIR a process builds, keyed by name and proof options, and serves its
/// clone after that: the first KECCAK AIR of the process would then carry its
/// bus interactions into every later prove, so a check that lives in an
/// interaction (the digest address on the link, lane 0's read, the round-chip
/// tokens on the mode bits) would stay in whatever is left out (box job l4d).
/// Each call takes its own query count (3 + a counter: a key no earlier call
/// used), and the KECCAK AIRs it proves with are checked to carry the
/// interactions the left-out checks make.
fn prove_without(left_out: &'static [SpongeCheck], build: fn() -> (Elf, Traces)) -> (bool, bool) {
    use stark::proof::view::StarkProofView;
    use stark::verifier::IsStarkVerifier;
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let _left = keccak::LeftOutEverywhere::new(left_out);
    let (program, mut traces) = build();
    let mut options = stark::proof::options::ProofOptions::default_test_options();
    options.fri_number_of_queries += 1 + CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let table_counts = traces.table_counts();
    let register_init =
        crate::tables::register::register_init_from_entry_point(program.entry_point);
    let register_commitment =
        crate::test_utils::trace_register_commitment(&options, &register_init, &traces);
    let airs = crate::VmAirs::new(
        &program,
        &options,
        false,
        &traces.page_configs,
        &table_counts,
        &crate::cpu_heights_from_traces(&traces),
        None,
        true,
        None,
        register_commitment.preprocessed(),
    );
    let shape = |interactions: &[stark::lookup::BusInteraction]| -> Vec<String> {
        interactions
            .iter()
            .map(|b| {
                format!(
                    "{} {} {:?} {:?}",
                    b.bus_id, b.is_sender, b.multiplicity, b.values
                )
            })
            .collect()
    };
    let want = shape(&keccak::bus_interactions());
    let refs = airs.air_refs();
    let keccaks: Vec<_> = refs
        .iter()
        .filter(|a| a.name().starts_with("KECCAK["))
        .collect();
    assert!(!keccaks.is_empty(), "a KECCAK AIR");
    for air in &keccaks {
        assert_eq!(
            shape(air.bus_interactions()),
            want,
            "{}: the AIR proved with carries the interactions of {left_out:?}",
            air.name()
        );
    }
    let pairs = airs.air_trace_pairs(&mut traces);
    let proof =
        crate::test_utils::multi_prove_ram(pairs, &mut crate::hash_pin::block_transcript(&[]))
            .unwrap_or_else(|e| panic!("prover failed, which is not a verifier rejection: {e:?}"));
    let views: Vec<StarkProofView<_, _, ()>> =
        proof.proofs.iter().map(StarkProofView::Owned).collect();
    let refs = airs.air_refs();
    let target = crate::compute_expected_commit_bus_balance_view(
        &refs,
        &views,
        &traces.public_output_bytes,
        0,
        &mut crate::hash_pin::block_transcript(&[]),
    )
    .expect("fingerprint collision in test");
    let verify = |target: &_| {
        crate::hash_pin::BlockVerifier::multi_verify_views(
            &refs,
            &views,
            &mut crate::hash_pin::block_transcript(&[]),
            target,
        )
    };
    let accepted = verify(&target);
    let mut sum = math::field::element::FieldElement::zero();
    for (air, view) in refs.iter().zip(views.iter()) {
        if air.has_trace_interaction()
            && let Some(contribution) = view.bus_table_contribution()
        {
            sum += contribution;
        }
    }
    (accepted, !accepted && verify(&sum))
}

#[test]
#[ignore = "box: proves test_keccak256 about 25 times; run alone with LAMBDA_VM_KECCAK256_BOX_ALONE=1"]
fn box_each_forgery_is_refused_by_the_verifier_and_accepted_without_its_checks() {
    assert_eq!(
        std::env::var(BOX_ALONE).as_deref(),
        Ok("1"),
        "run this test alone (--exact --test-threads=1) with {BOX_ALONE}=1: it leaves KECCAK's \
         checks out for the whole process while it runs"
    );
    let (_, honest) = traces();
    let (accepted, _) = prove_without(&[], traces);
    assert!(accepted, "the honest trace proves and verifies");
    let mut wrong = Vec::new();
    for forgery in &FORGERIES {
        let (_, forged) = (forgery.build)();
        assert_ne!(forged.public_output_bytes, honest.public_output_bytes);
        let (with, bus_only) = prove_without(&[], forgery.build);
        let (without, _) = prove_without(forgery.left_out, forgery.build);
        eprintln!(
            "KECCAK256-BOX {}: with every check accepted={with} (bus alone: {bus_only}); \
             without {:?} accepted={without}",
            forgery.name, forgery.left_out
        );
        if with || !without {
            wrong.push(forgery.name);
        }
    }
    // The row off the round chip with one of its two guards left out: the
    // other refuses it.
    for one in [&[SpongeCheck::ModeSum], &[SpongeCheck::TokensOnModes]] {
        let (accepted, _) = prove_without(one, row_off_the_round_chip);
        eprintln!("KECCAK256-BOX round chip, {one:?} alone left out: accepted={accepted}");
        if accepted {
            wrong.push("a row off the round chip with one guard left out");
        }
    }
    assert!(wrong.is_empty(), "not as expected: {wrong:?}");
}
