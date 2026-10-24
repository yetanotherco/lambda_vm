//! `Instr::HintOn`: a hint whose arena word the executor picks on a bit.
//!
//! The bit is a schedule dependency and never a read: the hint row is a plain
//! `LFM_HINT` row, the bit's multiplicity counts no `HintOn`, and every
//! executor (serial, level, streamed) runs the hint after the bit's writer.

use crate::tables::types::FE;

use super::builder::{Bit, Cell, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::exec_schedule;
use super::executor::{LfmExecError, StreamedExecution, execute, execute_serial};
use super::hash::HasherKind;
use super::instr::{Addr, Instr};
use super::validator::{LfmViolation, validate};
use super::word::{LfmWord, base_word};

const SOCKET: HasherKind = HasherKind::Poseidon1W16;

fn word(v: u64) -> LfmWord {
    [
        FE::from(v),
        FE::from(v + 1),
        FE::from(v + 2),
        FE::from(v + 3),
    ]
}

/// The published words of `program` over `arenas`, run serially and by
/// levels: the two must agree.
fn run(program: &LfmProgram, arenas: &[Vec<LfmWord>]) -> Result<Vec<LfmWord>, LfmExecError> {
    let serial = execute_serial(program, arenas, &SOCKET).map(|e| e.public_words);
    let levels = execute(program, arenas, &SOCKET).map(|e| e.public_words);
    assert_eq!(
        serial.is_ok(),
        levels.is_ok(),
        "serial {serial:?} against levels {levels:?}"
    );
    let words = serial?;
    assert_eq!(words, levels?, "both schedules publish the same words");
    Ok(words.into_iter().map(|(_, w)| w).collect())
}

/// Two bits of a hinted felt (low first), then a `HintOn` on each over a
/// two-word arena: word 0 on a clear bit, word 1 on a set one.
fn two_bit_program() -> (LfmProgram, [Bit; 2]) {
    let mut b = LfmBuilder::new();
    let a_idx = b.declare_arena(1);
    let a_words = b.declare_arena(2);
    let idx = b.hint_felt(a_idx, 0);
    let bits = b.bit_dec(idx, 2);
    for bit in &bits {
        let c = b.hint_word_on(a_words, 0, *bit, 1);
        b.public(c);
    }
    let program = compile(b.finish());
    (program, [bits[0], bits[1]])
}

/// ★ A `HintOn` reads `index` on a clear bit and `alt` on a set one, under
/// both schedules; it is an `LFM_HINT` row, and its bit's multiplicity counts
/// no read from it (the program validates, every multiplicity recounted).
#[test]
fn a_hint_on_reads_its_alternative_on_a_set_bit() {
    let (program, bits) = two_bit_program();
    validate(&program).expect("admissible: the bits' multiplicities count no HintOn");
    for bit in bits {
        let mult = program.instrs.iter().find_map(|i| match i {
            Instr::BitDec { bits, .. } => {
                bits.iter().find(|(a, _)| *a == bit.addr()).map(|(_, m)| *m)
            }
            _ => None,
        });
        assert_eq!(mult, Some(0), "no instruction reads the bit");
    }
    assert_eq!(
        program.groups.hint.real_rows, 3,
        "the index and two HintOn rows"
    );
    let words = vec![word(10), word(20)];
    for (idx, want) in [
        (0u64, [word(10), word(10)]),
        (1, [word(20), word(10)]),
        (2, [word(10), word(20)]),
        (3, [word(20), word(20)]),
    ] {
        let got =
            run(&program, &[vec![base_word(FE::from(idx))], words.clone()]).expect("executes");
        assert_eq!(got, want.to_vec(), "index {idx}");
    }
}

/// A `HintOn` on a cell that is not a bit has no execution.
#[test]
fn a_hint_on_a_non_boolean_cell_is_refused() {
    let mut b = LfmBuilder::new();
    let a = b.declare_arena(2);
    let v = b.hint_felt(a, 0);
    let c = b.hint_word_on(a, 0, Bit(v.addr()), 1);
    b.public(c);
    let program = compile(b.finish());
    let arena = vec![base_word(FE::from(2u64)), word(5)];
    assert!(matches!(
        execute_serial(&program, &[arena], &SOCKET),
        Err(LfmExecError::NonBooleanBit(_))
    ));
}

/// The validator holds a `HintOn`'s bit to a read's order (written, and
/// before the hint) and both its indices to the arena.
#[test]
fn the_validator_refuses_a_hint_on_out_of_order_or_out_of_bounds() {
    let (program, _) = two_bit_program();
    let at = program
        .instrs
        .iter()
        .position(|i| matches!(i, Instr::HintOn { .. }))
        .expect("a HintOn");
    let num_addrs = program.num_addrs;
    let mutate = |f: &dyn Fn(&mut Instr)| {
        let (mut p, _) = two_bit_program();
        f(&mut p.instrs[at]);
        validate(&p)
    };
    let own_out = |i: &mut Instr| {
        if let Instr::HintOn { bit, out, .. } = i {
            *bit = *out;
        }
    };
    assert!(matches!(
        mutate(&own_out),
        Err(LfmViolation::CyclicRead { .. })
    ));
    let unwritten = |i: &mut Instr| {
        if let Instr::HintOn { bit, .. } = i {
            *bit = Addr(num_addrs + 7);
        }
    };
    assert!(matches!(
        mutate(&unwritten),
        Err(LfmViolation::AddressOutOfRange { .. } | LfmViolation::ReadOfUnwritten { .. })
    ));
    let alt_past = |i: &mut Instr| {
        if let Instr::HintOn { alt, .. } = i {
            *alt = 2;
        }
    };
    assert!(matches!(
        mutate(&alt_past),
        Err(LfmViolation::ArenaOutOfBounds { index: 2, .. })
    ));
}

/// A bit that only exists after a permutation: a hinted 0/1 felt routed
/// through the first lane of a `Hash16` output times zero, so its writer sits
/// a level deep. Then a `HintOn` on it, over its own arena.
fn deep_bit_program() -> LfmProgram {
    let mut b = LfmBuilder::new();
    let a_x = b.declare_arena(1);
    let a_words = b.declare_arena(2);
    let x = b.hint_felt(a_x, 0);
    let h = b.hash16([x.as_cell(); 4]);
    let lane = b.unpack(h[0])[0];
    let zero = b.felt_const(FE::zero());
    let y = b.mul_add(lane, zero, x);
    let bit = b.bit_dec(y, 1)[0];
    let c: Cell = b.hint_word_on(a_words, 0, bit, 1);
    b.public(c);
    compile(b.finish())
}

/// ★ The level schedule puts a `HintOn` at or after its bit's level (the bit
/// is a schedule dependency), so both schedules execute the deep bit's hint.
#[test]
fn a_hint_on_runs_after_a_bit_drawn_from_a_permutation() {
    let program = deep_bit_program();
    validate(&program).expect("admissible");
    let levels = exec_schedule::build(&program);
    let level_of = |pred: &dyn Fn(&Instr) -> bool| {
        (0..levels.levels())
            .find(|&d| {
                levels
                    .others_at(d)
                    .iter()
                    .any(|&i| pred(&program.instrs[i as usize]))
            })
            .expect("scheduled")
    };
    let bit_level = level_of(&|i| matches!(i, Instr::BitDec { .. }));
    let hint_level = level_of(&|i| matches!(i, Instr::HintOn { .. }));
    assert!(bit_level >= 1, "the bit is a permutation deep");
    assert!(hint_level >= bit_level, "the hint runs after its bit");
    let words = vec![word(30), word(40)];
    for (x, want) in [(0u64, word(30)), (1, word(40))] {
        let got = run(&program, &[vec![base_word(FE::from(x))], words.clone()]).expect("executes");
        assert_eq!(got, vec![want], "x = {x}");
    }
}

/// ★ Streamed, the `HintOn` waits for its bit's group as well as its arena's:
/// landing its arena first runs nothing that reads the bit, and the execution
/// then equals the serial one.
#[test]
fn a_streamed_hint_on_waits_for_its_bits_group() {
    let program = deep_bit_program();
    let arenas = [vec![base_word(FE::from(1u64))], vec![word(30), word(40)]];
    let serial = execute_serial(&program, &arenas, &SOCKET).expect("executes");
    for order in [[1usize, 0], [0, 1]] {
        let mut ex = StreamedExecution::new(&program, vec![0, 1], &SOCKET).expect("starts");
        for g in order {
            ex.land(g, std::slice::from_ref(&arenas[g]))
                .unwrap_or_else(|e| panic!("order {order:?}: group {g} lands: {e:?}"));
        }
        let done = ex.finish().expect("finishes");
        assert_eq!(
            done.public_words, serial.public_words,
            "order {order:?}: the serial words"
        );
    }
}
