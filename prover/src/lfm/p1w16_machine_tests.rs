//! The width-16 socket wired into the machine: `Instr::Hash16` from the
//! builder through compilation, admission, both execution schedules and the
//! `LFM_HASH` trace, under `HasherKind::Poseidon1W16`.

use math::field::element::FieldElement;
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::table::TableView;
use stark::traits::TransitionEvaluationContext;

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
use crypto::hash::poseidon1_w16 as p1;

use super::builder::{Cell, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::executor::{LfmExecError, execute_serial, execute_with_merged_levels};
use super::hash::HasherKind;
use super::p1w16_socket::{P1W16SocketConstraints, SOCKET_FORM, cols};
use super::validator::{LfmViolation, validate};

type Gl = GoldilocksField;
type Gl3 = GoldilocksExtension;

fn word(seed: u64) -> [FE; 4] {
    core::array::from_fn(|i| FE::from(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i as u64))
}

/// Two chained width-16 permutations over constant cells, the first's cell 0
/// read twice and every output cell of the second published, so the
/// multiplicities are 2, 0, 0, 0 and then 1, 1, 1, 1. Returns the program and
/// the host's expected outputs of both rows.
fn chained_program() -> (LfmProgram, [FE; 16], [FE; 16]) {
    let mut b = LfmBuilder::new();
    let ins: [Cell; 4] = core::array::from_fn(|k| b.digest_const(word(k as u64 + 1)).as_cell());
    let first = b.hash16(ins);
    let zero = b.digest_const([FE::zero(); 4]).as_cell();
    let second = b.hash16([first[0], zero, first[0], ins[3]]);
    for c in second {
        b.public(c);
    }
    let program = compile(b.finish());

    let mut s0 = [FE::zero(); 16];
    for k in 0..4 {
        s0[4 * k..4 * k + 4].copy_from_slice(&word(k as u64 + 1));
    }
    let o0 = p1::permute(s0);
    let mut s1 = [FE::zero(); 16];
    s1[0..4].copy_from_slice(&o0[0..4]);
    s1[8..12].copy_from_slice(&o0[0..4]);
    s1[12..16].copy_from_slice(&word(4));
    (program, o0, p1::permute(s1))
}

fn socket_violations(row: &[FE]) -> Vec<usize> {
    let set = P1W16SocketConstraints { form: SOCKET_FORM };
    let n = ConstraintSet::<Gl, Gl3>::meta(&set).len();
    let no_ch: Vec<FieldElement<Gl3>> = vec![];
    let offset = FieldElement::<Gl3>::zero();
    let frame = Frame::<Gl, Gl3>::new(vec![TableView::new(vec![row.to_vec()], vec![vec![]])]);
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_ch, &no_ch, &offset);
    let mut base_out = vec![FE::zero(); n];
    let mut ext_out = vec![FieldElement::<Gl3>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base_out, &mut ext_out);
    set.eval(&mut folder);
    folder.assert_all_emitted();
    base_out
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != FE::zero())
        .map(|(i, _)| i)
        .collect()
}

#[test]
fn a_width16_program_compiles_admits_and_proves_under_the_socket() {
    let (program, _, _) = chained_program();
    assert!(program.hash16());
    assert_eq!(
        program.hasher(crate::hash_pin::BLOCK_HASHER),
        HasherKind::Poseidon1W16
    );
    assert_eq!(program.groups.hash.real_rows, 2);
    validate(&program).expect("admissible");

    // The instruction group is the socket's: addresses, IS_REAL, and the
    // backfilled multiplicities 2,0,0,0 then 1,1,1,1.
    let g = &program.groups.hash;
    assert_eq!(*g.at(0, cols::IS_REAL), FE::one());
    assert_eq!(*g.at(1, cols::IS_REAL), FE::one());
    let mults = |r: usize| -> Vec<FE> { (0..4).map(|k| *g.at(r, cols::MULT0 + k)).collect() };
    assert_eq!(mults(0), [2u64, 0, 0, 0].map(FE::from).to_vec());
    assert_eq!(mults(1), [1u64, 1, 1, 1].map(FE::from).to_vec());
}

#[test]
fn both_schedules_execute_the_permutations_and_agree() {
    let (program, o0, o1) = chained_program();
    let hasher = HasherKind::Poseidon1W16;
    let serial = execute_serial(&program, &[], &hasher).expect("serial");
    let level = execute_with_merged_levels(&program, &[], &hasher, 0, 1).expect("levels");
    for exec in [&serial, &level] {
        assert!(exec.records.hash.is_empty());
        assert_eq!(exec.records.hash16.len(), 2);
        assert_eq!(exec.records.hash16[0].outs, o0);
        assert_eq!(exec.records.hash16[1].outs, o1);
        let published: Vec<FE> = exec.public_words.iter().flat_map(|(_, w)| *w).collect();
        assert_eq!(published, o1.to_vec());
    }
    for (a, b) in serial.records.hash16.iter().zip(&level.records.hash16) {
        assert_eq!((a.ins, a.outs), (b.ins, b.outs));
    }
}

/// Every `LFM_HASH` row of the trace — the two real rows and the padding —
/// satisfies the socket's constraints, and a real row's preprocessed prefix is
/// its instruction's.
#[test]
fn the_hash_trace_satisfies_the_socket() {
    let (program, _, _) = chained_program();
    let hasher = HasherKind::Poseidon1W16;
    let exec = execute_serial(&program, &[], &hasher).expect("executes");
    let traces = super::trace::build_traces_with_hasher(&program, &exec.records, hasher);
    let t = &traces.hash;
    assert_eq!(t.num_cols(), SOCKET_FORM.num_columns());
    assert!(t.num_rows() >= 2);
    for r in 0..t.num_rows() {
        let row: Vec<FE> = (0..t.num_cols()).map(|c| *t.get_main(r, c)).collect();
        assert_eq!(socket_violations(&row), Vec::<usize>::new(), "row {r}");
        let real = row[cols::IS_REAL] == FE::one();
        assert_eq!(real, r < 2, "row {r}");
    }
}

#[test]
fn a_width16_row_under_a_twelve_felt_hasher_is_refused() {
    let (program, _, _) = chained_program();
    let err = execute_serial(&program, &[], &HasherKind::Rpx).map(|_| ());
    assert!(
        matches!(err, Err(LfmExecError::HasherRejected(_))),
        "{err:?}"
    );
}

#[test]
fn a_twelve_felt_row_under_the_socket_is_refused() {
    let mut b = LfmBuilder::new();
    let a = b.digest_const(word(1));
    let c = b.digest_const(word(2));
    let d = b.compress(a, c);
    b.public(d.as_cell());
    let program = compile(b.finish());
    assert!(!program.hash16());
    let err = execute_serial(&program, &[], &HasherKind::Poseidon1W16).map(|_| ());
    assert!(
        matches!(err, Err(LfmExecError::HasherRejected(_))),
        "{err:?}"
    );
}

/// The socket review's A2: under the width-16 socket the twelve-felt contract
/// hands out no hash at all — every hashing method refuses, so no host caller
/// can take an unproved twelve-felt Poseidon1 value for a proved one. (The
/// executor never gets that far: `admits` refuses the row first, as
/// [`a_twelve_felt_row_under_the_socket_is_refused`] shows.)
#[test]
fn the_twelve_felt_contract_hands_out_no_hash_under_the_socket() {
    use super::hash::LfmHasher;
    let h = HasherKind::Poseidon1W16;
    let w = word(3);
    type Call = fn(HasherKind, [FE; 4]);
    let calls: [(&str, Call); 4] = [
        ("permute", |h, _| {
            let _ = h.permute([FE::zero(); 12]);
        }),
        ("compress", |h, w| {
            let _ = h.compress(&w, &w);
        }),
        ("transcript", |h, w| {
            let _ = h.transcript(&w, &w);
        }),
        ("leaf", |h, w| {
            let _ = h.leaf(&w, &w);
        }),
    ];
    for (what, call) in calls {
        assert!(
            std::panic::catch_unwind(move || call(h, w)).is_err(),
            "{what} returned a twelve-felt hash under the socket"
        );
    }
}

/// A2's reachability, driven: every production entry point that hashes a
/// twelve-felt row with a program's hasher, handed a twelve-felt row of each
/// mode under the width-16 socket, refuses it as a TYPED error and never reaches
/// the refusing `permute` (a panic here fails the test).
///
/// - The executor, under all three schedules: `hash_compute` reads `mode_iv`
///   (the zero IV), then `admits` refuses before any hashing method runs.
/// - The trace filler over twelve-felt records (an RPX execution's) handed the
///   socket's hasher: its socket arm fills no witness and hashes nothing.
///
/// The other callers of the twelve-felt contract are not production entry
/// points with a program's hasher: the host block transcript is pinned to
/// `BLOCK_HASHER` at compile time (`hash_pin`), and `fixture`'s host sponge and
/// tree serve the fixture programs and the suites only.
#[test]
fn no_production_entry_point_reaches_the_twelve_felt_hash_under_the_socket() {
    use super::executor::execute;
    use super::trace::build_traces_with_hasher;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let programs: Vec<(&str, LfmProgram)> = vec![
        ("compress", {
            let mut b = LfmBuilder::new();
            let a = b.digest_const(word(1));
            let c = b.digest_const(word(2));
            let d = b.compress(a, c);
            b.public(d.as_cell());
            compile(b.finish())
        }),
        ("transcript", {
            let mut b = LfmBuilder::new();
            let a = b.digest_const(word(3));
            let c = b.digest_const(word(4));
            let d = b.transcript_step(a, c);
            b.public(d.as_cell());
            compile(b.finish())
        }),
        ("leaf", {
            let mut b = LfmBuilder::new();
            let a = b.digest_const(word(5));
            let f = b.digest_const(word(6)).as_cell();
            let d = b.leaf(a, f);
            b.public(d.as_cell());
            compile(b.finish())
        }),
        ("permute", {
            let mut b = LfmBuilder::new();
            let s: [Cell; 3] =
                core::array::from_fn(|k| b.digest_const(word(7 + k as u64)).as_cell());
            let out = b.permute(s);
            b.public(out[0]);
            compile(b.finish())
        }),
    ];
    let socket = HasherKind::Poseidon1W16;
    for (mode, program) in &programs {
        assert!(!program.hash16(), "{mode}: a twelve-felt program");
        type Entry = fn(&LfmProgram, HasherKind) -> Result<(), LfmExecError>;
        let entries: [(&str, Entry); 3] = [
            ("execute (the default schedule)", |p, h| {
                execute(p, &[], &h).map(|_| ())
            }),
            ("execute_serial", |p, h| {
                execute_serial(p, &[], &h).map(|_| ())
            }),
            ("execute_with_merged_levels", |p, h| {
                execute_with_merged_levels(p, &[], &h, 1, 2).map(|_| ())
            }),
        ];
        for (entry, run) in entries {
            let got = catch_unwind(AssertUnwindSafe(|| run(program, socket)));
            assert!(
                matches!(got, Ok(Err(LfmExecError::HasherRejected(_)))),
                "{entry} on a {mode} row under the socket: {:?}",
                got.map_err(|_| "panicked")
            );
        }
        // The trace filler: twelve-felt records, the socket's hasher.
        let exec = execute_serial(program, &[], &HasherKind::Rpx).expect("executes under RPX");
        let filled = catch_unwind(AssertUnwindSafe(|| {
            build_traces_with_hasher(program, &exec.records, socket);
        }));
        assert!(
            filled.is_ok(),
            "the trace filler hashed a {mode} row under the socket"
        );
    }
}

#[test]
fn a_program_mixing_the_two_widths_is_not_admitted() {
    let mut b = LfmBuilder::new();
    let ins: [Cell; 4] = core::array::from_fn(|k| b.digest_const(word(k as u64)).as_cell());
    let out = b.hash16(ins);
    let d = b.compress(out[0].as_digest(), out[1].as_digest());
    b.public(d.as_cell());
    let program = compile(b.finish());
    assert!(matches!(
        validate(&program),
        Err(LfmViolation::MixedHashWidths { .. })
    ));
}

/// The AIR set under the socket: `LFM_HASH` is the socket's AIR, the same
/// width as RPX's, with its eight interactions.
#[test]
fn the_air_set_under_the_socket_carries_the_socket_chip() {
    use super::airs::{ChipSet, LfmAirs, NUM_LFM_CHIPS};
    use super::chips::hash;
    let opts = super::proof::aggregation_wrap_options();
    let roots = [[0u8; 32]; NUM_LFM_CHIPS];
    let airs = LfmAirs::new_with_hasher(&roots, &opts, 1, HasherKind::Poseidon1W16, ChipSet::FULL);
    let names: Vec<&str> = airs.air_refs().iter().map(|a| a.name()).collect();
    assert!(names.contains(&"LFM_HASH"));
    assert_eq!(hash::num_columns(HasherKind::Poseidon1W16), 329);
    assert_eq!(hash::bus_interactions(HasherKind::Poseidon1W16).len(), 8);
    assert_eq!(HasherKind::Poseidon1W16.as_tag(), 5);
}
