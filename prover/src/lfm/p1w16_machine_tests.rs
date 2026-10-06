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
    assert!(program.hash16);
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
    assert!(!program.hash16);
    let err = execute_serial(&program, &[], &HasherKind::Poseidon1W16).map(|_| ());
    assert!(
        matches!(err, Err(LfmExecError::HasherRejected(_))),
        "{err:?}"
    );
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
