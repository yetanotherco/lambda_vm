//! An authenticated cap read at a computed index ([`Instr::CapTable`],
//! [`Instr::CapRead`]) and its two lean tables, `LEAN_CAP` and
//! `LEAN_CAP_READ`: a read is looked up as `(tag, index, word)` on
//! `BusId::CapLookup` against the table's rows, each of whose words is bound
//! to its node's cell on `LfmMem`. Round trip, then the forgeries: a word at
//! a wrong index, a table word not bound to its cell (and the mutation that
//! drops the binding), a read through one tree's handle answered from
//! another's, a dropped multiplicity, a padding row's offer (and the mutation
//! that drops its gate).

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};

use super::b3l_tables::{CAP_NAME, CAP_READ_NAME, test_cap};
use super::builder::{B3Block, B3Cv, Cell, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::edsl::WrapHash;
use super::executor::{LfmExecution, execute};
use super::hash::HasherKind;
use super::instr::Instr;
use super::wlfm::{
    WlfmBuild, WlfmProof, WlfmRate, build_wlfm, prove_wlfm, prove_wlfm_traces, test_format,
    verify_wlfm,
};
use super::word::LfmWord;

const HASHER: HasherKind = HasherKind::Poseidon1W16;

/// Tree A's eight nodes, tree B's four.
const A: [u64; 8] = [100, 101, 102, 103, 104, 105, 106, 107];
const B: [u64; 4] = [200, 201, 202, 203];

/// The reads, in program order: `(tree, index)`. Index 5 of A twice (a count
/// of two), index 0 of A (the row a padding row would imitate).
const READS: [(usize, u64); 5] = [(0, 5), (0, 3), (0, 5), (1, 2), (0, 0)];

fn word(v: u64) -> LfmWord {
    [FE::from(v); 4]
}

fn cell(b: &mut LfmBuilder, v: u64) -> Cell {
    b.digest_const(word(v)).as_cell()
}

fn lean_builder() -> LfmBuilder {
    LfmBuilder::new()
        .with_wrap_hash(WrapHash::Blake3)
        .with_b3_lean(true)
}

/// One compression (so the program has lean tables), the two cap tables,
/// and [`READS`], each index from its bits (low first), each read published.
fn program() -> LfmProgram {
    let mut b = lean_builder();
    let block: [Cell; 4] = core::array::from_fn(|i| cell(&mut b, i as u64 + 1));
    b.b3_compress(B3Cv::Iv, B3Block::Lanes(block), 64, 11);
    let a: Vec<Cell> = A.iter().map(|&v| cell(&mut b, v)).collect();
    let bt: Vec<Cell> = B.iter().map(|&v| cell(&mut b, v)).collect();
    let ha = b.cap_table(&a);
    let hb = b.cap_table(&bt);
    for (tree, index) in READS {
        let (handle, nbits) = if tree == 0 { (ha, 3) } else { (hb, 2) };
        let x = b.felt_const(FE::from(index));
        let bits = b.bit_dec(x, nbits);
        let out = b.cap_read(handle, &bits);
        b.public(out);
    }
    compile(b.finish())
}

fn build(program: &LfmProgram) -> WlfmBuild {
    build_wlfm(program, HASHER, WlfmRate::Two).unwrap_or_else(|e| panic!("the W-LFM build: {e}"))
}

fn honest_words() -> Vec<u64> {
    READS
        .iter()
        .map(|&(tree, i)| {
            if tree == 0 {
                A[i as usize]
            } else {
                B[i as usize]
            }
        })
        .collect()
}

/// Read `k`'s word forged to `w`, with the word it publishes forged to match:
/// the memory bus then balances, and only the cap lookup stands between the
/// forger and a false word.
fn forge_read(exec: &mut LfmExecution, k: usize, w: LfmWord) {
    exec.records.cap_read[k].out = w;
    exec.records.public[k] = w;
    exec.public_words[k].1 = w;
}

fn prove(build: &WlfmBuild, program: &LfmProgram, exec: LfmExecution) -> WlfmProof {
    let filled = super::proof::lfm_fill_executed_under(program, exec, HASHER, 0.0, &Vec::new());
    prove_wlfm_traces(build, filled, true).expect("the prover does not check the lookup")
}

fn verifies(build: &WlfmBuild, proof: &WlfmProof) -> bool {
    verify_wlfm(&build.artifacts, proof, &proof.public_words).is_ok()
}

/// The program's cap rows and reads, and the words it publishes: each read
/// is its tree's node at the index its bits spell.
#[test]
fn cap_reads_execute_to_the_indexed_nodes() {
    let program = program();
    super::validator::validate(&program).expect("admissible");
    assert_eq!(program.groups.cap.real_rows, A.len() + B.len());
    assert_eq!(program.groups.cap_read.real_rows, READS.len());
    let exec = execute(&program, &[], &HASHER).expect("executes");
    let published: Vec<u64> = exec
        .public_words
        .iter()
        .map(|(_, w)| w[0].canonical())
        .collect();
    assert_eq!(published, honest_words());
    assert_eq!(exec.records.cap.len(), A.len() + B.len());
    assert_eq!(exec.records.cap[5], word(A[5]));
    assert_eq!(exec.records.cap[A.len() + 2], word(B[2]));
}

/// ★ Cap tables and reads prove and verify through W-LFM: their chunks are
/// the last lean tables, after the byte table (and no mux chunk: this
/// program has no select).
#[test]
fn cap_reads_round_trip_through_wlfm() {
    let _fast = test_format::set(Some(2));
    let program = program();
    let build = build(&program);
    let a = &build.artifacts;
    // 12 table rows cut as 8 + 8 (the last chunk padded), 5 reads in one.
    assert_eq!(
        (a.b3l.caps.as_slice(), a.b3l.cap_reads.as_slice()),
        (&[8, 8][..], &[8][..])
    );
    let airs = a.airs();
    let names: Vec<&str> = airs.air_refs().iter().map(|r| r.name()).collect();
    assert_eq!(names[names.len() - 2..], [CAP_NAME, CAP_READ_NAME]);
    let proof = prove_wlfm(&program, &build, &[]).unwrap_or_else(|e| panic!("proves: {e}"));
    verify_wlfm(a, &proof, &proof.public_words).unwrap_or_else(|e| panic!("verifies: {e}"));
}

/// ★ FORGERY: a word at a wrong index. Read 0 (index 5 of A) answered with
/// node 4's word: refused, the lookup having no `(A, 5, node 4)` row. And the
/// same word claimed at index 4, the lookup then balanced: refused, the index
/// cell holding 5.
#[test]
fn a_read_at_a_wrong_index_is_refused() {
    let _fast = test_format::set(Some(2));
    let program = program();
    let build = build(&program);
    // The control: the same path, untampered, verifies.
    let exec = execute(&program, &[], &HASHER).expect("executes");
    assert!(verifies(&build, &prove(&build, &program, exec)));

    let mut exec = execute(&program, &[], &HASHER).expect("executes");
    forge_read(&mut exec, 0, word(A[4]));
    assert!(!verifies(&build, &prove(&build, &program, exec)));

    let mut exec = execute(&program, &[], &HASHER).expect("executes");
    forge_read(&mut exec, 0, word(A[4]));
    exec.records.cap_read[0].index = FE::from(4u64);
    assert!(!verifies(&build, &prove(&build, &program, exec)));
}

/// Every node cell's read by its cap table dropped from the constants'
/// multiplicities, instructions and prepared columns alike: the program a
/// table whose words are not read from the cells would be.
fn unbound(mut program: LfmProgram) -> LfmProgram {
    let nodes: Vec<u64> = program
        .instrs
        .iter()
        .filter_map(|i| match i {
            Instr::CapTable(t) => Some(t.nodes.iter().map(|a| a.0)),
            _ => None,
        })
        .flatten()
        .collect();
    assert_eq!(nodes.len(), A.len() + B.len());
    let mut row = 0usize;
    for instr in &mut program.instrs {
        if let Instr::Const { out, mult, .. } = instr {
            if nodes.contains(&out.0) {
                *mult -= 1;
                program
                    .groups
                    .const_
                    .set(row, super::layout::const_::MULT, FE::from(*mult));
            }
            row += 1;
        }
    }
    program
}

/// ★ FORGERY + MUTATION: a table word not bound to the authenticated cap.
/// Node 5 of A forged in its table row, both of its reads answered with the
/// forged word: refused while the row reads its node's cell; with that read
/// dropped (the program and the AIR alike), the forgery is accepted.
#[test]
fn a_table_word_off_its_cell_is_refused_and_the_cell_read_is_load_bearing() {
    let _fast = test_format::set(Some(2));
    let forged = word(999);
    let run = |drop: bool| {
        let _mutant = test_cap::set(test_cap::CapMutant {
            drop_node_read: drop,
            ..Default::default()
        });
        let program = if drop { unbound(program()) } else { program() };
        let build = build(&program);
        let mut exec = execute(&program, &[], &HASHER).expect("executes");
        exec.records.cap[5] = forged;
        forge_read(&mut exec, 0, forged);
        forge_read(&mut exec, 2, forged);
        let proof = prove(&build, &program, exec);
        verifies(&build, &proof).then_some(proof)
    };
    assert!(
        run(false).is_none(),
        "the node's cell read refuses the word"
    );
    let proof = run(true).expect("with the cell read dropped the forgery is accepted");
    assert_eq!(proof.public_words[0].1, forged);
}

/// ★ FORGERY: a tag cross-over. Read 1 (index 3 of A) claimed as tree B's
/// node 3, the lookup balanced by the counts: refused, A's handle holding A's
/// tag.
#[test]
fn a_read_through_one_trees_handle_is_not_answered_from_another() {
    let _fast = test_format::set(Some(2));
    let program = program();
    let build = build(&program);
    let mut exec = execute(&program, &[], &HASHER).expect("executes");
    forge_read(&mut exec, 1, word(B[3]));
    exec.records.cap_read[1].tag = FE::one();
    assert!(!verifies(&build, &prove(&build, &program, exec)));
}

/// The filled `LEAN_CAP` chunk holding table row `row` (wide), and `row`'s
/// row in it.
fn cap_chunk(
    filled: &mut super::proof::LfmFilled,
    row: usize,
) -> (
    &mut stark::trace::TraceTable<GoldilocksField, GoldilocksExtension>,
    usize,
) {
    let lean = filled.b3l_mut().expect("lean tables");
    let mut start = 0usize;
    for chunk in &mut lean.caps {
        let rows = chunk.rows();
        if row < start + rows {
            return match chunk {
                super::b3l_tables::HeldTable::Wide(t) => (t, row - start),
                _ => panic!("the cap chunk is held wide"),
            };
        }
        start += rows;
    }
    panic!("no cap chunk holds row {row}");
}

/// ★ FORGERY: a dropped multiplicity. Node 5 of A's count lowered from two to
/// one under its two reads: refused.
#[test]
fn a_dropped_multiplicity_is_refused() {
    use super::chips::cap::cols;
    let _fast = test_format::set(Some(2));
    let program = program();
    let build = build(&program);
    let exec = execute(&program, &[], &HASHER).expect("executes");
    let mut filled =
        super::proof::lfm_fill_executed_under(&program, exec, HASHER, 0.0, &Vec::new());
    let (t, r) = cap_chunk(&mut filled, 5);
    assert_eq!(
        *t.get_main(r, cols::M),
        FE::from(2u64),
        "two reads of A's node 5"
    );
    t.set_main(r, cols::M, FE::one());
    let proof = prove_wlfm_traces(&build, filled, true).expect("the prover does not check");
    assert!(!verifies(&build, &proof));
}

/// ★ FORGERY + MUTATION: a padding row's offer. The cap chunk's first
/// padding row (tag 0, index 0, its node read off) made to offer a forged
/// word once, read 4 (index 0 of A) answered with it and node 0's own count
/// dropped to match: refused by the padding gate; with the gate dropped, the
/// forgery is accepted.
#[test]
fn a_padding_rows_offer_is_refused_and_its_gate_is_load_bearing() {
    use super::chips::cap::cols;
    let _fast = test_format::set(Some(2));
    let forged = word(777);
    let program = program();
    let pad = A.len() + B.len();
    let run = |drop: bool| {
        let _mutant = test_cap::set(test_cap::CapMutant {
            drop_real_gate: drop,
            ..Default::default()
        });
        let build = build(&program);
        let mut exec = execute(&program, &[], &HASHER).expect("executes");
        forge_read(&mut exec, 4, forged);
        let mut filled =
            super::proof::lfm_fill_executed_under(&program, exec, HASHER, 0.0, &Vec::new());
        let (t, r) = cap_chunk(&mut filled, 0);
        assert_eq!(*t.get_main(r, cols::M), FE::one(), "one read of A's node 0");
        t.set_main(r, cols::M, FE::zero());
        let (t, r) = cap_chunk(&mut filled, pad);
        assert_eq!(*t.get_main(r, cols::IS_REAL), FE::zero(), "a padding row");
        assert_eq!(*t.get_main(r, cols::TAG), FE::zero());
        assert_eq!(*t.get_main(r, cols::INDEX), FE::zero());
        t.set_main(r, cols::M, FE::one());
        for (j, v) in forged.iter().enumerate() {
            t.set_main(r, cols::W0 + j, *v);
        }
        let proof = prove_wlfm_traces(&build, filled, true).expect("the prover does not check");
        verifies(&build, &proof).then_some(proof)
    };
    assert!(run(false).is_none(), "the padding gate refuses the offer");
    let proof = run(true).expect("with the gate dropped the forgery is accepted");
    assert_eq!(proof.public_words[4].1, forged);
}

/// An index past a table's last node has no row: the executor refuses it
/// (and the lookup could not match it).
#[test]
fn a_read_past_the_table_does_not_execute() {
    use super::executor::LfmExecError;
    let mut b = lean_builder();
    let block: [Cell; 4] = core::array::from_fn(|i| cell(&mut b, i as u64 + 1));
    b.b3_compress(B3Cv::Iv, B3Block::Lanes(block), 64, 11);
    let nodes: Vec<Cell> = A[..4].iter().map(|&v| cell(&mut b, v)).collect();
    let h = b.cap_table(&nodes);
    let past = b.felt_const(FE::from(4u64));
    let out = b.cap_read_at(h, past);
    b.public(out);
    let program = compile(b.finish());
    assert!(matches!(
        execute(&program, &[], &HASHER),
        Err(LfmExecError::CapIndexOutOfRange { tag: 0, index: 4 })
    ));
}

/// The admission validator refuses a cap table in a program with no lean
/// compression (no lean table would prove it), and a cap row whose tag is
/// not its table's ordinal.
#[test]
fn the_validator_refuses_cap_rows_out_of_shape() {
    use super::validator::{LfmViolation, validate};
    let mut b = lean_builder();
    let nodes: Vec<Cell> = A[..2].iter().map(|&v| cell(&mut b, v)).collect();
    let h = b.cap_table(&nodes);
    let x = b.felt_const(FE::one());
    let bits = b.bit_dec(x, 1);
    let out = b.cap_read(h, &bits);
    b.public(out);
    let no_lean = compile(b.finish());
    assert!(matches!(
        validate(&no_lean),
        Err(LfmViolation::CapWithoutLeanTables)
    ));

    let mut program = program();
    validate(&program).expect("admissible");
    // Tree B's first row retagged as tree A's: (0, 0) on two rows.
    program
        .groups
        .cap
        .set(A.len(), super::layout::cap::TAG, FE::zero());
    assert!(matches!(
        validate(&program),
        Err(LfmViolation::MalformedCapRow { row }) if row == A.len()
    ));
}

/// Cap tables are lean rows only: a builder that is not lean refuses one.
#[test]
#[should_panic(expected = "a cap table is a lean table")]
fn a_cap_table_outside_a_lean_program_is_refused_at_emission() {
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let nodes: Vec<Cell> = A[..2].iter().map(|&v| cell(&mut b, v)).collect();
    b.cap_table(&nodes);
}

/// The BLAKE3 W-leg verifies a child whose program reads caps: the leg is
/// emitted over the child's artifacts (its cap tables among them) and
/// executes over the child's proof.
#[test]
fn the_blake3_w_leg_walks_a_child_with_cap_reads() {
    use super::wlfm::{WlfmFamily, WlfmLevers, build_wlfm_under};
    let _fast = test_format::set(Some(2));
    let program = program();
    let child_build = build_wlfm_under(
        &program,
        HASHER,
        WlfmRate::Two,
        WlfmLevers::with_family(WlfmFamily::B3),
        true,
    )
    .expect("the child builds");
    let proof = prove_wlfm(&program, &child_build, &[]).expect("the child proves");
    verify_wlfm(&child_build.artifacts, &proof, &proof.public_words).expect("and verifies");
    let child = super::wlfm_leg::WlfmChild {
        artifacts: child_build.artifacts.clone(),
        num_public_words: proof.public_words.len(),
    };
    let leg = super::wlfm_leg::wlfm_leg_program(&child).expect("the BLAKE3 leg emits");
    let arena = super::wlfm_leg::wlfm_leg_arena(&child, &proof).expect("the leg's arena");
    execute(&leg, &[arena], &leg.hasher(HASHER)).expect("the leg executes over the child");
}
