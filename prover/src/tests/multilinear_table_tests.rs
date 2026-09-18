//! The multilinear table argument on **real VM tables**.
//!
//! Everything else exercising it uses fixtures of four or five columns. These
//! are the tables the VM actually proves, at the smallest scale each has:
//!
//! | table | columns | own constraints | shape it covers |
//! |---|---:|---:|---|
//! | EQ | 12 | 4 | compound packing `DWordWL`, a linear bus value with coefficients and constants, a constant element, **deduplicated** multiplicity |
//! | LT | 17 | several | signed comparison, deduplicated multiplicity |
//! | SHIFT | 29 | several | **no** deduplication — one row per operation, `μ = 1` |
//! | BYTEWISE | 26 | none | soundness entirely on the bus, the shape BITWISE/PAGE/REGISTER have |
//!
//! None of their buses balances on its own — they send to BITWISE and receive
//! from the ALU, whose counterparties are other tables — so what is checked is
//! each table's own argument, with the output fraction left to the caller.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as Fp,
};
use multilinear::whir_chain::{ChainConfig, GrindBits};
use stark::multilinear_air::Uniforms;
use stark::multilinear_table::{
    self, CommittedTable, CommittedTables, TableLayout, TableStatement,
};
use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use stark::constraints::builder::ConstraintSet;

use crate::tables::bytewise::{BytewiseOperation, generate_bytewise_trace};
use crate::tables::eq::{EqOperation, generate_eq_trace};
use crate::tables::lt::{LtOperation, generate_lt_trace};
use crate::tables::shift::{ShiftOperation, generate_shift_trace};
use crate::tables::trace_builder::Traces;
use crate::tables::{bytewise, eq, lt, shift};
use crate::test_utils::{
    ConcreteVmAir, create_bytewise_air, create_eq_air, create_lt_air, create_shift_air, run_asm_elf,
};
use executor::elf::Elf;
use executor::vm::execution::Executor;
use executor::vm::logs::Log;
use multilinear::whir_hash::KeccakWhir;

type ExtE = FieldElement<Ext>;

fn config() -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::default(),
    }
}

/// Four operations: equal and unequal, inverted and not. Distinct, so the
/// generator's deduplication leaves one row each.
fn eq_operations() -> Vec<EqOperation> {
    vec![
        EqOperation::new(7, 7, false),
        EqOperation::new(7, 9, false),
        EqOperation::new(1 << 40, 1 << 40, true),
        EqOperation::new(0, u64::MAX, true),
    ]
}

/// Signed and unsigned, both orders.
fn lt_operations() -> Vec<LtOperation> {
    vec![
        LtOperation::new(3, 9, false),
        LtOperation::new(9, 3, false),
        LtOperation::new(u64::MAX, 1, true),
        LtOperation::new(1, u64::MAX, true),
    ]
}

/// Left and right, arithmetic and logical, word and doubleword.
fn shift_operations() -> Vec<ShiftOperation> {
    vec![
        ShiftOperation::new(0x0123_4567_89ab_cdef, 4, false, false, false),
        ShiftOperation::new(0x0123_4567_89ab_cdef, 4, true, false, false),
        ShiftOperation::new(0xffff_ffff_ffff_0000, 8, true, true, false),
        ShiftOperation::new(0x0000_0000_dead_beef, 3, false, false, true),
    ]
}

fn bytewise_operations() -> Vec<BytewiseOperation> {
    vec![
        BytewiseOperation::new(0x0102_0304_0506_0708, 0x1112_1314_1516_1718, 0),
        BytewiseOperation::new(0, u64::MAX, 0),
        BytewiseOperation::new(7, 7, 1),
        BytewiseOperation::new(u64::MAX, 1, 1),
    ]
}

/// Runs the whole argument over one real table and returns its bus output.
fn argue<CS: ConstraintSet<Fp, Ext>>(
    air: &ConcreteVmAir<CS>,
    num_main_columns: usize,
    columns: Vec<Vec<FieldElement<Fp>>>,
) -> Result<(ExtE, ExtE), multilinear::Error> {
    let num_vars = columns[0].len().trailing_zeros() as usize;
    let layout = || {
        TableLayout::<Fp, Ext>::new(
            air.constraint_program(),
            air.constraints_meta(),
            air.bus_interactions(),
            num_main_columns,
            num_vars,
            Uniforms::default(),
        )
    };
    // The trace goes in as it is: base-field. Only the challenges are not.
    let table = CommittedTable::from_layout(layout()?, |col| columns[col as usize].clone())?;
    let committed = CommittedTables::<_, _, KeccakWhir>::commit(vec![table], &config())?;

    let mut prover = DefaultTranscript::<Ext>::new(b"vm-table");
    let proof = multilinear_table::multi_prove(&committed, &config(), &mut prover, None)?;

    // One commitment for the whole trace, one opening, one pass over the rows.
    assert_eq!(proof.roots.len(), 1);
    assert_eq!(proof.columns.len(), 1);
    assert_eq!(proof.columns[0].polys.len(), 1);
    assert_eq!(proof.tables[0].constraint.sumcheck.rounds.len(), num_vars);

    // The verifier rebuilds the layout from the AIR alone — no trace — and the
    // roots it absorbs come out of the proof.
    let verifier_layout = layout()?;
    let statement = verifier_layout.statement();

    // A single table's bus need not balance on its own, so the share it owes
    // is whatever it produced: what is under test here is the table, not the
    // balance.
    let owed = multilinear_table::contribution(&proof.tables[0].bus_output)
        .ok_or(multilinear::Error::BusImbalance)?;
    let mut verifier = DefaultTranscript::<Ext>::new(b"vm-table");
    multilinear_table::multi_verify::<_, _, _, KeccakWhir>(
        &proof,
        &[statement],
        std::slice::from_ref(committed.groups()[0].layout()),
        std::slice::from_ref(committed.groups()[0].domain()),
        committed.sizes(),
        &owed,
        &config(),
        &mut verifier,
        None,
    )?;
    Ok(proof.tables[0].bus_output)
}

fn argue_eq(columns: Vec<Vec<FieldElement<Fp>>>) -> Result<(ExtE, ExtE), multilinear::Error> {
    let options = ProofOptions::default_test_options();
    argue(&create_eq_air(&options), eq::cols::NUM_COLUMNS, columns)
}

/// The milestone: tables the VM actually proves — their own constraints and
/// their buses — each argued in one sumcheck against one commitment for the
/// whole trace.
#[test]
fn the_eq_table_argues_end_to_end() {
    let trace = generate_eq_trace(&eq_operations());
    assert_eq!(trace.columns_main().len(), eq::cols::NUM_COLUMNS);
    argue_eq(trace.columns_main()).unwrap();
}

#[test]
fn the_lt_table_argues_end_to_end() {
    let options = ProofOptions::default_test_options();
    let trace = generate_lt_trace(&lt_operations());
    argue(
        &create_lt_air(&options),
        lt::cols::NUM_COLUMNS,
        trace.columns_main(),
    )
    .unwrap();
}

/// SHIFT does not deduplicate: one row per operation with `μ = 1`, which is a
/// different multiplicity shape from EQ's and LT's.
#[test]
fn the_shift_table_argues_end_to_end() {
    let options = ProofOptions::default_test_options();
    let trace = generate_shift_trace(&shift_operations());
    argue(
        &create_shift_air(&options),
        shift::cols::NUM_COLUMNS,
        trace.columns_main(),
    )
    .unwrap();
}

/// BYTEWISE has no constraints of its own: its soundness is entirely its bus,
/// which is the shape BITWISE, PAGE and REGISTER have.
#[test]
fn the_bytewise_table_argues_end_to_end() {
    let options = ProofOptions::default_test_options();
    let trace = generate_bytewise_trace(&bytewise_operations());
    argue(
        &create_bytewise_air(&options),
        bytewise::cols::NUM_COLUMNS,
        trace.columns_main(),
    )
    .unwrap();
}

/// And a trace that breaks one of the table's own constraints is rejected.
#[test]
fn a_broken_eq_row_is_rejected() {
    let trace = generate_eq_trace(&eq_operations());
    let mut columns = trace.columns_main();
    // `res = eq XOR invert` is one of the four; flipping the output breaks it.
    columns[eq::cols::RES][0] += FieldElement::<Fp>::one();

    assert!(argue_eq(columns).is_err());
}

/// The multiplicity column is what the buses weigh by, so perturbing it moves
/// the table's share of the bus — the quantity a multi-table proof sums.
#[test]
fn the_multiplicity_column_moves_the_bus_share() {
    let trace = generate_eq_trace(&eq_operations());
    let honest = argue_eq(trace.columns_main()).unwrap();

    let mut columns = trace.columns_main();
    columns[eq::cols::MU][1] += FieldElement::<Fp>::one();
    let tampered = argue_eq(columns).unwrap();

    let share = |(p, q): (ExtE, ExtE)| p * q.inv().unwrap();
    assert_ne!(share(honest), share(tampered));
}

// ---------------------------------------------------------------
// Every live table of a real run.
// ---------------------------------------------------------------

/// Lays out one table through the `AIR` trait alone, which is all a verifier
/// has: the width and the height, never the values.
fn layout_dyn<'a>(
    air: &'a dyn AIR<Field = Fp, FieldExtension = Ext, PublicInputs = ()>,
    num_main_columns: usize,
    num_vars: usize,
) -> Result<TableLayout<'a, Fp, Ext>, multilinear::Error> {
    TableLayout::<Fp, Ext>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        num_main_columns,
        num_vars,
        Uniforms::default(),
    )
}

/// Runs a program and proves **every table it leaves in one multi-table
/// proof**, verified by `multi_verify` — which is what checks the bus balance
/// against what the statement owes.
///
/// The COMMIT table sends the program's public output on a bus whose
/// counterparty is not another table but the statement, so the tables only sum
/// to zero for a program that outputs nothing. `compute_commit_bus_offset` is
/// the same quantity the univariate verifier demands.
fn prove_and_verify_all_tables(elf: Elf, logs: &[Log]) -> usize {
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, logs, &Default::default(), &[]).unwrap();
    let public_output = traces.public_output_bytes.clone();

    let options = ProofOptions::default_test_options();
    let table_counts = traces.table_counts();
    let airs = crate::VmAirs::new(
        &elf,
        &options,
        true,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let pairs = airs.air_trace_pairs(&mut traces);

    // The shape of every table: width and height, which is all the verifier
    // needs to rebuild the layouts.
    let shapes: Vec<(usize, usize)> = pairs
        .iter()
        .map(|(air, trace, _)| {
            let columns = trace.columns_main();
            assert!(
                !columns.is_empty() && columns[0].len().is_power_of_two(),
                "{}: {} columns of {} rows",
                air.name(),
                columns.len(),
                columns.first().map_or(0, Vec::len)
            );
            (columns.len(), columns[0].len().trailing_zeros() as usize)
        })
        .collect();

    let mut tables = Vec::with_capacity(pairs.len());
    for ((air, trace, _), &(width, num_vars)) in pairs.iter().zip(&shapes) {
        let columns = trace.columns_main();
        let layout =
            layout_dyn(*air, width, num_vars).unwrap_or_else(|e| panic!("{}: {e:?}", air.name()));
        tables.push(
            CommittedTable::from_layout(layout, |col| columns[col as usize].clone())
                .unwrap_or_else(|e| panic!("{}: {e:?}", air.name())),
        );
    }
    let count = tables.len();
    // Every table's columns in one commitment: 55 of them still open once.
    let committed =
        CommittedTables::<_, _, KeccakWhir>::commit(tables, &config()).expect("commit every table");

    let mut prover = DefaultTranscript::<Ext>::new(b"vm-sweep");
    let proof = multilinear_table::multi_prove(&committed, &config(), &mut prover, None)
        .expect("prove every table");

    // Verifier side: the layouts are rebuilt from the AIRs and the shapes, with
    // no trace and no committed table in reach. The roots come from the proof,
    // and so does the stack — which the verifier rebuilds from the shapes too.
    let verifier_layouts: Vec<TableLayout<'_, Fp, Ext>> = pairs
        .iter()
        .zip(&shapes)
        .map(|((air, _, _), &(width, num_vars))| {
            layout_dyn(*air, width, num_vars).expect("rebuild the layout")
        })
        .collect();
    let statements: Vec<TableStatement<'_, Fp, Ext>> = verifier_layouts
        .iter()
        .map(TableLayout::statement)
        .collect();
    let stacked = multilinear_table::global_layout(&shapes).expect("rebuild the stack");
    let domain =
        multilinear::whir::Domain::<Fp>::new(stacked.n_stack() + config().log_blowup).unwrap();

    // The verifier redraws the shared LogUp challenges, so the offset has to be
    // computed against the same ones — which means replaying the transcript up
    // to that point exactly as `multi_verify` will.
    // The block is CALLED rather than re-spelled, so this promise stays true
    // when the block changes — which is how the epoch path's replay broke.
    let mut probe = DefaultTranscript::<Ext>::new(b"vm-sweep");
    let (z, alpha, _beta) =
        multilinear_table::absorb_roots_and_challenge::<Ext, _>(&mut probe, &proof.roots, &[]);
    // `start_index` is the carried x254: zero for a monolithic proof.
    let expected = crate::compute_commit_bus_offset(&public_output, 0, &z, &alpha)
        .expect("the commit fingerprints are invertible");

    let mut verifier = DefaultTranscript::<Ext>::new(b"vm-sweep");
    multilinear_table::multi_verify::<_, _, _, KeccakWhir>(
        &proof,
        &statements,
        std::slice::from_ref(&stacked),
        std::slice::from_ref(&domain),
        &[statements.len()],
        &expected,
        &config(),
        &mut verifier,
        None,
    )
    .expect("the whole table set verifies");

    count
}

/// **The whole VM through the multilinear path**: every live table of a real
/// run proved in one multi-table proof and verified, buses included.
///
/// The traces come from the executor, not from hand-written operations, so the
/// widths, the interaction counts, the packings and the multiplicity patterns
/// are whatever the VM actually produces.
#[test]
fn every_live_table_is_proved_and_verified() {
    let (elf, logs, _) = run_asm_elf("sub");
    let argued = prove_and_verify_all_tables(elf, &logs);
    assert!(argued >= 20, "expected the full table set, argued {argued}");
}

/// The same over the whole 64-bit instruction set, which lights up the tables a
/// two-instruction program never reaches.
#[test]
fn the_whole_instruction_set_is_proved_and_verified() {
    let (elf, logs, _) = run_asm_elf("all_instructions_64");
    assert!(prove_and_verify_all_tables(elf, &logs) >= 20);
}

/// And over a Rust program that calls the keccak precompile — which brings
/// KECCAK, KECCAK_RND and KECCAK_RC in, **and** writes public output, so the
/// statement's share of the bus is load-bearing here and nowhere else.
#[test]
fn a_program_using_a_precompile_is_proved_and_verified() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("executor/program_artifacts/rust/keccak.elf");
    let bytes = std::fs::read(&root).unwrap_or_else(|_| panic!("read {}", root.display()));
    let elf = Elf::load(&bytes).expect("load keccak.elf");
    let logs = Executor::new(&elf, vec![])
        .expect("executor")
        .run()
        .expect("run")
        .logs;

    assert!(prove_and_verify_all_tables(elf, &logs) >= 20);
}

/// A proof is only a proof if it can leave the process. Round-trips a real
/// table's proof through both formats the univariate path uses, and verifies
/// the **deserialized** one — a format that loses a field would still verify if
/// we only checked the original.
#[test]
fn a_real_table_proof_survives_serialization() {
    let options = ProofOptions::default_test_options();
    let air = create_eq_air(&options);
    let columns = generate_eq_trace(&eq_operations()).columns_main();
    let num_vars = columns[0].len().trailing_zeros() as usize;
    let layout = || {
        TableLayout::<Fp, Ext>::new(
            air.constraint_program(),
            air.constraints_meta(),
            air.bus_interactions(),
            eq::cols::NUM_COLUMNS,
            num_vars,
            Uniforms::default(),
        )
        .unwrap()
    };

    let table = CommittedTable::from_layout(layout(), |col| columns[col as usize].clone()).unwrap();
    let committed = CommittedTables::<_, _, KeccakWhir>::commit(vec![table], &config()).unwrap();
    let mut prover = DefaultTranscript::<Ext>::new(b"serialized");
    let proof = multilinear_table::multi_prove(&committed, &config(), &mut prover, None).unwrap();

    let json = serde_json::to_vec(&proof).expect("serde round trip");
    let from_json: multilinear_table::MultiProof<Fp, Ext> =
        serde_json::from_slice(&json).expect("serde round trip");

    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("rkyv round trip");
    let from_rkyv: multilinear_table::MultiProof<Fp, Ext> =
        rkyv::from_bytes::<_, rkyv::rancor::Error>(&bytes).expect("rkyv round trip");

    // The layout the verifier rebuilds, with the trace out of scope.
    let verifier_layout = layout();
    let statement = verifier_layout.statement();
    let owed = multilinear_table::contribution(&proof.tables[0].bus_output).unwrap();

    for (label, round_tripped) in [("serde", &from_json), ("rkyv", &from_rkyv)] {
        // The roots travel in the proof, so a format that dropped them would
        // fail here rather than pass on the original's.
        let mut verifier = DefaultTranscript::<Ext>::new(b"serialized");
        multilinear_table::multi_verify::<_, _, _, KeccakWhir>(
            round_tripped,
            &[statement],
            std::slice::from_ref(committed.groups()[0].layout()),
            std::slice::from_ref(committed.groups()[0].domain()),
            committed.sizes(),
            &owed,
            &config(),
            &mut verifier,
            None,
        )
        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
    }

    assert!(!bytes.is_empty());
}

/// Only the main columns are committed: the LogUp auxiliary columns the
/// univariate path needs are gone, replaced by the fraction tree. Checked on
/// every table, since the number of interactions — and so of aux columns —
/// differs.
#[test]
fn the_argument_commits_no_auxiliary_columns() {
    let options = ProofOptions::default_test_options();

    let eq_air = create_eq_air(&options);
    let lt_air = create_lt_air(&options);
    let shift_air = create_shift_air(&options);
    let bytewise_air = create_bytewise_air(&options);

    let cases: [(
        &dyn AIR<Field = Fp, FieldExtension = Ext, PublicInputs = ()>,
        usize,
    ); 4] = [
        (&eq_air, eq::cols::NUM_COLUMNS),
        (&lt_air, lt::cols::NUM_COLUMNS),
        (&shift_air, shift::cols::NUM_COLUMNS),
        (&bytewise_air, bytewise::cols::NUM_COLUMNS),
    ];

    for (air, num_main_columns) in cases {
        let program = air.constraint_program();
        // The program carries the table's own constraints and then the LogUp
        // ones, which this path drops.
        assert!(
            program.roots.len() > program.num_base,
            "the fixture must have LogUp constraints to drop"
        );
        assert!(
            air.trace_layout().1 > 0,
            "the univariate path commits auxiliary columns"
        );
        assert_eq!(air.trace_layout().0, num_main_columns);
    }
}

/// And the committed count on a real table is exactly its main width, in one
/// stacked commitment.
#[test]
fn a_real_table_is_one_commitment_for_its_main_columns() {
    let options = ProofOptions::default_test_options();
    let air = create_lt_air(&options);
    let trace = generate_lt_trace(&lt_operations());
    let columns = trace.columns_main();
    let num_vars = columns[0].len().trailing_zeros() as usize;
    // The trace goes in as it is: base-field. Only the challenges are not.
    let table = CommittedTable::<Fp, Ext>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        lt::cols::NUM_COLUMNS,
        num_vars,
        Uniforms::default(),
        |col| columns[col as usize].clone(),
    )
    .unwrap();

    assert_eq!(table.num_committed_columns(), lt::cols::NUM_COLUMNS);
    // And they all ride in one stacked polynomial, alone or alongside others.
    let committed = CommittedTables::<_, _, KeccakWhir>::commit(vec![table], &config()).unwrap();
    assert_eq!(committed.roots().len(), 1);
}

// =========================================================================
// W1-B step 2: the out-of-band preprocessed opening, prover side
// =========================================================================

/// Builds a table from the EQ trace plus a commitment over two of its columns,
/// standing in for DECODE's five ELF-derived ones at a scale a test can run.
fn eq_table_and_prepared_columns() -> (Vec<Vec<FieldElement<Fp>>>, Vec<multilinear::mle::Mle<Fp>>) {
    let trace = eq::generate_eq_trace(&eq_operations());
    let columns = trace.columns_main();
    let prepared: Vec<multilinear::mle::Mle<Fp>> = columns[..2]
        .iter()
        .map(|c| multilinear::mle::Mle::new(c.clone()).expect("mle"))
        .collect();
    (columns, prepared)
}

/// Proves the EQ table, optionally against a prepared commitment over its first
/// two columns, and hands back the proof.
///
/// ⚠ The TRACE IS AN ARGUMENT, not regenerated here, and that is load-bearing:
/// `generate_eq_trace` orders its rows by `HashMap` iteration, so two calls
/// produce two different row orders, two different commitments and two
/// different roots. Comparing proofs from two independently generated traces
/// would compare that instead of the thing under test — which is exactly what
/// the first draft of this helper did, and what `a850dd29` fixed for DECODE.
fn prove_eq_with_prepared(
    columns: &[Vec<FieldElement<Fp>>],
    prepared_columns: &[multilinear::mle::Mle<Fp>],
    prepared: bool,
    table_index: usize,
) -> Result<multilinear_table::MultiProof<Fp, Ext>, multilinear::Error> {
    let options = ProofOptions::default_test_options();
    let air = create_eq_air(&options);
    let num_vars = columns[0].len().trailing_zeros() as usize;

    let layout = TableLayout::<Fp, Ext>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        eq::cols::NUM_COLUMNS,
        num_vars,
        Uniforms::default(),
    )?;
    let table = CommittedTable::from_layout(layout, |col| columns[col as usize].clone())?;
    let committed = CommittedTables::<_, _, KeccakWhir>::commit(vec![table], &config())?;

    let refs: Vec<&multilinear::mle::Mle<Fp>> = prepared_columns.iter().collect();
    let out_of_band = multilinear::stacked_eval::StackedCommitment::<Fp, KeccakWhir>::commit(
        multilinear_table::global_layout(&[(refs.len(), num_vars)])?,
        &refs,
        None,
        &config(),
    )?;

    let mut prover = DefaultTranscript::<Ext>::new(b"w1b-step2");
    multilinear_table::multi_prove(
        &committed,
        &config(),
        &mut prover,
        prepared.then(|| multilinear_table::Prepared {
            commitment: &out_of_band,
            columns: &refs,
            table: table_index,
        }),
    )
}

/// ★★★ THE PROVER-SIDE WIRING GUARD: the prepared root reaches the sponge.
///
/// Nothing verifies the opening until step 3, so the question this answers is
/// the one that has gone wrong twice on this branch — **is the feature wired at
/// all?** The transcript defaulting to keccak while everything else moved, and
/// the grind dispatching keccak-only with a correct RPX kernel sitting unused,
/// were both "not wired", not "wired wrongly".
///
/// The absorb lands before `z`, so supplying a prepared commitment must move
/// the whole challenge stream and therefore the proof. If the proofs are equal,
/// the root did not reach the sponge — whatever else the opening contains.
///
/// ⚠ This is why a `preprocessed.is_some()` assertion would not do: the field
/// can be populated by an opening the transcript never saw.
///
/// # ⛔ WHAT IT STILL DOES NOT CATCH, measured not assumed
///
/// Moving the absorb to AFTER the first challenge leaves this test green. Run
/// as a mutation: the proof still differs from the no-prepared case, because
/// `alpha` and `beta` are drawn after the absorb wherever it sits, so the
/// inequality below is satisfied by a root that binds `z` to nothing.
///
/// So the three placements split cleanly by who catches them:
///
/// * **not absorbed at all** — this test;
/// * **aimed at the wrong table** — its sibling below;
/// * **absorbed after the first challenge** — NEITHER, today.
///
/// The last one is the dangerous one and it is currently guarded only by the
/// specification: `the_derived_root_is_absorbed_after_the_carried_ones` proves
/// the orderings are distinguishable, but nothing yet asserts which one the
/// production code uses. A round-trip will not close it either — prover and
/// verifier share the position, so a consistently late absorb verifies.
///
/// Step 3 closes it by comparing the verifier's roots block against an
/// independently built transcript, which is the same shape as every other check
/// on this branch that held: compare against a CONSTRUCTION, not against the
/// other side.
#[test]
fn a_prepared_commitment_moves_the_challenge_stream() {
    // One trace, two proves. See `prove_eq_with_prepared`'s note.
    let (columns, prepared_columns) = eq_table_and_prepared_columns();
    let without =
        prove_eq_with_prepared(&columns, &prepared_columns, false, 0).expect("prove without");
    let with = prove_eq_with_prepared(&columns, &prepared_columns, true, 0).expect("prove with");

    assert!(
        without.preprocessed.is_none(),
        "no prepared commitment was supplied, so nothing should be opened"
    );
    assert!(
        with.preprocessed.is_some(),
        "a prepared commitment was supplied and produced no opening"
    );

    // The roots the proof CARRIES are the same: the derived root is not one of
    // them, which is the other half of the design.
    assert_eq!(
        without.roots, with.roots,
        "the prepared root leaked into `MultiProof::roots`; it is derived by the \
         verifier and must never be carried"
    );

    // …and the challenges moved, which is the absorb having happened.
    let a = rkyv::to_bytes::<rkyv::rancor::Error>(&without.tables).expect("serialize");
    let b = rkyv::to_bytes::<rkyv::rancor::Error>(&with.tables).expect("serialize");
    assert_ne!(
        a.as_ref(),
        b.as_ref(),
        "the table arguments are byte-identical with and without the prepared \
         commitment — its root never reached the transcript, so it binds nothing"
    );
}

/// ★ The table index is checked, not trusted.
///
/// The opening binds the pinned columns to ONE table's reduced point. An index
/// that names no table must be an error rather than silently opening at
/// whatever point happens to be first.
#[test]
fn a_prepared_commitment_aimed_at_no_table_is_rejected() {
    let (columns, prepared_columns) = eq_table_and_prepared_columns();
    let err = prove_eq_with_prepared(&columns, &prepared_columns, true, 7)
        .expect_err("an out-of-range table index must fail");
    assert!(
        matches!(err, multilinear::Error::UnknownPolynomial { .. }),
        "expected the index to be reported as unknown, got {err:?}"
    );
}
