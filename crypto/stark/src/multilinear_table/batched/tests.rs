//! The batched argue (D-BATCH B-2) on the repo's multi-table example: round
//! trips across heights and bins, every negative D-BATCH §7 names, and the
//! mutation tests that show each new verifier check is load-bearing.

use super::*;
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as Fp,
};
use multilinear::whir_chain::{ChainFormat, GrindBits};

use crate::constraints::builder::{ConstraintBuilder, ConstraintSet, EmptyConstraints, RowDomain};
use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};
use crate::proof::options::ProofOptions;
use crate::traits::AIR;

type FE = FieldElement<Fp>;
type ExtE = FieldElement<Ext>;
type Air<CS> = AirWithBuses<Fp, Ext, NullBoundaryConstraintBuilder, (), CS>;
type Proof = BatchedMultiProof<Fp, Ext>;

const SEED: &[u8] = b"batched-argue";

fn config(argue: ArgueFormat) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::default(),
        format: ChainFormat {
            argue,
            ..ChainFormat::DEFAULT
        },
    }
}

fn batched(bin_log_cells: u8) -> ChainConfig {
    config(ArgueFormat::Batched { bin_log_cells })
}

/// The ADD table's own constraint: every row really is an addition.
struct AddConstraints;

impl ConstraintSet<Fp, Ext> for AddConstraints {
    fn max_degree(&self) -> usize {
        1
    }

    fn eval<B: ConstraintBuilder<Fp, Ext>>(&self, b: &mut B) {
        let a = b.main(0, 0);
        let addend = b.main(0, 1);
        let sum = b.main(0, 2);
        b.emit_base(0, a + addend - sum);
    }
}

/// The MUL table's, one degree higher.
struct MulConstraints;

impl ConstraintSet<Fp, Ext> for MulConstraints {
    fn max_degree(&self) -> usize {
        2
    }

    fn eval<B: ConstraintBuilder<Fp, Ext>>(&self, b: &mut B) {
        let a = b.main(0, 0);
        let factor = b.main(0, 1);
        let product = b.main(0, 2);
        b.emit_base(0, a * factor - product);
    }
}

/// The MUL table with a transition too — its multiplicity is the same on
/// every row but the last — so a factor reads the NEXT row: a shifted table,
/// which keeps its claim reduction.
struct ShiftedMulConstraints;

impl ConstraintSet<Fp, Ext> for ShiftedMulConstraints {
    fn max_degree(&self) -> usize {
        2
    }

    fn eval<B: ConstraintBuilder<Fp, Ext>>(&self, b: &mut B) {
        let a = b.main(0, 0);
        let factor = b.main(0, 1);
        let product = b.main(0, 2);
        b.emit_base(0, a * factor - product);
        let mult = b.main(0, 3);
        let next = b.main(1, 3);
        b.emit_base_rows(1, RowDomain::except_last(1), next - mult);
    }
}

struct Airs {
    cpu: Air<EmptyConstraints>,
    add: Air<AddConstraints>,
    mul: Air<MulConstraints>,
    shifted: Air<ShiftedMulConstraints>,
}

fn airs() -> Airs {
    let options = ProofOptions::default_test_options();
    let with = |interactions: &[BusInteraction]| AuxiliaryTraceBuildData {
        interactions: interactions.to_vec(),
    };
    let add_bus = new_add_air_with_lookup(&options);
    let mul_bus = new_mul_air_with_lookup(&options);
    Airs {
        cpu: new_cpu_air_with_lookup(&options),
        add: AirWithBuses::new(
            4,
            with(add_bus.bus_interactions()),
            &options,
            1,
            AddConstraints,
        ),
        mul: AirWithBuses::new(
            4,
            with(mul_bus.bus_interactions()),
            &options,
            1,
            MulConstraints,
        ),
        shifted: AirWithBuses::new(
            4,
            with(mul_bus.bus_interactions()),
            &options,
            1,
            ShiftedMulConstraints,
        ),
    }
}

/// CPU `2^(a+1)` rows (half add, half multiply), ADD `2^(a+pad)` (its adds,
/// then zero rows that add up and receive nothing), MUL `2^a`.
fn columns(a: usize, pad: usize) -> [Vec<Vec<FE>>; 3] {
    let count = 1u64 << a;
    let mut cpu = vec![Vec::new(); 5];
    let mut add = vec![Vec::new(); 4];
    let mut mul = vec![Vec::new(); 4];
    let push = |table: &mut Vec<Vec<FE>>, row: &[FE]| {
        for (column, value) in table.iter_mut().zip(row) {
            column.push(*value);
        }
    };
    for i in 0..count {
        let (x, y) = (FE::from(i + 1), FE::from(3 * i + 7));
        push(&mut cpu, &[FE::one(), FE::zero(), x, y, x + y]);
        push(&mut add, &[x, y, x + y, FE::one()]);
    }
    for _ in count..(count << pad) {
        push(&mut add, &[FE::zero(); 4]);
    }
    for i in 0..count {
        let (x, y) = (FE::from(i + 2), FE::from(5 * i + 3));
        push(&mut cpu, &[FE::zero(), FE::one(), x, y, x * y]);
        push(&mut mul, &[x, y, x * y, FE::one()]);
    }
    [cpu, add, mul]
}

fn table<'a, CS: ConstraintSet<Fp, Ext>>(
    air: &'a Air<CS>,
    columns: &[Vec<FE>],
) -> CommittedTable<'a, Fp, Ext> {
    let num_vars = columns[0].len().trailing_zeros() as usize;
    let owned = columns.to_vec();
    CommittedTable::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        columns.len(),
        num_vars,
        Uniforms::default(),
        |col| owned[col as usize].clone(),
    )
    .expect("a table")
}

/// CPU, ADD and MUL (or the shifted MUL) over `columns(a, pad)`.
fn tables<'a>(
    airs: &'a Airs,
    cols: &[Vec<Vec<FE>>; 3],
    shifted: bool,
) -> Vec<CommittedTable<'a, Fp, Ext>> {
    vec![
        table(&airs.cpu, &cols[0]),
        table(&airs.add, &cols[1]),
        if shifted {
            table(&airs.shifted, &cols[2])
        } else {
            table(&airs.mul, &cols[2])
        },
    ]
}

fn commit<'a>(
    tables: Vec<CommittedTable<'a, Fp, Ext>>,
    cfg: &ChainConfig,
) -> CommittedTables<'a, Fp, Ext, KeccakWhir> {
    CommittedTables::commit(tables, cfg).expect("committed")
}

fn prove(
    committed: &CommittedTables<'_, Fp, Ext, KeccakWhir>,
    cfg: &ChainConfig,
    faults: ProverFaults,
) -> Proof {
    let mut transcript = DefaultTranscript::<Ext>::new(SEED);
    prove_batched_with(committed, cfg, &mut transcript, None, faults).expect("the prover runs")
}

fn verify_under(
    committed: &CommittedTables<'_, Fp, Ext, KeccakWhir>,
    statements: &[TableStatement<'_, Fp, Ext>],
    proof: &Proof,
    cfg: &ChainConfig,
    checks: VerifierChecks,
) -> Result<(), MlError> {
    let mut transcript = DefaultTranscript::<Ext>::new(SEED);
    verify_batched_with::<_, _, _, KeccakWhir>(
        proof,
        statements,
        std::slice::from_ref(committed.groups()[0].layout()),
        std::slice::from_ref(committed.groups()[0].domain()),
        committed.sizes(),
        &ExtE::zero(),
        cfg,
        &mut transcript,
        None,
        false,
        checks,
    )
}

fn verify(
    committed: &CommittedTables<'_, Fp, Ext, KeccakWhir>,
    proof: &Proof,
    cfg: &ChainConfig,
) -> Result<(), MlError> {
    let statements: Vec<_> = committed.tables().iter().map(|t| t.statement()).collect();
    verify_under(committed, &statements, proof, cfg, VerifierChecks::ALL)
}

fn verify_with_checks(
    committed: &CommittedTables<'_, Fp, Ext, KeccakWhir>,
    proof: &Proof,
    cfg: &ChainConfig,
    checks: VerifierChecks,
) -> Result<(), MlError> {
    let statements: Vec<_> = committed.tables().iter().map(|t| t.statement()).collect();
    verify_under(committed, &statements, proof, cfg, checks)
}

/// A proof of `columns(a, pad)` under `cap`, and its verdict.
fn round_trip(a: usize, pad: usize, cap: u8, shifted: bool) -> Result<(), MlError> {
    let airs = airs();
    let cfg = batched(cap);
    let committed = commit(tables(&airs, &columns(a, pad), shifted), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    verify(&committed, &proof, &cfg)
}

fn shape(num_vars: usize, input_vars: usize) -> ArgueShape {
    ArgueShape {
        num_vars,
        input_vars,
        degree: 2,
        shifted: false,
    }
}

// ─── the plan ─────────────────────────────────────────────────────────────

/// First-fit-decreasing by `k`, ties by index; a table at or over the cap
/// alone; bins in the order they open, each in table order.
#[test]
fn the_plan_packs_first_fit_decreasing() {
    let shapes = [
        shape(3, 4),  // 16 cells
        shape(5, 6),  // 64
        shape(1, 2),  // 4
        shape(4, 5),  // 32
        shape(2, 3),  // 8
        shape(8, 10), // over the cap
        shape(3, 5),  // 32
    ];
    let plan = argue_plan(&shapes, 6); // cap 64
    // Order: 5 (alone), 1 (64 fills a bin), 3 (32), 6 (32 joins 3), 0 (16),
    // 4 (8 joins 0), 2 (4 joins 0).
    assert_eq!(plan.bins, vec![vec![5], vec![1], vec![3, 6], vec![0, 2, 4]]);
    assert_eq!(plan.num_vars, 8);
    assert_eq!(plan.degree, 2);
    // A roomy cap puts every table in one bin.
    assert_eq!(
        argue_plan(&shapes, 27).bins,
        vec![vec![0, 1, 2, 3, 4, 5, 6]]
    );
    // A cap of one cell: every table alone, largest first.
    assert_eq!(
        argue_plan(&shapes, 0).bins,
        vec![
            vec![5],
            vec![1],
            vec![3],
            vec![6],
            vec![0],
            vec![4],
            vec![2]
        ]
    );
}

/// The example's shapes, read off the statements: CPU `k = n + 1` (two
/// interactions), ADD and MUL `k = n` (one).
#[test]
fn the_shapes_come_from_the_statement() {
    let airs = airs();
    let committed = commit(tables(&airs, &columns(3, 1), true), &batched(27));
    let shapes: Vec<ArgueShape> = committed
        .tables()
        .iter()
        .map(|t| ArgueShape::of(&t.statement()))
        .collect();
    assert_eq!(
        shapes,
        vec![
            ArgueShape {
                num_vars: 4,
                input_vars: 5,
                degree: 2,
                shifted: false
            },
            ArgueShape {
                num_vars: 4,
                input_vars: 4,
                degree: 2,
                shifted: false
            },
            ArgueShape {
                num_vars: 3,
                input_vars: 3,
                degree: 3,
                shifted: true
            },
        ]
    );
}

// ─── round trips ─────────────────────────────────────────────────────────

/// ★ Three tables of different heights, virtually padded to the tallest, in
/// one bin, in bins of one or two, and with every table alone.
#[test]
fn tables_of_different_heights_round_trip_across_bins() {
    for (a, pad) in [(2, 0), (3, 1), (3, 2), (1, 3)] {
        for cap in [27u8, 5, 4, 3, 2, 0] {
            round_trip(a, pad, cap, false)
                .unwrap_or_else(|e| panic!("a={a} pad={pad} cap={cap}: {e:?}"));
        }
    }
}

/// One-row tables (`n_T = 0`: ADD and MUL at `a = 0`), a rootless one (CPU),
/// and a table alone in its bin.
#[test]
fn edge_shapes_round_trip() {
    round_trip(0, 0, 27, false).expect("two one-row tables and a two-row one");
    round_trip(0, 0, 0, false).expect("each alone");
    round_trip(0, 3, 27, false).expect("a one-row MUL beside taller tables");
}

/// A table with a shifted read keeps its claim reduction, at `r_{<n_T}`, and
/// the others still read their columns directly.
#[test]
fn a_shifted_table_keeps_its_reduction() {
    let airs = airs();
    let cfg = batched(27);
    let committed = commit(tables(&airs, &columns(3, 1), true), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    assert!(proof.argue.reduces[0].is_none());
    assert!(proof.argue.reduces[1].is_none());
    assert!(proof.argue.reduces[2].is_some());
    verify(&committed, &proof, &cfg).expect("verifies");
    round_trip(2, 2, 3, true).expect("across bins too");
}

/// The shape of a proof: one ladder per bin, `max k` steps each, and ONE
/// constraint sumcheck of `n_max` rounds for every table.
#[test]
fn one_constraint_sumcheck_for_every_table() {
    let airs = airs();
    let cfg = batched(27);
    let committed = commit(tables(&airs, &columns(3, 1), false), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    assert_eq!(proof.argue.gkr.len(), 1);
    assert_eq!(proof.argue.gkr[0].layers.len(), 5);
    assert_eq!(proof.argue.constraint.rounds.len(), 4);
    assert!(
        proof
            .argue
            .constraint
            .rounds
            .iter()
            .all(|r| r.evaluations.len() == 3),
        "D_max = 3 (MUL's degree-2 constraint, plus its weight)"
    );
    assert_eq!(proof.roots.len(), 1);
    assert_eq!(proof.columns.len(), 1);
}

// ─── negatives: tampered proofs ──────────────────────────────────────────

/// The fixture the tamper tests share: `a = 3`, ADD padded once, one bin —
/// CPU `k = 5`, ADD `k = 4`, MUL `k = 3`, so ladder steps 0..3 have three
/// trees.
fn fixture<'a>(airs: &'a Airs) -> (CommittedTables<'a, Fp, Ext, KeccakWhir>, Proof, ChainConfig) {
    let cfg = batched(27);
    let committed = commit(tables(airs, &columns(3, 1), false), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    verify(&committed, &proof, &cfg).expect("the untampered proof verifies");
    (committed, proof, cfg)
}

fn tampered(tamper: impl FnOnce(&mut Proof)) -> Result<(), MlError> {
    let airs = airs();
    let (committed, mut proof, cfg) = fixture(&airs);
    tamper(&mut proof);
    verify(&committed, &proof, &cfg)
}

#[test]
fn a_wrong_bus_output_is_refused() {
    for t in 0..3 {
        assert!(
            tampered(|p| p.argue.bus_outputs[t].0 += ExtE::one()).is_err(),
            "table {t}'s numerator"
        );
        assert!(
            tampered(|p| p.argue.bus_outputs[t].1 += ExtE::one()).is_err(),
            "table {t}'s denominator"
        );
    }
}

#[test]
fn a_missing_bus_output_is_refused() {
    assert_eq!(
        tampered(|p| {
            p.argue.bus_outputs.pop();
        }),
        Err(MlError::ArgueShapeMismatch { part: "tables" })
    );
}

/// One tree's halves wrong at a mid step of the three-tree bin: that step's
/// residual refuses it, and names the step.
#[test]
fn one_trees_halves_wrong_at_a_mid_step_are_refused() {
    for tree in 0..3 {
        assert_eq!(
            tampered(|p| p.argue.gkr[0].layers[2].halves[tree].p_hi += ExtE::one()),
            Err(MlError::LayerRelationMismatch { layer: 2 }),
            "tree {tree}"
        );
    }
    assert_eq!(
        tampered(|p| {
            p.argue.gkr[0].layers[2].halves.pop();
        }),
        Err(MlError::ArgueShapeMismatch {
            part: "ladder step"
        })
    );
}

#[test]
fn a_wrong_round_value_is_refused() {
    assert_eq!(
        tampered(|p| p.argue.gkr[0].layers[3].sumcheck.rounds[1].evaluations[0] += ExtE::one()),
        Err(MlError::LayerRelationMismatch { layer: 3 }),
        "a GKR round"
    );
    assert_eq!(
        tampered(|p| p.argue.constraint.rounds[1].evaluations[2] += ExtE::one()),
        Err(MlError::BatchMismatch),
        "a constraint round"
    );
}

#[test]
fn one_tables_factor_value_wrong_is_refused() {
    for t in 0..3 {
        assert_eq!(
            tampered(|p| p.argue.factor_values[t][1] += ExtE::one()),
            Err(MlError::BatchMismatch),
            "table {t}"
        );
    }
    assert_eq!(
        tampered(|p| {
            p.argue.factor_values[1].pop();
        }),
        Err(MlError::QueryCountMismatch {
            expected: 4,
            got: 3
        }),
        "a factor value short"
    );
}

/// ADD and MUL have the same shape; their messages swapped are refused.
#[test]
fn swapped_tables_are_refused() {
    let airs = airs();
    let cfg = batched(27);
    // Same heights, so the two are interchangeable in shape.
    let committed = commit(tables(&airs, &columns(3, 0), false), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    verify(&committed, &proof, &cfg).expect("honest");
    let mut outputs = proof.clone();
    outputs.argue.bus_outputs.swap(1, 2);
    assert!(verify(&committed, &outputs, &cfg).is_err(), "bus outputs");
    let mut values = proof.clone();
    values.argue.factor_values.swap(1, 2);
    assert!(verify(&committed, &values, &cfg).is_err(), "factor values");
    // And a verifier handed the statements in another order.
    let mut statements: Vec<_> = committed.tables().iter().map(|t| t.statement()).collect();
    statements.swap(1, 2);
    assert!(
        verify_under(&committed, &statements, &proof, &cfg, VerifierChecks::ALL).is_err(),
        "statements"
    );
}

/// A reduction present for an unshifted table, or absent for a shifted one.
#[test]
fn a_reduction_where_the_format_has_none_is_refused() {
    let airs = airs();
    let cfg = batched(27);
    let committed = commit(tables(&airs, &columns(3, 1), true), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    let reduce = proof.argue.reduces[2].clone().expect("the shifted table's");
    let mut extra = proof.clone();
    extra.argue.reduces[1] = Some(reduce);
    assert_eq!(
        verify(&committed, &extra, &cfg),
        Err(MlError::ArgueShapeMismatch { part: "reduction" })
    );
    let mut missing = proof.clone();
    missing.argue.reduces[2] = None;
    assert_eq!(
        verify(&committed, &missing, &cfg),
        Err(MlError::ArgueShapeMismatch { part: "reduction" })
    );
}

// ─── negatives: provers that break the format ────────────────────────────

fn faulted(faults: ProverFaults) -> Result<(), MlError> {
    let airs = airs();
    let cfg = batched(27);
    let committed = commit(tables(&airs, &columns(3, 1), false), &cfg);
    let proof = prove(&committed, &cfg, faults);
    verify(&committed, &proof, &cfg)
}

/// A prover padding a short table the `2^Δ` way instead of by `Π X_j`.
#[test]
fn a_table_padded_by_scaling_is_refused() {
    // MUL (n = 3) is the one shorter than `n_max = 4`.
    assert_eq!(
        faulted(ProverFaults {
            scaled_padding: Some(2),
            ..Default::default()
        }),
        Err(MlError::BatchMismatch)
    );
}

/// A table's zerocheck weighted by the suffix of `ξ`, not its prefix.
#[test]
fn a_suffix_of_xi_is_refused() {
    assert_eq!(
        faulted(ProverFaults {
            suffix_xi: Some(2),
            ..Default::default()
        }),
        Err(MlError::BatchMismatch)
    );
}

/// A wrong cross-table weight, and a table claim absorbed wrong.
#[test]
fn a_wrong_weight_or_claim_is_refused() {
    for t in 0..3 {
        assert_eq!(
            faulted(ProverFaults {
                wrong_weight: Some(t),
                ..Default::default()
            }),
            Err(MlError::BatchMismatch),
            "weight of table {t}"
        );
        assert_eq!(
            faulted(ProverFaults {
                tamper_claim: Some(t),
                ..Default::default()
            }),
            Err(MlError::BatchMismatch),
            "claim of table {t}"
        );
    }
}

/// A prover that bins under another cap argues a different partition.
#[test]
fn a_different_bin_partition_is_refused() {
    for cap in [4u8, 0] {
        assert!(
            faulted(ProverFaults {
                bin_log_cells: Some(cap),
                ..Default::default()
            })
            .is_err(),
            "cap {cap}"
        );
    }
}

/// A trace breaking one constraint of a small table, its bus still balanced:
/// only the constraint sumcheck's final equation can see it.
#[test]
fn a_violated_constraint_inside_a_bin_is_refused() {
    let airs = airs();
    let cfg = batched(27);
    let mut cols = columns(3, 1);
    // Row 2 of ADD, and the CPU's send of it, both claim a wrong sum (ADD's
    // `c` is column 2, the CPU's column 4; the columns are column-major).
    cols[1][2][2] += FE::one();
    cols[0][4][2] += FE::one();
    let committed = commit(tables(&airs, &cols, false), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    assert_eq!(
        verify(&committed, &proof, &cfg),
        Err(MlError::BatchMismatch)
    );
}

/// One multiplicity off: the bus no longer balances.
#[test]
fn one_bus_multiplicity_off_is_refused() {
    let airs = airs();
    let cfg = batched(27);
    let mut cols = columns(3, 1);
    cols[2][3][1] += FE::one();
    let committed = commit(tables(&airs, &cols, false), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    assert_eq!(verify(&committed, &proof, &cfg), Err(MlError::BusImbalance));
}

/// A verifier holding a preprocessed column the proof did not commit.
#[test]
fn a_wrong_preprocessed_column_is_refused() {
    let airs = airs();
    let cfg = batched(27);
    let cols = columns(3, 1);
    let committed = commit(tables(&airs, &cols, false), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    let right = vec![Mle::new(cols[1][0].clone()).unwrap()];
    let mut bad = cols[1][0].clone();
    bad[5] += FE::one();
    let wrong = vec![Mle::new(bad).unwrap()];
    let with = |pre: &[Mle<Fp>]| -> Result<(), MlError> {
        let tables = committed.tables();
        let statements = vec![
            tables[0].statement(),
            tables[1].layout().statement_with_preprocessed(pre),
            tables[2].statement(),
        ];
        verify_under(&committed, &statements, &proof, &cfg, VerifierChecks::ALL)
    };
    with(&right).expect("the right copy agrees");
    assert_eq!(with(&wrong), Err(MlError::EvaluationMismatch));
}

// ─── the format is the verifier's ────────────────────────────────────────

/// Neither format's prover nor verifier takes the other's config.
#[test]
fn a_proof_under_the_other_format_is_refused() {
    let airs = airs();
    let cols = columns(2, 1);
    let per_table = config(ArgueFormat::PerTable);
    let batch = batched(27);

    let committed = commit(tables(&airs, &cols, false), &per_table);
    let mut t = DefaultTranscript::<Ext>::new(SEED);
    assert_eq!(
        multi_prove_batched(&committed, &per_table, &mut t, None).unwrap_err(),
        MlError::ArgueFormatMismatch,
        "the batched prover under a per-table config"
    );
    let mut t = DefaultTranscript::<Ext>::new(SEED);
    assert_eq!(
        multi_prove(&committed, &batch, &mut t, None).unwrap_err(),
        MlError::ArgueFormatMismatch,
        "the per-table prover under a batched config"
    );

    // A per-table proof, under a verifier configured for the batched argue.
    let mut t = DefaultTranscript::<Ext>::new(SEED);
    let today = multi_prove(&committed, &per_table, &mut t, None).unwrap();
    let statements: Vec<_> = committed.tables().iter().map(|t| t.statement()).collect();
    let run_today = |cfg: &ChainConfig| {
        let mut t = DefaultTranscript::<Ext>::new(SEED);
        multi_verify::<_, _, _, KeccakWhir>(
            &today,
            &statements,
            std::slice::from_ref(committed.groups()[0].layout()),
            std::slice::from_ref(committed.groups()[0].domain()),
            committed.sizes(),
            &ExtE::zero(),
            cfg,
            &mut t,
            None,
        )
    };
    run_today(&per_table).expect("today's proof verifies today");
    assert_eq!(run_today(&batch), Err(MlError::ArgueFormatMismatch));

    // A batched proof, under a verifier configured for today's argue.
    let committed = commit(tables(&airs, &cols, false), &batch);
    let proof = prove(&committed, &batch, ProverFaults::default());
    verify(&committed, &proof, &batch).expect("verifies batched");
    assert_eq!(
        verify(&committed, &proof, &per_table),
        Err(MlError::ArgueFormatMismatch)
    );
}

/// ★ Default-off: today's format is `PerTable`, and the per-table proof of
/// the example is what it was — the batched module moved no byte of it.
#[test]
fn the_default_format_is_per_table() {
    assert_eq!(ChainFormat::DEFAULT.argue, ArgueFormat::PerTable);
    assert!(ChainFormat::DEFAULT.is_default());
    assert!(
        !ChainFormat {
            argue: ArgueFormat::BATCHED,
            ..ChainFormat::DEFAULT
        }
        .is_default()
    );
    assert_eq!(ArgueFormat::default(), ArgueFormat::PerTable);
}

// ─── mutations: each new check made inert lets its negative through ──────

fn inert(checks: VerifierChecks, tamper: impl FnOnce(&mut Proof)) -> Result<(), MlError> {
    let airs = airs();
    let (committed, mut proof, cfg) = fixture(&airs);
    tamper(&mut proof);
    verify_with_checks(&committed, &proof, &cfg, checks)
}

#[test]
fn the_ladder_residual_is_load_bearing() {
    let checks = VerifierChecks {
        residual: false,
        ..VerifierChecks::ALL
    };
    assert_ne!(
        inert(checks, |p| p.argue.gkr[0].layers[2].halves[1].p_hi +=
            ExtE::one()),
        Err(MlError::LayerRelationMismatch { layer: 2 }),
        "made inert, the step no longer refuses its own tamper"
    );
}

#[test]
fn the_constraint_equation_is_load_bearing() {
    let airs = airs();
    let cfg = batched(27);
    let mut cols = columns(3, 1);
    cols[1][2][2] += FE::one();
    cols[0][4][2] += FE::one();
    let committed = commit(tables(&airs, &cols, false), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    let checks = VerifierChecks {
        constraint: false,
        ..VerifierChecks::ALL
    };
    assert_eq!(
        verify_with_checks(&committed, &proof, &cfg, checks),
        Ok(()),
        "made inert, a violated constraint is accepted"
    );
}

#[test]
fn the_bus_balance_is_load_bearing() {
    let airs = airs();
    let cfg = batched(27);
    let mut cols = columns(3, 1);
    cols[2][3][1] += FE::one();
    let committed = commit(tables(&airs, &cols, false), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    let checks = VerifierChecks {
        balance: false,
        ..VerifierChecks::ALL
    };
    assert_eq!(
        verify_with_checks(&committed, &proof, &cfg, checks),
        Ok(()),
        "made inert, an unbalanced bus is accepted"
    );
}

#[test]
fn the_reduction_shape_is_load_bearing() {
    let airs = airs();
    let cfg = batched(27);
    let committed = commit(tables(&airs, &columns(3, 1), true), &cfg);
    let mut proof = prove(&committed, &cfg, ProverFaults::default());
    proof.argue.reduces[1] = proof.argue.reduces[2].clone();
    let checks = VerifierChecks {
        reduce_shape: false,
        ..VerifierChecks::ALL
    };
    assert_eq!(
        verify_with_checks(&committed, &proof, &cfg, checks),
        Ok(()),
        "made inert, a reduction the format has no place for is accepted"
    );
}

#[test]
fn the_preprocessed_check_is_load_bearing() {
    let airs = airs();
    let cfg = batched(27);
    let cols = columns(3, 1);
    let committed = commit(tables(&airs, &cols, false), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    let mut bad = cols[1][0].clone();
    bad[5] += FE::one();
    let wrong = vec![Mle::new(bad).unwrap()];
    let tables = committed.tables();
    let statements = vec![
        tables[0].statement(),
        tables[1].layout().statement_with_preprocessed(&wrong),
        tables[2].statement(),
    ];
    let checks = VerifierChecks {
        preprocessed: false,
        ..VerifierChecks::ALL
    };
    assert_eq!(
        verify_under(&committed, &statements, &proof, &cfg, checks),
        Ok(()),
        "made inert, a wrong preprocessed column is accepted"
    );
}

// ─── B-3: the card proves the host reference's bytes ─────────────────────

/// Whether a device promises any room (the parent module's probe).
fn a_device() -> bool {
    multilinear::gpu::reserve_budget() > 0
}

/// ★ D-BATCH B-3's gate: the prover on the card ([`Where::Device`]) proves the
/// host reference's canonical bytes — every tree, ladder step, round and value
/// — with the transcript in the same state after, and the proof verifies.
///
/// The tables are tall enough for the card to take them: CPU 2^13 rows (a tree
/// of 2^14 input cells, the device's tree threshold), ADD 2^13, MUL 2^12, all
/// past the fused rounds' 2^7. On a device the fused sessions and the ladder's
/// device rounds are counted, so the comparison is not of the host with
/// itself; without one it is the host reference against the device path's
/// host fallbacks, and says so.
///
/// ```text
/// cargo test --release -p stark --features cuda,multilinear/cuda --lib -- \
///     multilinear_table::batched::tests::the_card_proves_the_host_references_bytes --exact --nocapture
/// ```
#[test]
fn the_card_proves_the_host_references_bytes() {
    use crypto::fiat_shamir::is_transcript::IsTranscript;

    let airs = airs();
    for (a, pad, cap) in [(12usize, 1usize, 27u8), (12, 1, 14), (3, 1, 27)] {
        let cfg = batched(cap);
        let cols = columns(a, pad);
        let run = |at: Where| {
            let committed = commit(tables(&airs, &cols, false), &cfg);
            let fused = multilinear::gpu_fused::fused_sessions();
            let rounds = multilinear::gpu::sumcheck_rounds();
            let mut transcript = DefaultTranscript::<Ext>::new(SEED);
            let proof = prove_batched_on(
                &committed,
                &cfg,
                &mut transcript,
                None,
                ProverFaults::default(),
                at,
            )
            .unwrap_or_else(|e| panic!("{at:?}: {e:?}"));
            verify(&committed, &proof, &cfg).unwrap_or_else(|e| panic!("{at:?}: {e:?}"));
            let fused = multilinear::gpu_fused::fused_sessions() - fused;
            let rounds = multilinear::gpu::sumcheck_rounds() - rounds;
            (
                bincode::serialize(&proof).expect("bytes"),
                transcript.state(),
                transcript.sample_field_element(),
                fused,
                rounds,
            )
        };
        let host = run(Where::Host);
        let card = run(Where::Device);
        assert_eq!(host.3, 0, "the host reference ran no fused session");
        eprintln!(
            "batched argue B-3 a={a} pad={pad} cap={cap}: device {} fused sessions, {} device rounds; {} bytes",
            card.3,
            card.4,
            card.0.len()
        );
        if a_device() && a >= 7 {
            assert!(card.3 > 0, "a={a}: the card ran fused sessions");
            assert!(card.4 > 0, "a={a}: the card ran rounds");
        }
        assert!(
            host.0 == card.0,
            "a={a} cap={cap}: the canonical bytes differ"
        );
        assert_eq!(host.1, card.1, "a={a} cap={cap}: the transcripts parted");
        assert_eq!(host.2, card.2, "a={a} cap={cap}: the next challenge moved");
    }
}
