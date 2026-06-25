//! Differential tests for the monomorphized `ConstraintBuilder` migration: each
//! migrated table's `XxxDomain` must (a) fold the same residuals as the boxed
//! constraints on a synthetic row, and (b) produce a byte-identical proof on the
//! builder path vs the boxed path. Proofs use `grinding_factor = 0` so the parallel
//! FRI grinding nonce (a non-deterministic `find_any`) doesn't mask the comparison.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use stark::constraints::builder::{ConstraintContext, ProverConstraintBuilder, TableConstraints};
use stark::constraints::transition::TransitionConstraintEvaluator;
use stark::frame::Frame;
use stark::lookup::PackingShifts;
use stark::proof::options::ProofOptions;
use stark::table::TableView;
use stark::trace::{LDETraceTable, TraceTable};
use stark::traits::{AIR, TransitionEvaluationContext, ZerofierEvaluations};

use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::test_utils::multi_prove_ram;

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Serializes the global ConstraintBuilder flag toggling across the tests in this
/// module (they share a process-global flag; the atomic is thread-safe but the
/// off→on→off sequence must not interleave between tests).
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Fold every boxed constraint's `coeff * residual` at a synthetic single-row trace,
/// matching what a `ProverConstraintBuilder` does, so a table's domain folding can be
/// compared against its boxed constraints exactly.
fn boxed_residual_sum(
    boxed: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>>,
    row: &[FieldElement<F>],
    coeffs: &[FieldElement<E>],
    n: usize,
) -> FieldElement<E> {
    let step = TableView::<F, E>::new(vec![row.to_vec()], vec![]);
    let frame = Frame::<F, E>::new(vec![step]);
    let zero = FieldElement::<E>::zero();
    let shifts = PackingShifts::<F>::new();
    let octx = TransitionEvaluationContext::new_prover(&frame, &[], &[], &[], &zero, &shifts);
    let mut expected = FieldElement::<E>::zero();
    for b in boxed {
        let idx = TransitionConstraintEvaluator::constraint_idx(b.as_ref());
        let mut evals = vec![FieldElement::<E>::zero(); n];
        b.evaluate_verifier(&octx, &mut evals);
        expected = expected + &evals[idx] * &coeffs[idx];
    }
    expected
}

/// Assert that a table's migrated `domain` (a `TableConstraints`) folds the same
/// residuals as its `boxed` constraint list on a synthetic `num_columns`-column row.
/// Uses the object-safe `eval_prover` (a concrete `ProverConstraintBuilder`), so no
/// generic-fn-as-higher-ranked-`Fn` plumbing is needed.
fn assert_domain_matches_boxed(
    num_columns: usize,
    boxed: Vec<Box<dyn TransitionConstraintEvaluator<F, E>>>,
    domain: &dyn TableConstraints<F, E>,
) {
    let n = boxed.len();
    let row: Vec<FieldElement<F>> = (0..num_columns)
        .map(|i| FieldElement::<F>::from((i as u64) * 7 + 3))
        .collect();
    let main_columns: Vec<Vec<FieldElement<F>>> = row.iter().map(|v| vec![v.clone()]).collect();
    let lde = LDETraceTable::<F, E>::from_columns(main_columns, vec![], 1, 1);
    let zerofier = ZerofierEvaluations::<F> {
        groups: vec![vec![FieldElement::<F>::one()]],
        constraint_to_group: vec![0; n],
    };
    let coeffs: Vec<FieldElement<E>> = (0..n)
        .map(|i| FieldElement::<E>::from((i as u64) * 13 + 5))
        .collect();

    // Domain folding ignores the context; pass an empty one.
    let zero_e = FieldElement::<E>::zero();
    let shifts = PackingShifts::<F>::new();
    let ctx = ConstraintContext {
        rap_challenges: &[],
        logup_alpha_powers: &[],
        logup_table_offset: &zero_e,
        packing_shifts: &shifts,
        periodic: &[],
    };

    let mut cb = ProverConstraintBuilder::<F, E>::new(&lde, 0, &zerofier, &coeffs);
    domain.eval_prover(&mut cb, &ctx);
    let got = cb.finish();
    let expected = boxed_residual_sum(boxed, &row, &coeffs, n);
    assert_eq!(got, expected, "domain folding differs from boxed constraints");
}

/// Prove `air`/`base_trace` on the boxed path then the builder path and assert the
/// serialized proofs are byte-identical. The caller pins `grinding_factor = 0`.
fn assert_builder_path_byte_identical(
    air: &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
    base_trace: &TraceTable<F, E>,
    table: &str,
) {
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prove = || {
        let mut trace = base_trace.clone();
        let pairs: Vec<(
            &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
            _,
            _,
        )> = vec![(air, &mut trace, &())];
        multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[]))
            .unwrap_or_else(|_| panic!("{table} prove failed"))
    };
    stark::constraints::evaluator::set_constraint_builder(false);
    let off = prove();
    stark::constraints::evaluator::set_constraint_builder(true);
    let on = prove();
    stark::constraints::evaluator::set_constraint_builder(false);
    assert_eq!(
        bincode::serialize(&off).unwrap(),
        bincode::serialize(&on).unwrap(),
        "{table} builder path must produce a byte-identical proof"
    );
}

// =========================================================================
// EC_SCALAR
// =========================================================================

#[test]
fn ec_scalar_domain_eval_matches_boxed_residuals() {
    use crate::tables::ec_scalar::{cols, create_constraints, EcScalarDomain};
    assert_domain_matches_boxed(cols::NUM_COLUMNS, create_constraints(0).0, &EcScalarDomain);
}

#[test]
fn ec_scalar_constraint_builder_path_byte_identical() {
    use crate::tables::ec_scalar::{generate_ec_scalar_trace, rows_for_scalar};
    use crate::test_utils::create_ec_scalar_air;

    let mut proof_options = ProofOptions::default_test_options();
    proof_options.grinding_factor = 0;
    let air = create_ec_scalar_air(&proof_options);

    let mut k = [0u8; 32];
    k[0] = 0b1010_0101;
    k[1] = 0xFF;
    k[15] = 0x80;
    k[31] = 0x01;
    let ops = rows_for_scalar(444, 0x3000, &k);
    let base_trace = generate_ec_scalar_trace(&ops);

    assert_builder_path_byte_identical(&air, &base_trace, "EC_SCALAR");
}


// =========================================================================
// ECSM
// =========================================================================

#[test]
fn ecsm_domain_eval_matches_boxed_residuals() {
    use crate::tables::ecsm::{cols, create_constraints, EcsmDomain};
    assert_domain_matches_boxed(cols::NUM_COLUMNS, create_constraints(0).0, &EcsmDomain);
}

#[test]
fn ecsm_constraint_builder_path_byte_identical() {
    use crate::tables::ecsm::{generate_ecsm_trace, EcsmOperation};
    use crate::test_utils::create_ecsm_air;
    use ecsm::compute_witness;

    // secp256k1 Gx (little-endian).
    let gx_le = || {
        let mut be = [
            0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87,
            0x0B, 0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B,
            0x16, 0xF8, 0x17, 0x98,
        ];
        be.reverse();
        be
    };
    let k_le = |v: u64| {
        let mut k = [0u8; 32];
        k[..8].copy_from_slice(&v.to_le_bytes());
        k
    };
    let ops: Vec<EcsmOperation> = [1u64, 2, 5, 0xFFFF, 1_000_003]
        .iter()
        .map(|&k| EcsmOperation {
            timestamp: 444,
            addr_xg: 0x2000,
            addr_k: 0x3000,
            addr_xr: 0x1000,
            witness: compute_witness(&k_le(k), &gx_le()).unwrap(),
        })
        .collect();
    let base_trace = generate_ecsm_trace(&ops);

    let mut proof_options = ProofOptions::default_test_options();
    proof_options.grinding_factor = 0;
    let air = create_ecsm_air(&proof_options);
    assert_builder_path_byte_identical(&air, &base_trace, "ECSM");
}
