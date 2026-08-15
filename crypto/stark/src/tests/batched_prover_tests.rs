//! `multi_prove_batched` — the openings it produces, and the residency claim
//! MMCS-PLAN §3.3 asks to be made falsifiable at the PROVER level.
//!
//! The primitive-level access-window test
//! (`streaming_builder_serves_the_base_group_without_holding_it`, in
//! `fri/mmcs.rs`) shows that [`crate::fri::mmcs::StreamingMmcsBuilder`] CAN be
//! driven with peak residency one. It cannot show that the prover drives it that
//! way, because at the time it was written there was no batched prover. These
//! tests close that gap from the other side.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use crate::batched::proof::{BatchedMultiProof, BatchedProveStats};
use crate::batched::prover::multi_prove_batched;
use crate::batched::shape::{EpochShape, RoundShape};
use crate::config::KeccakStarkHash;
use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::fri::mmcs::{MixedMmcs, MixedOpening};
use crate::proof::options::ProofOptions;
use crate::prover::GenericProver;
use crate::residency_mode::ResidencyMode;
use crate::trace::TraceTable;
use crate::traits::AIR;

pub(crate) type F = GoldilocksField;
pub(crate) type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;
pub(crate) type Air = crate::lookup::AirWithBuses<
    F,
    E,
    crate::lookup::NullBoundaryConstraintBuilder,
    (),
    crate::constraints::builder::EmptyConstraints,
>;

/// Small `fri_final_poly_log_degree` so the tiny fixture below actually FOLDS.
/// At the default (7) every table in an 8-row epoch terminates immediately and
/// the batched FRI degenerates to one terminal polynomial — a real case, and
/// covered by `batched_prove_openings_authenticate` under the default options,
/// but not the one that exercises the injection recursion.
pub(crate) fn folding_options() -> ProofOptions {
    ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 4,
        coset_offset: 3,
        grinding_factor: 4,
        fri_final_poly_log_degree: 1,
    }
}

/// The bus-balanced CPU/ADD/MUL instance from the completeness tests, with the
/// CPU table one height above the other two so the epoch is genuinely mixed —
/// a same-height epoch would exercise neither the injection nor the index
/// reduction.
fn traces() -> (TraceTable<F, E>, TraceTable<F, E>, TraceTable<F, E>) {
    let cpu = TraceTable::from_columns_main(
        vec![
            vec![
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::one(),
                FE::zero(),
                FE::zero(),
            ],
            vec![
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::one(),
                FE::zero(),
                FE::zero(),
                FE::one(),
                FE::one(),
            ],
            (1..=8).map(FE::from).collect(),
            (1..=8).map(|i| FE::from(i * 10)).collect(),
            vec![
                FE::from(11),
                FE::from(40),
                FE::from(33),
                FE::from(160),
                FE::from(55),
                FE::from(66),
                FE::from(490),
                FE::from(640),
            ],
        ],
        1,
    );
    let add = TraceTable::from_columns_main(
        vec![
            vec![FE::from(1), FE::from(3), FE::from(5), FE::from(6)],
            vec![FE::from(10), FE::from(30), FE::from(50), FE::from(60)],
            vec![FE::from(11), FE::from(33), FE::from(55), FE::from(66)],
            vec![FE::one(); 4],
        ],
        1,
    );
    let mul = TraceTable::from_columns_main(
        vec![
            vec![FE::from(2), FE::from(4), FE::from(7), FE::from(8)],
            vec![FE::from(20), FE::from(40), FE::from(70), FE::from(80)],
            vec![FE::from(40), FE::from(160), FE::from(490), FE::from(640)],
            vec![FE::one(); 4],
        ],
        1,
    );
    (cpu, add, mul)
}

/// Prove `repeats` copies of the fixture as one epoch. `repeats == 1` is the
/// three-table epoch; higher values are how the residency claim is put on a
/// curve instead of a threshold.
pub(crate) fn prove_repeated(
    repeats: usize,
    options: &ProofOptions,
    residency: ResidencyMode,
) -> (
    Vec<Air>,
    BatchedMultiProof<F, E, ()>,
    BatchedProveStats,
    Vec<usize>,
) {
    prove_repeated_with(
        repeats,
        options,
        residency,
        &mut DefaultTranscript::<E>::new(&[]),
    )
}

/// As [`prove_repeated`], but against a caller-owned transcript — so a test can
/// read the state the PROVER ended in and compare it with the verifier's.
pub(crate) fn prove_repeated_with(
    repeats: usize,
    options: &ProofOptions,
    residency: ResidencyMode,
    transcript: &mut DefaultTranscript<E>,
) -> (
    Vec<Air>,
    BatchedMultiProof<F, E, ()>,
    BatchedProveStats,
    Vec<usize>,
) {
    let mut airs = Vec::new();
    let mut all_traces = Vec::new();
    for _ in 0..repeats {
        let (cpu, add, mul) = traces();
        airs.push(new_cpu_air_with_lookup(options));
        airs.push(new_add_air_with_lookup(options));
        airs.push(new_mul_air_with_lookup(options));
        all_traces.push(cpu);
        all_traces.push(add);
        all_traces.push(mul);
    }

    let unit = ();
    let pairs: Vec<_> = airs
        .iter()
        .zip(all_traces.iter_mut())
        .map(|(air, trace)| {
            (
                air as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
                trace,
                &unit,
            )
        })
        .collect();

    let trace_lengths: Vec<usize> = (0..repeats).flat_map(|_| [8usize, 4, 4]).collect();
    let (proof, stats) = multi_prove_batched::<F, E, (), KeccakStarkHash, GenericProver<F, E, (), KeccakStarkHash>>(
        pairs,
        transcript,
        None,
        #[cfg(feature = "disk-spill")]
        crate::storage_mode::StorageMode::Ram,
        residency,
    )
    .expect("the fixture is a well-shaped epoch");

    (airs, proof, stats, trace_lengths)
}

pub(crate) fn shape_of(airs: &[Air], trace_lengths: &[usize]) -> EpochShape {
    let refs: Vec<&dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>> = airs
        .iter()
        .map(|a| a as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>)
        .collect();
    EpochShape::derive(&refs, trace_lengths)
        .expect("the fixture is a well-shaped epoch")
        .0
}

/// Authenticate one round's opening the way a verifier must: reduce the shared
/// FRI index into the round's own index space first.
fn round_verifies<C: math::field::traits::IsField + 'static>(
    root: &crate::config::Commitment,
    opening: &MixedOpening<C>,
    round: &RoundShape,
    iota_fri: usize,
    h_max_fri: usize,
) -> bool
where
    FieldElement<C>: math::traits::AsBytes + Sync + Send,
{
    let Some(h_max_round) = round.h_max() else {
        return false;
    };
    let Some(iota) = crate::batched::round4::reduce_iota_to_round(iota_fri, h_max_fri, h_max_round)
    else {
        return false;
    };
    MixedMmcs::<C, KeccakStarkHash>::verify_batch(
        root,
        iota,
        opening,
        &round.heights(),
        &round.widths(),
    )
}

/// The honest path. Every query's opening of every batched round authenticates
/// against that round's root, under the index reduction the two different
/// `h_max` values force.
#[test_log::test]
fn batched_prove_openings_authenticate() {
    for options in [ProofOptions::default_test_options(), folding_options()] {
        let (airs, proof, _stats, lengths) = prove_repeated(1, &options, ResidencyMode::Retain);
        let shape = shape_of(&airs, &lengths);
        let h_max = shape.h_max();
        assert_eq!(proof.queries.len(), options.fri_number_of_queries);
        let iotas = recover_iotas(&proof, &shape, h_max);

        for (q, query) in proof.queries.iter().enumerate() {
            assert!(
                round_verifies(&proof.main_root, &query.main, &shape.main, iotas[q], h_max),
                "query {q}: main round must authenticate"
            );
            assert!(
                round_verifies(&proof.parts_root, &query.parts, &shape.parts, iotas[q], h_max),
                "query {q}: parts round must authenticate"
            );
            let (Some(root), Some(opening)) = (proof.aux_root, query.aux.as_ref()) else {
                panic!("the fixture's tables all have a RAP, so the aux round exists");
            };
            assert!(
                round_verifies(&root, opening, &shape.aux, iotas[q], h_max),
                "query {q}: aux round must authenticate"
            );
        }
    }
}

/// The query indices, recovered from the proof rather than read off the
/// prover's own state.
///
/// Deliberately NOT via `replay_epoch_transcript`, even though that exists: an
/// opening authenticates at exactly one leaf, so scanning the (tiny) index
/// space for the one that verifies is a derivation INDEPENDENT of the
/// transcript. These tests are then not circular — they do not check the
/// openings against indices produced by the same code path that has to be
/// right for the openings to mean anything. The epoch-level tests in
/// `batched_mmcs_soundness_tests::epoch` use the replay, so both derivations
/// are exercised and are pinned to each other by the honest path passing under
/// each.
fn recover_iotas(
    proof: &BatchedMultiProof<F, E, ()>,
    shape: &EpochShape,
    h_max: usize,
) -> Vec<usize> {
    proof
        .queries
        .iter()
        .map(|query| {
            (0..(1usize << (h_max - 1)))
                .find(|&candidate| {
                    round_verifies(&proof.main_root, &query.main, &shape.main, candidate, h_max)
                })
                .expect("an honest opening authenticates at its own index")
        })
        .collect()
}

/// ★ The acceptance test MMCS-PLAN §3.3 asks for, at the prover level.
///
/// Doubling the epoch must not double the trace-LDE residency. The assertion is
/// a SCALING one rather than a threshold: a threshold can be met by a prover
/// that holds everything for a small epoch, while the curve cannot. The retained
/// arm is the control — it proves the measurement can see growth, so a flat
/// recompute arm means the streaming discipline held, not that the ledger is
/// blind.
#[test_log::test]
fn streaming_prover_trace_residency_is_flat_in_the_table_count() {
    let options = folding_options();

    let (_, _, small_recompute, _) = prove_repeated(1, &options, ResidencyMode::RecomputeLde);
    let (_, _, large_recompute, _) = prove_repeated(2, &options, ResidencyMode::RecomputeLde);
    let (_, _, small_retain, _) = prove_repeated(1, &options, ResidencyMode::Retain);
    let (_, _, large_retain, _) = prove_repeated(2, &options, ResidencyMode::Retain);

    assert_eq!(
        small_recompute.peak_trace_lde_bytes, large_recompute.peak_trace_lde_bytes,
        "streaming the commitment rounds must make the trace-LDE peak independent of \
         how many tables the epoch has; it grew from {} to {} bytes",
        small_recompute.peak_trace_lde_bytes, large_recompute.peak_trace_lde_bytes
    );

    // The control. Without it a ledger that simply never counted anything would
    // pass the assertion above.
    assert!(
        large_retain.peak_trace_lde_bytes > small_retain.peak_trace_lde_bytes,
        "the retaining arm must show the growth the recomputing arm avoids \
         ({} vs {} bytes) — otherwise the measurement cannot see residency at all",
        small_retain.peak_trace_lde_bytes,
        large_retain.peak_trace_lde_bytes
    );
    assert!(
        large_retain.peak_trace_lde_bytes > large_recompute.peak_trace_lde_bytes,
        "at the same epoch the retaining arm must hold more than the recomputing one"
    );

    // The parts are `O(N)` by design and are accounted separately, so the claim
    // above is about the trace LDEs and is not quietly absorbing them.
    assert!(
        large_recompute.retained_parts_bytes > small_recompute.retained_parts_bytes,
        "the composition parts are retained per table and must be seen to grow"
    );
}

/// The recompute budget, stated as a test so it cannot drift silently. Six
/// tables, five phases that read a trace LDE — the commit, constraint
/// evaluation, the OOD evaluations, the DEEP codeword and the query openings —
/// and no barrier between them can be removed, so `RecomputeLde` pays one
/// forward NTT per table per phase.
#[test_log::test]
fn the_recompute_budget_is_five_expansions_per_table() {
    let options = folding_options();
    let (_, _, recompute, _) = prove_repeated(2, &options, ResidencyMode::RecomputeLde);
    let (_, _, retain, _) = prove_repeated(2, &options, ResidencyMode::Retain);

    let tables = 6;
    assert_eq!(
        recompute.main_lde_expansions,
        5 * tables,
        "main LDE: one expansion per table per phase that reads it"
    );
    assert_eq!(
        recompute.aux_lde_expansions,
        5 * tables,
        "aux LDE: every table in this fixture has a RAP, so the same five phases"
    );
    assert_eq!(
        retain.main_lde_expansions, tables,
        "retaining pays the floor: one expansion per table"
    );
    assert_eq!(
        retain.aux_lde_expansions, tables,
        "retaining pays the floor for aux too"
    );
    for stats in [recompute, retain] {
        assert_eq!(
            stats.parts_computations, tables,
            "composition parts are computed ONCE per table under either mode — \
             recomputing them would be a second constraint evaluation"
        );
    }
}

/// Residency is a performance choice and must not be a protocol one: the two
/// modes differ in when buffers exist, never in what is committed.
#[test_log::test]
fn residency_mode_does_not_move_any_batched_root() {
    let options = folding_options();
    let (_, retained, _, _) = prove_repeated(1, &options, ResidencyMode::Retain);
    let (_, recomputed, _, _) = prove_repeated(1, &options, ResidencyMode::RecomputeLde);

    assert_eq!(retained.prep_root, recomputed.prep_root);
    assert_eq!(retained.main_root, recomputed.main_root);
    assert_eq!(retained.aux_root, recomputed.aux_root);
    assert_eq!(retained.parts_root, recomputed.parts_root);
    assert_eq!(retained.fri_layer_roots, recomputed.fri_layer_roots);
    assert_eq!(retained.fri_final_poly_coeffs, recomputed.fri_final_poly_coeffs);
    assert_eq!(retained.nonce, recomputed.nonce);
}

/// A tampered row is rejected in EVERY round and at EVERY matrix, not only the
/// tallest one. A control that touched one matrix would pass even if the shorter
/// matrices were authenticated at the wrong leaf — which is precisely the silent
/// failure the index convention has.
#[test_log::test]
fn a_tampered_row_in_any_matrix_of_any_round_is_rejected() {
    let options = folding_options();
    let (airs, proof, _, lengths) = prove_repeated(1, &options, ResidencyMode::Retain);
    let shape = shape_of(&airs, &lengths);
    let h_max = shape.h_max();
    let iota_0 = recover_iotas(&proof, &shape, h_max)[0];
    let query = &proof.queries[0];

    for matrix in 0..shape.main.tables.len() {
        let mut tampered = query.main.clone();
        tampered.per_matrix[matrix].evaluations[0] += FE::one();
        assert!(
            !round_verifies(&proof.main_root, &tampered, &shape.main, iota_0, h_max),
            "main round, matrix {matrix}: a tampered row must be rejected"
        );
    }
    for matrix in 0..shape.parts.tables.len() {
        let mut tampered = query.parts.clone();
        tampered.per_matrix[matrix].evaluations_sym[0] += FieldElement::<E>::one();
        assert!(
            !round_verifies(&proof.parts_root, &tampered, &shape.parts, iota_0, h_max),
            "parts round, matrix {matrix}: a tampered symmetric row must be rejected"
        );
    }
    let aux_root = proof.aux_root.expect("the fixture has a RAP");
    for matrix in 0..shape.aux.tables.len() {
        let mut tampered = query.aux.clone().expect("the fixture has a RAP");
        tampered.per_matrix[matrix].evaluations[0] += FieldElement::<E>::one();
        assert!(
            !round_verifies(&aux_root, &tampered, &shape.aux, iota_0, h_max),
            "aux round, matrix {matrix}: a tampered row must be rejected"
        );
    }
}

/// The shape a round is verified under is the verifier's, not the proof's.
/// Feeding a width the epoch did not commit must reject — this is the
/// boundary-shift forgery `fri/mmcs.rs`'s width binding closes, reached through
/// the prover for the first time.
#[test_log::test]
fn a_width_the_epoch_did_not_commit_is_rejected() {
    let options = folding_options();
    let (airs, proof, _, lengths) = prove_repeated(1, &options, ResidencyMode::Retain);
    let mut shape = shape_of(&airs, &lengths);
    let h_max = shape.h_max();
    let iota_0 = recover_iotas(&proof, &shape, h_max)[0];

    shape.main.dims[0].1 += 1;
    assert!(
        !round_verifies(
            &proof.main_root,
            &proof.queries[0].main,
            &shape.main,
            iota_0,
            h_max
        ),
        "a main matrix width the epoch did not commit must be rejected"
    );
}
