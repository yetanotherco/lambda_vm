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
use crate::config::DefaultStarkHash;
use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::fri::mmcs::{MixedMmcs, MixedOpening};
use crate::proof::options::ProofOptions;
use crate::prover::{GenericProver, IsStarkProver};
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
    let (proof, stats) = multi_prove_batched::<
        F,
        E,
        (),
        DefaultStarkHash,
        GenericProver<F, E, (), DefaultStarkHash>,
    >(
        pairs,
        transcript,
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
    MixedMmcs::<C, DefaultStarkHash>::verify_batch(
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
                round_verifies(
                    &proof.parts_root,
                    &query.parts,
                    &shape.parts,
                    iotas[q],
                    h_max
                ),
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

    assert_eq!(retained.main_root, recomputed.main_root);
    assert_eq!(retained.aux_root, recomputed.aux_root);
    assert_eq!(retained.parts_root, recomputed.parts_root);
    assert_eq!(retained.fri_layer_roots, recomputed.fri_layer_roots);
    assert_eq!(
        retained.fri_final_poly_coeffs,
        recomputed.fri_final_poly_coeffs
    );
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

// ===========================================================================
// The PREPROCESSED tables (per-table trees inside the batched proof)
// ===========================================================================
//
// Preprocessed matrices are NOT a round of the mixed MMCS: each preprocessed
// table keeps its own row-pair tree — the one `air.precomputed_commitment()`
// pins — and both sides absorb that root from the AIR set. What still does
// real index work is the per-table reduction: a preprocessed table shorter
// than the FRI is opened at `reduce_iota_to_round(iota, h_max, height)`, and
// `fri/mmcs.rs`'s warning stands — a wrong convention is self-consistent, so
// the un-reduced control below is what makes the reduction load-bearing.

/// ADD and MUL, both declared preprocessed, at the SAME height but DIFFERENT
/// widths (2 and 3 precomputed columns). Each of those three facts is doing a
/// job:
///
/// - **two tables**, so "per matrix" in the tamper control below is a real
///   quantifier rather than a loop that runs once;
/// - **different widths**, so the width binding (from the AIR set, never the
///   proof) is exercised at two distinct values;
/// - **both below CPU's height**, so the per-table reduction keeps doing real
///   work.
///
/// ★ The per-AIR `precomputed_commitment()` values ARE read on the batched
/// path — that is the point of the per-table arrangement: the prover builds
/// each preprocessed table's own tree and fails the prove unless its root
/// equals the AIR's pinned value, and the verifier absorbs and compares those
/// same roots. The fixture therefore pins the REAL roots, computed by the same
/// routine the prover uses.
pub(crate) const PREP_WIDTHS: [usize; 2] = [2, 3];

/// The row-pair subset root over the first `width` columns of `trace`'s main
/// LDE — the value `air.precomputed_commitment()` must pin for the fixture to
/// prove.
fn real_prep_root(
    air: &Air,
    trace: &TraceTable<F, E>,
    width: usize,
) -> crate::config::Commitment {
    let (domain, twiddles) = crate::prover::domain_and_twiddles(
        air as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        trace.num_rows(),
    );
    let (data, total_cols) =
        GenericProver::<F, E, (), DefaultStarkHash>::expand_main_lde_row_major(
            trace,
            &domain,
            &twiddles,
            #[cfg(feature = "disk-spill")]
            crate::storage_mode::StorageMode::Ram,
        );
    GenericProver::<F, E, (), DefaultStarkHash>::commit_rows_bit_reversed_subset::<F>(
        &data, total_cols, 0, width,
    )
    .expect("the fixture trace has rows")
    .1
}

fn preprocessed_epoch(options: &ProofOptions) -> (Vec<Air>, Vec<TraceTable<F, E>>) {
    let (cpu, add, mul) = traces();
    let add_air = new_add_air_with_lookup(options);
    let mul_air = new_mul_air_with_lookup(options);
    let add_root = real_prep_root(&add_air, &add, PREP_WIDTHS[0]);
    let mul_root = real_prep_root(&mul_air, &mul, PREP_WIDTHS[1]);
    let airs = vec![
        new_cpu_air_with_lookup(options),
        add_air.with_preprocessed(add_root, PREP_WIDTHS[0]),
        mul_air.with_preprocessed(mul_root, PREP_WIDTHS[1]),
    ];
    (airs, vec![cpu, add, mul])
}

/// What a preprocessed-epoch prove hands back: the AIRs (borrowed by the shape
/// derivation), the proof, and the trace lengths the verifier would read off it.
pub(crate) type PreprocessedProve = (Vec<Air>, BatchedMultiProof<F, E, ()>, Vec<usize>);

pub(crate) fn prove_preprocessed() -> Result<PreprocessedProve, crate::prover::ProvingError> {
    let options = folding_options();
    let (airs, mut all_traces) = preprocessed_epoch(&options);
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
    let (proof, _) = multi_prove_batched::<
        F,
        E,
        (),
        DefaultStarkHash,
        GenericProver<F, E, (), DefaultStarkHash>,
    >(
        pairs,
        &mut DefaultTranscript::<E>::new(&[]),
        #[cfg(feature = "disk-spill")]
        crate::storage_mode::StorageMode::Ram,
        ResidencyMode::Retain,
    )?;
    Ok((airs, proof, vec![8, 4, 4]))
}

/// Per-table authentication of one preprocessed opening — the verifier's own
/// three steps (width bind, leaf hash, path walk), restated so the tamper
/// controls can drive them one matrix and one column at a time.
fn prep_table_verifies(
    root: &crate::config::Commitment,
    o: &crate::proof::stark::PolynomialOpenings<F>,
    leaf: usize,
    width: usize,
) -> bool {
    use crate::config::StarkHash;
    use crypto::merkle_tree::traits::IsStreamingLeafBackend;
    o.evaluations.len() == width && o.evaluations_sym.len() == width && {
        let leaf_hash =
            <<DefaultStarkHash as StarkHash>::Batched<F> as IsStreamingLeafBackend<F>>::hash_data_from_slices(
                &o.evaluations,
                &o.evaluations_sym,
            );
        crypto::merkle_tree::proof::verify_merkle_path_from_leaf_hash::<
            <DefaultStarkHash as StarkHash>::Batched<F>,
        >(&o.proof.merkle_path, root, leaf, leaf_hash)
    }
}

/// Honest path, plus the facts that make the rest of this section meaningful:
/// both preprocessed tables authenticate against the AIR's own pinned roots at
/// the reduced per-table index, the widths differ, and at least one table sits
/// strictly below the FRI so the reduction is non-trivial.
#[test_log::test]
fn the_preprocessed_tables_are_committed_and_authenticate() {
    let (airs, proof, lengths) = prove_preprocessed().expect("an honest preprocessed epoch");
    let shape = shape_of(&airs, &lengths);
    let h_max = shape.h_max();

    assert_eq!(
        shape.prep.widths(),
        PREP_WIDTHS,
        "two preprocessed tables at different widths"
    );
    let prep_h_max = shape.prep.h_max().expect("the fixture has preprocessed tables");
    assert!(
        prep_h_max < h_max,
        "the reduction must be non-trivial (prep {prep_h_max}, fri {h_max})"
    );

    for (q, iota) in recover_iotas(&proof, &shape, h_max).into_iter().enumerate() {
        let opening = &proof.queries[q];
        assert_eq!(opening.prep.len(), shape.prep.tables.len());
        for (k, &t) in shape.prep.tables.iter().enumerate() {
            let leaf = crate::batched::round4::reduce_iota_to_round(
                iota,
                h_max,
                shape.heights[t],
            )
            .expect("prep heights are a subset of table heights");
            assert!(
                prep_table_verifies(
                    &airs[t].precomputed_commitment(),
                    &opening.prep[k],
                    leaf,
                    airs[t].num_precomputed_columns(),
                ),
                "query {q}, prep table {t}: must authenticate against the AIR's own root"
            );
        }
    }
}

/// ★ The control the index convention needs. Reading a shorter preprocessed
/// table at the UN-reduced FRI index must fail — otherwise the reduction is
/// decoration and a prover free to pick either convention would be believed
/// under both.
#[test_log::test]
fn the_un_reduced_index_does_not_authenticate_a_preprocessed_table() {
    let (airs, proof, lengths) = prove_preprocessed().expect("an honest preprocessed epoch");
    let shape = shape_of(&airs, &lengths);
    let h_max = shape.h_max();

    let mut any_differed = false;
    for (q, iota) in recover_iotas(&proof, &shape, h_max).into_iter().enumerate() {
        for (k, &t) in shape.prep.tables.iter().enumerate() {
            let height = shape.heights[t];
            let reduced = crate::batched::round4::reduce_iota_to_round(iota, h_max, height)
                .expect("prep heights are a subset of table heights");
            if reduced == iota {
                continue;
            }
            any_differed = true;
            assert!(
                !prep_table_verifies(
                    &airs[t].precomputed_commitment(),
                    &proof.queries[q].prep[k],
                    iota,
                    airs[t].num_precomputed_columns(),
                ),
                "query {q}, prep table {t}: the un-reduced FRI index must not authenticate"
            );
        }
    }
    assert!(
        any_differed,
        "at least one query must have a reduced index different from the raw one, \
         or this test never exercised the convention it exists for"
    );
}

/// Per-matrix, per-column tamper control. The per-table arrangement must fail
/// if ANY single table's preprocessed value is wrong — the same quantifier the
/// fused-round design owed §3.3, kept under the new layout.
#[test_log::test]
fn a_tampered_precomputed_row_is_rejected_per_matrix() {
    let (airs, proof, lengths) = prove_preprocessed().expect("an honest preprocessed epoch");
    let shape = shape_of(&airs, &lengths);
    let h_max = shape.h_max();
    let iota_0 = recover_iotas(&proof, &shape, h_max)[0];

    for (k, &t) in shape.prep.tables.iter().enumerate() {
        let leaf = crate::batched::round4::reduce_iota_to_round(iota_0, h_max, shape.heights[t])
            .expect("prep heights are a subset of table heights");
        let root = airs[t].precomputed_commitment();
        let width = airs[t].num_precomputed_columns();
        let honest = &proof.queries[0].prep[k];
        assert!(
            prep_table_verifies(&root, honest, leaf, width),
            "honest-path control: prep table {t} must authenticate untampered"
        );
        for column in 0..width {
            let mut tampered = honest.clone();
            tampered.evaluations[column] += FE::one();
            assert!(
                !prep_table_verifies(&root, &tampered, leaf, width),
                "prep table {t}, column {column}: a tampered precomputed value \
                 must be rejected"
            );
        }
    }
}

/// A stale preprocessed constant fails the PROVE, not just every future
/// verify — the property the per-table path gets from `commit_main_trace`,
/// now unconditional on the batched path: the prover builds each preprocessed
/// tree and compares its root against the AIR's pinned value. (The old
/// registry-pin width tests have no analogue: widths come from the AIR set on
/// both sides, so there is no positionally-swappable width list left to pin.)
#[test_log::test]
fn a_stale_precomputed_constant_fails_the_prove() {
    let options = folding_options();
    let (cpu, add, mul) = traces();
    let mul_air = new_mul_air_with_lookup(&options);
    let mul_root = real_prep_root(&mul_air, &mul, PREP_WIDTHS[1]);
    let airs = vec![
        new_cpu_air_with_lookup(&options),
        // The stale constant: a root the trace's columns cannot reproduce.
        new_add_air_with_lookup(&options).with_preprocessed([7u8; 32], PREP_WIDTHS[0]),
        mul_air.with_preprocessed(mul_root, PREP_WIDTHS[1]),
    ];
    let mut all_traces = vec![cpu, add, mul];
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
    let result = multi_prove_batched::<
        F,
        E,
        (),
        DefaultStarkHash,
        GenericProver<F, E, (), DefaultStarkHash>,
    >(
        pairs,
        &mut DefaultTranscript::<E>::new(&[]),
        #[cfg(feature = "disk-spill")]
        crate::storage_mode::StorageMode::Ram,
        ResidencyMode::Retain,
    );
    assert!(
        matches!(
            result,
            Err(crate::prover::ProvingError::PrecomputedCommitmentMismatch)
        ),
        "a pinned root the trace cannot reproduce must fail the prove"
    );
}

/// ★ The preprocessed round can NEVER be taller than the FRI, so
/// `reduce_iota_to_round`'s shift is never negative and no supplementary index
/// derivation is needed for it.
///
/// This is a structural invariant of `EpochShape::derive`, not a property of any
/// fixture: a table's preprocessed matrix is pushed with the SAME `h` that goes
/// into `heights`, in the same loop iteration, so `prep.dims`'s heights are a
/// SUBSET of `heights` — and `EpochShape::h_max` is the max over all of
/// `heights`. A prep matrix at height H therefore implies a TABLE at height H,
/// which puts the FRI's `h_max` at H or above.
///
/// Worth pinning because the obvious worry is wrong in an expensive direction.
/// A preprocessed table can be enormous — the LFM machine's BITWISE is 2^20 rows
/// in every registry entry — and it looks as though widening a preprocessed
/// round to include it could push the round above a small epoch's FRI. It cannot:
/// a preprocessed matrix only ever enters through a table that is itself in the
/// epoch at that height. `reduce_iota_to_round` fails closed on the inverted
/// case, so had this invariant not held, batched mode would have died on every
/// affected epoch rather than gone wrong quietly.
#[test_log::test]
fn the_preprocessed_round_is_never_taller_than_the_fri() {
    let options = folding_options();

    // The preprocessed fixture, where the round is strictly SHORTER.
    let (airs, _proof, lengths) = prove_preprocessed().expect("an honest preprocessed epoch");
    let shape = shape_of(&airs, &lengths);
    let prep_h = shape.prep.h_max().expect("non-empty");
    assert!(
        prep_h < shape.h_max(),
        "this fixture is the strictly-shorter case (prep {prep_h}, fri {})",
        shape.h_max()
    );
    assert!(
        crate::batched::round4::reduce_iota_to_round(0, shape.h_max(), prep_h).is_some(),
        "the reduction must be defined"
    );

    // ★ The equal case, which is the one a widened round produces: make the
    // TALLEST table preprocessed. The round then reaches the FRI's own h_max and
    // the shift is exactly zero — never negative.
    let (cpu, add, mul) = traces();
    let tall_airs = vec![
        new_cpu_air_with_lookup(&options).with_preprocessed([5u8; 32], 2),
        new_add_air_with_lookup(&options),
        new_mul_air_with_lookup(&options),
    ];
    let _ = (cpu, add, mul);
    let tall = shape_of(&tall_airs, &[8, 4, 4]);
    let tall_prep_h = tall.prep.h_max().expect("CPU is preprocessed");
    assert_eq!(
        tall_prep_h,
        tall.h_max(),
        "a preprocessed tallest table puts the round AT the FRI's h_max"
    );
    assert_eq!(
        crate::batched::round4::reduce_iota_to_round(7, tall.h_max(), tall_prep_h),
        Some(7),
        "and the reduction is then the identity, not a negative shift"
    );

    // The invariant itself, over both shapes.
    for s in [&shape, &tall] {
        assert!(
            s.prep.h_max().is_none_or(|h| h <= s.h_max()),
            "prep heights are a subset of table heights, so the round can never \
             exceed the FRI"
        );
    }
}
