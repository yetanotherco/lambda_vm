//! The batched prover: four mixed-height MMCS roots and ONE FRI instance per
//! epoch.
//!
//! # The phase architecture, and why it is not `multi_prove` with a different
//! commit call
//!
//! `multi_prove` forks the transcript per table after the LogUp challenges and
//! then runs aux-build → aux-commit → rounds 2-4 FUSED per table, so a table
//! never waits on another. Batching cannot keep that: a batched root cannot be
//! absorbed until every contributing matrix exists, so each batched commitment
//! is a phase BARRIER. What survives of the fork is nothing — every challenge
//! here is drawn from the one shared transcript, in a fixed table order, and the
//! verifier replays that order exactly.
//!
//! ```text
//!   shape histogram                       <- bound BEFORE the first root
//!   per table: main LDE -> prep/main MMCS builders           [barrier]
//!   prep_root, main_root
//!   LogUp challenges
//!   per table: aux trace + aux LDE -> aux MMCS builder       [barrier]
//!   aux_root
//!   per table: bus contribution
//!   per table: beta_t, composition parts -> parts builder    [barrier]
//!   parts_root
//!   per table: z_t, OOD evaluations
//!   per table: gamma_t
//!   ONE batched FRI: alpha, (beta, layer root)*, terminal, grinding, iotas
//!   openings, one table at a time
//! ```
//!
//! # ★ The cost the plan does not price: the barriers force LDE rebuilds
//!
//! MMCS-PLAN §3.3 makes one memory argument — stream the tree build so a height
//! group's LDEs are not simultaneously resident — and [`StreamingMmcsBuilder`]
//! delivers it. But the tree build is not the only consumer of a table's LDE.
//! Constraint evaluation (round 2), the OOD evaluations (round 3), the DEEP
//! codeword (round 4) and the query openings all read it, and a barrier sits
//! between every pair of those: `beta` cannot be drawn before `aux_root` is
//! absorbed, `z` cannot be drawn before `parts_root` is, `alpha` cannot be drawn
//! before every table's OOD values are, and the query indices do not exist until
//! the FRI is over.
//!
//! So a table's main and aux LDEs are needed in FIVE phases that cannot be
//! merged, and the prover must either hold them (`O(N)`, which is what batching
//! was supposed to remove) or rebuild them (one forward NTT each, per phase).
//! [`ResidencyMode`] selects, exactly as it does in `multi_prove`, and
//! [`BatchedProveStats`] reports what it cost — `main_lde_expansions` and
//! `aux_lde_expansions` are the honest budget, not an estimate.
//!
//! The composition parts are the exception and are ALWAYS retained: recomputing
//! them means re-running constraint evaluation, which is the dominant cost of a
//! prove. `parts_computations` stays at one per table, and the counter is there
//! so that stops being silent if it ever changes.
//!
//! # What is deliberately absent
//!
//! No device paths. The GPU mixed-height MMCS exists (`crypto/math-cuda/`) and
//! is box-gated; wiring it in is a separate step, and a batched prover that
//! silently fell back between host and device arms would make the residency
//! numbers above unreproducible. Under `--features cuda` this path compiles and
//! runs on the host.

use math::fft::bit_reversing::in_place_bit_reverse_permute;
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::traits::AsBytes;

use crate::batched::proof::{
    BatchedMultiProof, BatchedProveStats, BatchedQueryOpening, BatchedTableData, ResidencyLedger,
    lde_bytes,
};
use crate::batched::round4::commit_batched_fri;
use crate::batched::shape::{EpochShape, PinnedPrep, RoundShape, ShapeError};
use crate::config::StarkHash;
use crate::domain::Domain;
use crate::fri::batched::HeightCombiner;
use crate::fri::mmcs::{BorrowedMatrix, LeafSource, MixedMmcs, MixedOpening, StreamingMmcsBuilder};
use crate::fri::terminal::coeffs_from_terminal_codeword;
use crate::lookup::{BusPublicInputs, LOGUP_NUM_CHALLENGES};
use crate::proof::stark::PolynomialOpenings;
use crate::prover::{IsStarkProver, ProvingError, domain_and_twiddles};
use crate::residency_mode::ResidencyMode;
#[cfg(feature = "disk-spill")]
use crate::storage_mode::StorageMode;
use crate::trace::{LDETraceTable, TraceTable};
use crate::traits::AIR;

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;

impl From<ShapeError> for ProvingError {
    fn from(e: ShapeError) -> Self {
        ProvingError::WrongParameter(format!("batched epoch shape: {e}"))
    }
}

/// The AIR, its trace and its public inputs, as `multi_prove` takes them.
pub type BatchedAirTracePair<'a, Field, FieldExtension, PI> = (
    &'a dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
    &'a mut TraceTable<Field, FieldExtension>,
    &'a PI,
);

/// A retention slot for one table's main LDE: `Some` while the buffer is being
/// held between phases, `None` while it is out on loan or was dropped.
type MainSlots<'a, Field> = &'a mut [Option<(Vec<FieldElement<Field>>, usize)>];
/// The same for the auxiliary LDE.
type AuxSlots<'a, FieldExtension> = &'a mut [Option<(Vec<FieldElement<FieldExtension>>, usize)>];

/// A table's LDE buffers, alive only for as long as the current phase needs
/// them, and accounted for while they are.
struct LdePair<Field: IsField, FieldExtension: IsField> {
    main: (Vec<FieldElement<Field>>, usize),
    aux: (Vec<FieldElement<FieldExtension>>, usize),
    bytes: usize,
}

/// Prove one epoch with batched commitments.
///
/// `expected_prep`, when supplied, is the registry's committed preprocessed root
/// and the widths it was committed over (M-6). The prover compares its own
/// against them and fails fast, preserving the property the per-table path gets
/// from `air.precomputed_commitment()`: a stale preprocessed constant is caught
/// here rather than by every future verifier.
///
/// `None` is permissive here — it is how the root is generated in the first
/// place. That is the opposite of the verifier's disposition; see
/// [`PinnedPrep`].
#[allow(clippy::too_many_arguments)]
pub fn multi_prove_batched<Field, FieldExtension, PI, H, P>(
    mut air_trace_pairs: Vec<BatchedAirTracePair<'_, Field, FieldExtension, PI>>,
    transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone + Send),
    expected_prep: Option<PinnedPrep<'_>>,
    #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
    residency: ResidencyMode,
) -> Result<
    (
        BatchedMultiProof<Field, FieldExtension, PI>,
        BatchedProveStats,
    ),
    ProvingError,
>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + Copy + 'static,
    FieldExtension: IsField + Send + Sync + Copy + 'static,
    FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
    FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    PI: Send + Sync + Clone,
    H: StarkHash,
    P: IsStarkProver<Field, FieldExtension, PI, H> + ?Sized,
    // The same two bounds `multi_prove` carries: under `disk-spill` the aux
    // trace is spilled through an mmap backing, which only a field whose
    // `BaseType` is plain data can be laid out in.
    <Field as IsField>::BaseType: math::spill_safe::SpillSafe,
    <FieldExtension as IsField>::BaseType: math::spill_safe::SpillSafe,
{
    let num_tables = air_trace_pairs.len();
    let mut stats = BatchedProveStats::default();
    // Two accounts, because they behave differently on purpose: the trace LDEs
    // must stay flat in the table count, the retained parts must not.
    let mut ledger = ResidencyLedger::default();
    let mut parts_ledger = ResidencyLedger::default();

    // =====================================================================
    // Phase 0 — domains, shape, and the shape binding
    // =====================================================================
    let mut domains = Vec::with_capacity(num_tables);
    let mut twiddles = Vec::with_capacity(num_tables);
    for (air, trace, _) in &*air_trace_pairs {
        let (domain, tw) = domain_and_twiddles(*air, trace.num_rows());
        domains.push(domain);
        twiddles.push(tw);
    }

    let airs: Vec<&dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>> =
        air_trace_pairs.iter().map(|(air, _, _)| *air).collect();
    let trace_lengths: Vec<usize> = domains
        .iter()
        .map(|d| d.interpolation_domain_size)
        .collect();
    let (shape, params) = EpochShape::derive(&airs, &trace_lengths)?;
    let h_max = shape.h_max();
    let coset_offset = FieldElement::<Field>::from(params.coset_offset);

    // ★ Addendum A's recommendation S, adopted. `commit_batched_fri` binds the
    // shape again in round 4, which is where the batched FRI's own challenges
    // need it; binding it HERE, before the first root, is what turns "no
    // rounds-1-3 challenge is shape-exploitable" from a collision-resistance
    // argument into a transcript-ordering one. Two field-sized absorptions per
    // table, and every later challenge inherits the binding.
    crate::fri::batched::absorb_shape_histogram::<FieldExtension, _>(
        transcript,
        &shape.heights,
        &shape.total_widths(),
    );

    // =====================================================================
    // Phase 1 — the preprocessed and main rounds, one main LDE pass per table
    // =====================================================================
    // Both builders are fed from the SAME expansion: a preprocessed table's
    // precomputed columns and its multiplicity columns are two column ranges of
    // one row-major main LDE, exactly as `commit_main_trace` splits them.
    let mut prep_builder =
        (!shape.prep.is_empty()).then(|| StreamingMmcsBuilder::<Field, H>::new(&shape.prep.dims));
    let mut main_builder = StreamingMmcsBuilder::<Field, H>::new(&shape.main.dims);
    let mut retained_main: Vec<Option<(Vec<FieldElement<Field>>, usize)>> =
        (0..num_tables).map(|_| None).collect();

    for table in 0..num_tables {
        let (_, trace, _) = &air_trace_pairs[table];
        let (main_data, total_cols) = P::expand_main_lde_row_major(
            trace,
            &domains[table],
            &twiddles[table],
            #[cfg(feature = "disk-spill")]
            storage_mode,
        );
        stats.main_lde_expansions += 1;
        let bytes = lde_bytes::<Field>(main_data.len());
        ledger.alloc(bytes);

        let height = shape.heights[table];
        let num_precomputed = total_cols - matrix_width(&shape.main, table);

        if let Some(builder) = prep_builder.as_mut()
            && num_precomputed > 0
        {
            let src = vec![BorrowedMatrix::RowMajorNatural {
                data: &main_data,
                stride: total_cols,
                col_start: 0,
                width: num_precomputed,
                log_height: height,
            }];
            builder.absorb(&src, 0);
        }
        let src = vec![BorrowedMatrix::RowMajorNatural {
            data: &main_data,
            stride: total_cols,
            col_start: num_precomputed,
            width: total_cols - num_precomputed,
            log_height: height,
        }];
        main_builder.absorb(&src, 0);

        // The root is what Fiat-Shamir needs; the buffer is not. Under
        // `RecomputeLde` it dies here and every later phase rebuilds it.
        match residency {
            ResidencyMode::Retain => retained_main[table] = Some((main_data, total_cols)),
            ResidencyMode::RecomputeLde => {
                drop(main_data);
                ledger.free(bytes);
            }
        }
    }

    let prep_mmcs = prep_builder.map(StreamingMmcsBuilder::finish);
    let main_mmcs = main_builder.finish();
    let prep_root = prep_mmcs.as_ref().map(MixedMmcs::root);
    // The widths are compared first because they are the more legible failure:
    // a registry whose entry predates a change to some AIR's precomputed column
    // count disagrees here in a way that names the cause, instead of surfacing
    // as a root mismatch that could equally be a stale constant.
    if let Some(expected) = expected_prep {
        if expected.widths != shape.prep.widths().as_slice() {
            return Err(ProvingError::PrecomputedCommitmentMismatch);
        }
        if prep_root.as_ref() != Some(expected.root) {
            return Err(ProvingError::PrecomputedCommitmentMismatch);
        }
    }
    if let Some(root) = prep_root {
        transcript.append_bytes(&root);
    }
    let main_root = main_mmcs.root();
    transcript.append_bytes(&main_root);

    // =====================================================================
    // Phase 2 — LogUp challenges, then the auxiliary round
    // =====================================================================
    let needs_lookup = airs.iter().any(|air| air.has_aux_trace());
    let lookup_challenges: Vec<FieldElement<FieldExtension>> = if needs_lookup {
        (0..LOGUP_NUM_CHALLENGES)
            .map(|_| transcript.sample_field_element())
            .collect()
    } else {
        Vec::new()
    };

    let mut bus_public_inputs: Vec<Option<BusPublicInputs<FieldExtension>>> =
        (0..num_tables).map(|_| None).collect();
    let mut aux_builder = (!shape.aux.is_empty())
        .then(|| StreamingMmcsBuilder::<FieldExtension, H>::new(&shape.aux.dims));
    let mut retained_aux: Vec<Option<(Vec<FieldElement<FieldExtension>>, usize)>> =
        (0..num_tables).map(|_| None).collect();

    for table in 0..num_tables {
        let (air, trace, _) = &mut air_trace_pairs[table];
        if !air.has_aux_trace() {
            continue;
        }
        bus_public_inputs[table] = air.build_auxiliary_trace(trace, &lookup_challenges);

        #[cfg(feature = "disk-spill")]
        if storage_mode == StorageMode::Disk {
            trace
                .spill_aux_to_disk()
                .map_err(|e| ProvingError::DiskSpill(format!("aux trace: {e}")))?;
        }

        let Some(builder) = aux_builder.as_mut() else {
            continue;
        };
        let (aux_data, aux_cols) = P::expand_aux_lde_row_major(
            trace,
            &domains[table],
            &twiddles[table],
            #[cfg(feature = "disk-spill")]
            storage_mode,
        );
        stats.aux_lde_expansions += 1;
        let bytes = lde_bytes::<FieldExtension>(aux_data.len());
        ledger.alloc(bytes);
        let src = vec![BorrowedMatrix::RowMajorNatural {
            data: &aux_data,
            stride: aux_cols,
            col_start: 0,
            width: aux_cols,
            log_height: shape.heights[table],
        }];
        builder.absorb(&src, 0);
        match residency {
            ResidencyMode::Retain => retained_aux[table] = Some((aux_data, aux_cols)),
            ResidencyMode::RecomputeLde => {
                drop(aux_data);
                ledger.free(bytes);
            }
        }
    }

    let aux_mmcs = aux_builder.map(StreamingMmcsBuilder::finish);
    let aux_root = aux_mmcs.as_ref().map(MixedMmcs::root);
    if let Some(root) = aux_root {
        transcript.append_bytes(&root);
    }

    // =====================================================================
    // Phase 3 — bus contributions, beta per table, the composition-parts round
    // =====================================================================
    for bpi in bus_public_inputs.iter().flatten() {
        transcript.append_field_element(&bpi.table_contribution);
    }

    let mut parts_builder = StreamingMmcsBuilder::<FieldExtension, H>::new(&shape.parts.dims);
    let mut retained_parts: Vec<Vec<Vec<FieldElement<FieldExtension>>>> =
        (0..num_tables).map(|_| Vec::new()).collect();

    for table in 0..num_tables {
        let beta: FieldElement<FieldExtension> = transcript.sample_field_element();
        let (air, _, pub_inputs) = &air_trace_pairs[table];
        let domain = &domains[table];

        let num_transition_constraints = air.context().num_transition_constraints;
        let num_boundary_constraints = air
            .boundary_constraints(
                pub_inputs,
                &lookup_challenges,
                bus_public_inputs[table].as_ref(),
                domain.interpolation_domain_size,
            )
            .constraints
            .len();
        let mut coefficients: Vec<FieldElement<FieldExtension>> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &beta))
                .take(num_boundary_constraints + num_transition_constraints)
                .collect();
        let transition_coefficients: Vec<_> =
            coefficients.drain(..num_transition_constraints).collect();
        let boundary_coefficients = coefficients;

        let ldes = materialize_ldes::<Field, FieldExtension, PI, H, P>(
            table,
            &air_trace_pairs,
            &domains,
            &twiddles,
            &shape,
            &mut retained_main,
            &mut retained_aux,
            &mut stats,
            &mut ledger,
            residency,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        );
        let mut lde_trace = lde_trace_of(&ldes, air.step_size(), domain.blowup_factor);

        let computed = P::compute_composition_parts(
            *air,
            pub_inputs,
            domain,
            &twiddles[table],
            &mut lde_trace,
            &lookup_challenges,
            bus_public_inputs[table].as_ref(),
            &transition_coefficients,
            &boundary_coefficients,
        )?;
        stats.parts_computations += 1;
        let parts = computed.parts;

        let parts_bytes: usize = parts
            .iter()
            .map(|p| lde_bytes::<FieldExtension>(p.len()))
            .sum();
        parts_ledger.alloc(parts_bytes);
        let src = vec![BorrowedMatrix::ColMajorNatural {
            cols: &parts,
            log_height: shape.heights[table],
        }];
        parts_builder.absorb(&src, 0);

        // Parts are RETAINED: rebuilding them is a second constraint evaluation.
        retained_parts[table] = parts;
        release_ldes(
            ldes,
            &mut retained_main,
            &mut retained_aux,
            table,
            &mut ledger,
            residency,
        );
        drop(lde_trace);
    }

    let parts_mmcs = parts_builder.finish();
    let parts_root = parts_mmcs.root();
    transcript.append_bytes(&parts_root);

    // =====================================================================
    // Phase 4 — z per table, OOD evaluations
    // =====================================================================
    let mut zs = Vec::with_capacity(num_tables);
    let mut round3s = Vec::with_capacity(num_tables);
    let mut ood_blocks = Vec::with_capacity(num_tables);

    for table in 0..num_tables {
        let (air, _, _) = &air_trace_pairs[table];
        let domain = &domains[table];
        // `sample_z_ood_with_domain_params` rather than `sample_z_ood`: the
        // verifier has the trace length and the blowup but not the domain
        // vectors, so naming the routine both sides can reach is what makes the
        // two agree by construction instead of by two call sites coinciding.
        let z = transcript.sample_z_ood_with_domain_params(
            domain.interpolation_domain_size,
            domain.interpolation_domain_size * domain.blowup_factor,
            &coset_offset,
        );

        let ldes = materialize_ldes::<Field, FieldExtension, PI, H, P>(
            table,
            &air_trace_pairs,
            &domains,
            &twiddles,
            &shape,
            &mut retained_main,
            &mut retained_aux,
            &mut stats,
            &mut ledger,
            residency,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        );
        let mut lde_trace = lde_trace_of(&ldes, air.step_size(), domain.blowup_factor);
        let round3 = P::round_3_evaluate_polynomials_in_out_of_domain_element(
            *air,
            domain,
            &mut lde_trace,
            &mut retained_parts[table],
            &z,
        );
        release_ldes(
            ldes,
            &mut retained_main,
            &mut retained_aux,
            table,
            &mut ledger,
            residency,
        );
        drop(lde_trace);

        let (block0, block1) = P::ood_layout(*air).split_full(&round3.trace_ood_evaluations);
        for block in [&block0, &block1] {
            for col in block.columns().iter() {
                for elem in col.iter() {
                    transcript.append_field_element(elem);
                }
            }
        }
        for element in round3.composition_poly_parts_ood_evaluation.iter() {
            transcript.append_field_element(element);
        }

        zs.push(z);
        ood_blocks.push((block0, block1));
        round3s.push(round3);
    }

    // =====================================================================
    // Phase 5 — gamma per table, then ONE batched FRI
    // =====================================================================
    let gammas: Vec<FieldElement<FieldExtension>> = (0..num_tables)
        .map(|_| transcript.sample_field_element())
        .collect();

    let mut standalone_coeffs: Vec<Option<Vec<FieldElement<FieldExtension>>>> =
        (0..num_tables).map(|_| None).collect();

    let commit = {
        let air_trace_pairs = &air_trace_pairs;
        let domains = &domains;
        let twiddles = &twiddles;
        let shape = &shape;
        // `&mut`: the DEEP host loop repopulates a table's part evals from the
        // resident handle when the device-only gate left them empty.
        let retained_parts = &mut retained_parts;
        let round3s = &round3s;
        let zs = &zs;
        let gammas = &gammas;
        let retained_main = &mut retained_main;
        let retained_aux = &mut retained_aux;
        let stats = &mut stats;
        let ledger = &mut ledger;
        let standalone_coeffs = &mut standalone_coeffs;
        let coset_offset_ref = &coset_offset;

        commit_batched_fri::<Field, FieldExtension, _, H, _>(
            transcript,
            &shape.heights,
            &shape.total_widths(),
            move |alpha, plan| {
                let mut combiner = HeightCombiner::new(*alpha);
                // Ascending table order, which is also `plan.batched`'s order —
                // absorption order is what defines the alpha powers, so the two
                // must not be allowed to drift apart.
                for table in 0..num_tables {
                    let (air, _, _) = &air_trace_pairs[table];
                    let domain = &domains[table];
                    let ldes = materialize_ldes::<Field, FieldExtension, PI, H, P>(
                        table,
                        air_trace_pairs,
                        domains,
                        twiddles,
                        shape,
                        retained_main,
                        retained_aux,
                        stats,
                        ledger,
                        residency,
                        #[cfg(feature = "disk-spill")]
                        storage_mode,
                    );
                    let mut lde_trace = lde_trace_of(&ldes, air.step_size(), domain.blowup_factor);
                    let mut deep = deep_codeword::<Field, FieldExtension, PI, H, P>(
                        *air,
                        domain,
                        &mut lde_trace,
                        &mut retained_parts[table],
                        &round3s[table],
                        &zs[table],
                        &gammas[table],
                    );
                    release_ldes(ldes, retained_main, retained_aux, table, ledger, residency);
                    drop(lde_trace);
                    in_place_bit_reverse_permute(&mut deep);

                    if plan.batched.contains(&table) {
                        combiner.absorb(&deep, shape.heights[table]);
                    } else {
                        // A standalone table's terminal codeword IS this
                        // codeword; the proof carries the polynomial it
                        // evaluates, at its own degree bound.
                        let log_degree = (shape.heights[table] as u32) - params.blowup_log;
                        standalone_coeffs[table] = Some(coeffs_from_terminal_codeword::<
                            Field,
                            FieldExtension,
                        >(
                            &deep, coset_offset_ref, log_degree
                        ));
                    }
                }
                combiner.finish()
            },
            &coset_offset,
            params.blowup_log,
            params.final_poly_log_degree,
            params.grinding_factor,
            params.num_queries,
        )
    };

    // =====================================================================
    // Phase 6 — openings, one table at a time
    // =====================================================================
    let iotas = commit.iotas.clone();
    let fri_decommitments = crate::fri::query_phase::<FieldExtension, H>(&commit.layers, &iotas);

    let mut prep_openings = empty_openings::<Field>(&iotas, shape.prep.tables.len());
    let mut main_openings = empty_openings::<Field>(&iotas, shape.main.tables.len());
    let mut aux_openings = empty_openings::<FieldExtension>(&iotas, shape.aux.tables.len());
    let mut parts_openings = empty_openings::<FieldExtension>(&iotas, shape.parts.tables.len());

    // ★ Each round is read in ITS OWN index space, and the reduction happens
    // exactly once, here. Doing it inside the read would be wrong twice over: a
    // round shorter than the FRI would be asked for a leaf it does not have
    // (the prep round's `h_max` is below the FRI's whenever the tallest
    // preprocessed table is not the tallest table), and a round that reduced
    // again on the way out would land somewhere else entirely.
    let prep_iotas = prep_mmcs
        .as_ref()
        .map(|mmcs| reduced_iotas(&iotas, h_max, mmcs.h_max()));
    let main_iotas = reduced_iotas(&iotas, h_max, main_mmcs.h_max());
    let aux_iotas = aux_mmcs
        .as_ref()
        .map(|mmcs| reduced_iotas(&iotas, h_max, mmcs.h_max()));
    let parts_iotas = reduced_iotas(&iotas, h_max, parts_mmcs.h_max());

    for table in 0..num_tables {
        let (air, _, _) = &air_trace_pairs[table];
        let ldes = materialize_ldes::<Field, FieldExtension, PI, H, P>(
            table,
            &air_trace_pairs,
            &domains,
            &twiddles,
            &shape,
            &mut retained_main,
            &mut retained_aux,
            &mut stats,
            &mut ledger,
            residency,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        );
        let _ = air;
        let height = shape.heights[table];
        let (main_data, total_cols) = &ldes.main;
        let num_precomputed = total_cols - matrix_width(&shape.main, table);

        if let (Some(mmcs), Some(m)) = (prep_mmcs.as_ref(), matrix_index(&shape.prep, table)) {
            let src = vec![BorrowedMatrix::RowMajorNatural {
                data: main_data,
                stride: *total_cols,
                col_start: 0,
                width: num_precomputed,
                log_height: height,
            }];
            let indices = prep_iotas.as_ref().expect("the prep MMCS exists here");
            fill_openings(mmcs, m, &src, indices, &mut prep_openings);
        }
        if let Some(m) = matrix_index(&shape.main, table) {
            let src = vec![BorrowedMatrix::RowMajorNatural {
                data: main_data,
                stride: *total_cols,
                col_start: num_precomputed,
                width: total_cols - num_precomputed,
                log_height: height,
            }];
            fill_openings(&main_mmcs, m, &src, &main_iotas, &mut main_openings);
        }
        if let (Some(mmcs), Some(m)) = (aux_mmcs.as_ref(), matrix_index(&shape.aux, table)) {
            let (aux_data, aux_cols) = &ldes.aux;
            let src = vec![BorrowedMatrix::RowMajorNatural {
                data: aux_data,
                stride: *aux_cols,
                col_start: 0,
                width: *aux_cols,
                log_height: height,
            }];
            let indices = aux_iotas.as_ref().expect("the aux MMCS exists here");
            fill_openings(mmcs, m, &src, indices, &mut aux_openings);
        }
        if let Some(m) = matrix_index(&shape.parts, table) {
            let src = vec![BorrowedMatrix::ColMajorNatural {
                cols: &retained_parts[table],
                log_height: height,
            }];
            fill_openings(&parts_mmcs, m, &src, &parts_iotas, &mut parts_openings);
        }

        release_ldes(
            ldes,
            &mut retained_main,
            &mut retained_aux,
            table,
            &mut ledger,
            residency,
        );
    }

    let queries = (0..iotas.len())
        .map(|q| BatchedQueryOpening {
            prep: prep_mmcs.as_ref().map(|mmcs| {
                let indices = prep_iotas.as_ref().expect("the prep MMCS exists here");
                assemble(mmcs, indices[q], &mut prep_openings, q)
                    .expect("the prep round was opened at these very indices")
            }),
            main: assemble(&main_mmcs, main_iotas[q], &mut main_openings, q)
                .expect("the main round was opened at these very indices"),
            aux: aux_mmcs.as_ref().map(|mmcs| {
                let indices = aux_iotas.as_ref().expect("the aux MMCS exists here");
                assemble(mmcs, indices[q], &mut aux_openings, q)
                    .expect("the aux round was opened at these very indices")
            }),
            parts: assemble(&parts_mmcs, parts_iotas[q], &mut parts_openings, q)
                .expect("the parts round was opened at these very indices"),
            fri: fri_decommitments[q].clone(),
        })
        .collect();

    let tables = (0..num_tables)
        .map(|table| {
            let (block0, block1) = ood_blocks[table].clone();
            BatchedTableData {
                trace_length: trace_lengths[table],
                trace_ood_evaluations: block0,
                trace_ood_next_evaluations: block1,
                composition_poly_parts_ood_evaluation: round3s[table]
                    .composition_poly_parts_ood_evaluation
                    .clone(),
                bus_public_inputs: bus_public_inputs[table].clone(),
                public_inputs: air_trace_pairs[table].2.clone(),
                standalone_final_poly_coeffs: standalone_coeffs[table].clone(),
            }
        })
        .collect();

    stats.peak_trace_lde_bytes = ledger.peak();
    stats.retained_parts_bytes = parts_ledger.peak();
    stats.peak_lde_bytes = stats.peak_trace_lde_bytes + stats.retained_parts_bytes;

    Ok((
        BatchedMultiProof {
            tables,
            prep_root,
            main_root,
            aux_root,
            parts_root,
            fri_layer_roots: commit.layer_roots,
            fri_final_poly_coeffs: commit.final_poly_coeffs,
            nonce: commit.nonce,
            queries,
        },
        stats,
    ))
}

/// Matrix index of `table` inside `round`, or `None` when it does not
/// contribute one.
fn matrix_index(round: &RoundShape, table: usize) -> Option<usize> {
    round.tables.iter().position(|&t| t == table)
}

/// The width `table` contributes to `round`. Zero when it contributes nothing.
fn matrix_width(round: &RoundShape, table: usize) -> usize {
    matrix_index(round, table).map_or(0, |m| round.dims[m].1)
}

#[allow(clippy::type_complexity)]
fn empty_openings<E: IsField>(
    iotas: &[usize],
    num_matrices: usize,
) -> Vec<Vec<Option<PolynomialOpenings<E>>>> {
    iotas
        .iter()
        .map(|_| (0..num_matrices).map(|_| None).collect())
        .collect()
}

/// Read one matrix's row pair at every query, so a table's openings are
/// harvested while its LDE is alive and never after.
fn fill_openings<E, H, S>(
    mmcs: &MixedMmcs<E, H>,
    matrix: usize,
    source: &S,
    iotas: &[usize],
    out: &mut [Vec<Option<PolynomialOpenings<E>>>],
) where
    E: IsField + 'static,
    H: StarkHash,
    S: LeafSource<E>,
    FieldElement<E>: AsBytes + Sync + Send,
{
    // `iotas` are in THIS round's index space already (see `reduced_iotas`).
    // Passing the FRI's raw indices here does not corrupt anything quietly: a
    // shorter round rejects them as out of range and produces no opening at
    // all, which is what `the_preprocessed_round_is_committed_and_authenticates`
    // caught the first time this was written the other way round.
    for (q, &iota) in iotas.iter().enumerate() {
        let Some(leaf) = mmcs.row_pair_leaf(iota, matrix) else {
            continue;
        };
        let mut evaluations = Vec::new();
        source.append_row(0, 2 * leaf, &mut evaluations);
        let mut evaluations_sym = Vec::new();
        source.append_row(0, 2 * leaf + 1, &mut evaluations_sym);
        out[q][matrix] = Some(PolynomialOpenings {
            proof: crypto::merkle_tree::proof::Proof {
                merkle_path: Vec::new(),
            },
            evaluations,
            evaluations_sym,
        });
    }
}

/// Reduce every FRI query index into one round's index space.
///
/// `h_max_round <= h_max_fri` always holds for a round of this epoch — a round
/// commits a subset of the epoch's tables, so its tallest matrix cannot exceed
/// the epoch's — which is why this is infallible here and
/// `reduce_iota_to_round` returns an `Option` on the verifier's path, where the
/// heights are proof-supplied.
fn reduced_iotas(iotas: &[usize], h_max_fri: usize, h_max_round: usize) -> Vec<usize> {
    iotas
        .iter()
        .map(|&iota| {
            crate::batched::round4::reduce_iota_to_round(iota, h_max_fri, h_max_round)
                .expect("a round of this epoch is never taller than the epoch")
        })
        .collect()
}

/// Turn one query's per-matrix rows into a [`MixedOpening`] by attaching the
/// round's shared authentication path.
///
/// `iota` is already in this round's index space — see [`reduced_iotas`].
fn assemble<E, H>(
    mmcs: &MixedMmcs<E, H>,
    iota: usize,
    openings: &mut [Vec<Option<PolynomialOpenings<E>>>],
    query: usize,
) -> Option<MixedOpening<E>>
where
    E: IsField + 'static,
    H: StarkHash,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let proof = mmcs.auth_path(iota)?;
    let per_matrix = openings[query]
        .iter_mut()
        .map(|slot| slot.take())
        .collect::<Option<Vec<_>>>()?;
    Some(MixedOpening { proof, per_matrix })
}

/// Build (or take back) a table's main and aux LDEs for the phase about to read
/// them.
#[allow(clippy::too_many_arguments)]
fn materialize_ldes<Field, FieldExtension, PI, H, P>(
    table: usize,
    air_trace_pairs: &[BatchedAirTracePair<'_, Field, FieldExtension, PI>],
    domains: &[std::sync::Arc<Domain<Field>>],
    twiddles: &[std::sync::Arc<crate::prover::LdeTwiddles<Field>>],
    shape: &EpochShape,
    retained_main: MainSlots<'_, Field>,
    retained_aux: AuxSlots<'_, FieldExtension>,
    stats: &mut BatchedProveStats,
    ledger: &mut ResidencyLedger,
    residency: ResidencyMode,
    #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
) -> LdePair<Field, FieldExtension>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + Copy + 'static,
    FieldExtension: IsField + Send + Sync + Copy + 'static,
    FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
    FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    H: StarkHash,
    P: IsStarkProver<Field, FieldExtension, PI, H> + ?Sized,
{
    let (_, trace, _) = &air_trace_pairs[table];
    let mut bytes = 0usize;

    let main = match retained_main[table].take() {
        Some(lde) => lde,
        None => {
            let lde = P::expand_main_lde_row_major(
                trace,
                &domains[table],
                &twiddles[table],
                #[cfg(feature = "disk-spill")]
                storage_mode,
            );
            stats.main_lde_expansions += 1;
            let b = lde_bytes::<Field>(lde.0.len());
            ledger.alloc(b);
            bytes += b;
            lde
        }
    };

    let aux = if matrix_index(&shape.aux, table).is_some() {
        match retained_aux[table].take() {
            Some(lde) => lde,
            None => {
                let lde = P::expand_aux_lde_row_major(
                    trace,
                    &domains[table],
                    &twiddles[table],
                    #[cfg(feature = "disk-spill")]
                    storage_mode,
                );
                stats.aux_lde_expansions += 1;
                let b = lde_bytes::<FieldExtension>(lde.0.len());
                ledger.alloc(b);
                bytes += b;
                lde
            }
        }
    } else {
        (Vec::new(), 0)
    };

    let _ = residency;
    LdePair { main, aux, bytes }
}

/// Give a table's LDEs back to the retention slots, or drop them.
fn release_ldes<Field: IsField, FieldExtension: IsField>(
    ldes: LdePair<Field, FieldExtension>,
    retained_main: MainSlots<'_, Field>,
    retained_aux: AuxSlots<'_, FieldExtension>,
    table: usize,
    ledger: &mut ResidencyLedger,
    residency: ResidencyMode,
) {
    match residency {
        ResidencyMode::Retain => {
            retained_main[table] = Some(ldes.main);
            if ldes.aux.1 > 0 {
                retained_aux[table] = Some(ldes.aux);
            }
        }
        ResidencyMode::RecomputeLde => {
            drop(ldes.main);
            drop(ldes.aux);
            ledger.free(ldes.bytes);
        }
    }
}

fn lde_trace_of<Field, FieldExtension>(
    ldes: &LdePair<Field, FieldExtension>,
    step_size: usize,
    blowup_factor: usize,
) -> LDETraceTable<Field, FieldExtension>
where
    Field: IsFFTField + IsSubFieldOf<FieldExtension>,
    FieldExtension: IsField,
{
    LDETraceTable::from_row_major(
        ldes.main.0.clone(),
        ldes.main.1,
        ldes.aux.0.clone(),
        ldes.aux.1,
        step_size,
        blowup_factor,
    )
}

/// One table's DEEP composition codeword, in NATURAL order.
#[allow(clippy::too_many_arguments)]
fn deep_codeword<Field, FieldExtension, PI, H, P>(
    air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
    domain: &Domain<Field>,
    // `&mut` to match `compute_deep_composition_poly_evaluations`, whose host
    // loop downloads the resident trace and part evals in place when the
    // device-only gate left them empty.
    lde_trace: &mut LDETraceTable<Field, FieldExtension>,
    composition_parts: &mut [Vec<FieldElement<FieldExtension>>],
    round3: &crate::prover::Round3<FieldExtension>,
    z: &FieldElement<FieldExtension>,
    gamma: &FieldElement<FieldExtension>,
) -> Vec<FieldElement<FieldExtension>>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + Copy + 'static,
    FieldExtension: IsField + Send + Sync + Copy + 'static,
    FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
    FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    H: StarkHash,
    P: IsStarkProver<Field, FieldExtension, PI, H> + ?Sized,
{
    let n_terms_composition_poly = composition_parts.len();
    let layout = P::ood_layout(air);
    let num_terms_trace = layout.num_surviving();

    let mut deep_composition_coefficients: Vec<FieldElement<FieldExtension>> =
        core::iter::successors(Some(FieldElement::one()), |x| Some(x * gamma))
            .take(n_terms_composition_poly + num_terms_trace)
            .collect();
    let trace_term_powers: Vec<_> = deep_composition_coefficients
        .drain(..num_terms_trace)
        .collect();
    let trace_term_coeffs = layout.build_trace_term_coeffs(&trace_term_powers);
    let gammas = deep_composition_coefficients;

    P::compute_deep_composition_poly_evaluations(
        lde_trace,
        composition_parts,
        round3,
        z,
        domain,
        &domain.trace_primitive_root,
        &gammas,
        &trace_term_coeffs,
    )
}
