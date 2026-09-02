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
//!   per table: main LDE -> per-table prep tree + main MMCS builder [barrier]
//!   per-table prep roots (from the AIR set), main_root
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
//! # The GPU commit
//!
//! Under `--features cuda`, on Goldilocks (base + ext3), all four MMCS trees
//! (main / aux / parts) are committed ON THE DEVICE: the keccak leaf hash and
//! the climb run on the GPU (`crypto/math-cuda/`'s mixed-height MMCS), and only
//! the digest layers come back to rebuild the host-side [`MixedMmcs`] via
//! [`MixedMmcs::from_heap_nodes`] — so the openings (row values from the LDE, one
//! shared auth path per query) are served exactly as before. The host
//! [`StreamingMmcsBuilder`] is skipped entirely for those rounds. This is a
//! DELIBERATE mode select (see `device_commit_enabled`), not a silent per-call
//! host↔device fallback — a device error is a hard abort. Off cuda / off
//! Goldilocks the host builder commits as it always did.
//!
//! ## Streaming, so it runs under either residency
//!
//! The device commit is STREAMING ([`math_cuda::mmcs::StreamingMixedMmcs`], the
//! device twin of [`StreamingMmcsBuilder`]): the main and aux rounds absorb each
//! table's LDE into the device tree INSIDE the round loop — the moment it is
//! produced — and free it, so only the per-leaf sponges stay resident, never all
//! LDEs at once. It therefore runs under BOTH `Retain` and `RecomputeLde`. This
//! is not cosmetic: the old all-at-once commit retained every LDE just to feed it
//! (O(N) memory), which OOM'd a large epoch; streaming under `RecomputeLde` is
//! what lets a big block commit on the GPU at all. Parts are ALWAYS retained
//! (recomputing them re-runs constraint eval), so that round streams one slab at
//! a time from the retained parts POST-loop instead.
//!
//! ## No silent fallback, enforced in release
//!
//! Because the device commit is authoritative, a silent device-side corruption
//! (a layout / heap bug that ships a wrong-but-self-consistent tree) would
//! otherwise ship a bad proof: the byte-identical device-vs-host parity that
//! guards these paths is `debug_assert!` / test-only and compiled OUT of a
//! release prove. So a device commit runs a canary (`xcheck_device_commit`) that
//! re-authenticates a handful of the committed tree's leaves against the LDE on
//! the host, through the same `open_batch` / `verify_batch` the verifier trusts.
//! Its cost is `k` leaf paths against an `O(N)` tree build; a mismatch is a hard
//! [`ProvingError`]. The canary reads the LDEs back, so main/aux run it only
//! under `Retain` (`RecomputeLde` dropped them — the debug parity tests cover
//! that path); parts, always retained, canary every time.
//!
//! The batched FRI commit follows the SAME policy. Its GPU drive is selected in
//! [`crate::batched::round4::commit_batched_fri`] (not inside
//! `crate::fri::batched`'s `batched_commit_phase`, which stays a pure host
//! build): when the device path is selected and a CUDA op fails, the error
//! propagates as a hard abort rather than silently restoring the transcript and
//! rebuilding on the host. The only sanctioned host build is NON-selection (off
//! cuda, off Goldilocks, or below the GPU threshold), reached before any
//! transcript mutation. (A silent FRI-side corruption — a wrong-but-no-error
//! device root — is not yet canaried here; the FRI verifier's own
//! fold-consistency check is the current backstop. Deferred.)

use math::fft::bit_reversing::in_place_bit_reverse_permute_row_major;
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use math::traits::AsBytes;

use crate::batched::proof::{
    BatchedMultiProof, BatchedProveStats, BatchedQueryOpening, BatchedTableData, ResidencyLedger,
    lde_bytes,
};
use crate::batched::round4::commit_batched_fri;
use crate::batched::shape::{EpochShape, RoundShape, ShapeError};
use crate::config::BatchedMerkleTreeBackend;
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
use crypto::merkle_tree::merkle::MerkleTree;

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
/// A preprocessed table's own row-pair tree — `None` for a table with no
/// preprocessed columns.
type PrepTreeSlot<B> = Option<std::sync::Arc<MerkleTree<B>>>;

/// A table's LDE buffers, alive only for as long as the current phase needs
/// them, and accounted for while they are.
struct LdePair<Field: IsField, FieldExtension: IsField> {
    main: (Vec<FieldElement<Field>>, usize),
    aux: (Vec<FieldElement<FieldExtension>>, usize),
    bytes: usize,
    /// Resident device LDE handles, present when `materialize_ldes` recomputed on
    /// the GPU. `lde_trace_take` attaches them to the `LDETraceTable` so R2
    /// constraint eval and R4 DEEP fire their device paths (they read
    /// `gpu_main`/`gpu_aux`); the host LDE above is kept alongside so `want_host`
    /// still drains parts to host and the host fallback stays valid
    /// (`host_trace_empty` = false). `None` on the host / retained paths.
    #[cfg(feature = "cuda")]
    gpu_main: Option<math_cuda::lde::GpuLdeBase>,
    #[cfg(feature = "cuda")]
    gpu_aux: Option<math_cuda::lde::GpuLdeExt3>,
}

/// Prove one epoch with batched commitments.
///
/// Preprocessed matrices are committed per table (the trees
/// `air.precomputed_commitment()` pins), so the per-table path's stale-constant
/// guard runs here unconditionally: a built prep tree that disagrees with the
/// AIR's own root fails the prove with the same error the per-table prover
/// raises.
#[allow(clippy::too_many_arguments)]
pub fn multi_prove_batched<Field, FieldExtension, PI, P>(
    air_trace_pairs: Vec<BatchedAirTracePair<'_, Field, FieldExtension, PI>>,
    transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone + Send),
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
    P: IsStarkProver<Field, FieldExtension, PI> + ?Sized,
    <Field as IsField>::BaseType: math::spill_safe::SpillSafe,
    <FieldExtension as IsField>::BaseType: math::spill_safe::SpillSafe,
{
    multi_prove_batched_carved::<Field, FieldExtension, PI, P>(
        air_trace_pairs,
        transcript,
        #[cfg(feature = "disk-spill")]
        storage_mode,
        residency,
        None,
    )
}

/// As [`multi_prove_batched`], with one table's main matrix carved into a
/// standalone row-pair tree ([`crate::batched::shape::CarvedMain`]).
///
/// The carved tree is built by `commit_rows_bit_reversed_subset` over the FULL
/// committed-main range of the same LDE expansion — the identical call the
/// per-table prover makes for a non-preprocessed table — so the carved root is
/// byte-identical to the root a per-table prove of the same trace commits.
/// The root is absorbed after the preprocessed roots and before `main_root`,
/// ahead of every challenge draw.
#[allow(clippy::too_many_arguments)]
pub fn multi_prove_batched_carved<Field, FieldExtension, PI, P>(
    mut air_trace_pairs: Vec<BatchedAirTracePair<'_, Field, FieldExtension, PI>>,
    transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone + Send),
    #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
    residency: ResidencyMode,
    carved_main: Option<usize>,
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
    P: IsStarkProver<Field, FieldExtension, PI> + ?Sized,
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
    let (shape, params) = EpochShape::derive_carved(&airs, &trace_lengths, carved_main)?;
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
    let t_phase = std::time::Instant::now();
    // Both builders are fed from the SAME expansion: a preprocessed table's
    // precomputed columns and its multiplicity columns are two column ranges of
    // one row-major main LDE, exactly as `commit_main_trace` splits them.
    // ★ Per-table preprocessed trees — #768's arrangement, kept for the same
    // reason (see `BatchedQueryOpening::prep`): each preprocessed table keeps
    // its OWN row-pair tree, the one `air.precomputed_commitment()` pins, and
    // both sides absorb that root FROM THE AIR SET, never from the proof —
    // the per-table path's critical soundness check, verbatim. The trees are
    // process-cached by root, so continuation epochs stop re-committing the
    // execution-independent tables (DECODE, BITWISE, ...), exactly as the
    // per-table prover does.
    let mut prep_trees: Vec<PrepTreeSlot<BatchedMerkleTreeBackend<Field>>> =
        (0..num_tables).map(|_| None).collect();

    // When the GPU can commit the main round (Goldilocks + Retain, so the LDEs
    // are still around to feed the device tree), the keccak leaf+climb runs on
    // the device and only the digest layers come back; the CPU
    // `StreamingMmcsBuilder` is skipped entirely. Otherwise it builds the tree.
    // Deliberate mode select (no silent per-call fallback); a device error is a
    // hard abort.
    let use_device_main = device_commit_enabled::<Field>(residency);
    let mut main_builder =
        (!use_device_main).then(|| StreamingMmcsBuilder::<Field>::new(&shape.main.dims));
    // The device twin of `main_builder`: created before the loop, absorbs each
    // table's LDE inside it (streaming), climbed after. Present iff device main.
    #[cfg(feature = "cuda")]
    let mut device_main: Option<math_cuda::mmcs::StreamingMixedMmcs> = if use_device_main {
        let h_max = round_h_max(&shape.main, "main")?;
        Some(
            math_cuda::mmcs::StreamingMixedMmcs::new(h_max as u64).map_err(|e| {
                ProvingError::WrongParameter(format!("device main commit init failed: {e:?}"))
            })?,
        )
    } else {
        None
    };
    let mut retained_main: Vec<Option<(Vec<FieldElement<Field>>, usize)>> =
        (0..num_tables).map(|_| None).collect();
    // The carved table's standalone main tree and its root. Built inside the
    // phase-1 loop from the same expansion every other table commits from; the
    // root is absorbed AFTER the loop (after every preprocessed root) and
    // before `main_root`.
    let mut carved_tree: Option<MerkleTree<BatchedMerkleTreeBackend<Field>>> = None;
    let mut carved_root: Option<crate::config::Commitment> = None;

    for table in 0..num_tables {
        let (air, trace, _) = &air_trace_pairs[table];
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
        let is_carved = shape.carved_main.map(|c| c.table) == Some(table);
        let num_precomputed = if is_carved {
            // `derive_carved` rejects a preprocessed carved table, so the
            // carved matrix is the full main range.
            0
        } else {
            total_cols - matrix_width(&shape.main, table)
        };

        if num_precomputed > 0 {
            // The root every verifier will absorb is the AIR's own; building a
            // tree that disagrees with it is a stale constant or a wrong LDE,
            // and the per-table path's error is the honest name for both.
            let expected = air.precomputed_commitment();
            let tree =
                match crate::prover::precomputed_tree_cache_get::<Field>(&expected) {
                    Some(tree) => tree,
                    None => {
                        let (tree, root) = P::commit_rows_bit_reversed_subset::<Field>(
                            &main_data,
                            total_cols,
                            0,
                            num_precomputed,
                        )
                        .ok_or(ProvingError::PrecomputedCommitmentMismatch)?;
                        if root != expected {
                            return Err(ProvingError::PrecomputedCommitmentMismatch);
                        }
                        let tree = std::sync::Arc::new(tree);
                        crate::prover::precomputed_tree_cache_put(
                            expected,
                            std::sync::Arc::clone(&tree),
                        );
                        tree
                    }
                };
            transcript.append_bytes(&expected);
            prep_trees[table] = Some(tree);
        }
        if is_carved {
            // The carve: the identical committer call the per-table prover
            // makes for a non-preprocessed table (`commit_rows_bit_reversed` =
            // the subset call over the full range), on the identical
            // expansion — the root is byte-identical to the per-table tree's.
            let (tree, root) =
                P::commit_rows_bit_reversed_subset::<Field>(&main_data, total_cols, 0, total_cols)
                    .ok_or_else(|| {
                        ProvingError::WrongParameter(
                            "the carved table's main matrix has no committable rows".to_string(),
                        )
                    })?;
            carved_tree = Some(tree);
            carved_root = Some(root);
        } else {
            if let Some(builder) = main_builder.as_mut() {
                let src = vec![BorrowedMatrix::RowMajorNatural {
                    data: &main_data,
                    stride: total_cols,
                    col_start: num_precomputed,
                    width: total_cols - num_precomputed,
                    log_height: height,
                }];
                builder.absorb(&src, 0);
            }
            // Streaming device absorb: hash this table into the device tree NOW,
            // so the LDE can be freed below (RecomputeLde) without retaining it.
            //
            // Prefer the RESIDENT path: expand this table's main LDE on the GPU
            // and absorb it column-major straight from VRAM — no host FFT upload,
            // the whole point of a device-resident batched prover. Tables below
            // the device LDE threshold decline the keep-expand and fall back to
            // uploading the host LDE (mixed eligibility; identical root either
            // way). Step 2: only the main COMMIT is device-resident here; the host
            // `main_data` above still feeds prep trees, `retained_main` and the
            // later phases, so this deliberately expands twice for now (step 3
            // drops the host expand and moves the later phases to device recompute).
            #[cfg(feature = "cuda")]
            if let Some(dev) = device_main.as_mut() {
                let (trace_slice, num_cols) = trace.main_data_row_major();
                debug_assert_eq!(
                    num_cols, total_cols,
                    "the resident and host expansions must see the same column count"
                );
                let n = if num_cols > 0 {
                    trace_slice.len() / num_cols
                } else {
                    0
                };
                let resident = crate::gpu_lde::try_expand_row_major_keep_no_tree::<Field, Field>(
                    trace_slice,
                    trace.main_rowmajor_dev(),
                    n,
                    num_cols,
                    domains[table].blowup_factor,
                    &twiddles[table].coset_weights,
                    false,
                );
                match resident {
                    // Resident LDE: `handle.buf` is column-major (`col*lde_size +
                    // row`); absorb the committed columns [num_precomputed,
                    // total_cols) in place. `wait_ready_on` orders the absorb after
                    // the producer's last kernel device-side (no host block). No
                    // per-table tree is built (the shared MMCS is the tree);
                    // dropping `handle` frees this table's LDE (stream-first).
                    Some((handle, _host_lde)) => {
                        let commit_stream = dev.stream();
                        handle.wait_ready_on(&commit_stream).map_err(|e| {
                            ProvingError::WrongParameter(format!(
                                "device main LDE ready-wait failed: {e:?}"
                            ))
                        })?;
                        dev.absorb_col_major_dev(
                            height as u64,
                            handle.buf.as_ref(),
                            handle.lde_size as u64,
                            num_precomputed as u64,
                            total_cols as u64,
                        )
                        .map_err(|e| {
                            ProvingError::WrongParameter(format!(
                                "device main absorb (resident) failed: {e:?}"
                            ))
                        })?;
                    }
                    // Below the device LDE threshold: upload the host LDE and
                    // absorb row-major, exactly as before.
                    None => {
                        // SAFETY: Goldilocks base (device_commit_enabled gated it);
                        // FieldElement<Gl> is #[repr(transparent)] over u64.
                        let data_u64 = unsafe {
                            std::slice::from_raw_parts(
                                main_data.as_ptr() as *const u64,
                                main_data.len(),
                            )
                        };
                        dev.absorb_row_major(
                            height as u64,
                            data_u64,
                            total_cols as u64,
                            num_precomputed as u64,
                            total_cols as u64,
                        )
                        .map_err(|e| {
                            ProvingError::WrongParameter(format!("device main absorb failed: {e:?}"))
                        })?;
                    }
                }
            }
        }

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

    // The carved root's transcript slot: after every preprocessed root, before
    // `main_root` — so every challenge (LogUp, beta, z, gamma, alpha, iotas) is
    // drawn after it. Proof-carried on the verifier's side, absorbed here from
    // the tree just built.
    if let Some(root) = carved_root.as_ref() {
        transcript.append_bytes(root);
    }

    let main_mmcs = if use_device_main {
        #[cfg(feature = "cuda")]
        {
            finalize_device_main(
                device_main
                    .take()
                    .expect("device_main present under use_device_main"),
                &retained_main,
                &shape,
                num_tables,
                residency,
            )?
        }
        #[cfg(not(feature = "cuda"))]
        {
            unreachable!("device commit requires the cuda feature")
        }
    } else {
        main_builder
            .take()
            .expect("CPU main builder present when device commit is disabled")
            .finish()
    };
    let main_root = main_mmcs.root();
    transcript.append_bytes(&main_root);

    // =====================================================================
    // Phase 2 — LogUp challenges, then the auxiliary round
    // =====================================================================
    stats.phase_wall[0] = t_phase.elapsed();
    let t_phase = std::time::Instant::now();
    let needs_lookup = airs.iter().any(|air| air.has_aux_trace());
    let lookup_challenges: Vec<FieldElement<FieldExtension>> = if needs_lookup {
        (0..LOGUP_NUM_CHALLENGES)
            .map(|_| transcript.sample_field_element())
            .collect()
    } else {
        Vec::new()
    };

    // The aux round expands its LDE from the host trace columns
    // (`expand_aux_lde_row_major` below); the device-resident aux build
    // returns the columns device-side only and leaves the host trace
    // unwritten, so it is disabled for every table here — the same switch
    // the per-table prover throws under disk-spill and `RecomputeLde`.
    #[cfg(feature = "cuda")]
    for (_, trace, _) in air_trace_pairs.iter_mut() {
        trace.set_resident_aux_ok(false);
    }

    let mut bus_public_inputs: Vec<Option<BusPublicInputs<FieldExtension>>> =
        (0..num_tables).map(|_| None).collect();
    let use_device_aux = device_commit_enabled::<FieldExtension>(residency);
    let mut aux_builder = (!shape.aux.is_empty() && !use_device_aux)
        .then(|| StreamingMmcsBuilder::<FieldExtension>::new(&shape.aux.dims));
    #[cfg(feature = "cuda")]
    let mut device_aux: Option<math_cuda::mmcs::StreamingMixedMmcs> =
        if use_device_aux && !shape.aux.is_empty() {
            let h_max = round_h_max(&shape.aux, "aux")?;
            Some(
                math_cuda::mmcs::StreamingMixedMmcs::new(h_max as u64).map_err(|e| {
                    ProvingError::WrongParameter(format!("device aux commit init failed: {e:?}"))
                })?,
            )
        } else {
            None
        };
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

        if shape.aux.is_empty() {
            continue;
        }
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
        if let Some(builder) = aux_builder.as_mut() {
            let src = vec![BorrowedMatrix::RowMajorNatural {
                data: &aux_data,
                stride: aux_cols,
                col_start: 0,
                width: aux_cols,
                log_height: shape.heights[table],
            }];
            builder.absorb(&src, 0);
        }
        // Streaming device absorb: expand this table's aux LDE on the GPU and
        // absorb it (slab layout) straight from VRAM — no host FFT upload. The
        // resident slab buffer yields the byte-identical aux leaf as the host
        // row-major absorb (both emit, per bit-reversed row, each column's 3
        // components consecutively). Tables below the device LDE threshold — and
        // disk-spilled traces (the natural aux trace lives on disk, the resident
        // expand needs it in memory) — fall back to uploading the host LDE
        // row-major (mixed eligibility, identical root). Like main, this is
        // step-2-shaped: the host `expand_aux_lde_row_major` above still feeds the
        // later phases; step 3 moves those to device recompute.
        #[cfg(feature = "cuda")]
        if let Some(dev) = device_aux.as_mut() {
            #[cfg(feature = "disk-spill")]
            let aux_resident_ok = storage_mode != StorageMode::Disk;
            #[cfg(not(feature = "disk-spill"))]
            let aux_resident_ok = true;

            let resident = if aux_resident_ok {
                let (trace_slice, num_cols) = trace.aux_data_row_major();
                debug_assert_eq!(
                    num_cols, aux_cols,
                    "the resident and host aux expansions must see the same column count"
                );
                let n = if num_cols > 0 {
                    trace_slice.len() / num_cols
                } else {
                    0
                };
                crate::gpu_lde::try_expand_ext3_row_major_keep_no_tree::<Field, FieldExtension>(
                    trace_slice,
                    n,
                    num_cols,
                    domains[table].blowup_factor,
                    &twiddles[table].coset_weights,
                    false,
                )
            } else {
                None
            };

            match resident {
                // Resident slab LDE: absorb columns [0, aux_cols) in place;
                // `wait_ready_on` orders the absorb after the producer device-side.
                Some((handle, _host_lde)) => {
                    let commit_stream = dev.stream();
                    handle.wait_ready_on(&commit_stream).map_err(|e| {
                        ProvingError::WrongParameter(format!(
                            "device aux LDE ready-wait failed: {e:?}"
                        ))
                    })?;
                    dev.absorb_ext3_slabs_dev(
                        shape.heights[table] as u64,
                        handle.buf.as_ref(),
                        handle.lde_size as u64,
                        aux_cols as u64,
                    )
                    .map_err(|e| {
                        ProvingError::WrongParameter(format!(
                            "device aux absorb (resident) failed: {e:?}"
                        ))
                    })?;
                }
                // Below threshold or disk-spilled: upload the host LDE row-major.
                None => {
                    // SAFETY: ext3 Goldilocks (device_commit_enabled gated it); an
                    // element is 3 consecutive u64.
                    let data_u64 = unsafe {
                        std::slice::from_raw_parts(
                            aux_data.as_ptr() as *const u64,
                            aux_data.len() * 3,
                        )
                    };
                    dev.absorb_ext3_row_major(
                        shape.heights[table] as u64,
                        data_u64,
                        aux_cols as u64,
                        0,
                        aux_cols as u64,
                    )
                    .map_err(|e| {
                        ProvingError::WrongParameter(format!("device aux absorb failed: {e:?}"))
                    })?;
                }
            }
        }
        match residency {
            ResidencyMode::Retain => retained_aux[table] = Some((aux_data, aux_cols)),
            ResidencyMode::RecomputeLde => {
                drop(aux_data);
                ledger.free(bytes);
            }
        }
    }

    let aux_mmcs = if use_device_aux && !shape.aux.is_empty() {
        #[cfg(feature = "cuda")]
        {
            Some(finalize_device_aux(
                device_aux.take().expect("device_aux present under use_device_aux"),
                &retained_aux,
                &shape,
                num_tables,
                residency,
            )?)
        }
        #[cfg(not(feature = "cuda"))]
        {
            unreachable!("device commit requires the cuda feature")
        }
    } else {
        aux_builder.map(StreamingMmcsBuilder::finish)
    };
    let aux_root = aux_mmcs.as_ref().map(MixedMmcs::root);
    if let Some(root) = aux_root {
        transcript.append_bytes(&root);
    }

    // =====================================================================
    // Phase 3 — bus contributions, beta per table, the composition-parts round
    // =====================================================================
    stats.phase_wall[1] = t_phase.elapsed();
    let t_phase = std::time::Instant::now();
    for bpi in bus_public_inputs.iter().flatten() {
        transcript.append_field_element(&bpi.table_contribution);
    }

    let use_device_parts = device_commit_enabled::<FieldExtension>(residency);
    let mut parts_builder = (!use_device_parts)
        .then(|| StreamingMmcsBuilder::<FieldExtension>::new(&shape.parts.dims));
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

        let ldes = materialize_ldes::<Field, FieldExtension, PI, P>(
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
            // R2: device-only recompute — constraint eval reads the resident LDE,
            // only the small composition parts come back to host (below).
            true,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        );
        let (mut lde_trace, carried_bytes) =
            lde_trace_take(ldes, air.step_size(), domain.blowup_factor);

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
        #[allow(unused_mut)]
        let mut parts = computed.parts;
        // Device-only R2: the constraint eval ran on the resident LDE and the
        // parts live in the handle (empty host placeholders). Download them —
        // small (2 composition-poly columns, not the full LDE) — for the host
        // parts commit + R3 coset-eval + R4 DEEP. No-op on the host path (parts
        // already populated) or any table whose device R2 declined (recovery
        // filled the host parts).
        #[cfg(feature = "cuda")]
        if parts.first().is_some_and(|p| p.is_empty()) {
            let handle = computed.gpu_parts.as_ref().expect(
                "device-only R2 must carry a resident parts handle when host parts are empty",
            );
            let stream = math_cuda::device::backend()
                .map_err(|e| {
                    ProvingError::WrongParameter(format!("backend for parts download: {e:?}"))
                })?
                .next_stream();
            parts = crate::gpu_lde::download_composition_parts_host::<FieldExtension>(
                handle, &stream,
            )
            .ok_or_else(|| {
                ProvingError::WrongParameter("device-only R2 parts download failed".to_string())
            })?;
        }

        let parts_bytes: usize = parts
            .iter()
            .map(|p| lde_bytes::<FieldExtension>(p.len()))
            .sum();
        parts_ledger.alloc(parts_bytes);
        if let Some(builder) = parts_builder.as_mut() {
            let src = vec![BorrowedMatrix::ColMajorNatural {
                cols: &parts,
                log_height: shape.heights[table],
            }];
            builder.absorb(&src, 0);
        }

        // Parts are RETAINED: rebuilding them is a second constraint evaluation.
        retained_parts[table] = parts;
        release_ldes(
            ldes_from_trace(lde_trace, carried_bytes),
            &mut retained_main,
            &mut retained_aux,
            table,
            &mut ledger,
            residency,
        );
    }

    let parts_mmcs = if use_device_parts {
        commit_parts_device(&retained_parts, &shape, num_tables)?
    } else {
        parts_builder
            .take()
            .expect("CPU parts builder present when device commit is disabled")
            .finish()
    };
    let parts_root = parts_mmcs.root();
    transcript.append_bytes(&parts_root);

    // =====================================================================
    // Phase 4 — z per table, OOD evaluations
    // =====================================================================
    stats.phase_wall[2] = t_phase.elapsed();
    let t_phase = std::time::Instant::now();
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

        // Phase 4 reads the trace ONLY at stride `blowup` — the size-`n`
        // coset evaluation. Under `Retain` the full LDE is already on hand
        // and the strided read is free; under `RecomputeLde` a full 4n
        // expansion here would be paid just to subsample it, so the
        // recompute arm materializes the n-sized evaluation directly
        // (bit-identical values, ~37% of the work, a quarter of the bytes)
        // and hands `round_3` a blowup-1 table, whose OWN stride the trace
        // reads follow.
        let round3 = if retained_main[table].is_some() {
            let ldes = materialize_ldes::<Field, FieldExtension, PI, P>(
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
                // R3 runs only under Retain here (RecomputeLde uses the coset-eval
                // branch below); the retained host LDE is used, so not device-only.
                false,
                #[cfg(feature = "disk-spill")]
                storage_mode,
            );
            let (mut lde_trace, carried_bytes) =
                lde_trace_take(ldes, air.step_size(), domain.blowup_factor);
            let round3 = P::round_3_evaluate_polynomials_in_out_of_domain_element(
                *air,
                domain,
                &mut lde_trace,
                &mut retained_parts[table],
                &z,
            );
            release_ldes(
                ldes_from_trace(lde_trace, carried_bytes),
                &mut retained_main,
                &mut retained_aux,
                table,
                &mut ledger,
                residency,
            );
            round3
        } else {
            let (_, trace, _) = &air_trace_pairs[table];
            let t_expand = std::time::Instant::now();
            let main = P::expand_main_coset_eval_row_major(trace, domain, &twiddles[table]);
            let aux = if matrix_index(&shape.aux, table).is_some() {
                let aux = P::expand_aux_coset_eval_row_major(trace, domain, &twiddles[table]);
                stats.aux_coset_evals += 1;
                aux
            } else {
                (Vec::new(), 0)
            };
            stats.lde_expansion_wall += t_expand.elapsed();
            stats.main_coset_evals += 1;
            let bytes = lde_bytes::<Field>(main.0.len()) + lde_bytes::<FieldExtension>(aux.0.len());
            ledger.alloc(bytes);
            let mut lde_trace =
                LDETraceTable::from_row_major(main.0, main.1, aux.0, aux.1, air.step_size(), 1);
            let round3 = P::round_3_evaluate_polynomials_in_out_of_domain_element(
                *air,
                domain,
                &mut lde_trace,
                &mut retained_parts[table],
                &z,
            );
            drop(lde_trace);
            ledger.free(bytes);
            round3
        };

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
    stats.phase_wall[3] = t_phase.elapsed();
    let t_phase = std::time::Instant::now();
    let gammas: Vec<FieldElement<FieldExtension>> = (0..num_tables)
        .map(|_| transcript.sample_field_element())
        .collect();

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
        let coset_offset_ref = &coset_offset;

        commit_batched_fri::<Field, FieldExtension, _, _>(
            transcript,
            &shape.heights,
            &shape.total_widths(),
            move |alpha, plan| {
                // The standalone class's terminal polynomials, handed back so
                // `commit_batched_fri` binds them into the transcript and the
                // wire carries the very coefficients that were bound.
                let mut standalone_coeffs: Vec<Option<Vec<FieldElement<FieldExtension>>>> =
                    (0..num_tables).map(|_| None).collect();
                let mut combiner = HeightCombiner::new(*alpha);
                // Ascending table order, which is also `plan.batched`'s order —
                // absorption order is what defines the alpha powers, so the two
                // must not be allowed to drift apart.
                for table in 0..num_tables {
                    let (air, _, _) = &air_trace_pairs[table];
                    let domain = &domains[table];
                    let ldes = materialize_ldes::<Field, FieldExtension, PI, P>(
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
                        // R4 DEEP: device-only recompute — reads the resident LDE
                        // + host parts (downloaded in R2); no full-LDE D2H.
                        true,
                        #[cfg(feature = "disk-spill")]
                        storage_mode,
                    );
                    let (mut lde_trace, carried_bytes) =
                        lde_trace_take(ldes, air.step_size(), domain.blowup_factor);
                    let mut deep = deep_codeword::<Field, FieldExtension, PI, P>(
                        *air,
                        domain,
                        &mut lde_trace,
                        &mut retained_parts[table],
                        &round3s[table],
                        &zs[table],
                        &gammas[table],
                    );
                    release_ldes(
                        ldes_from_trace(lde_trace, carried_bytes),
                        retained_main,
                        retained_aux,
                        table,
                        ledger,
                        residency,
                    );
                    // Row-major variant at one column = the parallel path; the
                    // serial swap loop was pure wall time, 27 times per epoch.
                    in_place_bit_reverse_permute_row_major(&mut deep, 1);

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
                (combiner.finish(), standalone_coeffs)
            },
            &coset_offset,
            params.blowup_log,
            params.final_poly_log_degree,
            params.grinding_factor,
            params.num_queries,
        )?
    };

    // =====================================================================
    // Phase 6 — openings, one table at a time
    // =====================================================================
    stats.phase_wall[4] = t_phase.elapsed();
    let t_phase = std::time::Instant::now();
    let iotas = commit.iotas.clone();
    let fri_decommitments = crate::fri::query_phase::<FieldExtension>(&commit.layers, &iotas);

    // Per-query, per-prep-table standard openings (prep-table order =
    // `shape.prep.tables`, which is AIR order).
    let mut prep_openings: Vec<Vec<crate::proof::stark::PolynomialOpenings<Field>>> =
        (0..iotas.len()).map(|_| Vec::new()).collect();
    let mut main_openings = empty_openings::<Field>(&iotas, shape.main.tables.len());
    let mut aux_openings = empty_openings::<FieldExtension>(&iotas, shape.aux.tables.len());
    let mut parts_openings = empty_openings::<FieldExtension>(&iotas, shape.parts.tables.len());
    let mut carved_openings: Vec<Option<crate::proof::stark::PolynomialOpenings<Field>>> =
        (0..iotas.len()).map(|_| None).collect();

    // ★ Each round is read in ITS OWN index space, and the reduction happens
    // exactly once, here. Doing it inside the read would be wrong twice over: a
    // round shorter than the FRI would be asked for a leaf it does not have
    // (the prep round's `h_max` is below the FRI's whenever the tallest
    // preprocessed table is not the tallest table), and a round that reduced
    // again on the way out would land somewhere else entirely.
    let main_iotas = reduced_iotas(&iotas, h_max, main_mmcs.h_max());
    let aux_iotas = aux_mmcs
        .as_ref()
        .map(|mmcs| reduced_iotas(&iotas, h_max, mmcs.h_max()));
    let parts_iotas = reduced_iotas(&iotas, h_max, parts_mmcs.h_max());

    for table in 0..num_tables {
        let (air, trace, _) = &air_trace_pairs[table];
        #[cfg(not(feature = "cuda"))]
        let _ = &trace;
        // Plain tables (no precomputed columns, not carved) gather their openings
        // straight off the resident LDE — no full-LDE download. Precomputed and
        // carved tables keep the host LDE (their per-table trees read it) but
        // still gather the shared-MMCS main/aux openings from the resident handle.
        let is_carved = shape.carved_main.map(|c| c.table) == Some(table);
        #[cfg(feature = "cuda")]
        let openings_device_only =
            !is_carved && trace.main_data_row_major().1 == matrix_width(&shape.main, table);
        #[cfg(not(feature = "cuda"))]
        let openings_device_only = false;
        let ldes = materialize_ldes::<Field, FieldExtension, PI, P>(
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
            openings_device_only,
            #[cfg(feature = "disk-spill")]
            storage_mode,
        );
        let _ = air;
        let height = shape.heights[table];
        let (main_data, total_cols) = &ldes.main;
        let num_precomputed = if is_carved {
            0
        } else {
            total_cols - matrix_width(&shape.main, table)
        };

        if is_carved {
            let tree = carved_tree
                .as_ref()
                .expect("the carved tree was built in phase 1");
            // The carved tree lives in the TABLE's own index space, exactly
            // like a preprocessed tree: reduce the shared FRI index once.
            let table_iotas = reduced_iotas(&iotas, h_max, height);
            for (q, &idx) in table_iotas.iter().enumerate() {
                carved_openings[q] = Some(P::open_polys_with(&domains[table], tree, idx, |row| {
                    main_data[row * total_cols..(row + 1) * total_cols].to_vec()
                }));
            }
        }

        if let Some(tree) = prep_trees[table].as_ref() {
            // The per-table tree lives in the TABLE's own index space; reduce
            // the shared FRI index by the height difference once, here.
            let table_iotas = reduced_iotas(&iotas, h_max, height);
            for (q, &idx) in table_iotas.iter().enumerate() {
                prep_openings[q].push(P::open_polys_with(&domains[table], tree, idx, |row| {
                    main_data[row * total_cols..row * total_cols + num_precomputed].to_vec()
                }));
            }
        }
        if let Some(m) = matrix_index(&shape.main, table) {
            #[cfg(feature = "cuda")]
            {
                // Gather this table's query row-pairs off the resident LDE handle
                // (a small D2H of only the queried rows) instead of downloading
                // the whole LDE. `None` when no handle (sub-threshold decline) →
                // the host LDE (present in that case) serves the openings.
                let gathered = ldes.gpu_main.as_ref().and_then(|h| {
                    let rows = query_row_indices(&main_mmcs, m, &main_iotas, h.lde_size);
                    let stream = math_cuda::device::backend().ok()?.next_stream();
                    let raw =
                        math_cuda::barycentric::gather_rows_base_on_device(h, &rows, &stream).ok()?;
                    crate::constraint_ir::gpu_interp::base_u64_to_field::<Field>(&raw)
                        .map(|v| (v, h.m))
                });
                match gathered {
                    Some((g, ncols)) => fill_openings_from_gathered(
                        &main_mmcs,
                        m,
                        &g,
                        ncols,
                        num_precomputed,
                        *total_cols,
                        &main_iotas,
                        &mut main_openings,
                    ),
                    None => {
                        let src = vec![BorrowedMatrix::RowMajorNatural {
                            data: main_data,
                            stride: *total_cols,
                            col_start: num_precomputed,
                            width: total_cols - num_precomputed,
                            log_height: height,
                        }];
                        fill_openings(&main_mmcs, m, &src, &main_iotas, &mut main_openings);
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let src = vec![BorrowedMatrix::RowMajorNatural {
                    data: main_data,
                    stride: *total_cols,
                    col_start: num_precomputed,
                    width: total_cols - num_precomputed,
                    log_height: height,
                }];
                fill_openings(&main_mmcs, m, &src, &main_iotas, &mut main_openings);
            }
        }
        if let (Some(mmcs), Some(m)) = (aux_mmcs.as_ref(), matrix_index(&shape.aux, table)) {
            let indices = aux_iotas.as_ref().expect("the aux MMCS exists here");
            #[cfg(feature = "cuda")]
            {
                let gathered = ldes.gpu_aux.as_ref().and_then(|h| {
                    let rows = query_row_indices(mmcs, m, indices, h.lde_size);
                    let stream = math_cuda::device::backend().ok()?.next_stream();
                    let raw =
                        math_cuda::barycentric::gather_rows_ext3_on_device(h, &rows, &stream).ok()?;
                    crate::constraint_ir::gpu_interp::ext3_u64_to_field::<FieldExtension>(&raw)
                        .map(|v| (v, h.m))
                });
                match gathered {
                    Some((g, ncols)) => fill_openings_from_gathered(
                        mmcs,
                        m,
                        &g,
                        ncols,
                        0,
                        ncols,
                        indices,
                        &mut aux_openings,
                    ),
                    None => {
                        let (aux_data, aux_cols) = &ldes.aux;
                        let src = vec![BorrowedMatrix::RowMajorNatural {
                            data: aux_data,
                            stride: *aux_cols,
                            col_start: 0,
                            width: *aux_cols,
                            log_height: height,
                        }];
                        fill_openings(mmcs, m, &src, indices, &mut aux_openings);
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let (aux_data, aux_cols) = &ldes.aux;
                let src = vec![BorrowedMatrix::RowMajorNatural {
                    data: aux_data,
                    stride: *aux_cols,
                    col_start: 0,
                    width: *aux_cols,
                    log_height: height,
                }];
                fill_openings(mmcs, m, &src, indices, &mut aux_openings);
            }
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
            prep: std::mem::take(&mut prep_openings[q]),
            main: assemble(&main_mmcs, main_iotas[q], &mut main_openings, q)
                .expect("the main round was opened at these very indices"),
            aux: aux_mmcs.as_ref().map(|mmcs| {
                let indices = aux_iotas.as_ref().expect("the aux MMCS exists here");
                assemble(mmcs, indices[q], &mut aux_openings, q)
                    .expect("the aux round was opened at these very indices")
            }),
            parts: assemble(&parts_mmcs, parts_iotas[q], &mut parts_openings, q)
                .expect("the parts round was opened at these very indices"),
            carved_main: carved_openings[q].take(),
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
                standalone_final_poly_coeffs: commit.standalone_coeffs[table].clone(),
            }
        })
        .collect();

    stats.phase_wall[5] = t_phase.elapsed();
    stats.peak_trace_lde_bytes = ledger.peak();
    stats.retained_parts_bytes = parts_ledger.peak();
    stats.peak_lde_bytes = stats.peak_trace_lde_bytes + stats.retained_parts_bytes;

    Ok((
        BatchedMultiProof {
            tables,
            main_root,
            carved_main_root: carved_root,
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

/// Whether a round is committed on the GPU (device tree authoritative) rather
/// than by the host `StreamingMmcsBuilder`. True with the `cuda` feature on a
/// Goldilocks field (base or ext3 — the only fields the kernels support), under
/// EITHER residency: the streaming device commit
/// ([`math_cuda::mmcs::StreamingMixedMmcs`]) absorbs each table's LDE inside the
/// round loop and frees it, keeping only the per-leaf sponges resident, so it no
/// longer needs `Retain` (which retained every LDE just to feed one all-at-once
/// commit — the O(N) memory that OOM'd a large epoch). A deliberate mode select;
/// there is no silent per-call host↔device fallback — a device error aborts.
#[cfg(feature = "cuda")]
fn device_commit_enabled<F: 'static>(_residency: ResidencyMode) -> bool {
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
    use math::field::goldilocks::GoldilocksField;
    use std::any::TypeId;
    TypeId::of::<F>() == TypeId::of::<GoldilocksField>()
        || TypeId::of::<F>() == TypeId::of::<Degree3GoldilocksExtensionField>()
}

#[cfg(not(feature = "cuda"))]
fn device_commit_enabled<F: 'static>(_residency: ResidencyMode) -> bool {
    false
}

/// Finalize the main round's STREAMING device commit: climb the tree from the
/// per-leaf sponges the round loop already absorbed into `device` (one table at
/// a time, then freed — see [`math_cuda::mmcs::StreamingMixedMmcs`]), rebuild the
/// host-side `MixedMmcs` from the heap, and — when the LDEs are still resident
/// (`Retain`) — run the release canary. Only reached under
/// [`device_commit_enabled`]; a device error is a hard abort, never a silent
/// host fallback.
#[cfg(feature = "cuda")]
fn finalize_device_main<Field: IsField + 'static>(
    device: math_cuda::mmcs::StreamingMixedMmcs,
    retained_main: &[Option<(Vec<FieldElement<Field>>, usize)>],
    shape: &EpochShape,
    num_tables: usize,
    residency: ResidencyMode,
) -> Result<MixedMmcs<Field>, ProvingError>
where
    FieldElement<Field>: math::traits::AsBytes + Sync + Send,
{
    let nodes = device
        .finish()
        .map_err(|e| ProvingError::WrongParameter(format!("device main commit failed: {e:?}")))?;
    let h_max = shape
        .main
        .dims
        .iter()
        .map(|&(lh, _)| lh)
        .max()
        .ok_or_else(|| ProvingError::WrongParameter("device main commit: empty round".to_string()))?;
    let mmcs = MixedMmcs::from_heap_nodes(shape.main.dims.clone(), h_max, &nodes);
    // The canary reads the retained LDEs back through a source in the SAME order
    // as the absorbed inputs; under `RecomputeLde` they are gone (the debug
    // parity tests cover that path).
    if residency == ResidencyMode::Retain {
        let mut source: Vec<BorrowedMatrix<Field>> = Vec::new();
        for table in 0..num_tables {
            if shape.carved_main.map(|c| c.table) == Some(table) {
                continue;
            }
            match &retained_main[table] {
                Some((data, total_cols)) => {
                    let width = matrix_width(&shape.main, table);
                    source.push(BorrowedMatrix::RowMajorNatural {
                        data,
                        stride: *total_cols,
                        col_start: *total_cols - width,
                        width,
                        log_height: shape.heights[table],
                    });
                }
                None => {
                    return Err(ProvingError::WrongParameter(
                        "device main commit canary: a non-carved main LDE was not retained"
                            .to_string(),
                    ));
                }
            }
        }
        xcheck_device_commit("main", &mmcs, source)?;
    }
    Ok(mmcs)
}

/// Finalize the aux round's STREAMING device commit (ext3 row-major). Aux twin
/// of [`finalize_device_main`]: the round loop absorbed each aux LDE into
/// `device`; here we climb + rebuild + (under `Retain`) canary. The aux round
/// commits every aux column, so `col_start = 0`.
#[cfg(feature = "cuda")]
fn finalize_device_aux<E: IsField + 'static>(
    device: math_cuda::mmcs::StreamingMixedMmcs,
    retained_aux: &[Option<(Vec<FieldElement<E>>, usize)>],
    shape: &EpochShape,
    num_tables: usize,
    residency: ResidencyMode,
) -> Result<MixedMmcs<E>, ProvingError>
where
    FieldElement<E>: math::traits::AsBytes + Sync + Send,
{
    let nodes = device
        .finish()
        .map_err(|e| ProvingError::WrongParameter(format!("device aux commit failed: {e:?}")))?;
    let h_max = round_h_max(&shape.aux, "aux")?;
    let mmcs = MixedMmcs::from_heap_nodes(shape.aux.dims.clone(), h_max, &nodes);
    if residency == ResidencyMode::Retain {
        let mut source: Vec<BorrowedMatrix<E>> = Vec::new();
        for table in 0..num_tables {
            if let Some((data, aux_cols)) = &retained_aux[table] {
                source.push(BorrowedMatrix::RowMajorNatural {
                    data,
                    stride: *aux_cols,
                    col_start: 0,
                    width: *aux_cols,
                    log_height: shape.heights[table],
                });
            }
        }
        xcheck_device_commit("aux", &mmcs, source)?;
    }
    Ok(mmcs)
}

/// Commit the parts round (column-major ext3 slabs) on the GPU via the streaming
/// [`math_cuda::mmcs::StreamingMixedMmcs`]: build one table's slab, absorb it,
/// free it, then the next — only one slab resident at a time instead of every
/// table's slab at once. Parts are ALWAYS retained (recomputing them re-runs
/// constraint eval, a prove's dominant cost), so — unlike main/aux — this reads
/// them post-loop and always runs the canary.
#[cfg(feature = "cuda")]
fn commit_parts_device<E: IsField + 'static>(
    retained_parts: &[Vec<Vec<FieldElement<E>>>],
    shape: &EpochShape,
    num_tables: usize,
) -> Result<MixedMmcs<E>, ProvingError>
where
    FieldElement<E>: math::traits::AsBytes + Sync + Send,
{
    let h_max = round_h_max(&shape.parts, "parts")?;
    let mut device = math_cuda::mmcs::StreamingMixedMmcs::new(h_max as u64).map_err(|e| {
        ProvingError::WrongParameter(format!("device parts commit init failed: {e:?}"))
    })?;
    for table in 0..num_tables {
        let parts = &retained_parts[table];
        if parts.is_empty() || parts[0].is_empty() {
            return Err(ProvingError::WrongParameter(
                "device parts commit: a table has no retained parts".to_string(),
            ));
        }
        let num_parts = parts.len();
        let num_rows = parts[0].len();
        let mut slab = vec![0u64; num_parts * 3 * num_rows];
        for (p, col) in parts.iter().enumerate() {
            for (row, elem) in col.iter().enumerate() {
                // SAFETY: ext3 Goldilocks (gated), `[u64; 3]` per element.
                let comps =
                    unsafe { std::slice::from_raw_parts(elem.value() as *const _ as *const u64, 3) };
                for (k, &c) in comps.iter().enumerate() {
                    slab[(p * 3 + k) * num_rows + row] = c;
                }
            }
        }
        device
            .absorb_ext3_slabs(
                shape.heights[table] as u64,
                &slab,
                num_rows as u64,
                num_parts as u64,
            )
            .map_err(|e| {
                ProvingError::WrongParameter(format!("device parts absorb failed: {e:?}"))
            })?;
        // `slab` is freed here — one table's slab resident at a time.
    }
    let nodes = device
        .finish()
        .map_err(|e| ProvingError::WrongParameter(format!("device parts commit failed: {e:?}")))?;
    let mmcs = MixedMmcs::from_heap_nodes(shape.parts.dims.clone(), h_max, &nodes);
    // Parts are always retained, so the canary source is always available.
    let mut source: Vec<BorrowedMatrix<E>> = Vec::new();
    for table in 0..num_tables {
        source.push(BorrowedMatrix::ColMajorNatural {
            cols: &retained_parts[table],
            log_height: shape.heights[table],
        });
    }
    xcheck_device_commit("parts", &mmcs, source)?;
    Ok(mmcs)
}

#[cfg(not(feature = "cuda"))]
fn commit_parts_device<E: IsField + 'static>(
    _retained_parts: &[Vec<Vec<FieldElement<E>>>],
    _shape: &EpochShape,
    _num_tables: usize,
) -> Result<MixedMmcs<E>, ProvingError>
where
    FieldElement<E>: math::traits::AsBytes + Sync + Send,
{
    unreachable!("device commit is disabled without the cuda feature")
}

/// How many leaves the [`xcheck_device_commit`] canary re-authenticates per
/// round. `k * (h_max - 1)` compressions and one row-pair hash per matrix — a
/// rounding error next to the `O(N)` device tree it guards.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const CANARY_SAMPLES: usize = 8;

/// Up to [`CANARY_SAMPLES`] DISTINCT leaf indices in `[0, n0)`, walked from a
/// root-seeded base by a root-seeded odd stride. The stride is coprime to the
/// power-of-two `n0`, so the first `k <= n0` steps never collide; seeding both
/// from the device root spreads the probes with the committed data (rather than
/// always testing leaf 0) without touching the transcript.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn canary_indices(root: &crate::config::Commitment, n0: usize) -> Vec<usize> {
    debug_assert!(n0 >= 1);
    let k = CANARY_SAMPLES.min(n0);
    let seed = u64::from_le_bytes(root[0..8].try_into().expect("a commitment is 32 bytes"));
    let base = (seed % n0 as u64) as usize;
    // `| 1` forces an odd stride; odd is coprime to any power of two.
    let stride = (u64::from_le_bytes(root[8..16].try_into().expect("a commitment is 32 bytes")) | 1)
        as usize;
    (0..k)
        .map(|s| base.wrapping_add(s.wrapping_mul(stride)) % n0)
        .collect()
}

/// Release-safe canary for a device MMCS commit. The device path is
/// AUTHORITATIVE — the host [`StreamingMmcsBuilder`] is skipped — and the
/// byte-identical device-vs-host parity that guards 3d/3e is `debug_assert!` /
/// test-only, i.e. compiled OUT of a release prove. This re-authenticates a
/// handful of the freshly committed tree's leaves against the retained LDE rows
/// on the host, through the very [`MixedMmcs::open_batch`] /
/// [`MixedMmcs::verify_batch`] the verifier trusts. A silent device-side
/// corruption — a layout / heap bug that ships a wrong-but-self-consistent tree
/// — then aborts the prove with a hard [`ProvingError`] instead of producing a
/// proof that authenticates the wrong data. `source` must describe the SAME
/// matrices, in the SAME order, as the ones fed to the device commit (the
/// callers build it in the very loop that assembles the device inputs).
///
/// Compiled unconditionally (only CALLED under cuda) so the host-only negative
/// test can exercise it without a GPU.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn xcheck_device_commit<E>(
    round: &str,
    mmcs: &MixedMmcs<E>,
    source: Vec<BorrowedMatrix<'_, E>>,
) -> Result<(), ProvingError>
where
    E: IsField + 'static,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let dims = mmcs.dims();
    if source.len() != dims.len() {
        return Err(ProvingError::WrongParameter(format!(
            "device {round} commit canary: leaf source has {} matrices but the tree committed {}",
            source.len(),
            dims.len()
        )));
    }
    let heights: Vec<usize> = dims.iter().map(|&(h, _)| h).collect();
    let widths: Vec<usize> = dims.iter().map(|&(_, w)| w).collect();
    let root = mmcs.root();
    let n0 = 1usize << (mmcs.h_max() - 1);

    for iota in canary_indices(&root, n0) {
        let opening = mmcs.open_batch(iota, &source);
        if !MixedMmcs::verify_batch(&root, iota, &opening, &heights, &widths) {
            return Err(ProvingError::WrongParameter(format!(
                "device {round} commit canary FAILED at leaf {iota}: the committed device \
                 tree does not re-authenticate against the retained LDE — a silent device-side \
                 corruption, aborting the prove (NO host fallback)"
            )));
        }
    }
    Ok(())
}

/// The tallest committed `log_height` in a round — the base layer of its device
/// tree. Errs on an empty round (no matrix to commit).
#[cfg(feature = "cuda")]
fn round_h_max(round: &RoundShape, name: &str) -> Result<usize, ProvingError> {
    round
        .dims
        .iter()
        .map(|&(lh, _)| lh)
        .max()
        .ok_or_else(|| ProvingError::WrongParameter(format!("device {name} commit: empty round")))
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
fn fill_openings<E, S>(
    mmcs: &MixedMmcs<E>,
    matrix: usize,
    source: &S,
    iotas: &[usize],
    out: &mut [Vec<Option<PolynomialOpenings<E>>>],
) where
    E: IsField + 'static,
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

/// Device counterpart of [`fill_openings`]: the query row-pairs were already
/// gathered off the resident LDE (`[even(q0), odd(q0), even(q1), odd(q1), ...]`,
/// each `ncols` field elements in FULL-row order); slice columns
/// `[col_start, col_end)` per query. Byte-identical output to `fill_openings`
/// (empty merkle path, filled by `assemble`), because the gathered rows are the
/// same rows `append_row` would have read from the host LDE — no full-LDE D2H.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn fill_openings_from_gathered<E>(
    mmcs: &MixedMmcs<E>,
    matrix: usize,
    gathered: &[FieldElement<E>],
    ncols: usize,
    col_start: usize,
    col_end: usize,
    iotas: &[usize],
    out: &mut [Vec<Option<PolynomialOpenings<E>>>],
) where
    E: IsField + 'static,
    FieldElement<E>: AsBytes + Sync + Send,
{
    for (q, &iota) in iotas.iter().enumerate() {
        // `row_pair_leaf` returning None means this round is shorter than the FRI
        // and does not have the leaf; gathered rows for such queries are the row-0
        // placeholder and skipped here (same as `fill_openings`).
        if mmcs.row_pair_leaf(iota, matrix).is_none() {
            continue;
        }
        let even = gathered[(2 * q) * ncols + col_start..(2 * q) * ncols + col_end].to_vec();
        let odd =
            gathered[(2 * q + 1) * ncols + col_start..(2 * q + 1) * ncols + col_end].to_vec();
        out[q][matrix] = Some(PolynomialOpenings {
            proof: crypto::merkle_tree::proof::Proof {
                merkle_path: Vec::new(),
            },
            evaluations: even,
            evaluations_sym: odd,
        });
    }
}

/// The NATURAL LDE row indices to gather for a matrix's query row-pairs:
/// per query, `rev(2*leaf)` and `rev(2*leaf+1)` (leaf = `row_pair_leaf`), which
/// is exactly what `append_row(2*leaf)` reads (it bit-reverses internally).
/// Out-of-range queries get row 0 (skipped by `fill_openings_from_gathered`).
#[cfg(feature = "cuda")]
fn query_row_indices<E>(
    mmcs: &MixedMmcs<E>,
    matrix: usize,
    iotas: &[usize],
    lde_size: usize,
) -> Vec<u32>
where
    E: IsField + 'static,
    FieldElement<E>: AsBytes + Sync + Send,
{
    let n = lde_size as u64;
    iotas
        .iter()
        .flat_map(|&iota| match mmcs.row_pair_leaf(iota, matrix) {
            Some(leaf) => [
                math::fft::bit_reversing::reverse_index(2 * leaf, n) as u32,
                math::fft::bit_reversing::reverse_index(2 * leaf + 1, n) as u32,
            ],
            None => [0u32, 0u32],
        })
        .collect()
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
fn assemble<E>(
    mmcs: &MixedMmcs<E>,
    iota: usize,
    openings: &mut [Vec<Option<PolynomialOpenings<E>>>],
    query: usize,
) -> Option<MixedOpening<E>>
where
    E: IsField + 'static,
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
/// Recompute a table's MAIN LDE on the GPU — the VRAM recompute mechanism: a
/// cheap device coset NTT. Returns the row-major host LDE (BYTE-IDENTICAL to
/// `expand_main_lde_row_major` — same DFT, exact modular arithmetic — kept so the
/// parts drain + the host fallback stay valid) AND the resident device handle,
/// which `lde_trace_take` attaches to the `LDETraceTable` so R2 constraint eval /
/// R4 DEEP fire their device paths (`evaluate_dev` / `try_deep_composition_gpu`
/// read `gpu_main`/`gpu_aux`, not `host_trace_empty`). `None` = ineligible (below
/// the device threshold / not Goldilocks / disk-spilled) → host FFT.
#[cfg(feature = "cuda")]
fn device_recompute_main_lde<Field, FieldExtension>(
    trace: &TraceTable<Field, FieldExtension>,
    domain: &Domain<Field>,
    twiddles: &crate::prover::LdeTwiddles<Field>,
    retain_host_lde: bool,
    #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
) -> Option<(Vec<FieldElement<Field>>, usize, math_cuda::lde::GpuLdeBase)>
where
    Field: IsFFTField + IsSubFieldOf<FieldExtension> + 'static,
    FieldExtension: IsField + 'static,
{
    // Disk mode spilled the trace; the device expand needs it in memory.
    #[cfg(feature = "disk-spill")]
    if storage_mode == StorageMode::Disk {
        return None;
    }
    let (trace_slice, num_cols) = trace.main_data_row_major();
    if num_cols == 0 {
        return None;
    }
    let n = trace_slice.len() / num_cols;
    // `retain_host_lde=false` = device-only: no D2H of the full LDE (host Vec
    // comes back empty). R2/R4 read the resident handle; the openings phase keeps
    // it true so its host row reads still work.
    let (handle, host_lde) = crate::gpu_lde::try_expand_row_major_keep_no_tree::<Field, Field>(
        trace_slice,
        trace.main_rowmajor_dev(),
        n,
        num_cols,
        domain.blowup_factor,
        &twiddles.coset_weights,
        retain_host_lde,
    )?;
    Some((host_lde, num_cols, handle))
}

/// Aux counterpart of [`device_recompute_main_lde`] (ext3 row-major LDE).
#[cfg(feature = "cuda")]
fn device_recompute_aux_lde<Field, FieldExtension>(
    trace: &TraceTable<Field, FieldExtension>,
    domain: &Domain<Field>,
    twiddles: &crate::prover::LdeTwiddles<Field>,
    retain_host_lde: bool,
    #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
) -> Option<(Vec<FieldElement<FieldExtension>>, usize, math_cuda::lde::GpuLdeExt3)>
where
    Field: IsFFTField + IsSubFieldOf<FieldExtension> + 'static,
    FieldExtension: IsField + 'static,
{
    #[cfg(feature = "disk-spill")]
    if storage_mode == StorageMode::Disk {
        return None;
    }
    let (trace_slice, num_cols) = trace.aux_data_row_major();
    if num_cols == 0 {
        return None;
    }
    let n = trace_slice.len() / num_cols;
    let (handle, host_lde) =
        crate::gpu_lde::try_expand_ext3_row_major_keep_no_tree::<Field, FieldExtension>(
            trace_slice,
            n,
            num_cols,
            domain.blowup_factor,
            &twiddles.coset_weights,
            retain_host_lde,
        )?;
    Some((host_lde, num_cols, handle))
}

fn materialize_ldes<Field, FieldExtension, PI, P>(
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
    // When true (R2 constraint eval, R4 DEEP), the GPU recompute is DEVICE-ONLY:
    // no D2H of the full LDE — those phases read the resident handle and only the
    // small composition parts come back to host. The openings phase passes false
    // (it reads host LDE rows), so the LDE is downloaded there. Ignored on the
    // host / retained paths.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] device_only: bool,
    #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
) -> LdePair<Field, FieldExtension>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + Copy + 'static,
    FieldExtension: IsField + Send + Sync + Copy + 'static,
    FieldElement<Field>: AsBytes + math::traits::ByteConversion + Sync + Send,
    FieldElement<FieldExtension>: AsBytes + math::traits::ByteConversion + Sync + Send,
    P: IsStarkProver<Field, FieldExtension, PI> + ?Sized,
{
    let (_, trace, _) = &air_trace_pairs[table];
    let mut bytes = 0usize;
    #[cfg(feature = "cuda")]
    let retain_host_lde = !device_only;
    // Resident device handles captured from a GPU recompute; attached to the
    // trace by `lde_trace_take` so R2/R4 fire their device paths.
    #[cfg(feature = "cuda")]
    let mut gpu_main: Option<math_cuda::lde::GpuLdeBase> = None;
    #[cfg(feature = "cuda")]
    let mut gpu_aux: Option<math_cuda::lde::GpuLdeExt3> = None;

    let main = match retained_main[table].take() {
        Some(lde) => lde,
        None => {
            let t_expand = std::time::Instant::now();
            // Recompute on the GPU (cheap device NTT) when eligible — the VRAM
            // recompute mechanism; the resident handle is kept + attached to the
            // trace so R2/R4 run on device. Falls back to the host FFT below
            // threshold / non-Goldilocks / disk-spill. Byte-identical either way.
            #[cfg(feature = "cuda")]
            let lde = match device_recompute_main_lde::<Field, FieldExtension>(
                trace,
                &domains[table],
                &twiddles[table],
                retain_host_lde,
                #[cfg(feature = "disk-spill")]
                storage_mode,
            ) {
                Some((host_lde, cols, handle)) => {
                    gpu_main = Some(handle);
                    (host_lde, cols)
                }
                None => P::expand_main_lde_row_major(
                    trace,
                    &domains[table],
                    &twiddles[table],
                    #[cfg(feature = "disk-spill")]
                    storage_mode,
                ),
            };
            #[cfg(not(feature = "cuda"))]
            let lde = P::expand_main_lde_row_major(
                trace,
                &domains[table],
                &twiddles[table],
                #[cfg(feature = "disk-spill")]
                storage_mode,
            );
            stats.lde_expansion_wall += t_expand.elapsed();
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
                let t_expand = std::time::Instant::now();
                #[cfg(feature = "cuda")]
                let lde = match device_recompute_aux_lde::<Field, FieldExtension>(
                    trace,
                    &domains[table],
                    &twiddles[table],
                    retain_host_lde,
                    #[cfg(feature = "disk-spill")]
                    storage_mode,
                ) {
                    Some((host_lde, cols, handle)) => {
                        gpu_aux = Some(handle);
                        (host_lde, cols)
                    }
                    None => P::expand_aux_lde_row_major(
                        trace,
                        &domains[table],
                        &twiddles[table],
                        #[cfg(feature = "disk-spill")]
                        storage_mode,
                    ),
                };
                #[cfg(not(feature = "cuda"))]
                let lde = P::expand_aux_lde_row_major(
                    trace,
                    &domains[table],
                    &twiddles[table],
                    #[cfg(feature = "disk-spill")]
                    storage_mode,
                );
                stats.lde_expansion_wall += t_expand.elapsed();
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
    LdePair {
        main,
        aux,
        bytes,
        #[cfg(feature = "cuda")]
        gpu_main,
        #[cfg(feature = "cuda")]
        gpu_aux,
    }
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

/// Move a table's LDE buffers into the trace view the phase reads — no copy.
/// The phases never mutate the buffers on the host path (the one bulk writer,
/// the cuda `set_host_data`, only FILLS deliberately-empty buffers), so the
/// same allocation flows phase → view → [`ldes_from_trace`] → retention, and
/// the transient double-residency the old clone created — one table's whole
/// main+aux LDE, invisible to the ledger — is gone.
fn lde_trace_take<Field, FieldExtension>(
    ldes: LdePair<Field, FieldExtension>,
    step_size: usize,
    blowup_factor: usize,
) -> (LDETraceTable<Field, FieldExtension>, usize)
where
    Field: IsFFTField + IsSubFieldOf<FieldExtension>,
    FieldExtension: IsField,
{
    let LdePair {
        main,
        aux,
        bytes,
        #[cfg(feature = "cuda")]
        gpu_main,
        #[cfg(feature = "cuda")]
        gpu_aux,
    } = ldes;
    // Detect device-only from the ACTUAL buffer state (empty host buffer + a
    // resident handle) BEFORE the Vecs are moved — mirrors the per-table
    // `build_lde_trace`. `device_only=false` (openings) keeps the host LDE, so
    // these stay false and the trace is a plain host trace with handles attached
    // (R2/R4 still read them via `gpu_main`/`gpu_aux`).
    #[cfg(feature = "cuda")]
    let main_empty = main.1 > 0 && main.0.is_empty() && gpu_main.is_some();
    #[cfg(feature = "cuda")]
    let host_trace_empty =
        main_empty || (aux.1 > 0 && aux.0.is_empty() && gpu_aux.is_some());
    #[cfg(feature = "cuda")]
    let device_rows = gpu_main
        .as_ref()
        .map(|h| h.lde_size)
        .or_else(|| gpu_aux.as_ref().map(|h| h.lde_size));
    #[allow(unused_mut)]
    let mut lde_trace =
        LDETraceTable::from_row_major(main.0, main.1, aux.0, aux.1, step_size, blowup_factor);
    #[cfg(feature = "cuda")]
    {
        if host_trace_empty {
            // `from_row_major` read num_rows from the empty host buffer (→ 0);
            // recover the true LDE row count from the resident handle.
            if let Some(n) = device_rows {
                lde_trace.set_num_rows(n);
            }
            lde_trace.set_host_trace_empty(true);
        }
        if let Some(h) = gpu_main {
            lde_trace.set_gpu_main(h);
        }
        if let Some(h) = gpu_aux {
            lde_trace.set_gpu_aux(h);
        }
    }
    (lde_trace, bytes)
}

/// Take the buffers back out of the trace view for release or retention —
/// the inverse of [`lde_trace_take`], carrying the byte account through.
fn ldes_from_trace<Field, FieldExtension>(
    lde_trace: LDETraceTable<Field, FieldExtension>,
    bytes: usize,
) -> LdePair<Field, FieldExtension>
where
    Field: IsFFTField + IsSubFieldOf<FieldExtension>,
    FieldExtension: IsField,
{
    LdePair {
        main: (lde_trace.main_data, lde_trace.num_main_cols),
        aux: (lde_trace.aux_data, lde_trace.num_aux_cols),
        bytes,
        // The resident handles (if any) live in `lde_trace` and drop with it here;
        // under RecomputeLde (the only path that attaches them) release drops
        // everything and the next phase recomputes.
        #[cfg(feature = "cuda")]
        gpu_main: None,
        #[cfg(feature = "cuda")]
        gpu_aux: None,
    }
}

/// One table's DEEP composition codeword, in NATURAL order.
#[allow(clippy::too_many_arguments)]
fn deep_codeword<Field, FieldExtension, PI, P>(
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
    P: IsStarkProver<Field, FieldExtension, PI> + ?Sized,
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

#[cfg(test)]
mod canary_tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField;

    type FE = FieldElement<GoldilocksField>;

    // Two mixed-height, row-major matrices (tallest first), the exact layout the
    // device main/aux commits hand the canary: matrix 0 is 8 rows × 2 cols
    // (log_height 3), matrix 1 is 4 rows × 1 col (log_height 2).
    fn source<'a>(d0: &'a [FE], d1: &'a [FE]) -> Vec<BorrowedMatrix<'a, GoldilocksField>> {
        vec![
            BorrowedMatrix::RowMajorNatural {
                data: d0,
                stride: 2,
                col_start: 0,
                width: 2,
                log_height: 3,
            },
            BorrowedMatrix::RowMajorNatural {
                data: d1,
                stride: 1,
                col_start: 0,
                width: 1,
                log_height: 2,
            },
        ]
    }

    fn fixture() -> (Vec<FE>, Vec<FE>) {
        let d0: Vec<FE> = (0..16u64).map(|i| FE::from(i + 1)).collect();
        let d1: Vec<FE> = (0..4u64).map(|i| FE::from(i + 100)).collect();
        (d0, d1)
    }

    #[test]
    fn canary_passes_on_a_tree_that_matches_its_data() {
        let (d0, d1) = fixture();
        let mmcs = MixedMmcs::commit(&source(&d0, &d1));
        assert!(
            xcheck_device_commit("test", &mmcs, source(&d0, &d1)).is_ok(),
            "the canary must accept a device tree that re-authenticates against its LDE"
        );
    }

    #[test]
    fn canary_fires_when_the_data_disagrees_with_the_tree() {
        let (d0, d1) = fixture();
        let mmcs = MixedMmcs::commit(&source(&d0, &d1));
        // The tallest matrix is hashed into EVERY leaf, so perturbing all its rows
        // makes every sampled leaf mismatch — a stand-in for a device that
        // committed the wrong data with a self-consistent tree.
        let d0_bad: Vec<FE> = (0..16u64).map(|i| FE::from(i + 2)).collect();
        assert!(
            xcheck_device_commit("test", &mmcs, source(&d0_bad, &d1)).is_err(),
            "the canary must reject a tree that disagrees with the retained LDE"
        );
    }

    #[test]
    fn canary_fires_when_a_committed_node_is_corrupted() {
        let (d0, d1) = fixture();
        let good = MixedMmcs::commit(&source(&d0, &d1));
        let dims = good.dims().to_vec();
        let h_max = good.h_max();
        // Flip a bit in the root digest (heap index 0); it sits on every leaf's
        // authentication path, so the recomputed climb from the honest data can
        // no longer reach it.
        let mut heap = good.heap_bytes();
        heap[0] ^= 0x01;
        let corrupted = MixedMmcs::from_heap_nodes(dims, h_max, &heap);
        assert!(
            xcheck_device_commit("test", &corrupted, source(&d0, &d1)).is_err(),
            "the canary must reject a tree with a corrupted committed node"
        );
    }
}
