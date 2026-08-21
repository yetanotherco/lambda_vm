use std::any::Any;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, OnceLock};
#[cfg(feature = "instruments")]
use std::time::{Duration, Instant};

use crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use math::fft::bit_reversing::{in_place_bit_reverse_permute, reverse_index};
use math::fft::bowers_fft::LayerTwiddles;
use math::fft::errors::FFTError;
use math::fft::two_half_fft::TwoHalfTwiddles;

use log::info;
use math::field::traits::{IsField, IsSubFieldOf};
use math::spill_safe::SpillSafe;
use math::traits::AsBytes;
use math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};

#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

#[cfg(feature = "debug-checks")]
use crate::debug::validate_trace;
use crate::fri;
use crate::lookup::LOGUP_NUM_CHALLENGES;
use crate::proof::stark::{DeepPolynomialOpenings, PolynomialOpenings};
#[cfg(feature = "disk-spill")]
use crate::storage_mode::StorageMode;
use crate::table::Table;
use crate::trace::LDETraceTable;

use super::config::{BatchedMerkleTree, BatchedMerkleTreeBackend, Commitment};
use super::constraints::evaluator::ConstraintEvaluator;
use super::domain::Domain;
use super::fri::fri_decommit::FriDecommitment;
use super::grinding;
use super::lookup::BusPublicInputs;
use super::proof::stark::{DeepPolynomialOpening, MultiProof, StarkProof};
use super::trace::TraceTable;
use super::traits::AIR;
#[cfg(feature = "cuda")]
use crypto::merkle_tree::proof::Proof;

pub use crate::commitment::{keccak_leaves_bit_reversed, keccak_leaves_row_pair_bit_reversed};

/// A triple of (AIR, TraceTable, PublicInputs) for proving.
type AirTracePair<'a, Field, FieldExtension, PI> = (
    &'a dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
    &'a mut TraceTable<Field, FieldExtension>,
    &'a PI,
);

/// A default STARK prover implementing `IsStarkProver`.
pub struct Prover<
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
    PI,
> {
    p: PhantomData<(Field, FieldExtension, PI)>,
}

impl<
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + 'static,
    FieldExtension: Send + Sync + IsField + 'static,
    PI,
> IsStarkProver<Field, FieldExtension, PI> for Prover<Field, FieldExtension, PI>
where
    FieldElement<Field>: math::traits::ByteConversion,
    FieldElement<FieldExtension>: math::traits::ByteConversion,
{
}

#[derive(Debug)]
pub enum ProvingError {
    WrongParameter(String),
    EmptyCommitment,
    /// The prover's recomputed preprocessed Merkle root did not match the
    /// commitment the AIR was constructed with (e.g. a stale static constant
    /// in a table module, or a wrong caller-supplied entry such as
    /// `page_commitments` / `decode_commitment`). Continuing would yield a
    /// proof an honest verifier always rejects — fail fast on the prover side
    /// with a localized error instead.
    PrecomputedCommitmentMismatch,
    /// I/O failure while spilling prover state (traces, LDE, Merkle trees) to disk:
    /// out of disk space, fd exhaustion, or mmap failure.
    #[cfg(feature = "disk-spill")]
    DiskSpill(String),
    /// An internal FFT/LDE computation failed (e.g. domain size exceeds the
    /// field's two-adicity, or a degenerate coset offset). Distinct from
    /// `WrongParameter` because the cause is internal prover machinery, not a
    /// caller-supplied parameter. Carries the underlying `FFTError`'s message.
    Fft(String),
}

impl From<FFTError> for ProvingError {
    fn from(e: FFTError) -> Self {
        ProvingError::Fft(format!("{e}"))
    }
}

/// Commitment artifacts for one trace table (main or auxiliary). Used for both
/// plain and preprocessed tables. Preprocessed tables additionally carry a
/// separate Merkle tree over their precomputed columns, hence the optional
/// `precomputed_tree`/`precomputed_root` pair and the `num_precomputed_cols`
/// index used when opening positions.
pub(crate) struct TableCommit<F: IsField>
where
    FieldElement<F>: AsBytes,
{
    /// Merkle tree over the trace columns (multiplicities only for preprocessed tables).
    pub(crate) tree: Arc<BatchedMerkleTree<F>>,
    /// Root of `tree`.
    pub(crate) root: Commitment,
    /// Preprocessed tables only: Merkle tree over precomputed columns.
    pub(crate) precomputed_tree: Option<Arc<BatchedMerkleTree<F>>>,
    /// Preprocessed tables only: root of `precomputed_tree`.
    pub(crate) precomputed_root: Option<Commitment>,
    /// Preprocessed tables only: number of precomputed columns. Zero otherwise.
    pub(crate) num_precomputed_cols: usize,
}

impl<F: IsField> TableCommit<F>
where
    FieldElement<F>: AsBytes,
{
    /// Build a `TableCommit` for a plain (non-preprocessed) table.
    fn plain(tree: BatchedMerkleTree<F>, root: Commitment) -> Self {
        Self {
            tree: Arc::new(tree),
            root,
            precomputed_tree: None,
            precomputed_root: None,
            num_precomputed_cols: 0,
        }
    }

    /// Build a `TableCommit` for a preprocessed table. The precomputed tree
    /// arrives as an `Arc` because it may be shared from the process-wide
    /// cache (see [`precomputed_tree_cache_get`]).
    fn preprocessed(
        tree: BatchedMerkleTree<F>,
        root: Commitment,
        precomputed_tree: Arc<BatchedMerkleTree<F>>,
        precomputed_root: Commitment,
        num_precomputed_cols: usize,
    ) -> Self {
        Self {
            tree: Arc::new(tree),
            root,
            precomputed_tree: Some(precomputed_tree),
            precomputed_root: Some(precomputed_root),
            num_precomputed_cols,
        }
    }

    /// Cheap clone. Only bumps Arc refcounts, no tree data is copied.
    fn share(&self) -> Self {
        Self {
            tree: Arc::clone(&self.tree),
            root: self.root,
            precomputed_tree: self.precomputed_tree.as_ref().map(Arc::clone),
            precomputed_root: self.precomputed_root,
            num_precomputed_cols: self.num_precomputed_cols,
        }
    }

    fn is_preprocessed(&self) -> bool {
        self.precomputed_tree.is_some()
    }
}

/// Process-wide cache of precomputed-column Merkle trees, keyed by their
/// commitment root. The root fully determines the tree (column content,
/// domain, blowup and leaf layout all feed the hash), so a hit needs no
/// re-verification: the lookup key IS the root a rebuild would be checked
/// against. This is what makes continuation epochs stop re-committing the
/// same DECODE/BITWISE/range tables once per epoch — those trees are
/// execution-independent; only the multiplicity columns change per run.
/// Type-erased so one static serves every field instantiation.
fn precomputed_tree_cache()
-> &'static Mutex<std::collections::HashMap<Commitment, Arc<dyn std::any::Any + Send + Sync>>> {
    static CACHE: OnceLock<
        Mutex<std::collections::HashMap<Commitment, Arc<dyn std::any::Any + Send + Sync>>>,
    > = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn precomputed_tree_cache_get<F: IsField + 'static>(
    root: &Commitment,
) -> Option<Arc<BatchedMerkleTree<F>>>
where
    FieldElement<F>: AsBytes,
{
    let cache = precomputed_tree_cache().lock().unwrap();
    cache
        .get(root)
        .cloned()
        .and_then(|any| any.downcast::<BatchedMerkleTree<F>>().ok())
}

fn precomputed_tree_cache_put<F: IsField + 'static>(
    root: Commitment,
    tree: Arc<BatchedMerkleTree<F>>,
) where
    FieldElement<F>: AsBytes,
{
    precomputed_tree_cache()
        .lock()
        .unwrap()
        .insert(root, tree as Arc<dyn std::any::Any + Send + Sync>);
}

/// A container for the results of the first round of the STARK Prove protocol.
pub(crate) struct Round1<Field, FieldExtension>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField,
    FieldExtension: IsField,
    FieldElement<Field>: AsBytes,
    FieldElement<FieldExtension>: AsBytes,
{
    /// The table of evaluations over the LDE of the main and auxiliary trace tables.
    pub(crate) lde_trace: LDETraceTable<Field, FieldExtension>,
    /// Commitment to the main trace.
    pub(crate) main: TableCommit<Field>,
    /// Commitment to the auxiliary (RAP) trace, if any.
    pub(crate) aux: Option<TableCommit<FieldExtension>>,
    /// The challenges of the RAP round.
    pub(crate) rap_challenges: Vec<FieldElement<FieldExtension>>,
    /// Bus interaction public inputs (initial and final aux column values).
    pub(crate) bus_public_inputs: Option<BusPublicInputs<FieldExtension>>,
}

/// Tuple returned by `commit_main_trace`: the commit, the cached LDE columns,
/// and (under cuda) the optional device LDE buffer kept alive for downstream
/// rounds when the R1 fused GPU pipeline ran.
#[cfg(feature = "cuda")]
type MainCommitTuple<F> = (
    TableCommit<F>,
    (Vec<FieldElement<F>>, usize),
    Option<math_cuda::lde::GpuLdeBase>,
);
#[cfg(not(feature = "cuda"))]
type MainCommitTuple<F> = (TableCommit<F>, (Vec<FieldElement<F>>, usize));

/// Round 1 commitment artifacts — Merkle trees, roots, challenges, and bus inputs.
/// Borrowed (not consumed) when building `Round1`.
pub(crate) struct Round1Commitments<Field, FieldExtension>
where
    Field: IsFFTField + IsSubFieldOf<FieldExtension>,
    FieldExtension: IsField,
    FieldElement<Field>: AsBytes,
    FieldElement<FieldExtension>: AsBytes,
{
    main: TableCommit<Field>,
    aux: Option<TableCommit<FieldExtension>>,
    rap_challenges: Vec<FieldElement<FieldExtension>>,
    bus_public_inputs: Option<BusPublicInputs<FieldExtension>>,
}

/// Main and auxiliary LDE columns, consumed by value when the table's `Round1`
/// is assembled.
///
/// Memory trade-off, asymmetric since the per-table scheduler fused aux build,
/// aux commit and rounds 2-4 into one task:
/// - main: produced by the Round 1 main commit, which is a phase-wide barrier,
///   so all N tables' main LDEs are live at once (O(N × main_cols × lde_size)).
/// - aux: produced and consumed inside the same fused task, so at most the
///   scheduler's `k` coexist (O(k × aux_cols × lde_size)) — which under `cuda`
///   is `num_airs`, so there they are all-N-live like the main ones.
///
/// Under `debug-checks` the fused task is split around the cross-table bus
/// balance check, so there the aux LDEs are all-N-live like the main ones.
struct Lde<Field: IsFFTField, FieldExtension: IsField> {
    /// Row-major main LDE buffer + its column count.
    main: (Vec<FieldElement<Field>>, usize),
    /// Row-major aux LDE buffer + its column count (`(vec![], 0)` if no aux).
    aux: (Vec<FieldElement<FieldExtension>>, usize),
    /// Device-side main LDE buffer, populated only when the R1 GPU fused
    /// pipeline ran for this table. Kept so R2/R3/R4 GPU paths can read
    /// the LDE without re-H2D.
    #[cfg(feature = "cuda")]
    gpu_main: Option<math_cuda::lde::GpuLdeBase>,
    #[cfg(feature = "cuda")]
    gpu_aux: Option<math_cuda::lde::GpuLdeExt3>,
}

impl<Field, FieldExtension> Round1Commitments<Field, FieldExtension>
where
    Field: IsFFTField + IsSubFieldOf<FieldExtension> + Send + Sync,
    FieldExtension: IsField + Send + Sync,
    FieldElement<Field>: AsBytes,
    FieldElement<FieldExtension>: AsBytes,
{
    /// Build a `Round1` by consuming a `Lde` and borrowing commitment data.
    /// The `TableCommit::share` calls are cheap — only bump Arc refcounts.
    fn build_round1(
        &self,
        lde: Lde<Field, FieldExtension>,
        step_size: usize,
        blowup_factor: usize,
    ) -> Round1<Field, FieldExtension> {
        let (main_data, num_main_cols) = lde.main;
        let (aux_data, num_aux_cols) = lde.aux;

        // Stage-3 device-only detection, inferred from the ACTUAL buffer state
        // (not the gate's intent): a table whose round-1 D2H was skipped has an
        // empty host buffer where it should have data. Using the real state is a
        // safety property — if the `device_only` gate held but the GPU keep path
        // fell back to CPU, the buffer is populated and this stays false, so the
        // proof runs on the host trace as normal. A mixed state (one buffer
        // empty, the other full) still sets the flag, and is legal rather than
        // an error: the aux commit may be more conservative than the main one
        // (never less), so an aux side that kept its host copy can sit next to
        // a device-only main. The R3 barycentric arms therefore guard on the
        // individual buffer — the side that still holds host data stays
        // readable — while the flag keeps the R4 and host-evaluator guards
        // armed. Reading the real state also picks up an R1 resident-aux
        // downgrade: it repopulates the host buffers before this point, so the
        // flag simply comes out false.
        #[cfg(feature = "cuda")]
        let main_empty = num_main_cols > 0 && main_data.is_empty();
        #[cfg(feature = "cuda")]
        let host_trace_empty =
            main_empty || (num_aux_cols > 0 && aux_data.is_empty() && lde.gpu_aux.is_some());
        #[cfg(feature = "cuda")]
        let device_num_rows = lde
            .gpu_main
            .as_ref()
            .map(|h| h.lde_size)
            .or_else(|| lde.gpu_aux.as_ref().map(|h| h.lde_size));

        #[allow(unused_mut)]
        let mut lde_trace = LDETraceTable::from_row_major(
            main_data,
            num_main_cols,
            aux_data,
            num_aux_cols,
            step_size,
            blowup_factor,
        );
        #[cfg(feature = "cuda")]
        {
            if host_trace_empty {
                // Recover the LDE row count from the resident device handle
                // whenever any host buffer is empty. `from_row_major` derives
                // `num_rows` from `main_data` (or `aux_data` when there are no
                // main columns); if that buffer was skipped it reads 0, so we
                // overwrite from the handle's `lde_size` (the true row count).
                // Idempotent when `from_row_major` already got it right, and it
                // covers the aux-only (`num_main_cols == 0`) device-only case
                // that a `main_empty`-only guard missed.
                if let Some(n) = device_num_rows {
                    lde_trace.set_num_rows(n);
                }
                lde_trace.set_host_trace_empty(true);
            }
            if let Some(h) = lde.gpu_main {
                lde_trace.set_gpu_main(h);
            }
            if let Some(h) = lde.gpu_aux {
                lde_trace.set_gpu_aux(h);
            }
        }
        Round1 {
            lde_trace,
            main: self.main.share(),
            aux: self.aux.as_ref().map(TableCommit::share),
            rap_challenges: self.rap_challenges.clone(),
            bus_public_inputs: self.bus_public_inputs.clone(),
        }
    }
}

/// Pre-computed twiddle factors and coset weights for a given domain size.
///
/// Shared across all columns of the same table, and across all phases (A, C, Rounds 2-4)
/// in sequential proving. This eliminates redundant root-of-unity generation:
/// many redundant `get_twiddles` calls reduced to one pair per distinct domain.
///
/// The `coset_weights` vector stores `[n_inv, n_inv*g, n_inv*g², ..., n_inv*g^{n-1}]`
/// where `g` is the coset offset and `n_inv = 1/n`. These are used in the iFFT+coset-shift
/// step of `expand_columns_to_lde`.
pub(crate) struct LdeTwiddles<F: IsFFTField> {
    /// Legacy per-column `LayerTwiddles`, only consumed by the debug-checks
    /// reconstruct path and the test-utils precomputed-commitment helper. Kept
    /// out of release builds so the production row-major LDE doesn't carry the
    /// extra (forward set is size `n·blowup`) twiddle memory for nothing.
    #[cfg(any(test, feature = "test-utils", feature = "debug-checks"))]
    inv: LayerTwiddles<F>,
    #[cfg(any(test, feature = "test-utils", feature = "debug-checks"))]
    fwd: LayerTwiddles<F>,
    /// Cache-blocked two-half twiddles for the batched row-major LDE path
    /// (`coset_lde_full_expand_row_major`). `two_half_inv` is size-`n` inverse,
    /// `two_half_fwd` size-`n·blowup` forward.
    two_half_inv: TwoHalfTwiddles<F>,
    two_half_fwd: TwoHalfTwiddles<F>,
    coset_weights: Vec<FieldElement<F>>,
    /// Composition half-extension cache, initialized only when the degree-2
    /// decomposition path actually runs on CPU.
    composition: OnceLock<CompositionLdeTwiddles<F>>,
    /// `1/(2·g·ωⁱ)` for the degree-2 quotient decomposition — see [`Self::inv_2x`].
    inv_2x: OnceLock<Arc<Vec<FieldElement<F>>>>,
}

pub(crate) struct CompositionLdeTwiddles<F: IsFFTField> {
    /// Inverse twiddles for the g²-coset halves of size `lde_size/2`.
    inv: LayerTwiddles<F>,
    /// Forward twiddles for the full g-coset of size `lde_size`.
    fwd: LayerTwiddles<F>,
    /// Weights `g⁻ʲ/(lde_size/2)` for the composition half-extension.
    weights: Vec<FieldElement<F>>,
}

impl<F: IsFFTField> CompositionLdeTwiddles<F> {
    fn new(half_size: usize, offset: &FieldElement<F>) -> Self {
        // Composition half-extension weights: g⁻ʲ / half_size. The constraint-
        // quotient halves live on the g²-coset of size `half_size`; the unnormalized
        // iFFT yields `n·cⱼ·(g²)ʲ` and these weights turn that into `cⱼ·gʲ` for the
        // forward FFT onto the g-coset.
        let half_size_fe = FieldElement::<F>::from(half_size as u64);
        let inv_half_size_offset = (&half_size_fe * offset)
            .inv()
            .expect("half_size and coset offset are non-zero");
        let half_size_inv = offset * &inv_half_size_offset;
        let offset_inv = &half_size_fe * &inv_half_size_offset;
        let weights = {
            let mut w = Vec::with_capacity(half_size);
            let mut cur = half_size_inv;
            for _ in 0..half_size {
                w.push(cur.clone());
                cur = &cur * &offset_inv;
            }
            w
        };

        Self {
            inv: LayerTwiddles::<F>::new_inverse(half_size.trailing_zeros() as u64)
                .expect("valid composition inverse twiddles"),
            fwd: LayerTwiddles::<F>::new((half_size * 2).trailing_zeros() as u64)
                .expect("valid composition forward twiddles"),
            weights,
        }
    }
}

impl<F: IsFFTField> LdeTwiddles<F> {
    /// Construct twiddles and coset weights for a domain of the given size and blowup factor.
    pub(crate) fn new(domain: &Domain<F>) -> Self {
        let domain_size = domain.interpolation_domain_size;
        let lde_size = domain_size * domain.blowup_factor;

        let domain_size_inv = FieldElement::<F>::from(domain_size as u64)
            .inv()
            .expect("domain_size is power of two");
        let offset = &domain.coset_offset;
        let coset_weights = {
            let mut w = Vec::with_capacity(domain_size);
            let mut offset_power = domain_size_inv;
            for _ in 0..domain_size {
                w.push(offset_power.clone());
                offset_power = offset * &offset_power;
            }
            w
        };

        Self {
            #[cfg(any(test, feature = "test-utils", feature = "debug-checks"))]
            inv: LayerTwiddles::<F>::new_inverse(domain_size.trailing_zeros() as u64)
                .expect("valid inverse twiddles"),
            #[cfg(any(test, feature = "test-utils", feature = "debug-checks"))]
            fwd: LayerTwiddles::<F>::new(lde_size.trailing_zeros() as u64)
                .expect("valid forward twiddles"),
            two_half_inv: TwoHalfTwiddles::<F>::new(domain_size.trailing_zeros() as usize, true)
                .expect("valid inverse two-half twiddles"),
            two_half_fwd: TwoHalfTwiddles::<F>::new(lde_size.trailing_zeros() as usize, false)
                .expect("valid forward two-half twiddles"),
            coset_weights,
            composition: OnceLock::new(),
            inv_2x: OnceLock::new(),
        }
    }

    fn composition(&self, domain: &Domain<F>) -> &CompositionLdeTwiddles<F> {
        let lde_size = domain.interpolation_domain_size * domain.blowup_factor;
        let half_size = lde_size / 2;
        debug_assert_eq!(self.coset_weights.len(), domain.interpolation_domain_size);
        self.composition
            .get_or_init(|| CompositionLdeTwiddles::new(half_size, &domain.coset_offset))
    }

    #[cfg(test)]
    pub(crate) fn has_composition_cache(&self) -> bool {
        self.composition.get().is_some()
    }

    /// `1/(2·g·ωⁱ)` for the degree-2 quotient decomposition, computed once per
    /// domain (an LDE/2-size batch inversion per table per epoch otherwise).
    /// `Arc`'d so the device-resident copy can pin it (see
    /// `gpu_interp::base_vec_device_handle`).
    fn inv_2x(&self, domain: &Domain<F>) -> &Arc<Vec<FieldElement<F>>> {
        self.inv_2x.get_or_init(|| {
            let n = domain.lde_roots_of_unity_coset.len() / 2;
            let mut inv: Vec<FieldElement<F>> = (0..n)
                // 2·(g·ωⁱ) = (g·ωⁱ).double() — one add, vs a base mul+reduce per element.
                .map(|i| domain.lde_roots_of_unity_coset[i].double())
                .collect();
            // Sequential: parallel inversion inside a OnceLock init can
            // deadlock the rayon pool (workers block on this same cell).
            FieldElement::inplace_batch_inverse_sequential(&mut inv)
                .expect("Coset points are non-zero");
            Arc::new(inv)
        })
    }
}

/// Process-wide `Domain` + `LdeTwiddles` cache keyed by
/// `(field, trace_length, blowup, coset_offset)`. Continuation epochs
/// otherwise rebuild the same ~24 MB `Domain` and
/// ~32 MB twiddle set per epoch; sharing the `Arc`s also lets every lazy
/// domain-derived cache (composition twiddles, `inv_2x`, OOD constants, FRI
/// inverse twiddles) fill once per process instead of once per epoch.
#[allow(clippy::type_complexity)]
fn domain_twiddle_cache() -> &'static std::sync::Mutex<
    std::collections::HashMap<(std::any::TypeId, usize, usize, u64), Box<dyn Any + Send + Sync>>,
> {
    static CACHE: OnceLock<
        std::sync::Mutex<
            std::collections::HashMap<
                (std::any::TypeId, usize, usize, u64),
                Box<dyn Any + Send + Sync>,
            >,
        >,
    > = OnceLock::new();
    CACHE.get_or_init(Default::default)
}

fn domain_and_twiddles<F, A>(air: &A, trace_length: usize) -> (Arc<Domain<F>>, Arc<LdeTwiddles<F>>)
where
    F: IsFFTField + 'static,
    FieldElement<F>: Send + Sync,
    A: AIR<Field = F> + ?Sized,
{
    type Entry<F> = (Arc<Domain<F>>, Arc<LdeTwiddles<F>>);
    let key = (
        std::any::TypeId::of::<F>(),
        trace_length,
        air.options().blowup_factor as usize,
        air.options().coset_offset,
    );
    {
        let cache = domain_twiddle_cache().lock().unwrap();
        if let Some(e) = cache.get(&key).and_then(|b| b.downcast_ref::<Entry<F>>()) {
            #[cfg(test)]
            crate::tests::domain_cache_stats::record(true);
            return e.clone();
        }
    }
    #[cfg(test)]
    crate::tests::domain_cache_stats::record(false);
    let d = Arc::new(Domain::new(air, trace_length));
    let t = Arc::new(LdeTwiddles::new(&d));
    // Pre-fill every lazy domain-derived cache from this setup thread, so no
    // rayon worker ever runs — or blocks waiting on — an initializer
    // mid-prove (a worker parked on a OnceLock can starve the initializer's
    // own pool work and deadlock the prove).
    let _ = d.ood_constants();
    let _ = d.fri_inv_twiddles();
    let _ = t.composition(&d);
    let _ = t.inv_2x(&d);
    let mut cache = domain_twiddle_cache().lock().unwrap();
    // Re-check under the lock: concurrent misses both build, and using the
    // loser would pin ITS per-instance vectors in the pointer-keyed device
    // caches for the process lifetime, duplicating VRAM. The winner stays.
    if let Some(e) = cache.get(&key).and_then(|b| b.downcast_ref::<Entry<F>>()) {
        return e.clone();
    }
    cache.insert(key, Box::new((d.clone(), t.clone())));
    (d, t)
}

/// Explicit `TABLE_PARALLELISM` override, honoured by both `k` values below so
/// setting it pins the scheduler and the storage estimate to the same number.
#[cfg(feature = "parallel")]
fn parallelism_override() -> Option<usize> {
    std::env::var("TABLE_PARALLELISM")
        .ok()
        .and_then(|s| s.parse().ok())
}

#[cfg(feature = "parallel")]
fn host_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Number of tables `multi_prove` proves concurrently, out of `num_airs` of
/// them.
///
/// Defaults: **every table** under `cuda`, `num_cores / 3` on CPU builds
/// (benchmarked optimal on both M3 Pro and EPYC 9454P — every table there is
/// pure host work, so `k` genuinely competes for cores). Both arms are
/// overridden by the `TABLE_PARALLELISM` env var, and the result is clamped to
/// `1..=num_airs`. Without the `parallel` feature this is 1 and the env var is
/// ignored.
///
/// # Why the `cuda` arm has no core term
///
/// Measured over 881 runs on two RTX 5090 boxes (sweep record linked from
/// PR #911): the work `k` divides is device- and workload-bound — invariant to
/// host core count over an 8× range — so `available_parallelism()` is the
/// wrong quantity to scale `k` by. `k` is not a thread count; it counts
/// concurrent drivers whose per-table work all runs on the one global rayon
/// pool. Worst case against the best measured `k`: `num_airs` +1.6 % (inside
/// noise), the old `cores*2/3` +13.0 %. Bounding concurrency is memory
/// admission's job (`VramGate`), not this count's.
pub fn table_parallelism(num_airs: usize) -> usize {
    #[cfg(feature = "parallel")]
    {
        // GPU builds: run every table. The work `k` divides is device- and
        // workload-bound, not core-bound — see the doc comment.
        #[cfg(feature = "cuda")]
        let k = parallelism_override().unwrap_or(num_airs);
        // CPU builds: every table is pure host work, so `k` competes for
        // the same cores the rayon pool wants.
        #[cfg(not(feature = "cuda"))]
        let k = parallelism_override().unwrap_or_else(|| (host_cores() / 3).max(1));
        k.clamp(1, num_airs.max(1))
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = num_airs;
        1
    }
}

/// How many tables' rounds 2-4 transients the *RAM* estimate assumes are alive
/// at once (`auto_storage::peak_bytes` sums the transient bytes of the top-k
/// tables, and `decide` turns that into RAM vs Disk).
///
/// Deliberately not `table_parallelism(num_airs)`. That is a ceiling, not a
/// bound: on a `cuda` build what actually limits how many tables are in flight
/// is `VramGate`'s byte budget, which this host-side estimate cannot see.
/// Feeding an unbounded count in here would sum *every* table's transients —
/// on many-PAGE shapes that inflates the estimate by up to +44 % (512 PAGE
/// tables at blowup 4) and would spill proofs to disk that fit in RAM. On the
/// shapes that reach this path today (~21 tables, one PAGE table) the top-k sum
/// has all but saturated, so this value and `num_airs` agree to well under 1 %.
///
/// Kept at exactly the value it had when the scheduler shared it, so splitting
/// the two does not move any storage decision.
///
/// TODO: derive this from a byte budget rather than a table count, so it
/// tracks what `VramGate` admits instead of standing in for it.
pub fn storage_estimate_parallelism() -> usize {
    #[cfg(feature = "parallel")]
    {
        parallelism_override().unwrap_or_else(|| {
            #[cfg(feature = "cuda")]
            {
                (host_cores() * 2 / 3).max(1)
            }
            #[cfg(not(feature = "cuda"))]
            {
                (host_cores() / 3).max(1)
            }
        })
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// Heuristic peak device bytes for one table: co-resident LDE columns plus the
/// resident Merkle trees, with a scratch factor for NTT and leaf transients. A
/// deliberate over estimate for a safety ceiling, not a precise allocator. Pass
/// aux_cols == 0 when the aux LDE is not yet resident (R1 main commit).
fn estimate_table_vram_bytes(main_cols: usize, aux_cols: usize, lde_size: usize) -> u64 {
    const BYTES_PER_BASE: u64 = 8;
    const EXT3_BYTES: u64 = 24;
    const SCRATCH_FACTOR: u64 = 2;
    const RESIDENT_TREE_BYTES_PER_LDE: u64 = 256;
    let lde = lde_size as u64;
    let per_row = (main_cols as u64).saturating_mul(BYTES_PER_BASE)
        + (aux_cols as u64).saturating_mul(EXT3_BYTES);
    let lde_term = lde.saturating_mul(per_row).saturating_mul(SCRATCH_FACTOR);
    let tree_term = lde.saturating_mul(RESIDENT_TREE_BYTES_PER_LDE);
    lde_term.saturating_add(tree_term)
}

/// Byte-budget admission gate for concurrently proven tables. `acquire`
/// blocks until the requested bytes fit under the budget, releasing on
/// permit drop. An oversized request is admitted alone (when nothing else
/// holds bytes), so tables larger than the whole budget still prove.
///
/// Only OS driver threads block here (see `run_admitted`) — never rayon
/// workers, whose pool the admitted tables use internally and which a
/// blocked worker would starve.
struct VramGate {
    used: std::sync::Mutex<u64>,
    freed: std::sync::Condvar,
    budget: u64,
}

struct VramPermit<'a> {
    gate: &'a VramGate,
    bytes: u64,
}

impl VramGate {
    fn new(budget: u64) -> Self {
        Self {
            used: std::sync::Mutex::new(0),
            freed: std::sync::Condvar::new(),
            budget,
        }
    }

    fn acquire(&self, bytes: u64) -> VramPermit<'_> {
        let mut used = self.used.lock().unwrap();
        loop {
            if *used == 0 || used.saturating_add(bytes) <= self.budget {
                *used = used.saturating_add(bytes);
                return VramPermit { gate: self, bytes };
            }
            used = self.freed.wait(used).unwrap();
        }
    }
}

impl Drop for VramPermit<'_> {
    fn drop(&mut self) {
        let mut used = self.gate.used.lock().unwrap();
        *used = used.saturating_sub(self.bytes);
        drop(used);
        self.gate.freed.notify_all();
    }
}

/// Run `task` once per table index on `workers` OS driver threads, admitting
/// each index through `gate` with its estimated bytes. `order` fixes the
/// start order (heaviest table first, so the long pole starts early and small
/// tables fill around it — the fixed chunks this replaces made every table
/// wait for the slowest of its chunk). Returns one slot per original index.
fn run_admitted<T: Send>(
    order: &[usize],
    estimates: &[u64],
    gate: &VramGate,
    workers: usize,
    task: impl Fn(usize) -> T + Sync,
) -> Vec<Option<T>> {
    let results: Vec<std::sync::Mutex<Option<T>>> = estimates
        .iter()
        .map(|_| std::sync::Mutex::new(None))
        .collect();
    let cursor = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..workers.max(1).min(order.len().max(1)) {
            scope.spawn(|| {
                loop {
                    let pos = cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if pos >= order.len() {
                        return;
                    }
                    let idx = order[pos];
                    let permit = gate.acquire(estimates[idx]);
                    let out = task(idx);
                    *results[idx].lock().unwrap() = Some(out);
                    drop(permit);
                }
            });
        }
    });
    results
        .into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect()
}

/// Table indices sorted heaviest-first by estimate.
fn heaviest_first(estimates: &[u64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..estimates.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(estimates[i]));
    order
}

/// A container for the results of the second round of the STARK Prove protocol.
pub(crate) struct Round2<F>
where
    F: IsField,
    FieldElement<F>: AsBytes,
{
    /// Evaluations of the composition polynomial parts over the LDE domain.
    pub(crate) lde_composition_poly_evaluations: Vec<Vec<FieldElement<F>>>,
    /// The Merkle tree built to compute the commitment to the composition polynomial parts.
    pub(crate) composition_poly_merkle_tree: BatchedMerkleTree<F>,
    /// The commitment to the composition polynomial parts.
    pub(crate) composition_poly_root: Commitment,
    /// The composition Merkle tree kept resident on device (when the R2 GPU tree
    /// path ran), so R4 openings gather paths on device instead of walking a host
    /// tree. When set, `composition_poly_merkle_tree` is a root only placeholder.
    /// `None` on the CPU path.
    #[cfg(feature = "cuda")]
    pub(crate) gpu_composition_tree: Option<math_cuda::lde::GpuMerkleTree>,
}

/// A container for the results of the third round of the STARK Prove protocol.
pub(crate) struct Round3<F: IsField> {
    /// Evaluations of the trace polynomials, main and auxiliary, at the out-of-domain challenge.
    trace_ood_evaluations: Table<F>,
    /// Evaluations of the composition polynomial parts at the out-of-domain challenge.
    composition_poly_parts_ood_evaluation: Vec<FieldElement<F>>,
}

/// A container for the results of the fourth round of the STARK Prove protocol.
pub(crate) struct Round4<F: IsSubFieldOf<E>, E: IsField> {
    /// Coefficients of the FRI final polynomial (degree < 2^k), emitted once
    /// folding reaches the terminal codeword.
    fri_final_poly_coeffs: Vec<FieldElement<E>>,
    /// The commitments to the fold polynomials of the inner layers of FRI.
    fri_layers_merkle_roots: Vec<Commitment>,
    /// The values and proofs of validity of the evaluations of the trace polynomials and the composition polynomials
    /// parts at the domain values corresponding to the FRI query challenges and their symmetric counterparts.
    deep_poly_openings: DeepPolynomialOpenings<F, E>,
    /// The values and proofs of validity of the evaluations of the fold polynomials of the inner
    /// layers of FRI at the values corresponding to the symmetrics of the FRI query challenges.
    query_list: Vec<FriDecommitment<E>>,
    /// The proof of work nonce.
    nonce: Option<u64>,
}

/// Returns the evaluations of the polynomial `p` over the lde domain defined by the given
/// `blowup_factor`, `domain_size` and `offset`. The number of evaluations returned is `domain_size
/// * blowup_factor`. The domain generator used is the one given by the implementation of `F` as `IsFFTField`.
pub fn evaluate_polynomial_on_lde_domain<F, E>(
    p: &Polynomial<FieldElement<E>>,
    blowup_factor: usize,
    domain_size: usize,
    offset: &FieldElement<F>,
) -> Result<Vec<FieldElement<E>>, FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
{
    let evaluations = Polynomial::evaluate_offset_fft(p, blowup_factor, Some(domain_size), offset)?;
    let step = evaluations.len() / (domain_size * blowup_factor);
    match step {
        1 => Ok(evaluations),
        _ => Ok(evaluations.into_iter().step_by(step).collect()),
    }
}

/// The functionality of a STARK prover providing methods to run the STARK Prove protocol
/// https://lambdaclass.github.io/lambdaworks/starks/protocol.html
/// The default implementation is complete and is compatible with Stone prover
/// https://github.com/starkware-libs/stone-prover
///
/// Note: many default-method signatures expose `pub(crate)` round-state types
/// (`Round1`, `Round2`, `Round3`, `Round4`, `LdeTwiddles`). These are internal
/// helpers — only `prove`, `multi_prove` are meant for callers. The
/// `private_interfaces` allow is removed once these helpers move off the trait.
#[allow(private_interfaces)]
pub trait IsStarkProver<
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync + 'static,
    FieldExtension: Send + Sync + IsField + 'static,
    PI,
> where
    FieldElement<Field>: math::traits::ByteConversion,
    FieldElement<FieldExtension>: math::traits::ByteConversion,
{
    /// Commit a row-major flat buffer (`num_rows * num_cols`) by hashing pairs
    /// of consecutive bit-reversed rows into each Merkle leaf (`ROWS_PER_LEAF = 2`).
    /// The byte layout per leaf matches `keccak_leaves_bit_reversed_grouped(columns, 2)`:
    /// leaf i = hash( row[br(2i)] ++ row[br(2i+1)] ), read as contiguous slices from
    /// the row-major buffer — no transpose needed.
    fn commit_rows_bit_reversed<E>(
        data: &[FieldElement<E>],
        num_cols: usize,
    ) -> Option<(BatchedMerkleTree<E>, Commitment)>
    where
        FieldElement<E>: AsBytes + Sync + Send + math::traits::ByteConversion,
        E: IsField,
    {
        Self::commit_rows_bit_reversed_subset(data, num_cols, 0, num_cols)
    }

    /// Subset variant of [`commit_rows_bit_reversed`]: hash pairs of bit-reversed rows
    /// from the column range `[col_start..col_end)`. Used for preprocessed traces where
    /// precomputed cols and multiplicity cols commit to separate Merkle trees from the
    /// same row-major buffer, both using the row-pair (`ROWS_PER_LEAF = 2`) leaf layout.
    fn commit_rows_bit_reversed_subset<E>(
        data: &[FieldElement<E>],
        num_cols: usize,
        col_start: usize,
        col_end: usize,
    ) -> Option<(BatchedMerkleTree<E>, Commitment)>
    where
        FieldElement<E>: AsBytes + Sync + Send + math::traits::ByteConversion,
        E: IsField,
    {
        use math::traits::ByteConversion;

        if num_cols == 0 || data.is_empty() || col_end <= col_start {
            return None;
        }
        debug_assert!(col_end <= num_cols);
        debug_assert_eq!(data.len() % num_cols, 0);
        let num_rows = data.len() / num_cols;
        if num_rows == 0 {
            return None;
        }
        debug_assert!(
            num_rows.is_power_of_two(),
            "num_rows must be a power of two for reverse_index"
        );

        // Local alias for the canonical constant, used several times below.
        const ROWS_PER_LEAF: usize = crate::commitment::ROWS_PER_LEAF;
        let num_leaves = num_rows / ROWS_PER_LEAF;
        let subset_cols = col_end - col_start;
        let byte_len = <FieldElement<E> as ByteConversion>::BYTE_LEN;
        let leaf_bytes = ROWS_PER_LEAF * subset_cols * byte_len;

        let hash_leaf = |buf: &mut [u8], leaf_idx: usize| -> Commitment {
            let mut offset = 0;
            for k in 0..ROWS_PER_LEAF {
                let br_idx = reverse_index(ROWS_PER_LEAF * leaf_idx + k, num_rows as u64);
                let row_start = br_idx * num_cols;
                let row = &data[row_start + col_start..row_start + col_end];
                for elem in row.iter() {
                    elem.write_bytes_be(&mut buf[offset..offset + byte_len]);
                    offset += byte_len;
                }
            }
            BatchedMerkleTreeBackend::<E>::hash_bytes(buf)
        };

        #[cfg(feature = "parallel")]
        let hashed_leaves: Vec<Commitment> = (0..num_leaves)
            .into_par_iter()
            .map_init(
                || vec![0u8; leaf_bytes],
                |buf, leaf_idx| hash_leaf(buf, leaf_idx),
            )
            .collect();
        #[cfg(not(feature = "parallel"))]
        let hashed_leaves: Vec<Commitment> = {
            let mut buf = vec![0u8; leaf_bytes];
            (0..num_leaves)
                .map(|leaf_idx| hash_leaf(&mut buf, leaf_idx))
                .collect()
        };

        let tree = BatchedMerkleTree::<E>::build_from_hashed_leaves(hashed_leaves)?;
        let root = tree.root;
        Some((tree, root))
    }

    /// Compute the LDE commitment for a subset of columns from a trace (for testing).
    ///
    /// This helper computes the same commitment the prover generates internally,
    /// useful for setting up soundness test scenarios. Only available under
    /// `cfg(test)` (in-crate) or with the `test-utils` Cargo feature
    /// (cross-crate tests).
    #[cfg(any(test, feature = "test-utils"))]
    fn compute_precomputed_commitment_for_testing(
        trace: &TraceTable<Field, FieldExtension>,
        air: &impl AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        num_precomputed_cols: usize,
    ) -> Option<Commitment>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let domain = Domain::new(air, trace.num_rows());
        let columns = trace.columns_main();
        let precomputed: Vec<_> = columns.into_iter().take(num_precomputed_cols).collect();
        let twiddles = LdeTwiddles::new(&domain);
        let evals =
            Self::compute_lde_from_columns_cached::<Field>(&precomputed, &domain, &twiddles);
        let (_, commitment) =
            crate::commitment::commit_bit_reversed(&evals, crate::commitment::ROWS_PER_LEAF)?;
        Some(commitment)
    }

    /// Compute LDE evaluations with pre-computed twiddle factors and coset weights.
    ///
    /// Accepts shared [`LdeTwiddles`] to avoid redundant twiddle generation and weight
    /// computation across phases (A, C, Rounds 2-4).
    ///
    /// Only the test-utils precomputed-commitment helper drives this; the
    /// production path commits the precomputed split via the row-major LDE.
    #[cfg(any(test, feature = "test-utils"))]
    fn compute_lde_from_columns_cached<E>(
        columns: &[Vec<FieldElement<E>>],
        domain: &Domain<Field>,
        twiddles: &LdeTwiddles<Field>,
    ) -> Vec<Vec<FieldElement<E>>>
    where
        E: IsSubFieldOf<FieldExtension> + Send + Sync,
        Field: IsSubFieldOf<E>,
        FieldElement<E>: Send + Sync,
    {
        if columns.is_empty() {
            return Vec::new();
        }

        crate::par::par_map_collect(0..columns.len(), |i| {
            Polynomial::coset_lde_full::<Field>(
                &columns[i],
                domain.blowup_factor,
                &twiddles.coset_weights,
                &twiddles.inv,
                &twiddles.fwd,
            )
            .expect("coset LDE computation")
        })
    }

    /// Expand each column in-place from N evaluations to N×blowup LDE evaluations.
    ///
    /// Performs iFFT + coset shift + FFT in place. Coset weights are pre-cached in
    /// `LdeTwiddles` to avoid recomputation across phases.
    ///
    /// Only the debug-checks reconstruct path uses this; production builds the
    /// main/aux LDE through the row-major two-half FFT.
    #[cfg(feature = "debug-checks")]
    fn expand_columns_to_lde<E>(
        columns: &mut [Vec<FieldElement<E>>],
        domain: &Domain<Field>,
        twiddles: &LdeTwiddles<Field>,
    ) where
        Field: IsSubFieldOf<E>,
        E: IsSubFieldOf<FieldExtension> + IsField + Send + Sync + 'static,
        FieldElement<E>: Send + Sync,
    {
        if columns.is_empty() {
            return;
        }

        // GPU batched fast path: all columns at once in one pipeline on one
        // stream. Falls through to per-column rayon when the table is too
        // small, the element type isn't Goldilocks, or the `cuda` feature is
        // off.
        #[cfg(feature = "cuda")]
        if crate::gpu_lde::try_expand_columns_batched::<Field, E>(
            columns,
            domain.blowup_factor,
            &twiddles.coset_weights,
        )
        .is_some()
        {
            return;
        }

        crate::par::par_for_each_mut(columns, |buf| {
            Polynomial::coset_lde_full_expand::<Field>(
                buf,
                domain.blowup_factor,
                &twiddles.coset_weights,
                &twiddles.inv,
                &twiddles.fwd,
            )
            .expect("coset LDE expansion");
        });
    }

    /// Stage-3 device-only gate for one table (see
    /// [`crate::gpu_lde::device_only_gate`]). Derived from the AIR + domain;
    /// the main commit uses it as is, while the aux commit additionally
    /// requires the main commit to have produced a device handle — the aux
    /// side may be more conservative than the main side (never less), which
    /// keeps a mixed GPU-aux/CPU-main state out.
    #[cfg(feature = "cuda")]
    fn device_only_for(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        domain: &Domain<Field>,
    ) -> bool {
        // Preconditions the downstream GPU paths require that the numeric gate
        // below does not capture. A table missing any of them would pass the
        // gate and skip its host D2H, leaving round 2 to recover through
        // `materialize_lde_trace_host` — correct, but a downgrade, and an
        // abort if the resident handles cannot serve the data:
        //  - R2 composition unconditionally needs a device aux handle
        //    (`gpu_aux()?`), so the table must declare an aux trace.
        //  - The composition path needs a uniform zerofier with ≥1 group. An
        //    empty constraint set makes `all(end_exemptions == 0)` vacuously
        //    true here but `is_uniform()` false downstream (0 groups).
        //  - Device-only is entered only for the d=2 quotient decomposition,
        //    checked below once `n` is in hand. The d=1 device R2 path exists
        //    too but keeps its host trace (never device-only), so it is not a
        //    concern here.
        if !air.has_aux_trace() || air.constraints_meta().is_empty() {
            return false;
        }
        let n = domain.interpolation_domain_size;
        // Device-only is entered only for the d=2 quotient decomposition. The d=1
        // (num_parts==1) device R2 path exists too, but keeps its host trace
        // (`want_host` stays true) so the preprocessed main commit still sees real
        // data — zeroing a preprocessed table's host trace fails its commitment
        // check. So d=1 tables run device-additive, never device-only.
        if air.composition_poly_degree_bound(n) / n != 2 {
            return false;
        }
        let lde_size = domain.interpolation_domain_size * domain.blowup_factor;
        let offsets_contiguous =
            crate::gpu_lde::offsets_are_contiguous(&air.context().transition_offsets);
        let zerofier_uniform = air.constraints_meta().iter().all(|m| m.end_exemptions == 0);
        crate::gpu_lde::device_only_gate::<Field, FieldExtension>(
            lde_size,
            n,
            offsets_contiguous,
            zerofier_uniform,
        )
    }

    /// Compute the main-trace LDE and commit. Returns a `TableCommit` along
    /// with the owned LDE columns (consumed later by the table's fused task)
    /// and (under cuda) the optional device LDE buffer kept alive for
    /// downstream rounds when the R1 fused GPU pipeline ran.
    ///
    /// `precomputed`: if present, the leading `num_cols` columns are committed
    /// as a separate Merkle tree (the precomputed split for preprocessed
    /// tables) and the root is checked against the AIR-hardcoded commitment.
    #[allow(clippy::type_complexity)]
    fn commit_main_trace(
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        twiddles: &LdeTwiddles<Field>,
        precomputed: Option<(Commitment, usize)>,
        #[cfg(feature = "cuda")] device_only: bool,
        #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
    ) -> Result<MainCommitTuple<Field>, ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let lde_size = domain.interpolation_domain_size * domain.blowup_factor;

        // Fused GPU path (cuda only): row-major NTT — single H2D from the
        // already-row-major trace, no column extraction, no transpose.
        // Falls back to CPU if GPU path returns None.
        #[cfg(feature = "cuda")]
        if precomputed.is_none() {
            let (trace_slice, num_cols) = trace.main_data_row_major();
            let n = if num_cols > 0 {
                trace_slice.len() / num_cols
            } else {
                0
            };
            #[cfg(feature = "instruments")]
            let t_sub = Instant::now();
            if let Some((tree, handle, main_data)) =
                crate::gpu_lde::try_expand_leaf_and_tree_row_major_keep::<
                    Field,
                    Field,
                    BatchedMerkleTreeBackend<Field>,
                >(
                    trace_slice,
                    trace.main_rowmajor_dev(),
                    n,
                    num_cols,
                    domain.blowup_factor,
                    &twiddles.coset_weights,
                    !device_only,
                )
            {
                #[cfg(feature = "instruments")]
                let main_lde_dur = t_sub.elapsed();
                let root = tree.root;
                #[cfg(feature = "instruments")]
                crate::instruments::accum_r1_main(main_lde_dur, std::time::Duration::ZERO);
                // Count a device-only main commit only once the GPU keep path
                // actually fired (handle produced + host trace intentionally
                // empty), so the counter reflects real residency, not the gate.
                if device_only {
                    crate::gpu_lde::GPU_DEVICE_ONLY_CALLS
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return Ok((
                    TableCommit::plain(tree, root),
                    (main_data, num_cols),
                    Some(handle),
                ));
            }
        }

        // Fused GPU split path for preprocessed tables (cuda only): one
        // row-major LDE of ALL columns plus two subset Merkle trees
        // (precomputed / multiplicity) built on device — leaves and levels are
        // bit-identical to `commit_rows_bit_reversed_subset`. The precomputed
        // tree comes back as a full host tree, so the process-wide
        // precomputed-tree cache works unchanged; the multiplicity tree stays
        // device-resident behind a root-only host tree and its opening paths
        // are gathered on device. The handle keeps the LDE device-resident for
        // the downstream GPU rounds.
        #[cfg(feature = "cuda")]
        if let Some((expected_precomputed_root, num_precomputed)) = precomputed {
            let (trace_slice, num_cols) = trace.main_data_row_major();
            let n = if num_cols > 0 {
                trace_slice.len() / num_cols
            } else {
                0
            };
            #[cfg(feature = "disk-spill")]
            let cache_ok = storage_mode != StorageMode::Disk;
            #[cfg(not(feature = "disk-spill"))]
            let cache_ok = true;
            let cached_pre = cache_ok
                .then(|| precomputed_tree_cache_get::<Field>(&expected_precomputed_root))
                .flatten();
            #[cfg(feature = "instruments")]
            let t_sub = Instant::now();
            if let Some((pre_tree, mult_tree, handle, main_data)) =
                crate::gpu_lde::try_expand_split_trees_row_major_keep::<
                    Field,
                    Field,
                    BatchedMerkleTreeBackend<Field>,
                >(
                    trace_slice,
                    trace.main_rowmajor_dev(),
                    n,
                    num_cols,
                    domain.blowup_factor,
                    &twiddles.coset_weights,
                    num_precomputed,
                    cached_pre.is_none(),
                    !device_only,
                )
            {
                #[cfg(feature = "instruments")]
                crate::instruments::accum_r1_main(t_sub.elapsed(), std::time::Duration::ZERO);
                if device_only {
                    crate::gpu_lde::GPU_DEVICE_ONLY_CALLS
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                let precomputed_tree = match cached_pre {
                    // Cache key == the root a rebuild would be verified
                    // against, so a hit needs no re-check.
                    Some(tree) => tree,
                    None => {
                        #[allow(unused_mut)]
                        let mut tree = pre_tree.expect("precomputed tree requested on cache miss");
                        if tree.root != expected_precomputed_root {
                            return Err(ProvingError::PrecomputedCommitmentMismatch);
                        }
                        #[cfg(feature = "disk-spill")]
                        Self::spill_tree(&mut tree, storage_mode, "precomputed Merkle tree")?;
                        let tree = Arc::new(tree);
                        if cache_ok {
                            precomputed_tree_cache_put::<Field>(
                                expected_precomputed_root,
                                Arc::clone(&tree),
                            );
                        }
                        tree
                    }
                };
                #[allow(unused_mut)]
                let mut mult_tree = mult_tree;
                #[cfg(feature = "disk-spill")]
                Self::spill_tree(&mut mult_tree, storage_mode, "mult Merkle tree")?;
                let mult_root = mult_tree.root;
                let commit = TableCommit::preprocessed(
                    mult_tree,
                    mult_root,
                    precomputed_tree,
                    expected_precomputed_root,
                    num_precomputed,
                );
                return Ok((commit, (main_data, num_cols), Some(handle)));
            }
            // GPU split path declined (size threshold / tower) → CPU path below.
        }

        // CPU path: the trace `Table` is already row-major, so copy it directly
        // (one memcpy — no transpose) and expand in place with the cache-blocked
        // batched two-half FFT. Row-major end-to-end: no LDE-size transpose,
        // contiguous Merkle leaves.
        let (trace_data, total_cols) = trace.main_data_row_major();

        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();

        let mut main_data: Vec<FieldElement<Field>> = Vec::with_capacity(lde_size * total_cols);
        main_data.extend_from_slice(trace_data);

        #[cfg(feature = "disk-spill")]
        if storage_mode == StorageMode::Disk {
            trace.main_table.advise_drop_cache();
        }

        Polynomial::<FieldElement<Field>>::coset_lde_full_expand_row_major::<Field>(
            &mut main_data,
            total_cols,
            domain.blowup_factor,
            &twiddles.coset_weights,
            &twiddles.two_half_inv,
            &twiddles.two_half_fwd,
        )
        .expect("row-major coset LDE expansion");

        #[cfg(feature = "instruments")]
        let main_lde_dur = t_sub.elapsed();

        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();

        let commit = match precomputed {
            None => {
                #[allow(unused_mut)]
                let (mut tree, root) = Self::commit_rows_bit_reversed(&main_data, total_cols)
                    .ok_or(ProvingError::EmptyCommitment)?;
                #[cfg(feature = "disk-spill")]
                Self::spill_tree(&mut tree, storage_mode, "main Merkle tree")?;
                TableCommit::plain(tree, root)
            }
            Some((expected_precomputed_root, num_precomputed)) => {
                // Only the multiplicity columns depend on the execution; the
                // precomputed-columns tree is a pure function of (content,
                // domain) already pinned by `expected_precomputed_root`, so it
                // is reused from the process cache when this exact commitment
                // was built before — across epochs and across proves. Bypassed
                // in disk-spill Disk mode, where trees are spilled (mutated).
                #[cfg(feature = "disk-spill")]
                let cache_ok = storage_mode != StorageMode::Disk;
                #[cfg(not(feature = "disk-spill"))]
                let cache_ok = true;
                let precomputed_tree = match cache_ok
                    .then(|| precomputed_tree_cache_get::<Field>(&expected_precomputed_root))
                    .flatten()
                {
                    // Cache key == the root a rebuild would be verified
                    // against, so a hit needs no re-check.
                    Some(tree) => tree,
                    None => {
                        #[allow(unused_mut)]
                        let (mut tree, root) = Self::commit_rows_bit_reversed_subset(
                            &main_data,
                            total_cols,
                            0,
                            num_precomputed,
                        )
                        .ok_or(ProvingError::EmptyCommitment)?;
                        if root != expected_precomputed_root {
                            return Err(ProvingError::PrecomputedCommitmentMismatch);
                        }
                        #[cfg(feature = "disk-spill")]
                        Self::spill_tree(&mut tree, storage_mode, "precomputed Merkle tree")?;
                        let tree = Arc::new(tree);
                        if cache_ok {
                            precomputed_tree_cache_put::<Field>(
                                expected_precomputed_root,
                                Arc::clone(&tree),
                            );
                        }
                        tree
                    }
                };
                #[allow(unused_mut)]
                let (mut mult_tree, mult_root) = Self::commit_rows_bit_reversed_subset(
                    &main_data,
                    total_cols,
                    num_precomputed,
                    total_cols,
                )
                .ok_or(ProvingError::EmptyCommitment)?;
                #[cfg(feature = "disk-spill")]
                Self::spill_tree(&mut mult_tree, storage_mode, "mult Merkle tree")?;
                TableCommit::preprocessed(
                    mult_tree,
                    mult_root,
                    precomputed_tree,
                    expected_precomputed_root,
                    num_precomputed,
                )
            }
        };

        #[cfg(feature = "instruments")]
        crate::instruments::accum_r1_main(main_lde_dur, t_sub.elapsed());

        #[cfg(feature = "cuda")]
        return Ok((commit, (main_data, total_cols), None));
        #[cfg(not(feature = "cuda"))]
        Ok((commit, (main_data, total_cols)))
    }

    /// Spill a committed Merkle tree to disk when `storage_mode` is `Disk`,
    /// tagging any I/O error with `label`. No-op otherwise. Shared by every commit
    /// site (main / preprocessed split / aux).
    #[cfg(feature = "disk-spill")]
    fn spill_tree<C>(
        tree: &mut BatchedMerkleTree<C>,
        storage_mode: StorageMode,
        label: &str,
    ) -> Result<(), ProvingError>
    where
        C: IsField,
        FieldElement<C>: AsBytes + Sync + Send,
    {
        if storage_mode == StorageMode::Disk {
            tree.spill_nodes_to_disk()
                .map_err(|e| ProvingError::DiskSpill(format!("{label}: {e}")))?;
        }
        Ok(())
    }

    /// Recompute Round1 from the trace, reusing the Merkle trees stored in commitments.
    ///
    /// Only used by `run_debug_checks` — the production path consumes the
    /// cached LDE directly and does not go through here.
    #[cfg(feature = "debug-checks")]
    fn reconstruct_round1(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        trace: &TraceTable<Field, FieldExtension>,
        domain: &Domain<Field>,
        commitment: &Round1Commitments<Field, FieldExtension>,
        twiddles: &LdeTwiddles<Field>,
    ) -> Result<Round1<Field, FieldExtension>, ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let lde_size = domain.interpolation_domain_size * domain.blowup_factor;

        // Column LDE then interleave to row-major (debug path: correctness over
        // speed; the values match the production row-major LDE).
        let mut main_cols = trace.extract_columns_main(lde_size);
        Self::expand_columns_to_lde::<Field>(&mut main_cols, domain, twiddles);
        let num_main_cols = main_cols.len();
        let main_rows = if num_main_cols > 0 {
            main_cols[0].len()
        } else {
            0
        };
        let mut main_data = vec![FieldElement::<Field>::zero(); main_rows * num_main_cols];
        if num_main_cols > 0 {
            for (row, dst) in main_data.chunks_exact_mut(num_main_cols).enumerate() {
                for (col, src) in main_cols.iter().enumerate() {
                    dst[col] = src[row].clone();
                }
            }
        }
        let main = (main_data, num_main_cols);

        let aux = if air.has_aux_trace() {
            let mut aux_cols = trace.extract_columns_aux(lde_size);
            Self::expand_columns_to_lde::<FieldExtension>(&mut aux_cols, domain, twiddles);
            let num_aux_cols = aux_cols.len();
            let aux_rows = if num_aux_cols > 0 {
                aux_cols[0].len()
            } else {
                0
            };
            let mut aux_data =
                vec![FieldElement::<FieldExtension>::zero(); aux_rows * num_aux_cols];
            if num_aux_cols > 0 {
                // clone required (generic conditionally-Copy extension element);
                // clippy's `clone_on_copy` here is a false positive.
                #[allow(clippy::clone_on_copy)]
                for (row, dst) in aux_data.chunks_exact_mut(num_aux_cols).enumerate() {
                    for (col, src) in aux_cols.iter().enumerate() {
                        dst[col] = src[row].clone();
                    }
                }
            }
            (aux_data, num_aux_cols)
        } else {
            (Vec::new(), 0)
        };

        Ok(commitment.build_round1(
            Lde {
                main,
                aux,
                #[cfg(feature = "cuda")]
                gpu_main: None,
                #[cfg(feature = "cuda")]
                gpu_aux: None,
            },
            air.step_size(),
            domain.blowup_factor,
        ))
    }

    /// Reconstruct Round1 for every table, print the bus balance report, and
    /// validate each trace. Called once after every table's aux commit, which
    /// under `debug-checks` means between the fused chain's two admitted
    /// passes — cross-table bus balance needs all the commitments at once.
    #[cfg(feature = "debug-checks")]
    fn run_debug_checks(
        pair_cells: &[std::sync::Mutex<AirTracePair<'_, Field, FieldExtension, PI>>],
        commitments: &[Round1Commitments<Field, FieldExtension>],
        domains: &[Arc<Domain<Field>>],
        twiddle_caches: &[Arc<LdeTwiddles<Field>>],
    ) where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        PI: Send + Sync + Clone,
    {
        let mut temp_results: Vec<Round1<Field, FieldExtension>> =
            Vec::with_capacity(pair_cells.len());
        for ((cell, commitment), (domain, twiddles)) in pair_cells
            .iter()
            .zip(commitments.iter())
            .zip(domains.iter().zip(twiddle_caches.iter()))
        {
            let pair = cell.lock().unwrap();
            let (air, trace, _) = &*pair;
            let result = Self::reconstruct_round1(*air, trace, domain, commitment, twiddles)
                .expect("reconstruct_round1 failed in debug-checks");
            temp_results.push(result);
        }

        let all_bus_public_inputs: Vec<Option<BusPublicInputs<FieldExtension>>> = temp_results
            .iter()
            .map(|r| r.bus_public_inputs.clone())
            .collect();
        print_bus_balance_report(&all_bus_public_inputs);

        for ((cell, round_1_result), domain) in pair_cells
            .iter()
            .zip(temp_results.iter())
            .zip(domains.iter())
        {
            let pair = cell.lock().unwrap();
            let (air, trace, pub_inputs) = &*pair;
            validate_trace(
                *air,
                *pub_inputs,
                trace,
                domain,
                &round_1_result.rap_challenges,
                round_1_result.bus_public_inputs.as_ref(),
            );
        }
    }

    /// Decompose the resident composition `H` into device-resident parts per the
    /// AIR's part count: the trivial d=1 de-interleave (`H` is the single part on
    /// the LDE coset) or the d=2 quotient split H₀/H₁. Both keep the parts
    /// device-resident (commit / R3 OOD / R4 DEEP / openings all read the count
    /// from the handle); `None` → the caller falls back to the host path. Shared
    /// by the R2 producer and the `xcheck` mirror so the two cannot drift.
    /// `want_host` gates the d=2 host drain only — d=1 tables are never
    /// device-only, so they always keep their host part.
    #[cfg(feature = "cuda")]
    fn decompose_comp_h_dev(
        number_of_parts: usize,
        h_dev: &math_cuda::constraint_interp::GpuCompH,
        domain: &Domain<Field>,
        twiddles: &LdeTwiddles<Field>,
        want_host: bool,
    ) -> Option<(
        Vec<Vec<FieldElement<FieldExtension>>>,
        math_cuda::lde::GpuLdeExt3,
    )> {
        if number_of_parts == 1 {
            crate::gpu_lde::try_deinterleave_comp_h_dev::<Field, FieldExtension>(h_dev)
        } else {
            crate::gpu_lde::try_decompose_extend_d2_dev::<Field, FieldExtension>(
                h_dev,
                twiddles.inv_2x(domain),
                &twiddles.composition(domain).weights,
                want_host,
            )
        }
    }

    /// Algebraically decompose H(x) = H₀(x²) + x·H₁(x²) on the LDE coset, then
    /// extend each half to the full LDE domain. This replaces the expensive
    /// iFFT(2N) + break_in_parts + FFT(2N)×2 pipeline with:
    ///   O(N) pointwise ops + iFFT(N)×2 + FFT(2N)×2
    ///
    /// The identity used:
    ///   H₀(x²) = (H(x) + H(-x)) / 2
    ///   H₁(x²) = (H(x) - H(-x)) / (2x)
    ///
    /// On the LDE coset {g·ω^i | i=0..2N-1}, we have -g·ω^i = g·ω^{i+N}
    /// since ω^N = -1 for a 2N-th root of unity ω.
    fn decompose_and_extend_d2(
        constraint_evaluations: &[FieldElement<FieldExtension>],
        domain: &Domain<Field>,
        twiddles: &LdeTwiddles<Field>,
    ) -> Vec<Vec<FieldElement<FieldExtension>>>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let two_n = constraint_evaluations.len();
        let n = two_n / 2;
        debug_assert_eq!(two_n, n * 2);

        // Step 1: 1/(2·g·ω^i) for i=0..N-1, cached once per domain in the
        // shared twiddles (base field — mixed F×E multiplication below).
        let inv_2x = twiddles.inv_2x(domain);
        debug_assert_eq!(inv_2x.len(), n);

        // Step 2: Pointwise decomposition.
        // H₀((g·ω^i)²) = (evals[i] + evals[i+N]) / 2
        // H₁((g·ω^i)²) = (evals[i] - evals[i+N]) / (2·g·ω^i)
        let two_inv = FieldElement::<Field>::from(2u64)
            .inv()
            .expect("2 is non-zero in the field");
        let (h0_evals, h1_evals) = crate::par::map_unzip(n, |i| {
            let sum = &constraint_evaluations[i] + &constraint_evaluations[i + n];
            let diff = &constraint_evaluations[i] - &constraint_evaluations[i + n];
            // F × E → E (base field scalar on left for mixed multiplication)
            (&two_inv * &sum, &inv_2x[i] * &diff)
        });

        // Step 3: Extend each part from n evals on the g²-coset to 2n evals on the
        // g-coset (the full LDE domain).

        // GPU fast path: batch both halves into one ext3 LDE call. Requires
        // `cuda` feature and a qualifying size. Falls through to CPU when not.
        #[cfg(feature = "cuda")]
        if let Some((lde_h0, lde_h1)) =
            crate::gpu_lde::try_extend_two_halves_gpu(&h0_evals, &h1_evals, domain)
        {
            return vec![lde_h0, lde_h1];
        }

        let composition_twiddles = twiddles.composition(domain);
        let (lde_h0, lde_h1) = crate::par::join(
            || Self::extend_half_to_lde(&h0_evals, composition_twiddles),
            || Self::extend_half_to_lde(&h1_evals, composition_twiddles),
        );
        vec![lde_h0, lde_h1]
    }

    /// Extend `half_evals` — `n = lde_size/2` evaluations of a degree-`<n` polynomial
    /// on the g²-coset — to `2n` evaluations on the g-coset (the full LDE domain).
    ///
    /// Fused: iFFT(n) → coset reshift g²→g → forward FFT(2n) in a single pass with no
    /// intermediate coefficient `Polynomial`. The twiddles and the weights `g⁻ʲ/n`
    /// (which fold the 1/n normalization and the net g²→g shift) are cached lazily
    /// once per domain in [`LdeTwiddles`].
    fn extend_half_to_lde(
        half_evals: &[FieldElement<FieldExtension>],
        twiddles: &CompositionLdeTwiddles<Field>,
    ) -> Vec<FieldElement<FieldExtension>>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        debug_assert_eq!(half_evals.len(), twiddles.weights.len());
        Polynomial::coset_lde_full::<Field>(
            half_evals,
            2,
            &twiddles.weights,
            &twiddles.inv,
            &twiddles.fwd,
        )
        .expect("coset extension")
    }

    /// Returns the result of the second round of the STARK Prove protocol.
    fn round_2_compute_composition_polynomial(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        pub_inputs: &PI,
        domain: &Domain<Field>,
        twiddles: &LdeTwiddles<Field>,
        round_1_result: &mut Round1<Field, FieldExtension>,
        transition_coefficients: &[FieldElement<FieldExtension>],
        boundary_coefficients: &[FieldElement<FieldExtension>],
    ) -> Result<Round2<FieldExtension>, ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        // Compute the evaluations of the composition polynomial on the LDE domain.
        let trace_length = domain.interpolation_domain_size;
        let evaluator = ConstraintEvaluator::new(
            air,
            pub_inputs,
            &round_1_result.rap_challenges,
            round_1_result.bus_public_inputs.as_ref(),
            trace_length,
        );
        let number_of_parts = air.composition_poly_degree_bound(trace_length) / trace_length;

        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();
        #[cfg(feature = "cuda")]
        let mut gpu_composition_parts: Option<math_cuda::lde::GpuLdeExt3> = None;

        // Fully device-resident d=2 path: H stays on device through decompose +
        // half extension, and the parts handle feeds the commit tree, R3 OOD,
        // R4 DEEP and the openings. The evaluations are drained to host only
        // while a host trace copy exists (fallback consumers); under
        // device-only nothing leaves the device and the placeholders below
        // stay empty. Any miss falls through to the host path (downloading H
        // when the evaluation itself already ran on device).
        #[cfg(feature = "cuda")]
        let mut precomputed_parts: Option<Vec<Vec<FieldElement<FieldExtension>>>> = None;
        // A downloaded `H` awaiting the host decompose: produced under the
        // lock below, consumed after it — the host iFFT + LDEs are pure CPU
        // work and must not serialize other tables' device windows.
        #[cfg(feature = "cuda")]
        let mut downloaded_h: Option<Vec<FieldElement<FieldExtension>>> = None;
        #[cfg(feature = "cuda")]
        if (number_of_parts == 1 || number_of_parts == 2) && !crate::gpu_lde::gpu_force_downgrade()
        {
            // Serializing this window across tables (device constraint eval +
            // decompose, where H is born) empirically eliminates a transient
            // whole-buffer H corruption seen under concurrent R2 windows on
            // VRAM pressure. What the guard orders is submission: a
            // device-only table's window is enqueue-only, so its kernels may
            // still overlap another table's on device. The commit, the host
            // decompose of a downloaded `H` and every host arm run outside
            // the lock. The force-downgrade test hook skips this fast path so
            // every device-only table exercises the host recovery below.
            let _r2_serial_guard = crate::gpu_lde::r2_serialize_guard();
            if let Some(h_dev) = evaluator.evaluate_dev(
                air,
                &round_1_result.lde_trace,
                domain,
                transition_coefficients,
                boundary_coefficients,
                &round_1_result.rap_challenges,
            ) {
                let want_host = !round_1_result.lde_trace.host_trace_empty();
                // num_parts==1 de-interleaves `H` (the single part); num_parts==2
                // runs the degree-2 quotient split. Both keep the parts resident.
                let decomposed = Self::decompose_comp_h_dev(
                    number_of_parts,
                    &h_dev,
                    domain,
                    twiddles,
                    want_host,
                );
                match decomposed {
                    Some((parts, handle)) => {
                        gpu_composition_parts = Some(handle);
                        precomputed_parts = Some(parts);
                    }
                    None => {
                        downloaded_h =
                            crate::gpu_lde::download_comp_h_to_field::<FieldExtension>(&h_dev);
                    }
                }
            }
        }
        #[cfg(feature = "cuda")]
        if let Some(h) = downloaded_h.take() {
            // num_parts==1: the downloaded `H` IS the single part (no host
            // decompose); num_parts==2: run the host degree-2 split + extend.
            precomputed_parts = Some(if number_of_parts == 1 {
                vec![h]
            } else {
                Self::decompose_and_extend_d2(&h, domain, twiddles)
            });
        }
        #[cfg(not(feature = "cuda"))]
        let precomputed_parts: Option<Vec<Vec<FieldElement<FieldExtension>>>> = None;

        #[cfg(feature = "instruments")]
        let constraints_dur = t_sub.elapsed();
        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();

        // Every arm below runs the HOST evaluator, which reads `get_main` /
        // `get_aux`. Under device-only those buffers are intentionally empty,
        // so landing here means the device decompose AND the `H` download both
        // failed. The gate is a static predicate and cannot mirror every
        // dynamic decline, so recover rather than abort: download the resident
        // LDEs into the host buffers (which also clears the device-only flag)
        // and let the host arms run — slower for this table, never wrong. The
        // assert is left for the case where the handles themselves cannot
        // serve the data, so that failure carries the device-only contract's
        // message rather than a bare index-out-of-bounds from somewhere inside
        // the evaluator.
        #[cfg(feature = "cuda")]
        if precomputed_parts.is_none() && round_1_result.lde_trace.host_trace_empty() {
            let recovered =
                crate::gpu_lde::materialize_lde_trace_host(&mut round_1_result.lde_trace);
            if recovered {
                // Rare by design; the name tells which condition the gate is
                // missing so it can be mirrored as an optimization.
                eprintln!(
                    "[gpu] device-only downgrade: table={} n={} num_parts={} \
                     (device R2 path declined; continuing on host)",
                    air.name(),
                    trace_length,
                    number_of_parts,
                );
            }
            assert!(
                recovered,
                "R2 composition fell back to the host evaluator on a device-only \
                 trace and the resident handles could not be downloaded: \
                 table={} n={} num_parts={} main_cols={} aux_cols={}",
                air.name(),
                trace_length,
                number_of_parts,
                round_1_result.lde_trace.num_main_cols(),
                round_1_result.lde_trace.num_aux_cols(),
            );
        }

        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut lde_composition_poly_parts_evaluations = if let Some(parts) = precomputed_parts {
            parts
        } else if number_of_parts == 2 {
            // Direct quotient decomposition: avoid full-size iFFT by algebraically
            // splitting H(x) = H₀(x²) + x·H₁(x²) using:
            //   H₀(x²) = (H(x) + H(-x)) / 2
            //   H₁(x²) = (H(x) - H(-x)) / (2x)
            // On the LDE coset {g·ω^i}, we have -g·ω^i = g·ω^{i+N} since ω^N = -1.
            let constraint_evaluations = evaluator.evaluate(
                air,
                &round_1_result.lde_trace,
                domain,
                transition_coefficients,
                boundary_coefficients,
                &round_1_result.rap_challenges,
            );
            Self::decompose_and_extend_d2(&constraint_evaluations, domain, twiddles)
        } else if number_of_parts == 1 {
            // Degree bound equals trace length: constraint evals are the LDE directly.
            vec![evaluator.evaluate(
                air,
                &round_1_result.lde_trace,
                domain,
                transition_coefficients,
                boundary_coefficients,
                &round_1_result.rap_challenges,
            )]
        } else {
            // Fallback for any future AIR with d > 2.
            let constraint_evaluations = evaluator.evaluate(
                air,
                &round_1_result.lde_trace,
                domain,
                transition_coefficients,
                boundary_coefficients,
                &round_1_result.rap_challenges,
            );
            let composition_poly =
                Polynomial::interpolate_offset_fft(&constraint_evaluations, &domain.coset_offset)?;
            let composition_poly_parts = composition_poly.break_in_parts(number_of_parts);

            let cpu_eval = || -> Result<Vec<Vec<FieldElement<FieldExtension>>>, ProvingError> {
                composition_poly_parts
                    .iter()
                    .map(|part| {
                        evaluate_polynomial_on_lde_domain(
                            part,
                            domain.blowup_factor,
                            domain.interpolation_domain_size,
                            &domain.coset_offset,
                        )
                        .map_err(ProvingError::from)
                    })
                    .collect()
            };

            // GPU fast path: batched ext3 LDE for all parts in one call.
            // `_keep` variant retains the de-interleaved device buffer as a
            // `GpuLdeExt3` handle stored on Round2 so R4 DEEP can skip the
            // `num_parts * 3 * lde_size * 8` byte H2D.
            #[cfg(feature = "cuda")]
            {
                let parts_slices: Vec<&[FieldElement<FieldExtension>]> = composition_poly_parts
                    .iter()
                    .map(|p| p.coefficients.as_slice())
                    .collect();
                match crate::gpu_lde::try_evaluate_parts_on_lde_gpu_keep::<Field, FieldExtension>(
                    &parts_slices,
                    domain.blowup_factor,
                    domain.interpolation_domain_size,
                    &domain.coset_offset,
                ) {
                    Some((evals, handle)) => {
                        gpu_composition_parts = Some(handle);
                        evals
                    }
                    None => cpu_eval()?,
                }
            }
            #[cfg(not(feature = "cuda"))]
            cpu_eval()?
        };

        #[cfg(feature = "instruments")]
        let fft_dur = t_sub.elapsed();

        // Fold the R2 device composition parts handle into the session
        // (resident R2 to R4) before the commit: the tree build below, its
        // recovery, R3 OOD, R4 DEEP and the openings all read it from the
        // trace. The host evaluations stay in `Round2` for the R4 openings.
        #[cfg(feature = "cuda")]
        if let Some(handle) = gpu_composition_parts {
            round_1_result.lde_trace.set_gpu_composition_parts(handle);
        }

        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();
        // GPU fast path for the comp-poly Merkle commit: hash straight from
        // the resident parts handle when R2 kept one (no host pack + H2D
        // re-upload); otherwise wrap the host eval Vecs. Either way the tree
        // stays resident on device (no whole-tree copy), a root-only host tree
        // is returned, and the device tree is threaded to R4 in
        // `Round2.gpu_composition_tree`.
        #[cfg(feature = "cuda")]
        let (composition_poly_merkle_tree, composition_poly_root, gpu_composition_tree) =
            match round_1_result
                .lde_trace
                .gpu_composition_parts()
                .and_then(|h| {
                    crate::gpu_lde::try_build_comp_poly_tree_gpu_from_dev::<
                        FieldExtension,
                        BatchedMerkleTreeBackend<FieldExtension>,
                    >(h)
                })
                .or_else(|| {
                    crate::gpu_lde::try_build_comp_poly_tree_gpu::<
                        FieldExtension,
                        BatchedMerkleTreeBackend<FieldExtension>,
                    >(&lde_composition_poly_parts_evaluations)
                }) {
                Some((host_tree, dev_tree)) => {
                    let root = host_tree.root;
                    (host_tree, root, Some(dev_tree))
                }
                None => {
                    // The host part evals are empty under device-only (the R2
                    // drain is skipped) — repopulate them from the resident
                    // parts handle rather than abort. Gate on the parts the
                    // CPU fallback actually consumes, not on
                    // `host_trace_empty()`: the trace can stay device-resident
                    // while these parts were downloaded to the host anyway (the
                    // GPU decompose fell back to `decompose_and_extend_d2`), in
                    // which case the materialize is a no-op. The assert fires
                    // only when the handle cannot serve the data.
                    let recovered = crate::gpu_lde::materialize_composition_parts_host(
                        &round_1_result.lde_trace,
                        &mut lde_composition_poly_parts_evaluations,
                    );
                    assert!(
                        recovered,
                        "R2 composition commit fell back to the host part evals \
                         on a device-only table and the resident parts handle \
                         could not be downloaded"
                    );
                    let (tree, root) = crate::commitment::commit_bit_reversed(
                        &lde_composition_poly_parts_evaluations,
                        crate::commitment::ROWS_PER_LEAF,
                    )
                    .ok_or(ProvingError::EmptyCommitment)?;
                    (tree, root, None)
                }
            };
        #[cfg(not(feature = "cuda"))]
        let (composition_poly_merkle_tree, composition_poly_root) =
            crate::commitment::commit_bit_reversed(
                &lde_composition_poly_parts_evaluations,
                crate::commitment::ROWS_PER_LEAF,
            )
            .ok_or(ProvingError::EmptyCommitment)?;
        #[cfg(feature = "instruments")]
        let merkle_dur = t_sub.elapsed();

        #[cfg(feature = "instruments")]
        crate::instruments::store_r2_sub(constraints_dur, fft_dur, merkle_dur);

        Ok(Round2 {
            lde_composition_poly_evaluations: lde_composition_poly_parts_evaluations,
            composition_poly_merkle_tree,
            composition_poly_root,
            #[cfg(feature = "cuda")]
            gpu_composition_tree,
        })
    }

    /// Returns the result of the third round of the STARK Prove protocol.
    fn round_3_evaluate_polynomials_in_out_of_domain_element(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        domain: &Domain<Field>,
        round_1_result: &mut Round1<Field, FieldExtension>,
        round_2_result: &mut Round2<FieldExtension>,
        z: &FieldElement<FieldExtension>,
    ) -> Round3<FieldExtension>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let num_parts = round_2_result.lde_composition_poly_evaluations.len();
        let z_power = z.pow(num_parts);
        let domain_size = domain.interpolation_domain_size;
        let blowup_factor = domain.blowup_factor;

        // === Shared domain constants for barycentric evaluation (cached per domain) ===
        let dc = domain.ood_constants();

        // === Composition poly parts: barycentric evaluation at z^num_parts ===
        let comp_z_pow_n = z_power.pow(domain_size);

        // GPU fast path: strided barycentric straight over the resident R2
        // parts handle (device inv_denoms for the single point z^P), skipping
        // the host stride-extract and the sequential CPU fold per part.
        #[cfg(feature = "cuda")]
        let gpu_parts_ood: Option<Vec<FieldElement<FieldExtension>>> =
            round_1_result
                .lde_trace
                .gpu_composition_parts()
                .and_then(|parts_dev| {
                    let dispatch = |inv_host: &[FieldElement<FieldExtension>],
                                ctx: Option<(&crate::gpu_lde::R3DevContext, usize)>| {
                    crate::gpu_lde::try_barycentric_ext3_on_ext3_handle::<Field, FieldExtension>(
                        parts_dev,
                        blowup_factor,
                        &dc.points,
                        &dc.offset_pow_n,
                        &dc.size_inv,
                        &dc.offset_pow_n_inv,
                        &comp_z_pow_n,
                        inv_host,
                        ctx,
                    )
                };
                    match crate::gpu_lde::try_prep_r3_dev_context::<Field, FieldExtension>(
                        &dc.points,
                        std::slice::from_ref(&z_power),
                        round_1_result.lde_trace.bound_stream(),
                    ) {
                        Some(ctx) => dispatch(&[], Some((&ctx, 0))),
                        // Below the dev-context threshold (single eval point):
                        // host inv_denoms + the same strided kernel, mirroring the
                        // trace OOD's mixed arm.
                        None => {
                            let inv =
                                math::polynomial::barycentric_inv_denoms(&z_power, &dc.points);
                            dispatch(&inv, None)
                        }
                    }
                });
        #[cfg(not(feature = "cuda"))]
        let gpu_parts_ood: Option<Vec<FieldElement<FieldExtension>>> = None;

        let composition_poly_parts_ood_evaluation: Vec<_> = match gpu_parts_ood {
            Some(v) => v,
            None => {
                // The host part evals are empty under device-only (the R2
                // drain is skipped) — repopulate them from the resident parts
                // handle rather than abort; the assert fires only when the
                // handle cannot serve the data.
                #[cfg(feature = "cuda")]
                {
                    let recovered = crate::gpu_lde::materialize_composition_parts_host(
                        &round_1_result.lde_trace,
                        &mut round_2_result.lde_composition_poly_evaluations,
                    );
                    assert!(
                        recovered,
                        "R3 parts OOD fell back to the host part evals on a \
                         device-only table and the resident parts handle could \
                         not be downloaded"
                    );
                }
                let comp_inv_denoms =
                    math::polynomial::barycentric_inv_denoms(&z_power, &dc.points);
                round_2_result
                    .lde_composition_poly_evaluations
                    .iter()
                    .map(|lde_evals| {
                        // Extract trace-size evaluations (stride = blowup_factor)
                        let evals: Vec<FieldElement<FieldExtension>> = (0..domain_size)
                            .map(|i| lde_evals[i * blowup_factor].clone())
                            .collect();
                        math::polynomial::interpolate_coset_eval_ext_with_g_n_inv(
                            &comp_z_pow_n,
                            &dc.offset_pow_n,
                            &dc.size_inv,
                            &dc.offset_pow_n_inv,
                            &dc.points,
                            &evals,
                            &comp_inv_denoms,
                        )
                    })
                    .collect()
            }
        };

        // === Trace polynomials: barycentric evaluation via LDE ===
        let trace_ood_evaluations = crate::trace::get_trace_evaluations_from_lde(
            &mut round_1_result.lde_trace,
            domain,
            z,
            &air.context().transition_offsets,
            air.step_size(),
            dc,
        );

        Round3 {
            trace_ood_evaluations,
            composition_poly_parts_ood_evaluation,
        }
    }

    /// The pruned-OOD layout for this AIR — the single place in the prover that
    /// reads the shape metadata (`trace_columns`, `step_size`, the
    /// transition-offset count, and the next-row column set). The round-3 block
    /// split and the round-4 DEEP-coefficient assignment both derive from the
    /// returned [`crate::ood::OodLayout`], which the verifier rebuilds identically
    /// (invariant I3).
    fn ood_layout(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
    ) -> crate::ood::OodLayout {
        crate::ood::OodLayout::new(
            air.context().trace_columns,
            air.context().transition_offsets.len() * air.step_size(),
            air.step_size(),
            air.trace_ood_next_row_columns(),
        )
    }

    /// Returns the result of the fourth round of the STARK Prove protocol.
    fn round_4_compute_and_run_fri_on_the_deep_composition_polynomial(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        domain: &Domain<Field>,
        round_1_result: &mut Round1<Field, FieldExtension>,
        round_2_result: &mut Round2<FieldExtension>,
        round_3_result: &Round3<FieldExtension>,
        z: &FieldElement<FieldExtension>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
    ) -> Round4<Field, FieldExtension>
    where
        FieldElement<FieldExtension>: AsBytes,
        FieldElement<Field>: AsBytes,
    {
        let coset_offset_u64 = air.context().proof_options.coset_offset;
        let coset_offset = FieldElement::<Field>::from(coset_offset_u64);

        let gamma = transcript.sample_field_element();

        let n_terms_composition_poly = round_2_result.lde_composition_poly_evaluations.len();
        // g·z pruning: only the current-row block (all columns) plus the masked
        // next-row columns get an opening / DEEP coefficient.
        let layout = Self::ood_layout(air);
        let num_terms_trace = layout.num_surviving();

        // <<<< Receive challenges: 𝛾, 𝛾'
        let mut deep_composition_coefficients: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
                .take(n_terms_composition_poly + num_terms_trace)
                .collect();

        let trace_term_powers: Vec<_> = deep_composition_coefficients
            .drain(..num_terms_trace)
            .collect();
        // Rectangular W×num_eval_points grid with the sampled powers at surviving
        // positions and zeros at pruned next-row positions, so the DEEP loop
        // below (and the GPU path) stay unchanged — zero-coefficient terms vanish.
        let trace_term_coeffs = layout.build_trace_term_coeffs(&trace_term_powers);

        // <<<< Receive challenges: 𝛾ⱼ, 𝛾ⱼ'
        let gammas = deep_composition_coefficients;

        let domain_size = domain.lde_roots_of_unity_coset.len();

        // Fully device-resident DEEP → FRI: the codeword is computed, bit-
        // reversed, and folded on device without crossing PCIe. On any miss
        // (gates, cudarc failure — the FRI driver restores the transcript)
        // the host path below recomputes DEEP through its own arms.
        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();
        #[cfg(feature = "cuda")]
        let precomputed_fri = Self::try_compute_deep_dev(
            &round_1_result.lde_trace,
            round_2_result,
            round_3_result,
            z,
            domain,
            &domain.trace_primitive_root,
            &gammas,
            &trace_term_coeffs,
        )
        .and_then(|dw| {
            crate::gpu_lde::try_fri_commit_gpu_from_dev(
                dw,
                transcript,
                &coset_offset,
                domain.blowup_factor.trailing_zeros(),
                air.options().fri_final_poly_log_degree as u32,
                domain.fri_inv_twiddles(),
                !round_1_result.lde_trace.host_trace_empty(),
            )
        });
        #[cfg(not(feature = "cuda"))]
        #[allow(clippy::type_complexity)]
        let precomputed_fri: Option<(
            Vec<FieldElement<FieldExtension>>,
            Vec<
                crate::fri::fri_commitment::FriLayer<
                    FieldExtension,
                    crate::config::FriLayerMerkleTreeBackend<FieldExtension>,
                >,
            >,
        )> = None;
        #[cfg(feature = "instruments")]
        let mut other_dur_1 = t_sub.elapsed();
        #[cfg(feature = "instruments")]
        let mut r4_fft_dur = Duration::ZERO;
        #[cfg(feature = "instruments")]
        let mut r4_merkle_dur = Duration::ZERO;

        let (fri_final_poly_coeffs, fri_layers) = if let Some(res) = precomputed_fri {
            res
        } else {
            // Compute p₀ (deep composition polynomial) as N evaluations on the LDE coset
            #[cfg(feature = "instruments")]
            let t_sub = Instant::now();
            let deep_evals = Self::compute_deep_composition_poly_evaluations(
                &mut round_1_result.lde_trace,
                round_2_result,
                round_3_result,
                z,
                domain,
                &domain.trace_primitive_root,
                &gammas,
                &trace_term_coeffs,
            );
            #[cfg(feature = "instruments")]
            {
                other_dur_1 += t_sub.elapsed();
            }

            // DEEP evaluations are already at 2N LDE points — just bit-reverse for FRI.
            // No iFFT+FFT extension needed (Plonky3-style direct LDE computation).
            #[cfg(feature = "instruments")]
            let t_sub = Instant::now();
            let mut lde_evals = deep_evals;
            in_place_bit_reverse_permute(&mut lde_evals);
            #[cfg(feature = "instruments")]
            {
                r4_fft_dur = t_sub.elapsed();
            }

            // FRI commit phase from pre-computed evaluations
            #[cfg(feature = "instruments")]
            let t_sub = Instant::now();
            let res = fri::commit_phase_from_evaluations(
                lde_evals,
                transcript,
                &coset_offset,
                domain_size,
                domain.blowup_factor.trailing_zeros(),
                air.options().fri_final_poly_log_degree as u32,
                domain.fri_inv_twiddles(),
            );
            #[cfg(feature = "instruments")]
            {
                r4_merkle_dur = t_sub.elapsed();
            }
            res
        };

        // grinding: generate nonce and append it to the transcript
        #[cfg(feature = "instruments")]
        let t_sub = Instant::now();
        let security_bits = air.context().proof_options.grinding_factor;
        let mut nonce = None;
        if security_bits > 0 {
            let nonce_value =
                grinding::generate_nonce_maybe_gpu(&transcript.state(), security_bits)
                    .expect("nonce not found");
            transcript.append_bytes(&nonce_value.to_be_bytes());
            nonce = Some(nonce_value);
        }

        let number_of_queries = air.options().fri_number_of_queries;
        let iotas = Self::sample_query_indexes(number_of_queries, domain, transcript);

        let query_list = fri::query_phase(&fri_layers, &iotas);

        let fri_layers_merkle_roots: Vec<_> = fri_layers
            .iter()
            .map(|layer| layer.merkle_tree.root)
            .collect();

        let deep_poly_openings =
            Self::open_deep_composition_poly(domain, round_1_result, round_2_result, &iotas);

        #[cfg(feature = "instruments")]
        {
            let queries_dur = t_sub.elapsed();
            crate::instruments::store_r4_sub(r4_fft_dur, r4_merkle_dur, other_dur_1, queries_dur);
        }

        Round4 {
            fri_final_poly_coeffs,
            fri_layers_merkle_roots,
            deep_poly_openings,
            query_list,
            nonce,
        }
    }

    fn sample_query_indexes(
        number_of_queries: usize,
        domain: &Domain<Field>,
        transcript: &mut impl IsStarkTranscript<FieldExtension, Field>,
    ) -> Vec<usize> {
        let domain_size = domain.lde_roots_of_unity_coset.len() as u64;
        (0..number_of_queries)
            .map(|_| (transcript.sample_u64(domain_size >> 1)) as usize)
            .collect::<Vec<usize>>()
    }

    /// Computes the DEEP composition polynomial at all 2N LDE points (Plonky3-style).
    ///
    /// Evaluates directly on the full LDE domain, eliminating the iFFT(N)+FFT(2N)
    /// extension that was needed when computing at only N trace-coset points.
    /// The result is ready for FRI after bit-reversal — no FFT needed.
    ///
    /// The DEEP polynomial is:
    ///   deep(X) = Σ_j γ_j * (H_j(X) - H_j(z^K)) / (X - z^K)
    ///           + Σ_{j,k} γ'_{j,k} * (t_j(X) - t_j(z·w^k)) / (X - z·w^k)
    #[allow(clippy::too_many_arguments)]
    /// Fully device-resident DEEP: device inv-denoms + resident parts handle,
    /// codeword kept on device in FRI order for [`gpu_lde::try_fri_commit_gpu_from_dev`].
    /// `None` → the host DEEP path (which retries its own GPU arms).
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn try_compute_deep_dev(
        lde_trace: &LDETraceTable<Field, FieldExtension>,
        round_2_result: &Round2<FieldExtension>,
        round_3_result: &Round3<FieldExtension>,
        z: &FieldElement<FieldExtension>,
        domain: &Domain<Field>,
        primitive_root: &FieldElement<Field>,
        composition_poly_gammas: &[FieldElement<FieldExtension>],
        trace_terms_gammas: &[Vec<FieldElement<FieldExtension>>],
    ) -> Option<math_cuda::deep::GpuDeepCodeword>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let parts_dev = lde_trace.gpu_composition_parts()?;
        let num_parts = round_2_result.lde_composition_poly_evaluations.len();
        let z_power = z.pow(num_parts);
        let num_eval_points = if trace_terms_gammas.is_empty() {
            0
        } else {
            trace_terms_gammas[0].len()
        };
        let mut z_shifted = Vec::with_capacity(num_eval_points);
        let mut current_z = z.clone();
        for _ in 0..num_eval_points {
            z_shifted.push(current_z.clone());
            current_z = primitive_root * &current_z;
        }
        let z_scalars: Vec<FieldElement<FieldExtension>> =
            core::iter::once(z_power).chain(z_shifted).collect();
        let (inv_dev, stream) =
            crate::gpu_lde::try_inv_denoms_dev_with_stream::<Field, FieldExtension>(
                &domain.lde_roots_of_unity_coset,
                &z_scalars,
                math_cuda::inverse::DenomSign::XMinusZ,
                lde_trace.bound_stream(),
            )?;
        crate::gpu_lde::try_deep_composition_gpu_keep::<Field, FieldExtension>(
            lde_trace,
            parts_dev,
            &round_3_result.composition_poly_parts_ood_evaluation,
            &round_3_result.trace_ood_evaluations.columns(),
            composition_poly_gammas,
            trace_terms_gammas,
            (&inv_dev, &stream),
            num_eval_points,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_deep_composition_poly_evaluations(
        lde_trace: &mut LDETraceTable<Field, FieldExtension>,
        round_2_result: &mut Round2<FieldExtension>,
        round_3_result: &Round3<FieldExtension>,
        z: &FieldElement<FieldExtension>,
        domain: &Domain<Field>,
        primitive_root: &FieldElement<Field>,
        composition_poly_gammas: &[FieldElement<FieldExtension>],
        trace_terms_gammas: &[Vec<FieldElement<FieldExtension>>],
    ) -> Vec<FieldElement<FieldExtension>>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let num_parts = round_2_result.lde_composition_poly_evaluations.len();
        let z_power = z.pow(num_parts); // pole for H terms

        // Number of evaluation points per trace column (= transition_offsets.len() * step_size)
        let num_eval_points = if trace_terms_gammas.is_empty() {
            0
        } else {
            trace_terms_gammas[0].len()
        };

        // Trace poles: z_shifted[k] = primitive_root^k * z for k = 0..num_eval_points
        let mut z_shifted = Vec::with_capacity(num_eval_points);
        let mut current_z = z.clone();
        for _ in 0..num_eval_points {
            z_shifted.push(current_z.clone());
            current_z = primitive_root * &current_z;
        }

        // Number of main and aux columns in the LDE trace
        let num_main_cols = lde_trace.num_main_cols();
        let num_aux_cols = lde_trace.num_aux_cols();
        let lde_size = domain.lde_roots_of_unity_coset.len();

        // OOD evaluations
        let h_ood = &round_3_result.composition_poly_parts_ood_evaluation;
        let trace_ood_columns = round_3_result.trace_ood_evaluations.columns();
        let num_total_cols = num_main_cols + num_aux_cols;

        // Fully device-resident GPU fast path: build inv_denoms on device
        // ([z^K, z_shifted[0..]] over the full LDE coset), then run R4
        // DEEP composition reading the same device buffer. Skips the
        // CPU `inplace_batch_inverse` on the happy path; on any GPU
        // failure we fall through and compute denoms on CPU below.
        #[cfg(feature = "cuda")]
        {
            let z_scalars: Vec<FieldElement<FieldExtension>> = core::iter::once(z_power.clone())
                .chain(z_shifted.iter().cloned())
                .collect();
            if let Some((inv_dev, stream)) =
                crate::gpu_lde::try_inv_denoms_dev_with_stream::<Field, FieldExtension>(
                    &domain.lde_roots_of_unity_coset,
                    &z_scalars,
                    math_cuda::inverse::DenomSign::XMinusZ,
                    lde_trace.bound_stream(),
                )
                && let Some(deep_evals) =
                    crate::gpu_lde::try_deep_composition_gpu::<Field, FieldExtension>(
                        lde_trace,
                        lde_trace.gpu_composition_parts(),
                        &round_2_result.lde_composition_poly_evaluations,
                        h_ood,
                        &trace_ood_columns,
                        composition_poly_gammas,
                        trace_terms_gammas,
                        &[],
                        Some((&inv_dev, &stream)),
                        num_eval_points,
                    )
            {
                return deep_evals;
            }
        }

        // CPU denoms + batch inverse for the fallback paths below.
        // Single-source helper shared with the GPU parity test so any
        // sign/ordering/layout drift breaks the test instead of silently
        // diverging CUDA vs non-CUDA proofs.
        let denoms = crate::r4_denoms::build_r4_inv_denoms_cpu::<Field, FieldExtension>(
            &domain.lde_roots_of_unity_coset,
            &z_power,
            &z_shifted,
        )
        .expect("R4 inv denoms: coset points are base field, poles are extension field");

        let inv_h = &denoms[0..lde_size];

        // GPU mixed path: dev parts (when R2 keep handle exists) + host
        // inv_denoms. Used when the dev-inv-denoms path above didn't fire
        // (e.g., cudarc error in compute_denoms / scan).
        #[cfg(feature = "cuda")]
        {
            if let Some(deep_evals) =
                crate::gpu_lde::try_deep_composition_gpu::<Field, FieldExtension>(
                    lde_trace,
                    lde_trace.gpu_composition_parts(),
                    &round_2_result.lde_composition_poly_evaluations,
                    h_ood,
                    &trace_ood_columns,
                    composition_poly_gammas,
                    trace_terms_gammas,
                    &denoms,
                    None,
                    num_eval_points,
                )
            {
                return deep_evals;
            }
        }

        // Reaching here means both GPU DEEP arms fell through to the host loop
        // below, which reads the host trace (`get_main`/`get_aux`) AND the
        // host part evals. Under the device-only gate either may be empty —
        // download the resident data rather than abort; the asserts fire only
        // when a resident handle cannot serve it.
        #[cfg(feature = "cuda")]
        {
            if lde_trace.host_trace_empty() {
                let recovered = crate::gpu_lde::materialize_lde_trace_host(lde_trace);
                assert!(
                    recovered,
                    "R4 DEEP composition fell back to the host trace on a \
                     device-only table and the resident handles could not be \
                     downloaded"
                );
            }
            let parts_recovered = crate::gpu_lde::materialize_composition_parts_host(
                lde_trace,
                &mut round_2_result.lde_composition_poly_evaluations,
            );
            assert!(
                parts_recovered,
                "R4 DEEP composition fell back to the host part evals on a \
                 device-only table and the resident parts handle could not be \
                 downloaded"
            );
        }

        // OOD column compression (Plonky3-style): precompute one value per eval point,
        //   ood_compressed_k = Σ_j gamma[j][k] * ood[j][k].
        // The per-LDE-point trace column sums are NOT precomputed — they are fused
        // directly into the hot loop below. DEEP is evaluated at all 2N LDE points
        // (no stride), so every row is used.
        let mut ood_compressed: Vec<FieldElement<FieldExtension>> =
            vec![FieldElement::zero(); num_eval_points];
        for j in 0..num_total_cols {
            let ood_evals_j = &trace_ood_columns[j];
            let gammas_j = &trace_terms_gammas[j];
            for k in 0..num_eval_points {
                ood_compressed[k] += &gammas_j[k] * &ood_evals_j[k];
            }
        }

        // Fused single-pass: compute column compression AND DEEP polynomial inline.
        // Eliminates the intermediate `compressed` allocation (~400 MB for CPU table)
        // and reduces to a single rayon dispatch instead of num_eval_points + 1.
        // Each row i's column data is reused across all eval points k within a rayon
        // task, so the k=1 read hits L1 cache after k=0 just loaded it.

        // Pre-gather gamma references per eval point for cache-friendly access.
        let main_gammas_by_k: Vec<Vec<&FieldElement<FieldExtension>>> = (0..num_eval_points)
            .map(|k| {
                (0..num_main_cols)
                    .map(|j| &trace_terms_gammas[j][k])
                    .collect()
            })
            .collect();
        let aux_gammas_by_k: Vec<Vec<&FieldElement<FieldExtension>>> = (0..num_eval_points)
            .map(|k| {
                (0..num_aux_cols)
                    .map(|j| &trace_terms_gammas[num_main_cols + j][k])
                    .collect()
            })
            .collect();

        crate::par::par_map_collect(0..lde_size, |i| {
            let mut result = FieldElement::<FieldExtension>::zero();

            // H terms
            for j in 0..num_parts {
                let h_j_val = &round_2_result.lde_composition_poly_evaluations[j][i];
                let h_j_ood = &h_ood[j];
                result += &composition_poly_gammas[j] * (h_j_val - h_j_ood) * &inv_h[i];
            }

            // Trace terms: for each eval point k, compute the column sum inline
            // and multiply by the denominator inverse in one pass.
            for k in 0..num_eval_points {
                let inv_t_k_i = &denoms[(1 + k) * lde_size + i];
                let mut col_sum = FieldElement::<FieldExtension>::zero();
                for (j, gamma) in main_gammas_by_k[k].iter().enumerate() {
                    col_sum += lde_trace.get_main(i, j) * *gamma;
                }
                for (j, gamma) in aux_gammas_by_k[k].iter().enumerate() {
                    col_sum += lde_trace.get_aux(i, j) * *gamma;
                }
                result += inv_t_k_i * (col_sum - &ood_compressed[k]);
            }

            result
        })
    }

    /// Computes values and validity proofs of the evaluations of the composition polynomial parts
    /// at the domain value corresponding to the FRI query challenge `index` and its symmetric
    /// element.
    fn open_composition_poly(
        composition_poly_merkle_tree: &BatchedMerkleTree<FieldExtension>,
        lde_composition_poly_evaluations: &[Vec<FieldElement<FieldExtension>>],
        index: usize,
    ) -> PolynomialOpenings<FieldExtension>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let proof = composition_poly_merkle_tree
            .get_proof_by_pos(index)
            .expect("FRI query index in bounds");

        let lde_composition_poly_parts_evaluation: Vec<_> = lde_composition_poly_evaluations
            .iter()
            .flat_map(|part| {
                vec![
                    part[reverse_index(index * 2, part.len() as u64)].clone(),
                    part[reverse_index(index * 2 + 1, part.len() as u64)].clone(),
                ]
            })
            .collect();

        PolynomialOpenings {
            proof,
            evaluations: lde_composition_poly_parts_evaluation
                .clone()
                .into_iter()
                .step_by(2)
                .collect(),
            evaluations_sym: lde_composition_poly_parts_evaluation
                .into_iter()
                .skip(1)
                .step_by(2)
                .collect(),
        }
    }

    /// Like [`Self::open_composition_poly`] but uses a Merkle proof already
    /// gathered from the resident device composition tree
    /// ([`crate::gpu_lde::gather_proofs_dev`]) instead of walking a host tree.
    /// Row-pair leaf: one proof at position `index` authenticates both rows.
    #[cfg(feature = "cuda")]
    fn open_composition_poly_with_proof(
        proof: Proof<Commitment>,
        lde_composition_poly_evaluations: &[Vec<FieldElement<FieldExtension>>],
        index: usize,
    ) -> PolynomialOpenings<FieldExtension>
    where
        FieldElement<Field>: AsBytes + Sync + Send,
        FieldElement<FieldExtension>: AsBytes + Sync + Send,
    {
        let lde_composition_poly_parts_evaluation: Vec<_> = lde_composition_poly_evaluations
            .iter()
            .flat_map(|part| {
                vec![
                    part[reverse_index(index * 2, part.len() as u64)].clone(),
                    part[reverse_index(index * 2 + 1, part.len() as u64)].clone(),
                ]
            })
            .collect();

        PolynomialOpenings {
            proof,
            evaluations: lde_composition_poly_parts_evaluation
                .clone()
                .into_iter()
                .step_by(2)
                .collect(),
            evaluations_sym: lde_composition_poly_parts_evaluation
                .into_iter()
                .skip(1)
                .step_by(2)
                .collect(),
        }
    }

    /// Computes values and validity proofs of the evaluations of trace polynomials at
    /// the FRI query challenge `challenge` and its symmetric counterpart. The caller
    /// supplies a `gather` closure that pulls the row data from the column-major LDE
    /// storage (full main row, ranged main row, or aux row).
    fn open_polys_with<C, G>(
        domain: &Domain<Field>,
        tree: &BatchedMerkleTree<C>,
        challenge: usize,
        gather: G,
    ) -> PolynomialOpenings<C>
    where
        C: IsField,
        FieldElement<C>: AsBytes + Sync + Send,
        G: Fn(usize) -> Vec<FieldElement<C>>,
    {
        let domain_size = domain.lde_roots_of_unity_coset.len() as u64;
        // Rows `2·challenge` and `2·challenge+1` are committed together as the
        // single leaf at position `challenge`; one Merkle path authenticates both
        // the queried row and its symmetric counterpart.
        PolynomialOpenings {
            proof: tree
                .get_proof_by_pos(challenge)
                .expect("FRI query index in bounds"),
            evaluations: gather(reverse_index(challenge * 2, domain_size)),
            evaluations_sym: gather(reverse_index(challenge * 2 + 1, domain_size)),
        }
    }

    /// Like [`Self::open_polys_with`], but uses a Merkle proof already gathered
    /// from the resident device tree (see [`crate::gpu_lde::gather_proofs_dev`])
    /// instead of walking a host tree. Row-pair leaf: one proof at position
    /// `challenge` authenticates both the queried row and its symmetric
    /// counterpart. Evaluations still come from the host LDE columns via `gather`.
    #[cfg(feature = "cuda")]
    fn open_polys_with_proofs<C, G>(
        domain: &Domain<Field>,
        proof: Proof<Commitment>,
        challenge: usize,
        gather: G,
    ) -> PolynomialOpenings<C>
    where
        C: IsField,
        FieldElement<C>: AsBytes + Sync + Send,
        G: Fn(usize) -> Vec<FieldElement<C>>,
    {
        let domain_size = domain.lde_roots_of_unity_coset.len() as u64;
        PolynomialOpenings {
            proof,
            evaluations: gather(reverse_index(challenge * 2, domain_size)),
            evaluations_sym: gather(reverse_index(challenge * 2 + 1, domain_size)),
        }
    }

    /// Build a [`PolynomialOpenings`] from a device Merkle proof and a pair of
    /// row-value vectors already gathered off the resident device LDE — the
    /// fully device-sourced counterpart of [`Self::open_polys_with_proofs`],
    /// which still reads its evaluations from the host LDE.
    #[cfg(feature = "cuda")]
    fn open_polys_from_values<C: IsField>(
        proof: Proof<Commitment>,
        evaluations: Vec<FieldElement<C>>,
        evaluations_sym: Vec<FieldElement<C>>,
    ) -> PolynomialOpenings<C> {
        PolynomialOpenings {
            proof,
            evaluations,
            evaluations_sym,
        }
    }

    /// Slice out query `qi`'s even/odd row (each `ncols` field elements) from the
    /// row-major device gather `[even(q0), odd(q0), even(q1), odd(q1), ...]`.
    #[cfg(feature = "cuda")]
    fn device_row_pair<C: IsField>(
        vals: &[FieldElement<C>],
        qi: usize,
        ncols: usize,
    ) -> (Vec<FieldElement<C>>, Vec<FieldElement<C>>) {
        let even = vals[(2 * qi) * ncols..(2 * qi + 1) * ncols].to_vec();
        let odd = vals[(2 * qi + 1) * ncols..(2 * qi + 2) * ncols].to_vec();
        (even, odd)
    }

    /// Gather every query's row-pair off a device-resident LDE (a small D2H of
    /// only the queried rows), lifting the raw limbs to field elements via
    /// `convert`. Returns `None` (→ the host arms of the openings) when the
    /// gather fails or the tower is not Goldilocks; a gather failure is fatal
    /// only under device-only, where no host copy exists to fall back to. Bumps
    /// the opening-gather counter exactly when values are produced, so the
    /// counter reflects the device path actually serving the openings. One body
    /// for the main and aux arms — `what` only labels the messages.
    #[cfg(feature = "cuda")]
    fn gather_query_rows_device<C: IsField>(
        lde_trace: &LDETraceTable<Field, FieldExtension>,
        what: &str,
        gather: impl FnOnce(&std::sync::Arc<math_cuda::CudaStream>) -> math_cuda::Result<Vec<u64>>,
        convert: impl FnOnce(&[u64]) -> Option<Vec<FieldElement<C>>>,
    ) -> Option<Vec<FieldElement<C>>> {
        let stream = lde_trace
            .bound_stream()
            .expect("bound stream for device-resident row gather");
        let raw = match gather(&stream) {
            Ok(v) => v,
            Err(e) => {
                assert!(
                    !lde_trace.host_trace_empty(),
                    "device {what}-row gather failed and the trace is device-only \
                     (no host fallback): {e:?}"
                );
                return None;
            }
        };
        let vals = convert(&raw);
        if vals.is_some() {
            crate::gpu_lde::GPU_OPENING_GATHER_CALLS
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        vals
    }

    /// One query's trace-poly opening with the device-resident fast paths:
    /// device Merkle proof + device-gathered values when both are present, the
    /// device proof with a host gather when only the tree is resident, and the
    /// full host walk otherwise. One body for the main, aux and preprocessed
    /// multiplicity arms, so the device↔host cross-check and the R4
    /// `host_trace_empty` hard-abort guards exist exactly once. The device
    /// gather always pulls the full `ncols` row; `col_range` selects the
    /// committed subset (the full row for plain arms, `[split, ncols)` for the
    /// multiplicity subset) and must match what `gather` returns.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn open_trace_polys_device<C, G>(
        domain: &Domain<Field>,
        lde_trace: &LDETraceTable<Field, FieldExtension>,
        dev_proofs: Option<&Vec<Proof<Commitment>>>,
        dev_values: Option<&Vec<FieldElement<C>>>,
        tree: &BatchedMerkleTree<C>,
        qi: usize,
        challenge: usize,
        ncols: usize,
        col_range: std::ops::Range<usize>,
        what: &str,
        gather: G,
    ) -> PolynomialOpenings<C>
    where
        C: IsField,
        FieldElement<C>: AsBytes + Sync + Send,
        G: Fn(usize) -> Vec<FieldElement<C>>,
    {
        let Some(proofs) = dev_proofs else {
            assert!(
                !lde_trace.host_trace_empty(),
                "R4 {what} opening fell back to the host tree, but it is device-only (empty)"
            );
            // A root-only host tree means the nodes are device-resident, so a
            // broken proofs↔tree pairing must abort here. `get_proof_by_pos`
            // already refuses a root-only tree, but the panic it produces
            // downstream reads "FRI query index in bounds" — this names the
            // real cause instead.
            assert!(
                !tree.is_root_only(),
                "R4 {what} opening fell back to a root-only host tree (nodes device-resident)"
            );
            return Self::open_polys_with(domain, tree, challenge, gather);
        };
        let proof = proofs[qi].clone();
        let Some(dev_vals) = dev_values else {
            // Device tree resident but the value gather is absent: reading the
            // host gather is invalid under device-only.
            assert!(
                !lde_trace.host_trace_empty(),
                "R4 {what} opening fell back to the host gather, but it is device-only (empty)"
            );
            return Self::open_polys_with_proofs(domain, proof, challenge, gather);
        };
        let (even, odd) = Self::device_row_pair(dev_vals, qi, ncols);
        let (even, odd) = (even[col_range.clone()].to_vec(), odd[col_range].to_vec());
        // Cross-check the device gather against the host LDE. Skipped under
        // device-only (host trace empty): the gather was proven bit-identical
        // while the host copy was resident, and there is nothing to check
        // against. Release keeps query 0 as a canary (the GPU test suites run
        // --release, and gather failure modes — stride/offset/layout — are
        // systematic, so one query catches them); debug checks every query.
        if (cfg!(debug_assertions) || qi == 0) && !lde_trace.host_trace_empty() {
            let domain_size = domain.lde_roots_of_unity_coset.len() as u64;
            let r_even = reverse_index(challenge * 2, domain_size);
            let r_odd = reverse_index(challenge * 2 + 1, domain_size);
            assert_eq!(
                even,
                gather(r_even),
                "device {what}-row gather mismatch (even), query {qi}"
            );
            assert_eq!(
                odd,
                gather(r_odd),
                "device {what}-row gather mismatch (odd), query {qi}"
            );
        }
        Self::open_polys_from_values(proof, even, odd)
    }

    /// Open the deep composition polynomial on a list of indexes and their symmetric elements.
    fn open_deep_composition_poly(
        domain: &Domain<Field>,
        round_1_result: &Round1<Field, FieldExtension>,
        round_2_result: &Round2<FieldExtension>,
        indexes_to_open: &[usize],
    ) -> DeepPolynomialOpenings<Field, FieldExtension>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let mut openings = Vec::with_capacity(indexes_to_open.len());

        let lde_trace = &round_1_result.lde_trace;
        let main_commit = &round_1_result.main;
        let is_preprocessed = main_commit.is_preprocessed();
        let num_precomputed_cols = main_commit.num_precomputed_cols;
        let total_cols = lde_trace.num_main_cols();

        // Row-pair LDE positions for every query, `[even(q0), odd(q0), ...]`.
        // Each query opens the leaf at `challenge`, which pairs LDE rows
        // `reverse_index(2·challenge)` (the queried point) and
        // `reverse_index(2·challenge+1)` (its symmetric `-x` point).
        #[cfg(feature = "cuda")]
        let domain_size = domain.lde_roots_of_unity_coset.len() as u64;
        #[cfg(feature = "cuda")]
        let query_rows: Vec<u32> = indexes_to_open
            .iter()
            .flat_map(|&c| {
                [
                    reverse_index(c * 2, domain_size) as u32,
                    reverse_index(c * 2 + 1, domain_size) as u32,
                ]
            })
            .collect();

        // R4 trace proofs from the resident device trees, gathered in one batch
        // over all query positions instead of walking the host trees (byte
        // identical to the host proofs, guarded by the `merkle_gather` test).
        // `*_dev_proofs` is `Some` exactly when the tree is device resident (so
        // the host tree is a root only placeholder). In that case the gather
        // must succeed: there is no host tree to fall back to, so a gather error
        // is a hard abort. When the tree is not device resident the value is
        // `None` and the openings below walk the full host tree.
        // For preprocessed tables the resident tree is the multiplicity subset
        // tree (the host `main_commit.tree` is root only); values come from the
        // same device row gather as plain tables, sliced per subset below.
        #[cfg(feature = "cuda")]
        let main_dev_proofs: Option<Vec<Proof<Commitment>>> = lde_trace
            .gpu_main()
            .and_then(|h| h.tree.as_ref())
            .map(|tree| {
                let stream = lde_trace
                    .bound_stream()
                    .expect("bound stream for device-resident main-tree opening");
                // Row-pair leaves: one proof per query at position `challenge`.
                crate::gpu_lde::gather_proofs_dev(tree, indexes_to_open, &stream)
                    .expect("device main-tree gather failed; resident tree has no host fallback")
            });

        // Same for the aux trace tree, when it is device resident.
        #[cfg(feature = "cuda")]
        let aux_dev_proofs: Option<Vec<Proof<Commitment>>> = round_1_result
            .aux
            .as_ref()
            .and_then(|_aux| lde_trace.gpu_aux().and_then(|h| h.tree.as_ref()))
            .map(|tree| {
                let stream = lde_trace
                    .bound_stream()
                    .expect("bound stream for device-resident aux-tree opening");
                // Row-pair leaves: one proof per query at position `challenge`.
                crate::gpu_lde::gather_proofs_dev(tree, indexes_to_open, &stream)
                    .expect("device aux-tree gather failed; resident tree has no host fallback")
            });

        // Composition tree: openings open a single position `index` (row pair
        // leaf), so gather one proof per query challenge from the device tree.
        #[cfg(feature = "cuda")]
        let comp_dev_proofs: Option<Vec<Proof<Commitment>>> =
            round_2_result.gpu_composition_tree.as_ref().map(|tree| {
                let stream = lde_trace
                    .bound_stream()
                    .expect("bound stream for device-resident composition-tree opening");
                crate::gpu_lde::gather_proofs_dev(tree, indexes_to_open, &stream).expect(
                    "device composition-tree gather failed; resident tree has no host fallback",
                )
            });

        // Full-residency Stage 2: gather each query's row-pair straight off the
        // resident device LDE (a small D2H of only the queried rows), instead of
        // indexing the full host LDE trace. The host trace is still resident this
        // stage and every device row is cross-checked against it in the loop
        // below; Stage 3 drops the host copy once this path is proven. `None`
        // when the LDE is not device resident or the tower is not Goldilocks (→
        // host gather). Row-major: `q`-th row's columns at `[q*ncols ..]`.
        // Gate the value gathers on the corresponding device-tree proofs so the
        // two device arms stay aligned (`*_dev_proofs.is_some() ⇔
        // *_dev_values.is_some()` on the Goldilocks path) and we never gather
        // rows for a tree that is not device resident.
        #[cfg(feature = "cuda")]
        let main_dev_values: Option<Vec<FieldElement<Field>>> =
            main_dev_proofs.as_ref().and_then(|_| {
                lde_trace.gpu_main().and_then(|h| {
                    Self::gather_query_rows_device(
                        lde_trace,
                        "main",
                        |stream| {
                            math_cuda::barycentric::gather_rows_base_on_device(
                                h,
                                &query_rows,
                                stream,
                            )
                        },
                        |raw| crate::constraint_ir::gpu_interp::base_u64_to_field::<Field>(raw),
                    )
                })
            });

        #[cfg(feature = "cuda")]
        let aux_dev_values: Option<Vec<FieldElement<FieldExtension>>> =
            aux_dev_proofs.as_ref().and_then(|_| {
                lde_trace.gpu_aux().and_then(|h| {
                    Self::gather_query_rows_device(
                        lde_trace,
                        "aux",
                        |stream| {
                            math_cuda::barycentric::gather_rows_ext3_on_device(
                                h,
                                &query_rows,
                                stream,
                            )
                        },
                        |raw| {
                            crate::constraint_ir::gpu_interp::ext3_u64_to_field::<FieldExtension>(
                                raw,
                            )
                        },
                    )
                })
            });

        // Composition part values off the resident R2 parts handle (one ext3
        // "column" per part), same row-pair gather as main/aux above.
        #[cfg(feature = "cuda")]
        let comp_num_parts = lde_trace
            .gpu_composition_parts()
            .map(|h| h.m)
            .unwrap_or_else(|| round_2_result.lde_composition_poly_evaluations.len());
        #[cfg(feature = "cuda")]
        let comp_dev_values: Option<Vec<FieldElement<FieldExtension>>> =
            comp_dev_proofs.as_ref().and_then(|_| {
                lde_trace.gpu_composition_parts().and_then(|h| {
                    Self::gather_query_rows_device(
                        lde_trace,
                        "composition",
                        |stream| {
                            math_cuda::barycentric::gather_rows_ext3_on_device(
                                h,
                                &query_rows,
                                stream,
                            )
                        },
                        |raw| {
                            crate::constraint_ir::gpu_interp::ext3_u64_to_field::<FieldExtension>(
                                raw,
                            )
                        },
                    )
                })
            });

        for (qi, index) in indexes_to_open.iter().enumerate() {
            #[cfg(not(feature = "cuda"))]
            let _ = qi;
            // For preprocessed tables, open the main split (multiplicities only);
            // for normal tables, open all main columns.
            let main_trace_opening = if is_preprocessed {
                // Multiplicity subset: same device fast paths as the plain
                // arm, sliced to the committed `[split, total)` column range.
                #[cfg(feature = "cuda")]
                {
                    Self::open_trace_polys_device(
                        domain,
                        lde_trace,
                        main_dev_proofs.as_ref(),
                        main_dev_values.as_ref(),
                        &main_commit.tree,
                        qi,
                        *index,
                        total_cols,
                        num_precomputed_cols..total_cols,
                        "multiplicity",
                        |row| {
                            lde_trace.gather_main_row_range(row, num_precomputed_cols, total_cols)
                        },
                    )
                }
                #[cfg(not(feature = "cuda"))]
                Self::open_polys_with(domain, &main_commit.tree, *index, |row| {
                    lde_trace.gather_main_row_range(row, num_precomputed_cols, total_cols)
                })
            } else {
                #[cfg(feature = "cuda")]
                {
                    Self::open_trace_polys_device(
                        domain,
                        lde_trace,
                        main_dev_proofs.as_ref(),
                        main_dev_values.as_ref(),
                        &main_commit.tree,
                        qi,
                        *index,
                        total_cols,
                        0..total_cols,
                        "main",
                        |row| lde_trace.gather_main_row(row),
                    )
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Self::open_polys_with(domain, &main_commit.tree, *index, |row| {
                        lde_trace.gather_main_row(row)
                    })
                }
            };

            // For preprocessed tables, also open the precomputed-columns tree.
            // The tree is always a full host tree (process-wide cache), so the
            // Merkle path comes from the host walk; the VALUES come from the
            // device row gather when the LDE is resident (sliced to the
            // `[0, split)` range), host range gather otherwise.
            let precomputed_trace_opening = main_commit.precomputed_tree.as_ref().map(|tree| {
                #[cfg(feature = "cuda")]
                {
                    match main_dev_values.as_ref() {
                        Some(vals) => {
                            let (even, odd) = Self::device_row_pair(vals, qi, total_cols);
                            let (even, odd) = (
                                even[..num_precomputed_cols].to_vec(),
                                odd[..num_precomputed_cols].to_vec(),
                            );
                            // Query 0 stays a release canary, same rationale
                            // as `open_trace_polys_device`.
                            if (cfg!(debug_assertions) || qi == 0) && !lde_trace.host_trace_empty()
                            {
                                let r_even = reverse_index(*index * 2, domain_size);
                                let r_odd = reverse_index(*index * 2 + 1, domain_size);
                                assert_eq!(
                                    even,
                                    lde_trace.gather_main_row_range(
                                        r_even,
                                        0,
                                        num_precomputed_cols
                                    ),
                                    "device precomputed-row gather mismatch (even), query {qi}"
                                );
                                assert_eq!(
                                    odd,
                                    lde_trace.gather_main_row_range(r_odd, 0, num_precomputed_cols),
                                    "device precomputed-row gather mismatch (odd), query {qi}"
                                );
                            }
                            Self::open_polys_from_values(
                                tree.get_proof_by_pos(*index)
                                    .expect("FRI query index in bounds"),
                                even,
                                odd,
                            )
                        }
                        None => {
                            assert!(
                                !lde_trace.host_trace_empty(),
                                "R4 precomputed opening fell back to the host gather, \
                                 but it is device-only (empty)"
                            );
                            Self::open_polys_with(domain, tree, *index, |row| {
                                lde_trace.gather_main_row_range(row, 0, num_precomputed_cols)
                            })
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                Self::open_polys_with(domain, tree, *index, |row| {
                    lde_trace.gather_main_row_range(row, 0, num_precomputed_cols)
                })
            });

            let composition_openings = {
                #[cfg(feature = "cuda")]
                {
                    match (&comp_dev_proofs, &comp_dev_values) {
                        (Some(proofs), Some(vals)) => {
                            let (even, odd) = Self::device_row_pair(vals, qi, comp_num_parts);
                            // Cross-check against the host part evals while
                            // they are still resident (absent under full
                            // residency, where the gather is the only source).
                            // Query 0 stays a release canary, same rationale
                            // as `open_trace_polys_device`.
                            if (cfg!(debug_assertions) || qi == 0)
                                && round_2_result
                                    .lde_composition_poly_evaluations
                                    .first()
                                    .is_some_and(|p| !p.is_empty())
                            {
                                let expected = Self::open_composition_poly_with_proof(
                                    proofs[qi].clone(),
                                    &round_2_result.lde_composition_poly_evaluations,
                                    *index,
                                );
                                assert_eq!(
                                    even, expected.evaluations,
                                    "device composition-row gather mismatch (even), query {qi}"
                                );
                                assert_eq!(
                                    odd, expected.evaluations_sym,
                                    "device composition-row gather mismatch (odd), query {qi}"
                                );
                            }
                            PolynomialOpenings {
                                proof: proofs[qi].clone(),
                                evaluations: even,
                                evaluations_sym: odd,
                            }
                        }
                        (Some(proofs), None) => {
                            assert!(
                                round_2_result
                                    .lde_composition_poly_evaluations
                                    .first()
                                    .is_none_or(|p| !p.is_empty()),
                                "R4 composition opening fell back to the host part evals, \
                                 but they are device-only (empty)"
                            );
                            Self::open_composition_poly_with_proof(
                                proofs[qi].clone(),
                                &round_2_result.lde_composition_poly_evaluations,
                                *index,
                            )
                        }
                        _ => Self::open_composition_poly(
                            &round_2_result.composition_poly_merkle_tree,
                            &round_2_result.lde_composition_poly_evaluations,
                            *index,
                        ),
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Self::open_composition_poly(
                        &round_2_result.composition_poly_merkle_tree,
                        &round_2_result.lde_composition_poly_evaluations,
                        *index,
                    )
                }
            };

            let aux_trace_polys = round_1_result.aux.as_ref().map(|aux| {
                #[cfg(feature = "cuda")]
                {
                    Self::open_trace_polys_device(
                        domain,
                        lde_trace,
                        aux_dev_proofs.as_ref(),
                        aux_dev_values.as_ref(),
                        &aux.tree,
                        qi,
                        *index,
                        lde_trace.num_aux_cols(),
                        0..lde_trace.num_aux_cols(),
                        "aux",
                        |row| lde_trace.gather_aux_row(row),
                    )
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Self::open_polys_with(domain, &aux.tree, *index, |row| {
                        lde_trace.gather_aux_row(row)
                    })
                }
            });

            openings.push(DeepPolynomialOpening {
                composition_poly: composition_openings,
                main_trace_polys: main_trace_opening,
                precomputed_trace_polys: precomputed_trace_opening,
                aux_trace_polys,
            });
        }

        openings
    }

    // TODO: propagate errors instead of unwrap() in commit_main_trace, reconstruct_round1, and expand_columns_to_lde
    /// Generates STARK proofs for one or more AIRs with a shared transcript.
    ///
    /// # Multi-Table Proving with LogUp
    ///
    /// When proving multiple tables that communicate via LogUp (lookup arguments),
    /// all tables must use the **same** random challenges (z, α) for the LogUp bus
    /// to balance correctly. This function ensures challenge sharing by:
    ///
    /// 1. **Commit all main traces**: All main trace commitments go into the
    ///    transcript before any challenges are sampled.
    /// 2. **Sample shared LogUp challenges**: The challenges (z, α) are sampled
    ///    once from the transcript and shared by all AIRs.
    /// 3. **Build auxiliary traces**: Each AIR builds its LogUp running-sum
    ///    columns using the shared challenges.
    /// 4. **Rounds 2-4**: Standard STARK protocol rounds for each AIR.
    ///
    /// # Warning
    ///
    /// The transcript must be safely initialized before passing it to this method.
    fn multi_prove(
        #[allow(unused_mut)] mut air_trace_pairs: Vec<AirTracePair<'_, Field, FieldExtension, PI>>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone + Send),
        #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
    ) -> Result<MultiProof<Field, FieldExtension, PI>, ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        PI: Send + Sync + Clone,
        Field: Copy + 'static,
        FieldExtension: Copy + 'static,
        <Field as IsField>::BaseType: SpillSafe,
        <FieldExtension as IsField>::BaseType: SpillSafe,
    {
        info!("Started proof generation...");

        #[cfg(feature = "instruments")]
        crate::instruments::reset_all();
        #[cfg(feature = "instruments")]
        let mut heap_snaps: Vec<crate::instruments::HeapSnapshot> = Vec::new();

        let num_airs = air_trace_pairs.len();

        // Check if any AIR has an auxiliary trace
        let needs_lookup_challenges = air_trace_pairs
            .iter()
            .any(|(air, _, _)| air.has_aux_trace());

        // =====================================================================
        // Pre-pass: compute domains and twiddles
        // =====================================================================

        #[cfg(feature = "instruments")]
        let phase_start = Instant::now();
        #[cfg(feature = "instruments")]
        let __sp = crate::instruments::span("r1_prepass");

        let mut domains = Vec::with_capacity(num_airs);
        let mut twiddle_caches: Vec<Arc<LdeTwiddles<Field>>> = Vec::with_capacity(num_airs);

        for (air, trace, _pub_inputs) in &*air_trace_pairs {
            let (domain, twiddles) = domain_and_twiddles(*air, trace.num_rows());
            domains.push(domain);
            twiddle_caches.push(twiddles);
        }

        let k = table_parallelism(num_airs);

        // VRAM budgeted admission. The budget caps the summed device working set
        // of the tables proved concurrently so large blocks don't exhaust VRAM.
        // It is an extra ceiling on top of `k` (it never raises concurrency). On
        // non-cuda builds, or when the budget can't be queried, it is `u64::MAX`
        // and the gate is inert — concurrency is then bounded by `k` alone.
        #[cfg(feature = "cuda")]
        let vram_budget = math_cuda::device::backend()
            .map(|b| b.vram_budget_bytes())
            .unwrap_or(u64::MAX);
        #[cfg(not(feature = "cuda"))]
        let vram_budget = u64::MAX;

        // NOTE: an earlier revision published prove-wide pinned-staging size
        // hints here so worker slabs allocated once at final size. Measured on
        // a 5090 it BACKFIRED: every worker slot then pays a max-size
        // cuMemHostAlloc (~160ms avg, 7.5s total vs 4.2s of ladder churn), and
        // those allocations convoy the driver lock. The mechanism was removed;
        // don't re-add pre-sizing without a shared-slab design that bounds the
        // number of allocations.

        let vram_gate = VramGate::new(vram_budget);

        // R1 main commit: only the main LDE and its Merkle scratch are resident,
        // so the aux columns add nothing to this phase's working set.
        let main_estimates: Vec<u64> = air_trace_pairs
            .iter()
            .enumerate()
            .map(|(idx, (_, trace, _))| {
                let lde_size = domains[idx].interpolation_domain_size * domains[idx].blowup_factor;
                estimate_table_vram_bytes(trace.num_main_columns, 0, lde_size)
            })
            .collect();

        // Spill main traces to mmap before Round 1 LDE.
        #[cfg(feature = "disk-spill")]
        if storage_mode == StorageMode::Disk {
            crate::par::par_try_for_each_mut(&mut air_trace_pairs, |(_, trace, _)| {
                trace
                    .main_table
                    .spill_to_disk()
                    .map_err(|e| ProvingError::DiskSpill(format!("early main: {e}")))
            })?;
        }

        #[cfg(feature = "instruments")]
        drop(__sp);
        #[cfg(feature = "instruments")]
        let prepass_elapsed = phase_start.elapsed();
        #[cfg(feature = "instruments")]
        if let Some(s) = crate::instruments::snap("After pool alloc") {
            heap_snaps.push(s);
        }

        // =====================================================================
        // Round 1: Commit all main traces (VRAM-admitted, up to K concurrent)
        // =====================================================================
        // All main trace commitments must be in the transcript before sampling
        // LogUp challenges.

        #[cfg(feature = "instruments")]
        let phase_start = Instant::now();
        #[cfg(feature = "instruments")]
        let __sp = crate::instruments::span("r1_main_commit");

        let mut main_commits: Vec<TableCommit<Field>> = Vec::with_capacity(num_airs);
        let mut main_ldes: Vec<(Vec<FieldElement<Field>>, usize)> = Vec::with_capacity(num_airs);
        // Optional device-side LDE handle per table, populated only when the
        // R1 fused GPU pipeline produced one. Pairing is by index: this vector
        // is moved into the per-table `gpu_main_cells` mutex slots below, and
        // each driver only ever touches `gpu_main_cells[idx]` for its own
        // table. (It used to ride a zip chain through the old phase D.)
        #[cfg(feature = "cuda")]
        let mut main_gpu_handles: Vec<Option<math_cuda::lde::GpuLdeBase>> =
            Vec::with_capacity(num_airs);

        // All main commits with continuous VRAM admission (no chunk barriers);
        // the transcript only needs the roots absorbed in index order, done
        // sequentially below once every commit completed — the one ordering
        // Fiat-Shamir requires before sampling the shared challenges.
        let main_results = run_admitted(
            &heaviest_first(&main_estimates),
            &main_estimates,
            &vram_gate,
            k,
            |idx| {
                let (air, trace, _) = &air_trace_pairs[idx];
                let domain = &domains[idx];
                let twiddles = &twiddle_caches[idx];

                let precomputed = air
                    .is_preprocessed()
                    .then(|| (air.precomputed_commitment(), air.num_precomputed_columns()));

                // Stage-3 device-only gate: when it holds, `commit_main_trace`
                // keeps the R1 LDE device-resident and skips the host D2H.
                #[cfg(feature = "cuda")]
                let device_only = Self::device_only_for(*air, domain);

                Self::commit_main_trace(
                    *trace,
                    domain,
                    twiddles,
                    precomputed,
                    #[cfg(feature = "cuda")]
                    device_only,
                    #[cfg(feature = "disk-spill")]
                    storage_mode,
                )
            },
        );
        for result in main_results {
            let result = result.expect("run_admitted fills every slot");
            #[cfg(feature = "cuda")]
            let (commit, cached_main, gpu_main) = result?;
            #[cfg(not(feature = "cuda"))]
            let (commit, cached_main) = result?;
            if let Some(ref pre_root) = commit.precomputed_root {
                transcript.append_bytes(pre_root);
            }
            transcript.append_bytes(&commit.root);
            main_commits.push(commit);
            main_ldes.push(cached_main);
            #[cfg(feature = "cuda")]
            main_gpu_handles.push(gpu_main);
        }

        #[cfg(feature = "instruments")]
        drop(__sp);
        #[cfg(feature = "instruments")]
        let main_commits_elapsed = phase_start.elapsed();
        #[cfg(feature = "instruments")]
        if let Some(s) = crate::instruments::snap("After main commits") {
            heap_snaps.push(s);
        }

        // =====================================================================
        // Round 1: Sample shared LogUp challenges
        // =====================================================================

        let lookup_challenges: Vec<FieldElement<FieldExtension>> = if needs_lookup_challenges {
            (0..LOGUP_NUM_CHALLENGES)
                .map(|_| transcript.sample_field_element())
                .collect()
        } else {
            Vec::new()
        };

        // =====================================================================
        // Aux build + aux commit + Rounds 2-4: fused per table
        // =====================================================================
        // Each table gets an independent transcript fork (cloned from the shared
        // state after the LogUp challenges, domain-separated by table index).
        // This matches the verifier's forking and makes per-table proving
        // independent.
        //
        // Aux build, aux commit and rounds 2-4 run FUSED per table below (one
        // driver chains all three for its table, so tables never wait on a
        // phase barrier); only this sequential prep runs here.

        // Disk-spill needs the aux columns in the host trace to spill them, so
        // disable the GPU-resident aux build (it would keep them device-only).
        #[cfg(all(feature = "cuda", feature = "disk-spill"))]
        if storage_mode == StorageMode::Disk {
            for (_, trace, _) in air_trace_pairs.iter_mut() {
                trace.set_resident_aux_ok(false);
            }
        }

        // Thread each table's device-resident trace-domain main columns (kept by
        // the R1 main LDE) onto its trace so the LogUp aux fingerprint kernel
        // reads them in place instead of re-uploading ~3 GB. Preprocessed tables
        // also carry a handle with `trace_dev` (the split-tree path); only
        // CPU-LDE tables fall back to the host upload path.
        #[cfg(all(feature = "cuda", not(feature = "debug-checks")))]
        for ((_, trace, _), gpu_main) in air_trace_pairs.iter_mut().zip(main_gpu_handles.iter()) {
            if let Some(handle) = gpu_main
                && let Some(td) = &handle.trace_dev
            {
                trace.set_main_trace_dev(std::sync::Arc::clone(td), handle.trace_rows);
            }
        }

        // Pre-fork all transcripts (cheap, sequential — must match verifier ordering)
        let table_transcripts: Vec<_> = (0..num_airs)
            .map(|idx| {
                let mut t = transcript.clone();
                if num_airs > 1 {
                    t.append_bytes(&(idx as u64).to_le_bytes());
                }
                t
            })
            .collect();

        // The aux stage of the fused chain returns a cfg-gated AuxResult. Under
        // cuda it carries the optional ext3 GPU LDE handle as a third element,
        // so the handle stays inside its own table's task and never needs a
        // separate handle vector.
        #[cfg(feature = "cuda")]
        type AuxResult<FE> = (
            Option<TableCommit<FE>>,
            (Vec<FieldElement<FE>>, usize),
            Option<math_cuda::lde::GpuLdeExt3>,
        );
        #[cfg(not(feature = "cuda"))]
        type AuxResult<FE> = (Option<TableCommit<FE>>, (Vec<FieldElement<FE>>, usize));
        // R1 aux commit and rounds 2 to 4 share the peak working set: the main
        // and aux LDEs are co-resident, plus the composition and Merkle
        // transients (in the scratch factor). The aux width comes from the AIR
        // layout (the aux build itself runs inside the admitted chain below).
        let peak_estimates: Vec<u64> = air_trace_pairs
            .iter()
            .enumerate()
            .map(|(idx, (air, trace, _))| {
                let lde_size = domains[idx].interpolation_domain_size * domains[idx].blowup_factor;
                let (_, aux_cols) = air.trace_layout();
                estimate_table_vram_bytes(trace.num_main_columns, aux_cols, lde_size)
            })
            .collect();

        // Per-table slots for the fused chain: each driver takes or locks only
        // its own index, so every mutex is uncontended by construction.
        let pair_cells: Vec<std::sync::Mutex<AirTracePair<'_, Field, FieldExtension, PI>>> =
            air_trace_pairs
                .into_iter()
                .map(std::sync::Mutex::new)
                .collect();
        let main_commit_cells: Vec<std::sync::Mutex<Option<TableCommit<Field>>>> = main_commits
            .into_iter()
            .map(|c| std::sync::Mutex::new(Some(c)))
            .collect();
        #[allow(clippy::type_complexity)]
        let main_lde_cells: Vec<
            std::sync::Mutex<Option<(Vec<FieldElement<Field>>, usize)>>,
        > = main_ldes
            .into_iter()
            .map(|l| std::sync::Mutex::new(Some(l)))
            .collect();
        #[cfg(feature = "cuda")]
        let gpu_main_cells: Vec<std::sync::Mutex<Option<math_cuda::lde::GpuLdeBase>>> =
            main_gpu_handles
                .into_iter()
                .map(std::sync::Mutex::new)
                .collect();
        let transcript_cells: Vec<_> = table_transcripts
            .into_iter()
            .map(std::sync::Mutex::new)
            .collect();
        #[cfg(feature = "instruments")]
        #[allow(clippy::type_complexity)]
        let table_timings_mx: std::sync::Mutex<
            Vec<(String, usize, Duration, crate::instruments::TableSubOps)>,
        > = std::sync::Mutex::new(Vec::new());

        // Fused chain, stage 1: aux build → aux commit → aux root into the
        // table's transcript fork → Round1 assembly.
        #[allow(clippy::type_complexity)]
        let aux_stage = |idx: usize| -> Result<
            (
                Round1Commitments<Field, FieldExtension>,
                Lde<Field, FieldExtension>,
            ),
            ProvingError,
        > {
            let mut pair = pair_cells[idx].lock().unwrap();
            let (air, trace, _) = &mut *pair;
            let domain = &domains[idx];
            let twiddles = &twiddle_caches[idx];

            #[cfg(feature = "instruments")]
            let __sp = crate::instruments::span("r1_aux_build_table");
            let bus_public_inputs = if air.has_aux_trace() {
                air.build_auxiliary_trace(*trace, &lookup_challenges)
            } else {
                None
            };
            // The trace-domain snapshot retained by the R1 main LDE has exactly
            // one consumer — the aux build above. Reclaim it before this
            // table's aux-commit + DEEP/FRI VRAM peak.
            #[cfg(feature = "cuda")]
            {
                trace.clear_main_trace_dev();
                trace.clear_main_rowmajor_dev();
                if let Some(handle) = gpu_main_cells[idx].lock().unwrap().as_mut() {
                    handle.trace_dev = None;
                    handle.trace_rows = 0;
                }
            }
            #[cfg(feature = "disk-spill")]
            if storage_mode == StorageMode::Disk && air.has_aux_trace() {
                trace
                    .spill_aux_to_disk()
                    .map_err(|e| ProvingError::DiskSpill(format!("aux trace: {e}")))?;
            }
            #[cfg(feature = "instruments")]
            drop(__sp);

            #[cfg(feature = "instruments")]
            let __sp = crate::instruments::span("r1_aux_commit_table");
            let aux_full: AuxResult<FieldExtension> =
                (|| -> Result<AuxResult<FieldExtension>, ProvingError> {
                    if air.has_aux_trace() {
                        let lde_size = domain.interpolation_domain_size * domain.blowup_factor;

                        // Device-only for the aux commit: the main commit's
                        // gate AND a produced main device handle. The aux side
                        // may be MORE conservative than main (never less) — if
                        // the GPU main commit declined and fell back to CPU,
                        // skipping the aux D2H here would leave a device-only
                        // trace with no main handle to serve it.
                        #[cfg(feature = "cuda")]
                        let mut device_only = Self::device_only_for(*air, domain)
                            && gpu_main_cells[idx].lock().unwrap().is_some();

                        // Resident GPU path: aux columns already on device (from
                        // the resident LogUp aux build) — LDE straight from device
                        // memory, no upload, no host column extraction. When the
                        // resident build fired the host aux trace is empty, so a
                        // device LDE failure downloads the resident aux trace and
                        // continues on the host arms below (falling through as-is
                        // would commit a zero aux trace).
                        #[cfg(feature = "cuda")]
                        if trace.aux_resident().is_some() {
                            #[cfg(feature = "instruments")]
                            let t_sub = Instant::now();
                            let num_cols = trace.aux_resident().map_or(0, |ra| ra.num_aux_cols);
                            let expand = |ra: &math_cuda::logup::ResidentAux| {
                                crate::gpu_lde::try_expand_leaf_and_tree_ext3_row_major_keep_dev::<
                                    Field,
                                    FieldExtension,
                                    BatchedMerkleTreeBackend<FieldExtension>,
                                >(
                                    ra,
                                    domain.blowup_factor,
                                    &twiddles.coset_weights,
                                    !device_only,
                                )
                            };
                            let mut expanded = expand(trace.aux_resident().expect("checked above"));
                            if expanded.is_none()
                                && let Ok(be) = math_cuda::device::backend()
                                && be.ctx.synchronize().is_ok()
                            {
                                // The decline is usually transient VRAM
                                // pressure from concurrent tables; a device
                                // drain releases those peaks, so one retry
                                // tends to keep the table fully resident
                                // instead of paying the host downgrade.
                                crate::gpu_lde::GPU_RESIDENT_AUX_RETRIES
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                eprintln!(
                                    "[gpu] resident aux LDE declined: table={} \
                                     (retrying after device drain)",
                                    air.name(),
                                );
                                expanded = expand(trace.aux_resident().expect("checked above"));
                            }
                            if let Some((tree, handle, aux_data)) = expanded {
                                #[cfg(feature = "instruments")]
                                crate::instruments::accum_r1_aux(t_sub.elapsed(), Duration::ZERO);
                                let root = tree.root;
                                return Ok((
                                    Some(TableCommit::plain(tree, root)),
                                    (aux_data, num_cols),
                                    Some(handle),
                                ));
                            }
                            // The device aux LDE declined at runtime (transient
                            // VRAM pressure, usually) and there is no host aux
                            // trace to fall back to. Same class as the R2
                            // downgrade: download the resident aux trace — and
                            // the main LDE if this table was device-only — and
                            // continue fully host-backed on the arms below.
                            let mut recovered = crate::gpu_lde::materialize_aux_trace_host(*trace);
                            // Once the aux download lands, the host aux trace is
                            // populated: a later failure is the main-LDE
                            // download's, and the error has to name that step
                            // instead of claiming an empty aux trace.
                            let aux_recovered = recovered;
                            if recovered && device_only {
                                let mut cell = main_lde_cells[idx].lock().unwrap();
                                if let Some((data, _)) = cell.as_mut()
                                    && data.is_empty()
                                    && trace.num_main_columns > 0
                                {
                                    recovered = match (
                                        gpu_main_cells[idx].lock().unwrap().as_ref(),
                                        math_cuda::device::backend(),
                                    ) {
                                        (Some(h), Ok(be)) => {
                                            match crate::gpu_lde::download_main_lde_row_major::<Field>(
                                                h,
                                                &be.next_stream(),
                                            ) {
                                                Some(v) => {
                                                    *data = v;
                                                    true
                                                }
                                                None => false,
                                            }
                                        }
                                        _ => false,
                                    };
                                }
                            }
                            if !recovered {
                                return Err(ProvingError::Fft(
                                    if aux_recovered {
                                        "resident aux LDE declined; the aux trace was recovered \
                                         but the main-LDE download failed"
                                    } else {
                                        "resident aux LDE declined and the aux-trace download \
                                         recovery failed"
                                    }
                                    .to_string(),
                                ));
                            }
                            eprintln!(
                                "[gpu] resident-aux downgrade: table={} rows={} \
                                 (device aux LDE declined; continuing on host)",
                                air.name(),
                                trace.num_rows(),
                            );
                            device_only = false;
                        }

                        // Fused GPU path (cuda only): row-major ext3 NTT — single
                        // H2D, no column extraction, no CPU transpose.
                        #[cfg(feature = "cuda")]
                        {
                            let (trace_slice, num_cols) = trace.aux_data_row_major();
                            let n = if num_cols > 0 {
                                trace_slice.len() / num_cols
                            } else {
                                0
                            };
                            #[cfg(feature = "instruments")]
                            let t_sub = Instant::now();
                            if let Some((tree, handle, aux_data)) =
                                crate::gpu_lde::try_expand_leaf_and_tree_ext3_row_major_keep::<
                                    Field,
                                    FieldExtension,
                                    BatchedMerkleTreeBackend<FieldExtension>,
                                >(
                                    trace_slice,
                                    n,
                                    num_cols,
                                    domain.blowup_factor,
                                    &twiddles.coset_weights,
                                    !device_only,
                                )
                            {
                                #[cfg(feature = "instruments")]
                                let aux_lde_dur = t_sub.elapsed();
                                let root = tree.root;
                                #[cfg(feature = "instruments")]
                                crate::instruments::accum_r1_aux(aux_lde_dur, Duration::ZERO);
                                return Ok((
                                    Some(TableCommit::plain(tree, root)),
                                    (aux_data, num_cols),
                                    Some(handle),
                                ));
                            }
                        }

                        // CPU path: copy the already-row-major aux trace directly
                        // (one memcpy — no transpose) and expand with the
                        // cache-blocked batched two-half FFT.
                        let (trace_data, total_cols) = trace.aux_data_row_major();

                        #[cfg(feature = "instruments")]
                        let t_sub = Instant::now();

                        let mut aux_data: Vec<FieldElement<FieldExtension>> =
                            Vec::with_capacity(lde_size * total_cols);
                        aux_data.extend_from_slice(trace_data);

                        #[cfg(feature = "disk-spill")]
                        if storage_mode == StorageMode::Disk {
                            trace.aux_table.advise_drop_cache();
                        }

                        Polynomial::<FieldElement<FieldExtension>>::coset_lde_full_expand_row_major::<Field>(
                            &mut aux_data,
                            total_cols,
                            domain.blowup_factor,
                            &twiddles.coset_weights,
                            &twiddles.two_half_inv,
                            &twiddles.two_half_fwd,
                        )
                        .expect("row-major aux coset LDE expansion");

                        #[cfg(feature = "instruments")]
                        let aux_lde_dur = t_sub.elapsed();
                        #[cfg(feature = "instruments")]
                        let t_sub = Instant::now();
                        #[allow(unused_mut)]
                        let (mut tree, root) =
                            Self::commit_rows_bit_reversed(&aux_data, total_cols)
                                .ok_or(ProvingError::EmptyCommitment)?;
                        #[cfg(feature = "disk-spill")]
                        Self::spill_tree(&mut tree, storage_mode, "aux Merkle tree")?;
                        let commit = TableCommit::plain(tree, root);
                        #[cfg(feature = "instruments")]
                        crate::instruments::accum_r1_aux(aux_lde_dur, t_sub.elapsed());

                        #[cfg(feature = "cuda")]
                        return Ok((Some(commit), (aux_data, total_cols), None));
                        #[cfg(not(feature = "cuda"))]
                        Ok((Some(commit), (aux_data, total_cols)))
                    } else {
                        #[cfg(feature = "cuda")]
                        return Ok((None, (Vec::new(), 0), None));
                        #[cfg(not(feature = "cuda"))]
                        Ok((None, (Vec::new(), 0)))
                    }
                })()?;
            // Tuple shape is cfg-gated; `.0` is the optional TableCommit in
            // both variants. Aux roots go to the table's OWN fork, so no
            // cross-table ordering is needed here.
            if let Some(ref c) = aux_full.0 {
                transcript_cells[idx].lock().unwrap().append_bytes(&c.root);
            }
            #[cfg(feature = "instruments")]
            drop(__sp);

            #[cfg(feature = "cuda")]
            let (aux_commit, cached_aux, gpu_aux) = aux_full;
            #[cfg(not(feature = "cuda"))]
            let (aux_commit, cached_aux) = aux_full;
            let main_commit = main_commit_cells[idx]
                .lock()
                .unwrap()
                .take()
                .expect("main commit consumed once per table");
            let main_lde = main_lde_cells[idx]
                .lock()
                .unwrap()
                .take()
                .expect("main lde consumed once per table");
            #[cfg(feature = "cuda")]
            let gpu_main = gpu_main_cells[idx].lock().unwrap().take();
            let commitment = Round1Commitments {
                main: main_commit,
                aux: aux_commit,
                rap_challenges: lookup_challenges.clone(),
                bus_public_inputs,
            };
            #[cfg(feature = "cuda")]
            let lde = Lde {
                main: main_lde,
                aux: cached_aux,
                gpu_main,
                gpu_aux,
            };
            #[cfg(not(feature = "cuda"))]
            let lde = Lde {
                main: main_lde,
                aux: cached_aux,
            };
            Ok((commitment, lde))
        };

        // Fused chain, stage 2: Round1 from the cached LDE (consumed by value,
        // no recomputation) → rounds 2-4 against the table's transcript fork.
        let rounds_stage = |idx: usize,
                            commitment: Round1Commitments<Field, FieldExtension>,
                            lde: Lde<Field, FieldExtension>|
         -> Result<StarkProof<Field, FieldExtension, PI>, ProvingError> {
            let pair = pair_cells[idx].lock().unwrap();
            let (air, trace, pub_inputs) = &*pair;
            let _ = trace; // used by instruments
            let domain = &domains[idx];

            #[cfg(feature = "instruments")]
            let __sp = crate::instruments::span("rounds_2to4_table");
            #[cfg(feature = "instruments")]
            let table_start = Instant::now();

            let mut round_1_result =
                commitment.build_round1(lde, air.step_size(), domain.blowup_factor);

            let mut tguard = transcript_cells[idx].lock().unwrap();
            if let Some(ref bpi) = round_1_result.bus_public_inputs {
                tguard.append_field_element(&bpi.table_contribution);
            }

            let proof = Self::prove_rounds_2_to_4(
                *air,
                *pub_inputs,
                &mut round_1_result,
                &mut *tguard,
                domain,
                &twiddle_caches[idx],
            )?;

            #[cfg(feature = "instruments")]
            {
                let sub_ops = crate::instruments::take_round_sub_ops().unwrap_or_default();
                table_timings_mx.lock().unwrap().push((
                    air.name().to_string(),
                    trace.num_rows(),
                    table_start.elapsed(),
                    sub_ops,
                ));
            }
            Ok(proof)
        };

        #[cfg(feature = "instruments")]
        let phase_start = Instant::now();
        // Phase-level span for the whole fused region, opened here on the
        // calling thread. The per-table spans inside it (`*_table`) are one
        // instance per table and `phase_table.py` sums same-label spans, so
        // they cannot stand in for the phase wall: their sum runs up to `k`
        // times over it. This is also the span `LAMBDA_VM_NSYS_CAPTURE_SPAN`
        // brackets, which needs exactly one instance to start/stop the
        // profiler around.
        #[cfg(feature = "instruments")]
        let __sp = crate::instruments::span("rounds_2to4");

        let peak_order = heaviest_first(&peak_estimates);

        // One fused task per table: while a heavy table works through a
        // host-bound stretch, the others' GPU stages fill the device. The
        // shared transcript is untouched past this point (each fork is
        // per-table), so any order is sound; proofs are drained in index order.
        #[cfg(not(feature = "debug-checks"))]
        let table_results = run_admitted(&peak_order, &peak_estimates, &vram_gate, k, |idx| {
            let (commitment, lde) = aux_stage(idx)?;
            rounds_stage(idx, commitment, lde)
        });

        // debug-checks needs every table's commitments and traces between the
        // aux and rounds stages (cross-table bus balance), so it splits the
        // fused chain into two admitted passes around the check.
        #[cfg(feature = "debug-checks")]
        let table_results = {
            let aux_outs = run_admitted(&peak_order, &peak_estimates, &vram_gate, k, aux_stage);
            let mut commitments = Vec::with_capacity(num_airs);
            let mut ldes = Vec::with_capacity(num_airs);
            for out in aux_outs {
                let (c, l) = out.expect("run_admitted fills every slot")?;
                commitments.push(c);
                ldes.push(l);
            }
            Self::run_debug_checks(&pair_cells, &commitments, &domains, &twiddle_caches);
            #[allow(clippy::type_complexity)]
            let staged: Vec<
                std::sync::Mutex<
                    Option<(
                        Round1Commitments<Field, FieldExtension>,
                        Lde<Field, FieldExtension>,
                    )>,
                >,
            > = commitments
                .into_iter()
                .zip(ldes)
                .map(|p| std::sync::Mutex::new(Some(p)))
                .collect();
            run_admitted(&peak_order, &peak_estimates, &vram_gate, k, |idx| {
                let (c, l) = staged[idx].lock().unwrap().take().unwrap();
                rounds_stage(idx, c, l)
            })
        };

        let mut proofs = Vec::with_capacity(num_airs);
        for result in table_results {
            proofs.push(result.expect("run_admitted fills every slot")?);
        }
        #[cfg(feature = "instruments")]
        drop(__sp);
        #[cfg(feature = "instruments")]
        let table_timings = table_timings_mx.into_inner().unwrap();
        #[cfg(feature = "instruments")]
        {
            // Store timing data for the top-level report in prove_with_options.
            // Uses a thread-local to avoid changing multi_prove's return type.
            crate::instruments::store(crate::instruments::MultiProveTiming {
                prepass: prepass_elapsed,
                main_commits: main_commits_elapsed,
                rounds_2_4: phase_start.elapsed(),
                round1_sub: crate::instruments::take_r1_sub(),
                table_timings,
                heap_snapshots: heap_snaps,
            });
        }

        Ok(MultiProof { proofs })
    }

    /// Generate a STARK proof for a single AIR/trace.
    /// This is equivalent to calling `multi_prove` with a single-element slice.
    fn prove(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        trace: &mut TraceTable<Field, FieldExtension>,
        pub_inputs: &PI,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone + Send),
    ) -> Result<StarkProof<Field, FieldExtension, PI>, ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        PI: Send + Sync + Clone,
        Field: Copy + 'static,
        FieldExtension: Copy + 'static,
        <Field as IsField>::BaseType: SpillSafe,
        <FieldExtension as IsField>::BaseType: SpillSafe,
    {
        let air_trace_pairs = vec![(air, trace, pub_inputs)];
        Self::multi_prove(
            air_trace_pairs,
            transcript,
            #[cfg(feature = "disk-spill")]
            StorageMode::Ram,
        )
        .map(|mut multi_proof| multi_proof.proofs.remove(0))
    }

    // TODO: propagate errors instead of unwrap() in open_deep_composition_poly and FRI operations
    /// Executes rounds 2-4 and generates a STARK proof for the trace `main_trace` with public inputs `pub_inputs`.
    /// Warning: the transcript must be safely initialized before passing it to this method.
    /// Diagnostic (see `gpu_lde::gpu_xcheck`): the verifier's step-2
    /// composition consistency check run in-process on the freshly computed
    /// R3 values — H(z) reconstructed from the trace OOD evaluations must
    /// match the folded parts OOD. Near-zero cost (one constraint evaluation
    /// at a single point), so it can run on every table without disturbing
    /// the timing that provokes VRAM-pressure bugs. Mirrors
    /// `step_2_verify_claimed_composition_polynomial` in `verifier.rs`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn composition_ood_consistent(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        pub_inputs: &PI,
        domain: &Domain<Field>,
        rap_challenges: &[FieldElement<FieldExtension>],
        bus_public_inputs: Option<&BusPublicInputs<FieldExtension>>,
        transition_coefficients: &[FieldElement<FieldExtension>],
        boundary_coefficients: &[FieldElement<FieldExtension>],
        z: &FieldElement<FieldExtension>,
        trace_ood: &Table<FieldExtension>,
        parts_ood: &[FieldElement<FieldExtension>],
    ) -> bool {
        use crate::lookup::{LOGUP_CHALLENGE_ALPHA, compute_alpha_powers};
        use crate::traits::TransitionEvaluationContext;

        let trace_length = domain.interpolation_domain_size;
        let boundary_constraints =
            air.boundary_constraints(pub_inputs, rap_challenges, bus_public_inputs, trace_length);
        let mut step_to_point: std::collections::HashMap<usize, FieldElement<Field>> =
            std::collections::HashMap::new();
        let boundary_points: Vec<FieldElement<Field>> = boundary_constraints
            .constraints
            .iter()
            .map(|c| {
                step_to_point
                    .entry(c.step)
                    .or_insert_with(|| domain.trace_primitive_root.pow(c.step as u64))
                    .clone()
            })
            .collect();

        let main_trace_width = air.trace_layout().0;
        let ood_row = trace_ood.get_row(0);
        let (nums, mut dens): (
            Vec<FieldElement<FieldExtension>>,
            Vec<FieldElement<FieldExtension>>,
        ) = boundary_constraints
            .constraints
            .iter()
            .zip(&boundary_points)
            .map(|(c, point)| {
                let column_idx = if c.is_aux {
                    main_trace_width + c.col
                } else {
                    c.col
                };
                (-&c.value + &ood_row[column_idx], -point + z)
            })
            .unzip();
        if FieldElement::inplace_batch_inverse(&mut dens).is_err() {
            return false;
        }
        let boundary_sum: FieldElement<FieldExtension> = nums
            .iter()
            .zip(&dens)
            .zip(boundary_coefficients)
            .map(|((num, den), beta)| num * den * beta)
            .fold(FieldElement::zero(), |acc, x| acc + x);

        let Some(num_main_trace_columns) =
            trace_ood.width.checked_sub(air.num_auxiliary_rap_columns())
        else {
            return false;
        };
        let logup_alpha_powers: Vec<FieldElement<FieldExtension>> =
            if rap_challenges.len() > LOGUP_CHALLENGE_ALPHA {
                compute_alpha_powers(
                    &rap_challenges[LOGUP_CHALLENGE_ALPHA],
                    air.max_bus_elements(),
                )
            } else {
                Vec::new()
            };
        let logup_table_offset = match bus_public_inputs {
            Some(bpi) => {
                let n = FieldElement::<Field>::from(trace_length as u64);
                match n.inv() {
                    Ok(n_inv) => n_inv * &bpi.table_contribution,
                    Err(_) => return false,
                }
            }
            None => FieldElement::zero(),
        };

        // Frame over the OOD grid, mirroring `StarkTableView::into_frame`
        // (that view carries rkyv bounds this generic context lacks).
        let step_size = air.step_size();
        debug_assert!(trace_ood.height.is_multiple_of(step_size));
        let steps: Vec<crate::table::TableView<FieldExtension, FieldExtension>> = (0..trace_ood
            .height)
            .step_by(step_size)
            .map(|initial| {
                let mut main = Vec::new();
                let mut aux = Vec::new();
                for row_idx in initial..initial + step_size {
                    let row = trace_ood.get_row(row_idx);
                    main.push(row[..num_main_trace_columns].to_vec());
                    aux.push(row[num_main_trace_columns..].to_vec());
                }
                crate::table::TableView::new(main, aux)
            })
            .collect();
        let ood_frame = crate::frame::Frame::new(steps);
        let ctx = TransitionEvaluationContext::new_verifier(
            &ood_frame,
            rap_challenges,
            &logup_alpha_powers,
            &logup_table_offset,
        );
        let transition_evals = air.compute_transition(&ctx);

        let mut denominators =
            vec![FieldElement::<FieldExtension>::zero(); air.num_transition_constraints()];
        air.constraints_meta().iter().for_each(|m| {
            denominators[m.constraint_idx] = crate::constraints::zerofier::evaluate_zerofier(
                m,
                z,
                &domain.trace_primitive_root,
                trace_length,
            );
        });
        let transition_sum = transition_evals
            .into_iter()
            .zip(transition_coefficients)
            .zip(denominators)
            .fold(FieldElement::zero(), |acc, ((eval, beta), den)| {
                acc + beta * eval * &den
            });

        let ood_evaluation = &boundary_sum + transition_sum;
        let claimed = parts_ood
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, coeff| acc * z + coeff);
        claimed == ood_evaluation
    }

    /// Diagnostic follow-up when [`Self::composition_ood_consistent`] fails:
    /// recompute each device-derived stage on host for THIS table only and
    /// report which one diverges, then panic (the proof would not verify).
    /// Runs after the corruption already happened, so the expensive host
    /// recomputes cannot mask the failure they are diagnosing.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn xcheck_post_mortem(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        pub_inputs: &PI,
        domain: &Domain<Field>,
        twiddles: &LdeTwiddles<Field>,
        round_1_result: &mut Round1<Field, FieldExtension>,
        transition_coefficients: &[FieldElement<FieldExtension>],
        boundary_coefficients: &[FieldElement<FieldExtension>],
        round_2_result: &Round2<FieldExtension>,
        round_3_result: &Round3<FieldExtension>,
        z: &FieldElement<FieldExtension>,
    ) where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
    {
        let name = air.name();
        let trace_length = domain.interpolation_domain_size;
        eprintln!("[xcheck] FAIL composition consistency: table={name} n={trace_length}");

        if round_1_result.lde_trace.host_trace_empty()
            && !crate::gpu_lde::materialize_lde_trace_host(&mut round_1_result.lde_trace)
        {
            panic!("[xcheck] table={name}: cannot materialize host trace for post-mortem");
        }

        // Stage 1: R2 parts (device H + decompose) vs full host recompute.
        let evaluator = ConstraintEvaluator::new(
            air,
            pub_inputs,
            &round_1_result.rap_challenges,
            round_1_result.bus_public_inputs.as_ref(),
            trace_length,
        );
        let host_h = evaluator.evaluate(
            air,
            &round_1_result.lde_trace,
            domain,
            transition_coefficients,
            boundary_coefficients,
            &round_1_result.rap_challenges,
        );
        // num_parts==1: `H` IS the single part (no host decompose); num_parts==2:
        // the degree-2 split. Mirrors the R2 producer so the compare is apples-to-apples.
        let number_of_parts = air.composition_poly_degree_bound(trace_length) / trace_length;
        let host_parts = if number_of_parts == 1 {
            vec![host_h]
        } else {
            Self::decompose_and_extend_d2(&host_h, domain, twiddles)
        };
        let device_parts: Option<Vec<Vec<FieldElement<FieldExtension>>>> = if round_2_result
            .lde_composition_poly_evaluations
            .first()
            .is_some_and(|p| !p.is_empty())
        {
            Some(round_2_result.lde_composition_poly_evaluations.clone())
        } else {
            round_1_result
                .lde_trace
                .gpu_composition_parts()
                .and_then(crate::gpu_lde::download_ext3_columns::<FieldExtension>)
        };
        let mut r2_verdict = "UNAVAILABLE (no device parts to compare)".to_string();
        if let Some(dev) = &device_parts {
            r2_verdict = "ok".to_string();
            'outer: for (pi, (hp, dp)) in host_parts.iter().zip(dev).enumerate() {
                if hp.len() != dp.len() {
                    r2_verdict =
                        format!("LEN MISMATCH part={pi} host={} dev={}", hp.len(), dp.len());
                    break;
                }
                for (ri, (x, y)) in hp.iter().zip(dp.iter()).enumerate() {
                    if x != y {
                        r2_verdict = format!("MISMATCH part={pi} row={ri} host={x:?} device={y:?}");
                        break 'outer;
                    }
                }
            }
        }
        eprintln!("[xcheck] table={name} R2 parts: {r2_verdict}");

        // Corruption shape: how much of each part differs, and where. A whole
        // buffer points at H itself; a contiguous chunk at one kernel pass; a
        // strided pattern at slab/component confusion.
        if let Some(dev) = &device_parts {
            for (pi, (hp, dp)) in host_parts.iter().zip(dev).enumerate() {
                if hp.len() != dp.len() {
                    continue;
                }
                let mism: Vec<usize> = hp
                    .iter()
                    .zip(dp.iter())
                    .enumerate()
                    .filter(|(_, (x, y))| x != y)
                    .map(|(i, _)| i)
                    .collect();
                if !mism.is_empty() {
                    eprintln!(
                        "[xcheck] table={name} part={pi}: {} of {} rows differ, first={} last={}",
                        mism.len(),
                        hp.len(),
                        mism[0],
                        mism[mism.len() - 1],
                    );
                }
            }
        }

        // Rerun the device R2 chain for this table now that the storm has
        // passed: a correct rerun means a transient race during the original
        // run; the same wrong values mean a persistently corrupted device
        // input (zerofiers, IR buffers, resident LDEs).
        let rerun: Option<Vec<Vec<FieldElement<FieldExtension>>>> = evaluator
            .evaluate_dev(
                air,
                &round_1_result.lde_trace,
                domain,
                transition_coefficients,
                boundary_coefficients,
                &round_1_result.rap_challenges,
            )
            .and_then(|h_dev| {
                Self::decompose_comp_h_dev(number_of_parts, &h_dev, domain, twiddles, true)
                    .map(|(parts, _handle)| parts)
            });
        let rerun_verdict = match &rerun {
            None => "device rerun declined".to_string(),
            Some(p2) if *p2 == host_parts => {
                "rerun matches HOST (transient race in the original run)".to_string()
            }
            Some(p2) if device_parts.as_ref().is_some_and(|dp| p2 == dp) => {
                "rerun matches ORIGINAL DEVICE (persistent corrupted device input)".to_string()
            }
            Some(_) => "rerun matches NEITHER".to_string(),
        };
        eprintln!("[xcheck] table={name} R2 rerun: {rerun_verdict}");

        // Stage 2: R3 trace OOD vs the host arms.
        let dc = domain.ood_constants();
        let host_ood = crate::trace::with_r3_force_host(|| {
            crate::trace::get_trace_evaluations_from_lde(
                &mut round_1_result.lde_trace,
                domain,
                z,
                &air.context().transition_offsets,
                air.step_size(),
                dc,
            )
        });
        let got = &round_3_result.trace_ood_evaluations;
        let mut r3_trace_verdict = "ok".to_string();
        if host_ood.width != got.width || host_ood.height != got.height {
            r3_trace_verdict = "SHAPE MISMATCH".to_string();
        } else {
            'outer: for r in 0..host_ood.height {
                for c in 0..host_ood.width {
                    if host_ood.get(r, c) != got.get(r, c) {
                        r3_trace_verdict = format!(
                            "MISMATCH row={r} col={c} host={:?} device={:?}",
                            host_ood.get(r, c),
                            got.get(r, c)
                        );
                        break 'outer;
                    }
                }
            }
        }
        eprintln!("[xcheck] table={name} R3 trace_ood: {r3_trace_verdict}");

        // Stage 3: R3 parts OOD vs the host arm over the HOST-recomputed parts
        // (independent of the device H), and over the device parts when
        // available (isolates barycentric vs upstream).
        let num_parts = round_3_result.composition_poly_parts_ood_evaluation.len();
        let z_power = z.pow(num_parts);
        let comp_z_pow_n = z_power.pow(trace_length);
        let comp_inv_denoms = math::polynomial::barycentric_inv_denoms(&z_power, &dc.points);
        let ood_of =
            |parts: &[Vec<FieldElement<FieldExtension>>]| -> Vec<FieldElement<FieldExtension>> {
                parts
                    .iter()
                    .map(|lde_evals| {
                        let evals: Vec<FieldElement<FieldExtension>> = (0..trace_length)
                            .map(|i| lde_evals[i * domain.blowup_factor].clone())
                            .collect();
                        math::polynomial::interpolate_coset_eval_ext_with_g_n_inv(
                            &comp_z_pow_n,
                            &dc.offset_pow_n,
                            &dc.size_inv,
                            &dc.offset_pow_n_inv,
                            &dc.points,
                            &evals,
                            &comp_inv_denoms,
                        )
                    })
                    .collect()
            };
        let host_parts_ood = ood_of(&host_parts);
        eprintln!(
            "[xcheck] table={name} R3 parts_ood: claimed={:?} host_from_host_parts={:?} host_from_device_parts={:?}",
            round_3_result.composition_poly_parts_ood_evaluation,
            host_parts_ood,
            device_parts.as_deref().map(ood_of),
        );

        eprintln!(
            "[xcheck] table={name}: composition OOD inconsistency (R2 parts: {r2_verdict}; \
             R2 rerun: {rerun_verdict}; R3 trace_ood: {r3_trace_verdict}); aborting"
        );
        // abort() and not panic!: a panicking prover thread deadlocks the
        // epoch pipeline (producer stuck in a bounded send), which would turn
        // every diagnostic catch into a hung process.
        std::process::abort();
    }

    fn prove_rounds_2_to_4(
        air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
        pub_inputs: &PI,
        round_1_result: &mut Round1<Field, FieldExtension>,
        transcript: &mut (impl IsStarkTranscript<FieldExtension, Field> + Clone),
        domain: &Domain<Field>,
        twiddles: &LdeTwiddles<Field>,
    ) -> Result<StarkProof<Field, FieldExtension, PI>, ProvingError>
    where
        FieldElement<Field>: AsBytes,
        FieldElement<FieldExtension>: AsBytes,
        PI: Send + Sync + Clone,
    {
        log::debug!("Started proof generation...");

        // ===================================
        // ==========|   Round 2   |==========
        // ===================================

        // <<<< Receive challenge: 𝛽
        let beta = transcript.sample_field_element();
        let trace_length = domain.interpolation_domain_size;
        let num_boundary_constraints = air
            .boundary_constraints(
                pub_inputs,
                &round_1_result.rap_challenges,
                round_1_result.bus_public_inputs.as_ref(),
                trace_length,
            )
            .constraints
            .len();

        let num_transition_constraints = air.context().num_transition_constraints;

        let mut coefficients: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &beta))
                .take(num_boundary_constraints + num_transition_constraints)
                .collect();

        let transition_coefficients: Vec<_> =
            coefficients.drain(..num_transition_constraints).collect();
        let boundary_coefficients = coefficients;

        let mut round_2_result = Self::round_2_compute_composition_polynomial(
            air,
            pub_inputs,
            domain,
            twiddles,
            round_1_result,
            &transition_coefficients,
            &boundary_coefficients,
        )?;

        // >>>> Send commitments: [H₁], [H₂]
        transcript.append_bytes(&round_2_result.composition_poly_root);

        // ===================================
        // ==========|   Round 3   |==========
        // ===================================

        // <<<< Receive challenge: z
        let z = transcript.sample_z_ood(
            &domain.lde_roots_of_unity_coset,
            &domain.trace_roots_of_unity,
        );

        #[cfg(feature = "instruments")]
        let t_r3 = Instant::now();
        let round_3_result = Self::round_3_evaluate_polynomials_in_out_of_domain_element(
            air,
            domain,
            round_1_result,
            &mut round_2_result,
            &z,
        );
        #[cfg(feature = "instruments")]
        let round_3_dur = t_r3.elapsed();

        // Diagnostic: verifier-equivalent composition consistency check, run
        // per table at negligible cost; on failure, per-stage host recompute
        // names where the corruption entered (then panics).
        #[cfg(feature = "cuda")]
        if crate::gpu_lde::gpu_xcheck()
            && !Self::composition_ood_consistent(
                air,
                pub_inputs,
                domain,
                &round_1_result.rap_challenges,
                round_1_result.bus_public_inputs.as_ref(),
                &transition_coefficients,
                &boundary_coefficients,
                &z,
                &round_3_result.trace_ood_evaluations,
                &round_3_result.composition_poly_parts_ood_evaluation,
            )
        {
            Self::xcheck_post_mortem(
                air,
                pub_inputs,
                domain,
                twiddles,
                round_1_result,
                &transition_coefficients,
                &boundary_coefficients,
                &round_2_result,
                &round_3_result,
                &z,
            );
        }

        // >>>> Send values: tⱼ(zgᵏ). g·z pruning: split the full OOD table into
        // the current-row block (all columns) and the pruned next-row block
        // (masked columns only), and absorb only the surviving values — the
        // verifier absorbs the identical two blocks in the same order.
        let (ood_block0, ood_block1) =
            Self::ood_layout(air).split_full(&round_3_result.trace_ood_evaluations);
        for block in [&ood_block0, &ood_block1] {
            for col in block.columns().iter() {
                for elem in col.iter() {
                    transcript.append_field_element(elem);
                }
            }
        }

        // >>>> Send values: Hᵢ(z^N)
        for element in round_3_result.composition_poly_parts_ood_evaluation.iter() {
            transcript.append_field_element(element);
        }

        // ===================================
        // ==========|   Round 4   |==========
        // ===================================

        // Part of this round is running FRI, which is an interactive
        // protocol on its own. Therefore we pass it the transcript
        // to simulate the interactions with the verifier.
        let round_4_result = Self::round_4_compute_and_run_fri_on_the_deep_composition_polynomial(
            air,
            domain,
            round_1_result,
            &mut round_2_result,
            &round_3_result,
            &z,
            transcript,
        );

        #[cfg(feature = "instruments")]
        {
            let zero = Duration::ZERO;
            let (r2_constraints, r2_fft, r2_merkle) =
                crate::instruments::take_r2_sub().unwrap_or((zero, zero, zero));
            let (r4_fft, r4_merkle, r4_deep_comp, r4_queries) =
                crate::instruments::take_r4_sub().unwrap_or((zero, zero, zero, zero));
            crate::instruments::store_round_sub_ops(crate::instruments::TableSubOps {
                constraints: r2_constraints,
                comp_decompose: r2_fft,
                comp_commit: r2_merkle,
                ood: round_3_dur,
                deep_comp: r4_deep_comp,
                deep_extend: r4_fft,
                fri_commit: r4_merkle,
                queries: r4_queries,
            });
        }

        log::debug!("End proof generation");

        Ok(StarkProof {
            // [t]
            lde_trace_main_merkle_root: round_1_result.main.root,
            // [t]
            lde_trace_aux_merkle_root: round_1_result.aux.as_ref().map(|x| x.root),
            // For preprocessed tables: commitment to precomputed columns only
            lde_trace_precomputed_merkle_root: round_1_result.main.precomputed_root,
            // tⱼ(zgᵏ): current-row block + pruned next-row block.
            trace_ood_evaluations: ood_block0,
            trace_ood_next_evaluations: ood_block1,
            // [H₁] and [H₂]
            composition_poly_root: round_2_result.composition_poly_root,
            // Hᵢ(z^N)
            composition_poly_parts_ood_evaluation: round_3_result
                .composition_poly_parts_ood_evaluation,
            // [pₖ]
            fri_layers_merkle_roots: round_4_result.fri_layers_merkle_roots,
            // FRI final polynomial coefficients
            fri_final_poly_coeffs: round_4_result.fri_final_poly_coeffs,
            // Open(p₀(D₀), 𝜐ₛ), Open(pₖ(Dₖ), −𝜐ₛ^(2ᵏ))
            query_list: round_4_result.query_list,
            // Open(H₁(D_LDE, 𝜐₀), Open(H₂(D_LDE, 𝜐₀), Open(tⱼ(D_LDE), 𝜐₀)
            // Open(H₁(D_LDE, -𝜐ᵢ), Open(H₂(D_LDE, -𝜐ᵢ), Open(tⱼ(D_LDE), -𝜐ᵢ)
            deep_poly_openings: round_4_result.deep_poly_openings,
            // nonce obtained from grinding
            nonce: round_4_result.nonce,
            // Bus interaction public inputs (for boundary constraints and bus balance check)
            bus_public_inputs: round_1_result.bus_public_inputs.clone(),
            // Public inputs for boundary constraints
            public_inputs: pub_inputs.clone(),
            trace_length: domain.interpolation_domain_size,
        })
    }
}

/// Print a global bus balance report aggregating per-bus sums across all tables.
///
/// Uses numeric bus IDs only (no VM-specific names) to keep the stark crate generic.
/// For bus ID → name mapping, see `BusId` in the prover crate.
#[cfg(feature = "debug-checks")]
fn print_bus_balance_report<FieldExtension>(
    all_bus_public_inputs: &[Option<BusPublicInputs<FieldExtension>>],
) where
    FieldExtension: IsField,
{
    use std::collections::HashMap;

    let has_logup = all_bus_public_inputs.iter().any(|r| r.is_some());
    if !has_logup {
        return;
    }

    let mut global_bus_sums: HashMap<u64, FieldElement<FieldExtension>> = HashMap::new();
    let mut bus_senders: HashMap<u64, Vec<(String, FieldElement<FieldExtension>)>> = HashMap::new();
    let mut bus_receivers: HashMap<u64, Vec<(String, FieldElement<FieldExtension>)>> =
        HashMap::new();
    let mut global_sender_sums: HashMap<u64, FieldElement<FieldExtension>> = HashMap::new();
    let mut global_receiver_sums: HashMap<u64, FieldElement<FieldExtension>> = HashMap::new();

    for bus_inputs in all_bus_public_inputs.iter().flatten() {
        for (&bus_id, sum) in &bus_inputs.per_bus_sums {
            *global_bus_sums
                .entry(bus_id)
                .or_insert(FieldElement::zero()) += sum.clone();
        }
        for (&bus_id, sum) in &bus_inputs.per_bus_sender_sums {
            *global_sender_sums
                .entry(bus_id)
                .or_insert(FieldElement::zero()) += sum.clone();
            bus_senders
                .entry(bus_id)
                .or_default()
                .push((bus_inputs.table_name.clone(), sum.clone()));
        }
        for (&bus_id, sum) in &bus_inputs.per_bus_receiver_sums {
            *global_receiver_sums
                .entry(bus_id)
                .or_insert(FieldElement::zero()) += sum.clone();
            bus_receivers
                .entry(bus_id)
                .or_default()
                .push((bus_inputs.table_name.clone(), sum.clone()));
        }
    }

    eprintln!("\n=== GLOBAL BUS BALANCE REPORT ===");
    let zero = FieldElement::<FieldExtension>::zero();
    let mut bus_ids: Vec<_> = global_bus_sums.keys().copied().collect();
    bus_ids.sort();
    for bus_id in bus_ids {
        let total = &global_bus_sums[&bus_id];
        if *total != zero {
            eprintln!("Bus {:2}: IMBALANCED", bus_id);

            if let Some(senders) = bus_senders.get(&bus_id) {
                eprintln!("  SENDERS:");
                for (table_name, sum) in senders {
                    eprintln!("    [{:12}]: {:?}", table_name, sum);
                }
                if let Some(total_sent) = global_sender_sums.get(&bus_id) {
                    eprintln!("    → Total sent: {:?}", total_sent);
                }
            }

            if let Some(receivers) = bus_receivers.get(&bus_id) {
                eprintln!("  RECEIVERS:");
                for (table_name, sum) in receivers {
                    eprintln!("    [{:12}]: {:?}", table_name, sum);
                }
                if let Some(total_recv) = global_receiver_sums.get(&bus_id) {
                    eprintln!("    → Total received: {:?}", total_recv);
                }
            }

            eprintln!("  IMBALANCE: {:?}\n", total);
        } else {
            eprintln!("Bus {:2}: BALANCED ✓", bus_id);
        }
    }
    eprintln!("=================================\n");

    {
        use crate::bus_debug::BUS_DEBUG_TRACKER;

        let tracker = BUS_DEBUG_TRACKER
            .lock()
            .expect("[BusDebugTracker] mutex poisoned — debug data may be inconsistent");
        if !tracker.is_empty() {
            if tracker.is_truncated() {
                eprintln!(
                    "[BusDebugTracker] WARNING: Log truncated at {} entries — results may be incomplete",
                    tracker.len()
                );
            }
            eprintln!(
                "[BusDebugTracker] Logged {} interactions, running analysis...",
                tracker.len()
            );
            let report = tracker.analyze_mismatches();
            report.print_summary();
        }
    }
}
