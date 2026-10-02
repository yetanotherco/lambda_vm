//! One proof per block, no epochs: the monolithic prover over a whole block at
//! uniform full-height table instances, on the device.
//!
//! The proof is the monolithic [`VmProof`], checked by the unchanged
//! [`crate::verify_with_options`]: the statement (`StatementKind::Monolithic`),
//! then every instance's main root in AIR order, one set of LogUp challenges,
//! and a transcript fork per instance for its aux, quotient and FRI. Memory
//! across the block is the monolithic PAGE argument; there is no
//! local-to-global table and no global proof.
//!
//! What makes a whole block fit the card is the residency mode, not the
//! protocol: under [`ResidencyMode::RecomputeLdeDevice`] Round 1 keeps only
//! each instance's root, and each instance's fused task commits its trace on
//! the device again (root checked equal to the absorbed one). The host keeps
//! the traces between the two phases.
//!
//! This is a measurement driver beside the epoch pipeline
//! ([`crate::continuation`]), which stays the production path.

use std::time::Instant;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use stark::prover::IsStarkProver;
use stark::residency_mode::ResidencyMode;

use crate::statement::{StatementKind, absorb_statement};
use crate::tables::MaxRowsConfig;
use crate::tables::register;
use crate::tables::trace_builder::{
    ChunkJob, DecodeArtifacts, StreamTable, StreamedChunk, Traces, WindowedTraceBuilder,
    build_initial_image,
};
use crate::test_utils::{
    VmAir, create_cpu_air, create_load_air, create_lt_air, create_memw_air,
    create_memw_aligned_air, create_memw_register_air, create_shift_air, create_store_air,
};
use crate::{AcceleratorShape, Commitment, Error, ProofOptions, VmAirs, VmProof};

/// A helper thread's result, its panic re-raised on the caller.
fn join<T>(handle: std::thread::ScopedJoinHandle<'_, T>) -> T {
    handle
        .join()
        .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
}

/// Rows per full-height instance: every splittable table is cut into
/// instances of `2^BLOCK_ROWS_LOG2` rows (the tail padded to its power of two).
pub const BLOCK_ROWS_LOG2: u32 = 21;

/// KECCAK_RND rows per instance: the block format's cap
/// ([`crate::BLOCK_KECCAK_RND_MAX_ROWS`], 2^16), so the batching phase keeps
/// its 128 bits (D-NOEPOCH §6) and no instance outweighs the VRAM gate.
pub const BLOCK_KECCAK_RND_ROWS_LOG2: u32 = crate::BLOCK_KECCAK_RND_MAX_ROWS.trailing_zeros();

/// ECDAS rows per instance: the block format's cap
/// ([`crate::BLOCK_ECDAS_MAX_ROWS`], 2^17).
pub const BLOCK_ECDAS_ROWS_LOG2: u32 = crate::BLOCK_ECDAS_MAX_ROWS.trailing_zeros();

/// `LAMBDA_VM_BLOCK_KECCAK_RND_LOG2`: `5..=16` sets KECCAK_RND's chunk height;
/// unset is [`BLOCK_KECCAK_RND_ROWS_LOG2`]. Anything else aborts: the block
/// verifiers refuse a KECCAK_RND instance above
/// [`crate::BLOCK_KECCAK_RND_MAX_ROWS`], so one table (`off`, the A arm of the
/// chunking A/B) or a taller chunk no longer makes a verifiable block.
fn block_keccak_rnd_rows() -> usize {
    match std::env::var("LAMBDA_VM_BLOCK_KECCAK_RND_LOG2")
        .ok()
        .as_deref()
    {
        None => 1 << BLOCK_KECCAK_RND_ROWS_LOG2,
        Some(v) => {
            let n: u32 = v
                .parse()
                .ok()
                .filter(|n| (5..=BLOCK_KECCAK_RND_ROWS_LOG2).contains(n))
                .unwrap_or_else(|| {
                    panic!(
                        "LAMBDA_VM_BLOCK_KECCAK_RND_LOG2 must be 5..={BLOCK_KECCAK_RND_ROWS_LOG2}, \
                         got `{v}`"
                    )
                });
            1 << n
        }
    }
}

/// The block's table caps: [`BLOCK_ROWS_LOG2`] for every splittable table,
/// KECCAK_RND chunked (see [`block_keccak_rnd_rows`]) and ECDAS chunked at
/// [`BLOCK_ECDAS_ROWS_LOG2`].
pub fn block_max_rows() -> MaxRowsConfig {
    MaxRowsConfig {
        keccak_rnd: block_keccak_rnd_rows(),
        ecdas: 1 << BLOCK_ECDAS_ROWS_LOG2,
        ..MaxRowsConfig::uniform(1 << BLOCK_ROWS_LOG2)
    }
}

/// The block verifier: [`crate::verify_with_options`] under
/// [`AcceleratorShape::BlockChunked`], the shape [`prove_block`] proves —
/// KECCAK_RND and ECDAS chunked, each instance under its cap. Every other
/// verifier keeps both to one table.
pub fn verify_block(
    vm_proof: &VmProof,
    elf_bytes: &[u8],
    opts: &ProofOptions,
) -> Result<bool, Error> {
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let elf_digest = crate::statement::elf_digest(elf_bytes);
    crate::verify_prepared_shaped(
        vm_proof,
        &program,
        &elf_digest,
        opts,
        None,
        None,
        AcceleratorShape::BlockChunked,
    )
}

/// `LAMBDA_VM_BLOCK_WARM_PRECOMPUTED=0` leaves the ELF data pages' preprocessed
/// commitments to Round 1 (the A arm of the warm A/B); unset or anything else
/// derives them beside the execution.
fn warm_precomputed() -> bool {
    std::env::var("LAMBDA_VM_BLOCK_WARM_PRECOMPUTED").map_or(true, |v| v != "0")
}

/// Derive every ELF data page's preprocessed root, both leaf layouts, with the
/// device commit where it admits the shape, and record them
/// ([`crate::tables::page::record_data_page_commitment`]) for the PAGE AIRs'
/// lazy commitments to return. FAST 354 measured the alternative: derived on the
/// host beside Round 1 they shortened Round 1 by 1.3 s and lengthened its
/// prepass by as much (the host work competed with the prove's own).
fn warm_data_page_commitments(program: &Elf, opts: &ProofOptions) {
    use crate::tables::page;
    use stark::leaf_layout::LeafLayout;
    let t = Instant::now();
    let mut n = 0;
    for config in Traces::page_configs_from_elf(program) {
        if config.init_values.is_none() || config.is_private_input {
            continue;
        }
        let group = page::preprocessed_group(&config);
        for layout in [LeafLayout::RowPair, LeafLayout::Row] {
            let root =
                crate::lfm::commit::commit_group_device_or_host_with("PAGE", &group, opts, layout);
            page::record_data_page_commitment(&config, opts, layout, root);
            n += 1;
        }
    }
    eprintln!(
        "BLOCK WARM: {n} data-page commitments in {:.2}s, beside the execution",
        t.elapsed().as_secs_f64()
    );
}

/// Wall times of one block prove, in seconds, for the readout.
#[derive(Debug, Clone, Default)]
pub struct BlockTimes {
    /// The whole block's execution (DECODE's root is derived beside it).
    pub execute: f64,
    /// Every trace, built from the execution logs.
    pub build: f64,
    /// AIR construction, the statement and the transcript.
    pub setup: f64,
    /// `multi_prove`: both phases, every instance.
    pub prove: f64,
}

impl BlockTimes {
    /// Execution to proof.
    pub fn total(&self) -> f64 {
        self.execute + self.build + self.setup + self.prove
    }
}

/// Levels of each plain table's tree the block's phase B does NOT keep: it
/// keeps the rest on the host after Round 1 and recomputes the LDE alone (no
/// second hash), rebuilding each queried subtree of `2^k` leaves at the
/// openings (FAST 358: −4.89 s at k = 3). At k = 6 the kept tops are 1/64 of
/// each tree (FAST 503: the 1× peak −1.2 GiB, the 1.57× peak −2.3 GiB, the
/// base unmoved). `LAMBDA_VM_RECOMMIT_TOP_LEVELS` overrides it (0 = the full
/// recommit).
pub const BLOCK_RECOMMIT_TOP_LEVELS: usize = 6;

/// [`prove_block_with`] at the block's shape: [`block_max_rows`] and
/// [`ResidencyMode::RecomputeLdeDevice`].
pub fn prove_block(
    elf_bytes: &[u8],
    private_input: &[u8],
    opts: &ProofOptions,
) -> Result<(VmProof, BlockTimes), Error> {
    prove_block_with(
        elf_bytes,
        private_input,
        opts,
        &block_max_rows(),
        ResidencyMode::RecomputeLdeDevice,
    )
}

/// [`prove_block`], handing `on_shape` the proof's [`BlockShape`] once the
/// traces are built, before the prove: what a consumer of the proof will
/// receive, so the recursion's programs can be derived while the base proves.
///
/// [`BlockShape`]: crate::lfm::block_plan::BlockShape
pub fn prove_block_observed(
    elf_bytes: &[u8],
    private_input: &[u8],
    opts: &ProofOptions,
    on_shape: &mut dyn FnMut(&crate::lfm::block_plan::BlockShape),
) -> Result<(VmProof, BlockTimes), Error> {
    prove_block_with_observed(
        elf_bytes,
        private_input,
        opts,
        &block_max_rows(),
        ResidencyMode::RecomputeLdeDevice,
        on_shape,
    )
}

/// Execute the whole program, build every trace at `max_rows`, and prove it as
/// one monolithic proof under `residency`. DECODE's commitment, a function of
/// the ELF and `opts` alone, is derived on a helper thread beside the
/// execution and handed to the AIRs, as the epoch pipeline does.
pub fn prove_block_with(
    elf_bytes: &[u8],
    private_input: &[u8],
    opts: &ProofOptions,
    max_rows: &MaxRowsConfig,
    residency: ResidencyMode,
) -> Result<(VmProof, BlockTimes), Error> {
    prove_block_with_observed(
        elf_bytes,
        private_input,
        opts,
        max_rows,
        residency,
        &mut |_| {},
    )
}

fn prove_block_with_observed(
    elf_bytes: &[u8],
    private_input: &[u8],
    opts: &ProofOptions,
    max_rows: &MaxRowsConfig,
    residency: ResidencyMode,
    on_shape: &mut dyn FnMut(&crate::lfm::block_plan::BlockShape),
) -> Result<(VmProof, BlockTimes), Error> {
    let mut times = BlockTimes::default();
    #[cfg(feature = "cuda")]
    stark::prover::set_default_recommit_top_levels(BLOCK_RECOMMIT_TOP_LEVELS);
    #[cfg(feature = "cuda")]
    stark::prover::set_default_pack_after_commit(narrow_streamed());
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let ledger = memlog().then(|| std::sync::Arc::new(MemLedger::new()));
    let _sampler = ledger.clone().map(MemSampler::start);
    // The finish's phases, on the ledger's clock, while phase A runs.
    if let Some(ledger) = &ledger {
        let ledger = ledger.clone();
        crate::tables::trace_builder::set_finish_marks(Some(std::sync::Arc::new(move |label| {
            ledger.line(&format!("finish {label}"))
        })));
    }
    let (mut traces, decode_commitment, precommits) = if stream_phase_a() {
        build_streamed(
            &program,
            private_input,
            opts,
            max_rows,
            residency,
            &mut times,
            &StreamConfig::from_env(),
            ledger.as_deref(),
        )?
    } else {
        let (traces, decode) = build_serial(&program, private_input, opts, max_rows, &mut times)?;
        (traces, decode, Vec::new())
    };
    if let Some(ledger) = &ledger {
        crate::tables::trace_builder::set_finish_marks(None);
        ledger.line("phase A end");
    }

    let proof = prove_block_traces(
        elf_bytes,
        &program,
        &mut traces,
        opts,
        Some(decode_commitment),
        residency,
        precommits,
        &mut times,
        on_shape,
    )?;
    if let Some(ledger) = &ledger {
        ledger.line("prove end");
    }
    eprintln!(
        "BLOCK PHASE total {:.2}s (execute {:.2} · build {:.2} · setup {:.2} · prove {:.2})",
        times.total(),
        times.execute,
        times.build,
        times.setup,
        times.prove
    );
    Ok((proof, times))
}

/// Phase A's producer, serial: execute the whole run, then collect, then build
/// every trace (DECODE's root, the DECODE artifacts, the initial image and the
/// data-page roots beside the execution).
fn build_serial(
    program: &Elf,
    private_input: &[u8],
    opts: &ProofOptions,
    max_rows: &MaxRowsConfig,
    times: &mut BlockTimes,
) -> Result<(Traces, Commitment), Error> {
    // Beside the execution: DECODE's root (a function of the ELF and `opts`),
    // the DECODE artifacts and the initial image — none reads the execution.
    let t = Instant::now();
    let (decode_commitment, artifacts, initial_image, run) = std::thread::scope(|s| {
        let decode = s.spawn(|| {
            crate::tables::decode::commitment_from_elf_device_or_host(program, opts)
                .map_err(|e| Error::Recursion(format!("DECODE commitment from ELF: {e}")))
        });
        let artifacts = s.spawn(|| DecodeArtifacts::from_elf(program));
        // The ELF data pages' preprocessed roots, on the card while it is
        // otherwise idle; Round 1 would otherwise derive each on the host.
        if warm_precomputed() {
            s.spawn(|| warm_data_page_commitments(program, opts));
        }
        let image = s.spawn(|| build_initial_image(program, private_input));
        let run = Executor::new(program, private_input.to_vec())
            .map_err(|e| Error::Execution(format!("{e}")))
            .and_then(|executor| executor.run().map_err(|e| Error::Execution(format!("{e}"))));
        (join(decode), join(artifacts), join(image), run)
    });
    let (decode_commitment, artifacts, run) = (decode_commitment?, artifacts?, run?);
    times.execute = t.elapsed().as_secs_f64();
    eprintln!("BLOCK PHASE execute {:.2}s", times.execute);

    // `Traces::from_elf_and_logs`, step by step, for the stamps.
    let t = Instant::now();
    let register_init = register::register_init_from_entry_point(program.entry_point);
    let collected =
        Traces::collect_epoch(&artifacts, &initial_image, &register_init, &run.logs, true)?;
    drop(run);
    let collect = t.elapsed().as_secs_f64();
    let traces = Traces::build_from_collected(
        &artifacts,
        collected,
        Some(&initial_image),
        &register_init,
        max_rows,
        private_input,
        true,
        false,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )?;
    drop(initial_image);
    times.build = t.elapsed().as_secs_f64();
    eprintln!(
        "BLOCK PHASE build {:.2}s (collect {collect:.2} · generate {:.2})",
        times.build,
        times.build - collect
    );

    Ok((traces, decode_commitment))
}

/// A streamed instance's Round-1 commit, made in phase A's stream.
type Precommit = stark::prover::PrecommittedMain<
    crate::tables::types::GoldilocksField,
    crate::hash_pin::BlockStarkHash,
>;

/// A streamed instance back from its committer: the chunk (for
/// `Traces::insert_streamed`), its AIR name and its Round-1 commit.
type Committed = (StreamedChunk, String, Precommit);

/// Phase A's output: the traces, DECODE's root and the streamed instances'
/// precommits by AIR name.
type Produced = (Traces, Commitment, Vec<(String, Precommit)>);

/// `LAMBDA_VM_BLOCK_STREAM=0` builds phase A serially (the A arm of the stream
/// A/B); unset or anything else streams it.
fn stream_phase_a() -> bool {
    std::env::var("LAMBDA_VM_BLOCK_STREAM").map_or(true, |v| v != "0")
}

/// `LAMBDA_VM_BLOCK_STREAM=push` streams phase A with the builder's first
/// schedule (a measurement arm): one thread walks, routes and generates each
/// window's chunks in turn ([`WindowedTraceBuilder::push`]). Otherwise the walk
/// has a thread of its own, the producer routes, and the committers generate.
fn stream_by_push() -> bool {
    std::env::var("LAMBDA_VM_BLOCK_STREAM").is_ok_and(|v| v == "push")
}

/// `LAMBDA_VM_BLOCK_DROP_OPS=0` keeps every walked window's op lists to the
/// finish (the A arm of the drop A/B); unset or anything else drops each
/// streamed chunk's ops as the chunk leaves
/// ([`WindowedTraceBuilder::drop_streamed_ops`]), so phase A does not hold the
/// run's op lists to its end. The traces are the same either way.
fn drop_streamed_ops() -> bool {
    std::env::var("LAMBDA_VM_BLOCK_DROP_OPS").map_or(true, |v| v.trim() != "0")
}

/// Narrow storage, `LAMBDA_VM_BLOCK_NARROW`: `0` keeps every main trace at 8
/// bytes a cell (the A arm), `1` packs each plain table after its Round-1
/// commit ([`narrow_streamed`] alone), and unset or anything else is `2`: both
/// [`narrow_streamed`] and [`narrow_finished`] (D-MEMORY M3).
fn narrow_level() -> u8 {
    match std::env::var("LAMBDA_VM_BLOCK_NARROW")
        .as_deref()
        .map(str::trim)
    {
        Ok("0") => 0,
        Ok("1") => 1,
        _ => 2,
    }
}

/// Every plain table's main trace is packed at the bytes its columns need once
/// its Round-1 commit exists, about 2 bytes a cell instead of 8. The device
/// packs it from the commit's snapshot
/// (`stark::prover::set_default_pack_after_commit`): the streamed instances on
/// their committers, the rest in phase B's Round 1. A table committed on the
/// host is packed on the host (`TraceTable::pack_main_narrow`). Phase B uploads
/// it packed and widens it on the device, or on the host where the device path
/// does not run. The words are the same, so no proof byte moves. On unless
/// `LAMBDA_VM_BLOCK_NARROW=0`.
fn narrow_streamed() -> bool {
    narrow_level() >= 1
}

/// [`narrow_streamed`], and the tables phase A's finish builds are packed as
/// each is generated ([`WindowedTraceBuilder::pack_finished_tables`]), so the
/// finish never holds them at 8 bytes a cell; their Round-1 commits read the
/// packed columns. On unless `LAMBDA_VM_BLOCK_NARROW` is `0` or `1`.
fn narrow_finished() -> bool {
    narrow_level() == 2
}

/// Committer threads for the streamed instances.
const STREAM_COMMITTERS: usize = 3;

/// `LAMBDA_VM_BLOCK_FINISH_WIDE=n` (n >= 1): with [`narrow_finished`], phase
/// A's finish generates at most `n` chunks at 8 bytes a cell at once
/// ([`WindowedTraceBuilder::bound_finished_generation`]); unset or `0` is no
/// bound. A measurement knob until its gate.
fn finish_wide_chunks() -> usize {
    std::env::var("LAMBDA_VM_BLOCK_FINISH_WIDE")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

/// `LAMBDA_VM_BLOCK_COMMITTERS=n` (1..=8): committer threads, a measurement
/// knob; unset is [`STREAM_COMMITTERS`].
fn stream_committers() -> usize {
    std::env::var("LAMBDA_VM_BLOCK_COMMITTERS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| (1..=8).contains(n))
        .unwrap_or(STREAM_COMMITTERS)
}

/// `LAMBDA_VM_BLOCK_GENERATORS=n` (1..=16): `n` threads generate each streamed
/// chunk, and pack it on the host under narrow storage, before a committer
/// takes it, so the committers only commit (and a chunk waits packed rather
/// than as its ops). Unset or `0`: the committers generate. A measurement
/// knob until its gate.
fn stream_generators() -> usize {
    std::env::var("LAMBDA_VM_BLOCK_GENERATORS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n <= 16)
        .unwrap_or(0)
}

/// `LAMBDA_VM_BLOCK_READY_MIB=n` (n >= 1): with [`stream_generators`], the
/// generated chunks waiting for a committer hold at most `n` MiB; unset or `0`:
/// unbounded.
fn ready_queue_budget() -> Option<usize> {
    std::env::var("LAMBDA_VM_BLOCK_READY_MIB")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .map(|n| n << 20)
}

/// `LAMBDA_VM_BLOCK_QUEUE_MIB=n` (n >= 1): the streamed chunks waiting for a
/// committer hold at most `n` MiB of ops ([`QueueRoom`]); the producer waits
/// for room, and so do the walk and the executor behind it. Unset or `0`:
/// unbounded. A measurement knob until its gate.
fn commit_queue_budget() -> Option<usize> {
    std::env::var("LAMBDA_VM_BLOCK_QUEUE_MIB")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .map(|n| n << 20)
}

/// How phase A's stream hands its chunks to the device: committer and
/// generator threads, and the byte budgets of what waits for them.
#[derive(Clone, Copy, Debug)]
struct StreamConfig {
    committers: usize,
    generators: usize,
    ops_budget: Option<usize>,
    ready_budget: Option<usize>,
}

impl StreamConfig {
    /// The block's: [`stream_committers`], [`stream_generators`],
    /// [`commit_queue_budget`], [`ready_queue_budget`].
    fn from_env() -> Self {
        Self {
            committers: stream_committers(),
            generators: stream_generators(),
            ops_budget: commit_queue_budget(),
            ready_budget: ready_queue_budget(),
        }
    }
}

/// The streamed chunks on their way to a committer, by the bytes their ops
/// hold: counted always (the most at once is reported), and bounded by a
/// budget when there is one. A chunk is admitted when the queue is empty or it
/// fits, so a chunk bigger than the budget still goes, alone. It leaves the
/// queue once its committer has generated it (its ops are freed then).
struct QueueRoom {
    budget: Option<usize>,
    /// (bytes, chunks) admitted and not yet generated.
    held: std::sync::Mutex<(usize, usize)>,
    room: std::sync::Condvar,
    /// The most bytes, and the most chunks, held at once.
    most: std::sync::Mutex<(usize, usize)>,
    /// Seconds the producer waited for room.
    waited: std::sync::Mutex<f64>,
    /// A committer stopped: nothing waits for room any more.
    closed: std::sync::atomic::AtomicBool,
}

impl QueueRoom {
    fn new(budget: Option<usize>) -> Self {
        Self {
            budget,
            held: std::sync::Mutex::new((0, 0)),
            room: std::sync::Condvar::new(),
            most: std::sync::Mutex::new((0, 0)),
            waited: std::sync::Mutex::new(0.0),
            closed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Wait until a chunk of `bytes` fits, then count it in.
    fn admit(&self, bytes: usize) {
        let t = Instant::now();
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        while self
            .budget
            .is_some_and(|budget| held.0 > 0 && held.0 + bytes > budget)
            && !self.closed.load(std::sync::atomic::Ordering::Relaxed)
        {
            held = self.room.wait(held).unwrap_or_else(|e| e.into_inner());
        }
        held.0 += bytes;
        held.1 += 1;
        let mut most = self.most.lock().unwrap_or_else(|e| e.into_inner());
        *most = (most.0.max(held.0), most.1.max(held.1));
        *self.waited.lock().unwrap_or_else(|e| e.into_inner()) += t.elapsed().as_secs_f64();
    }

    /// A chunk of `bytes` left the queue.
    fn release(&self, bytes: usize) {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        held.0 = held.0.saturating_sub(bytes);
        held.1 = held.1.saturating_sub(1);
        self.room.notify_all();
    }

    /// A committer stopped on an error: the producer must not wait on it.
    fn close(&self) {
        self.closed
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let _held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        self.room.notify_all();
    }
}

/// `LAMBDA_VM_BLOCK_PURGE=1`: once the run is walked, before the finish
/// builds its tables, the allocator returns the pages freed so far to the OS
/// ([`purge_freed_pages`]), so the finish does not stack its working set on
/// pages its threads cannot reuse. A measurement knob, off by default.
fn purge_before_finish() -> bool {
    std::env::var("LAMBDA_VM_BLOCK_PURGE").is_ok_and(|v| v.trim() == "1")
}

/// Every jemalloc arena's freed (dirty and muzzy) pages returned to the OS now,
/// the decay settings left as they were (never, under the posture): each
/// arena's decay is set to 0, which purges at once, then back. In the lib's
/// tests, which install jemalloc (`lib.rs`); `false` elsewhere.
#[cfg(test)]
fn purge_freed_pages() -> bool {
    use tikv_jemalloc_ctl::{arenas, raw};
    let Ok(n) = arenas::narenas::read() else {
        return false;
    };
    for i in 0..n {
        for kind in ["dirty", "muzzy"] {
            let name = format!("arena.{i}.{kind}_decay_ms\0");
            // SAFETY: a NUL-terminated mallctl name whose value is an ssize_t.
            unsafe {
                let Ok(was) = raw::read::<isize>(name.as_bytes()) else {
                    continue;
                };
                let _ = raw::write(name.as_bytes(), 0isize);
                let _ = raw::write(name.as_bytes(), was);
            }
        }
    }
    true
}

/// Outside the lib's tests the allocator is not driven.
#[cfg(not(test))]
fn purge_freed_pages() -> bool {
    false
}

/// `LAMBDA_VM_BLOCK_DRAIN_PURGE=1`: from the walk's end to phase A's end, the
/// allocator arenas of the walker and the accumulator (where the streamed
/// chunks' ops were allocated) return each freed page at once, so the ops the
/// committers free while the finish runs do not stay resident beside the
/// finish's working set; every other arena keeps its pages. A measurement
/// knob, off by default.
fn drain_purge() -> bool {
    std::env::var("LAMBDA_VM_BLOCK_DRAIN_PURGE").is_ok_and(|v| v.trim() == "1")
}

/// The calling thread's jemalloc arena, in the lib's tests (`lib.rs` installs
/// jemalloc); `None` elsewhere.
#[cfg(test)]
fn current_arena() -> Option<u32> {
    // SAFETY: a NUL-terminated mallctl name whose value is an unsigned.
    unsafe { tikv_jemalloc_ctl::raw::read::<u32>(b"thread.arena\0").ok() }
}

#[cfg(not(test))]
fn current_arena() -> Option<u32> {
    None
}

/// Set `arena`'s dirty and muzzy decay (milliseconds; 0 purges at once and on
/// every later free, -1 never), returning what it had. Lib tests only.
#[cfg(test)]
fn swap_arena_decay(arena: u32, (dirty, muzzy): (isize, isize)) -> Option<(isize, isize)> {
    use tikv_jemalloc_ctl::raw;
    let (d, m) = (
        format!("arena.{arena}.dirty_decay_ms\0"),
        format!("arena.{arena}.muzzy_decay_ms\0"),
    );
    // SAFETY: NUL-terminated mallctl names whose values are ssize_t.
    unsafe {
        let was = (
            raw::read::<isize>(d.as_bytes()).ok()?,
            raw::read::<isize>(m.as_bytes()).ok()?,
        );
        raw::write(d.as_bytes(), dirty).ok()?;
        raw::write(m.as_bytes(), muzzy).ok()?;
        Some(was)
    }
}

#[cfg(not(test))]
fn swap_arena_decay(_arena: u32, _decay: (isize, isize)) -> Option<(isize, isize)> {
    None
}

/// `LAMBDA_VM_BLOCK_MEMLOG=1`: [`MemLedger`]'s `BLOCK MEM` lines, every half
/// second from the block's start to its proof and at the phase marks. A
/// measurement knob, off by default.
fn memlog() -> bool {
    std::env::var("LAMBDA_VM_BLOCK_MEMLOG").is_ok_and(|v| v.trim() == "1")
}

/// Where the block's host memory is: what phase A holds in the places it knows
/// of, printed ([`MemLedger::line`]) beside the process's resident set and, in
/// the lib's tests (which run jemalloc), the allocator's live, active and
/// resident bytes. Lists count at their capacities. It changes no trace.
struct MemLedger {
    start: Instant,
    /// Streamed chunks sent to the committers and not taken yet, and their
    /// bytes (a job's ops, or the `push` arm's generated chunk).
    queued: std::sync::atomic::AtomicUsize,
    queued_bytes: std::sync::atomic::AtomicUsize,
    /// What the committers (or the generators) hold: a taken job's ops, then
    /// its 64-bit trace until it is packed or kept.
    committing_bytes: std::sync::atomic::AtomicUsize,
    /// Chunks generated (and packed) waiting for a committer, and their bytes
    /// ([`stream_generators`]).
    ready: std::sync::atomic::AtomicUsize,
    ready_bytes: std::sync::atomic::AtomicUsize,
    /// Committed streamed chunks waiting for the finish, and their main
    /// traces' bytes, packed and 64-bit.
    committed: std::sync::atomic::AtomicUsize,
    committed_packed: std::sync::atomic::AtomicUsize,
    committed_wide: std::sync::atomic::AtomicUsize,
    /// The executor's windows of logs on their way to the walker, and the
    /// walked windows on their way to the accumulator.
    logs_bytes: std::sync::atomic::AtomicUsize,
    walked_bytes: std::sync::atomic::AtomicUsize,
    /// The builder's run so far after the last absorbed window, the walk's
    /// carried memory state, and the executor's memory, as last published.
    builder_bytes: std::sync::atomic::AtomicUsize,
    walk_bytes: std::sync::atomic::AtomicUsize,
    executor_bytes: std::sync::atomic::AtomicUsize,
}

impl MemLedger {
    fn new() -> Self {
        use std::sync::atomic::AtomicUsize;
        Self {
            start: Instant::now(),
            queued: AtomicUsize::new(0),
            queued_bytes: AtomicUsize::new(0),
            committing_bytes: AtomicUsize::new(0),
            ready: AtomicUsize::new(0),
            ready_bytes: AtomicUsize::new(0),
            committed: AtomicUsize::new(0),
            committed_packed: AtomicUsize::new(0),
            committed_wide: AtomicUsize::new(0),
            logs_bytes: AtomicUsize::new(0),
            walked_bytes: AtomicUsize::new(0),
            builder_bytes: AtomicUsize::new(0),
            walk_bytes: AtomicUsize::new(0),
            executor_bytes: AtomicUsize::new(0),
        }
    }

    /// A chunk of `bytes` sent to the committers.
    fn queue(&self, bytes: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        self.queued.fetch_add(1, Relaxed);
        self.queued_bytes.fetch_add(bytes, Relaxed);
    }

    /// A committer took a chunk of `bytes`.
    fn take(&self, bytes: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        self.queued.fetch_sub(1, Relaxed);
        self.queued_bytes.fetch_sub(bytes, Relaxed);
        self.committing_bytes.fetch_add(bytes, Relaxed);
    }

    /// A committer turned `ops` bytes of a job into a `wide` 64-bit trace.
    fn generated(&self, ops: usize, wide: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        self.committing_bytes.fetch_add(wide, Relaxed);
        self.committing_bytes.fetch_sub(ops, Relaxed);
    }

    /// A generator handed a chunk it held at `wide` bytes to the committers
    /// as `held` bytes (packed, or still 64-bit).
    fn ready(&self, wide: usize, held: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        self.committing_bytes.fetch_sub(wide, Relaxed);
        self.ready.fetch_add(1, Relaxed);
        self.ready_bytes.fetch_add(held, Relaxed);
    }

    /// A committer took a generated chunk of `held` bytes.
    fn take_ready(&self, held: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        self.ready.fetch_sub(1, Relaxed);
        self.ready_bytes.fetch_sub(held, Relaxed);
        self.committing_bytes.fetch_add(held, Relaxed);
    }

    /// A committer kept a committed chunk it held at `held` bytes as `packed`
    /// bytes (`None`: kept 64-bit).
    fn keep(&self, held: usize, packed: Option<usize>) {
        use std::sync::atomic::Ordering::Relaxed;
        self.committing_bytes.fetch_sub(held, Relaxed);
        self.committed.fetch_add(1, Relaxed);
        match packed {
            Some(packed) => self.committed_packed.fetch_add(packed, Relaxed),
            None => self.committed_wide.fetch_add(held, Relaxed),
        };
    }

    fn line(&self, label: &str) {
        use std::sync::atomic::Ordering::Relaxed;
        const GIB: f64 = (1u64 << 30) as f64;
        let g = |a: &std::sync::atomic::AtomicUsize| a.load(Relaxed) as f64 / GIB;
        let rss = proc_rss_bytes().map_or("n/a".to_string(), |b| format!("{:.2}", b as f64 / GIB));
        let heap = heap_stats().map_or("heap n/a".to_string(), |[live, active, resident, mapped, retained]| {
            let g = |b: usize| b as f64 / GIB;
            format!(
                "heap live {:.2} · active {:.2} · resident {:.2} (resident − live {:.2}) · mapped {:.2} · retained {:.2}",
                g(live),
                g(active),
                g(resident),
                g(resident.saturating_sub(live)),
                g(mapped),
                g(retained),
            )
        });
        eprintln!(
            "BLOCK MEM {label} t={:.1} · rss {rss} · {heap} · queue {} chunks {:.2} · committing {:.2} · \
             ready {} packed {:.2} · committed {} ({:.2} packed + {:.2} 64-bit) · logs {:.2} · walked {:.2} · \
             builder {:.2} · walk {:.2} · executor {:.2} (GiB)",
            self.start.elapsed().as_secs_f64(),
            self.queued.load(Relaxed),
            g(&self.queued_bytes),
            g(&self.committing_bytes),
            self.ready.load(Relaxed),
            g(&self.ready_bytes),
            self.committed.load(Relaxed),
            g(&self.committed_packed),
            g(&self.committed_wide),
            g(&self.logs_bytes),
            g(&self.walked_bytes),
            g(&self.builder_bytes),
            g(&self.walk_bytes),
            g(&self.executor_bytes),
        );
    }
}

/// The ledger's line every half second on a thread of its own, until dropped.
struct MemSampler {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl MemSampler {
    fn start(ledger: std::sync::Arc<MemLedger>) -> Self {
        use std::sync::atomic::Ordering::Relaxed;
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = stop.clone();
        let handle = std::thread::Builder::new()
            .name("block-memlog".to_string())
            .spawn(move || {
                while !flag.load(Relaxed) {
                    std::thread::park_timeout(std::time::Duration::from_millis(500));
                    if !flag.load(Relaxed) {
                        ledger.line("tick");
                    }
                }
            })
            .ok();
        Self { stop, handle }
    }
}

impl Drop for MemSampler {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
    }
}

/// This process's resident set (`VmRSS`), on Linux.
fn proc_rss_bytes() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kib: usize = status
        .lines()
        .find(|l| l.starts_with("VmRSS:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(kib * 1024)
}

/// jemalloc's allocated, active, resident, mapped and retained bytes: in the
/// lib's tests, which install jemalloc as the allocator (`lib.rs`).
#[cfg(test)]
fn heap_stats() -> Option<[usize; 5]> {
    use tikv_jemalloc_ctl::{epoch, stats};
    // The statistics are cached until the epoch turns.
    epoch::advance().ok()?;
    Some([
        stats::allocated::read().ok()?,
        stats::active::read().ok()?,
        stats::resident::read().ok()?,
        stats::mapped::read().ok()?,
        stats::retained::read().ok()?,
    ])
}

/// Outside the lib's tests the allocator's statistics are not read.
#[cfg(not(test))]
fn heap_stats() -> Option<[usize; 5]> {
    None
}

/// The bytes a streamed chunk on its way to a committer holds.
fn streamed_bytes(streamed: &Streamed) -> usize {
    match streamed {
        Streamed::Job(job) => job.op_bytes(),
        Streamed::Chunk(chunk) => wide_bytes(&chunk.trace),
    }
}

/// A main trace's bytes at 8 bytes a cell.
fn wide_bytes(
    trace: &stark::trace::TraceTable<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
    >,
) -> usize {
    trace.num_rows() * trace.num_main_columns * std::mem::size_of::<u64>()
}

/// A streamed chunk on its way to a committer: a job the committer generates,
/// or (the `push` arm) a chunk the producer generated.
enum Streamed {
    Job(ChunkJob),
    Chunk(StreamedChunk),
}

/// The AIR a streamed chunk is proved under, named as [`VmAirs::new`] names it.
fn stream_air(table: StreamTable, index: usize, opts: &ProofOptions) -> VmAir {
    match table {
        StreamTable::Cpu => Box::new(create_cpu_air(opts).with_name(&format!("CPU[{index}]"))),
        StreamTable::MemwRegister => {
            Box::new(create_memw_register_air(opts).with_name(&format!("MEMW_R[{index}]")))
        }
        StreamTable::MemwAligned => {
            Box::new(create_memw_aligned_air(opts).with_name(&format!("MEMW_A[{index}]")))
        }
        StreamTable::Memw => Box::new(create_memw_air(opts).with_name(&format!("MEMW[{index}]"))),
        StreamTable::Load => Box::new(create_load_air(opts).with_name(&format!("LOAD[{index}]"))),
        StreamTable::Lt => Box::new(create_lt_air(opts).with_name(&format!("LT[{index}]"))),
        StreamTable::Shift => {
            Box::new(create_shift_air(opts).with_name(&format!("SHIFT[{index}]")))
        }
        StreamTable::Store => {
            Box::new(create_store_air(opts).with_name(&format!("STORE[{index}]")))
        }
    }
}

/// Phase A's producer, streamed. The executor runs in windows of
/// `max_rows.cpu` cycles on its own thread; the shared windowed builder
/// ([`WindowedTraceBuilder`]) collects each window and hands out every full
/// chunk of the eight tables whose rows grow in execution order; committer
/// threads generate each chunk and make its Round-1 commit on the device
/// ([`IsStarkProver::precommit_main`]) while the run is still being executed and
/// collected. The builder is split ([`WindowedTraceBuilder::split`]): a walker
/// thread does nothing but walk, so the executor waits on the walk alone, and
/// this thread keeps, routes and hands out each walked window. A window is
/// walked once the next one arrives (the last goes to `finish`). The traces are
/// a whole-run build's (the builder's own tests).
#[allow(clippy::too_many_arguments)]
fn build_streamed(
    program: &Elf,
    private_input: &[u8],
    opts: &ProofOptions,
    max_rows: &MaxRowsConfig,
    residency: ResidencyMode,
    times: &mut BlockTimes,
    stream: &StreamConfig,
    ledger: Option<&MemLedger>,
) -> Result<Produced, Error> {
    use std::sync::Mutex;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::mpsc;

    let window = max_rows.cpu;
    let t = Instant::now();
    let (job_tx, job_rx) = mpsc::channel::<Streamed>();
    let job_rx = Mutex::new(job_rx);
    let committer_count = stream.committers;
    let queue = QueueRoom::new(stream.ops_budget);
    let generators = stream.generators;
    let ready = QueueRoom::new(stream.ready_budget);
    let (ready_tx, ready_rx) = mpsc::channel::<(StreamedChunk, usize)>();
    let ready_rx = Mutex::new(ready_rx);
    let committed: Mutex<Vec<Committed>> = Mutex::new(Vec::new());
    let commit_secs = Mutex::new(0.0f64);
    let generate_secs = Mutex::new(0.0f64);
    // Packed instances: (wide bytes, packed bytes, seconds packing on the
    // host, instances the device packed).
    let narrowed = Mutex::new((0usize, 0usize, 0.0f64, 0usize));
    let narrow = narrow_streamed();
    let drain = drain_purge();
    // The walker's and the accumulator's arenas ([`drain_purge`]), and the
    // decay each had while they purge at once.
    let producer_arenas: &Mutex<Vec<u32>> = &Mutex::new(Vec::new());
    let drained: Mutex<Vec<(u32, (isize, isize))>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        // The executor, window by window, a bounded two windows ahead.
        let (log_tx, log_rx) = mpsc::sync_channel::<Vec<executor::vm::logs::Log>>(2);
        let exec = s.spawn(move || -> Result<f64, Error> {
            let t = Instant::now();
            let mut executor = Executor::new(program, private_input.to_vec())
                .map_err(|e| Error::Execution(format!("{e}")))?;
            while let Some(logs) = executor
                .resume_with_limit(window)
                .map_err(|e| Error::Execution(format!("{e}")))?
            {
                let logs = logs.to_vec();
                let logs_bytes = logs.capacity() * std::mem::size_of::<executor::vm::logs::Log>();
                if log_tx.send(logs).is_err() {
                    break;
                }
                if let Some(ledger) = ledger {
                    ledger.logs_bytes.fetch_add(logs_bytes, Relaxed);
                    ledger
                        .executor_bytes
                        .store(executor.memory().heap_bytes(), Relaxed);
                }
            }
            Ok(t.elapsed().as_secs_f64())
        });
        let decode = s.spawn(|| {
            crate::tables::decode::commitment_from_elf_device_or_host(program, opts)
                .map_err(|e| Error::Recursion(format!("DECODE commitment from ELF: {e}")))
        });
        if warm_precomputed() {
            s.spawn(|| warm_data_page_commitments(program, opts));
        }

        // Generators ([`stream_generators`]): each streamed chunk generated,
        // and packed under narrow storage, ahead of the committers.
        for _ in 0..generators {
            let (job_rx, queue, ready, narrowed, generate_secs) =
                (&job_rx, &queue, &ready, &narrowed, &generate_secs);
            let ready_tx = ready_tx.clone();
            s.spawn(move || {
                loop {
                    let job = job_rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                    let Ok(job) = job else {
                        return;
                    };
                    let job_bytes = streamed_bytes(&job);
                    if let Some(ledger) = ledger {
                        ledger.take(job_bytes);
                    }
                    let t = Instant::now();
                    let mut chunk = match job {
                        Streamed::Job(job) => job.generate(),
                        Streamed::Chunk(chunk) => chunk,
                    };
                    queue.release(job_bytes);
                    let wide = wide_bytes(&chunk.trace);
                    if let Some(ledger) = ledger {
                        ledger.generated(job_bytes, wide);
                    }
                    *generate_secs.lock().unwrap_or_else(|e| e.into_inner()) +=
                        t.elapsed().as_secs_f64();
                    let tp = Instant::now();
                    if narrow && chunk.trace.pack_main_narrow() {
                        narrowed.lock().unwrap_or_else(|e| e.into_inner()).2 +=
                            tp.elapsed().as_secs_f64();
                    }
                    let held = chunk.trace.narrow_main().map_or(wide, |t| t.data().len());
                    if let Some(ledger) = ledger {
                        ledger.ready(wide, held);
                    }
                    ready.admit(held);
                    if ready_tx.send((chunk, held)).is_err() {
                        ready.release(held);
                    }
                }
            });
        }
        drop(ready_tx);

        // Committers: each streamed chunk generated (unless a generator did)
        // and Round-1 committed, as the chunks complete.
        let mut committers = Vec::new();
        for _ in 0..committer_count {
            committers.push(s.spawn(|| -> Result<(), Error> {
                let commit_all = || -> Result<(), Error> {
                    loop {
                        // `t`: from the chunk's arrival (its generation
                        // included when this committer generates it).
                        let (mut chunk, wide, held, t) = if generators > 0 {
                            let next = ready_rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                            let Ok((chunk, held)) = next else {
                                return Ok(());
                            };
                            let t = Instant::now();
                            ready.release(held);
                            if let Some(ledger) = ledger {
                                ledger.take_ready(held);
                            }
                            let wide = wide_bytes(&chunk.trace);
                            (chunk, wide, held, t)
                        } else {
                            let job = job_rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                            let Ok(job) = job else {
                                return Ok(());
                            };
                            let t = Instant::now();
                            let job_bytes = streamed_bytes(&job);
                            if let Some(ledger) = ledger {
                                ledger.take(job_bytes);
                            }
                            let chunk = match job {
                                Streamed::Job(job) => job.generate(),
                                Streamed::Chunk(chunk) => chunk,
                            };
                            queue.release(job_bytes);
                            let wide = wide_bytes(&chunk.trace);
                            if let Some(ledger) = ledger {
                                ledger.generated(job_bytes, wide);
                            }
                            *generate_secs.lock().unwrap_or_else(|e| e.into_inner()) +=
                                t.elapsed().as_secs_f64();
                            (chunk, wide, wide, t)
                        };
                        let air = stream_air(chunk.table, chunk.index, opts);
                        let name = air.name().to_string();
                        #[allow(unused_mut)]
                        let mut pre = crate::hash_pin::BlockProver::precommit_main(
                            air.as_ref(),
                            &chunk.trace,
                            #[cfg(feature = "disk-spill")]
                            stark::storage_mode::StorageMode::Ram,
                            residency,
                        )
                        .map_err(|e| Error::Prover(format!("{e:?}")))?;
                        *commit_secs.lock().unwrap_or_else(|e| e.into_inner()) +=
                            t.elapsed().as_secs_f64();
                        if narrow {
                            let tp = Instant::now();
                            // The device packed it from the commit's snapshot, or
                            // the host packs it here.
                            let by_device = pre
                                .take_narrow()
                                .is_some_and(|t| chunk.trace.install_main_narrow(t));
                            if by_device || chunk.trace.pack_main_narrow() {
                                let packed =
                                    chunk.trace.narrow_main().map_or(0, |t| t.data().len());
                                let mut n = narrowed.lock().unwrap_or_else(|e| e.into_inner());
                                n.0 += wide;
                                n.1 += packed;
                                if by_device {
                                    n.3 += 1;
                                } else {
                                    n.2 += tp.elapsed().as_secs_f64();
                                }
                            }
                        }
                        if let Some(ledger) = ledger {
                            let packed = chunk.trace.narrow_main().map(|t| t.data().len());
                            ledger.keep(held, packed);
                        }
                        committed
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push((chunk, name, pre));
                    }
                };
                let result = commit_all();
                if result.is_err() {
                    queue.close();
                    ready.close();
                }
                result
            }));
        }

        let produce = || -> Result<(Traces, f64), Error> {
            if drain {
                producer_arenas
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .extend(current_arena());
            }
            let mut builder = WindowedTraceBuilder::new(program, private_input, max_rows)?;
            if drop_streamed_ops() {
                builder = builder.drop_streamed_ops()?;
            }
            if narrow_finished() {
                builder = builder
                    .pack_finished_tables()
                    .bound_finished_generation(finish_wide_chunks());
            }
            if let Some(ledger) = ledger {
                eprintln!(
                    "BLOCK MEM builder created: initial image {:.2} GiB",
                    builder.image_bytes() as f64 / (1u64 << 30) as f64
                );
                ledger.line("start");
            }
            let mut collect_secs = 0.0;
            let last = if stream_by_push() {
                let mut held: Option<Vec<executor::vm::logs::Log>> = None;
                for logs in log_rx.iter() {
                    if let Some(prev) = held.replace(logs) {
                        let t = Instant::now();
                        for chunk in builder.push(&prev)? {
                            let chunk = Streamed::Chunk(chunk);
                            let bytes = streamed_bytes(&chunk);
                            queue.admit(bytes);
                            if let Some(ledger) = ledger {
                                ledger.queue(bytes);
                            }
                            let _ = job_tx.send(chunk);
                        }
                        collect_secs += t.elapsed().as_secs_f64();
                    }
                }
                held.unwrap_or_default()
            } else {
                let (mut walker, mut accumulator) = builder.split();
                std::thread::scope(|inner| -> Result<_, Error> {
                    let (walked_tx, walked_rx) = mpsc::sync_channel(2);
                    let walking = inner.spawn(move || -> Result<_, Error> {
                        if drain {
                            producer_arenas
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .extend(current_arena());
                        }
                        // One window held back: the run's last is `finish`'s.
                        let mut held: Option<Vec<executor::vm::logs::Log>> = None;
                        for logs in log_rx.iter() {
                            if let Some(prev) = held.replace(logs) {
                                let walked = walker.walk(&prev)?;
                                if let Some(ledger) = ledger {
                                    ledger.walk_bytes.store(walker.state_bytes(), Relaxed);
                                    ledger.logs_bytes.fetch_sub(
                                        prev.capacity()
                                            * std::mem::size_of::<executor::vm::logs::Log>(),
                                        Relaxed,
                                    );
                                    ledger.walked_bytes.fetch_add(walked.heap_bytes(), Relaxed);
                                }
                                if walked_tx.send(walked).is_err() {
                                    break;
                                }
                            }
                        }
                        Ok(held.unwrap_or_default())
                    });
                    for walked in walked_rx {
                        let t = Instant::now();
                        if let Some(ledger) = ledger {
                            ledger.walked_bytes.fetch_sub(walked.heap_bytes(), Relaxed);
                        }
                        for job in accumulator.absorb(walked) {
                            let job = Streamed::Job(job);
                            let bytes = streamed_bytes(&job);
                            queue.admit(bytes);
                            if let Some(ledger) = ledger {
                                ledger.queue(bytes);
                            }
                            let _ = job_tx.send(job);
                        }
                        if let Some(ledger) = ledger {
                            ledger
                                .builder_bytes
                                .store(accumulator.held_bytes(), Relaxed);
                        }
                        collect_secs += t.elapsed().as_secs_f64();
                    }
                    join(walking)
                })?
            };
            drop(job_tx);
            if let Some(ledger) = ledger {
                ledger.line("windows walked");
            }
            if drain {
                let (t, before) = (Instant::now(), proc_rss_bytes());
                let mut arenas = producer_arenas.lock().unwrap_or_else(|e| e.into_inner());
                arenas.sort_unstable();
                arenas.dedup();
                let swapped: Vec<(u32, (isize, isize))> = arenas
                    .iter()
                    .filter_map(|&a| swap_arena_decay(a, (0, 0)).map(|was| (a, was)))
                    .collect();
                eprintln!(
                    "BLOCK DRAIN PURGE: arenas {:?} return freed pages at once until phase A ends ({:.2} s) · \
                     VmRSS {:.2} → {:.2} GiB",
                    swapped.iter().map(|(a, _)| a).collect::<Vec<_>>(),
                    t.elapsed().as_secs_f64(),
                    before.map_or(-1.0, |b| b as f64 / (1u64 << 30) as f64),
                    proc_rss_bytes().map_or(-1.0, |b| b as f64 / (1u64 << 30) as f64),
                );
                *drained.lock().unwrap_or_else(|e| e.into_inner()) = swapped;
            }
            if purge_before_finish() {
                let (t, before) = (Instant::now(), proc_rss_bytes());
                let purged = purge_freed_pages();
                let gib = |b: Option<usize>| b.map_or(-1.0, |b| b as f64 / (1u64 << 30) as f64);
                eprintln!(
                    "BLOCK PURGE before the finish: {} in {:.2} s · VmRSS {:.2} → {:.2} GiB",
                    if purged { "purged" } else { "not available" },
                    t.elapsed().as_secs_f64(),
                    gib(before),
                    gib(proc_rss_bytes()),
                );
            }
            let windows = builder.stamps();
            let t = Instant::now();
            let traces = builder.finish(&last)?;
            let finish_secs = t.elapsed().as_secs_f64();
            if let Some(ledger) = ledger {
                ledger.line("finish done");
            }
            eprintln!(
                "BLOCK STREAM builder: {} windows · walk {:.2} · route {:.2} · hand-out {:.2} · \
                 finish {finish_secs:.2} (s) · streamed ops dropped {}",
                windows.windows,
                windows.walk,
                windows.route,
                windows.generate,
                if drop_streamed_ops() { "on" } else { "off" },
            );
            collect_secs += finish_secs;
            Ok((traces, collect_secs))
        };
        let produced = produce();
        let mut errors = Vec::new();
        for c in committers {
            if let Err(e) = join(c) {
                errors.push(e);
            }
        }
        // Phase A's committers are done: the producer's arenas keep their
        // pages again.
        for (arena, was) in std::mem::take(&mut *drained.lock().unwrap_or_else(|e| e.into_inner()))
        {
            let _ = swap_arena_decay(arena, was);
        }
        let (mut traces, collect_secs) = produced?;
        if let Some(e) = errors.into_iter().next() {
            return Err(e);
        }
        times.execute = join(exec)?;
        let decode_commitment = join(decode)?;

        let committed: Vec<Committed> =
            std::mem::take(&mut *committed.lock().unwrap_or_else(|e| e.into_inner()));
        let n = committed.len();
        let mut chunks = Vec::with_capacity(n);
        let mut precommits = Vec::with_capacity(n);
        for (chunk, name, pre) in committed {
            chunks.push(chunk);
            precommits.push((name, pre));
        }
        traces.insert_streamed(chunks)?;
        let total = t.elapsed().as_secs_f64();
        times.build = total - times.execute;
        eprintln!(
            "BLOCK PHASE stream {total:.2}s (execute {:.2} on its thread · collect+build {collect_secs:.2} · \
             build+commit of {n} streamed instances {:.2} on {committer_count} threads, generate {:.2} {})",
            times.execute,
            *commit_secs.lock().unwrap_or_else(|e| e.into_inner()),
            *generate_secs.lock().unwrap_or_else(|e| e.into_inner()),
            if generators > 0 {
                format!("on {generators} generator threads")
            } else {
                "of it".to_string()
            },
        );
        if narrow {
            let (wide, packed, secs, by_device) =
                *narrowed.lock().unwrap_or_else(|e| e.into_inner());
            eprintln!(
                "BLOCK NARROW: streamed main traces {:.2} GiB packed to {:.2} GiB ({:.3} B/cell) · \
                 {by_device} of {n} packed by the device · host packing {secs:.2} s on the committers",
                wide as f64 / (1u64 << 30) as f64,
                packed as f64 / (1u64 << 30) as f64,
                if wide > 0 {
                    packed as f64 * 8.0 / wide as f64
                } else {
                    0.0
                },
            );
        }
        let (most_bytes, most_chunks) = *queue.most.lock().unwrap_or_else(|e| e.into_inner());
        let (ready_bytes, ready_chunks) = *ready.most.lock().unwrap_or_else(|e| e.into_inner());
        let budget =
            |b: Option<usize>| b.map_or("none".to_string(), |b| format!("{} MiB", b >> 20));
        eprintln!(
            "BLOCK QUEUE: {committer_count} committers · {generators} generators · ops budget {} · at most \
             {:.2} GiB in {most_chunks} chunks waiting as ops · the producer waited {:.2} s for room · \
             ready budget {} · at most {:.2} GiB in {ready_chunks} chunks waiting generated · the \
             generators waited {:.2} s for room",
            budget(queue.budget),
            most_bytes as f64 / (1u64 << 30) as f64,
            *queue.waited.lock().unwrap_or_else(|e| e.into_inner()),
            budget(ready.budget),
            ready_bytes as f64 / (1u64 << 30) as f64,
            *ready.waited.lock().unwrap_or_else(|e| e.into_inner()),
        );
        Ok((traces, decode_commitment, precommits))
    })
}

/// The prove step over built traces: the AIRs [`crate::prove_with_options_and_inputs`]
/// builds (with DECODE's commitment supplied when given), the monolithic
/// statement in the transcript, and one `multi_prove` under `residency`.
/// Prints the instance census and hands `on_shape` the proof's shape before
/// proving.
#[allow(clippy::too_many_arguments)]
pub fn prove_block_traces(
    elf_bytes: &[u8],
    program: &Elf,
    traces: &mut Traces,
    opts: &ProofOptions,
    decode_commitment: Option<Commitment>,
    residency: ResidencyMode,
    precommits: Vec<(String, Precommit)>,
    times: &mut BlockTimes,
    on_shape: &mut dyn FnMut(&crate::lfm::block_plan::BlockShape),
) -> Result<VmProof, Error> {
    let t = Instant::now();
    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        program,
        opts,
        false,
        &traces.page_configs,
        &table_counts,
        decode_commitment,
        true,
        None,
        None,
        None,
    );
    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();
    let mut transcript = crate::hash_pin::block_transcript(&[]);
    absorb_statement(
        &mut transcript,
        StatementKind::Monolithic,
        elf_bytes,
        &traces.public_output_bytes,
        &table_counts,
        num_private_input_pages,
        &runtime_page_ranges,
        opts.fri_final_poly_log_degree,
    );
    let public_output = traces.public_output_bytes.clone();
    let pairs = airs.air_trace_pairs(traces);
    print_census(
        pairs
            .iter()
            .map(|(air, trace, _)| (air.name(), trace.num_rows(), trace.num_main_columns)),
    );
    if memlog() {
        // The main traces the prove starts from: packed, and at 8 bytes a cell.
        let (mut packed, mut wide, mut n_packed) = (0usize, 0usize, 0usize);
        for (_, trace, _) in &pairs {
            match trace.narrow_main() {
                Some(narrow) => {
                    packed += narrow.data().len();
                    n_packed += 1;
                }
                None => wide += wide_bytes(trace),
            }
        }
        eprintln!(
            "BLOCK MEM traces: {} instances · {n_packed} packed {:.2} GiB · {} at 8 B/cell {:.2} GiB",
            pairs.len(),
            packed as f64 / (1u64 << 30) as f64,
            pairs.len() - n_packed,
            wide as f64 / (1u64 << 30) as f64,
        );
    }
    on_shape(&crate::lfm::block_plan::BlockShape {
        table_counts: table_counts.clone(),
        runtime_page_ranges: runtime_page_ranges.clone(),
        num_private_input_pages,
        public_output_len: public_output.len(),
        trace_lengths: pairs.iter().map(|(_, trace, _)| trace.num_rows()).collect(),
    });
    times.setup = t.elapsed().as_secs_f64();
    eprintln!(
        "BLOCK PHASE setup {:.2}s · residency {residency:?}",
        times.setup
    );

    // Each streamed instance's precommit goes to its AIR's index (by name); the
    // rest commit in Round 1.
    let n_precommits = precommits.len();
    let mut by_name: std::collections::BTreeMap<String, Precommit> =
        precommits.into_iter().collect();
    let precommitted: Vec<Option<Precommit>> = pairs
        .iter()
        .map(|(air, _, _)| by_name.remove(air.name()))
        .collect();
    let placed = precommitted.iter().filter(|p| p.is_some()).count();
    if placed != n_precommits || !by_name.is_empty() {
        return Err(Error::Prover(format!(
            "{n_precommits} streamed precommits, {placed} placed on an AIR"
        )));
    }

    let t = Instant::now();
    let proof = crate::hash_pin::BlockProver::multi_prove_precommitted(
        pairs,
        &mut transcript,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
        residency,
        precommitted,
    )
    .map_err(|e| Error::Prover(format!("{e:?}")))?;
    times.prove = t.elapsed().as_secs_f64();
    eprintln!("BLOCK PHASE prove {:.2}s", times.prove);

    Ok(VmProof {
        proof,
        runtime_page_ranges,
        table_counts,
        public_output,
        num_private_input_pages,
    })
}

/// One `BLOCK CENSUS` line per table type (instances, rows, main cells) and a
/// total, from `(AIR name, rows, main columns)` in AIR order. A type is the
/// name before its `[i]` instance suffix.
fn print_census<'a>(instances: impl Iterator<Item = (&'a str, usize, usize)>) {
    let mut types: Vec<(String, usize, u64, u64)> = Vec::new();
    for (name, rows, cols) in instances {
        let ty = name.split('[').next().unwrap_or(name).to_string();
        let cells = (rows * cols) as u64;
        match types.iter_mut().find(|(t, ..)| *t == ty) {
            Some(entry) => {
                entry.1 += 1;
                entry.2 += rows as u64;
                entry.3 += cells;
            }
            None => types.push((ty, 1, rows as u64, cells)),
        }
    }
    let (mut instances_total, mut cells_total) = (0usize, 0u64);
    for (ty, count, rows, cells) in &types {
        eprintln!("BLOCK CENSUS {ty}: {count} instance(s), {rows} rows, {cells} main cells");
        instances_total += count;
        cells_total += cells;
    }
    eprintln!(
        "BLOCK CENSUS total: {instances_total} sub-proofs, {} table types, {cells_total} main cells",
        types.len()
    );
}

/// [`build_streamed`] of `program` (no input) under a stream of `committers`
/// committers, `generators` generators and `budget` bytes for each queue, for
/// the tests: the traces, and the AIR of each streamed instance precommitted.
#[cfg(test)]
pub(crate) fn stream_for_test(
    program: &Elf,
    opts: &ProofOptions,
    max_rows: &MaxRowsConfig,
    committers: usize,
    generators: usize,
    budget: Option<usize>,
) -> Result<(Traces, Vec<String>), Error> {
    let stream = StreamConfig {
        committers,
        generators,
        ops_budget: budget,
        ready_budget: budget,
    };
    let (traces, _, precommits) = build_streamed(
        program,
        &[],
        opts,
        max_rows,
        ResidencyMode::RecomputeLdeDevice,
        &mut BlockTimes::default(),
        &stream,
        None,
    )?;
    Ok((
        traces,
        precommits.into_iter().map(|(name, _)| name).collect(),
    ))
}

#[cfg(test)]
mod memlog_tests {
    use std::sync::atomic::Ordering::Relaxed;

    use super::MemLedger;

    /// A chunk queued, taken, generated and kept leaves nothing queued or in a
    /// committer's hands, and counts once as committed (packed or 64-bit).
    #[test]
    fn the_ledger_moves_each_chunk_from_the_queue_to_the_committed() {
        let ledger = MemLedger::new();
        ledger.queue(300);
        ledger.queue(200);
        assert_eq!(ledger.queued.load(Relaxed), 2);
        assert_eq!(ledger.queued_bytes.load(Relaxed), 500);

        ledger.take(300);
        ledger.generated(300, 800);
        assert_eq!(ledger.committing_bytes.load(Relaxed), 800);
        ledger.keep(800, Some(220));

        ledger.take(200);
        ledger.generated(200, 640);
        ledger.keep(640, None);

        assert_eq!(ledger.queued.load(Relaxed), 0);
        assert_eq!(ledger.queued_bytes.load(Relaxed), 0);
        assert_eq!(ledger.committing_bytes.load(Relaxed), 0);
        assert_eq!(ledger.committed.load(Relaxed), 2);
        assert_eq!(ledger.committed_packed.load(Relaxed), 220);
        assert_eq!(ledger.committed_wide.load(Relaxed), 640);
        // In the lib's tests jemalloc is the allocator: the line reads it.
        assert!(super::heap_stats().is_some_and(|[live, ..]| live > 0));
        ledger.line("test");
    }
}

#[cfg(test)]
mod queue_tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use super::QueueRoom;

    /// Within a budget the producer waits for room: a chunk that would pass the
    /// budget is admitted only once an earlier one left. A chunk bigger than
    /// the budget still goes when the queue is empty.
    #[test]
    fn a_bounded_queue_admits_a_chunk_once_it_fits() {
        let queue = QueueRoom::new(Some(100));
        queue.admit(60);
        queue.admit(40);
        let admitted = AtomicBool::new(false);
        std::thread::scope(|s| {
            s.spawn(|| {
                queue.admit(30);
                admitted.store(true, Ordering::SeqCst);
            });
            std::thread::sleep(Duration::from_millis(100));
            assert!(!admitted.load(Ordering::SeqCst), "admitted past the budget");
            queue.release(60);
        });
        assert!(admitted.load(Ordering::SeqCst));
        assert_eq!(*queue.held.lock().unwrap(), (70, 2));
        assert_eq!(*queue.most.lock().unwrap(), (100, 2));
        queue.release(40);
        queue.release(30);
        queue.admit(500);
        assert_eq!(*queue.held.lock().unwrap(), (500, 1));
    }

    /// Unbounded, nothing waits; a stopped committer frees a waiting producer.
    #[test]
    fn an_unbounded_or_closed_queue_never_waits() {
        let unbounded = QueueRoom::new(None);
        for _ in 0..10 {
            unbounded.admit(1 << 30);
        }
        assert_eq!(unbounded.most.lock().unwrap().1, 10);

        let queue = QueueRoom::new(Some(10));
        queue.admit(10);
        std::thread::scope(|s| {
            let waiting = s.spawn(|| queue.admit(10));
            std::thread::sleep(Duration::from_millis(50));
            queue.close();
            waiting.join().unwrap();
        });
        assert_eq!(*queue.held.lock().unwrap(), (20, 2));
    }
}

#[cfg(test)]
mod wide_permit_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::tables::trace_builder::WidePermits;

    /// However many workers want one, at most `n` hold a permit at once, and
    /// every worker gets one in the end.
    #[test]
    fn at_most_n_chunks_hold_a_permit() {
        let permits = WidePermits::new(2);
        let (holding, most, done) = (
            AtomicUsize::new(0),
            AtomicUsize::new(0),
            AtomicUsize::new(0),
        );
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    let _permit = permits.acquire();
                    let now = holding.fetch_add(1, Ordering::SeqCst) + 1;
                    most.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    holding.fetch_sub(1, Ordering::SeqCst);
                    done.fetch_add(1, Ordering::SeqCst);
                });
            }
        });
        assert_eq!(done.load(Ordering::SeqCst), 8);
        assert_eq!(most.load(Ordering::SeqCst), 2);
    }
}

#[cfg(test)]
mod purge_tests {
    /// The purge reaches jemalloc in the lib's tests and leaves every arena's
    /// decay settings as they were.
    #[test]
    fn a_purge_leaves_the_decay_settings_as_they_were() {
        use tikv_jemalloc_ctl::raw;
        let name = b"arena.0.dirty_decay_ms\0";
        // SAFETY: a NUL-terminated mallctl name whose value is an ssize_t.
        let before = unsafe { raw::read::<isize>(name) }.expect("arena 0 decay");
        // Something to purge: a freed allocation.
        drop(vec![1u8; 64 << 20]);
        assert!(super::purge_freed_pages());
        let after = unsafe { raw::read::<isize>(name) }.expect("arena 0 decay");
        assert_eq!(before, after);
    }

    /// A thread finds its own arena, and a swapped decay comes back as it was.
    #[test]
    fn an_arena_decay_swaps_and_swaps_back() {
        let arena = super::current_arena().expect("this thread's arena");
        let was = super::swap_arena_decay(arena, (0, 0)).expect("swap to 0");
        assert_eq!(super::swap_arena_decay(arena, was), Some((0, 0)));
        assert_eq!(super::swap_arena_decay(arena, was), Some(was));
    }
}
