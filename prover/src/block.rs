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
/// openings (FAST 358: −4.89 s at k = 3). `LAMBDA_VM_RECOMMIT_TOP_LEVELS`
/// overrides it (0 = the full recommit).
pub const BLOCK_RECOMMIT_TOP_LEVELS: usize = 3;

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
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let (mut traces, decode_commitment, precommits) = if stream_phase_a() {
        build_streamed(
            &program,
            private_input,
            opts,
            max_rows,
            residency,
            &mut times,
        )?
    } else {
        let (traces, decode) = build_serial(&program, private_input, opts, max_rows, &mut times)?;
        (traces, decode, Vec::new())
    };

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

/// `LAMBDA_VM_BLOCK_DROP_OPS=1`: the builder drops each streamed chunk's ops as
/// the chunk leaves ([`WindowedTraceBuilder::drop_streamed_ops`]), so phase A
/// does not hold the run's op lists to its end; the traces are the same. Off by
/// default until its box gate (D-MEMORY M1).
fn drop_streamed_ops() -> bool {
    std::env::var("LAMBDA_VM_BLOCK_DROP_OPS").is_ok_and(|v| v.trim() == "1")
}

/// Committer threads for the streamed instances.
const STREAM_COMMITTERS: usize = 3;

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
fn build_streamed(
    program: &Elf,
    private_input: &[u8],
    opts: &ProofOptions,
    max_rows: &MaxRowsConfig,
    residency: ResidencyMode,
    times: &mut BlockTimes,
) -> Result<Produced, Error> {
    use std::sync::Mutex;
    use std::sync::mpsc;

    let window = max_rows.cpu;
    let t = Instant::now();
    let (job_tx, job_rx) = mpsc::channel::<Streamed>();
    let job_rx = Mutex::new(job_rx);
    let committed: Mutex<Vec<Committed>> = Mutex::new(Vec::new());
    let commit_secs = Mutex::new(0.0f64);
    let generate_secs = Mutex::new(0.0f64);
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
                if log_tx.send(logs.to_vec()).is_err() {
                    break;
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

        // Committers: each streamed chunk generated and Round-1 committed, as
        // the chunks complete.
        let mut committers = Vec::new();
        for _ in 0..STREAM_COMMITTERS {
            committers.push(s.spawn(|| -> Result<(), Error> {
                loop {
                    let job = job_rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                    let Ok(job) = job else {
                        return Ok(());
                    };
                    let t = Instant::now();
                    let chunk = match job {
                        Streamed::Job(job) => job.generate(),
                        Streamed::Chunk(chunk) => chunk,
                    };
                    *generate_secs.lock().unwrap_or_else(|e| e.into_inner()) +=
                        t.elapsed().as_secs_f64();
                    let air = stream_air(chunk.table, chunk.index, opts);
                    let name = air.name().to_string();
                    let pre = crate::hash_pin::BlockProver::precommit_main(
                        air.as_ref(),
                        &chunk.trace,
                        #[cfg(feature = "disk-spill")]
                        stark::storage_mode::StorageMode::Ram,
                        residency,
                    )
                    .map_err(|e| Error::Prover(format!("{e:?}")))?;
                    *commit_secs.lock().unwrap_or_else(|e| e.into_inner()) +=
                        t.elapsed().as_secs_f64();
                    committed
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push((chunk, name, pre));
                }
            }));
        }

        let produce = || -> Result<(Traces, f64), Error> {
            let mut builder = WindowedTraceBuilder::new(program, private_input, max_rows)?;
            if drop_streamed_ops() {
                builder = builder.drop_streamed_ops()?;
            }
            let mut collect_secs = 0.0;
            let last = if stream_by_push() {
                let mut held: Option<Vec<executor::vm::logs::Log>> = None;
                for logs in log_rx.iter() {
                    if let Some(prev) = held.replace(logs) {
                        let t = Instant::now();
                        for chunk in builder.push(&prev)? {
                            let _ = job_tx.send(Streamed::Chunk(chunk));
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
                        // One window held back: the run's last is `finish`'s.
                        let mut held: Option<Vec<executor::vm::logs::Log>> = None;
                        for logs in log_rx.iter() {
                            if let Some(prev) = held.replace(logs)
                                && walked_tx.send(walker.walk(&prev)?).is_err()
                            {
                                break;
                            }
                        }
                        Ok(held.unwrap_or_default())
                    });
                    for walked in walked_rx {
                        let t = Instant::now();
                        for job in accumulator.absorb(walked) {
                            let _ = job_tx.send(Streamed::Job(job));
                        }
                        collect_secs += t.elapsed().as_secs_f64();
                    }
                    join(walking)
                })?
            };
            drop(job_tx);
            let windows = builder.stamps();
            let t = Instant::now();
            let traces = builder.finish(&last)?;
            let finish_secs = t.elapsed().as_secs_f64();
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
             build+commit of {n} streamed instances {:.2} on {STREAM_COMMITTERS} threads, generate {:.2} of it)",
            times.execute,
            *commit_secs.lock().unwrap_or_else(|e| e.into_inner()),
            *generate_secs.lock().unwrap_or_else(|e| e.into_inner()),
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
