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
    DecodeArtifacts, Traces, WindowedCollector, build_initial_image, generate_cpu_window,
};
use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::test_utils::create_cpu_air;
use crate::{AcceleratorShape, Commitment, Error, ProofOptions, VmAirs, VmProof};
use stark::trace::TraceTable as StarkTraceTable;

/// A helper thread's result, its panic re-raised on the caller.
fn join<T>(handle: std::thread::ScopedJoinHandle<'_, T>) -> T {
    handle
        .join()
        .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
}

/// Rows per full-height instance: every splittable table is cut into
/// instances of `2^BLOCK_ROWS_LOG2` rows (the tail padded to its power of two).
pub const BLOCK_ROWS_LOG2: u32 = 21;

/// KECCAK_RND rows per instance: 2^16, so the batching phase keeps its 128
/// bits (D-NOEPOCH §6) and no instance outweighs the VRAM gate.
pub const BLOCK_KECCAK_RND_ROWS_LOG2: u32 = 16;

/// `LAMBDA_VM_BLOCK_KECCAK_RND_LOG2`: `off` proves KECCAK_RND as one table (the
/// A arm of the chunking A/B), `5..=26` sets its cap; unset is
/// [`BLOCK_KECCAK_RND_ROWS_LOG2`]. Anything else aborts.
fn block_keccak_rnd_rows() -> usize {
    match std::env::var("LAMBDA_VM_BLOCK_KECCAK_RND_LOG2")
        .ok()
        .as_deref()
    {
        None => 1 << BLOCK_KECCAK_RND_ROWS_LOG2,
        Some("off") => crate::tables::KECCAK_RND_UNCHUNKED,
        Some(v) => {
            let n: u32 = v
                .parse()
                .ok()
                .filter(|n| (5..=26).contains(n))
                .unwrap_or_else(|| {
                    panic!("LAMBDA_VM_BLOCK_KECCAK_RND_LOG2 must be `off` or 5..=26, got `{v}`")
                });
            1 << n
        }
    }
}

/// The block's table caps: [`BLOCK_ROWS_LOG2`] for every splittable table and
/// KECCAK_RND chunked (see [`block_keccak_rnd_rows`]).
pub fn block_max_rows() -> MaxRowsConfig {
    MaxRowsConfig {
        keccak_rnd: block_keccak_rnd_rows(),
        ..MaxRowsConfig::uniform(1 << BLOCK_ROWS_LOG2)
    }
}

/// The block verifier: [`crate::verify_with_options`] that also accepts a
/// chunked KECCAK_RND ([`AcceleratorShape::KeccakRndChunked`]), the shape
/// [`prove_block`] proves. Every other verifier keeps KECCAK_RND to one table.
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
        AcceleratorShape::KeccakRndChunked,
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
    let mut times = BlockTimes::default();
    let program = Elf::load(elf_bytes).map_err(|e| Error::ElfLoad(format!("{e}")))?;
    let (mut traces, decode_commitment, cpu_precommits) = if stream_phase_a() {
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
        cpu_precommits,
        &mut times,
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

/// A CPU instance's trace and its Round-1 commit, made in phase A's stream.
type CpuPrecommit = stark::prover::PrecommittedMain<
    crate::tables::types::GoldilocksField,
    crate::hash_pin::BlockStarkHash,
>;

/// A streamed CPU instance: its index, its trace and its Round-1 commit.
type BuiltCpu = (
    usize,
    StarkTraceTable<GoldilocksField, GoldilocksExtension>,
    CpuPrecommit,
);

/// `LAMBDA_VM_BLOCK_STREAM=0` builds phase A serially (the A arm of the stream
/// A/B); unset or anything else streams it.
fn stream_phase_a() -> bool {
    std::env::var("LAMBDA_VM_BLOCK_STREAM").map_or(true, |v| v != "0")
}

/// Committer threads for the streamed CPU instances.
const STREAM_BUILDERS: usize = 2;

/// Phase A's producer, streamed. The executor runs in windows of one CPU
/// instance (`max_rows.cpu` cycles) on its own thread; the collector walks
/// each window as it arrives; and for each window a builder generates that CPU
/// instance's trace from the window's logs and makes its Round-1 commit on the
/// device ([`IsStarkProver::precommit_main`]) while the run is still being
/// executed and collected. After the last window the other tables are built as
/// before. The traces are the serial build's (the windowed walk is the one walk;
/// each window's CPU table is `chunk_and_generate`'s chunk of the same ops).
fn build_streamed(
    program: &Elf,
    private_input: &[u8],
    opts: &ProofOptions,
    max_rows: &MaxRowsConfig,
    residency: ResidencyMode,
    times: &mut BlockTimes,
) -> Result<(Traces, Commitment, Vec<CpuPrecommit>), Error> {
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};

    let window = max_rows.cpu;
    let t = Instant::now();
    std::thread::scope(|outer| {
        // The executor, window by window, a bounded two windows ahead.
        let (log_tx, log_rx) = mpsc::sync_channel::<Arc<Vec<executor::vm::logs::Log>>>(2);
        let exec = outer.spawn(move || -> Result<f64, Error> {
            let t = Instant::now();
            let mut executor = Executor::new(program, private_input.to_vec())
                .map_err(|e| Error::Execution(format!("{e}")))?;
            while let Some(logs) = executor
                .resume_with_limit(window)
                .map_err(|e| Error::Execution(format!("{e}")))?
            {
                if log_tx.send(Arc::new(logs.to_vec())).is_err() {
                    break;
                }
            }
            Ok(t.elapsed().as_secs_f64())
        });
        let decode = outer.spawn(|| {
            crate::tables::decode::commitment_from_elf_device_or_host(program, opts)
                .map_err(|e| Error::Recursion(format!("DECODE commitment from ELF: {e}")))
        });
        if warm_precomputed() {
            outer.spawn(|| warm_data_page_commitments(program, opts));
        }
        let image = outer.spawn(|| build_initial_image(program, private_input));
        let artifacts = DecodeArtifacts::from_elf(program)?;
        let initial_image = join(image);
        let register_init = register::register_init_from_entry_point(program.entry_point);

        let (job_tx, job_rx) = mpsc::channel::<(usize, usize, Arc<Vec<executor::vm::logs::Log>>)>();
        let job_rx = Mutex::new(job_rx);
        let built: Mutex<Vec<BuiltCpu>> = Mutex::new(Vec::new());
        let builder_secs = Mutex::new(0.0f64);
        let (collect_secs, collected) = std::thread::scope(|inner| -> Result<_, Error> {
            let mut builders = Vec::new();
            for _ in 0..STREAM_BUILDERS {
                builders.push(inner.spawn(|| -> Result<(), Error> {
                    loop {
                        let job = job_rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                        let Ok((k, first_cycle, logs)) = job else {
                            return Ok(());
                        };
                        let t = Instant::now();
                        let trace = generate_cpu_window(&logs, &artifacts, first_cycle)?;
                        drop(logs);
                        let air = create_cpu_air(opts).with_name(&format!("CPU[{k}]"));
                        let pre = crate::hash_pin::BlockProver::precommit_main(
                            &air,
                            &trace,
                            #[cfg(feature = "disk-spill")]
                            stark::storage_mode::StorageMode::Ram,
                            residency,
                        )
                        .map_err(|e| Error::Prover(format!("{e:?}")))?;
                        *builder_secs.lock().unwrap_or_else(|e| e.into_inner()) +=
                            t.elapsed().as_secs_f64();
                        built
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push((k, trace, pre));
                    }
                }));
            }
            let mut collector =
                WindowedCollector::new(&artifacts, &initial_image, &register_init, 1 << 27);
            let mut collect_secs = 0.0;
            for (k, logs) in log_rx.iter().enumerate() {
                let _ = job_tx.send((k, collector.cycles(), Arc::clone(&logs)));
                let t = Instant::now();
                collector.push_window(&logs)?;
                collect_secs += t.elapsed().as_secs_f64();
            }
            drop(job_tx);
            let t = Instant::now();
            let collected = collector.finish(true);
            collect_secs += t.elapsed().as_secs_f64();
            for b in builders {
                join(b)?;
            }
            Ok((collect_secs, collected))
        })?;
        times.execute = join(exec)?;
        let decode_commitment = join(decode)?;
        let streamed = t.elapsed().as_secs_f64();

        let t_build = Instant::now();
        let mut traces = Traces::build_from_collected(
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
        let generate = t_build.elapsed().as_secs_f64();
        let mut built = built.into_inner().unwrap_or_else(|e| e.into_inner());
        built.sort_by_key(|(k, ..)| *k);
        let mut precommits = Vec::with_capacity(built.len());
        for (k, (i, trace, pre)) in built.into_iter().enumerate() {
            debug_assert_eq!(i, k, "one CPU instance per window, in order");
            traces.cpus.push(trace);
            precommits.push(pre);
        }
        times.build = t.elapsed().as_secs_f64() - times.execute;
        eprintln!(
            "BLOCK PHASE stream {streamed:.2}s (execute {:.2} on its thread · collect {collect_secs:.2} · \
             CPU build+commit {:.2} on {STREAM_BUILDERS} threads, {} instances) · generate {generate:.2}",
            times.execute,
            builder_secs.into_inner().unwrap_or_else(|e| e.into_inner()),
            precommits.len(),
        );
        Ok((traces, decode_commitment, precommits))
    })
}

/// The prove step over built traces: the AIRs [`crate::prove_with_options_and_inputs`]
/// builds (with DECODE's commitment supplied when given), the monolithic
/// statement in the transcript, and one `multi_prove` under `residency`.
/// Prints the instance census before proving.
#[allow(clippy::too_many_arguments)]
pub fn prove_block_traces(
    elf_bytes: &[u8],
    program: &Elf,
    traces: &mut Traces,
    opts: &ProofOptions,
    decode_commitment: Option<Commitment>,
    residency: ResidencyMode,
    cpu_precommits: Vec<CpuPrecommit>,
    times: &mut BlockTimes,
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
    times.setup = t.elapsed().as_secs_f64();
    eprintln!(
        "BLOCK PHASE setup {:.2}s · residency {residency:?}",
        times.setup
    );

    // `CPU[i]`'s precommit goes to that AIR's index; the rest commit in Round 1.
    let n_cpu_precommits = cpu_precommits.len();
    let mut cpu_precommits: Vec<Option<CpuPrecommit>> =
        cpu_precommits.into_iter().map(Some).collect();
    let precommitted: Vec<Option<CpuPrecommit>> = pairs
        .iter()
        .map(|(air, _, _)| {
            air.name()
                .strip_prefix("CPU[")
                .and_then(|rest| rest.strip_suffix(']'))
                .and_then(|i| i.parse::<usize>().ok())
                .and_then(|i| cpu_precommits.get_mut(i).and_then(Option::take))
        })
        .collect();
    let placed = precommitted.iter().filter(|p| p.is_some()).count();
    if placed != n_cpu_precommits {
        return Err(Error::Prover(format!(
            "{n_cpu_precommits} CPU precommits for {placed} CPU tables"
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
