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

use crate::finish_sink::{self, FinishSink, FinishedTable};
use crate::statement::{StatementKind, absorb_statement};
use crate::tables::MaxRowsConfig;
use crate::tables::gpack::{self, TraceForm};
use crate::tables::register;
use crate::tables::trace_builder::{
    ChunkJob, DecodeArtifacts, StreamedChunk, Traces, WindowedTraceBuilder, build_initial_image,
};
use crate::{AcceleratorShape, Commitment, Error, ProofOptions, VmAirs, VmProof};

/// A helper thread's result, its panic re-raised on the caller.
fn join<T>(handle: std::thread::ScopedJoinHandle<'_, T>) -> T {
    handle
        .join()
        .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
}

/// `scope.spawn`, the thread named `name` so a per-thread sampler can tell the
/// stream's threads apart. A thread that cannot start is fatal, as with
/// `scope.spawn`.
fn spawn_named<'scope, 'env, T: Send + 'scope>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    name: String,
    f: impl FnOnce() -> T + Send + 'scope,
) -> std::thread::ScopedJoinHandle<'scope, T> {
    std::thread::Builder::new()
        .name(name)
        .spawn_scoped(scope, f)
        .expect("spawn a block stream thread")
}

/// Rows per full-height instance: every splittable table is cut into
/// instances of `2^BLOCK_ROWS_LOG2` rows (the tail padded to its power of two).
pub const BLOCK_ROWS_LOG2: u32 = 21;

/// KECCAK rows per instance: the block format's cap
/// ([`crate::BLOCK_KECCAK_MAX_ROWS`], 2^18), so the batching phase keeps its
/// bits and no instance outweighs the VRAM gate however many permutations the
/// block calls.
pub const BLOCK_KECCAK_ROWS_LOG2: u32 = crate::BLOCK_KECCAK_MAX_ROWS.trailing_zeros();

/// KECCAK_RND rows per instance: the block format's cap
/// ([`crate::BLOCK_KECCAK_RND_MAX_ROWS`], 2^16), so the batching phase keeps
/// its 128 bits (D-NOEPOCH §6) and no instance outweighs the VRAM gate.
pub const BLOCK_KECCAK_RND_ROWS_LOG2: u32 = crate::BLOCK_KECCAK_RND_MAX_ROWS.trailing_zeros();

/// ECSM rows per instance: the block format's cap
/// ([`crate::BLOCK_ECSM_MAX_ROWS`], 2^17).
pub const BLOCK_ECSM_ROWS_LOG2: u32 = crate::BLOCK_ECSM_MAX_ROWS.trailing_zeros();

/// ECDAS rows per instance: the block format's cap
/// ([`crate::BLOCK_ECDAS_MAX_ROWS`], 2^17).
pub const BLOCK_ECDAS_ROWS_LOG2: u32 = crate::BLOCK_ECDAS_MAX_ROWS.trailing_zeros();

/// A chunked table's rows per instance from the knob `var`: `min..=cap` (log2)
/// sets it, unset is `2^cap`. Anything else aborts: the block verifiers refuse
/// an instance above its cap, so a taller chunk (or one table, `off`) no
/// longer makes a verifiable block.
fn block_chunk_rows(var: &str, min: u32, cap: u32) -> usize {
    match std::env::var(var).ok().as_deref() {
        None => 1 << cap,
        Some(v) => {
            let n: u32 = v
                .parse()
                .ok()
                .filter(|n| (min..=cap).contains(n))
                .unwrap_or_else(|| panic!("{var} must be {min}..={cap}, got `{v}`"));
            1 << n
        }
    }
}

/// The block's table caps: [`BLOCK_ROWS_LOG2`] for every splittable table, and
/// the chunked accelerators at their caps unless a measurement knob lowers
/// one: `LAMBDA_VM_BLOCK_KECCAK_LOG2` (`2..=18`), `LAMBDA_VM_BLOCK_KECCAK_RND_LOG2`
/// (`5..=16`, whole permutations of 24 rows) and `LAMBDA_VM_BLOCK_ECSM_LOG2`
/// (`2..=17`) ([`block_chunk_rows`]); ECDAS at [`BLOCK_ECDAS_ROWS_LOG2`].
pub fn block_max_rows() -> MaxRowsConfig {
    MaxRowsConfig {
        keccak: block_chunk_rows("LAMBDA_VM_BLOCK_KECCAK_LOG2", 2, BLOCK_KECCAK_ROWS_LOG2),
        keccak_rnd: block_chunk_rows(
            "LAMBDA_VM_BLOCK_KECCAK_RND_LOG2",
            5,
            BLOCK_KECCAK_RND_ROWS_LOG2,
        ),
        ecsm: block_chunk_rows("LAMBDA_VM_BLOCK_ECSM_LOG2", 2, BLOCK_ECSM_ROWS_LOG2),
        ecdas: 1 << BLOCK_ECDAS_ROWS_LOG2,
        ..MaxRowsConfig::uniform(1 << BLOCK_ROWS_LOG2)
    }
}

/// The block verifier: [`crate::verify_with_options`] under
/// [`AcceleratorShape::BlockChunked`], the shape [`prove_block`] proves —
/// KECCAK, KECCAK_RND, ECSM and ECDAS chunked, each instance under its cap.
/// Every other verifier keeps all four to one table.
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

/// The block's cap on a kept top's rebuilt subtree, in field elements: each
/// plain table leaves out of its kept top at most [`BLOCK_RECOMMIT_TOP_LEVELS`]
/// levels, fewer when its row is wide (KECCAK_RND 2, ECDAS 3, KECCAK 4, every
/// table of ≤ 64 columns 6). At the median block phase B −8.44 s (KECCAK_RND's
/// queries 0.67 → 0.09 s a table, the fused region's head idle 9–10 → under
/// 1 s), the 1× base −0.70 s, kept tops +0.15 GiB (BIG 466). Proofs are the
/// same bytes. `LAMBDA_VM_KEPT_SUBTREE_ELEMS` overrides it (0 = one depth for
/// every table, as before).
pub const BLOCK_KEPT_SUBTREE_ELEMS: usize = 8192;

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
    // With the table timeline on, the global rayon pool's workers are named
    // `rayon-<n>` for a per-thread sampler; only a pool nothing built yet can
    // be named, so a process that used rayon before says so.
    #[cfg(feature = "parallel")]
    if std::env::var("LAMBDA_VM_TABLE_TIMELINE").is_ok_and(|v| v == "1") {
        let named = rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("rayon-{i}"))
            .build_global()
            .is_ok();
        eprintln!(
            "[block] the global rayon pool's workers {}",
            if named {
                "are named rayon-<n>"
            } else {
                "keep their names (the pool was built before the block)"
            }
        );
    }
    #[cfg(feature = "cuda")]
    stark::prover::set_default_recommit_top_levels(BLOCK_RECOMMIT_TOP_LEVELS);
    #[cfg(feature = "cuda")]
    stark::prover::set_default_kept_subtree_elems(BLOCK_KEPT_SUBTREE_ELEMS);
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
    // A new block: no memory pressure seen yet (`alloc_purge`'s `auto`).
    crate::alloc_purge::clear_memory_pressure();
    // The spill outlives the prove: phase B reads the spilled traces back.
    let spill = stream_phase_a()
        .then(|| Spill::open(spill_policy()))
        .flatten();
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
            spill.as_ref(),
        )?
    } else {
        let (traces, decode) = build_serial(&program, private_input, opts, max_rows, &mut times)?;
        (traces, decode, Vec::new())
    };
    if let Some(ledger) = &ledger {
        crate::tables::trace_builder::set_finish_marks(None);
        ledger.line("phase A end");
    }
    if let Some(spill) = &spill {
        eprintln!("BLOCK SPILL phase A: {}", spill.report());
    }
    // Phase A's freed pages (the finish's lists, spilled traces) back to the OS
    // before phase B allocates: under memory pressure by default, or as
    // `LAMBDA_VM_ALLOC_PURGE` names it.
    crate::alloc_purge::purge_point("phase-a");

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
    if let Some(spill) = &spill {
        eprintln!("BLOCK SPILL end: {}", spill.report());
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

/// An instance back from its committer: the table (for
/// [`finish_sink::insert_finished`]), its AIR name and its Round-1 commit.
type Committed = (FinishedTable, String, Precommit);

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

/// `LAMBDA_VM_BLOCK_LT_CONCAT=1`: the finish concatenates LT's ops into one
/// list before chunking them ([`WindowedTraceBuilder::concat_lt`]), the A arm
/// of keeping them as segments; the tables are the same.
fn lt_concat() -> bool {
    std::env::var("LAMBDA_VM_BLOCK_LT_CONCAT").is_ok_and(|v| v.trim() == "1")
}

/// `LAMBDA_VM_BLOCK_COMPACT_LT=0`: the LT ops derived from the MEMW ops the
/// builder drops are held at 24 bytes each until the finish
/// ([`WindowedTraceBuilder::raw_memw_lt`]), the A arm of keeping them compact;
/// unset or anything else keeps them compact. The tables are the same.
fn compact_lt() -> bool {
    !std::env::var("LAMBDA_VM_BLOCK_COMPACT_LT").is_ok_and(|v| v.trim() == "0")
}

/// With [`narrow_finished`], the finish builds KECCAK_RND and LT packed a
/// block at a time; `LAMBDA_VM_BLOCK_PACKED_BUILD=0` builds them at 8 bytes a
/// cell and packs them afterwards, as the other tables
/// ([`WindowedTraceBuilder::build_wide_then_pack`]). The words are the same
/// either way. On by default: at the median block on BIG the wide builds'
/// 64-bit KECCAK_RND chunks in flight held the peak ≈ 14 GiB higher (BIG 107:
/// 96.7 against 82.7 GiB), which a block at the top of the July median range
/// (≈ 12×) does not have room for, and the packed builds cost no base time
/// there (−1.85 s; phase B +0.7 s, the finish −2.6 s).
fn packed_builds() -> bool {
    !std::env::var("LAMBDA_VM_BLOCK_PACKED_BUILD").is_ok_and(|v| v.trim() == "0")
}

/// G-pack (`tables::gpack`), `LAMBDA_VM_BLOCK_GPACK`: under narrow storage the
/// generators write each streamed chunk packed as they generate it, and the
/// finish each table it builds ([`WindowedTraceBuilder::generate_packed`]),
/// with no 64-bit copy; `0` builds each at 8 bytes a cell and packs it
/// afterwards (the A arm). The bytes are the same either way.
fn gpack() -> bool {
    !std::env::var("LAMBDA_VM_BLOCK_GPACK").is_ok_and(|v| v.trim() == "0")
}

/// `LAMBDA_VM_BLOCK_KR_WIDE_CAP=n` (1..=64): with KECCAK_RND built wide, then
/// packed (`LAMBDA_VM_BLOCK_PACKED_BUILD=0`), the finish builds at most `n` of its chunks at
/// 8 bytes a cell at once ([`WindowedTraceBuilder::cap_kr_wide`]); unset,
/// 0 or anything else, no cap. The tables are the same.
fn kr_wide_cap() -> usize {
    std::env::var("LAMBDA_VM_BLOCK_KR_WIDE_CAP")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| (1..=64).contains(n))
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

/// Generator threads for the streamed instances ([`stream_generators`]):
/// eight. Six fall behind the producer's walk levers (BIG 467: 125 chunks
/// waiting at windows walked). At the median block on BIG the walk levers with
/// eight generators end phase A 3.45 s sooner than without them on six; with
/// ten, 2.23 s sooner, the walk itself 2.1 s slower beside the extra threads
/// (BIG 468).
const STREAM_GENERATORS: usize = 8;

/// `LAMBDA_VM_BLOCK_GENERATORS=n` (0..=16): `n` threads generate each streamed
/// chunk, and pack it on the host under narrow storage, before a committer
/// takes it, so the committers only commit and a chunk waits packed rather
/// than as its ops. `0`: the committers generate (the A arm). Unset is
/// [`STREAM_GENERATORS`]: at the median block on BIG the queue of ops fell
/// from 32.8 to 1.4 GiB, the peak 13.75 GiB and the base 12.2 s (BIG 103).
fn stream_generators() -> usize {
    std::env::var("LAMBDA_VM_BLOCK_GENERATORS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n <= 16)
        .unwrap_or(STREAM_GENERATORS)
}

/// When the tables the finish builds get their Round-1 commits
/// (`LAMBDA_VM_BLOCK_FINISH_COMMIT`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FinishCommit {
    /// `0`: in phase B's Round 1 (the A arm).
    PhaseB,
    /// Unset or `1`: on phase A's committers, each as soon as the finish has
    /// built it ([`WindowedTraceBuilder::finish_handing`]), while the finish
    /// goes on. At the median block on BIG, with the committers' own pool:
    /// phase A +5.45 s, phase B −13.32 s, base −7.85 s (BIG 463, three runs an
    /// arm).
    Handed,
}

/// [`FinishCommit`] from `LAMBDA_VM_BLOCK_FINISH_COMMIT`; anything else stops
/// the run.
fn finish_commit() -> FinishCommit {
    match std::env::var("LAMBDA_VM_BLOCK_FINISH_COMMIT")
        .as_deref()
        .map(str::trim)
    {
        Err(_) | Ok("1") => FinishCommit::Handed,
        Ok("0") => FinishCommit::PhaseB,
        Ok(other) => panic!("LAMBDA_VM_BLOCK_FINISH_COMMIT must be `0` or `1`, got `{other}`"),
    }
}

/// Phase A's card gate, in bytes: each commit a committer makes is admitted
/// by its device set (the estimate phase B's Round 1 admits the same commit
/// against) under the card's admission budget, so more committers, or larger
/// tables beside them, cannot over-fill the card. Without it each commit is
/// admitted alone against the whole budget and nothing sums concurrent ones.
/// `LAMBDA_VM_BLOCK_CARD_GATE=0` drops it (the A arm); off the device there is
/// no budget.
fn card_gate_budget() -> Option<usize> {
    if std::env::var("LAMBDA_VM_BLOCK_CARD_GATE").is_ok_and(|v| v.trim() == "0") {
        return None;
    }
    #[cfg(feature = "cuda")]
    return stark::gpu_lde::device_vram_budget_bytes().map(|b| b as usize);
    #[cfg(not(feature = "cuda"))]
    return None;
}

/// The card bytes `air`'s Round-1 commit of `trace` is admitted for: its
/// device set (LDE, trace snapshot, tree, scratch) at the trace's leaf layout,
/// as phase B's Round 1 estimates it.
fn commit_card_bytes(air: &crate::test_utils::VmAir, trace: &finish_sink::Trace) -> usize {
    let n = trace.num_rows();
    let layout = stark::leaf_layout::table_leaf_layout(air.as_ref(), n);
    stark::device_set::commit_device_set_rpl(
        n,
        trace.num_main_columns,
        air.options().blowup_factor as usize,
        true,
        layout.rows_per_leaf(),
    )
    .total() as usize
}

/// Phase A's card gate ([`card_gate_budget`]): a commit is admitted when the
/// card holds nothing or its bytes fit beside what it holds, so a commit
/// bigger than the budget still runs, alone. Without a budget it only counts.
/// The most it held, and how long the committers waited, are reported.
struct CardGate {
    budget: Option<usize>,
    /// (bytes, commits) admitted now.
    held: std::sync::Mutex<(usize, usize)>,
    room: std::sync::Condvar,
    most: std::sync::Mutex<(usize, usize)>,
    waited: std::sync::Mutex<f64>,
    /// A committer stopped: nothing waits any more.
    closed: std::sync::atomic::AtomicBool,
}

impl CardGate {
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

    /// Wait until a commit of `bytes` fits, then count it in.
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

    /// A commit of `bytes` is done.
    fn release(&self, bytes: usize) {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        held.0 = held.0.saturating_sub(bytes);
        held.1 = held.1.saturating_sub(1);
        self.room.notify_all();
    }

    /// A committer stopped on an error: the others must not wait on it.
    fn close(&self) {
        self.closed
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let _held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        self.room.notify_all();
    }
}

/// Threads in the committers' own rayon pool ([`commit_pool_threads`]).
const COMMIT_POOL_THREADS: usize = 8;

/// `LAMBDA_VM_BLOCK_COMMIT_POOL=n` (0..=32): the committers make their commits
/// inside a rayon pool of `n` threads of their own, so the host-side parallel
/// work of a commit (a small table's host LDE and tree, a packed trace's
/// widening) does not queue behind the finish's generation, which holds every
/// worker of the global pool while it runs (BIG 462: during p5 the finish handed
/// 140 tables and none was committed until p5 ended; BIG 463 with the pool:
/// 127–141 committed in p5). `0`: the global pool (the A arm). Unset is
/// [`COMMIT_POOL_THREADS`].
fn commit_pool_threads() -> usize {
    std::env::var("LAMBDA_VM_BLOCK_COMMIT_POOL")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n <= 32)
        .unwrap_or(COMMIT_POOL_THREADS)
}

/// How phase A's stream hands its chunks to the device: committer and
/// generator threads, when the finish's tables are committed, the card's byte
/// budget for the commits, the committers' own pool, and whether the tables
/// are written packed as they are generated ([`gpack`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct StreamConfig {
    committers: usize,
    generators: usize,
    finish: FinishCommit,
    card_budget: Option<usize>,
    commit_pool: usize,
    gpack: bool,
}

impl StreamConfig {
    /// The block's: [`stream_committers`], [`stream_generators`],
    /// [`finish_commit`], [`card_gate_budget`], [`commit_pool_threads`],
    /// [`gpack`].
    fn from_env() -> Self {
        Self {
            committers: stream_committers(),
            generators: stream_generators(),
            finish: finish_commit(),
            card_budget: card_gate_budget(),
            commit_pool: commit_pool_threads(),
            gpack: gpack(),
        }
    }

    /// This config with G-pack on or off ([`gpack`]).
    #[cfg(test)]
    pub(crate) fn with_gpack(self, gpack: bool) -> Self {
        Self { gpack, ..self }
    }
}

/// What phase A does with each committed instance's packed main trace
/// (D-ANYBLOCK S2, `LAMBDA_VM_BLOCK_SPILL`): keep it on the host, or write it to
/// a spill file ([`stark::spill`]) for phase B to read back ahead of its walks.
/// The words that come back are the words that went out (digest-checked), so
/// no proof byte depends on the policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SpillPolicy {
    /// Keep every trace.
    Off,
    /// Spill every committed instance that is packed (a measurement arm).
    Always,
    /// Keep at most this many bytes of committed packed traces on the host;
    /// spill the rest.
    Budget(u64),
    /// Spill once the host would pass the target ([`spill_target_bytes`]):
    /// see [`spill_decision`]. The default: a block that fits spills nothing.
    Auto,
}

/// `LAMBDA_VM_BLOCK_SPILL`: `auto` (and unset) | `off` | `always` | `<GiB>` (a
/// resident budget for committed packed traces). Anything else is `off`.
fn spill_policy() -> SpillPolicy {
    parse_spill_policy(std::env::var("LAMBDA_VM_BLOCK_SPILL").ok().as_deref())
}

fn parse_spill_policy(value: Option<&str>) -> SpillPolicy {
    match value.map(str::trim) {
        Some("always") => SpillPolicy::Always,
        Some("auto") => SpillPolicy::Auto,
        Some(gib) => gib
            .parse::<f64>()
            .ok()
            .filter(|g| g.is_finite() && *g >= 0.0)
            .map_or(SpillPolicy::Off, |g| {
                SpillPolicy::Budget((g * (1u64 << 30) as f64) as u64)
            }),
        None => SpillPolicy::Auto,
    }
}

/// `auto`'s target for the host: `LAMBDA_VM_BLOCK_SPILL_TARGET_GIB`, else the
/// smaller of the cgroup's memory limit (v2 or v1, [`cgroup_memory`]) and
/// `MemTotal`, less 10 GiB ([`spill_target_from`]).
fn spill_target_bytes() -> u64 {
    if let Some(gib) = std::env::var("LAMBDA_VM_BLOCK_SPILL_TARGET_GIB")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|g| g.is_finite() && *g > 0.0)
    {
        return (gib * (1u64 << 30) as f64) as u64;
    }
    let limit = cgroup_memory(
        &std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default(),
        std::path::Path::new(CGROUP_ROOT),
        CgroupValue::File("memory.max"),
        CgroupValue::File("memory.limit_in_bytes"),
    );
    let mem_total = std::fs::read_to_string("/proc/meminfo").ok().and_then(|m| {
        m.lines()
            .find_map(|l| l.strip_prefix("MemTotal:"))
            .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
            .map(|kib| kib << 10)
    });
    spill_target_from(limit, mem_total)
}

/// The smaller of a cgroup limit and `MemTotal`, less 10 GiB; no spill
/// (`u64::MAX`) when neither is known. A v1 cgroup without a limit reads a
/// sentinel far above `MemTotal`, which the minimum discards.
fn spill_target_from(limit: Option<u64>, mem_total: Option<u64>) -> u64 {
    const MARGIN: u64 = 10 << 30;
    match (limit, mem_total) {
        (Some(limit), Some(total)) => limit.min(total),
        (Some(bytes), None) | (None, Some(bytes)) => bytes,
        (None, None) => return u64::MAX,
    }
    .saturating_sub(MARGIN)
}

/// Where the cgroup filesystem is mounted.
const CGROUP_ROOT: &str = "/sys/fs/cgroup";

/// One number of a cgroup's memory files: a file that holds just the number,
/// or one `key value` line of `memory.stat`.
#[derive(Clone, Copy)]
enum CgroupValue<'a> {
    File(&'a str),
    Stat(&'a str),
}

impl CgroupValue<'_> {
    fn read(self, dir: &std::path::Path) -> Option<u64> {
        let (file, key) = match self {
            CgroupValue::File(file) => (file, None),
            CgroupValue::Stat(key) => ("memory.stat", Some(key)),
        };
        let text = std::fs::read_to_string(dir.join(file)).ok()?;
        match key {
            None => text.trim().parse::<u64>().ok(),
            Some(key) => text.lines().find_map(|l| {
                let (k, v) = l.split_once(' ')?;
                (k == key).then(|| v.trim().parse::<u64>().ok()).flatten()
            }),
        }
    }
}

/// A number from this process's cgroup memory files: `v2` under the unified
/// hierarchy (`memory.max`, `memory.current`, `memory.stat`'s `inactive_file`),
/// else `v1` under the memory controller (`memory.limit_in_bytes`,
/// `memory.usage_in_bytes`, `memory.stat`'s `total_inactive_file`).
/// `proc_cgroup` is `/proc/self/cgroup`, `root` the cgroup mount. Each is read
/// at the process's cgroup path, then at its hierarchy's root, which a
/// container without a cgroup namespace sees as its own cgroup. `None` where
/// neither reads as a number (v2's `max` included).
fn cgroup_memory(
    proc_cgroup: &str,
    root: &std::path::Path,
    v2: CgroupValue,
    v1: CgroupValue,
) -> Option<u64> {
    let at = |base: &std::path::Path, path: &str, value: CgroupValue| {
        value
            .read(&base.join(path.trim().trim_start_matches('/')))
            .or_else(|| value.read(base))
    };
    let (mut unified, mut memory) = (None, None);
    for line in proc_cgroup.lines() {
        let mut fields = line.splitn(3, ':');
        let (Some(id), Some(controllers), Some(path)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        if id == "0" && controllers.is_empty() {
            unified = Some(path);
        } else if controllers.split(',').any(|c| c == "memory") {
            memory = Some(path);
        }
    }
    unified
        .and_then(|path| at(root, path, v2))
        .or_else(|| memory.and_then(|path| at(&root.join("memory"), path, v1)))
}

/// What `auto` reads of the host ([`HostReading::bytes`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct HostReading {
    /// The process's peak resident set (`VmHWM`).
    hwm: u64,
    /// The cgroup's charge (`memory.current`, or v1's `memory.usage_in_bytes`),
    /// page cache included.
    charged: u64,
    /// Its inactive file pages (`memory.stat`'s `inactive_file`, or v1's
    /// `total_inactive_file`): the page cache the kernel reclaims first.
    inactive_file: u64,
}

impl HostReading {
    fn now() -> Self {
        let hwm = std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find_map(|l| l.strip_prefix("VmHWM:"))
                    .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
                    .map(|kib| kib << 10)
            });
        let proc_cgroup = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
        let root = std::path::Path::new(CGROUP_ROOT);
        let charged = cgroup_memory(
            &proc_cgroup,
            root,
            CgroupValue::File("memory.current"),
            CgroupValue::File("memory.usage_in_bytes"),
        );
        let inactive_file = cgroup_memory(
            &proc_cgroup,
            root,
            CgroupValue::Stat("inactive_file"),
            CgroupValue::Stat("total_inactive_file"),
        );
        Self {
            hwm: hwm.unwrap_or(0),
            charged: charged.unwrap_or(0),
            inactive_file: inactive_file.unwrap_or(0),
        }
    }

    /// The host's bytes: the larger of the peak resident set (the resident set
    /// itself is not monotone under the allocator's posture, which drops and
    /// re-faults recycled extents) and the cgroup's working set, its charge
    /// less its inactive file pages (the kubelet's measure). Page cache the
    /// kernel would give back first is not counted; active file pages are,
    /// and the target's 10 GiB margin covers them.
    fn bytes(&self) -> u64 {
        self.hwm
            .max(self.charged.saturating_sub(self.inactive_file))
    }
}

/// `auto`'s reserve beside the host's bytes: what the rest of the block will
/// still need once the instances committed so far (`cells` main cells) are
/// on the host. With the finish's tables committed in phase A (and spilled
/// by the same policy), that is the finish's transients, ≈ 0.18 GiB per G
/// cells committed (0.135 per total G cells ÷ the streamed share 0.763,
/// I-MEM §8), plus 6 GiB for phase B's bump and the read-back window.
fn spill_reserve_bytes(cells: u64) -> u64 {
    const PER_G: f64 = 0.18 * (1u64 << 30) as f64;
    (cells as f64 / 1e9 * PER_G) as u64 + (6 << 30)
}

/// The policy's choice for one committed instance of `bytes` packed bytes:
/// `resident` are the committed packed bytes kept so far, `cells` the main
/// cells committed so far (this instance's included), `host` the host's bytes
/// ([`HostReading::bytes`]), read only by `auto`.
fn spill_decision(
    policy: SpillPolicy,
    target: u64,
    resident: u64,
    cells: u64,
    bytes: u64,
    host: impl FnOnce() -> u64,
) -> bool {
    match policy {
        SpillPolicy::Off => false,
        SpillPolicy::Always => true,
        SpillPolicy::Budget(budget) => resident + bytes > budget,
        SpillPolicy::Auto => {
            host()
                .saturating_add(spill_reserve_bytes(cells))
                .saturating_add(bytes)
                > target
        }
    }
}

/// The upstream queues' byte budgets while spilling: the ops waiting for a
/// generator, and the generated chunks waiting for a committer. A disk slower
/// than the stream fills the spill's writer queue, the committers wait on it,
/// and these make the walk wait in turn instead of piling work up.
const SPILL_QUEUE_BYTES: usize = 4 << 30;
const SPILL_READY_BYTES: usize = 2 << 30;

/// The budgets above arm only once a spill is plausible: always under a policy
/// that spills regardless of the host (`always`, a budget), and under `auto`
/// once the host and the reserve reach this share of the target. Where nothing
/// can spill (a block far below the target) the queues run as with the spill
/// off.
const SPILL_ARM_SHARE: f64 = 0.85;

/// Whether a spill is plausible ([`SPILL_ARM_SHARE`]): `host` the host's bytes
/// ([`HostReading::bytes`]), `reserve` [`spill_reserve_bytes`], read only by
/// `auto`.
fn spill_plausible(
    policy: SpillPolicy,
    target: u64,
    host_and_reserve: impl FnOnce() -> u64,
) -> bool {
    match policy {
        SpillPolicy::Off => false,
        SpillPolicy::Always | SpillPolicy::Budget(_) => true,
        SpillPolicy::Auto => host_and_reserve() as f64 >= SPILL_ARM_SHARE * target as f64,
    }
}

/// What arms a queue's byte budget ([`QueueRoom::admit`]).
trait Arming: Sync {
    /// Whether the budget binds now. Once true it stays true.
    fn armed(&self) -> bool;
}

/// Phase A's spill: the policy, its store, and what it decided.
struct Spill {
    policy: SpillPolicy,
    target: u64,
    store: stark::spill::SpillStore,
    /// Committed packed bytes kept on the host, and committed main cells.
    resident: std::sync::atomic::AtomicU64,
    cells: std::sync::atomic::AtomicU64,
    /// (instances, bytes) spilled, and instances kept.
    spilled: std::sync::Mutex<(u64, u64, u64)>,
    /// The largest host reading `auto` decided on.
    host_most: std::sync::Mutex<Option<HostReading>>,
    /// When the store opened, and the seconds after it at which the queue
    /// budgets armed ([`Arming`]).
    opened: Instant,
    armed_at: std::sync::Mutex<Option<f64>>,
}

impl Spill {
    /// The policy's store, opened; `None` when the policy is `off`, or when
    /// the store will not open (the block then runs resident, as without a
    /// policy, and says why).
    fn open(policy: SpillPolicy) -> Option<Self> {
        if policy == SpillPolicy::Off {
            return None;
        }
        // `disk-spill`'s storage keeps a trace's words in an mmap of its own;
        // a spilled trace keeps none in its table. The two are not combined
        // (I-SPILL R13): with that feature the block runs resident.
        if cfg!(feature = "disk-spill") {
            eprintln!(
                "BLOCK SPILL: {policy:?} wanted, but not with the disk-spill feature: resident"
            );
            return None;
        }
        match stark::spill::SpillStore::open(stark::spill::SpillOptions::default()) {
            Ok(store) => Some(Self {
                policy,
                target: spill_target_bytes(),
                store,
                resident: std::sync::atomic::AtomicU64::new(0),
                cells: std::sync::atomic::AtomicU64::new(0),
                spilled: std::sync::Mutex::new((0, 0, 0)),
                host_most: std::sync::Mutex::new(None),
                opened: Instant::now(),
                armed_at: std::sync::Mutex::new(None),
            }),
            Err(e) => {
                eprintln!(
                    "BLOCK SPILL: {policy:?} wanted, but the store did not open ({e}): resident"
                );
                None
            }
        }
    }

    /// A committed instance's packed trace: spilled when the policy says so
    /// and the store takes it. Returns the bytes spilled (0: kept).
    fn consider(&self, trace: &mut finish_sink::Trace) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(bytes) = trace.narrow_main().map(|t| t.data().len()) else {
            return 0;
        };
        let cells = (trace.num_rows() * trace.num_main_columns) as u64;
        let cells = self.cells.fetch_add(cells, Relaxed) + cells;
        let resident = self.resident.load(Relaxed);
        let wanted = spill_decision(
            self.policy,
            self.target,
            resident,
            cells,
            bytes as u64,
            || {
                let reading = HostReading::now();
                let mut most = self.host_most.lock().unwrap_or_else(|e| e.into_inner());
                if most.is_none_or(|m| reading.bytes() > m.bytes()) {
                    *most = Some(reading);
                }
                reading.bytes()
            },
        );
        let mut s = self.spilled.lock().unwrap_or_else(|e| e.into_inner());
        if wanted && trace.spill_main(&self.store) {
            s.0 += 1;
            s.1 += bytes as u64;
            bytes
        } else {
            s.2 += 1;
            self.resident.fetch_add(bytes as u64, Relaxed);
            0
        }
    }

    /// The `BLOCK SPILL` line: the policy, its decisions and the store.
    fn report(&self) -> String {
        let (n, bytes, kept) = *self.spilled.lock().unwrap_or_else(|e| e.into_inner());
        let g = |b: u64| b as f64 / (1u64 << 30) as f64;
        let host = match *self.host_most.lock().unwrap_or_else(|e| e.into_inner()) {
            Some(h) => format!(
                " · host at most {:.2} GiB (VmHWM {:.2} · charge {:.2} − inactive file {:.2})",
                g(h.bytes()),
                g(h.hwm),
                g(h.charged),
                g(h.inactive_file),
            ),
            None => String::new(),
        };
        format!(
            "{:?} · target {:.1} GiB · spilled {n} instances {:.2} GiB · kept {kept} ({:.2} GiB packed){host} · {}",
            self.policy,
            g(self.target),
            g(bytes),
            g(self.resident.load(std::sync::atomic::Ordering::Relaxed)),
            self.store.stats(),
        )
    }
}

impl Arming for Spill {
    fn armed(&self) -> bool {
        let mut armed_at = self.armed_at.lock().unwrap_or_else(|e| e.into_inner());
        if armed_at.is_some() {
            return true;
        }
        let plausible = spill_plausible(self.policy, self.target, || {
            let cells = self.cells.load(std::sync::atomic::Ordering::Relaxed);
            HostReading::now()
                .bytes()
                .saturating_add(spill_reserve_bytes(cells))
        });
        if plausible {
            *armed_at = Some(self.opened.elapsed().as_secs_f64());
            // Memory is short: `auto` returns the allocator's freed pages at
            // phase A's end and the base's end (`alloc_purge`).
            crate::alloc_purge::note_memory_pressure();
        }
        plausible
    }
}

impl Spill {
    /// The `BLOCK QUEUE` line's word on the budgets.
    fn arming_report(&self) -> String {
        match *self.armed_at.lock().unwrap_or_else(|e| e.into_inner()) {
            Some(t) => format!(" · budgets armed at {t:.1} s"),
            None => " · budgets never armed".to_string(),
        }
    }
}

/// The streamed chunks waiting for a committer (or, generated, for a
/// committer to take them), by their bytes: the most held at once is reported.
/// A chunk leaves once it is generated (its ops are freed then), or once a
/// committer takes it.
struct QueueRoom<'a> {
    /// (bytes, chunks) held now.
    held: std::sync::Mutex<(usize, usize)>,
    /// The most bytes, and the most chunks, held at once.
    most: std::sync::Mutex<(usize, usize)>,
    /// With a budget, once it is armed, [`Self::admit`] waits until the chunk
    /// fits beside what is held (a chunk alone always fits): the spill's
    /// back-pressure upstream, so a disk slower than the stream throttles the
    /// walk instead of piling ops and generated chunks up
    /// ([`SPILL_QUEUE_BYTES`], [`SPILL_ARM_SHARE`]).
    budget: Option<(usize, &'a dyn Arming)>,
    room: std::sync::Condvar,
    /// Seconds spent waiting for room.
    waited: std::sync::Mutex<f64>,
}

impl<'a> QueueRoom<'a> {
    fn with_budget(budget: Option<(usize, &'a dyn Arming)>) -> Self {
        Self {
            held: std::sync::Mutex::new((0, 0)),
            most: std::sync::Mutex::new((0, 0)),
            budget,
            room: std::sync::Condvar::new(),
            waited: std::sync::Mutex::new(0.0),
        }
    }

    /// A chunk of `bytes` joined the queue (once it fits, with a budget).
    fn admit(&self, bytes: usize) {
        let t = Instant::now();
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        while let Some((budget, arming)) = self.budget {
            if held.0 == 0 || held.0 + bytes <= budget || !arming.armed() {
                break;
            }
            held = self.room.wait(held).unwrap_or_else(|e| e.into_inner());
        }
        held.0 += bytes;
        held.1 += 1;
        let mut most = self.most.lock().unwrap_or_else(|e| e.into_inner());
        *most = (most.0.max(held.0), most.1.max(held.1));
        if self.budget.is_some() {
            *self.waited.lock().unwrap_or_else(|e| e.into_inner()) += t.elapsed().as_secs_f64();
        }
    }

    /// A chunk of `bytes` left the queue.
    fn release(&self, bytes: usize) {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        held.0 = held.0.saturating_sub(bytes);
        held.1 = held.1.saturating_sub(1);
        self.room.notify_all();
    }
}

/// `LAMBDA_VM_BLOCK_MEMLOG=1`: [`MemLedger`]'s `BLOCK MEM` lines, every half
/// second from the block's start to its proof and at the phase marks. A
/// measurement knob, off by default.
fn memlog() -> bool {
    std::env::var("LAMBDA_VM_BLOCK_MEMLOG").is_ok_and(|v| v.trim() == "1")
}

/// Where the block's host memory is: what phase A holds in the places it knows
/// of, printed ([`MemLedger::line`]) beside the process's resident set and,
/// where the allocator can be read ([`heap_stats`]), its live, active and
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
    /// Committed instances' packed traces sent to the spill file.
    committed_spilled: std::sync::atomic::AtomicUsize,
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
            committed_spilled: AtomicUsize::new(0),
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

    /// A chunk of `bytes` counted by [`Self::queue`] was never sent.
    fn unqueue(&self, bytes: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        self.queued.fetch_sub(1, Relaxed);
        self.queued_bytes.fetch_sub(bytes, Relaxed);
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

    /// A committed instance's packed trace went to the spill file instead of
    /// staying here: `bytes` move from the committed packed to the spilled.
    fn spilled(&self, bytes: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        self.committed_packed.fetch_sub(bytes, Relaxed);
        self.committed_spilled.fetch_add(bytes, Relaxed);
    }

    /// One line of `what`'s parts, largest first, those of at least 0.01 GiB.
    fn parts(&self, what: &str, mut parts: Vec<(String, usize)>) {
        const GIB: f64 = (1u64 << 30) as f64;
        let total: usize = parts.iter().map(|(_, b)| b).sum();
        parts.sort_by(|a, b| b.1.cmp(&a.1));
        let listed: Vec<String> = parts
            .iter()
            .filter(|(_, b)| *b as f64 >= 0.01 * GIB)
            .map(|(name, b)| format!("{name} {:.2}", *b as f64 / GIB))
            .collect();
        eprintln!(
            "BLOCK MEM {what} parts: {:.2} GiB · {}",
            total as f64 / GIB,
            listed.join(" · ")
        );
    }

    fn line(&self, label: &str) {
        use std::sync::atomic::Ordering::Relaxed;
        const GIB: f64 = (1u64 << 30) as f64;
        let g = |a: &std::sync::atomic::AtomicUsize| a.load(Relaxed) as f64 / GIB;
        let rss = proc_rss_bytes().map_or("n/a".to_string(), |b| format!("{:.2}", b as f64 / GIB));
        let faults =
            proc_minor_faults().map_or("n/a".to_string(), |f| format!("{:.2}", f as f64 / 1e6));
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
            "BLOCK MEM {label} t={:.1} · rss {rss} · minflt {faults} M · {heap} · queue {} chunks {:.2} · committing {:.2} · \
             ready {} packed {:.2} · committed {} ({:.2} packed + {:.2} 64-bit) · spilled {:.2} · logs {:.2} · walked {:.2} · \
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
            g(&self.committed_spilled),
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

/// This process's minor page faults so far (`/proc/self/stat`, field 10), on
/// Linux: each a first touch of a page the allocator had not kept.
fn proc_minor_faults() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The command name may hold spaces; the fields after it do not.
    let after = stat.rsplit_once(')')?.1;
    after.split_whitespace().nth(7)?.parse().ok()
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

/// The allocator's allocated, active, resident, mapped and retained bytes,
/// when it can be read ([`crate::alloc_purge::stats`]: the binary's hooks, or
/// the jemalloc the lib's tests install).
fn heap_stats() -> Option<[usize; 5]> {
    crate::alloc_purge::stats().map(|s| [s.allocated, s.active, s.resident, s.mapped, s.retained])
}

/// The bytes a streamed chunk on its way to a committer holds.
fn streamed_bytes(streamed: &Streamed) -> usize {
    match streamed {
        Streamed::Job(job) => job.op_bytes(),
        Streamed::Chunk(chunk) => wide_bytes(&chunk.trace),
        Streamed::Finished(table) => held_bytes(&table.trace),
    }
}

/// A main trace's bytes as held: packed, or 64-bit.
fn held_bytes(trace: &finish_sink::Trace) -> usize {
    trace
        .narrow_main()
        .map_or_else(|| wide_bytes(trace), |t| t.data().len())
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
/// (the `push` arm) a chunk the producer generated, or a table the finish
/// handed off ([`BlockFinishSink`]), generated already.
enum Streamed {
    Job(ChunkJob),
    Chunk(StreamedChunk),
    Finished(FinishedTable),
}

/// A generated table on its way from a generator to a committer: the table,
/// the bytes it holds, and whether the finish handed it off.
struct ToCommit {
    table: FinishedTable,
    held: usize,
    finished: bool,
}

/// The finish's sink in phase A: each table it is handed goes into the
/// committers' queue, behind the streamed chunks, and is counted in the
/// ledger as queued. It never waits (the finish calls it from its rayon
/// workers): the ops budget does not apply to a table that is already built.
/// Once the queue is closed (a committer stopped) it gives the table back.
struct BlockFinishSink<'a> {
    tx: std::sync::mpsc::Sender<Streamed>,
    ledger: Option<&'a MemLedger>,
}

impl FinishSink for BlockFinishSink<'_> {
    fn hand(&self, table: FinishedTable) -> Option<FinishedTable> {
        let bytes = held_bytes(&table.trace);
        if let Some(ledger) = self.ledger {
            ledger.queue(bytes);
        }
        match self.tx.send(Streamed::Finished(table)) {
            Ok(()) => None,
            Err(refused) => {
                if let Some(ledger) = self.ledger {
                    ledger.unqueue(bytes);
                }
                match refused.0 {
                    Streamed::Finished(table) => Some(table),
                    Streamed::Job(_) | Streamed::Chunk(_) => None,
                }
            }
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
    spill: Option<&Spill>,
) -> Result<Produced, Error> {
    use std::sync::Mutex;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::mpsc;

    let window = max_rows.cpu;
    let t = Instant::now();
    let (job_tx, job_rx) = mpsc::channel::<Streamed>();
    let job_rx = Mutex::new(job_rx);
    let committer_count = stream.committers;
    // While spilling, both queues have a byte budget: a slow disk throttles
    // the walk ([`SPILL_QUEUE_BYTES`]).
    let queue = QueueRoom::with_budget(spill.map(|s| (SPILL_QUEUE_BYTES, s as &dyn Arming)));
    let generators = stream.generators;
    let ready = QueueRoom::with_budget(spill.map(|s| (SPILL_READY_BYTES, s as &dyn Arming)));
    let (ready_tx, ready_rx) = mpsc::channel::<ToCommit>();
    let ready_rx = Mutex::new(ready_rx);
    // Phase A's card gate ([`card_gate_budget`]): every commit admitted by its
    // device set; the most it held and the committers' wait are reported.
    let card = CardGate::new(stream.card_budget);
    // The committers' own pool ([`commit_pool_threads`]). Each committer is an
    // OS thread of this scope, never a worker of either pool, so `install`
    // only parks the committer until its commit is done.
    #[cfg(feature = "parallel")]
    let commit_pool = match stream.commit_pool {
        0 => None,
        n => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .thread_name(|i| format!("block-commit-{i}"))
                .build()
                .map_err(|e| Error::Prover(format!("the committers' pool: {e}")))?,
        ),
    };
    #[cfg(not(feature = "parallel"))]
    let _ = stream.commit_pool;
    // Tables the finish handed off and the committers committed, and their
    // seconds from arrival to commit.
    let finished_committed = Mutex::new((0usize, 0.0f64));
    let committed: Mutex<Vec<Committed>> = Mutex::new(Vec::new());
    let commit_secs = Mutex::new(0.0f64);
    let generate_secs = Mutex::new(0.0f64);
    // Packed instances: (wide bytes, packed bytes, seconds packing on the
    // host, instances the device packed).
    let narrowed = Mutex::new((0usize, 0usize, 0.0f64, 0usize));
    let narrow = narrow_streamed();
    // The form the generators write each streamed chunk in: packed as it is
    // generated under G-pack, else at 8 bytes a cell (packed below).
    let form = if narrow && stream.gpack {
        TraceForm::Narrow
    } else {
        TraceForm::Wide
    };
    gpack::reset_counts();
    std::thread::scope(|s| {
        // The executor, window by window, a bounded two windows ahead.
        let (log_tx, log_rx) = mpsc::sync_channel::<Vec<executor::vm::logs::Log>>(2);
        let exec = spawn_named(
            s,
            "block-exec".to_string(),
            move || -> Result<f64, Error> {
                let t = Instant::now();
                let mut executor = Executor::new(program, private_input.to_vec())
                    .map_err(|e| Error::Execution(format!("{e}")))?;
                while let Some(logs) = executor
                    .resume_with_limit(window)
                    .map_err(|e| Error::Execution(format!("{e}")))?
                {
                    let logs = logs.to_vec();
                    let logs_bytes =
                        logs.capacity() * std::mem::size_of::<executor::vm::logs::Log>();
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
            },
        );
        let decode = s.spawn(|| {
            crate::tables::decode::commitment_from_elf_device_or_host(program, opts)
                .map_err(|e| Error::Recursion(format!("DECODE commitment from ELF: {e}")))
        });
        if warm_precomputed() {
            s.spawn(|| warm_data_page_commitments(program, opts));
        }

        // Generators ([`stream_generators`]): each streamed chunk generated,
        // and packed under narrow storage, ahead of the committers.
        for g in 0..generators {
            let (job_rx, queue, ready, narrowed, generate_secs) =
                (&job_rx, &queue, &ready, &narrowed, &generate_secs);
            let ready_tx = ready_tx.clone();
            spawn_named(s, format!("gen-{g}"), move || {
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
                    // A table the finish handed off is built already, and it
                    // never took room in the ops queue.
                    let (mut table, finished) = match job {
                        Streamed::Job(job) => (FinishedTable::from(job.generate_as(form)), false),
                        Streamed::Chunk(chunk) => (FinishedTable::from(chunk), false),
                        Streamed::Finished(table) => (table, true),
                    };
                    if !finished {
                        queue.release(job_bytes);
                    }
                    // As generated: packed under G-pack, else 64-bit.
                    let wide = held_bytes(&table.trace);
                    if let Some(ledger) = ledger {
                        ledger.generated(job_bytes, wide);
                    }
                    if !finished {
                        *generate_secs.lock().unwrap_or_else(|e| e.into_inner()) +=
                            t.elapsed().as_secs_f64();
                    }
                    let tp = Instant::now();
                    if narrow
                        && table.trace.narrow_main().is_none()
                        && table.trace.pack_main_narrow()
                    {
                        narrowed.lock().unwrap_or_else(|e| e.into_inner()).2 +=
                            tp.elapsed().as_secs_f64();
                    }
                    let held = held_bytes(&table.trace);
                    if let Some(ledger) = ledger {
                        ledger.ready(wide, held);
                    }
                    ready.admit(held);
                    if let Err(refused) = ready_tx.send(ToCommit {
                        table,
                        held,
                        finished,
                    }) {
                        ready.release(refused.0.held);
                    }
                }
            });
        }
        drop(ready_tx);

        // Committers: each streamed chunk generated (unless a generator did)
        // and Round-1 committed, as the chunks complete.
        let mut committers = Vec::new();
        for c in 0..committer_count {
            committers.push(spawn_named(
                s,
                format!("commit-{c}"),
                || -> Result<(), Error> {
                    let commit_all = || -> Result<(), Error> {
                        loop {
                            // `t`: from the chunk's arrival (its generation
                            // included when this committer generates it).
                            let (mut chunk, wide, held, finished, t) = if generators > 0 {
                                let next =
                                    ready_rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                                let Ok(ToCommit {
                                    table,
                                    held,
                                    finished,
                                }) = next
                                else {
                                    return Ok(());
                                };
                                let t = Instant::now();
                                ready.release(held);
                                if let Some(ledger) = ledger {
                                    ledger.take_ready(held);
                                }
                                let wide = wide_bytes(&table.trace);
                                (table, wide, held, finished, t)
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
                                let (table, finished) = match job {
                                    Streamed::Job(job) => {
                                        (FinishedTable::from(job.generate()), false)
                                    }
                                    Streamed::Chunk(chunk) => (FinishedTable::from(chunk), false),
                                    Streamed::Finished(table) => (table, true),
                                };
                                let wide = wide_bytes(&table.trace);
                                let held = if finished {
                                    job_bytes
                                } else {
                                    queue.release(job_bytes);
                                    if let Some(ledger) = ledger {
                                        ledger.generated(job_bytes, wide);
                                    }
                                    *generate_secs.lock().unwrap_or_else(|e| e.into_inner()) +=
                                        t.elapsed().as_secs_f64();
                                    wide
                                };
                                (table, wide, held, finished, t)
                            };
                            let air = finish_sink::air_for(chunk.kind, chunk.index, opts);
                            let name = air.name().to_string();
                            let on_card = commit_card_bytes(&air, &chunk.trace);
                            card.admit(on_card);
                            let commit = || {
                                crate::hash_pin::BlockProver::precommit_main(
                                    air.as_ref(),
                                    &chunk.trace,
                                    #[cfg(feature = "disk-spill")]
                                    stark::storage_mode::StorageMode::Ram,
                                    residency,
                                )
                            };
                            #[cfg(feature = "parallel")]
                            let pre = match &commit_pool {
                                Some(pool) => pool.install(commit),
                                None => commit(),
                            };
                            #[cfg(not(feature = "parallel"))]
                            let pre = commit();
                            let pre = pre.map_err(|e| Error::Prover(format!("{e:?}")));
                            card.release(on_card);
                            #[allow(unused_mut)]
                            let mut pre = pre?;
                            *commit_secs.lock().unwrap_or_else(|e| e.into_inner()) +=
                                t.elapsed().as_secs_f64();
                            if finished {
                                let mut f =
                                    finished_committed.lock().unwrap_or_else(|e| e.into_inner());
                                f.0 += 1;
                                f.1 += t.elapsed().as_secs_f64();
                            }
                            if finished {
                                // What phase B's Round 1 does after the same commit:
                                // the device's packed copy replaces the 64-bit one.
                                if let Some(packed) = pre.take_narrow() {
                                    chunk.trace.install_main_narrow(packed);
                                }
                            } else if narrow {
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
                            let packed = chunk.trace.narrow_main().map(|t| t.data().len());
                            let spilled = spill.map_or(0, |spill| spill.consider(&mut chunk.trace));
                            if let Some(ledger) = ledger {
                                ledger.keep(held, packed);
                                if spilled > 0 {
                                    ledger.spilled(spilled);
                                }
                            }
                            committed
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .push((chunk, name, pre));
                        }
                    };
                    let result = commit_all();
                    if result.is_err() {
                        card.close();
                    }
                    result
                },
            ));
        }

        let produce = || -> Result<(Traces, f64), Error> {
            let mut builder = WindowedTraceBuilder::new(program, private_input, max_rows)?;
            if drop_streamed_ops() {
                builder = builder.drop_streamed_ops()?;
            }
            if narrow_finished() {
                builder = builder.pack_finished_tables();
                if stream.gpack {
                    builder = builder.generate_packed();
                }
                if !packed_builds() {
                    builder = builder.build_wide_then_pack().cap_kr_wide(kr_wide_cap());
                }
            }
            if lt_concat() {
                builder = builder.concat_lt();
            }
            if !compact_lt() {
                builder = builder.raw_memw_lt();
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
                    let walking = spawn_named(
                        inner,
                        "block-walk".to_string(),
                        move || -> Result<_, Error> {
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
                        },
                    );
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
            // The finish's sink keeps the committers' queue open until the
            // finish's tables are handed off.
            let sink = (stream.finish != FinishCommit::PhaseB).then(|| BlockFinishSink {
                tx: job_tx.clone(),
                ledger,
            });
            drop(job_tx);
            if let Some(ledger) = ledger {
                ledger.line("windows walked");
                ledger.parts("builder", builder.heap_parts());
            }
            let windows = builder.stamps();
            let t = Instant::now();
            let traces =
                builder.finish_handing(&last, sink.as_ref().map(|sink| sink as &dyn FinishSink))?;
            drop(sink);
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
        let (mut traces, collect_secs) = produced?;
        if let Some(e) = errors.into_iter().next() {
            return Err(e);
        }
        times.execute = join(exec)?;
        let decode_commitment = join(decode)?;

        let committed: Vec<Committed> =
            std::mem::take(&mut *committed.lock().unwrap_or_else(|e| e.into_inner()));
        let (n_finished, finished_secs) =
            *finished_committed.lock().unwrap_or_else(|e| e.into_inner());
        let n = committed.len() - n_finished;
        let mut tables = Vec::with_capacity(committed.len());
        let mut precommits = Vec::with_capacity(committed.len());
        for (table, name, pre) in committed {
            tables.push(table);
            precommits.push((name, pre));
        }
        finish_sink::insert_finished(&mut traces, tables)?;
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
            // With generators on, they pack every table they hand over (the
            // finish's included); the committers only pack without them.
            let packers = if generators > 0 {
                "generators"
            } else {
                "committers"
            };
            let [direct, narrowed_after, rewritten, wide_then_packed] = gpack::counts();
            eprintln!(
                "BLOCK NARROW: streamed main traces {:.2} GiB packed to {:.2} GiB ({:.3} B/cell) · \
                 {by_device} of {n} packed by the device · host packing {secs:.2} s on the {packers} · \
                 G-pack {}: {direct} written packed, {narrowed_after} narrowed after, {rewritten} \
                 written again, {wide_then_packed} built wide then packed",
                wide as f64 / (1u64 << 30) as f64,
                packed as f64 / (1u64 << 30) as f64,
                if wide > 0 {
                    packed as f64 * 8.0 / wide as f64
                } else {
                    0.0
                },
                if form == TraceForm::Narrow {
                    "on"
                } else {
                    "off"
                },
            );
        }
        let (card_most, card_commits) = *card.most.lock().unwrap_or_else(|e| e.into_inner());
        eprintln!(
            "BLOCK CARD GATE: budget {} · at most {:.2} GiB in {card_commits} commits at once · the \
             committers waited {:.2} s for the card · {n_finished} tables of the finish committed in \
             phase A ({finished_secs:.2} s on the committers)",
            card.budget.map_or("none".to_string(), |b| format!(
                "{:.2} GiB",
                b as f64 / (1u64 << 30) as f64
            )),
            card_most as f64 / (1u64 << 30) as f64,
            *card.waited.lock().unwrap_or_else(|e| e.into_inner()),
        );
        let (most_bytes, most_chunks) = *queue.most.lock().unwrap_or_else(|e| e.into_inner());
        let (ready_bytes, ready_chunks) = *ready.most.lock().unwrap_or_else(|e| e.into_inner());
        eprintln!(
            "BLOCK QUEUE: {committer_count} committers · {generators} generators · at most {:.2} GiB in \
             {most_chunks} chunks waiting as ops · at most {:.2} GiB in {ready_chunks} chunks waiting \
             generated · waited for room {:.2} s (ops) {:.2} s (generated){}",
            most_bytes as f64 / (1u64 << 30) as f64,
            ready_bytes as f64 / (1u64 << 30) as f64,
            *queue.waited.lock().unwrap_or_else(|e| e.into_inner()),
            *ready.waited.lock().unwrap_or_else(|e| e.into_inner()),
            spill.map(Spill::arming_report).unwrap_or_default(),
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
    // The heap before the AIRs exist (memlog): the setup line splits what
    // they add from what phase A left.
    let live_before_airs = memlog().then(heap_stats).flatten().map(|[live, ..]| live);
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
        // The main traces the prove starts from: packed, spilled (their bytes
        // in the heap only while a slot still holds them), and at 8 bytes a
        // cell.
        let (mut packed, mut wide, mut n_packed) = (0usize, 0usize, 0usize);
        let (mut spilled, mut spilled_resident, mut n_spilled) = (0usize, 0usize, 0usize);
        for (_, trace, _) in &pairs {
            if let Some(narrow) = trace.narrow_main() {
                packed += narrow.data().len();
                n_packed += 1;
            } else if let Some(slot) = trace.spilled_main() {
                spilled += slot.len();
                if slot.is_resident() {
                    spilled_resident += slot.len();
                }
                n_spilled += 1;
            } else {
                wide += wide_bytes(trace);
            }
        }
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        eprintln!(
            "BLOCK MEM traces: {} instances · {n_packed} packed {:.2} GiB · {n_spilled} spilled \
             {:.2} GiB ({:.2} in memory) · {} at 8 B/cell {:.2} GiB",
            pairs.len(),
            gib(packed),
            gib(spilled),
            gib(spilled_resident),
            pairs.len() - n_packed - n_spilled,
            gib(wide),
        );
        // The traces' bytes in the heap.
        let traces_held = packed + spilled_resident + wide;
        // What else the heap holds as the prove starts: the precommits (the
        // streamed and phase-A-committed instances'), the precomputed-tree
        // cache, and what no gauge names (the AIRs just built included).
        let mut held = [0usize; 4];
        for (_, pre) in &precommits {
            for (h, b) in held.iter_mut().zip(pre.host_bytes()) {
                *h += b;
            }
        }
        let cache = stark::prover::precomputed_tree_cache_bytes_inserted() as usize;
        let g = |b: usize| b as f64 / (1u64 << 30) as f64;
        eprintln!(
            "BLOCK MEM precommits: {} · trees {:.2} · kept tops {:.2} · cached main {:.2} · packed copies {:.2} (GiB)",
            precommits.len(),
            g(held[0]),
            g(held[1]),
            g(held[2]),
            g(held[3]),
        );
        if let Some([live, ..]) = heap_stats() {
            let airs = live_before_airs.map_or(0, |before| live.saturating_sub(before));
            let named = traces_held + held.iter().sum::<usize>() + cache + airs;
            eprintln!(
                "BLOCK MEM setup: heap live {:.2} = traces {:.2} + precommits {:.2} + precomputed trees {:.2} + the AIRs and their pairing {:.2} + unnamed {:.2} (GiB)",
                g(live),
                g(traces_held),
                g(held.iter().sum()),
                g(cache),
                g(airs),
                g(live.saturating_sub(named)),
            );
        }
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
/// committers and `generators` generators, for the tests: the traces, and the
/// AIR of each streamed instance precommitted.
#[cfg(test)]
pub(crate) fn stream_for_test(
    program: &Elf,
    opts: &ProofOptions,
    max_rows: &MaxRowsConfig,
    committers: usize,
    generators: usize,
) -> Result<(Traces, Vec<String>), Error> {
    stream_config_for_test(
        program,
        opts,
        max_rows,
        StreamConfig {
            committers,
            generators,
            finish: FinishCommit::PhaseB,
            card_budget: None,
            commit_pool: 0,
            gpack: true,
        },
    )
}

/// [`stream_for_test`] with the finish's tables handed to phase A's
/// committers as the finish builds them ([`FinishCommit::Handed`]), the
/// commits admitted through a card gate of `card_budget` bytes and made in a
/// pool of `commit_pool` threads of their own (0: the global pool), and the
/// ledger on.
#[cfg(test)]
pub(crate) fn stream_finish_commit_for_test(
    program: &Elf,
    opts: &ProofOptions,
    max_rows: &MaxRowsConfig,
    committers: usize,
    generators: usize,
    card_budget: Option<usize>,
    commit_pool: usize,
) -> Result<(Traces, Vec<String>), Error> {
    stream_config_for_test(
        program,
        opts,
        max_rows,
        StreamConfig {
            committers,
            generators,
            finish: FinishCommit::Handed,
            card_budget,
            commit_pool,
            gpack: true,
        },
    )
}

#[cfg(test)]
fn stream_config_for_test(
    program: &Elf,
    opts: &ProofOptions,
    max_rows: &MaxRowsConfig,
    stream: StreamConfig,
) -> Result<(Traces, Vec<String>), Error> {
    stream_spill_for_test(program, opts, max_rows, stream, None).map(|(t, n, _)| (t, n))
}

/// Phase A's stream with the spill policy `policy` (`None`: no spill):
/// the traces (spilled ones still spilled), the instances precommitted, and
/// the `BLOCK SPILL` report.
#[cfg(test)]
pub(crate) fn stream_spill_for_test(
    program: &Elf,
    opts: &ProofOptions,
    max_rows: &MaxRowsConfig,
    stream: StreamConfig,
    policy: Option<SpillPolicy>,
) -> Result<(Traces, Vec<String>, String), Error> {
    let spill = policy.map(|p| Spill::open(p).expect("the spill store opens"));
    let ledger = (stream.finish != FinishCommit::PhaseB || spill.is_some()).then(MemLedger::new);
    let (traces, _, precommits) = build_streamed(
        program,
        &[],
        opts,
        max_rows,
        ResidencyMode::RecomputeLdeDevice,
        &mut BlockTimes::default(),
        &stream,
        ledger.as_ref(),
        spill.as_ref(),
    )?;
    let report = spill.as_ref().map(Spill::report).unwrap_or_default();
    if let Some(ledger) = &ledger {
        use std::sync::atomic::Ordering::Relaxed;
        // Every table the ledger saw queued was taken and kept.
        assert_eq!(ledger.queued.load(Relaxed), 0, "queued chunks left");
        assert_eq!(ledger.ready.load(Relaxed), 0, "ready chunks left");
        assert_eq!(
            ledger.committing_bytes.load(Relaxed),
            0,
            "bytes left committing"
        );
    }
    // Every precommit reports the host bytes it holds (BLOCK MEM precommits):
    // at least its commitment's Merkle nodes or kept top levels.
    for (name, pre) in &precommits {
        let [trees, tops, _, _] = pre.host_bytes();
        if trees + tops == 0 {
            return Err(Error::Prover(format!(
                "{name}: a precommit reports no host bytes"
            )));
        }
    }
    Ok((
        traces,
        precommits.into_iter().map(|(name, _)| name).collect(),
        report,
    ))
}

/// The stream's configuration for a test: `committers` / `generators`, the
/// finish's tables committed in phase A or B, no card gate, the global pool.
#[cfg(test)]
pub(crate) fn stream_config(
    committers: usize,
    generators: usize,
    finish_in_a: bool,
) -> StreamConfig {
    StreamConfig {
        committers,
        generators,
        finish: if finish_in_a {
            FinishCommit::Handed
        } else {
            FinishCommit::PhaseB
        },
        card_budget: None,
        commit_pool: 0,
        gpack: true,
    }
}

#[cfg(test)]
mod spill_policy_tests {
    use super::{
        CgroupValue, HostReading, SpillPolicy, cgroup_memory, parse_spill_policy, spill_decision,
        spill_plausible, spill_reserve_bytes, spill_target_from,
    };

    const GIB: u64 = 1 << 30;

    /// `LAMBDA_VM_BLOCK_SPILL`'s values: auto (and unset), off, always, a
    /// budget in GiB; anything else is off.
    #[test]
    fn the_spill_policy_reads_its_knob() {
        assert_eq!(parse_spill_policy(None), SpillPolicy::Auto);
        assert_eq!(parse_spill_policy(Some("off")), SpillPolicy::Off);
        assert_eq!(parse_spill_policy(Some("always")), SpillPolicy::Always);
        assert_eq!(parse_spill_policy(Some(" auto ")), SpillPolicy::Auto);
        assert_eq!(
            parse_spill_policy(Some("40")),
            SpillPolicy::Budget(40 * GIB)
        );
        assert_eq!(
            parse_spill_policy(Some("0.5")),
            SpillPolicy::Budget(GIB / 2)
        );
        assert_eq!(parse_spill_policy(Some("-1")), SpillPolicy::Off);
        assert_eq!(parse_spill_policy(Some("lots")), SpillPolicy::Off);
    }

    /// The cgroup memory files, v2 and v1, from fake `/proc/self/cgroup` texts
    /// and cgroup trees: v2 at the process's path, v2's `max` falling through
    /// to v1, v1 at its path or (a container without a cgroup namespace) at
    /// the controller's root, a v1 controller sharing its hierarchy, and none.
    #[test]
    fn the_cgroup_memory_files_read_v2_then_v1() {
        let root = std::env::temp_dir().join(format!("imem3-cgroup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let put = |rel: &str, file: &str, value: &str| {
            let dir = root.join(rel);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(file), value).unwrap();
        };
        let read = |proc_cgroup: &str, v2, v1| cgroup_memory(proc_cgroup, &root, v2, v1);
        let limit = |proc_cgroup: &str| {
            read(
                proc_cgroup,
                CgroupValue::File("memory.max"),
                CgroupValue::File("memory.limit_in_bytes"),
            )
        };
        let charge = |proc_cgroup: &str| {
            read(
                proc_cgroup,
                CgroupValue::File("memory.current"),
                CgroupValue::File("memory.usage_in_bytes"),
            )
        };
        let inactive = |proc_cgroup: &str| {
            read(
                proc_cgroup,
                CgroupValue::Stat("inactive_file"),
                CgroupValue::Stat("total_inactive_file"),
            )
        };
        // v2 at the process's path, its charge and its inactive file pages,
        // read by their exact key (a decoy ending in the same name comes
        // first).
        put("a/b", "memory.max", "129584070656\n");
        put("a/b", "memory.current", "4096\n");
        put(
            "a/b",
            "memory.stat",
            "anon 100\nfile 9000\nx_inactive_file 1\nactive_file 10\ninactive_file 8192\n",
        );
        assert_eq!(limit("0::/a/b\n"), Some(129_584_070_656));
        assert_eq!(charge("0::/a/b\n"), Some(4096));
        assert_eq!(inactive("0::/a/b\n"), Some(8192));
        // v2 unlimited and no v1: no limit.
        put("c", "memory.max", "max\n");
        assert_eq!(limit("0::/c\n"), None);
        // A hybrid host, as FAST: the unified line names a path with no
        // memory files, and the memory controller's own cgroup is its root.
        // v1's inactive file pages are its hierarchical key, not the local
        // one that ends in the same name.
        put("memory", "memory.limit_in_bytes", "61774757888\n");
        put("memory", "memory.usage_in_bytes", "17855025152\n");
        put(
            "memory",
            "memory.stat",
            "cache 16000\ninactive_file 5\ntotal_cache 16000\ntotal_inactive_file 15180000000\n",
        );
        let fast = "12:memory:/docker/d005\n9:cpu,cpuacct:/docker/d005\n0::/docker/d005\n";
        assert_eq!(limit(fast), Some(61_774_757_888));
        assert_eq!(charge(fast), Some(17_855_025_152));
        assert_eq!(inactive(fast), Some(15_180_000_000));
        // v1 at the process's path wins over the controller's root, and a
        // controller sharing its hierarchy is found.
        put("memory/docker/d005", "memory.limit_in_bytes", "777\n");
        assert_eq!(limit(fast), Some(777));
        put("memory/p", "memory.limit_in_bytes", "888\n");
        assert_eq!(limit("4:cpu,memory:/p\n"), Some(888));
        // v2's `max` falls through to v1.
        assert_eq!(limit("4:memory:/p\n0::/c\n"), Some(888));
        // Nothing readable.
        assert_eq!(limit(""), None);
        assert_eq!(limit("5:pids:/x\n"), None);
        assert_eq!(inactive("0::/c\n"), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// `auto`'s host bytes: the larger of the peak resident set and the
    /// cgroup's working set (its charge less its inactive file pages). FAST at
    /// 17:28Z: 40.58 GB charged with 15.18 GB inactive file reads 25.40 GB, not
    /// 40.58; inactive pages past the charge read zero; the peak resident set
    /// wins when it is larger.
    #[test]
    fn auto_reads_the_hosts_working_set() {
        let reading = |hwm, charged, inactive_file| HostReading {
            hwm,
            charged,
            inactive_file,
        };
        assert_eq!(
            reading(24_680_000_000, 40_580_000_000, 15_180_000_000).bytes(),
            25_400_000_000
        );
        assert_eq!(reading(0, 4 * GIB, 6 * GIB).bytes(), 0);
        assert_eq!(reading(30 * GIB, 40 * GIB, 15 * GIB).bytes(), 30 * GIB);
        assert_eq!(reading(0, 120 * GIB, 0).bytes(), 120 * GIB);
    }

    /// The queue budgets arm where a spill is plausible: always under
    /// `always` and a budget (the host unread), never with the spill off, and
    /// under `auto` once the host and reserve reach 85 % of the target (BIG's
    /// 110.7 GiB target: 94.1 GiB).
    #[test]
    fn the_budgets_arm_where_a_spill_is_plausible() {
        let never = || panic!("only auto reads the host");
        let target = 110 * GIB;
        assert!(!spill_plausible(SpillPolicy::Off, target, never));
        assert!(spill_plausible(SpillPolicy::Always, target, never));
        assert!(spill_plausible(
            SpillPolicy::Budget(40 * GIB),
            target,
            never
        ));
        assert!(!spill_plausible(SpillPolicy::Auto, target, || 93 * GIB));
        assert!(spill_plausible(SpillPolicy::Auto, target, || 94 * GIB));
        // The median's 1× block: ≈ 16 GiB of host, 6.6 GiB of reserve.
        assert!(!spill_plausible(SpillPolicy::Auto, target, || 23 * GIB));
    }

    /// The target is the smaller of the cgroup limit and `MemTotal`, less
    /// 10 GiB: v1's unlimited sentinel gives `MemTotal`'s, a limit above
    /// `MemTotal` is capped, and with neither nothing spills.
    #[test]
    fn the_spill_target_is_the_smaller_limit_less_ten_gib() {
        let total = 62_840_956u64 << 10;
        assert_eq!(
            spill_target_from(Some(61_774_757_888), Some(total)),
            61_774_757_888 - 10 * GIB
        );
        assert_eq!(
            spill_target_from(Some(9_223_372_036_854_771_712), Some(total)),
            total - 10 * GIB
        );
        assert_eq!(
            spill_target_from(Some(200 * GIB), Some(125 * GIB)),
            115 * GIB
        );
        assert_eq!(spill_target_from(None, Some(64 * GIB)), 54 * GIB);
        assert_eq!(
            spill_target_from(Some(129_584_070_656), None),
            129_584_070_656 - 10 * GIB
        );
        assert_eq!(spill_target_from(Some(8 * GIB), Some(16 * GIB)), 0);
        assert_eq!(spill_target_from(None, None), u64::MAX);
    }

    /// Off keeps, always spills, a budget spills what would pass it, and auto
    /// spills once the host, the reserve for what the block has committed and
    /// the instance would pass the target — and, the host flat, keeps spilling
    /// as the block commits more (its reserve only grows).
    #[test]
    fn the_spill_decision_follows_its_policy() {
        let never = || panic!("only auto reads the host");
        assert!(!spill_decision(SpillPolicy::Off, 0, 0, 0, GIB, never));
        assert!(spill_decision(
            SpillPolicy::Always,
            u64::MAX,
            0,
            0,
            1,
            never
        ));
        let budget = SpillPolicy::Budget(10 * GIB);
        assert!(!spill_decision(budget, 0, 9 * GIB, 0, GIB, never));
        assert!(spill_decision(budget, 0, 9 * GIB, 0, GIB + 1, never));
        let target = 110 * GIB;
        // The median at the walk's end: 72 GiB on the host, 23 G cells.
        assert!(!spill_decision(
            SpillPolicy::Auto,
            target,
            0,
            23_000_000_000,
            GIB / 10,
            || 72 * GIB
        ));
        // A block twice the median: the same host, more committed.
        assert!(spill_decision(
            SpillPolicy::Auto,
            target,
            0,
            60_000_000_000,
            GIB / 10,
            || 100 * GIB
        ));
        let mut spilling = false;
        for g in (1..400).map(|g| g * 1_000_000_000u64) {
            let now = spill_decision(SpillPolicy::Auto, target, 0, g, GIB / 10, || 95 * GIB);
            assert!(now || !spilling, "auto stopped spilling at {g} cells");
            spilling = now;
        }
        assert!(spilling);
        assert!(spill_reserve_bytes(0) == 6 * GIB && spill_reserve_bytes(10_000_000_000) > 7 * GIB);
    }
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
    use super::{Arming, QueueRoom};
    use std::sync::atomic::{AtomicBool, Ordering};

    /// A test arming: armed while its flag is set.
    struct Flag(AtomicBool);

    impl Arming for Flag {
        fn armed(&self) -> bool {
            self.0.load(Ordering::Relaxed)
        }
    }

    /// The queue reports the most bytes and chunks it held at once.
    #[test]
    fn the_queue_reports_the_most_it_held() {
        let queue = QueueRoom::with_budget(None);
        queue.admit(60);
        queue.admit(40);
        queue.release(60);
        queue.admit(10);
        assert_eq!(*queue.held.lock().unwrap(), (50, 2));
        assert_eq!(*queue.most.lock().unwrap(), (100, 2));
    }

    /// With a budget, a chunk that does not fit beside what is held waits
    /// until enough is released, and a chunk alone always fits.
    #[test]
    fn a_budgeted_queue_waits_for_room() {
        static ARMED: Flag = Flag(AtomicBool::new(true));
        let queue = QueueRoom::with_budget(Some((100, &ARMED)));
        queue.admit(500);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            s.spawn(|| {
                queue.admit(60);
                tx.send(()).unwrap();
            });
            assert!(
                rx.recv_timeout(std::time::Duration::from_millis(200))
                    .is_err(),
                "admitted beside a full queue"
            );
            queue.release(500);
            rx.recv_timeout(std::time::Duration::from_secs(5))
                .expect("admitted once there is room");
        });
        assert_eq!(*queue.held.lock().unwrap(), (60, 1));
    }

    /// An unarmed budget binds nothing: chunks go in beside a full queue, as
    /// with no budget, until the arming flips; then an over-budget chunk
    /// waits for room again.
    #[test]
    fn an_unarmed_budget_binds_nothing() {
        static ARMING: Flag = Flag(AtomicBool::new(false));
        let queue = QueueRoom::with_budget(Some((100, &ARMING)));
        queue.admit(500);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            s.spawn(|| {
                queue.admit(60);
                tx.send(()).unwrap();
            });
            let went_in = rx.recv_timeout(std::time::Duration::from_secs(1)).is_ok();
            if !went_in {
                // Let the waiter finish before failing.
                queue.release(500);
            }
            assert!(went_in, "an unarmed budget made a chunk wait");
        });
        assert_eq!(*queue.held.lock().unwrap(), (560, 2));
        ARMING.0.store(true, Ordering::Relaxed);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            s.spawn(|| {
                queue.admit(40);
                tx.send(()).unwrap();
            });
            assert!(
                rx.recv_timeout(std::time::Duration::from_millis(200))
                    .is_err(),
                "armed: admitted beside a full queue"
            );
            queue.release(500);
            queue.release(60);
            rx.recv_timeout(std::time::Duration::from_secs(5))
                .expect("admitted once there is room");
        });
    }
}
