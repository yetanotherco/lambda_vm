//! Lambda VM CLI - execute, prove, and verify RISC-V programs.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueHint};

#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// jemalloc, never purging.
//
// The allocator itself is unchanged: jemalloc is here because the platform
// allocator keeps freed arena chunks resident, and on the recursion campaign's
// branch the same proves read up to 13 GiB higher under glibc. What this sets
// is jemalloc's *decay* timers, which hand freed pages back to the OS. The
// prover allocates and frees multi-hundred-MiB host buffers continuously, so
// those pages come straight back as minor faults on the worker threads.
//
// Measured on an RTX 5090 box on the recursion campaign's branches, ABBA in
// each, every arm at one commit and one set of knobs:
//   * the WHIR prover (keccak, `whir/lfm` @ 64393da9) — 39.69-39.88 s a block
//     with this setting against 43.94-44.07 s without, the default costing
//     +13 M minor faults and +15 s of system time per run on this arm;
//   * the per-table STARK tree (0e4f4610) — 187 / 163 / 166 / 171 s, both
//     never-purge arms under both default arms, 5-24 s a block, with the proof
//     bytes unmoved (30 identical lines, 0 differing).
// The cost is peak RSS: +1.4 GiB and +2.9-3.3 GiB respectively, a ninth to a
// quarter of the 13 GiB the allocator choice itself is worth — which is why the
// lever is the decay setting and not the allocator. `background_thread:true` recovers
// none of it: the cost is the re-touch, not the `madvise` call.
//
// This binary's own pipeline has not been measured under the setting; the
// numbers above are from the campaign's branches, where the prover's
// allocation pattern is the same.
//
// `_RJEM_MALLOC_CONF` in the environment still overrides this, which is how a
// measurement arm puts the default policy back. It has to be that spelling:
// `tikv-jemalloc-sys` builds with `--with-jemalloc-prefix=_rjem_` under default
// features, and jemalloc then reads one env name chosen at configure time
// (`jemalloc.c`, `obtain_malloc_conf` source 3) — so plain `MALLOC_CONF` is read
// by nothing here and sets an arm to the default policy without saying it did
// not. The file source is prefixed too: `/etc/_rjem_malloc.conf`.
//
// jemalloc reads this symbol as a `const char *` before `main` is entered, so
// the value has to be in the initializer, and the name is the prefixed one
// `tikv-jemalloc-sys` declares (`#[cfg_attr(prefixed, link_name =
// "_rjem_malloc_conf")]`, its `src/lib.rs`). None of that is compiler-checked.
// `prover/tests/jemalloc_conf.rs` reads both options back out of jemalloc, but
// it carries its own copy of this block and reads its own process — it pins the
// pattern. This binary's own unit tests read this export back
// (`the_binary_runs_its_never_purge_posture`): deleting the lines below turns
// that test red.
const NEVER_PURGE: &[u8] = b"dirty_decay_ms:-1,muzzy_decay_ms:-1\0";

#[allow(non_upper_case_globals)]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static malloc_conf: Option<&'static core::ffi::c_char> =
    Some(unsafe { &*(NEVER_PURGE.as_ptr() as *const core::ffi::c_char) });

/// This binary's jemalloc, as the prover asks it (`prover::alloc_purge`): its
/// statistics, and a purge of every arena's dirty pages. Installed at the top
/// of `main`, so the block's purge points (`LAMBDA_VM_ALLOC_PURGE`, `auto` by
/// default) run here as they do in the prover's own tests. It also reads back
/// the never-purge posture it runs for the `BLOCK POSTURE SET` line and its
/// test.
mod allocator {
    use prover::alloc_purge::{AllocStats, AllocatorHooks};

    pub const HOOKS: AllocatorHooks = AllocatorHooks { stats, purge_all };

    /// jemalloc's statistics, the epoch turned first (they are cached until
    /// it turns).
    fn stats() -> Option<AllocStats> {
        use tikv_jemalloc_ctl::{epoch, stats};
        epoch::advance().ok()?;
        Some(AllocStats {
            allocated: stats::allocated::read().ok()?,
            active: stats::active::read().ok()?,
            resident: stats::resident::read().ok()?,
            mapped: stats::mapped::read().ok()?,
            retained: stats::retained::read().ok()?,
        })
    }

    /// `arena.<MALLCTL_ARENAS_ALL>.purge` (4096, jemalloc's "every arena"
    /// index): a control that neither reads nor writes, so every pointer is
    /// null.
    fn purge_all() -> bool {
        // SAFETY: the name is NUL-terminated, and a control that takes no
        // value is called with null old and new pointers and zero lengths, as
        // jemalloc's `NEITHER_READ_NOR_WRITE` requires.
        let rc = unsafe {
            tikv_jemalloc_sys::mallctl(
                c"arena.4096.purge".as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 {
            eprintln!("ALLOC PURGE: arena.4096.purge returned {rc}");
        }
        rc == 0
    }

    /// The running `opt.dirty_decay_ms` and `opt.muzzy_decay_ms`: `-1` is the
    /// never-purge posture this binary compiles in.
    pub fn decay_ms() -> Option<(isize, isize)> {
        // SAFETY: both names are NUL-terminated `ssize_t` options.
        unsafe {
            Some((
                tikv_jemalloc_ctl::raw::read(b"opt.dirty_decay_ms\0").ok()?,
                tikv_jemalloc_ctl::raw::read(b"opt.muzzy_decay_ms\0").ok()?,
            ))
        }
    }
}

use executor::vm::instruction::decoding::Instruction;
use executor::vm::instruction::execution::{Accelerator, SyscallNumbers};
use executor::{elf::Elf, flamegraph::FlamegraphGenerator, vm::execution::Executor};
use prover::VmProof;
use stark::proof::options::GoldilocksCubicProofOptions;

const DEFAULT_CONTINUATION_EPOCH_SIZE_LOG2: u32 = 20;
const MIN_CONTINUATION_EPOCH_SIZE_LOG2: u32 = 18;

/// Read a file into a buffer aligned for `rkyv::from_bytes`. A plain
/// `Vec<u8>` from `std::fs::read` is align-1 by the type system even though
/// the allocator happens to return well-aligned memory in practice — read
/// straight into an `AlignedVec` instead of relying on that.
fn read_aligned_file(path: &Path) -> std::io::Result<rkyv::util::AlignedVec<16>> {
    use std::os::unix::fs::FileExt;

    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len() as usize;
    let mut aligned = rkyv::util::AlignedVec::<16>::with_capacity(len);
    aligned.resize(len, 0);
    file.read_exact_at(&mut aligned, 0)?;
    Ok(aligned)
}

/// Polls jemalloc `stats.allocated` every 10ms from a background thread,
/// tracking the high-water mark. Near-zero overhead because jemalloc uses
/// thread-local caches — `epoch::advance()` just merges cached counters.
#[cfg(feature = "jemalloc-stats")]
mod heap_tracker {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    use tikv_jemalloc_ctl::{epoch, stats};

    pub struct HeapTracker {
        stop: Arc<AtomicBool>,
        peak: Arc<AtomicUsize>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl HeapTracker {
        pub fn start() -> Self {
            let stop = Arc::new(AtomicBool::new(false));
            let peak = Arc::new(AtomicUsize::new(0));
            let stop_clone = stop.clone();
            let peak_clone = peak.clone();

            let handle = thread::spawn(move || {
                while !stop_clone.load(Ordering::Relaxed) {
                    // Refresh jemalloc's cached stats
                    epoch::advance().ok();
                    if let Ok(allocated) = stats::allocated::read() {
                        peak_clone.fetch_max(allocated, Ordering::Relaxed);
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                // One final sample after stop signal
                epoch::advance().ok();
                if let Ok(allocated) = stats::allocated::read() {
                    peak_clone.fetch_max(allocated, Ordering::Relaxed);
                }
            });

            Self {
                stop,
                peak,
                handle: Some(handle),
            }
        }

        pub fn stop(mut self) -> usize {
            self.shutdown();
            self.peak.load(Ordering::Relaxed)
        }

        fn shutdown(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(h) = self.handle.take() {
                h.join().ok();
            }
        }
    }

    impl Drop for HeapTracker {
        fn drop(&mut self) {
            self.shutdown();
        }
    }
}

#[derive(Parser)]
#[command(author, version, about = "Lambda VM - RISC-V zkVM", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Execute an ELF program without generating a proof
    Execute {
        /// Path to the ELF file
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        elf: PathBuf,

        /// Path to the private input file
        #[arg(long, value_hint = ValueHint::FilePath)]
        private_input: Option<PathBuf>,

        /// Generate flamegraph folded stacks to file
        #[arg(long, value_hint = ValueHint::FilePath)]
        flamegraph: Option<PathBuf>,

        /// Key the folded stacks by raw hex address instead of resolving
        /// through the ELF symtab (pairs with scripts/enrich_flamegraph.py).
        /// Only meaningful with --flamegraph.
        #[arg(long, requires = "flamegraph")]
        flamegraph_raw: bool,

        /// Checkpoint the flamegraph's folded output to --flamegraph every N
        /// cycles, so a killed run still leaves usable (partial) output on
        /// disk. Only meaningful with --flamegraph.
        #[arg(long, requires = "flamegraph")]
        flamegraph_checkpoint_cycles: Option<u64>,

        /// Stop execution early once at least this many cycles have run.
        #[arg(long)]
        cycle_budget: Option<u64>,

        /// Print the dynamic instruction (cycle) count, plus `Keccak calls` /
        /// `Ecsm calls` (accelerator syscall invocations). The accelerator lines
        /// are omitted when combined with --flamegraph (that path has no per-log
        /// data).
        #[arg(long)]
        cycles: bool,
    },

    /// Generate a proof for an ELF program
    Prove {
        /// Path to the ELF file
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        elf: PathBuf,

        /// Output path for the proof bundle
        #[arg(short, long, value_hint = ValueHint::FilePath)]
        output: PathBuf,

        /// Path to the private input file
        #[arg(long, value_hint = ValueHint::FilePath)]
        private_input: Option<PathBuf>,

        /// Blowup factor (power of 2). Higher = fewer queries, smaller proof, slower proving.
        #[arg(long, default_value = "2")]
        blowup: u8,

        /// Print proving time
        #[arg(long)]
        time: bool,

        /// Execute once outside the timer and print dynamic instruction count
        #[arg(long)]
        cycles: bool,

        /// Build traces and print total main-trace field elements (rows × columns summed across
        /// all tables) and aux-trace field elements (committed EF columns × rows)
        #[arg(long, conflicts_with = "continuations")]
        elements: bool,

        /// Prove with continuations (split execution into epochs; flat peak memory)
        #[arg(long)]
        continuations: bool,

        /// Prove with the multilinear backend (WHIR) instead of FRI.
        ///
        /// A different proof format: verify it with `verify --whir`. The
        /// multilinear path derives its own rate and query count from the
        /// trace, so `--blowup` does not apply to it.
        #[arg(long, conflicts_with = "continuations")]
        whir: bool,

        /// Continuation epoch size as log2(cycles); e.g. 20 means 1,048,576 cycles.
        #[arg(
            long,
            value_name = "N",
            requires = "continuations",
            value_parser = parse_epoch_size_log2,
            long_help = "Continuation epoch size as log2(cycles); e.g. 20 means 1,048,576 cycles.\n\nDefault when omitted: 20. Values below 18 are rejected for the CLI because tiny epochs are dominated by fixed overhead. Indicative ethrex 10-transfer distinct-account peak heap from a local sweep: 19 ~= 6.9 GB, 20 ~= 9.5 GB, 21 ~= 15.8 GB, 22 ~= 26.8 GB. Higher values reduce epoch count, continuation bundle size, and fixed per-epoch overhead, but increase peak memory. For a new workload, try the highest value your machine can run without swapping."
        )]
        epoch_size_log2: Option<u32>,
    },

    /// Verify a proof bundle
    Verify {
        /// Path to the proof bundle file
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        proof: PathBuf,

        /// Path to the ELF file (required for DECODE table verification)
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        elf: PathBuf,

        /// Blowup factor used during proving (must match)
        #[arg(long, default_value = "2")]
        blowup: u8,

        /// Print verification time
        #[arg(long)]
        time: bool,

        /// Verify a continuation proof bundle (produced by `prove --continuations`)
        #[arg(long)]
        continuations: bool,

        /// Verify a multilinear proof (produced by `prove --whir`)
        #[arg(long, conflicts_with = "continuations")]
        whir: bool,
    },

    /// Prove a whole block as a recursion tree, base to top node
    ///
    /// The block's no-epoch WHIR base proof, the leaves that verify its
    /// groups, the nodes and the top: one proof a consumer verifies with
    /// `verify-block` against the ELF. Needs a `--features cuda` build. The
    /// production posture (table parallelism, retention, the grind's search,
    /// ...) is set for every knob the environment leaves unset; the `BLOCK
    /// POSTURE` lines on stderr say which. The base hash is `--base-hash`'s,
    /// a proof-format constant `verify-block` must be given too.
    ProveBlock {
        /// Path to the guest ELF
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        elf: PathBuf,

        /// The block's private input (the guest reads it with `get_private_input()`)
        #[arg(long, value_hint = ValueHint::FilePath)]
        input: Option<PathBuf>,

        /// Output path for the block proof
        #[arg(short, long, value_hint = ValueHint::FilePath)]
        output: PathBuf,

        /// Print the whole run's wall (base to top node)
        #[arg(long)]
        time: bool,

        /// Print the blake3 digest of the top proof's bytes
        #[arg(long)]
        digest: bool,

        /// The base proof's hash (a proof-format constant)
        #[arg(long, value_enum, default_value_t = BaseHash::Rpx)]
        base_hash: BaseHash,
    },

    /// Verify a block proof produced by `prove-block`
    VerifyBlock {
        /// Path to the block proof
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        proof: PathBuf,

        /// Path to the guest ELF the block ran (trusted)
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        elf: PathBuf,

        /// Print verification time
        #[arg(long)]
        time: bool,

        /// The base hash the block was proved over (`prove-block --base-hash`);
        /// the verifier's own constant, never read from the proof
        #[arg(long, value_enum, default_value_t = BaseHash::Rpx)]
        base_hash: BaseHash,
    },

    /// Count main-trace and aux-trace field elements without proving
    CountElements {
        /// Path to the ELF file
        #[arg(value_parser, value_hint = ValueHint::FilePath)]
        elf: PathBuf,

        /// Path to the private input file
        #[arg(long, value_hint = ValueHint::FilePath)]
        private_input: Option<PathBuf>,
    },
}

/// A block's base hash (`--base-hash`): a proof-format constant the prover
/// and the verifier each take from the operator, never from the proof file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum BaseHash {
    /// RPX256, the production base
    Rpx,
    /// ZisK's Poseidon1 at width 16 over 4-ary trees (opt-in)
    P1,
}

impl BaseHash {
    fn format(self) -> stark::proof::options::BaseFormat {
        match self {
            Self::Rpx => stark::proof::options::BaseFormat::RPX,
            Self::P1 => stark::proof::options::BaseFormat::P1_WHIR,
        }
    }
}

fn main() -> ExitCode {
    env_logger::init();
    prover::alloc_purge::install(allocator::HOOKS);
    let cli = Cli::parse();
    // Still single-threaded here: clap and env_logger spawn nothing.
    if matches!(
        cli.command,
        Commands::ProveBlock { .. } | Commands::VerifyBlock { .. }
    ) {
        // SAFETY: no thread but this one exists yet.
        for line in unsafe { apply_posture() } {
            eprintln!("{line}");
        }
    }

    match cli.command {
        Commands::Execute {
            elf,
            private_input,
            flamegraph,
            flamegraph_raw,
            flamegraph_checkpoint_cycles,
            cycle_budget,
            cycles,
        } => cmd_execute(
            elf,
            private_input,
            FlamegraphCliOptions {
                path: flamegraph,
                raw: flamegraph_raw,
                checkpoint_cycles: flamegraph_checkpoint_cycles,
            },
            cycle_budget,
            cycles,
        ),
        Commands::Prove {
            elf,
            output,
            private_input,
            blowup,
            time,
            cycles,
            elements,
            continuations,
            epoch_size_log2,
            whir,
        } => {
            if continuations {
                cmd_prove_continuation(
                    elf,
                    output,
                    private_input,
                    epoch_size_log2,
                    blowup,
                    time,
                    cycles,
                )
            } else if whir {
                cmd_prove_whir(elf, output, private_input, blowup, time)
            } else {
                cmd_prove(elf, output, private_input, blowup, time, cycles, elements)
            }
        }
        Commands::Verify {
            proof,
            elf,
            blowup,
            time,
            continuations,
            whir,
        } => {
            if continuations {
                cmd_verify_continuation(proof, elf, blowup, time)
            } else if whir {
                cmd_verify_whir(proof, elf, blowup, time)
            } else {
                cmd_verify(proof, elf, blowup, time)
            }
        }
        Commands::ProveBlock {
            elf,
            input,
            output,
            time,
            digest,
            base_hash,
        } => cmd_prove_block(elf, input, output, time, digest, base_hash),
        Commands::VerifyBlock {
            proof,
            elf,
            time,
            base_hash,
        } => cmd_verify_block(proof, elf, time, base_hash),
        Commands::CountElements { elf, private_input } => cmd_count_elements(elf, private_input),
    }
}

/// What the posture does in an environment: the knobs it sets (each posture
/// knob the environment leaves unset), the ones the environment already sets,
/// and notes on the knobs it set to something other than the table's value.
struct PosturePlan {
    set: Vec<(&'static str, &'static str)>,
    env: Vec<String>,
    notes: Vec<String>,
}

/// The posture against the environment `env` and the host `target` (bytes):
/// the precomputed-tree cache's cap follows the target
/// ([`prover::lfm::whir_block_tree::posture_tree_cache_cap`]).
fn posture_plan(env: impl Fn(&str) -> Option<String>, target: impl Fn() -> u64) -> PosturePlan {
    use prover::lfm::whir_block_tree::{POSTURE, POSTURE_TREE_CACHE_KNOB, posture_tree_cache_cap};
    let mut plan = PosturePlan {
        set: Vec::new(),
        env: Vec::new(),
        notes: Vec::new(),
    };
    for &(name, value) in POSTURE {
        if let Some(v) = env(name) {
            plan.env.push(format!("{name}={v}"));
            continue;
        }
        if name == POSTURE_TREE_CACHE_KNOB {
            let target = target();
            let cap = posture_tree_cache_cap(target);
            if cap != value {
                plan.notes.push(format!(
                    "BLOCK POSTURE: {name}={cap}, not {value}: the host target is {:.1} GiB \
                     (LAMBDA_VM_BLOCK_SPILL_TARGET_GIB, else the cgroup or MemTotal less 10 GiB)",
                    target as f64 / (1u64 << 30) as f64
                ));
            }
            plan.set.push((name, cap));
            continue;
        }
        plan.set.push((name, value));
    }
    plan
}

/// The block's production posture ([`prover::lfm::whir_block_tree::POSTURE`]),
/// set for every knob the environment leaves unset ([`posture_plan`]), for
/// `verify-block` too. The base hash is not posture: it is `--base-hash`'s.
/// Returns the lines that say what it set and what it left.
///
/// # Safety
///
/// Sets environment variables: no other thread may exist.
unsafe fn apply_posture() -> Vec<String> {
    let plan = posture_plan(
        |name| std::env::var(name).ok(),
        prover::block_whir::spill_target_bytes,
    );
    for (name, value) in &plan.set {
        // SAFETY: the caller's: no other thread exists.
        unsafe { std::env::set_var(name, value) };
    }
    let decay = match allocator::decay_ms() {
        Some((-1, -1)) => "never purge (dirty -1, muzzy -1)".to_string(),
        Some((d, m)) => format!("dirty_decay_ms {d}, muzzy_decay_ms {m} (_RJEM_MALLOC_CONF)"),
        None => "unread".to_string(),
    };
    let words = |w: Vec<String>| {
        if w.is_empty() {
            "none".to_string()
        } else {
            w.join(" ")
        }
    };
    let mut lines = vec![format!(
        "BLOCK POSTURE SET: {} · from the environment: {} · jemalloc: {decay}",
        words(plan.set.iter().map(|(n, v)| format!("{n}={v}")).collect()),
        words(plan.env),
    )];
    lines.extend(plan.notes);
    lines
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn cmd_prove_block(
    elf_path: PathBuf,
    input_path: Option<PathBuf>,
    output_path: PathBuf,
    time: bool,
    digest: bool,
    base_hash: BaseHash,
) -> ExitCode {
    use prover::lfm::whir_block_tree::{
        StderrSink, WhirBlockTreeProof, WhirTreeConfig, prove_whir_block_tree,
    };

    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {e}");
            return ExitCode::FAILURE;
        }
    };
    let input = match read_private_input(input_path.as_ref()) {
        Ok(input) => input,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let mut cfg = match WhirTreeConfig::from_env() {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    cfg.format.zf = cfg.format.zf.with_base(base_hash.format());
    // A forced leaf count, another fan-in or another argue is another tree than
    // the one `verify-block` derives: harness arms, not a production proof.
    if !cfg.at_presets() {
        eprintln!(
            "W3_LEAVES, W3_FAN_IN and BLOCK_WHIR_ARGUE make another tree than the block \
             verifier derives: unset them to prove a block the CLI can verify"
        );
        return ExitCode::FAILURE;
    }
    let mut run = match prove_whir_block_tree(&elf_data, &input, &cfg, &StderrSink) {
        Ok(run) => run,
        Err(e) => {
            eprintln!("Block proof failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let whole = run.times.whole;
    let Some((_, top)) = run.proofs.pop() else {
        eprintln!("Block proof failed: the tree has no top");
        return ExitCode::FAILURE;
    };
    let top_digest = digest.then(|| match rkyv::to_bytes::<rkyv::rancor::Error>(&top.proof) {
        Ok(bytes) => format!(
            "BLOCK TOP DIGEST: {} (blake3 of the top proof's {} bytes)",
            &blake3::hash(&bytes).to_hex()[..32],
            bytes.len()
        ),
        Err(e) => format!("BLOCK TOP DIGEST: none ({e})"),
    });
    let output_hex = hex(&run.statement.public_output);
    let file = WhirBlockTreeProof::new(run.statement, top);
    let bytes = match file.to_bytes() {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("Failed to serialize the block proof: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = std::fs::write(&output_path, &bytes) {
        eprintln!("Failed to write {output_path:?}: {e}");
        return ExitCode::FAILURE;
    }
    eprintln!(
        "Block proof written to {output_path:?} ({} bytes)",
        bytes.len()
    );
    println!("Output: {output_hex}");
    if let Some(line) = top_digest {
        println!("{line}");
    }
    if time {
        println!("Proving time: {whole:.3}s");
    }
    ExitCode::SUCCESS
}

fn cmd_verify_block(
    proof_path: PathBuf,
    elf_path: PathBuf,
    time: bool,
    base_hash: BaseHash,
) -> ExitCode {
    use prover::lfm::whir_block_tree::{
        WhirBlockTreeProof, base_line, verify_whir_block_tree_proof_based,
    };

    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {e}");
            return ExitCode::FAILURE;
        }
    };
    let bytes = match read_aligned_file(&proof_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read proof file: {e}");
            return ExitCode::FAILURE;
        }
    };
    let proof = match WhirBlockTreeProof::from_bytes(&bytes) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to read the block proof: {e}");
            return ExitCode::FAILURE;
        }
    };
    drop(bytes);
    let base = base_hash.format();
    eprintln!("{}", base_line(&base));
    eprintln!("Verifying block proof...");
    let start = Instant::now();
    let result = verify_whir_block_tree_proof_based(&elf_data, &base, &proof);
    let verify_elapsed = start.elapsed();
    match result {
        Ok(()) => {
            eprintln!("Verification succeeded!");
            println!("Output: {}", hex(&proof.public_output));
            if time {
                println!("Verification time: {:.3}s", verify_elapsed.as_secs_f64());
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Verification failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn read_private_input(path: Option<&PathBuf>) -> Result<Vec<u8>, String> {
    match path {
        Some(path) => {
            eprintln!("Reading private input file...");
            std::fs::read(path).map_err(|e| format!("Failed to read private input file: {e}"))
        }
        None => Ok(vec![]),
    }
}

fn count_cycles(elf_data: &[u8], private_inputs: &[u8]) -> Result<u64, String> {
    let program =
        Elf::load(elf_data).map_err(|e| format!("Failed to load ELF for cycle count: {e:?}"))?;
    let executor = Executor::new(&program, private_inputs.to_vec())
        .map_err(|e| format!("Failed to create executor for cycle count: {e:?}"))?;
    executor
        .run()
        .map(|result| result.logs.len() as u64)
        .map_err(|e| format!("Execution failed during cycle count: {e:?}"))
}

/// Write the flamegraph's current (possibly partial) folded output to
/// `output_path`, replacing any previous contents. Used both for the final
/// write and for periodic checkpoints during a long run.
///
/// Writes to a `tempfile` in the same directory, flushes it, then persists
/// (renames) it over `output_path` — the whole file is replaced atomically,
/// so a kill mid-write can never leave `output_path` empty or torn (the
/// previous good checkpoint stays put until the new one is fully on disk).
fn write_flamegraph_checkpoint(
    output_path: &PathBuf,
    generator: &FlamegraphGenerator,
    raw: bool,
) -> Result<(), String> {
    let dir = output_path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = tempfile::NamedTempFile::new_in(dir)
        .map_err(|e| format!("Failed to create temp output file: {e}"))?;

    let mut writer = BufWriter::new(tmp.as_file());
    let result = if raw {
        generator.write_folded_raw(&mut writer)
    } else {
        generator.write_folded(&mut writer)
    };
    result.map_err(|e| format!("Failed to write flamegraph output: {e:?}"))?;
    writer
        .flush()
        .map_err(|e| format!("Failed to flush flamegraph output: {e}"))?;
    drop(writer);

    tmp.persist(output_path)
        .map_err(|e| format!("Failed to replace {output_path:?} with temp output: {e}"))?;
    Ok(())
}

/// Flamegraph-related flags grouped so `cmd_execute` doesn't need a flat
/// 8-argument signature.
struct FlamegraphCliOptions {
    path: Option<PathBuf>,
    raw: bool,
    checkpoint_cycles: Option<u64>,
}

/// Classifies one executed instruction as an accelerator syscall invocation.
///
/// Delegates to the executor's canonical `SyscallNumbers::accelerator()` so the
/// CLI's counts equal the prover's chip-trigger counts by construction: the
/// prover sets `ecall_keccak`/`ecall_ecsm` from `f.ecall && log.src1_val ==
/// <SYSCALL_NUMBER>`. Here `f.ecall` is the instruction at the log's
/// `current_pc` being `EcallEbreak`, and `src1_val` carries a7 (the syscall
/// number) on ECALL logs. (`get_private_input` is a memory-mapped read, not a
/// syscall, so it never reaches this path.)
fn accelerator_of(instruction: Option<&Instruction>, src1_val: u64) -> Option<Accelerator> {
    if !matches!(instruction, Some(Instruction::EcallEbreak)) {
        return None;
    }
    SyscallNumbers::try_from(src1_val)
        .ok()
        .and_then(|s| s.accelerator())
}

fn cmd_execute(
    elf_path: PathBuf,
    private_input_path: Option<PathBuf>,
    flamegraph: FlamegraphCliOptions,
    cycle_budget: Option<u64>,
    cycles: bool,
) -> ExitCode {
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let program = match Elf::load(&elf_data) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to load ELF program: {:?}", e);
            return ExitCode::FAILURE;
        }
    };

    let private_inputs = match read_private_input(private_input_path.as_ref()) {
        Ok(inputs) => inputs,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    // Accelerator invocation counts, tallied only in the plain streaming path
    // below (the flamegraph path drives execution inside the executor and does
    // not expose per-log data). `None` means "not counted", so the accel lines
    // are omitted rather than printed as misleading zeros.
    let mut accel_counts: Option<(u64, u64, u64)> = None;

    let cycle_count = if let Some(ref output_path) = flamegraph.path {
        // Shared execute+flamegraph path (executor::flamegraph) instead of
        // hand-rolling the SymbolTable/Executor/drive-loop wiring here.
        let mut next_checkpoint = flamegraph.checkpoint_cycles;
        let result = executor::flamegraph::run_with_flamegraph(
            &elf_data,
            &program,
            private_inputs,
            cycle_budget,
            |total_cycles, generator| {
                let Some(threshold) = next_checkpoint else {
                    return;
                };
                if total_cycles < threshold {
                    return;
                }
                if let Err(e) = write_flamegraph_checkpoint(output_path, generator, flamegraph.raw)
                {
                    eprintln!("Warning: flamegraph checkpoint failed: {e}");
                }
                next_checkpoint = flamegraph.checkpoint_cycles.map(|step| threshold + step);
            },
        );

        let (generator, result) = result;
        let total_cycles = match result {
            Ok(total_cycles) => total_cycles,
            Err(e) => {
                eprintln!("Execution failed: {:?}", e);
                // Best-effort: persist whatever the generator accumulated
                // before the fault instead of discarding it outright.
                match write_flamegraph_checkpoint(output_path, &generator, flamegraph.raw) {
                    Ok(()) => eprintln!(
                        "Partial flamegraph written to {:?} ({} instructions)",
                        output_path,
                        generator.total_instructions()
                    ),
                    Err(e) => eprintln!("Warning: failed to write partial flamegraph: {e}"),
                }
                return ExitCode::FAILURE;
            }
        };

        if let Err(e) = write_flamegraph_checkpoint(output_path, &generator, flamegraph.raw) {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
        eprintln!(
            "Flamegraph written to {:?} ({} instructions)",
            output_path,
            generator.total_instructions()
        );

        total_cycles
    } else {
        let mut executor = match Executor::new(&program, private_inputs) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Failed to create executor: {:?}", e);
                return ExitCode::FAILURE;
            }
        };

        let mut cycle_count: u64 = 0;
        let mut keccak_calls: u64 = 0;
        let mut blake3_calls: u64 = 0;
        let mut ecsm_calls: u64 = 0;
        // Reused per chunk: `(current_pc, a7)` for logs whose a7 matches an
        // accelerator syscall number. This is a cheap superset — a non-ECALL
        // instruction can hold the same value in src1 — that `accelerator_of`
        // confirms below, once the chunk's `&Log` borrow (tied to the executor's
        // `&mut`) is released so the instruction cache can be read again.
        let mut accel_candidates: Vec<(u64, u64)> = Vec::new();
        loop {
            let logs = match executor.resume_budgeted(cycle_count, cycle_budget) {
                Ok(logs) => logs,
                Err(e) => {
                    eprintln!("Execution failed: {:?}", e);
                    return ExitCode::FAILURE;
                }
            };
            let Some(logs) = logs else { break };
            cycle_count += logs.len() as u64;
            if cycles {
                for log in logs {
                    if SyscallNumbers::try_from(log.src1_val)
                        .map(|s| s.accelerator().is_some())
                        .unwrap_or(false)
                    {
                        accel_candidates.push((log.current_pc, log.src1_val));
                    }
                }
            }
            // `logs` is no longer used, so the executor's `&mut` borrow is free
            // and the instruction cache can be read to confirm each candidate.
            for (pc, a7) in accel_candidates.drain(..) {
                match accelerator_of(executor.instructions.get(pc), a7) {
                    Some(Accelerator::Keccak) => keccak_calls += 1,
                    Some(Accelerator::Blake3) => blake3_calls += 1,
                    Some(Accelerator::Ecsm) => ecsm_calls += 1,
                    None => {}
                }
            }
            if cycle_budget.is_some_and(|budget| cycle_count >= budget) {
                break;
            }
        }

        if let Err(e) = executor.finish() {
            eprintln!("Failed to finish execution: {:?}", e);
            return ExitCode::FAILURE;
        }

        if cycles {
            accel_counts = Some((keccak_calls, blake3_calls, ecsm_calls));
        }
        cycle_count
    };

    if cycles {
        println!("Cycles: {}", cycle_count);
        if let Some((keccak_calls, blake3_calls, ecsm_calls)) = accel_counts {
            println!("Keccak calls: {}", keccak_calls);
            println!("Blake3 calls: {}", blake3_calls);
            println!("Ecsm calls: {}", ecsm_calls);
        }
    }

    ExitCode::SUCCESS
}

fn cmd_prove(
    elf_path: PathBuf,
    output_path: PathBuf,
    private_input_path: Option<PathBuf>,
    blowup: u8,
    time: bool,
    cycles: bool,
    elements: bool,
) -> ExitCode {
    eprintln!("Reading ELF file...");
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let private_inputs = match read_private_input(private_input_path.as_ref()) {
        Ok(inputs) => inputs,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    // Pre-pass: execute once outside the timer to count dynamic instructions.
    // Mirrors SP1's cycle-count pass so both provers report the same kind of
    // number without inflating the measured proving time.
    let cycle_count = if cycles {
        match count_cycles(&elf_data, &private_inputs) {
            Ok(count) => Some(count),
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

    // Pre-pass: build traces and count field elements without running the proof.
    let element_count = if elements {
        match prover::count_elements(&elf_data, &private_inputs) {
            Ok(counts) => Some(counts),
            Err(e) => {
                eprintln!("Failed to count elements: {:?}", e);
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

    #[cfg(feature = "jemalloc-stats")]
    let tracker = heap_tracker::HeapTracker::start();

    #[cfg(all(feature = "jemalloc-stats", feature = "instruments"))]
    stark::instruments::set_heap_reader(|| {
        tikv_jemalloc_ctl::epoch::advance().ok();
        tikv_jemalloc_ctl::stats::allocated::read().ok()
    });

    let start = Instant::now();
    let opts = match GoldilocksCubicProofOptions::with_blowup(blowup) {
        Ok(opts) => opts,
        Err(e) => {
            eprintln!("Invalid proof options: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "Generating proof (blowup={blowup}, queries={})...",
        opts.fri_number_of_queries
    );
    let proof = prover::prove_with_options_and_inputs(
        &elf_data,
        &private_inputs,
        &opts,
        &Default::default(),
    );
    let prove_elapsed = start.elapsed();
    let proof = match proof {
        Ok(proof) => proof,
        Err(e) => {
            eprintln!("Proof generation failed: {}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Writing proof...");
    let file = match File::create(&output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create output file: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let mut writer = BufWriter::new(file);

    let bytes = match rkyv::to_bytes::<rkyv::rancor::Error>(&proof) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to serialize proof: {}", e);
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = writer.write_all(&bytes) {
        eprintln!("Failed to write proof: {}", e);
        return ExitCode::FAILURE;
    }

    eprintln!("Proof written to {:?}", output_path);
    if let Some(c) = cycle_count {
        println!("Cycles: {}", c);
    }
    if let Some((main, aux)) = element_count {
        println!("Elements: {}", main);
        println!("Aux elements (EF-cols): {}", aux);
    }
    if time {
        println!("Proving time: {:.3}s", prove_elapsed.as_secs_f64());
    }
    #[cfg(feature = "jemalloc-stats")]
    {
        let peak_bytes = tracker.stop();
        println!("Peak heap: {} MB", peak_bytes / (1024 * 1024));
    }
    ExitCode::SUCCESS
}

fn cmd_verify(proof_path: PathBuf, elf_path: PathBuf, blowup: u8, time: bool) -> ExitCode {
    eprintln!("Reading ELF file...");
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Reading proof...");
    let proof_bytes = match read_aligned_file(&proof_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read proof file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let proof: VmProof = match rkyv::from_bytes::<VmProof, rkyv::rancor::Error>(&proof_bytes) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to deserialize proof: {}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Verifying proof...");
    let start = Instant::now();
    let opts = match GoldilocksCubicProofOptions::with_blowup(blowup) {
        Ok(opts) => opts,
        Err(e) => {
            eprintln!("Invalid proof options: {e}");
            return ExitCode::FAILURE;
        }
    };
    let result = prover::verify_with_options(&proof, &elf_data, &opts, None, None);
    let verify_elapsed = start.elapsed();
    let result = match result {
        Ok(valid) => valid,
        Err(e) => {
            eprintln!("Verification error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    if result {
        eprintln!("Verification succeeded!");
        if time {
            println!("Verification time: {:.3}s", verify_elapsed.as_secs_f64());
        }
        ExitCode::SUCCESS
    } else {
        eprintln!("Verification failed! Ensure --blowup matches the value used for proving.");
        ExitCode::FAILURE
    }
}

/// `prove --whir`: the multilinear backend.
///
/// Same executor, same traces, same AIRs as [`cmd_prove`]; each table is argued
/// in one sumcheck against a WHIR commitment instead of a composition
/// polynomial committed by FRI. The proof is a different type, so `verify`
/// needs `--whir` too.
///
/// `blowup` reaches the AIRs (it sets the preprocessed commitments) but not the
/// multilinear argument, which derives its own rate and query count from the
/// tallest stacked polynomial in the proof.
fn cmd_prove_whir(
    elf_path: PathBuf,
    output_path: PathBuf,
    private_input_path: Option<PathBuf>,
    blowup: u8,
    time: bool,
) -> ExitCode {
    eprintln!("Reading ELF file...");
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let private_inputs = match read_private_input(private_input_path.as_ref()) {
        Ok(inputs) => inputs,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let opts = match GoldilocksCubicProofOptions::with_blowup(blowup) {
        Ok(opts) => opts,
        Err(e) => {
            eprintln!("Invalid proof options: {e}");
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Generating proof (multilinear/WHIR)...");
    let start = Instant::now();
    let proof = prover::multilinear_prove::prove_with_options_and_inputs(
        &elf_data,
        &private_inputs,
        &opts,
        &Default::default(),
    );
    let prove_elapsed = start.elapsed();
    let proof = match proof {
        Ok(proof) => proof,
        Err(e) => {
            eprintln!("Proof generation failed: {}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Writing proof...");
    let bytes = match rkyv::to_bytes::<rkyv::rancor::Error>(&proof) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to serialize proof: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let file = match File::create(&output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create output file: {}", e);
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = BufWriter::new(file).write_all(&bytes) {
        eprintln!("Failed to write proof: {}", e);
        return ExitCode::FAILURE;
    }

    eprintln!("Proof written to {:?}", output_path);
    println!("Tables: {}", proof.proof.tables.len());
    println!("Commitments: {}", proof.proof.roots.len());
    if time {
        println!("Proving time: {:.3}s", prove_elapsed.as_secs_f64());
    }
    ExitCode::SUCCESS
}

/// `verify --whir`: the counterpart of [`cmd_prove_whir`].
fn cmd_verify_whir(proof_path: PathBuf, elf_path: PathBuf, blowup: u8, time: bool) -> ExitCode {
    eprintln!("Reading ELF file...");
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Reading proof...");
    let proof_bytes = match read_aligned_file(&proof_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read proof file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let proof = match rkyv::from_bytes::<
        prover::multilinear_prove::MultilinearVmProof,
        rkyv::rancor::Error,
    >(&proof_bytes)
    {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to deserialize proof: {e}");
            eprintln!("(a proof produced without `--whir` needs `verify` without it)");
            return ExitCode::FAILURE;
        }
    };

    let opts = match GoldilocksCubicProofOptions::with_blowup(blowup) {
        Ok(opts) => opts,
        Err(e) => {
            eprintln!("Invalid proof options: {e}");
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Verifying proof (multilinear/WHIR)...");
    let start = Instant::now();
    let result = prover::multilinear_prove::verify_with_options(&proof, &elf_data, &opts);
    let verify_elapsed = start.elapsed();
    let result = match result {
        Ok(valid) => valid,
        Err(e) => {
            eprintln!("Verification error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    if result {
        eprintln!("Verification succeeded!");
        if time {
            println!("Verification time: {:.3}s", verify_elapsed.as_secs_f64());
        }
        ExitCode::SUCCESS
    } else {
        eprintln!("Verification failed! Ensure --blowup matches the value used for proving.");
        ExitCode::FAILURE
    }
}

fn cmd_prove_continuation(
    elf_path: PathBuf,
    output_path: PathBuf,
    private_input_path: Option<PathBuf>,
    epoch_size_log2: Option<u32>,
    blowup: u8,
    time: bool,
    cycles: bool,
) -> ExitCode {
    eprintln!("Reading ELF file...");
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let private_inputs = match read_private_input(private_input_path.as_ref()) {
        Ok(inputs) => inputs,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let cycle_count = if cycles {
        match count_cycles(&elf_data, &private_inputs) {
            Ok(count) => Some(count),
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };

    let epoch_size_log2 = epoch_size_log2.unwrap_or(DEFAULT_CONTINUATION_EPOCH_SIZE_LOG2);
    let epoch_size = match continuation_epoch_size(epoch_size_log2) {
        Ok(size) => size,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let opts = match GoldilocksCubicProofOptions::with_blowup(blowup) {
        Ok(opts) => opts,
        Err(e) => {
            eprintln!("Invalid proof options: {e}");
            return ExitCode::FAILURE;
        }
    };

    eprintln!(
        "Generating continuation proof (blowup={blowup}, epoch_size_log2={epoch_size_log2}, epoch_size={epoch_size})...",
    );
    // Same tracker as the monolithic path: the peak here is the flat per-epoch
    // working set rather than a whole-trace high-water mark, and it is the metric
    // that decides which epoch size a machine can run, so the benchmarks need it
    // reported identically on both paths.
    #[cfg(feature = "jemalloc-stats")]
    let tracker = heap_tracker::HeapTracker::start();
    let start = Instant::now();
    let bundle = match prover::continuation::prove_continuation(
        &elf_data,
        &private_inputs,
        epoch_size_log2,
        &opts,
    ) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Continuation proof generation failed: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let prove_elapsed = start.elapsed();

    eprintln!("Writing proof...");
    let file = match File::create(&output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create output file: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let mut writer = BufWriter::new(file);
    let bytes = match rkyv::to_bytes::<rkyv::rancor::Error>(&bundle) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to serialize proof: {}", e);
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = writer.write_all(&bytes) {
        eprintln!("Failed to write proof: {}", e);
        return ExitCode::FAILURE;
    }

    eprintln!("Proof written to {:?}", output_path);
    if let Some(c) = cycle_count {
        println!("Cycles: {}", c);
    }
    println!("Epochs: {}", bundle.num_epochs());
    if time {
        println!("Proving time: {:.3}s", prove_elapsed.as_secs_f64());
    }
    #[cfg(feature = "jemalloc-stats")]
    {
        let peak_bytes = tracker.stop();
        println!("Peak heap: {} MB", peak_bytes / (1024 * 1024));
    }
    ExitCode::SUCCESS
}

fn cmd_verify_continuation(
    proof_path: PathBuf,
    elf_path: PathBuf,
    blowup: u8,
    time: bool,
) -> ExitCode {
    eprintln!("Reading ELF file...");
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Reading proof...");
    let proof_bytes = match read_aligned_file(&proof_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read proof file: {}", e);
            return ExitCode::FAILURE;
        }
    };
    let bundle: prover::continuation::ContinuationProof =
        match rkyv::from_bytes::<prover::continuation::ContinuationProof, rkyv::rancor::Error>(
            &proof_bytes,
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to deserialize proof: {}", e);
                return ExitCode::FAILURE;
            }
        };

    let opts = match GoldilocksCubicProofOptions::with_blowup(blowup) {
        Ok(opts) => opts,
        Err(e) => {
            eprintln!("Invalid proof options: {e}");
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Verifying continuation proof...");
    let start = Instant::now();
    let result = prover::continuation::verify_continuation(&elf_data, &bundle, &opts);
    let verify_elapsed = start.elapsed();

    match result {
        Ok(Some(output)) => {
            eprintln!("Verification succeeded!");
            let hex: String = output.iter().map(|b| format!("{:02x}", b)).collect();
            println!("Output: {}", hex);
            if time {
                println!("Verification time: {:.3}s", verify_elapsed.as_secs_f64());
            }
            ExitCode::SUCCESS
        }
        Ok(None) => {
            eprintln!("Verification failed! Ensure --blowup matches the value used for proving.");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("Verification error: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn cmd_count_elements(elf_path: PathBuf, private_input_path: Option<PathBuf>) -> ExitCode {
    let elf_data = match std::fs::read(&elf_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read ELF file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    let private_inputs = match read_private_input(private_input_path.as_ref()) {
        Ok(inputs) => inputs,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    match prover::count_elements(&elf_data, &private_inputs) {
        Ok((main, aux)) => {
            println!("Elements: {}", main);
            println!("Aux elements (EF-cols): {}", aux);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Failed to count elements: {:?}", e);
            ExitCode::FAILURE
        }
    }
}

fn continuation_epoch_size(epoch_size_log2: u32) -> Result<usize, String> {
    if epoch_size_log2 < MIN_CONTINUATION_EPOCH_SIZE_LOG2 {
        return Err(format!(
            "--epoch-size-log2 must be at least {MIN_CONTINUATION_EPOCH_SIZE_LOG2} for CLI proving"
        ));
    }
    1usize.checked_shl(epoch_size_log2).ok_or_else(|| {
        format!("--epoch-size-log2 {epoch_size_log2} is too large for this platform")
    })
}

fn parse_epoch_size_log2(value: &str) -> Result<u32, String> {
    let epoch_size_log2 = value
        .parse::<u32>()
        .map_err(|_| format!("--epoch-size-log2 must be an integer, got `{value}`"))?;
    continuation_epoch_size(epoch_size_log2)?;
    Ok(epoch_size_log2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    // The arg graph is well-formed (e.g. `requires`/`conflicts_with` reference real args).
    #[test]
    fn cli_command_is_valid() {
        Cli::command().debug_assert();
    }

    // The continuation epoch flag requires --continuations.
    #[test]
    fn epoch_size_log2_requires_continuations() {
        let r = Cli::command().try_get_matches_from([
            "cli",
            "prove",
            "prog.elf",
            "-o",
            "out",
            "--epoch-size-log2",
            "20",
        ]);
        assert!(r.is_err());
    }

    #[test]
    fn epoch_size_log2_accepts_continuations() {
        let r = Cli::command().try_get_matches_from([
            "cli",
            "prove",
            "prog.elf",
            "-o",
            "out",
            "--continuations",
            "--epoch-size-log2",
            "20",
        ]);
        assert!(r.is_ok());
    }

    #[test]
    fn cycles_accepts_continuations() {
        let r = Cli::command().try_get_matches_from([
            "cli",
            "prove",
            "prog.elf",
            "-o",
            "out",
            "--continuations",
            "--cycles",
        ]);
        assert!(r.is_ok());
    }

    #[test]
    fn elements_conflicts_with_continuations() {
        let r = Cli::command().try_get_matches_from([
            "cli",
            "prove",
            "prog.elf",
            "-o",
            "out",
            "--continuations",
            "--elements",
        ]);
        assert!(r.is_err());
    }

    #[test]
    fn epoch_size_log2_rejects_tiny_cli_values() {
        let r = Cli::command().try_get_matches_from([
            "cli",
            "prove",
            "prog.elf",
            "-o",
            "out",
            "--continuations",
            "--epoch-size-log2",
            "17",
        ]);
        assert!(r.is_err());
    }

    #[test]
    fn old_epoch_size_flag_is_rejected() {
        let r = Cli::command().try_get_matches_from([
            "cli",
            "prove",
            "prog.elf",
            "-o",
            "out",
            "--continuations",
            "--epoch-size",
            "1048576",
        ]);
        assert!(r.is_err());
    }

    #[test]
    fn old_num_epochs_flag_is_rejected() {
        let r = Cli::command().try_get_matches_from([
            "cli",
            "prove",
            "prog.elf",
            "-o",
            "out",
            "--continuations",
            "--num-epochs",
            "4",
        ]);
        assert!(r.is_err());
    }

    #[test]
    fn prove_help_omits_removed_epoch_flags() {
        let mut cmd = Cli::command();
        let prove = cmd.find_subcommand_mut("prove").unwrap();
        let mut help = Vec::new();
        prove.write_long_help(&mut help).unwrap();
        let help = String::from_utf8(help).unwrap();

        assert!(help.contains("--epoch-size-log2 <N>"));
        assert!(!help.contains("--num-epochs"));
        assert!(!help.contains("--epoch-size <"));
    }

    /// `prove-block` takes the ELF, an optional input and a required output;
    /// `verify-block` the proof and the ELF.
    #[test]
    fn the_block_commands_take_their_arguments() {
        let ok = |args: &[&str]| Cli::command().try_get_matches_from(args).is_ok();
        assert!(ok(&[
            "cli",
            "prove-block",
            "guest.elf",
            "-o",
            "block.proof"
        ]));
        assert!(ok(&[
            "cli",
            "prove-block",
            "guest.elf",
            "--input",
            "block.bin",
            "-o",
            "block.proof",
            "--time",
            "--digest"
        ]));
        assert!(
            !ok(&["cli", "prove-block", "guest.elf"]),
            "an output is required"
        );
        assert!(ok(&[
            "cli",
            "verify-block",
            "block.proof",
            "guest.elf",
            "--time"
        ]));
        assert!(
            !ok(&["cli", "verify-block", "block.proof"]),
            "the ELF is required"
        );
    }

    /// `--base-hash` takes `rpx` (the default) or `p1` on both block commands,
    /// and maps onto the format's bases; nothing else parses.
    #[test]
    fn the_block_commands_take_a_base_hash() {
        let base = |args: &[&str]| match Cli::try_parse_from(args).map(|c| c.command) {
            Ok(
                Commands::ProveBlock { base_hash, .. } | Commands::VerifyBlock { base_hash, .. },
            ) => Some(base_hash),
            _ => None,
        };
        let prove = ["cli", "prove-block", "guest.elf", "-o", "block.proof"];
        let verify = ["cli", "verify-block", "block.proof", "guest.elf"];
        for cmd in [&prove[..], &verify[..]] {
            assert_eq!(base(cmd), Some(BaseHash::Rpx), "RPX by default");
            for (word, want) in [("rpx", BaseHash::Rpx), ("p1", BaseHash::P1)] {
                let args = [cmd, &["--base-hash", word]].concat();
                assert_eq!(base(&args), Some(want), "{word}");
            }
            for bad in ["keccak", "poseidon1", "P1W"] {
                let args = [cmd, &["--base-hash", bad]].concat();
                assert_eq!(base(&args), None, "{bad} is not a base");
            }
        }
        assert_eq!(
            BaseHash::Rpx.format(),
            prover::block_whir::BlockFormat::production().zf.base,
            "the default is the production base"
        );
        assert_eq!(
            BaseHash::P1.format(),
            stark::proof::options::BaseFormat::P1_WHIR
        );
    }

    /// The posture sets exactly the knobs the environment leaves unset.
    #[test]
    fn the_posture_sets_only_unset_knobs() {
        use prover::lfm::whir_block_tree::POSTURE;
        let roomy = || 110u64 << 30;
        let all = posture_plan(|_| None, roomy);
        assert_eq!(all.set, POSTURE.to_vec());
        assert!(all.env.is_empty() && all.notes.is_empty());
        let mine = posture_plan(
            |n| (n == "TABLE_PARALLELISM").then(|| "8".to_string()),
            roomy,
        );
        assert!(mine.set.iter().all(|(n, _)| *n != "TABLE_PARALLELISM"));
        assert_eq!(mine.env, vec!["TABLE_PARALLELISM=8".to_string()]);
    }

    /// The precomputed-tree cache's posture follows the host target: 64
    /// entries from a 64 GiB target up, 16 below it (with a note), and a value
    /// the environment sets is left alone.
    #[test]
    fn the_tree_cache_posture_follows_the_host_target() {
        use prover::lfm::whir_block_tree::POSTURE_TREE_CACHE_KNOB as CAP;
        let none = |_: &str| None;
        let cap = |plan: &PosturePlan| plan.set.iter().find(|(n, _)| *n == CAP).map(|(_, v)| *v);
        let big = posture_plan(none, || 64u64 << 30);
        assert_eq!(cap(&big), Some("64"));
        assert!(big.notes.is_empty());
        let small = posture_plan(none, || 38u64 << 30);
        assert_eq!(cap(&small), Some("16"));
        assert_eq!(small.notes.len(), 1, "the note says why");
        let set = posture_plan(|n| (n == CAP).then(|| "32".to_string()), || 38u64 << 30);
        assert_eq!(cap(&set), None, "the environment's value stays");
        assert_eq!(set.env, vec![format!("{CAP}=32")]);
    }

    /// The binary runs the allocator posture it compiles in: jemalloc never
    /// purges (`opt.dirty_decay_ms` and `opt.muzzy_decay_ms` are -1), unless
    /// `_RJEM_MALLOC_CONF` says otherwise. Deleting the `malloc_conf` export
    /// turns this red.
    #[test]
    fn the_binary_runs_its_never_purge_posture() {
        if std::env::var_os("_RJEM_MALLOC_CONF").is_some() {
            println!("_RJEM_MALLOC_CONF is set: the compiled posture is overridden, not read");
            return;
        }
        assert_eq!(allocator::decay_ms(), Some((-1, -1)));
    }

    /// The hooks the binary installs reach its jemalloc: the prover reads the
    /// statistics through them, and the purge hands back the pages 512 freed
    /// buffers of 1 MiB left behind (under the never-purge posture they stay
    /// resident until then).
    #[test]
    fn the_installed_hooks_read_and_purge_this_binarys_jemalloc() {
        prover::alloc_purge::install(allocator::HOOKS);
        let resident = || {
            prover::alloc_purge::stats()
                .expect("the prover reads the installed hooks")
                .resident
        };
        let buffers: Vec<Vec<u8>> = (0..512).map(|_| vec![1u8; 1 << 20]).collect();
        drop(buffers);
        let retained = resident();
        assert!((allocator::HOOKS.purge_all)(), "the purge runs");
        let purged = resident();
        assert!(
            retained >= purged + (256 << 20),
            "resident {} → {} MiB: the purge returned less than half of 512 MiB freed",
            retained >> 20,
            purged >> 20
        );
    }

    #[test]
    fn continuation_epoch_size_rejects_tiny_cli_values() {
        assert!(continuation_epoch_size(17).is_err());
    }

    #[test]
    fn continuation_epoch_size_uses_exact_power_of_two() {
        assert_eq!(continuation_epoch_size(20).unwrap(), 1 << 20);
    }

    // `accelerator_of` must match the prover's `CpuOperation::from_log`: count an
    // invocation only when the instruction is an ECALL AND a7 is the accelerator
    // syscall number. Covers both accelerators, the non-accelerator syscalls, a
    // non-ECALL whose src1 collides with an accelerator number, and a cache miss.
    #[test]
    fn accelerator_of_mirrors_prover_classification() {
        use executor::vm::instruction::execution::{ECSM_SYSCALL_NUMBER, KECCAK_SYSCALL_NUMBER};

        let ecall = Instruction::EcallEbreak;

        assert_eq!(
            accelerator_of(Some(&ecall), KECCAK_SYSCALL_NUMBER),
            Some(Accelerator::Keccak)
        );
        assert_eq!(
            accelerator_of(Some(&ecall), ECSM_SYSCALL_NUMBER),
            Some(Accelerator::Ecsm)
        );

        // Non-accelerator syscalls (Commit=64, Halt=93) count as neither.
        assert_eq!(
            accelerator_of(Some(&ecall), SyscallNumbers::Commit as u64),
            None
        );
        assert_eq!(
            accelerator_of(Some(&ecall), SyscallNumbers::Halt as u64),
            None
        );

        // A non-ECALL instruction whose src1 happens to equal an accelerator a7
        // must not count — this is the `f.ecall &&` guard the prover applies.
        assert_eq!(
            accelerator_of(Some(&Instruction::Fence), KECCAK_SYSCALL_NUMBER),
            None
        );

        // No decoded instruction at the pc (cache miss) counts as neither.
        assert_eq!(accelerator_of(None, KECCAK_SYSCALL_NUMBER), None);
    }
}
