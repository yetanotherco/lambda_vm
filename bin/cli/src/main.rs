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
        /// `Ecsm calls` / `Dma calls` (accelerator syscall invocations), and for
        /// DMA the `Dma bytes` copied and the `Dma rows` those copies add to the
        /// trace before its power-of-two padding. One `memcpy` is chunked into
        /// several DMA ecalls, so the byte and row lines, not the call count, are
        /// what the copies cost. The accelerator lines are omitted when combined
        /// with --flamegraph (that path has no per-log data).
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
    /// POSTURE` lines on stderr say which. The base hash is the build's
    /// (`hash_pin::BLOCK_BASE`), a constant `verify-block` shares.
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

        /// How each group's stacked polynomials are opened: one chain per run
        /// of up to four (`indexed`, production) or one per polynomial
        /// (`per-poly`). A format choice: `verify-block` must be given the
        /// same one; the proof file does not carry it.
        #[arg(long, value_parser = ["per-poly", "indexed"], default_value = "indexed")]
        group: String,

        /// The WHIR base code's blowup: 2 (production) or 4 (the legacy rate),
        /// its query count from the code's own rule. A format choice:
        /// `verify-block` must be given the same one; the proof file does not
        /// carry it.
        #[arg(long, value_parser = ["2", "4"], default_value = "2")]
        blowup: String,

        /// Where the WHIR round-0 trees put their leaves: each query's coset
        /// together (`coset`, production) or in the codeword's order
        /// (`natural`). A format choice: `verify-block` must be given the same
        /// one; the proof file does not carry it.
        #[arg(long, value_parser = ["natural", "coset"], default_value = "coset")]
        leaves: String,

        /// The recursion's leaf scale: a count `T` (production 10) doubles the
        /// leaves' chip heights when the single heights take `T` leaves or
        /// more; `off` keeps every leaf at the single heights. A format choice:
        /// `verify-block` must be given the same one; the proof file does not
        /// carry it.
        #[arg(long, value_parser = leaf_double_from_word, default_value = "10")]
        leaf_double_from: String,
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

        /// The group opening the block was proved with (`prove-block
        /// --group`): the verifier's constant, never read from the proof.
        #[arg(long, value_parser = ["per-poly", "indexed"], default_value = "indexed")]
        group: String,

        /// The WHIR blowup the block was proved with (`prove-block
        /// --blowup`): the verifier's constant, never read from the proof.
        #[arg(long, value_parser = ["2", "4"], default_value = "2")]
        blowup: String,

        /// The round-0 leaf order the block was proved with (`prove-block
        /// --leaves`): the verifier's constant, never read from the proof.
        #[arg(long, value_parser = ["natural", "coset"], default_value = "coset")]
        leaves: String,

        /// The leaf scale the block was proved with (`prove-block
        /// --leaf-double-from`): the verifier's constant, never read from the
        /// proof.
        #[arg(long, value_parser = leaf_double_from_word, default_value = "10")]
        leaf_double_from: String,
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
            group,
            blowup,
            leaves,
            leaf_double_from,
        } => cmd_prove_block(
            elf,
            input,
            output,
            time,
            digest,
            BlockChoices {
                group: &group,
                blowup: &blowup,
                leaves: &leaves,
                leaf_double_from: &leaf_double_from,
            },
        ),
        Commands::VerifyBlock {
            proof,
            elf,
            time,
            group,
            blowup,
            leaves,
            leaf_double_from,
        } => cmd_verify_block(
            proof,
            elf,
            time,
            BlockChoices {
                group: &group,
                blowup: &blowup,
                leaves: &leaves,
                leaf_double_from: &leaf_double_from,
            },
        ),
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
/// `verify-block` too. The base hash is not posture: it is the build's
/// (`hash_pin::BLOCK_BASE`).
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

/// The block commands' format choices, the words of `--group`, `--blowup`,
/// `--leaves` and `--leaf-double-from`: `verify-block` must be given the ones
/// `prove-block` was, and the proof file carries none of them.
struct BlockChoices<'a> {
    group: &'a str,
    blowup: &'a str,
    leaves: &'a str,
    leaf_double_from: &'a str,
}

/// `--leaf-double-from`'s words: `off` or a leaf count
/// (`whir_block::leaf_scale_setting`).
fn leaf_double_from_word(word: &str) -> Result<String, String> {
    prover::lfm::whir_block::leaf_scale_setting(Some(word))?;
    Ok(word.to_string())
}

/// The leaf scale of `--leaf-double-from`: the caller's, so the format's own
/// environment knob is refused rather than silently outranked.
fn block_leaf_scale(word: &str) -> Result<prover::lfm::whir_block::LeafScale, String> {
    use prover::lfm::whir_block::{LEAF_DOUBLE_FROM_ENV, leaf_scale_setting};
    if std::env::var_os(LEAF_DOUBLE_FROM_ENV).is_some() {
        return Err(format!(
            "{LEAF_DOUBLE_FROM_ENV} is the harness's knob: give the CLI --leaf-double-from instead"
        ));
    }
    leaf_scale_setting(Some(word))
}

fn cmd_prove_block(
    elf_path: PathBuf,
    input_path: Option<PathBuf>,
    output_path: PathBuf,
    time: bool,
    digest: bool,
    choices: BlockChoices<'_>,
) -> ExitCode {
    use prover::lfm::whir_block_tree::{
        StderrSink, WhirBlockTreeProof, WhirTreeConfig, group_setting, prove_whir_block_tree,
    };
    let BlockChoices {
        group,
        blowup,
        leaves,
        leaf_double_from,
    } = choices;

    // The group opening is the caller's constant, given on the command line
    // and to `verify-block` alike; the harness's environment word for it is
    // refused here rather than silently outranked.
    if std::env::var_os("LAMBDA_VM_WHIR_GROUP").is_some() {
        eprintln!("LAMBDA_VM_WHIR_GROUP is the harness's knob: give the CLI --group instead");
        return ExitCode::FAILURE;
    }
    // So are the WHIR blowup and the round-0 leaf order, installed as the
    // process format before anything reads it: the banner then states the
    // format this block is proved under.
    let (blowup, order) = match install_block_format(blowup, leaves) {
        Ok(choices) => choices,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("BLOCK WHIR BLOWUP: {blowup}");
    eprintln!("BLOCK WHIR LEAVES: {leaves}");
    // So is the leaves' scale: the tree's format below takes the flag's.
    let leaf_scale = match block_leaf_scale(leaf_double_from) {
        Ok(scale) => scale,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("BLOCK LEAF SCALE: {leaf_scale}");
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
    cfg.format.group = match group_setting(Some(group)) {
        Ok(group) => group,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("BLOCK GROUP OPENING: {group}");
    cfg.format.leaf_scale = leaf_scale;
    assert_eq!(
        (cfg.format.zf.whir_blowup, cfg.format.zf.whir_leaves),
        (blowup.factor(), order),
        "the tree's format is the installed one"
    );
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
    choices: BlockChoices<'_>,
) -> ExitCode {
    use prover::lfm::whir_block_tree::{
        WhirBlockTreeProof, base_line, group_setting, verify_whir_block_tree_proof_at,
    };
    let BlockChoices {
        group,
        blowup,
        leaves,
        leaf_double_from,
    } = choices;

    let fold = match group_setting(Some(group)) {
        Ok(fold) => fold,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    // The verifier is told the rate and the leaf order as it is told the group
    // opening, and both are installed as the process format before anything
    // reads it.
    let (blowup, order) = match install_block_format(blowup, leaves) {
        Ok(choices) => choices,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    // So is the leaves' scale, before the production format below reads the
    // environment's knob, which this refuses.
    let leaf_scale = match block_leaf_scale(leaf_double_from) {
        Ok(scale) => scale,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let zf = prover::block_whir::BlockFormat::production_with(fold).zf;
    assert_eq!(
        (zf.whir_blowup, zf.whir_leaves),
        (blowup.factor(), order),
        "the verifier's process format is the installed one"
    );

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
    // The block's base is its proof family's: the verifier's constant, never
    // read from the file.
    eprintln!("{}", base_line(&prover::hash_pin::BLOCK_BASE));
    // So is the group opening: the caller's, never the file's.
    eprintln!("BLOCK GROUP OPENING: {group}");
    eprintln!("BLOCK WHIR BLOWUP: {blowup}");
    eprintln!("BLOCK WHIR LEAVES: {leaves}");
    eprintln!("BLOCK LEAF SCALE: {leaf_scale}");
    eprintln!("Verifying block proof...");
    let start = Instant::now();
    let result =
        verify_whir_block_tree_proof_at(&elf_data, &proof, fold, blowup, order, leaf_scale);
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

/// The block commands' process format: the environment's ZF knobs with the
/// WHIR blowup of `--blowup` (`blowup`) and the round-0 leaf order of
/// `--leaves` (`leaves`), installed before anything reads the process format,
/// so its banner — printed once, on stdout — and every reader of it state the
/// blowup and the order the block is proved or verified under. The format's
/// own environment knobs for both are refused: the flags are the caller's.
fn install_block_format(
    blowup: &str,
    leaves: &str,
) -> Result<(prover::zf_format::WhirBlowup, prover::zf_format::LeafOrder), String> {
    use prover::zf_format::{ENV_WHIR_BLOWUP, ENV_WHIR_LEAVES, ZfFormat};
    for (knob, flag) in [(ENV_WHIR_BLOWUP, "--blowup"), (ENV_WHIR_LEAVES, "--leaves")] {
        if std::env::var_os(knob).is_some() {
            return Err(format!(
                "{knob} is the harness's knob: give the CLI {flag} instead"
            ));
        }
    }
    let blowup = prover::lfm::whir_block_tree::blowup_setting(Some(blowup))?;
    let leaves = prover::lfm::whir_block_tree::leaves_setting(Some(leaves))?;
    let format = ZfFormat {
        whir_blowup: blowup.factor(),
        whir_leaves: leaves,
        ..ZfFormat::from_env()?
    };
    format.check_leaves()?;
    ZfFormat::install(format)?;
    Ok((blowup, leaves))
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

/// One tally per [`Accelerator`] variant, printed by `execute --cycles`.
#[derive(Default)]
struct AccelCounts {
    keccak: u64,
    blake3: u64,
    ecsm: u64,
    dma: u64,
    /// Bytes copied and DMA table rows those copies consume. Keccak and ECSM
    /// cost the same per call, so DMA is the only accelerator whose report needs
    /// a size next to its count: one `memcpy` becomes as many ecalls as the
    /// guest stub chunks it into, which makes `dma` alone a poor cost proxy.
    dma_bytes: u64,
    dma_rows: u64,
}

impl AccelCounts {
    /// Exhaustive `match`: a new `Accelerator` variant is a compile error here,
    /// so it cannot be executed without also being reported. `dst_val` is the
    /// ECALL's logged operands: `dst_val` is the chunk's byte count for DMA and
    /// unused for the others; `rows` is the COPY row count the executor derived
    /// at the ecall. The width follows from `count` alone today, so the row count is
    /// recoverable from it; the slot is kept because the AIR permits a schedule that
    /// reads both ends' residues and a future one may.
    fn tally(&mut self, accelerator: Accelerator, dst_val: u64, rows: u64) {
        match accelerator {
            Accelerator::Keccak => self.keccak += 1,
            Accelerator::Blake3 => self.blake3 += 1,
            Accelerator::Ecsm => self.ecsm += 1,
            Accelerator::Dma => {
                self.dma += 1;
                self.dma_bytes += dst_val;
                self.dma_rows += rows;
            }
        }
    }
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
    let mut accel_counts: Option<AccelCounts> = None;

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
        let mut counts = AccelCounts::default();
        // Reused per chunk: `(current_pc, a7, dst_val)` for logs whose a7 matches
        // an accelerator syscall number. This is a cheap superset — a non-ECALL
        // instruction can hold the same value in src1 — that `accelerator_of`
        // confirms below, once the chunk's `&Log` borrow (tied to the executor's
        // `&mut`) is released so the instruction cache can be read again.
        let mut accel_candidates: Vec<(u64, u64, u64, u64)> = Vec::new();
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
                        accel_candidates.push((
                            log.current_pc,
                            log.src1_val,
                            log.dst_val,
                            log.src2_val,
                        ));
                    }
                }
            }
            // `logs` is no longer used, so the executor's `&mut` borrow is free
            // and the instruction cache can be read to confirm each candidate.
            for (pc, a7, dst_val, rows) in accel_candidates.drain(..) {
                if let Some(accelerator) = accelerator_of(executor.instructions.get(pc), a7) {
                    counts.tally(accelerator, dst_val, rows);
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
            accel_counts = Some(counts);
        }
        cycle_count
    };

    if cycles {
        println!("Cycles: {}", cycle_count);
        if let Some(counts) = accel_counts {
            println!("Keccak calls: {}", counts.keccak);
            println!("Blake3 calls: {}", counts.blake3);
            println!("Ecsm calls: {}", counts.ecsm);
            println!("Dma calls: {}", counts.dma);
            println!("Dma bytes: {}", counts.dma_bytes);
            println!("Dma rows: {}", counts.dma_rows);
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
    let max = prover::tables::cpu::MAX_EPOCH_SIZE_LOG2;
    if epoch_size_log2 > max {
        return Err(format!(
            "--epoch-size-log2 must be at most {max} (an epoch has fewer than 2^30 CPU rows)"
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

    /// Both block commands take the group opening, one chain per run unless
    /// told otherwise, and nothing but its two words.
    #[test]
    fn the_block_commands_take_the_group_opening() {
        let parse = |args: &[&str]| Cli::try_parse_from(args);
        let prove = ["cli", "prove-block", "guest.elf", "-o", "block.proof"];
        let verify = ["cli", "verify-block", "block.proof", "guest.elf"];
        for cmd in [&prove[..], &verify[..]] {
            let group = |args: &[&str]| match parse(args).map(|cli| cli.command) {
                Ok(Commands::ProveBlock { group, .. } | Commands::VerifyBlock { group, .. }) => {
                    Some(group)
                }
                _ => None,
            };
            assert_eq!(group(cmd).as_deref(), Some("indexed"), "the default");
            for word in ["per-poly", "indexed"] {
                let args = [cmd, &["--group", word]].concat();
                assert_eq!(group(&args).as_deref(), Some(word));
                assert!(
                    prover::lfm::whir_block_tree::group_setting(Some(word)).is_ok(),
                    "{word} reads"
                );
            }
            let args = [cmd, &["--group", "index"]].concat();
            assert!(parse(&args).is_err(), "another word is refused");
        }
    }

    /// Both block commands take the WHIR blowup, 2 unless told otherwise, and
    /// nothing but its two words; each word reads to the blowup it names.
    #[test]
    fn the_block_commands_take_the_blowup() {
        let parse = |args: &[&str]| Cli::try_parse_from(args);
        let prove = ["cli", "prove-block", "guest.elf", "-o", "block.proof"];
        let verify = ["cli", "verify-block", "block.proof", "guest.elf"];
        for cmd in [&prove[..], &verify[..]] {
            let blowup = |args: &[&str]| match parse(args).map(|cli| cli.command) {
                Ok(Commands::ProveBlock { blowup, .. } | Commands::VerifyBlock { blowup, .. }) => {
                    Some(blowup)
                }
                _ => None,
            };
            assert_eq!(blowup(cmd).as_deref(), Some("2"), "the default");
            for (word, want) in [
                ("4", prover::zf_format::WhirBlowup::Four),
                ("2", prover::zf_format::WhirBlowup::Two),
            ] {
                let args = [cmd, &["--blowup", word]].concat();
                assert_eq!(blowup(&args).as_deref(), Some(word));
                assert_eq!(
                    prover::lfm::whir_block_tree::blowup_setting(Some(word)),
                    Ok(want),
                    "{word} reads"
                );
            }
            for word in ["1", "8", "half"] {
                let args = [cmd, &["--blowup", word]].concat();
                assert!(parse(&args).is_err(), "--blowup {word} is refused");
            }
        }
        assert_eq!(
            prover::lfm::whir_block_tree::blowup_setting(None).map(|b| b.factor()),
            Ok(prover::block_whir::BlockFormat::production().zf.whir_blowup),
            "unset is the production blowup"
        );
    }

    /// Both block commands take the round-0 leaf order, the coset order unless told
    /// otherwise, and nothing but its two words; each word reads to the order
    /// it names.
    #[test]
    fn the_block_commands_take_the_leaf_order() {
        use prover::zf_format::LeafOrder;
        let parse = |args: &[&str]| Cli::try_parse_from(args);
        let prove = ["cli", "prove-block", "guest.elf", "-o", "block.proof"];
        let verify = ["cli", "verify-block", "block.proof", "guest.elf"];
        for cmd in [&prove[..], &verify[..]] {
            let leaves = |args: &[&str]| match parse(args).map(|cli| cli.command) {
                Ok(Commands::ProveBlock { leaves, .. } | Commands::VerifyBlock { leaves, .. }) => {
                    Some(leaves)
                }
                _ => None,
            };
            assert_eq!(leaves(cmd).as_deref(), Some("coset"), "the default");
            for (word, want) in [("natural", LeafOrder::Natural), ("coset", LeafOrder::Coset)] {
                let args = [cmd, &["--leaves", word]].concat();
                assert_eq!(leaves(&args).as_deref(), Some(word));
                assert_eq!(
                    prover::lfm::whir_block_tree::leaves_setting(Some(word)),
                    Ok(want),
                    "{word} reads"
                );
            }
            for word in ["Coset", "bit-reversed", "1"] {
                let args = [cmd, &["--leaves", word]].concat();
                assert!(parse(&args).is_err(), "--leaves {word} is refused");
            }
        }
        assert_eq!(
            prover::lfm::whir_block_tree::leaves_setting(None),
            Ok(prover::zf_format::ZfFormat::DEFAULT.whir_leaves),
            "unset is the production order"
        );
    }

    /// Both block commands take the leaf scale, production's threshold of 10
    /// unless told otherwise, and nothing but `off` or a leaf count; each word
    /// reads to the scale it names.
    #[test]
    fn the_block_commands_take_the_leaf_scale() {
        use prover::lfm::whir_block::LeafScale;
        let parse = |args: &[&str]| Cli::try_parse_from(args);
        let prove = ["cli", "prove-block", "guest.elf", "-o", "block.proof"];
        let verify = ["cli", "verify-block", "block.proof", "guest.elf"];
        for cmd in [&prove[..], &verify[..]] {
            let scale = |args: &[&str]| match parse(args).map(|cli| cli.command) {
                Ok(
                    Commands::ProveBlock {
                        leaf_double_from, ..
                    }
                    | Commands::VerifyBlock {
                        leaf_double_from, ..
                    },
                ) => Some(leaf_double_from),
                _ => None,
            };
            assert_eq!(scale(cmd).as_deref(), Some("10"), "the default");
            for (word, want) in [
                ("off", LeafScale::Fixed),
                ("12", LeafScale::DoubleFrom(12)),
                ("10", LeafScale::DoubleFrom(10)),
            ] {
                let args = [cmd, &["--leaf-double-from", word]].concat();
                assert_eq!(scale(&args).as_deref(), Some(word));
                assert_eq!(
                    prover::lfm::whir_block::leaf_scale_setting(Some(word)),
                    Ok(want),
                    "{word} reads"
                );
            }
            for word in ["0", "on", "twelve", "-3"] {
                let args = [cmd, &["--leaf-double-from", word]].concat();
                assert!(
                    parse(&args).is_err(),
                    "--leaf-double-from {word} is refused"
                );
            }
        }
        assert_eq!(
            prover::lfm::whir_block::leaf_scale_setting(Some("10")),
            Ok(LeafScale::DEFAULT),
            "the flag's default is production's"
        );
    }

    /// The block commands take no base hash: the block's base is its proof
    /// family's, a constant of the build, so `--base-hash` is refused on both.
    #[test]
    fn the_block_commands_take_no_base_hash() {
        let ok = |args: &[&str]| Cli::try_parse_from(args).is_ok();
        let prove = ["cli", "prove-block", "guest.elf", "-o", "block.proof"];
        let verify = ["cli", "verify-block", "block.proof", "guest.elf"];
        for cmd in [&prove[..], &verify[..]] {
            assert!(ok(cmd));
            for word in ["rpx", "p1"] {
                let args = [cmd, &["--base-hash", word]].concat();
                assert!(!ok(&args), "--base-hash {word} is refused");
            }
        }
        assert_eq!(
            prover::block_whir::BlockFormat::production().zf.base,
            stark::proof::options::BaseFormat::P1_WHIR,
            "the production base is the family's"
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
    fn continuation_epoch_size_rejects_epochs_over_the_cpu_row_bound() {
        assert!(continuation_epoch_size(prover::tables::cpu::MAX_EPOCH_SIZE_LOG2).is_ok());
        assert!(continuation_epoch_size(prover::tables::cpu::MAX_EPOCH_SIZE_LOG2 + 1).is_err());
    }

    #[test]
    fn continuation_epoch_size_uses_exact_power_of_two() {
        assert_eq!(continuation_epoch_size(20).unwrap(), 1 << 20);
    }

    /// The chip each syscall must drive, written out here rather than read back
    /// from `SyscallNumbers::accelerator()`. Comparing the CLI against the
    /// executor alone would pass if both agreed on the wrong answer — a chip
    /// demoted to `None` has to fail somewhere, and this is that somewhere.
    ///
    /// Cross-checked row by row against `SyscallNumbers::ALL`, which the
    /// executor's macro generates from the enum, so a new syscall fails the test
    /// until it gets a row here.
    const EXPECTED_ACCELERATORS: &[(SyscallNumbers, Option<Accelerator>)] = &[
        (SyscallNumbers::KeccakPermute, Some(Accelerator::Keccak)),
        (SyscallNumbers::Blake3Compress, Some(Accelerator::Blake3)),
        (SyscallNumbers::Blake3Absorb, Some(Accelerator::Blake3)),
        (SyscallNumbers::Ecsm, Some(Accelerator::Ecsm)),
        (SyscallNumbers::DmaMemcpy, Some(Accelerator::Dma)),
        (SyscallNumbers::DmaMemset, Some(Accelerator::Dma)),
        (SyscallNumbers::Print, None),
        (SyscallNumbers::Panic, None),
        (SyscallNumbers::Commit, None),
        (SyscallNumbers::Halt, None),
        // `hint` drives its own HINT table, but the executor maps it to no
        // `Accelerator`: the ecall adds no correctness constraint, so there is no
        // accelerated work to attribute. `execute --cycles` reports no hint line.
        (SyscallNumbers::Hint, None),
    ];

    // `accelerator_of` must match the prover's `CpuOperation::from_log`: count an
    // invocation only when the instruction is an ECALL AND a7 is the accelerator
    // syscall number.
    #[test]
    fn accelerator_of_mirrors_prover_classification() {
        let ecall = Instruction::EcallEbreak;

        for &syscall in SyscallNumbers::ALL {
            let rows = EXPECTED_ACCELERATORS
                .iter()
                .filter(|(listed, _)| *listed == syscall)
                .count();
            assert_eq!(
                rows, 1,
                "{syscall:?} needs exactly one row in EXPECTED_ACCELERATORS"
            );
        }

        for &(syscall, expected) in EXPECTED_ACCELERATORS {
            assert_eq!(
                accelerator_of(Some(&ecall), syscall.raw()),
                expected,
                "ECALL with a7 of {syscall:?} must classify as {expected:?}"
            );
            // A non-ECALL instruction whose src1 happens to equal a syscall a7
            // must not count — this is the `f.ecall &&` guard the prover applies.
            assert_eq!(
                accelerator_of(Some(&Instruction::Fence), syscall.raw()),
                None,
                "non-ECALL with a7 of {syscall:?} must not count"
            );
            // No decoded instruction at the pc (cache miss) counts as neither.
            assert_eq!(accelerator_of(None, syscall.raw()), None);
        }
    }

    // Every tallied accelerator gets its own counter: no two variants may share
    // a field, and each must land in the one the report prints.
    #[test]
    fn accel_counts_tallies_each_accelerator_separately() {
        for &(_, expected_accelerator) in EXPECTED_ACCELERATORS {
            let Some(accelerator) = expected_accelerator else {
                continue;
            };
            let mut counts = AccelCounts::default();
            counts.tally(accelerator, 0, 0);
            assert_eq!(
                counts.keccak + counts.blake3 + counts.ecsm + counts.dma,
                1,
                "{accelerator:?} must increment exactly one counter"
            );
            let expected = match accelerator {
                Accelerator::Keccak => counts.keccak,
                Accelerator::Blake3 => counts.blake3,
                Accelerator::Ecsm => counts.ecsm,
                Accelerator::Dma => counts.dma,
            };
            assert_eq!(
                expected, 1,
                "{accelerator:?} must increment its own counter"
            );
        }
    }

    // The byte and row lines are what make the DMA report a cost figure rather
    // than a call count, so they must accumulate across chunked ecalls and use
    // the executor's row formula — the same one trace generation sizes with.
    #[test]
    fn accel_counts_sizes_dma_calls() {
        let mut counts = AccelCounts::default();
        // Rows as the executor derives them at each ecall.
        for bytes in [256, 256, 8, 3, 0] {
            let rows =
                executor::vm::instruction::execution::copy_trace_rows(0x1000, 0x2000, bytes, false);
            counts.tally(Accelerator::Dma, bytes, rows);
        }

        assert_eq!(counts.dma, 5, "every DMA ecall counts as one call");
        assert_eq!(counts.dma_bytes, 523);
        // The rows are the executor's, derived where src, dst and count are all known;
        // the report only sums them.
        assert_eq!(counts.dma_rows, 32 + 32 + 1 + 3 + 1);

        // The other accelerators must leave the DMA size lines alone.
        let mut others = AccelCounts::default();
        others.tally(Accelerator::Keccak, 200, 0);
        others.tally(Accelerator::Ecsm, 32, 0);
        assert_eq!((others.dma, others.dma_bytes, others.dma_rows), (0, 0, 0));
    }
}
