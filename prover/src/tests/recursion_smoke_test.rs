//! End-to-end naive recursion pipeline smoke tests.
//!
//! Each test:
//! 1. Proves an inner program on the host.
//! 2. Serializes `(VmProof, inner_elf)` with postcard.
//! 3. Hands that as private input to the recursion guest.
//! 4. Either **proves** the recursion guest's execution (memory-bounded via
//!    continuations) and verifies the outer proof (`OuterMode::Prove`), or
//!    merely **executes** the guest in-VM and reads the committed marker
//!    straight off the executor's memory (`OuterMode::ExecuteOnly`) — a cheaper
//!    tier that skips the LDE/FRI that dominate the full pipeline.
//!
//! The guest ELFs are assumed built by `make compile-recursion-elfs`.

use std::ops::ControlFlow;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

/// Read a recursion-suite guest ELF artifact, built by `make compile-recursion-elfs`.
fn read_guest_elf(root: &std::path::Path, name: &str) -> Vec<u8> {
    let path = root.join(format!("executor/program_artifacts/recursion/{name}.elf"));
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read {} — run `make compile-recursion-elfs`: {e}",
            path.display()
        )
    })
}

/// Minimum-security FRI parameters: blowup=2, a single FRI query. Security is
/// intentionally terrible — used by the capacity-probing test and every cheap
/// diagnostic below, where the goal is the smallest possible inner proof, not
/// a sound one. (`GoldilocksCubicProofOptions::with_blowup` derives a query
/// count from a 128-bit target, far more than we want here.)
const MIN_PROOF_OPTIONS: stark::proof::options::ProofOptions =
    stark::proof::options::ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 1,
        coset_offset: 3,
        grinding_factor: 1,
    };

/// Prove `inner_elf` (fed `inner_input`) under `opts`, then package
/// `(proof, elf, opts)` into the postcard blob the recursion and
/// deserialize-only guests consume as their private input. `tag` prefixes the
/// progress lines. Returns the inner proof — callers that re-verify it on the
/// host need it — next to the encoded blob.
fn prove_inner_and_encode_blob(
    tag: &str,
    inner_elf: &[u8],
    inner_input: &[u8],
    opts: &stark::proof::options::ProofOptions,
) -> (crate::VmProof, Vec<u8>) {
    eprintln!(
        "[{tag}] proving inner (blowup={}, fri_queries={}) ...",
        opts.blowup_factor, opts.fri_number_of_queries
    );
    let inner_proof = crate::prove_with_options_and_inputs(
        inner_elf,
        inner_input,
        opts,
        &crate::MaxRowsConfig::default(),
    )
    .expect("inner prove should succeed");

    let blob =
        postcard::to_allocvec(&(&inner_proof, &inner_elf, opts)).expect("postcard encode failed");
    eprintln!("[{tag}] postcard blob: {} bytes", blob.len());
    (inner_proof, blob)
}

/// How far to take the recursion guest after it has been handed the inner
/// proof. The guest under test is the verifier either way — this only chooses
/// whether we also prove the guest's own execution.
#[derive(Clone, Copy, Debug)]
enum OuterMode {
    /// Execute the guest in-VM and read the committed marker straight off the
    /// executor's memory. Streams logs via `Executor::resume()` and never
    /// builds a `Traces`, so footprint stays bounded to the VM's touched
    /// memory + instruction cache. Skips the LDE/FRI of the full pipeline entirely.
    ExecuteOnly,
    /// Prove the guest's execution via continuations, then verify the outer
    /// proof on the host. `prove_and_verify_continuation` retains every epoch's
    /// STARK proof in the bundle before verifying, so peak RAM grows with epoch
    /// count. Heavy — excluded from CI, run manually. A future verify-one-and-
    /// discard API extension would make this memory-friendlier.
    Prove,
}

/// Execute the recursion guest in-VM on `blob` and return the bytes it
/// committed (the success marker the in-VM verifier emits).
///
/// Streams execution via `Executor::resume()`. The committed marker is
/// read directly off the executor's memory. This avoids OOMs.
fn execute_outer_and_commit(label: &str, recursion_elf_bytes: &[u8], blob: &[u8]) -> Vec<u8> {
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    eprintln!("[{label}] executing outer (recursion guest, in-VM verify, streaming) ...");
    let program = Elf::load(recursion_elf_bytes).expect("load recursion elf");
    let mut executor = Executor::new(&program, blob.to_vec()).expect("executor new");

    let (total_cycles, exec_time) =
        drive_executor(&mut executor, |_log| ControlFlow::Continue(()), |_, _, _| {});

    let committed = executor
        .finish()
        .expect("read committed output after execution")
        .memory_values;

    eprintln!(
        "[{label}] {total_cycles} cycles in {exec_time:?}; committed {} bytes: {:?} (as str: {:?})",
        committed.len(),
        committed,
        String::from_utf8_lossy(&committed),
    );
    committed
}

/// Epoch size for the outer prove: 2^16 ≈ 65K cycles. Small so one epoch's
/// trace+LDE stays under the ~16GiB CI runners.
const OUTER_EPOCH_SIZE_LOG2: u32 = 16;

/// Prove the recursion guest's execution on `blob` memory-bounded via
/// continuations and verify the bundle on the host, returning the bytes the
/// guest committed.
fn prove_outer_and_commit(label: &str, recursion_elf_bytes: &[u8], blob: &[u8]) -> Vec<u8> {
    let opts =
        crate::GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is always valid");
    eprintln!(
        "[{label}] proving outer (recursion guest) via continuations \
         (epoch=2^{OUTER_EPOCH_SIZE_LOG2} cycles) ..."
    );
    let committed = crate::continuation::prove_and_verify_continuation(
        recursion_elf_bytes,
        blob,
        OUTER_EPOCH_SIZE_LOG2,
        &opts,
    )
    .expect("outer continuation prove/verify errored")
    .expect("outer continuation proof must verify on host");
    eprintln!("[{label}] outer continuation proof generated and verified");
    committed
}

/// Stream a guest's execution via `Executor::resume()`, calling `on_log` for
/// every `Log` without ever buffering the full log stream (`Executor::run`
/// would accumulate tens of millions of `Log`s and OOM even a 125 GB box).
/// `on_log` returns `ControlFlow::Break(())` to stop the run early (e.g. once a
/// cycle budget is hit); `Continue(())` to keep going. `on_progress(chunks,
/// total_cycles, elapsed)` fires once per resumed chunk; callers throttle and
/// format their own progress lines. Returns `(total_cycles, wall_time)` —
/// `total_cycles` counts logs actually visited, so it is exact even when a run
/// breaks mid-chunk.
fn drive_executor(
    executor: &mut executor::vm::execution::Executor,
    mut on_log: impl FnMut(&executor::vm::logs::Log) -> ControlFlow<()>,
    mut on_progress: impl FnMut(usize, u64, std::time::Duration),
) -> (u64, std::time::Duration) {
    let start = std::time::Instant::now();
    let mut total_cycles: u64 = 0;
    let mut chunks: usize = 0;
    while let Some(logs) = executor.resume().expect("executor resume failed (guest panicked in-VM?)") {
        let mut stop = false;
        for log in logs {
            total_cycles += 1;
            if on_log(log).is_break() {
                stop = true;
                break;
            }
        }
        chunks += 1;
        on_progress(chunks, total_cycles, start.elapsed());
        if stop {
            break;
        }
    }
    (total_cycles, start.elapsed())
}

/// Shared preamble for every execute-only diagnostic below: build the standard
/// recursion private-input blob (an `empty`-program inner proof produced under
/// `opts`), load guest `guest_name`, and stand up an executor over it. Returns
/// the guest's raw ELF bytes (callers that resolve PCs pass them to
/// [`executor::elf::SymbolTable::parse`]), the loaded program, and the
/// ready-to-drive executor.
fn setup_guest_run(
    label: &str,
    guest_name: &str,
    opts: &stark::proof::options::ProofOptions,
) -> (
    Vec<u8>,
    executor::elf::Elf,
    executor::vm::execution::Executor,
) {
    let root = workspace_root();
    let empty_elf_bytes = read_guest_elf(&root, "empty");
    let guest_elf_bytes = read_guest_elf(&root, guest_name);

    let (_inner_proof, blob) = prove_inner_and_encode_blob(label, &empty_elf_bytes, &[], opts);

    let program = executor::elf::Elf::load(&guest_elf_bytes).expect("ELF load failed");
    assert_ne!(
        program.entry_point, 0,
        "{guest_name} ELF has entry_point=0 — build artifact is malformed"
    );
    let executor =
        executor::vm::execution::Executor::new(&program, blob).expect("Executor::new failed");
    (guest_elf_bytes, program, executor)
}

/// A `drive_executor` progress callback that prints the throttled
/// `[label]   ... N chunks, M cycles, T elapsed` line every `stride` chunks —
/// the readout the counting diagnostics share. Tests that need extra live state
/// (unique PC count, active step bucket) keep their own closure instead. Takes
/// `impl Into<String>` so it works with both `&'static` tags and a run's
/// dynamic `label`.
fn log_progress(label: impl Into<String>, stride: usize) -> impl FnMut(usize, u64, std::time::Duration) {
    let label = label.into();
    move |chunks, cycles, elapsed| {
        if chunks.is_multiple_of(stride) {
            eprintln!("[{label}]   ... {chunks} chunks, {cycles} cycles, {elapsed:?} elapsed");
        }
    }
}

/// Resolve a guest PC to its (demangled) enclosing function name using the
/// ELF's own symbol table — the same data `executor::flamegraph` resolves
/// against. `<unknown>` when no function symbol covers the PC (e.g. PLT stubs
/// or a release build that dropped symbols). No file:line: the symbol table
/// carries function ranges only, not DWARF line info.
fn resolve_pc(symbols: &executor::elf::SymbolTable, pc: u64) -> String {
    symbols.lookup(pc).map_or_else(
        || "<unknown>".to_string(),
        |s| executor::flamegraph::demangle(&s.name),
    )
}

/// Verifier sub-routines in execution order. LLVM inlines the step bodies, but
/// closures inside each keep the method name in their mangled symbol, so
/// `run_profile` advances the step bucket by substring-matching the enclosing
/// symbol. A step with no matching symbol merges into the previous bucket.
const VERIFIER_STEP_KEYWORDS: [&str; 4] = [
    "replay_rounds_after_round_1",
    "step_2_verify_claimed_composition_polynomial",
    "step_3_verify_fri",
    "step_4_verify_trace_and_composition_openings",
];

/// `blowup=8` inner-proof options: the security-derived multi-query count (tens
/// of queries, 128-bit) used by every `multiquery` profiling variant.
fn blowup8() -> stark::proof::options::ProofOptions {
    crate::GoldilocksCubicProofOptions::with_blowup(8).expect("blowup=8 is always valid")
}

/// Fold the PC histogram by enclosing function and print the top-25 by cycles.
/// Folded because an inlined kernel spreads across many PCs; no per-address
/// table since a bare PC isn't actionable without file:line.
fn print_function_table(
    symbols: &executor::elf::SymbolTable,
    pc_hist: std::collections::HashMap<u64, u64>,
    total_cycles: u64,
) {
    let mut by_function: std::collections::HashMap<String, (u64, u64)> =
        std::collections::HashMap::new();
    for (pc, count) in &pc_hist {
        let entry = by_function.entry(resolve_pc(symbols, *pc)).or_insert((0, 0));
        entry.0 += *count; // cycles
        entry.1 += 1; // distinct PCs folded into this function
    }
    let mut fn_entries: Vec<(String, (u64, u64))> = by_function.into_iter().collect();
    fn_entries.sort_unstable_by_key(|(_name, (cycles, _pcs))| std::cmp::Reverse(*cycles));

    let pct = |n: u64| 100.0 * (n as f64) / (total_cycles as f64);
    eprintln!("  Unique PCs   : {}", pc_hist.len());
    eprintln!();
    eprintln!("  Top 25 functions by cycle count (aggregated over their PCs):");
    eprintln!("  rank          cycles        %    cum %    PCs  function");
    let mut fn_cumulative: u64 = 0;
    for (rank, (name, (cycles, pcs))) in fn_entries.iter().take(25).enumerate() {
        fn_cumulative += cycles;
        eprintln!(
            "  {:>4}  {:>14}  {:>6.2}%  {:>6.2}%  {:>5}  {}",
            rank + 1,
            cycles,
            pct(*cycles),
            pct(fn_cumulative),
            pcs,
            name,
        );
    }
}

/// Print the monotonic per-verifier-step cycle bucketing. `buckets[0]` is
/// pre-step-1 setup (alloc + postcard decode + `VmAirs::new`); `buckets[i]` is
/// verifier step i (with a missing step's cycles merged into the previous one).
fn print_step_breakdown(buckets: &[u64; 5], total_cycles: u64) {
    let labels = [
        "0. setup (alloc + postcard decode + VmAirs::new + pre-step-1)",
        "1. step 1: replay_rounds_after_round_1",
        "2. step 2: verify_claimed_composition_polynomial",
        "3. step 3: verify_fri",
        "4. step 4: verify_trace_and_composition_openings (+ wrap-up)",
    ];
    eprintln!();
    eprintln!("  Per-step cycle breakdown (monotonic state machine):");
    eprintln!("  {:<60}  {:>14}  {:>7}", "bucket", "cycles", "%");
    for (label, cycles) in labels.iter().zip(buckets.iter()) {
        let pct = if total_cycles > 0 {
            100.0 * (*cycles as f64) / (total_cycles as f64)
        } else {
            0.0
        };
        eprintln!("  {:<60}  {:>14}  {:>6.2}%", label, cycles, pct);
    }
}

/// Single-pass execute-only profiler. Always prints total cycles + wall time +
/// a rough trace/LDE size estimate. With `detailed`, the same pass also builds
/// the PC histogram and verifier-step bucketing and prints the top-25 functions
/// and the per-step breakdown (the two always come together); `!detailed` does
/// no per-log work, so it's just a fast cycle counter. `progress_stride`
/// throttles the readout (recursion large, the deserialize-only control small).
fn run_profile(
    guest_name: &str,
    progress_stride: usize,
    opts: stark::proof::options::ProofOptions,
    detailed: bool,
) {
    use std::collections::HashMap;

    let (guest_elf_bytes, _program, mut executor) = setup_guest_run("profile", guest_name, &opts);
    let symbols = executor::elf::SymbolTable::parse(&guest_elf_bytes);

    let mut pc_hist: HashMap<u64, u64> = HashMap::new();
    let mut buckets = [0u64; 5];
    let mut last_range: Option<(u64, u64)> = None;
    let mut last_advance: u8 = 0;
    let bucket = std::cell::Cell::new(0u8);
    let unique = std::cell::Cell::new(0usize);

    if detailed {
        assert!(
            !symbols.is_empty(),
            "{guest_name} ELF has no symbol table — was it stripped?"
        );
        for (i, kw) in VERIFIER_STEP_KEYWORDS.iter().enumerate() {
            let n = symbols.functions().iter().filter(|f| f.name.contains(kw)).count();
            eprintln!(
                "[profile] step {}: keyword={kw:?} -> {n} symbol(s) {}",
                i + 1,
                if n > 0 { "" } else { "(no match; merges into previous bucket)" },
            );
        }
    }

    eprintln!(
        "[profile] executing {guest_name} guest ({}) ...",
        if detailed { "histogram + steps" } else { "cycle counter" }
    );
    let (total_cycles, exec_time) = drive_executor(
        &mut executor,
        |log| {
            if detailed {
                let pc = log.current_pc;
                *pc_hist.entry(pc).or_insert(0) += 1;
                unique.set(pc_hist.len());

                let in_cached = matches!(last_range, Some((s, e)) if pc >= s && pc < e);
                if !in_cached {
                    if let Some(sym) = symbols.lookup(pc) {
                        last_range = Some((sym.address, sym.address + sym.size.max(1)));
                        last_advance = 0;
                        for (i, kw) in VERIFIER_STEP_KEYWORDS.iter().enumerate() {
                            if sym.name.contains(kw) {
                                last_advance = (i + 1) as u8;
                            }
                        }
                    } else {
                        last_range = None;
                        last_advance = 0;
                    }
                }
                if bucket.get() < last_advance {
                    bucket.set(last_advance);
                }
                buckets[bucket.get() as usize] += 1;
            }
            ControlFlow::Continue(())
        },
        |chunks, cycles, elapsed| {
            if chunks.is_multiple_of(progress_stride) {
                if detailed {
                    eprintln!(
                        "[profile]   ... {chunks} chunks, {cycles} cycles, {} unique PCs, bucket={}, {elapsed:?}",
                        unique.get(),
                        bucket.get(),
                    );
                } else {
                    eprintln!("[profile]   ... {chunks} chunks, {cycles} cycles, {elapsed:?}");
                }
            }
        },
    );

    eprintln!();
    eprintln!("============================================================");
    eprintln!(
        "  {} GUEST PROFILE (blowup={}, {} queries)",
        guest_name.to_uppercase(),
        opts.blowup_factor,
        opts.fri_number_of_queries,
    );
    eprintln!("============================================================");
    eprintln!("  Total cycles : {total_cycles}");
    eprintln!("  Exec time    : {exec_time:?}");
    eprintln!();
    eprintln!("  Rough trace/LDE size if this guest were proven:");
    let approx_columns = 250u64;
    let main_trace_bytes = total_cycles * approx_columns * 8;
    eprintln!(
        "    main trace          : ~{:.2} GB ({total_cycles} cycles × ~{approx_columns} cols × 8 B)",
        main_trace_bytes as f64 / 1e9,
    );
    eprintln!(
        "    main LDE (blowup=2) : ~{:.2} GB  (+aux ≈ 50% more → peak ≈ 2-3× LDE)",
        (main_trace_bytes * 2) as f64 / 1e9,
    );

    if detailed {
        eprintln!();
        print_function_table(&symbols, pc_hist, total_cycles);
        print_step_breakdown(&buckets, total_cycles);
    }
    eprintln!("============================================================");
}

/// Core pipeline: prove an inner program with the given options, hand the
/// proof+ELF+options to the recursion guest, then take the guest to `mode`
/// (execute-only or full prove) and assert it committed the `[1]` success
/// marker — i.e. the in-VM verifier accepted the inner proof.
fn run_recursion_pipeline_with_options(
    label: &str,
    inner_elf_bytes: &[u8],
    inner_private_input: &[u8],
    inner_proof_options: stark::proof::options::ProofOptions,
    mode: OuterMode,
) {
    let root = workspace_root();
    let recursion_elf_bytes = read_guest_elf(&root, "recursion");

    let (inner_proof, blob) = prove_inner_and_encode_blob(
        label,
        inner_elf_bytes,
        inner_private_input,
        &inner_proof_options,
    );

    assert!(
        crate::verify_with_options(
            &inner_proof,
            inner_elf_bytes,
            &inner_proof_options,
            None,
            None
        )
        .expect("inner verify errored"),
        "inner proof must verify on host"
    );
    assert!(
        blob.len() <= executor::vm::memory::MAX_PRIVATE_INPUT_SIZE as usize,
        "recursion input exceeds MAX_PRIVATE_INPUT_SIZE"
    );

    let committed = match mode {
        OuterMode::ExecuteOnly => execute_outer_and_commit(label, &recursion_elf_bytes, &blob),
        OuterMode::Prove => prove_outer_and_commit(label, &recursion_elf_bytes, &blob),
    };

    assert_eq!(
        committed,
        vec![1u8],
        "recursion guest must commit the [1] success marker (in-VM verify accepted)"
    );
    eprintln!("[{label}] guest committed [1]: in-VM verify accepted ✓");
}

/// Convenience wrapper using `blowup=8` for the inner proof — the default for
/// the `empty` and `fibonacci` cases, chosen to keep outer-prove memory tractable.
fn run_recursion_pipeline(
    label: &str,
    inner_elf_bytes: &[u8],
    inner_private_input: &[u8],
    mode: OuterMode,
) {
    let inner_proof_options = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(8)
        .expect("blowup=8 is always valid");
    run_recursion_pipeline_with_options(
        label,
        inner_elf_bytes,
        inner_private_input,
        inner_proof_options,
        mode,
    );
}

/// Reproduce the recursion guest's EXACT path on the host — decode the postcard
/// blob into `(VmProof, Vec<u8>, ProofOptions)` and call `verify_with_options`.
/// Cheap regression guard.
#[test]
#[ignore = "needs prebuilt guest ELF (make compile-recursion-elfs)"]
fn test_recursion_blob_decodes_and_verifies_on_host() {
    let root = workspace_root();
    let empty_elf_bytes = read_guest_elf(&root, "empty");
    let (_inner, blob) =
        prove_inner_and_encode_blob("roundtrip", &empty_elf_bytes, &[], &MIN_PROOF_OPTIONS);

    // Decode exactly as the guest does.
    let decoded: Result<(crate::VmProof, Vec<u8>, crate::ProofOptions), _> =
        postcard::from_bytes(&blob);
    let (vm_proof, inner_elf, options) = match decoded {
        Ok(t) => t,
        Err(e) => panic!("[roundtrip] postcard DECODE failed (this is the guest panic): {e}"),
    };
    eprintln!(
        "[roundtrip] decode ok: elf {} bytes, blowup {}",
        inner_elf.len(),
        options.blowup_factor
    );

    match crate::verify_with_options(&vm_proof, &inner_elf, &options, None, None) {
        Ok(true) => eprintln!("[roundtrip] verify ok=true — guest path is sound"),
        Ok(false) => panic!(
            "[roundtrip] verify returned FALSE (guest hits assert!(ok)) — proof did not survive the postcard round-trip"
        ),
        Err(e) => panic!("[roundtrip] verify ERRORED (guest hits .expect): {e:?}"),
    }
}

// === Execute-only tier ========================================================

/// Execute-only mirror of `test_recursion_prove_empty`: verify a `blowup=8`
/// proof of the empty program in-VM.
#[test]
#[ignore = "slow: runs the in-VM STARK verifier (minutes on CI)"]
fn test_recursion_execute_empty() {
    let root = workspace_root();
    let empty_elf_bytes = read_guest_elf(&root, "empty");
    run_recursion_pipeline(
        "recursion-exec-empty",
        &empty_elf_bytes,
        &[],
        OuterMode::ExecuteOnly,
    );
}

/// Execute-only mirror of `test_recursion_prove_1query`: smallest possible
/// inner proof (blowup=2, 1 query) → least guest work.
#[test]
#[ignore = "slow: runs the in-VM STARK verifier (minutes on CI)"]
fn test_recursion_execute_1query() {
    let root = workspace_root();
    let empty_elf_bytes = read_guest_elf(&root, "empty");
    run_recursion_pipeline_with_options(
        "recursion-exec-1query",
        &empty_elf_bytes,
        &[],
        MIN_PROOF_OPTIONS,
        OuterMode::ExecuteOnly,
    );
}

/// Execute-only mirror of `test_recursion_prove`: verify a `blowup=8` proof of
/// fibonacci(10) in-VM.
#[test]
#[ignore = "slow: runs the in-VM STARK verifier (minutes on CI)"]
fn test_recursion_execute() {
    let root = workspace_root();
    let fib_elf_bytes = read_guest_elf(&root, "fibonacci");

    let n: u64 = 10;
    let inner_private_input = n.to_le_bytes().to_vec();

    run_recursion_pipeline(
        "recursion-exec-fib",
        &fib_elf_bytes,
        &inner_private_input,
        OuterMode::ExecuteOnly,
    );
}

// === Full-prove tier ==========================================================

/// Inner program: empty (halt immediately). Useful for measuring the
/// verifier's intrinsic recursion overhead.
#[test]
#[ignore = "slow: memory-bounded continuation prove of the verifier-in-VM"]
fn test_recursion_prove_empty() {
    let root = workspace_root();
    let empty_elf_bytes = read_guest_elf(&root, "empty");
    run_recursion_pipeline(
        "recursion-prove-empty",
        &empty_elf_bytes,
        &[],
        OuterMode::Prove,
    );
}

/// Inner program: empty, but with the absolute-minimum FRI parameters
/// (blowup=2, **fri_number_of_queries=1**). For quick profiling only.
#[test]
#[ignore = "slow: memory-bounded continuation prove of the verifier-in-VM"]
fn test_recursion_prove_1query() {
    let root = workspace_root();
    let empty_elf_bytes = read_guest_elf(&root, "empty");

    run_recursion_pipeline_with_options(
        "recursion-prove-1query",
        &empty_elf_bytes,
        &[],
        MIN_PROOF_OPTIONS,
        OuterMode::Prove,
    );
}

/// Diagnostic: build the inner proof and dump the recursion guest's private-input
/// blob to `/tmp/recursion_input.bin` so the CLI's `execute --flamegraph` can
/// consume it.
///
/// Usage after running this test:
/// ```
/// cargo run -p cli --release -- execute \
///     bench_vs/lambda/recursion/target/riscv64im-lambda-vm-elf/release/recursion-bench \
///     --private-input /tmp/recursion_input.bin \
///     --flamegraph /tmp/recursion_folded.txt
/// cat /tmp/recursion_folded.txt | inferno-flamegraph > /tmp/recursion_flamegraph.svg
/// ```
#[test]
#[ignore = "diagnostic: writes recursion private input to /tmp/recursion_input.bin"]
fn test_dump_recursion_input() {
    let root = workspace_root();
    let empty_elf_bytes = read_guest_elf(&root, "empty");

    let (_inner_proof, blob) =
        prove_inner_and_encode_blob("dump-input", &empty_elf_bytes, &[], &MIN_PROOF_OPTIONS);

    let path = "/tmp/recursion_input.bin";
    std::fs::write(path, &blob).expect("write blob");
    eprintln!("[dump-input] wrote {} bytes to {path}", blob.len());
}

/// Cycle count only of the recursion guest verifying a 1-query inner proof.
#[test]
#[ignore = "diagnostic: fast; recursion guest cycle count (1 query)"]
fn test_recursion_cycles_1query() {
    run_profile("recursion", 500, MIN_PROOF_OPTIONS, false);
}

/// Cycle count only at 128-bit security: more FRI queries → more verifier cycles.
#[test]
#[ignore = "diagnostic: fast; recursion guest cycle count (multi-query)"]
fn test_recursion_cycles_multiquery() {
    run_profile("recursion", 500, blowup8(), false);
}

/// Full profile (top-25 functions + per-step breakdown) of the 1-query run —
/// the cheapest verifier run, dominated by fixed setup.
#[test]
#[ignore = "diagnostic: ~8 min; recursion guest histogram + steps (1 query)"]
fn test_recursion_profile_1query() {
    run_profile("recursion", 500, MIN_PROOF_OPTIONS, true);
}

/// Full profile at 128-bit security: weight shifts toward per-query FRI/Merkle.
#[test]
#[ignore = "diagnostic: heavy; recursion guest histogram + steps (multi-query)"]
fn test_recursion_profile_multiquery() {
    run_profile("recursion", 500, blowup8(), true);
}

/// Diagnostic: count the distinct 4 KB memory pages the recursion guest
/// touches when verifying a small inner proof.
///
/// We suspect the outer prover's 125 GB OOM wall is dominated by per-page
/// PAGE-table overhead. The number of PAGE tables the prover would build
/// equals the number of distinct 4 KB pages the executor touches — code,
/// heap, private input, and stack. This test surfaces that count without
/// running the prover.
///
/// Layout (per `executor::constants` + `bench_vs/lambda/recursion/src/main.rs`):
/// - Code/static: whatever PT_LOAD segments the recursion ELF carries.
/// - Heap: `_end .. 0xC000_0000` (`MAX_MEMORY_SIZE`); `TlsfHeap` scatters
///   allocations across this region.
/// - Private input: starts at `PRIVATE_INPUT_START_INDEX = 0xFF000000`.
/// - Stack: top of address space (down from `STACK_TOP = 0xFFFFFFFFFFFFFFF0`).
///
/// Interpretation (rough):
/// - <1,000 pages: PAGE-table overhead is not the bottleneck.
/// - 10k-100k pages: TLSF heap fragmentation; design a tighter bump allocator
///   and re-measure.
/// - >100k pages: postcard decode dominates; consider streaming decode.
#[test]
#[ignore = "diagnostic: counts distinct 4 KB memory pages touched by the recursion guest"]
fn test_recursion_page_count() {
    use executor::vm::memory::PRIVATE_INPUT_START_INDEX;
    use std::collections::HashSet;

    let (_bytes, program, mut executor) =
        setup_guest_run("page-count", "recursion", &MIN_PROOF_OPTIONS);

    // Precompute the recursion ELF's PT_LOAD ranges so we can bucket code/
    // static pages separately from heap. `Elf::load` already expands BSS
    // (memsz > filesz) into zero-valued words, so these ranges cover
    // .text + .rodata + .data + .bss.
    let segment_ranges: Vec<(u64, u64)> = program
        .data
        .iter()
        .map(|seg| (seg.base_addr, seg.base_addr + (seg.values.len() as u64 * 4)))
        .collect();
    eprintln!(
        "[page-count] recursion ELF: {} PT_LOAD segment(s)",
        segment_ranges.len(),
    );
    for (i, (lo, hi)) in segment_ranges.iter().enumerate() {
        eprintln!(
            "[page-count]   segment[{i}]: 0x{lo:016x} .. 0x{hi:016x} ({} bytes)",
            hi - lo,
        );
    }

    // Stream through execution — running to completion via `Executor::run`
    // would accumulate ~67 M `Log` records (~2.7 GB) we don't need. We only
    // care about the *final* memory state.
    eprintln!("[page-count] executing recursion guest (streaming) ...");
    let (total_cycles, exec_time) = drive_executor(
        &mut executor,
        |_log| ControlFlow::Continue(()),
        log_progress("page-count", 50),
    );

    // Collect the set of distinct 4 KB pages from every cell touched during
    // (a) program loading, (b) private-input loading, (c) execution.
    const PAGE_MASK: u64 = !0xFFFu64;
    let cells = executor.memory().cells();
    let total_cells = cells.len();
    let pages: HashSet<u64> = cells.keys().map(|&a| a & PAGE_MASK).collect();

    // Bucket by region. A "code/static" page is any page that overlaps a
    // PT_LOAD segment. Stack lives near the top of the 64-bit address
    // space; private input lives in the [0xFF000000, ...) window above the
    // 3 GB heap ceiling.
    const HEAP_CEILING: u64 = 0xC000_0000;
    const STACK_FLOOR: u64 = 0xFFFF_FFFF_0000_0000;

    let mut code_pages = 0usize;
    let mut heap_pages = 0usize;
    let mut private_input_pages = 0usize;
    let mut stack_pages = 0usize;
    let mut other_pages = 0usize;

    for &page in &pages {
        let page_end = page.saturating_add(0x1000);
        let in_code = segment_ranges
            .iter()
            .any(|&(lo, hi)| page < hi && lo < page_end);
        if in_code {
            code_pages += 1;
        } else if page >= STACK_FLOOR {
            stack_pages += 1;
        } else if page >= PRIVATE_INPUT_START_INDEX {
            private_input_pages += 1;
        } else if page < HEAP_CEILING {
            heap_pages += 1;
        } else {
            other_pages += 1;
        }
    }

    eprintln!();
    eprintln!("============================================================");
    eprintln!("  RECURSION GUEST PAGE-COUNT SUMMARY");
    eprintln!("============================================================");
    eprintln!("  Total cycles                  : {total_cycles}");
    eprintln!("  Executor wall time            : {exec_time:?}");
    eprintln!("  Memory cells touched (4 B ea) : {total_cells}");
    eprintln!("  Distinct 4 KB pages touched   : {}", pages.len());
    eprintln!();
    eprintln!("  Pages per region:");
    eprintln!("    code/static (ELF segments)     : {code_pages}");
    eprintln!("    heap (0..0xC000_0000)          : {heap_pages}");
    eprintln!("    private input (0xFF000000..)   : {private_input_pages}");
    eprintln!("    stack (>= 0xFFFFFFFF_00000000) : {stack_pages}");
    if other_pages > 0 {
        eprintln!("    other (unclassified)           : {other_pages}");
    }
    eprintln!();
    eprintln!("  Interpretation (PAGE-table overhead):");
    eprintln!("    <1k pages     → PAGE overhead is not the bottleneck.");
    eprintln!("    10k-100k      → TLSF heap fragmentation; try a bump alloc.");
    eprintln!("    >100k         → postcard decode dominates; stream-decode?");
    eprintln!("============================================================");
}

/// Diagnostic: build a **sampled** call-stack histogram of the recursion guest.
///
/// Like `test_recursion_pc_histogram` but groups by full call stack (not PC).
/// To stay fast, only every `SAMPLE_RATE`-th log is recorded into the histogram.
/// The call stack itself is updated on every log (skipping would corrupt it).
///
/// Output is written to `/tmp/recursion_folded_sampled.txt` in
/// inferno-flamegraph "folded stacks" format. Pipe it through:
///
///     cat /tmp/recursion_folded_sampled.txt | inferno-flamegraph > svg.svg
///
/// Expect ~10-20 minutes for SAMPLE_RATE=100 on a 40B-cycle guest.
#[test]
#[ignore = "diagnostic: sampled flamegraph for the verifier-in-VM"]
fn test_recursion_sampled_flamegraph() {
    use executor::flamegraph::FlamegraphGenerator;
    use std::io::BufWriter;

    /// 1 in N logs is fed to `process_logs`, which both updates the call
    /// stack and records a sample. At 1, every cycle goes through — the call
    /// stack stays exactly in sync with execution so frame widths are
    /// trustworthy, but the per-cycle cost (~57µs) limits how many cycles
    /// we can cover within a wall-clock budget.
    ///
    /// At SAMPLE_RATE > 1, every CALL/RETURN that lands on a skipped cycle
    /// silently desyncs the stack, producing the "stuck-in-visit_seq" effect
    /// we saw at 1:1000. Use values > 1 only when stack accuracy is
    /// expendable.
    const SAMPLE_RATE: usize = 1;

    /// Stop the executor early once we've covered this many cycles.
    /// Set to 0 to run to completion (40B+ cycles, hours at SAMPLE_RATE=1).
    /// At SAMPLE_RATE=1, ~57µs per cycle means 5M cycles ≈ 5 min wall time.
    const CYCLE_BUDGET: u64 = 5_000_000;

    let (recursion_elf_bytes, program, mut executor) =
        setup_guest_run("sampled-fg", "recursion", &MIN_PROOF_OPTIONS);

    eprintln!("[sampled-fg] executing recursion guest (sampling 1-in-{SAMPLE_RATE}) ...",);
    let symbols = executor::elf::SymbolTable::parse(&recursion_elf_bytes);
    let entry_point = program.entry_point;

    // Build our own instruction cache from the same segments `Executor::new`
    // decodes internally. Owning it (rather than reading `executor.instructions`
    // mid-loop) is what lets the per-log closure call `process_logs` without
    // borrowing `executor`, which `drive_executor` holds mutably for `resume()`.
    let instructions = executor::vm::execution::InstructionCache::new(&program.data)
        .expect("instruction cache build failed");

    // RefCell so the per-log closure (`process_logs`, &mut self) and the
    // progress closure (`write_folded`, &self) can both reach the generator —
    // their calls never overlap, so the runtime borrow check never trips.
    let generator = std::cell::RefCell::new(FlamegraphGenerator::new(symbols, entry_point));

    // Path is defined here (not after the loop) so the periodic checkpoint
    // writes below can target it. The final write at the end still happens.
    let path = "/tmp/recursion_folded_sampled.txt";

    let mut i = 0usize;
    let (total_cycles, exec_time) = drive_executor(
        &mut executor,
        |log| {
            // 1-in-SAMPLE_RATE logs are fed to `process_logs`. At SAMPLE_RATE==1
            // this is the identity filter (`_ % 1 == 0`); the `#[allow]` keeps
            // the general form so SAMPLE_RATE can be bumped without touching the
            // body. Skipped logs lose stack accuracy — acceptable diagnostic
            // quality at higher rates.
            #[allow(clippy::modulo_one)]
            let take = i % SAMPLE_RATE == 0;
            if take {
                generator
                    .borrow_mut()
                    .process_logs(std::slice::from_ref(log), &instructions)
                    .expect("flamegraph process_logs");
            }
            i += 1;

            // Early exit once we've covered the cycle budget. The dominant hot
            // kernels are ~uniform across the verifier's runtime, so a partial
            // run still surfaces them. `#[allow]` lets CYCLE_BUDGET be const-0
            // (full run) without tripping clippy.
            #[allow(clippy::absurd_extreme_comparisons)]
            if CYCLE_BUDGET > 0 && i as u64 >= CYCLE_BUDGET {
                eprintln!("[sampled-fg] hit cycle budget ({CYCLE_BUDGET} cycles), stopping early");
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        },
        |chunks, cycles, elapsed| {
            if chunks.is_multiple_of(500) {
                eprintln!(
                    "[sampled-fg]   ... {chunks} chunks, {cycles} cycles, {elapsed:?} elapsed"
                );
                // Checkpoint: re-write the folded file in place so a killed run
                // still leaves a usable (if partial) flamegraph on disk.
                let file = std::fs::File::create(path).expect("create output file");
                let mut writer = BufWriter::new(file);
                generator
                    .borrow()
                    .write_folded(&mut writer)
                    .expect("write folded output");
            }
        },
    );

    let file = std::fs::File::create(path).expect("create output file");
    let mut writer = BufWriter::new(file);
    generator
        .borrow()
        .write_folded(&mut writer)
        .expect("write folded output");

    eprintln!();
    eprintln!("============================================================");
    eprintln!("  SAMPLED FLAMEGRAPH SUMMARY");
    eprintln!("============================================================");
    eprintln!("  Total cycles : {total_cycles}");
    eprintln!("  Sample rate  : 1 in {SAMPLE_RATE}");
    eprintln!("  Exec time    : {exec_time:?}");
    eprintln!("  Output file  : {path}");
    eprintln!("============================================================");
    eprintln!();
    eprintln!("  To render SVG (requires inferno):");
    eprintln!("    cat {path} | inferno-flamegraph > /tmp/recursion_flamegraph_sampled.svg");
    eprintln!("============================================================");
}

/// Diagnostic: host-side per-step timings for the verifier.
///
/// Runs an inner prove (empty guest, blowup=2, 1 query) and then verifies it
/// on the host. When built with `--features stark/instruments`, the verifier
/// prints `Time spent: ...` for each of the four steps (replay challenges,
/// composition polynomial, FRI, DEEP openings) plus the step-1-replay it
/// does before step 2. Lets us see the host-side split in seconds, without
/// running anything inside the VM.
///
/// Usage:
/// ```
/// cargo test --release -p lambda-vm-prover --features stark/instruments \
///   --lib test_host_verify_step_timings -- --ignored --nocapture
/// ```
#[test]
#[ignore = "diagnostic: prints host-side verifier step timings"]
fn test_host_verify_step_timings() {
    let root = workspace_root();
    let empty_path =
        root.join("bench_vs/lambda/empty/target/riscv64im-lambda-vm-elf/release/empty-bench");
    let empty_elf_bytes = std::fs::read(&empty_path).expect("read empty-bench");

    let inner_proof_options = MIN_PROOF_OPTIONS;

    eprintln!("[host-verify] proving empty (blowup=2, fri_queries=1) ...");
    let inner_proof = crate::prove_with_options_and_inputs(
        &empty_elf_bytes,
        &[],
        &inner_proof_options,
        &crate::MaxRowsConfig::default(),
    )
    .expect("inner prove should succeed");

    eprintln!("[host-verify] verifying on host (with instruments) ...");
    let ok = crate::verify_with_options(
        &inner_proof,
        &empty_elf_bytes,
        &inner_proof_options,
        None,
        None,
    )
    .expect("verify errored");
    assert!(ok, "proof must verify");
    eprintln!("[host-verify] verified OK");
}

// Control guest: decodes the blob and halts. Its cycle count subtracted from
// the matching recursion run isolates the in-VM verifier cost.

#[test]
#[ignore = "diagnostic: fast; deserialize-only guest cycle count (1 query)"]
fn test_deserialize_only_cycles_1query() {
    run_profile("deserialize-only", 50, MIN_PROOF_OPTIONS, false);
}

#[test]
#[ignore = "diagnostic: fast; deserialize-only guest cycle count (multi-query)"]
fn test_deserialize_only_cycles_multiquery() {
    run_profile("deserialize-only", 50, blowup8(), false);
}

#[test]
#[ignore = "diagnostic: ~1 min; deserialize-only guest histogram (1 query)"]
fn test_deserialize_only_profile_1query() {
    run_profile("deserialize-only", 50, MIN_PROOF_OPTIONS, true);
}

#[test]
#[ignore = "diagnostic: deserialize-only guest histogram (multi-query)"]
fn test_deserialize_only_profile_multiquery() {
    run_profile("deserialize-only", 50, blowup8(), true);
}

/// Inner program: fibonacci(10).
#[test]
#[ignore = "slow: memory-bounded continuation prove of the verifier-in-VM"]
fn test_recursion_prove() {
    let root = workspace_root();
    let fib_elf_bytes = read_guest_elf(&root, "fibonacci");

    let n: u64 = 10;
    let inner_private_input = n.to_le_bytes().to_vec();

    run_recursion_pipeline(
        "recursion-prove-fib",
        &fib_elf_bytes,
        &inner_private_input,
        OuterMode::Prove,
    );
}
