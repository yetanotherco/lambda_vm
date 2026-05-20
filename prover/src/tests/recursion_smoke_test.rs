//! End-to-end naive recursion pipeline smoke tests.
//!
//! Each test:
//! 1. Proves an inner program on the host.
//! 2. Serializes `(VmProof, inner_elf)` with postcard.
//! 3. Hands that as private input to the recursion guest.
//! 4. Proves the recursion guest's execution.
//! 5. Verifies the outer proof.
//!
//! The ELFs are built on demand by `bench_vs/build_recursion_elfs.sh`.
//!
//! Tests are `#[ignore]`d because the outer proof runs the full STARK verifier
//! inside the VM (minutes per run, large memory footprint).

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn build_elfs(root: &std::path::Path) {
    let status = Command::new("bash")
        .arg(root.join("bench_vs/build_recursion_elfs.sh"))
        .status()
        .expect("failed to spawn build helper");
    assert!(status.success(), "ELF build script failed");
}

/// Read a guest ELF artifact from a bench_vs/lambda/<name>/ build.
fn read_guest_elf(root: &std::path::Path, name: &str, bin_name: &str) -> Vec<u8> {
    let path = root.join(format!(
        "bench_vs/lambda/{name}/target/riscv64im-lambda-vm-elf/release/{bin_name}"
    ));
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Core pipeline: prove an inner program with the given options, hand the
/// proof+ELF+options to the recursion guest, then prove and verify the outer
/// proof.
fn run_recursion_pipeline_with_options(
    label: &str,
    inner_elf_bytes: &[u8],
    inner_private_input: &[u8],
    inner_proof_options: stark::proof::options::ProofOptions,
) {
    let root = workspace_root();
    build_elfs(&root);
    let recursion_elf_bytes = read_guest_elf(&root, "recursion", "recursion-bench");

    eprintln!(
        "[{label}] proving inner (blowup={}, fri_queries={}) ...",
        inner_proof_options.blowup_factor, inner_proof_options.fri_number_of_queries
    );
    let inner_proof = crate::prove_with_options_and_inputs(
        inner_elf_bytes,
        inner_private_input,
        &inner_proof_options,
        &crate::MaxRowsConfig::default(),
    )
    .expect("inner prove should succeed");
    eprintln!("[{label}] inner proof generated");

    assert!(
        crate::verify_with_options(&inner_proof, inner_elf_bytes, &inner_proof_options)
            .expect("inner verify errored"),
        "inner proof must verify on host"
    );

    let elf_for_vkey = executor::elf::Elf::load(inner_elf_bytes).expect("ELF load failed");
    let vkey = crate::VmVerifyingKey::from_elf_and_options(&elf_for_vkey, &inner_proof_options);
    let blob =
        postcard::to_allocvec(&(&inner_proof, &inner_elf_bytes, &inner_proof_options, &vkey))
            .expect("postcard encode failed");
    eprintln!(
        "[{label}] postcard blob: {} bytes (limit: MAX_PRIVATE_INPUT_SIZE)",
        blob.len()
    );
    assert!(
        blob.len() < executor::constants::MAX_PRIVATE_INPUT_SIZE as usize,
        "recursion input exceeds MAX_PRIVATE_INPUT_SIZE"
    );

    eprintln!("[{label}] proving outer (recursion guest) ...");
    let outer_proof =
        crate::prove_with_inputs(&recursion_elf_bytes, &blob).expect("outer prove should succeed");
    eprintln!("[{label}] outer proof generated");

    assert!(
        crate::verify(&outer_proof, &recursion_elf_bytes).expect("outer verify errored"),
        "outer proof must verify on host"
    );

    assert_eq!(
        outer_proof.public_output,
        vec![1u8],
        "guest should commit success marker"
    );
}

/// Convenience wrapper using `blowup=8` for the inner proof — the default for
/// the existing smoke tests, chosen to keep outer-prove memory tractable.
fn run_recursion_pipeline(label: &str, inner_elf_bytes: &[u8], inner_private_input: &[u8]) {
    let inner_proof_options = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(8)
        .expect("blowup=8 is always valid");
    run_recursion_pipeline_with_options(
        label,
        inner_elf_bytes,
        inner_private_input,
        inner_proof_options,
    );
}

/// Inner program: empty (halt immediately). Useful for measuring the
/// lambda-vm verifier's intrinsic recursion overhead — i.e. what it costs
/// to verify the smallest possible lambda-vm proof, with no inner workload.
#[test]
#[ignore = "slow: runs the full STARK verifier inside the VM"]
fn test_recursion_smoke_empty() {
    let root = workspace_root();
    build_elfs(&root);
    let empty_elf_bytes = read_guest_elf(&root, "empty", "empty-bench");
    run_recursion_pipeline("recursion-empty", &empty_elf_bytes, &[]);
}

/// Inner program: empty, but with the absolute-minimum FRI parameters
/// (blowup=2, **fri_number_of_queries=1**). This is a "can the pipeline even
/// run end-to-end on a 125 GB box" experiment — security is intentionally
/// terrible. Use only for capacity probing.
#[test]
#[ignore = "slow: runs the full STARK verifier inside the VM"]
fn test_recursion_smoke_1query() {
    let root = workspace_root();
    build_elfs(&root);
    let empty_elf_bytes = read_guest_elf(&root, "empty", "empty-bench");

    // Construct ProofOptions directly so we can pin fri_number_of_queries = 1.
    // (GoldilocksCubicProofOptions::with_blowup derives queries from a 128-bit
    // security target — way more than we want here.)
    let inner_proof_options = stark::proof::options::ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 1,
        coset_offset: 3,
        grinding_factor: 1,
    };

    run_recursion_pipeline_with_options(
        "recursion-1query",
        &empty_elf_bytes,
        &[],
        inner_proof_options,
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
    build_elfs(&root);
    let empty_elf_bytes = read_guest_elf(&root, "empty", "empty-bench");

    let inner_proof_options = stark::proof::options::ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 1,
        coset_offset: 3,
        grinding_factor: 1,
    };

    eprintln!("[dump-input] proving inner ...");
    let inner_proof = crate::prove_with_options_and_inputs(
        &empty_elf_bytes,
        &[],
        &inner_proof_options,
        &crate::MaxRowsConfig::default(),
    )
    .expect("inner prove should succeed");

    let elf_for_vkey = executor::elf::Elf::load(&empty_elf_bytes).expect("ELF load failed");
    let vkey = crate::VmVerifyingKey::from_elf_and_options(&elf_for_vkey, &inner_proof_options);
    let blob =
        postcard::to_allocvec(&(&inner_proof, &empty_elf_bytes, &inner_proof_options, &vkey))
            .expect("postcard encode failed");

    let path = "/tmp/recursion_input.bin";
    std::fs::write(path, &blob).expect("write blob");
    eprintln!("[dump-input] wrote {} bytes to {path}", blob.len());
}

/// Diagnostic: build the inner proof + recursion guest input, then **execute
/// only** the recursion guest (no STARK proving) and report cycle counts +
/// trace size estimates.
///
/// This is the cheap way to find out how many RISC-V instructions the
/// verifier actually executes inside the guest — a much faster signal than
/// running the full outer prove (which can OOM on a 125 GB machine).
#[test]
#[ignore = "diagnostic: runs the executor only, prints cycle counts"]
fn test_recursion_cycle_count() {
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    let root = workspace_root();
    build_elfs(&root);
    let empty_elf_bytes = read_guest_elf(&root, "empty", "empty-bench");
    let recursion_elf_bytes = read_guest_elf(&root, "recursion", "recursion-bench");

    // Build the inner proof exactly as the smoke test does, with the
    // absolute-minimum FRI params so the inner is as small as possible.
    let inner_proof_options = stark::proof::options::ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 1,
        coset_offset: 3,
        grinding_factor: 1,
    };

    eprintln!("[cycle-count] proving inner (empty, blowup=2, fri_queries=1) ...");
    let inner_proof = crate::prove_with_options_and_inputs(
        &empty_elf_bytes,
        &[],
        &inner_proof_options,
        &crate::MaxRowsConfig::default(),
    )
    .expect("inner prove should succeed");

    let elf_for_vkey = executor::elf::Elf::load(&empty_elf_bytes).expect("ELF load failed");
    let vkey = crate::VmVerifyingKey::from_elf_and_options(&elf_for_vkey, &inner_proof_options);
    let blob =
        postcard::to_allocvec(&(&inner_proof, &empty_elf_bytes, &inner_proof_options, &vkey))
            .expect("postcard encode failed");
    eprintln!("[cycle-count] postcard blob: {} bytes", blob.len());

    // Execute (NOT prove) the recursion guest. Use `resume()` in a loop and
    // only count chunk sizes — never accumulate logs in memory. This avoids
    // the Vec<Log> blow-up that OOMs even a 125 GB server (one Log is 40 B;
    // a few billion of them is hundreds of GB).
    eprintln!("[cycle-count] executing recursion guest (streaming counter only) ...");
    let program = Elf::load(&recursion_elf_bytes).expect("ELF load failed");
    let mut executor = Executor::new(&program, blob).expect("Executor::new failed");
    let start = std::time::Instant::now();
    let mut cycle_count: usize = 0;
    let mut chunks: usize = 0;
    while let Some(logs) = executor.resume().expect("executor resume failed") {
        cycle_count += logs.len();
        chunks += 1;
        if chunks.is_multiple_of(50) {
            eprintln!(
                "[cycle-count]   ... {chunks} chunks, {cycle_count} cycles, {:?} elapsed",
                start.elapsed()
            );
        }
    }
    let exec_time = start.elapsed();

    eprintln!();
    eprintln!("============================================================");
    eprintln!("  RECURSION GUEST EXECUTION SUMMARY");
    eprintln!("============================================================");
    eprintln!("  Cycle count           : {cycle_count}");
    eprintln!("  Executor wall time    : {exec_time:?}");
    eprintln!();
    eprintln!("  Rough memory estimate for outer prove:");
    let bytes_per_field = 8usize;
    let approx_columns = 250usize; // CPU + MEMW + DECODE + bus columns combined
    let main_trace_bytes = cycle_count * approx_columns * bytes_per_field;
    let blowup = 2usize;
    let lde_main_bytes = main_trace_bytes * blowup;
    eprintln!(
        "    main trace            : ~{:.2} GB ({} cycles × ~{} cols × 8 B)",
        main_trace_bytes as f64 / 1e9,
        cycle_count,
        approx_columns
    );
    eprintln!(
        "    main LDE (blowup={})   : ~{:.2} GB",
        blowup,
        lde_main_bytes as f64 / 1e9
    );
    eprintln!("  (aux trace adds roughly 50% more, so peak peak ≈ 2-3× LDE)");
    eprintln!("============================================================");
}

/// Diagnostic: build a PC histogram of the recursion guest's execution.
///
/// Streams chunks of logs via `Executor::resume()` so the in-memory state
/// stays bounded to the histogram itself (~MB for ~hundreds of thousands of
/// unique PCs). Prints the top 100 PCs by cycle count, plus cumulative %.
/// Pipe the output through `addr2line` to map PCs to source functions.
#[test]
#[ignore = "diagnostic: ~8 minutes; prints PC histogram of the verifier-in-VM"]
fn test_recursion_pc_histogram() {
    use executor::elf::Elf;
    use executor::vm::execution::Executor;
    use std::collections::HashMap;

    let root = workspace_root();
    build_elfs(&root);
    let empty_elf_bytes = read_guest_elf(&root, "empty", "empty-bench");
    let recursion_elf_bytes = read_guest_elf(&root, "recursion", "recursion-bench");

    let inner_proof_options = stark::proof::options::ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 1,
        coset_offset: 3,
        grinding_factor: 1,
    };

    eprintln!("[pc-hist] proving inner (empty, blowup=2, fri_queries=1) ...");
    let inner_proof = crate::prove_with_options_and_inputs(
        &empty_elf_bytes,
        &[],
        &inner_proof_options,
        &crate::MaxRowsConfig::default(),
    )
    .expect("inner prove should succeed");

    let elf_for_vkey = executor::elf::Elf::load(&empty_elf_bytes).expect("ELF load failed");
    let vkey = crate::VmVerifyingKey::from_elf_and_options(&elf_for_vkey, &inner_proof_options);
    let blob =
        postcard::to_allocvec(&(&inner_proof, &empty_elf_bytes, &inner_proof_options, &vkey))
            .expect("postcard encode failed");
    eprintln!("[pc-hist] postcard blob: {} bytes", blob.len());

    eprintln!("[pc-hist] executing recursion guest (building PC histogram) ...");
    let program = Elf::load(&recursion_elf_bytes).expect("ELF load failed");
    let mut executor = Executor::new(&program, blob).expect("Executor::new failed");

    let start = std::time::Instant::now();
    let mut pc_hist: HashMap<u64, u64> = HashMap::with_capacity(300_000);
    let mut total_cycles: u64 = 0;
    let mut chunks: usize = 0;
    while let Some(logs) = executor.resume().expect("executor resume failed") {
        for log in logs {
            *pc_hist.entry(log.current_pc).or_insert(0) += 1;
        }
        total_cycles += logs.len() as u64;
        chunks += 1;
        if chunks.is_multiple_of(500) {
            eprintln!(
                "[pc-hist]   ... {chunks} chunks, {total_cycles} cycles, {} unique PCs, {:?}",
                pc_hist.len(),
                start.elapsed()
            );
        }
    }
    let exec_time = start.elapsed();

    let mut entries: Vec<(u64, u64)> = pc_hist.into_iter().collect();
    entries.sort_unstable_by_key(|(_pc, count)| std::cmp::Reverse(*count));

    eprintln!();
    eprintln!("============================================================");
    eprintln!("  RECURSION GUEST PC HISTOGRAM");
    eprintln!("============================================================");
    eprintln!("  Total cycles : {total_cycles}");
    eprintln!("  Unique PCs   : {}", entries.len());
    eprintln!("  Exec time    : {exec_time:?}");
    eprintln!();
    eprintln!("  Top 100 PCs by cycle count:");
    eprintln!(
        "  {:>4}  {:>18}  {:>14}  {:>7}  {:>7}",
        "rank", "pc", "cycles", "%", "cum %"
    );
    let mut cumulative: u64 = 0;
    for (rank, (pc, count)) in entries.iter().take(100).enumerate() {
        cumulative += count;
        let pct = 100.0 * (*count as f64) / (total_cycles as f64);
        let cum_pct = 100.0 * (cumulative as f64) / (total_cycles as f64);
        eprintln!(
            "  {:>4}  {:#018x}  {:>14}  {:>6.2}%  {:>6.2}%",
            rank + 1,
            pc,
            count,
            pct,
            cum_pct
        );
    }
    eprintln!("============================================================");
    eprintln!();
    eprintln!("  To map PCs to source functions, on the same machine that has");
    eprintln!("  the recursion ELF (and ideally a debug build for line info):");
    eprintln!("    addr2line -e <recursion-bench-elf> -f -i -C 0x<pc>");
    eprintln!("  Or for symbol-range lookup without DWARF:");
    eprintln!("    nm --print-size <recursion-bench-elf> | rustfilt | sort");
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
    use executor::elf::Elf;
    use executor::flamegraph::FlamegraphGenerator;
    use executor::vm::execution::Executor;
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

    let root = workspace_root();
    build_elfs(&root);
    let empty_elf_bytes = read_guest_elf(&root, "empty", "empty-bench");
    let recursion_elf_bytes = read_guest_elf(&root, "recursion", "recursion-bench");

    let inner_proof_options = stark::proof::options::ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 1,
        coset_offset: 3,
        grinding_factor: 1,
    };

    eprintln!("[sampled-fg] proving inner (empty, blowup=2, fri_queries=1) ...");
    let inner_proof = crate::prove_with_options_and_inputs(
        &empty_elf_bytes,
        &[],
        &inner_proof_options,
        &crate::MaxRowsConfig::default(),
    )
    .expect("inner prove should succeed");

    let elf_for_vkey = executor::elf::Elf::load(&empty_elf_bytes).expect("ELF load failed");
    let vkey = crate::VmVerifyingKey::from_elf_and_options(&elf_for_vkey, &inner_proof_options);
    let blob =
        postcard::to_allocvec(&(&inner_proof, &empty_elf_bytes, &inner_proof_options, &vkey))
            .expect("postcard encode failed");
    eprintln!("[sampled-fg] postcard blob: {} bytes", blob.len());

    eprintln!("[sampled-fg] executing recursion guest (sampling 1-in-{SAMPLE_RATE}) ...",);
    let program = Elf::load(&recursion_elf_bytes).expect("ELF load failed");
    let symbols = executor::elf::SymbolTable::parse(&recursion_elf_bytes);
    let entry_point = program.entry_point;
    let mut executor = Executor::new(&program, blob).expect("Executor::new failed");

    let mut generator = FlamegraphGenerator::new(symbols, entry_point);

    // Path is defined here (not after the loop) so the periodic checkpoint
    // writes below can target it. The final write at the end still happens.
    let path = "/tmp/recursion_folded_sampled.txt";

    let start = std::time::Instant::now();
    let mut total_cycles: u64 = 0;
    let mut chunks: usize = 0;
    while let Some(logs) = executor.resume().expect("executor resume failed") {
        // Pull the chunk into an owned Vec so we can use it after dropping the
        // immutable borrow of `executor`.
        let (sampled, chunk_len) = {
            let len = logs.len();
            // When SAMPLE_RATE == 1, this is the identity filter — `_ % 1 == 0`
            // is trivially true. clippy::modulo_one is fired so we suppress it
            // here; the generality of the filter is the point (lets us flip
            // SAMPLE_RATE without touching the loop body).
            #[allow(clippy::modulo_one)]
            let sampled: Vec<_> = logs
                .iter()
                .enumerate()
                .filter(|(i, _)| i % SAMPLE_RATE == 0)
                .map(|(_, log)| log.clone())
                .collect();
            (sampled, len)
        };

        // Now we can re-borrow executor.instructions immutably for the
        // flamegraph generator. We build the sampled subset of logs (every Nth)
        // and call process_logs on it. THIS LOSES STACK ACCURACY for skipped
        // logs but is fast — acceptable for diagnostic-quality data at this
        // sample rate.
        generator
            .process_logs(&sampled, &executor.instructions)
            .expect("flamegraph process_logs");

        total_cycles += chunk_len as u64;
        chunks += 1;
        if chunks.is_multiple_of(500) {
            eprintln!(
                "[sampled-fg]   ... {chunks} chunks, {total_cycles} cycles, {:?} elapsed",
                start.elapsed()
            );
            // Checkpoint: re-write the folded file in place so a killed run
            // still leaves a usable (if partial) flamegraph on disk.
            let file = std::fs::File::create(path).expect("create output file");
            let mut writer = BufWriter::new(file);
            generator
                .write_folded(&mut writer)
                .expect("write folded output");
        }

        // Early exit once we've covered the cycle budget. The flamegraph will
        // reflect only the cycles we processed, but the dominant hot kernels
        // are typically uniformly distributed across the verifier's runtime so
        // a partial run still surfaces them clearly. Wrapped in #[allow] so
        // CYCLE_BUDGET can be const-0 (full run) without tripping clippy.
        #[allow(clippy::absurd_extreme_comparisons)]
        if CYCLE_BUDGET > 0 && total_cycles >= CYCLE_BUDGET {
            eprintln!("[sampled-fg] hit cycle budget ({CYCLE_BUDGET} cycles), stopping early");
            break;
        }
    }
    let exec_time = start.elapsed();

    let file = std::fs::File::create(path).expect("create output file");
    let mut writer = BufWriter::new(file);
    generator
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
    if !empty_path.exists() {
        build_elfs(&root);
    }
    let empty_elf_bytes = std::fs::read(&empty_path).expect("read empty-bench");

    let inner_proof_options = stark::proof::options::ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 1,
        coset_offset: 3,
        grinding_factor: 1,
    };

    eprintln!("[host-verify] proving empty (blowup=2, fri_queries=1) ...");
    let inner_proof = crate::prove_with_options_and_inputs(
        &empty_elf_bytes,
        &[],
        &inner_proof_options,
        &crate::MaxRowsConfig::default(),
    )
    .expect("inner prove should succeed");

    eprintln!("[host-verify] verifying on host (with instruments) ...");
    let ok = crate::verify_with_options(&inner_proof, &empty_elf_bytes, &inner_proof_options)
        .expect("verify errored");
    assert!(ok, "proof must verify");
    eprintln!("[host-verify] verified OK");
}

/// Diagnostic: cycle count for the **deserialize-only** counterpart of the
/// recursion guest. Same input layout (`(VmProof, Vec<u8>, ProofOptions)`)
/// and same proof, but the guest just postcard-decodes the blob and halts —
/// it never calls `verify_with_options`.
///
/// The cycle delta between this and `test_recursion_cycle_count` is the
/// actual cost of the STARK verifier inside the VM. The flamegraph
/// suggested postcard decode was ~93% of the recursion guest's cycles; this
/// test pins down that number directly.
#[test]
#[ignore = "diagnostic: runs the deserialize-only guest, prints cycle count"]
fn test_deserialize_only_cycle_count() {
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    let root = workspace_root();
    build_elfs(&root);
    let empty_elf_bytes = read_guest_elf(&root, "empty", "empty-bench");
    let deser_elf_bytes = read_guest_elf(&root, "deserialize-only", "deserialize-only-bench");

    let inner_proof_options = stark::proof::options::ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 1,
        coset_offset: 3,
        grinding_factor: 1,
    };

    eprintln!("[deser-only] proving inner (empty, blowup=2, fri_queries=1) ...");
    let inner_proof = crate::prove_with_options_and_inputs(
        &empty_elf_bytes,
        &[],
        &inner_proof_options,
        &crate::MaxRowsConfig::default(),
    )
    .expect("inner prove should succeed");

    let elf_for_vkey = executor::elf::Elf::load(&empty_elf_bytes).expect("ELF load failed");
    let vkey = crate::VmVerifyingKey::from_elf_and_options(&elf_for_vkey, &inner_proof_options);
    let blob =
        postcard::to_allocvec(&(&inner_proof, &empty_elf_bytes, &inner_proof_options, &vkey))
            .expect("postcard encode failed");
    eprintln!("[deser-only] postcard blob: {} bytes", blob.len());

    eprintln!("[deser-only] executing deserialize-only guest (streaming) ...");
    let program = Elf::load(&deser_elf_bytes).expect("ELF load failed");
    eprintln!(
        "[deser-only] ELF: {} bytes, entry_point=0x{:x}",
        deser_elf_bytes.len(),
        program.entry_point,
    );
    assert_ne!(
        program.entry_point, 0,
        "deserialize-only ELF has entry_point=0 — build artifact is malformed"
    );
    let mut executor = Executor::new(&program, blob).expect("Executor::new failed");

    let start = std::time::Instant::now();
    let mut cycle_count: usize = 0;
    let mut chunks: usize = 0;
    while let Some(logs) = executor.resume().expect("executor resume failed") {
        cycle_count += logs.len();
        chunks += 1;
        if chunks.is_multiple_of(50) {
            eprintln!(
                "[deser-only]   ... {chunks} chunks, {cycle_count} cycles, {:?} elapsed",
                start.elapsed()
            );
        }
    }
    let exec_time = start.elapsed();

    eprintln!();
    eprintln!("============================================================");
    eprintln!("  DESERIALIZE-ONLY GUEST EXECUTION SUMMARY");
    eprintln!("============================================================");
    eprintln!("  Cycle count           : {cycle_count}");
    eprintln!("  Executor wall time    : {exec_time:?}");
    eprintln!();
    eprintln!("  Compare against test_recursion_cycle_count (~40.5B cycles");
    eprintln!("  with the same proof). Delta = verifier-in-VM cost.");
    eprintln!("============================================================");
}

/// Diagnostic: bucket the recursion guest's cycles by which verifier step
/// is currently executing.
///
/// The verifier's hot path is `verify_rounds_2_to_4`, which calls four
/// sub-routines in a fixed order:
///   1. `replay_rounds_after_round_1`               (recover challenges)
///   2. `step_2_verify_claimed_composition_polynomial`
///   3. `step_3_verify_fri`
///   4. `step_4_verify_trace_and_composition_openings`
///
/// We resolve each sub-routine's entry PC from the recursion ELF's symbol
/// table, then run a monotonic state machine over the execution stream:
/// the active bucket only advances 0 → 1 → 2 → 3 → 4 (never backwards),
/// so cycles inside a step's callees stay attributed to that step.
///
/// Bucket 0 ("setup") captures everything before step 1 is entered — the
/// allocator init, postcard decode, and `VmAirs::new` (which contains the
/// expensive preprocessed-commitment FFTs).
///
/// Streams chunks via `Executor::resume()` so memory stays bounded.
#[test]
#[ignore = "diagnostic: ~13 min; buckets the 40B cycles by verifier step"]
fn test_recursion_step_breakdown() {
    use executor::elf::{Elf, SymbolTable};
    use executor::vm::execution::Executor;

    let root = workspace_root();
    build_elfs(&root);
    let empty_elf_bytes = read_guest_elf(&root, "empty", "empty-bench");
    let recursion_elf_bytes = read_guest_elf(&root, "recursion", "recursion-bench");

    let inner_proof_options = stark::proof::options::ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 1,
        coset_offset: 3,
        grinding_factor: 1,
    };

    eprintln!("[step-bkd] proving inner (empty, blowup=2, fri_queries=1) ...");
    let inner_proof = crate::prove_with_options_and_inputs(
        &empty_elf_bytes,
        &[],
        &inner_proof_options,
        &crate::MaxRowsConfig::default(),
    )
    .expect("inner prove should succeed");

    let elf_for_vkey = executor::elf::Elf::load(&empty_elf_bytes).expect("ELF load failed");
    let vkey = crate::VmVerifyingKey::from_elf_and_options(&elf_for_vkey, &inner_proof_options);
    let blob =
        postcard::to_allocvec(&(&inner_proof, &empty_elf_bytes, &inner_proof_options, &vkey))
            .expect("postcard encode failed");
    eprintln!("[step-bkd] postcard blob: {} bytes", blob.len());

    // Build a per-step "advance bucket to N" lookup. The verifier's step
    // functions get inlined by LLVM in release mode, so we can't rely on
    // matching their entry PCs directly. Instead we anchor on closures the
    // compiler emits *inside* each step's body — iterator combinators like
    // `.fold(|...|)` keep the step's method name as a substring in their
    // mangled symbol. Any PC that resolves to a symbol containing step N's
    // keyword advances the bucket to N (monotonically).
    //
    // If step N has no matching symbol at all (e.g. step 4 is fully inlined
    // with no closure children of its own), its cycles get attributed to the
    // previous bucket. We report that explicitly in the summary.
    let symbols = SymbolTable::parse(&recursion_elf_bytes);
    assert!(
        !symbols.is_empty(),
        "recursion ELF has no symbol table — was it stripped?"
    );

    let step_keywords = [
        "replay_rounds_after_round_1",
        "step_2_verify_claimed_composition_polynomial",
        "step_3_verify_fri",
        "step_4_verify_trace_and_composition_openings",
    ];
    let step_found: [bool; 4] = std::array::from_fn(|i| {
        symbols
            .functions()
            .iter()
            .any(|f| f.name.contains(step_keywords[i]))
    });
    for (i, found) in step_found.iter().enumerate() {
        let n_matches = symbols
            .functions()
            .iter()
            .filter(|f| f.name.contains(step_keywords[i]))
            .count();
        eprintln!(
            "[step-bkd] step {}: keyword={:?} -> {} symbol(s) {}",
            i + 1,
            step_keywords[i],
            n_matches,
            if *found {
                ""
            } else {
                "(fully inlined; will merge into the previous bucket)"
            }
        );
    }

    // Monotonic state machine: 0=setup, 1..=4=inside step N (or its callees /
    // inlined-step-N-cycles attributed here because step N+1 is missing).
    let mut bucket: u8 = 0;
    let mut buckets = [0u64; 5];

    eprintln!("[step-bkd] executing recursion guest (streaming) ...");
    let program = Elf::load(&recursion_elf_bytes).expect("ELF load failed");
    let mut executor = Executor::new(&program, blob).expect("Executor::new failed");

    // Cache the last symbol-table hit so we only do a binary search on
    // function transitions, not on every cycle. Functions are typically
    // long-running (>>1 instruction), so this cache hits ~all of the time.
    let mut last_range: Option<(u64, u64)> = None;
    let mut last_advance: u8 = 0;

    let start = std::time::Instant::now();
    let mut total_cycles: u64 = 0;
    let mut chunks: usize = 0;
    while let Some(logs) = executor.resume().expect("executor resume failed") {
        for log in logs {
            let pc = log.current_pc;
            let in_cached = matches!(last_range, Some((s, e)) if pc >= s && pc < e);
            if !in_cached {
                // Slow path: refresh the cache from the symbol table.
                if let Some(sym) = symbols.lookup(pc) {
                    // SymbolTable accepts size=0 symbols as "any address >="; for
                    // those we'd need the next symbol's start for a real upper
                    // bound. Cheapest workaround: set a tiny range so we re-resolve
                    // soon enough that wrong attribution is bounded.
                    let end = sym.address + sym.size.max(1);
                    last_range = Some((sym.address, end));
                    last_advance = 0;
                    for (i, kw) in step_keywords.iter().enumerate() {
                        if sym.name.contains(kw) {
                            last_advance = (i + 1) as u8;
                        }
                    }
                } else {
                    last_range = None;
                    last_advance = 0;
                }
            }
            if bucket < last_advance {
                bucket = last_advance;
            }
            buckets[bucket as usize] += 1;
        }
        total_cycles += logs.len() as u64;
        chunks += 1;
        if chunks.is_multiple_of(500) {
            eprintln!(
                "[step-bkd]   ... {chunks} chunks, {total_cycles} cycles, bucket={bucket}, {:?}",
                start.elapsed()
            );
        }
    }
    let exec_time = start.elapsed();

    let labels = [
        "0. setup (alloc + postcard decode + VmAirs::new + pre-step-1)",
        "1. step 1: replay_rounds_after_round_1",
        "2. step 2: verify_claimed_composition_polynomial",
        "3. step 3: verify_fri",
        "4. step 4: verify_trace_and_composition_openings (+ wrap-up)",
    ];

    eprintln!();
    eprintln!("============================================================");
    eprintln!("  RECURSION GUEST PER-STEP CYCLE BREAKDOWN");
    eprintln!("============================================================");
    eprintln!("  Total cycles : {total_cycles}");
    eprintln!("  Exec time    : {exec_time:?}");
    eprintln!();
    eprintln!("  {:<60}  {:>14}  {:>7}", "bucket", "cycles", "%");
    for (label, cycles) in labels.iter().zip(buckets.iter()) {
        let pct = if total_cycles > 0 {
            100.0 * (*cycles as f64) / (total_cycles as f64)
        } else {
            0.0
        };
        eprintln!("  {:<60}  {:>14}  {:>6.2}%", label, cycles, pct);
    }
    eprintln!("============================================================");
}

/// Inner program: fibonacci(10).
#[test]
#[ignore = "slow: runs the full STARK verifier inside the VM"]
fn test_recursion_smoke() {
    let root = workspace_root();
    build_elfs(&root);
    let fib_elf_bytes = read_guest_elf(&root, "fibonacci", "fibonacci-bench");

    let n: u64 = 10;
    let inner_private_input = n.to_le_bytes().to_vec();

    run_recursion_pipeline("recursion-smoke", &fib_elf_bytes, &inner_private_input);
}
