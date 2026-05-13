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

    let blob = postcard::to_allocvec(&(&inner_proof, &inner_elf_bytes, &inner_proof_options))
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

    let blob = postcard::to_allocvec(&(&inner_proof, &empty_elf_bytes, &inner_proof_options))
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
    loop {
        match executor.resume().expect("executor resume failed") {
            Some(logs) => {
                cycle_count += logs.len();
                chunks += 1;
                if chunks.is_multiple_of(50) {
                    eprintln!(
                        "[cycle-count]   ... {chunks} chunks, {cycle_count} cycles, {:?} elapsed",
                        start.elapsed()
                    );
                }
            }
            None => break,
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

    let blob = postcard::to_allocvec(&(&inner_proof, &empty_elf_bytes, &inner_proof_options))
        .expect("postcard encode failed");
    eprintln!("[pc-hist] postcard blob: {} bytes", blob.len());

    eprintln!("[pc-hist] executing recursion guest (building PC histogram) ...");
    let program = Elf::load(&recursion_elf_bytes).expect("ELF load failed");
    let mut executor = Executor::new(&program, blob).expect("Executor::new failed");

    let start = std::time::Instant::now();
    let mut pc_hist: HashMap<u64, u64> = HashMap::with_capacity(300_000);
    let mut total_cycles: u64 = 0;
    let mut chunks: usize = 0;
    loop {
        match executor.resume().expect("executor resume failed") {
            Some(logs) => {
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
            None => break,
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
