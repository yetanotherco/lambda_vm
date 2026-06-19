//! Profile-compare: run the prover in CPU-only and CPU+CUDA modes, capture
//! structured JSON timings for each run, and emit a per-phase comparison
//! table with median + min + max across the runs.
//!
//! Discards run #1 of each mode as warmup (cold disk cache, allocator
//! warmup, PTX JIT compile for the cuda binary). Reports across runs 2..N.
//!
//! Usage (from repo root):
//!   cargo run --release --manifest-path tools/profile_compare/Cargo.toml
//!
//! All JSON snapshots are written to /tmp/profile_{mode}_{i}.json and kept
//! after the run — re-running overwrites them in place.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const RUNS: usize = 5;
const WARMUP_DISCARD: usize = 1;
const WORKLOAD: &str = "ethrex_simple_tx";
const TEST_NAME: &str = "profile_trace_expansion";
const TEST_FN: &str = "profile_ethrex_single_tx";

#[allow(dead_code)] // some fields kept for future report extensions
#[derive(Debug, Deserialize, Default, Clone)]
struct Snapshot {
    mode: String,
    execute_s: f64,
    trace_build_s: f64,
    air_construction_s: f64,
    total_s: f64,
    #[serde(default)]
    trace_build_tables: BTreeMap<String, f64>,
    #[serde(default)]
    pre_pass_s: f64,
    #[serde(default)]
    round1_total_s: f64,
    #[serde(default)]
    round1_main_lde_s: f64,
    #[serde(default)]
    round1_main_merkle_s: f64,
    #[serde(default)]
    round1_aux_lde_s: f64,
    #[serde(default)]
    round1_aux_merkle_s: f64,
    #[serde(default)]
    rounds_2_4_s: f64,
    #[serde(default)]
    table_timings: Vec<TableTiming>,
}

#[allow(dead_code)] // sub-op fields are parsed for future report extensions
#[derive(Debug, Deserialize, Default, Clone)]
struct TableTiming {
    name: String,
    #[serde(default)]
    rows: u64,
    total_s: f64,
    #[serde(default)]
    constraints_s: f64,
    #[serde(default)]
    comp_decompose_s: f64,
    #[serde(default)]
    comp_commit_s: f64,
    #[serde(default)]
    ood_s: f64,
    #[serde(default)]
    deep_comp_s: f64,
    #[serde(default)]
    deep_extend_s: f64,
    #[serde(default)]
    fri_commit_s: f64,
    #[serde(default)]
    queries_s: f64,
}

fn repo_root() -> PathBuf {
    // tools/profile_compare/Cargo.toml's parent's parent.
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // profile_compare
    p.pop(); // tools
    p
}

fn build_test_binary(repo: &Path, features: &str) -> Result<(), String> {
    eprintln!("==> cargo test --no-run --features {features} (release)");
    let status = Command::new("cargo")
        .arg("test")
        .arg("--release")
        .arg("-p")
        .arg("lambda-vm-prover")
        .arg("--features")
        .arg(features)
        .arg("--test")
        .arg(TEST_NAME)
        .arg("--no-run")
        .current_dir(repo)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| format!("cargo test --no-run failed to spawn: {e}"))?;
    if !status.success() {
        return Err(format!(
            "cargo test --no-run for features '{features}' exited with {status}"
        ));
    }
    Ok(())
}

fn run_one(repo: &Path, features: &str, json_path: &Path) -> Result<(), String> {
    eprintln!("    -> writing {}", json_path.display());
    let status = Command::new("cargo")
        .arg("test")
        .arg("--release")
        .arg("-p")
        .arg("lambda-vm-prover")
        .arg("--features")
        .arg(features)
        .arg("--test")
        .arg(TEST_NAME)
        .arg("--")
        .arg("--ignored")
        .arg("--exact")
        .arg(TEST_FN)
        .env("LAMBDA_VM_PROFILE_JSON", json_path)
        .current_dir(repo)
        .stdout(Stdio::null())
        .stderr(Stdio::null()) // keep run logs quiet; we read the JSON
        .status()
        .map_err(|e| format!("test run failed to spawn: {e}"))?;
    if !status.success() {
        return Err(format!("test run exited with {status}"));
    }
    Ok(())
}

fn load_snapshot(path: &Path) -> Result<Snapshot, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("reading {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
}

#[derive(Default)]
struct Stats {
    median: f64,
    min: f64,
    max: f64,
}

fn stats(values: &[f64]) -> Stats {
    if values.is_empty() {
        return Stats::default();
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let min = *sorted.first().unwrap();
    let max = *sorted.last().unwrap();
    Stats { median, min, max }
}

fn must_stay_cpu(label: &str) -> bool {
    // Phases that fundamentally stay on CPU regardless of any future port.
    matches!(label, "Execute" | "AIR construction" | "Round 4 queries")
}

fn already_on_gpu_when_cuda(label: &str) -> bool {
    // Phases dispatched to GPU today (PR-2 through PR-5). Trace build and
    // R2 constraint evaluation are NOT here — they stay CPU even with cuda
    // on. We can't break R2 into constraints vs LDE+Merkle from the current
    // snapshot schema, so the comparison reports `Rounds 2-4 total` and
    // attributes whatever delta exists to GPU dispatches.
    matches!(
        label,
        "Pre-pass (domains/twiddles)"
            | "Round 1 (main + aux LDE+Merkle)"
            | "Rounds 2-4 total"
    )
}

fn fmt_secs(v: f64) -> String {
    format!("{v:>7.2}s")
}

fn fmt_range(s: &Stats) -> String {
    format!("({:.2}-{:.2})", s.min, s.max)
}

fn label_status(label: &str) -> &'static str {
    if must_stay_cpu(label) {
        "CPU (stays)"
    } else if already_on_gpu_when_cuda(label) {
        "GPU when cuda"
    } else {
        "CPU (portable)"
    }
}

fn run_mode(
    repo: &Path,
    features: &str,
    mode_tag: &str,
) -> Result<Vec<Snapshot>, String> {
    build_test_binary(repo, features)?;
    let mut snapshots = Vec::with_capacity(RUNS);
    for i in 1..=RUNS {
        eprintln!("==> [{mode_tag}] run {i}/{RUNS}");
        let path = PathBuf::from(format!("/tmp/profile_{mode_tag}_{i}.json"));
        // Remove any stale snapshot from a previous run so a silent failure
        // is detected (load_snapshot will then fail loudly).
        let _ = std::fs::remove_file(&path);
        run_one(repo, features, &path)?;
        let snap = load_snapshot(&path)?;
        snapshots.push(snap);
    }
    Ok(snapshots)
}

fn aggregate_phase<F: Fn(&Snapshot) -> f64>(snaps: &[Snapshot], pick: F) -> Stats {
    let vals: Vec<f64> = snaps.iter().skip(WARMUP_DISCARD).map(pick).collect();
    stats(&vals)
}

fn main() {
    let repo = repo_root();
    eprintln!("profile_compare: workload={WORKLOAD} runs={RUNS} (run 1 discarded as warmup)");
    eprintln!("repo root: {}", repo.display());
    eprintln!();

    let cpu_snaps = match run_mode(&repo, "instruments", "cpu") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("CPU mode failed: {e}");
            std::process::exit(1);
        }
    };
    let gpu_snaps = match run_mode(&repo, "instruments,cuda", "cuda") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("CUDA mode failed: {e}");
            std::process::exit(1);
        }
    };

    type Picker = Box<dyn Fn(&Snapshot) -> f64>;
    let phases: Vec<(&'static str, Picker)> = vec![
        ("Execute", Box::new(|s| s.execute_s)),
        ("Trace build", Box::new(|s| s.trace_build_s)),
        ("AIR construction", Box::new(|s| s.air_construction_s)),
        ("Pre-pass (domains/twiddles)", Box::new(|s| s.pre_pass_s)),
        (
            "Round 1 (main + aux LDE+Merkle)",
            Box::new(|s| s.round1_total_s),
        ),
        ("Rounds 2-4 total", Box::new(|s| s.rounds_2_4_s)),
        ("TOTAL", Box::new(|s| s.total_s)),
    ];

    println!();
    println!("=== PROFILE COMPARISON ===");
    println!(
        "Workload: {WORKLOAD}    Runs: {RUNS} (first discarded, median over {} hot)",
        RUNS - WARMUP_DISCARD
    );
    println!();
    println!(
        "  {:<32} {:>10} {:>14}   {:>10} {:>14}   {:>10}   Status",
        "Phase", "CPU med", "CPU (min-max)", "GPU med", "GPU (min-max)", "Saved"
    );
    println!("  {}", "-".repeat(120));

    let mut total_must_stay = 0.0;
    let mut total_portable = 0.0;
    let mut total_gpu_already = 0.0;
    let mut total_cpu = 0.0;
    let mut total_gpu = 0.0;

    for (label, pick) in &phases {
        let cpu = aggregate_phase(&cpu_snaps, &**pick);
        let gpu = aggregate_phase(&gpu_snaps, &**pick);
        let saved = (cpu.median - gpu.median).max(0.0);
        let status = label_status(label);

        println!(
            "  {:<32} {:>10} {:>14}   {:>10} {:>14}   {:>10}   {}",
            label,
            fmt_secs(cpu.median),
            fmt_range(&cpu),
            fmt_secs(gpu.median),
            fmt_range(&gpu),
            fmt_secs(saved),
            status,
        );

        if *label == "TOTAL" {
            total_cpu = cpu.median;
            total_gpu = gpu.median;
            continue;
        }
        if must_stay_cpu(label) {
            total_must_stay += cpu.median;
        } else if already_on_gpu_when_cuda(label) {
            total_gpu_already += saved;
        } else {
            total_portable += cpu.median;
        }
    }

    println!("  {}", "-".repeat(120));
    println!();
    println!("Summary:");
    println!(
        "  CPU-only total wall:                {}",
        fmt_secs(total_cpu)
    );
    println!(
        "  CPU + GPU total wall:               {}        (speedup {:.2}x)",
        fmt_secs(total_gpu),
        if total_gpu > 0.0 { total_cpu / total_gpu } else { 0.0 }
    );
    println!(
        "  Time saved by GPU (already done):   {}",
        fmt_secs(total_gpu_already)
    );
    println!(
        "  Still on CPU but portable:          {}        <- upper bound of future GPU savings",
        fmt_secs(total_portable)
    );
    println!(
        "  Must stay CPU (Execute + AIR + Q):  {}",
        fmt_secs(total_must_stay)
    );
    println!();

    // Per-table trace-build breakdown (medians, sorted descending).
    println!("Trace build per-table breakdown (medians, CPU-only run — same on both modes today):");
    let mut by_table: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for snap in cpu_snaps.iter().skip(WARMUP_DISCARD) {
        for (name, dur) in &snap.trace_build_tables {
            by_table.entry(name.clone()).or_default().push(*dur);
        }
    }
    let mut rows: Vec<(String, f64)> = by_table
        .into_iter()
        .map(|(name, vals)| (name, stats(&vals).median))
        .collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (name, med) in &rows {
        println!("    {:<24} {}", name, fmt_secs(*med));
    }
    println!();
}
