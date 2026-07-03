//! End-to-end naive recursion pipeline smoke tests: prove an inner program,
//! hand `(VmProof, elf, opts, vkey)` to the in-VM verifier guest, then either prove
//! the guest's execution (`OuterMode::Prove`) or just execute it
//! (`OuterMode::ExecuteOnly`). Guest ELFs come from `make compile-recursion-elfs`.
//!
//! Every pipeline host-verifies the inner proof, so building with
//! `--features stark/instruments` makes any of these tests print the verifier's
//! per-step `Time spent:` timings.

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

/// Smallest possible inner proof (blowup=2, 1 query). Intentionally insecure —
/// for the cheap diagnostics, not soundness.
const MIN_PROOF_OPTIONS: stark::proof::options::ProofOptions =
    stark::proof::options::ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 1,
        coset_offset: 3,
        grinding_factor: 1,
    };

/// Prove `inner_elf` under `opts` and rkyv-encode a `RecursionInput` into the
/// guest's private-input blob (magic/version prefix + archive). Returns the
/// proof and the blob.
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

    let elf_for_vkey = executor::elf::Elf::load(inner_elf).expect("ELF load failed");
    let page_configs = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
        &elf_for_vkey,
        &inner_proof.runtime_page_ranges,
        inner_proof.num_private_input_pages,
    );
    let vkey =
        crate::VmVerifyingKey::from_elf_and_options(&elf_for_vkey, opts, None, &page_configs);
    let input = crate::RecursionInput {
        vm_proof: inner_proof,
        inner_elf: inner_elf.to_vec(),
        options: opts.clone(),
        vkey,
    };
    let blob = crate::encode_recursion_input(&input).expect("encode recursion input");
    eprintln!("[{tag}] rkyv blob: {} bytes", blob.len());
    (input.vm_proof, blob)
}

/// Whether to also prove the guest's own execution after handing it the proof.
#[derive(Clone, Copy, Debug)]
enum OuterMode {
    /// Execute in-VM, read the committed marker off memory; no LDE/FRI.
    ExecuteOnly,
    /// Prove the execution (memory-bounded via continuations) and verify on host.
    Prove,
}

/// Execute the recursion guest in-VM on `blob` and return its committed bytes,
/// read straight off the executor's memory after a streamed run.
fn execute_outer_and_commit(label: &str, recursion_elf_bytes: &[u8], blob: &[u8]) -> Vec<u8> {
    use executor::elf::Elf;
    use executor::vm::execution::Executor;

    eprintln!("[{label}] executing outer (recursion guest, in-VM verify, streaming) ...");
    let program = Elf::load(recursion_elf_bytes).expect("load recursion elf");
    let mut executor = Executor::new(&program, blob.to_vec()).expect("executor new");

    let (total_cycles, exec_time) = drive_executor(
        &mut executor,
        |_log| ControlFlow::Continue(()),
        |_, _, _| {},
    );

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

/// Prove the guest's execution via continuations, verify on host, return the
/// committed bytes.
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

/// Stream a guest's execution via `Executor::resume()` without buffering the log
/// stream. `on_log` returns `Break` to stop early; `on_progress` fires per chunk.
/// Returns `(total_cycles, wall_time)`, exact even on an early break.
fn drive_executor(
    executor: &mut executor::vm::execution::Executor,
    mut on_log: impl FnMut(&executor::vm::logs::Log) -> ControlFlow<()>,
    mut on_progress: impl FnMut(usize, u64, std::time::Duration),
) -> (u64, std::time::Duration) {
    let start = std::time::Instant::now();
    let mut total_cycles: u64 = 0;
    let mut chunks: usize = 0;
    while let Some(logs) = executor
        .resume()
        .expect("executor resume failed (guest panicked in-VM?)")
    {
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

/// Shared preamble: build the blob (an `empty` inner proof under `opts`), load
/// `guest_name`, and stand up an executor. Returns `(elf_bytes, program, executor)`.
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

/// Demangled enclosing-function name for a PC via the ELF symbol table;
/// `<unknown>` if none covers it. No file:line (symtab has no DWARF).
fn resolve_pc(symbols: &executor::elf::SymbolTable, pc: u64) -> String {
    symbols.lookup(pc).map_or_else(
        || "<unknown>".to_string(),
        |s| executor::flamegraph::demangle(&s.name),
    )
}

/// Verifier sub-steps in execution order, keyed by `stark::profile_markers::STEP_*`
/// value. `run_profile` buckets cycles by the latest marker observed so far
/// (`decode_step_marker`, defaulting to bucket 0 until the first marker fires),
/// so `multi_verify`'s per-table `3,4,5,6` repetition re-attributes cycles to
/// the correct step on each table's `6->3` transition instead of latching at 6.
const STEP_LABELS: [&str; 7] = [
    "0. setup (alloc init + blob prefix check)",
    "1. airs_and_bus_balance (Elf::load/VmAirs::new preprocessed FFT+Merkle/bus balance)",
    "2. multi_verify setup (transcript replay phase A/B, per-table fork)",
    "3. step 1: replay_rounds_after_round_1",
    "4. step 2: verify_claimed_composition_polynomial",
    "5. step 3: verify_fri",
    "6. step 4: verify_trace_and_composition_openings (+ wrap-up)",
];

/// `blowup=8` (128-bit, multi-query) options for the `multiquery` variants.
fn blowup8() -> stark::proof::options::ProofOptions {
    crate::GoldilocksCubicProofOptions::with_blowup(8).expect("blowup=8 is always valid")
}

/// Short per-step tag for the function table, keyed by the same bucket index
/// used in `STEP_LABELS`/`buckets`.
fn step_tag(bucket: u8) -> &'static str {
    match bucket {
        0 => "setup",
        1 => "airs_bus_balance",
        2 => "multi_verify_setup",
        3 => "step1:replay",
        4 => "step2:claimed",
        5 => "step3:fri",
        6 => "step4:openings",
        _ => "?",
    }
}

/// Print one top-25 table: `rows` is `(name, cycles, distinct_pcs)`, already
/// unsorted; `denom_cycles` is the denominator for percentages — the global
/// total for the all-steps table, but *that step's own total* for a per-step
/// table, so `%`/`cum %` show what dominates within that step (a `keccak`
/// that's 90% of a cheap step should read as 90%, not as a fraction of a
/// percent of the whole run).
fn print_top25_table(rows: &mut [(String, u64, u64)], denom_cycles: u64) {
    rows.sort_unstable_by_key(|(_name, cycles, _pcs)| std::cmp::Reverse(*cycles));
    let pct = |n: u64| 100.0 * (n as f64) / (denom_cycles as f64);
    eprintln!("  rank          cycles        %    cum %    PCs  function");
    let mut cumulative: u64 = 0;
    for (rank, (name, cycles, pcs)) in rows.iter().take(25).enumerate() {
        cumulative += cycles;
        eprintln!(
            "  {:>4}  {:>14}  {:>6.2}%  {:>6.2}%  {:>5}  {}",
            rank + 1,
            cycles,
            pct(*cycles),
            pct(cumulative),
            pcs,
            name,
        );
    }
}

/// Print the global top-25 functions by cycle count, then one top-25 table
/// per verifier step — so e.g. how much of `step4:openings` is spent in
/// `keccak` is visible at a glance, instead of only the function's total
/// across all steps.
fn print_function_table(
    symbols: &executor::elf::SymbolTable,
    pc_hist: std::collections::HashMap<(u64, u8), u64>,
    total_cycles: u64,
) {
    let mut by_function: std::collections::HashMap<String, (u64, u64)> =
        std::collections::HashMap::new();
    let mut by_function_per_step: std::collections::HashMap<
        u8,
        std::collections::HashMap<String, (u64, u64)>,
    > = std::collections::HashMap::new();
    let mut unique_pcs: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for ((pc, bucket), count) in &pc_hist {
        unique_pcs.insert(*pc);
        let name = resolve_pc(symbols, *pc);

        let entry = by_function.entry(name.clone()).or_insert((0, 0));
        entry.0 += *count; // cycles
        entry.1 += 1; // distinct PCs folded into this function

        let step_entry = by_function_per_step
            .entry(*bucket)
            .or_default()
            .entry(name)
            .or_insert((0, 0));
        step_entry.0 += *count;
        step_entry.1 += 1;
    }

    eprintln!("  Unique PCs   : {}", unique_pcs.len());
    eprintln!();
    eprintln!(
        "  Top 25 functions by cycle count (aggregated over their PCs, all steps; % of total cycles):"
    );
    let mut rows: Vec<(String, u64, u64)> = by_function
        .into_iter()
        .map(|(name, (cycles, pcs))| (name, cycles, pcs))
        .collect();
    print_top25_table(&mut rows, total_cycles);

    for bucket in 0u8..STEP_LABELS.len() as u8 {
        let Some(by_step_function) = by_function_per_step.remove(&bucket) else {
            continue;
        };
        let step_total: u64 = by_step_function.values().map(|(cycles, _pcs)| cycles).sum();
        eprintln!();
        eprintln!(
            "  Top 25 functions by cycle count — step {} (% of this step's {} cycles):",
            step_tag(bucket),
            step_total,
        );
        let mut rows: Vec<(String, u64, u64)> = by_step_function
            .into_iter()
            .map(|(name, (cycles, pcs))| (name, cycles, pcs))
            .collect();
        print_top25_table(&mut rows, step_total);
    }
}

/// Print the per-verifier-step cycle bucketing (`buckets[0]` = setup).
fn print_step_breakdown(buckets: &[u64; 7], total_cycles: u64) {
    eprintln!();
    eprintln!("  Per-step cycle breakdown (latest-marker state machine):");
    eprintln!("  {:<70}  {:>14}  {:>7}", "bucket", "cycles", "%");
    for (label, cycles) in STEP_LABELS.iter().zip(buckets.iter()) {
        let pct = if total_cycles > 0 {
            100.0 * (*cycles as f64) / (total_cycles as f64)
        } else {
            0.0
        };
        eprintln!("  {:<60}  {:>14}  {:>6.2}%", label, cycles, pct);
    }
}

/// Single-pass execute-only profiler. Always prints total cycles, the
/// per-step cycle breakdown (marker decode is cheap — one `InstructionCache`
/// lookup per cycle), and a rough trace/LDE estimate; with `detailed`, also
/// the top-25 functions table (needs a `pc_hist` HashMap, so gated).
fn run_profile(
    guest_name: &str,
    progress_stride: usize,
    opts: stark::proof::options::ProofOptions,
    detailed: bool,
) {
    use std::collections::HashMap;

    let (guest_elf_bytes, program, mut executor) = setup_guest_run("profile", guest_name, &opts);
    let symbols = executor::elf::SymbolTable::parse(&guest_elf_bytes);
    let instructions = executor::vm::execution::InstructionCache::new(&program.data)
        .expect("instruction cache build failed");

    let mut pc_hist: HashMap<(u64, u8), u64> = HashMap::new();
    let mut buckets = [0u64; 7];
    let bucket = std::cell::Cell::new(0u8);
    let unique = std::cell::Cell::new(0usize);

    eprintln!(
        "[profile] executing {guest_name} guest ({}) ...",
        if detailed {
            "histogram + steps"
        } else {
            "steps"
        }
    );
    let (total_cycles, exec_time) = drive_executor(
        &mut executor,
        |log| {
            let pc = log.current_pc;

            if let Some(marker) = executor::vm::execution::decode_step_marker(&instructions, pc) {
                bucket.set(marker as u8);
            }
            buckets[bucket.get() as usize] += 1;

            if detailed {
                *pc_hist.entry((pc, bucket.get())).or_insert(0) += 1;
                unique.set(pc_hist.len());
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
                    eprintln!(
                        "[profile]   ... {chunks} chunks, {cycles} cycles, bucket={}, {elapsed:?}",
                        bucket.get(),
                    );
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

    eprintln!();
    print_step_breakdown(&buckets, total_cycles);
    if detailed {
        eprintln!();
        print_function_table(&symbols, pc_hist, total_cycles);
    }
    eprintln!("============================================================");
}

/// Core pipeline: prove the inner program, run the guest to `mode`, assert it
/// committed `vk_digest ‖ inner public output` — the outer-verifier check:
/// the digest of the vkey used in-guest must match one derived on the host
/// from the trusted inner ELF.
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

    let mut expected = inner_proof.vk_digest.to_vec();
    expected.extend_from_slice(&inner_proof.public_output);
    assert_eq!(
        committed, expected,
        "recursion guest must commit vk_digest ‖ inner public output"
    );
    eprintln!("[{label}] guest committed vk_digest ‖ output: in-VM verify accepted ✓");
}

/// `run_recursion_pipeline_with_options` with `blowup=8` (the `empty`/`fibonacci` default).
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

/// Verify the blob on the host exactly as the guest does (zero-copy through
/// `verify_recursion_blob`) — a cheap guard on the encode/verify contract
/// without running the VM. Also checks the guest's misaligned read conditions
/// and that a tampered proof is rejected.
#[test]
#[ignore = "needs prebuilt guest ELF (make compile-recursion-elfs)"]
fn test_recursion_blob_decodes_and_verifies_on_host() {
    let root = workspace_root();
    let empty_elf_bytes = read_guest_elf(&root, "empty");
    let (inner_proof, blob) =
        prove_inner_and_encode_blob("roundtrip", &empty_elf_bytes, &[], &MIN_PROOF_OPTIONS);

    let verification = crate::verify_recursion_blob(&blob).expect("verify_recursion_blob errored");
    assert!(verification.ok, "zero-copy path must accept a valid proof");
    assert_eq!(
        verification.public_output, inner_proof.public_output,
        "public output must round-trip through the blob"
    );
    assert_eq!(
        verification.vk_digest, inner_proof.vk_digest,
        "vk digest must match the proof's"
    );

    // Host buffers carry no alignment guarantee, so `verify_recursion_blob`
    // must accept the blob at any base alignment (falling back to an aligned
    // copy when needed). The plain call above already exercises the common
    // misaligned case (`Vec` base + 12-byte prefix → 4-aligned archive);
    // shifting the base by 4 covers another residue class.
    let mut padded: Vec<u8> = Vec::with_capacity(blob.len() + 4);
    padded.extend_from_slice(&[0u8; 4]);
    padded.extend_from_slice(&blob);
    let ok_shifted = crate::verify_recursion_blob(&padded[4..])
        .expect("verify_recursion_blob errored on shifted buffer")
        .ok;
    assert!(ok_shifted, "path must accept the proof from a shifted buffer");

    // A bad magic must be rejected before the unsafe access.
    let mut bad_magic = blob.clone();
    bad_magic[0] ^= 0xFF;
    assert!(
        crate::verify_recursion_blob(&bad_magic).is_err(),
        "bad magic must be rejected"
    );

    // Soundness: a single-byte tamper in the proof payload must make the
    // zero-copy verifier reject (Fiat-Shamir / Merkle openings stop matching).
    let mut tampered = blob.clone();
    let tamper_idx = tampered.len() - 64;
    tampered[tamper_idx] ^= 0x01;
    let tampered_result = crate::verify_recursion_blob(&tampered);
    assert!(
        !matches!(tampered_result.map(|v| v.ok), Ok(true)),
        "zero-copy verifier must NOT accept a tampered proof"
    );
}

// === Execute-only tier ========================================================

/// Execute-only: verify a `blowup=8` proof of the empty program in-VM.
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

/// Execute-only: smallest inner proof (blowup=2, 1 query) → least guest work.
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

/// Regression test for the marker mechanism itself: every `STEP_*` marker
/// must be observed at least once during a full verifier run, and each
/// transition between consecutive markers must be a valid step in the
/// verifier's state machine.
///
/// `multi_verify` re-runs `replay_rounds_after_round_1 -> step_2 -> step_3 ->
/// step_4` once per AIR table (see `crypto/stark/src/verifier.rs`), so the
/// full marker sequence isn't monotonic overall — it's `STEP_DECODE_DONE ->
/// STEP_AIRS_AND_BUS_BALANCE_DONE` once each, followed by N repetitions of
/// the `3,4,5,6` cycle (one per table). A transition outside
/// `{1->2, 2->3, 3->4, 4->5, 5->6, 6->3}` means the marker convention broke —
/// wrong immediate decoded, or a stale/mismatched build.
#[test]
#[ignore = "slow: runs the in-VM STARK verifier (minutes on CI)"]
fn test_recursion_step_markers_observed_in_order() {
    let (_bytes, program, mut executor) =
        setup_guest_run("step-markers", "recursion", &MIN_PROOF_OPTIONS);
    let instructions = executor::vm::execution::InstructionCache::new(&program.data)
        .expect("instruction cache build failed");

    let decode_done = stark::profile_markers::STEP_DECODE_DONE;
    let airs_ready = stark::profile_markers::STEP_AIRS_AND_BUS_BALANCE_DONE;
    let replay = stark::profile_markers::STEP_REPLAY_ROUNDS_AFTER_ROUND_1;
    let claimed = stark::profile_markers::STEP_VERIFY_CLAIMED_COMPOSITION_POLYNOMIAL;
    let fri = stark::profile_markers::STEP_VERIFY_FRI;
    let openings = stark::profile_markers::STEP_VERIFY_TRACE_AND_COMPOSITION_OPENINGS;

    let mut last_marker: Option<u32> = None;
    let mut seen = std::collections::HashSet::new();
    drive_executor(
        &mut executor,
        |log| {
            if let Some(marker) =
                executor::vm::execution::decode_step_marker(&instructions, log.current_pc)
            {
                let valid_transition = match last_marker {
                    None => marker == decode_done,
                    Some(last) if last == decode_done => marker == airs_ready,
                    Some(last) if last == airs_ready => marker == replay,
                    Some(last) if last == replay => marker == claimed,
                    Some(last) if last == claimed => marker == fri,
                    Some(last) if last == fri => marker == openings,
                    Some(last) if last == openings => marker == replay,
                    Some(_) => false,
                };
                assert!(
                    valid_transition,
                    "invalid step marker transition: {last_marker:?} -> {marker}"
                );
                last_marker = Some(marker);
                seen.insert(marker);
            }
            ControlFlow::Continue(())
        },
        |_, _, _| {},
    );

    for step in [decode_done, airs_ready, replay, claimed, fri, openings] {
        assert!(seen.contains(&step), "marker {step} was never observed");
    }
}

/// Execute-only: verify a `blowup=8` proof of fibonacci(10) in-VM.
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

/// Inner program: empty — the verifier's intrinsic recursion overhead.
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

/// Inner program: empty, blowup=2/1-query. Quick profiling only.
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

/// Dump the guest's private-input blob to `/tmp/recursion_input.bin` for the
/// CLI's `execute --flamegraph`.
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

/// Full profile (top-25 + per-step) of the 1-query run.
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
