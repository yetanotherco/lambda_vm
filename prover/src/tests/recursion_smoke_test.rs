//! End-to-end naive recursion pipeline smoke tests: prove an inner program,
//! build the guest's private-input blob with `recursion::encode_guest_input`,
//! hand it to the in-VM verifier guest, then execute or prove the guest. Guest
//! ELFs come from `make compile-recursion-elfs`. `ProofOptions` is fixed per
//! preset at build time (`recursion::Preset`); `decode_commitment`/
//! `page_commitments` are private input, precomputed host-side. Each pipeline
//! host-verifies the inner proof via full recompute (`None, None`) as a
//! ground-truth check, then runs the production consumer check
//! (`recursion::check_attestation`) over the guest's committed attestation.

use std::ops::ControlFlow;
use std::path::PathBuf;

use crate::recursion::{self, MIN_PROOF_OPTIONS, Preset};

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

/// Prove `inner_elf` under `opts` and build the guest's private-input blob via
/// [`recursion::encode_guest_input`] (which precomputes the DECODE/page roots
/// and rkyv-encodes the [`crate::GuestInput`]). Returns the proof and the
/// blob.
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

    let blob = recursion::encode_guest_input(&inner_proof, inner_elf, opts)
        .expect("recursion::encode_guest_input failed");
    eprintln!("[{tag}] rkyv blob: {} bytes", blob.len());
    (inner_proof, blob)
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

/// The identity + output a correct in-VM run must commit — the profile
/// tests' correctness oracle, computed host-side before the guest runs and
/// checked against its committed attestation in [`run_profile_from`].
struct ExpectedAttestation {
    id: [u8; 32],
    output: Vec<u8>,
}

/// Shared preamble: build the blob (an `empty` inner proof under the preset's
/// options), load the `recursion-<preset>.elf` verifier, and stand up an
/// executor. Returns `(elf_bytes, program, executor, expected_attestation)`.
fn setup_guest_run(
    label: &str,
    preset: Preset,
) -> (
    Vec<u8>,
    executor::elf::Elf,
    executor::vm::execution::Executor,
    ExpectedAttestation,
) {
    let root = workspace_root();
    let empty_elf_bytes = read_guest_elf(&root, "empty");
    let guest_elf_bytes = read_guest_elf(&root, preset.artifact_stem());

    let (inner_proof, blob) =
        prove_inner_and_encode_blob(label, &empty_elf_bytes, &[], &preset.options());

    let expected = ExpectedAttestation {
        id: recursion::expected_program_id(&empty_elf_bytes, &preset.options())
            .expect("expected_program_id errored"),
        output: inner_proof.public_output,
    };

    let program = executor::elf::Elf::load(&guest_elf_bytes).expect("ELF load failed");
    assert_ne!(
        program.entry_point,
        0,
        "recursion-{} ELF has entry_point=0 — build artifact is malformed",
        preset.name()
    );
    let executor =
        executor::vm::execution::Executor::new(&program, blob).expect("Executor::new failed");
    (guest_elf_bytes, program, executor, expected)
}

/// [`setup_guest_run`]'s fixture-based counterpart for a real ethrex block:
/// reads a pre-proved continuation input (`make recursion-profile-block-input`)
/// instead of proving one in-process, so this test only ever measures the
/// verifier guest, never the inner prove.
fn setup_block4_blowup4_guest_run() -> (
    Vec<u8>,
    executor::elf::Elf,
    executor::vm::execution::Executor,
    ExpectedAttestation,
) {
    let root = workspace_root();
    let guest_elf_bytes = read_guest_elf(&root, "recursion-cont-blowup4");

    let art = root.join("executor/program_artifacts/recursion");
    let blob_path = art.join("recursion-cont-blowup4-block4.bin");
    let blob = std::fs::read(&blob_path).unwrap_or_else(|e| {
        panic!(
            "failed to read {} — run `make recursion-profile-block-input`: {e}",
            blob_path.display()
        )
    });
    let expected_path = art.join("recursion-cont-blowup4-block4.bin.expected");
    let expected_bytes = std::fs::read(&expected_path).unwrap_or_else(|e| {
        panic!(
            "failed to read {} — run `make recursion-profile-block-input`: {e}",
            expected_path.display()
        )
    });
    let (id_bytes, output) = expected_bytes.split_at(32);
    let expected = ExpectedAttestation {
        id: id_bytes
            .try_into()
            .expect("expected sidecar id is 32 bytes"),
        output: output.to_vec(),
    };

    let program = executor::elf::Elf::load(&guest_elf_bytes).expect("ELF load failed");
    assert_ne!(
        program.entry_point, 0,
        "recursion-cont-blowup4 ELF has entry_point=0 — build artifact is malformed",
    );
    let executor =
        executor::vm::execution::Executor::new(&program, blob).expect("Executor::new failed");
    (guest_elf_bytes, program, executor, expected)
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

/// Print one top-25 table. `rows` is `(name, cycles, distinct_pcs)`;
/// `denom_cycles` is the percentage denominator (global total for the all-steps
/// table, that step's own total for a per-step table).
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

/// Single-pass execute-only profiler over the `empty` inner program (the
/// verifier's intrinsic recursion overhead, not a real workload). Always
/// prints total cycles, the per-step cycle breakdown (marker decode is cheap —
/// one `InstructionCache` lookup per cycle), and a rough trace/LDE estimate;
/// with `detailed`, also the top-25 functions table (needs a `pc_hist`
/// HashMap, so gated).
fn run_profile(preset: Preset, progress_stride: usize, detailed: bool) {
    let (guest_elf_bytes, program, executor, expected) = setup_guest_run("profile", preset);
    run_profile_from(
        preset,
        &guest_elf_bytes,
        &program,
        executor,
        progress_stride,
        detailed,
        &expected,
    );
}

/// Shared profiling loop: runs an already-set-up guest executor and prints
/// the same cycle/step/function breakdown regardless of the inner program.
fn run_profile_from(
    preset: Preset,
    guest_elf_bytes: &[u8],
    program: &executor::elf::Elf,
    mut executor: executor::vm::execution::Executor,
    progress_stride: usize,
    detailed: bool,
    expected: &ExpectedAttestation,
) {
    use std::collections::HashMap;

    let opts = preset.options();
    let symbols = executor::elf::SymbolTable::parse(guest_elf_bytes);
    let instructions = executor::vm::execution::InstructionCache::new(&program.data)
        .expect("instruction cache build failed");

    let mut pc_hist: HashMap<(u64, u8), u64> = HashMap::new();
    let mut buckets = [0u64; 7];
    let bucket = std::cell::Cell::new(0u8);
    let unique = std::cell::Cell::new(0usize);

    eprintln!(
        "[profile] executing recursion-{} guest ({}) ...",
        preset.name(),
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

    // Correctness, not just crash-freedom: check the guest's committed
    // attestation against the trusted host recompute (`expected`).
    let committed = executor
        .finish()
        .expect("read committed output after execution")
        .memory_values;
    let (id, output) = recursion::split_attestation(&committed)
        .expect("attestation too short (guest committed fewer than 32 bytes)");
    assert_eq!(
        id, expected.id,
        "guest attestation program_id mismatch — in-VM verify accepted a different \
         (ELF, roots) identity than the trusted host recompute"
    );
    assert_eq!(
        output,
        expected.output.as_slice(),
        "attested inner public output mismatch — the in-VM verify's committed output \
         diverges from the trusted host recompute"
    );
    eprintln!(
        "[profile] guest attestation matched the trusted host recompute (program_id + inner public output) ✓"
    );

    eprintln!();
    eprintln!("============================================================");
    eprintln!(
        "  RECURSION-{} GUEST PROFILE (blowup={}, {} queries)",
        preset.name().to_uppercase(),
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

/// Core pipeline: prove the inner program under `preset.options()`, run the
/// guest (`recursion-<preset>.elf`) to `mode`, then run the production consumer
/// check — recompute the program id from the trusted inner ELF and compare it
/// against the guest's committed attestation (`program_id || inner_public_output`)
/// via [`recursion::check_attestation`]. A match means the in-VM verifier
/// accepted the proof and the host-side identity binding holds.
fn run_recursion_pipeline_with_options(
    label: &str,
    inner_elf_bytes: &[u8],
    inner_private_input: &[u8],
    preset: Preset,
    mode: OuterMode,
) {
    let root = workspace_root();
    let recursion_elf_bytes = read_guest_elf(&root, preset.artifact_stem());
    let opts = preset.options();

    let (inner_proof, blob) =
        prove_inner_and_encode_blob(label, inner_elf_bytes, inner_private_input, &opts);

    assert!(
        crate::verify_with_options(&inner_proof, inner_elf_bytes, &opts, None, None)
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

    // Production consumer path: recompute the id from the trusted inner ELF and
    // compare against the guest's committed attestation (the real host-side
    // binding). `Some(inner_public_output)` iff the recompute matches.
    let inner_output = recursion::check_attestation(&committed, inner_elf_bytes, &opts)
        .expect("check_attestation errored")
        .expect(
            "guest attestation must match the trusted inner ELF (program_id recompute+compare)",
        );
    assert_eq!(
        inner_output, inner_proof.public_output,
        "attested inner public output must equal the inner proof's public output"
    );
    eprintln!("[{label}] guest attestation matched the trusted inner ELF (program_id recompute) ✓");
}

/// `run_recursion_pipeline_with_options` at `blowup=8` (the `empty`/`fibonacci`
/// default), i.e. [`Preset::Blowup8`].
fn run_recursion_pipeline(
    label: &str,
    inner_elf_bytes: &[u8],
    inner_private_input: &[u8],
    mode: OuterMode,
) {
    run_recursion_pipeline_with_options(
        label,
        inner_elf_bytes,
        inner_private_input,
        Preset::Blowup8,
        mode,
    );
}

/// Decode the blob on the host and mirror the guest's verify+attest, then run
/// the consumer check — a cheap guard on the encode/decode/attest contract
/// without running the VM.
#[test]
fn test_recursion_blob_decodes_and_verifies_on_host() {
    let root = workspace_root();
    let empty_elf_bytes = read_guest_elf(&root, "empty");
    let (inner, blob) =
        prove_inner_and_encode_blob("roundtrip", &empty_elf_bytes, &[], &MIN_PROOF_OPTIONS);

    // Mirror the guest exactly: decode + verify_and_attest_blob over the blob.
    let attestation = match recursion::verify_and_attest_blob(&blob, &MIN_PROOF_OPTIONS) {
        Ok(Some(a)) => {
            eprintln!("[roundtrip] verify_and_attest_blob accepted — guest path is sound");
            a
        }
        Ok(None) => panic!(
            "[roundtrip] verify_and_attest_blob returned None (guest hits the failed-verification expect) — proof did not survive the rkyv round-trip"
        ),
        Err(e) => panic!("[roundtrip] verify_and_attest_blob ERRORED (guest hits .expect): {e:?}"),
    };

    // Consumer check: the committed attestation must bind to the trusted inner
    // ELF and carry the inner proof's public output.
    let output = recursion::check_attestation(&attestation, &empty_elf_bytes, &MIN_PROOF_OPTIONS)
        .expect("check_attestation errored")
        .expect("attestation must match the trusted inner ELF (program_id recompute+compare)");
    assert_eq!(
        output, inner.public_output,
        "attested public output must equal the inner proof's public output"
    );

    // Host buffers carry no alignment guarantee, so `verify_recursion_blob`
    // must accept the blob at any base alignment (falling back to an aligned
    // copy when needed). The plain call above already exercises the common
    // misaligned case (`Vec` base + 12-byte prefix → 4-aligned archive);
    // shifting the base by 4 covers another residue class.
    let mut padded: Vec<u8> = Vec::with_capacity(blob.len() + 4);
    padded.extend_from_slice(&[0u8; 4]);
    padded.extend_from_slice(&blob);
    let v = crate::verify_recursion_blob(&padded[4..], &MIN_PROOF_OPTIONS)
        .expect("verify_recursion_blob errored on misaligned buffer");
    assert!(v.ok, "misaligned-buffer verify must also succeed");
}

/// Continuation flavor of the roundtrip guard: prove the empty program via
/// continuations (tiny epochs so the bundle is genuinely multi-epoch), encode
/// the [`recursion::ContinuationGuestInput`] blob, decode it exactly as the
/// intended `continuation`-feature guest would, and mirror its
/// `verify_continuation_and_attest` call — a cheap host-side check of the
/// encode/decode/verify/attest contract without running the VM.
#[test]
fn test_recursion_continuation_blob_decodes_and_verifies_on_host() {
    let root = workspace_root();
    let fib_elf_bytes = read_guest_elf(&root, "fibonacci");
    let inner_input = 10u64.to_le_bytes();

    let bundle = crate::continuation::prove_continuation(
        &fib_elf_bytes,
        &inner_input,
        4,
        &MIN_PROOF_OPTIONS,
    )
    .expect("continuation prove should succeed");
    assert!(
        bundle.num_epochs() > 1,
        "epoch=2^4 must split fibonacci(10) into multiple epochs for this test to bite"
    );
    // Ground truth: the trustless recompute path must accept the bundle.
    let expected_output =
        crate::continuation::verify_continuation(&fib_elf_bytes, &bundle, &MIN_PROOF_OPTIONS)
            .expect("verify_continuation errored")
            .expect("bundle must verify with recomputed roots");
    // Consumer re-bind values, computed before the encode consumes the bundle:
    // recompute the roots from the bundle + trusted ELF and compare ids (the
    // continuation analog of check_attestation).
    let (expected_decode, expected_pages) =
        crate::continuation::continuation_precomputed_commitments(
            &fib_elf_bytes,
            &bundle,
            &MIN_PROOF_OPTIONS,
        )
        .expect("continuation_precomputed_commitments errored");
    let expected_id =
        recursion::program_id_from_elf(&fib_elf_bytes, &expected_decode, &expected_pages)
            .expect("program_id_from_elf errored");

    let blob =
        recursion::encode_continuation_guest_input(bundle, &fib_elf_bytes, &MIN_PROOF_OPTIONS)
            .expect("encode_continuation_guest_input failed");

    // Verify exactly as the guest does (built with `continuation` + `min`):
    // prefix validation + rkyv access + deserialize + verify + attest.
    let attestation = recursion::verify_continuation_and_attest(&blob, &MIN_PROOF_OPTIONS)
        .expect("verify_continuation_and_attest errored")
        .expect("continuation proof did not survive the rkyv round-trip");
    let (id, output) = recursion::split_attestation(&attestation).expect("attestation too short");
    assert_eq!(
        id, expected_id,
        "attested id must match the honest recompute"
    );
    assert_eq!(
        output,
        &expected_output[..],
        "supplied-roots output must match the recompute path's output"
    );
}

/// Corrupting a private-input commitment on an *honest* proof makes
/// verification fail (`Ok(false)`). Necessary but not sufficient alone — a
/// custom prover can supply consistent mismatched roots (see
/// `recursion_soundness_gap_poc`); the identity binding is the `program_id`
/// fold, not this check.
#[test]
fn test_recursion_rejects_corrupted_commitment() {
    let root = workspace_root();
    let empty_elf_bytes = read_guest_elf(&root, "empty");
    let (vm_proof, _blob) = prove_inner_and_encode_blob(
        "corrupt-commitment",
        &empty_elf_bytes,
        &[],
        &MIN_PROOF_OPTIONS,
    );
    let (mut decode_commitment, page_commitments) =
        recursion::precomputed_commitments(&empty_elf_bytes, &MIN_PROOF_OPTIONS)
            .expect("precomputed_commitments failed");
    decode_commitment[0] ^= 0xFF;

    let ok = crate::verify_with_options(
        &vm_proof,
        &empty_elf_bytes,
        &MIN_PROOF_OPTIONS,
        Some(decode_commitment),
        Some(&page_commitments),
    )
    .expect("verify errored");
    assert!(
        !ok,
        "corrupted decode_commitment must be rejected, not silently accepted"
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
        Preset::Min,
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
    let (_bytes, program, mut executor, _expected) = setup_guest_run("step-markers", Preset::Min);
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
        Preset::Min,
        OuterMode::Prove,
    );
}

/// Dump the guest's private-input blob to `/tmp/recursion_input.bin` for the
/// CLI's `execute --flamegraph` and `scripts/bench_recursion_cycles.sh`.
///
/// Env knobs:
/// * `RECURSION_DUMP_PRESET` (`min`|`blowup2`|`blowup4`|`blowup8`, default
///   `min`) — must match the `recursion-<preset>.elf` the blob is fed to.
/// * `RECURSION_DUMP_INNER_ELF` (path, default the `empty` guest).
/// * `RECURSION_DUMP_INNER_INPUT` (path, default none).
/// * `RECURSION_DUMP_EPOCH_LOG2` (int, default unset = monolithic) — prove via
///   continuations with `2^n`-cycle epochs and encode a
///   [`recursion::ContinuationGuestInput`] blob for `recursion-cont-<preset>.elf`.
#[test]
#[ignore = "diagnostic: writes recursion private input to /tmp/recursion_input.bin"]
fn test_dump_recursion_input() {
    let root = workspace_root();

    let preset_name = std::env::var("RECURSION_DUMP_PRESET").unwrap_or_else(|_| "min".to_string());
    let preset = Preset::ALL
        .into_iter()
        .find(|p| p.name() == preset_name)
        .unwrap_or_else(|| {
            panic!(
                "unknown RECURSION_DUMP_PRESET '{preset_name}' (expected min|blowup2|blowup4|blowup8)"
            )
        });

    let (inner_elf_bytes, inner_label) = match std::env::var("RECURSION_DUMP_INNER_ELF") {
        Ok(p) => (
            std::fs::read(&p).unwrap_or_else(|e| panic!("read RECURSION_DUMP_INNER_ELF {p}: {e}")),
            p,
        ),
        Err(_) => (read_guest_elf(&root, "empty"), "empty".to_string()),
    };
    let inner_input = match std::env::var("RECURSION_DUMP_INNER_INPUT") {
        Ok(p) => {
            std::fs::read(&p).unwrap_or_else(|e| panic!("read RECURSION_DUMP_INNER_INPUT {p}: {e}"))
        }
        Err(_) => Vec::new(),
    };

    // Continuation dumps also get an `.expected` sidecar (32-byte id || inner
    // public output), computed here while the `ContinuationProof` bundle
    // still exists (`encode_continuation_guest_input` consumes it) — lets a
    // consumer check the pre-proved fixture without re-deriving it.
    let (blob, expected_sidecar) = match std::env::var("RECURSION_DUMP_EPOCH_LOG2") {
        Ok(s) => {
            // No recursion-cont-blowup8.elf is built (RECURSION_CONT_PRESETS
            // stops at blowup4).
            assert_ne!(
                preset,
                Preset::Blowup8,
                "RECURSION_DUMP_PRESET=blowup8 has no recursion-cont-blowup8.elf guest; \
                 continuation mode only supports min|blowup2|blowup4"
            );
            let epoch_log2: u32 = s
                .parse()
                .unwrap_or_else(|e| panic!("bad RECURSION_DUMP_EPOCH_LOG2 '{s}': {e}"));
            let opts = preset.options();
            eprintln!(
                "[dump-input] proving inner continuation (blowup={}, fri_queries={}, epoch=2^{epoch_log2}) ...",
                opts.blowup_factor, opts.fri_number_of_queries
            );
            let bundle = crate::continuation::prove_continuation(
                &inner_elf_bytes,
                &inner_input,
                epoch_log2,
                &opts,
            )
            .expect("inner continuation prove should succeed");
            eprintln!("[dump-input] continuation epochs: {}", bundle.num_epochs());

            let expected_output =
                crate::continuation::verify_continuation(&inner_elf_bytes, &bundle, &opts)
                    .expect("verify_continuation errored")
                    .expect("continuation bundle must verify on host before dumping");
            let (expected_decode, expected_pages) =
                crate::continuation::continuation_precomputed_commitments(
                    &inner_elf_bytes,
                    &bundle,
                    &opts,
                )
                .expect("continuation_precomputed_commitments errored");
            let expected_id =
                recursion::program_id_from_elf(&inner_elf_bytes, &expected_decode, &expected_pages)
                    .expect("program_id_from_elf errored");

            let blob = recursion::encode_continuation_guest_input(bundle, &inner_elf_bytes, &opts)
                .expect("recursion::encode_continuation_guest_input failed");
            (blob, Some((expected_id, expected_output)))
        }
        Err(_) => {
            let (_inner_proof, blob) = prove_inner_and_encode_blob(
                "dump-input",
                &inner_elf_bytes,
                &inner_input,
                &preset.options(),
            );
            (blob, None)
        }
    };
    assert!(
        blob.len() <= executor::vm::memory::MAX_PRIVATE_INPUT_SIZE as usize,
        "recursion input exceeds MAX_PRIVATE_INPUT_SIZE"
    );

    let path = "/tmp/recursion_input.bin";
    std::fs::write(path, &blob).expect("write blob");
    eprintln!(
        "[dump-input] preset={} inner={inner_label} wrote {} bytes to {path}",
        preset.name(),
        blob.len()
    );

    if let Some((id, output)) = expected_sidecar {
        let mut sidecar_data = Vec::with_capacity(32 + output.len());
        sidecar_data.extend_from_slice(&id);
        sidecar_data.extend_from_slice(&output);
        let sidecar_path = format!("{path}.expected");
        std::fs::write(&sidecar_path, &sidecar_data).expect("write expected sidecar");
        eprintln!(
            "[dump-input] wrote {} bytes to {sidecar_path}",
            sidecar_data.len()
        );
    }
}

/// Cycle count only of the recursion guest verifying a 1-query inner proof.
#[test]
#[ignore = "diagnostic: fast; recursion guest cycle count (1 query)"]
fn test_recursion_cycles_1query() {
    run_profile(Preset::Min, 500, false);
}

/// Cycle count only at 128-bit security: more FRI queries → more verifier cycles.
#[test]
#[ignore = "diagnostic: fast; recursion guest cycle count (multi-query)"]
fn test_recursion_cycles_multiquery() {
    run_profile(Preset::Blowup8, 500, false);
}

/// Cycle count only at 128-bit security with the realistic base-layer shape:
/// blowup=2 yields ~0.49 bits/query, so the full 219-query FRI dominates.
#[test]
#[ignore = "diagnostic: recursion guest cycle count (blowup=2, 219 queries)"]
fn test_recursion_cycles_blowup2() {
    run_profile(Preset::Blowup2, 500, false);
}

/// Cycle count only at 128-bit security, blowup=4 (110 queries) — the other
/// realistic base-layer point.
#[test]
#[ignore = "diagnostic: recursion guest cycle count (blowup=4, 110 queries)"]
fn test_recursion_cycles_blowup4() {
    run_profile(Preset::Blowup4, 500, false);
}

/// Full profile (top-25 + per-step) of the recursion `continuation` guest
/// verifying a REAL ethrex block (4 transfers), blowup=4 — not the
/// `empty`-program diagnostic floor `test_recursion_profile_1query`/
/// `_multiquery` measure. Requires `make recursion-profile-block-input`.
#[test]
#[ignore = "diagnostic: heavy; recursion guest histogram + steps over a real ethrex block (blowup=4)"]
fn test_recursion_profile_blowup4_block() {
    let (guest_elf_bytes, program, executor, expected) = setup_block4_blowup4_guest_run();
    run_profile_from(
        Preset::Blowup4,
        &guest_elf_bytes,
        &program,
        executor,
        500,
        true,
        &expected,
    );
}

/// Full profile (top-25 + per-step) of the 1-query run.
#[test]
#[ignore = "diagnostic: ~8 min; recursion guest histogram + steps (1 query)"]
fn test_recursion_profile_1query() {
    run_profile(Preset::Min, 500, true);
}

/// Full profile at 128-bit security: weight shifts toward per-query FRI/Merkle.
#[test]
#[ignore = "diagnostic: heavy; recursion guest histogram + steps (multi-query)"]
fn test_recursion_profile_multiquery() {
    run_profile(Preset::Blowup8, 500, true);
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
