//! DEGREE-LANE EXPERIMENT (temporary, not for merge).
//!
//! Measures what constraint degree costs the *verifier*, which for recursion is
//! the number that matters: the next layer pays it in-circuit, per query.
//!
//! Degree reaches the verifier through exactly one channel — the composition
//! polynomial's part count, `parts = max_degree - 1`. The verifier never
//! evaluates constraints on the LDE domain (only once, at the OOD point z), so
//! constraint complexity is invisible to it. Per query it does:
//!   * one leaf hash over `2 * parts` extension elements, and
//!   * one Merkle path whose length depends on the domain, NOT on parts.
//! So degree widens the leaf but does not lengthen the walk. This test measures
//! that split directly via the `hash-count` counters.
//!
//! ⚠ The hash counters are process-global atomics, and the prover/verifier hash
//! on rayon worker threads, so a second test running concurrently in the same
//! process corrupts the reading (measured: 158,730 → 655,613 permutations when
//! the prove tests ran alongside). Always select the counting test EXACTLY —
//! `degree_cost_verifier_hashes` — and never with a prefix that also matches
//! the prove arms. The prove arms are deliberately named `degree_prove_*` so
//! that a `degree_cost` filter cannot catch them.
//!
//! Sweep the arms with the `VM_MAX_DEGREE` constant (crate root) and the
//! `LVM_DEGREE_BLOWUP` env var. Requires `--features hash-count`; counting is
//! deterministic, so one run is an exact reading.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::proof::view::StarkProofView;
use stark::traits::AIR;
#[allow(unused_imports)]
use stark::proof::view::MultiProofView;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::VmAirs;
use crate::VmProof;
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::test_utils::multi_prove_ram;

use executor::elf::Elf;
use executor::vm::execution::Executor;

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Blowup for this arm (`LVM_DEGREE_BLOWUP`, default 4). Query count follows
/// from the Johnson-bound formula, so these are real 128-bit-target shapes:
/// blowup 2 → 219 q, 4 → 110 q, 8 → 73 q.
fn arm_options() -> ProofOptions {
    let blowup: u8 = std::env::var("LVM_DEGREE_BLOWUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    GoldilocksCubicProofOptions::with_blowup(blowup).expect("valid blowup")
}


/// Number of degree-lane measurement arms currently running in this process.
///
/// The hash counters are process-global and the prover hashes on rayon workers,
/// so a concurrently running arm silently corrupts a reading. This turns that
/// into a loud failure: every arm registers itself, and the counting arm
/// asserts it is alone. Verified to fire — running the three arms together
/// under one filter trips it, and the corrupt reading it prevents was
/// 551,944 permutations against a true 158,730.
static ACTIVE_ARMS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct ArmGuard;

impl ArmGuard {
    fn enter() -> Self {
        ACTIVE_ARMS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Self
    }
    /// Assert this arm has the process to itself. Must be checked at BOTH ends
    /// of the measured window, not just at test entry: pollution happens when
    /// another arm hashes *between* the counter reset and the read, which an
    /// entry-only check cannot see.
    fn assert_exclusive(&self, when: &str) {
        let n = ACTIVE_ARMS.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            n, 1,
            "degree-lane ({when}): {n} measurement arms are running concurrently, \
             so the process-global hash counters are corrupted. Select exactly one \
             arm (e.g. `degree_cost_verifier_hashes`) and run with --test-threads=1."
        );
    }
}

impl Drop for ArmGuard {
    fn drop(&mut self) {
        ACTIVE_ARMS.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// The workload for this arm. `LVM_DEGREE_ELF_PATH` takes an arbitrary ELF
/// (real-scale programs live outside `program_artifacts/asm`);
/// `LVM_DEGREE_ELF` names one of the asm artifacts. Taller traces mean deeper
/// Merkle trees, which shifts the balance between the per-query part cost and
/// the per-query-per-level path cost.
fn arm_elf() -> (String, Vec<u8>) {
    if let Ok(path) = std::env::var("LVM_DEGREE_ELF_PATH") {
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let name = std::path::Path::new(&path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone());
        return (name, bytes);
    }
    let name = std::env::var("LVM_DEGREE_ELF").unwrap_or_else(|_| "sub".to_string());
    let bytes = crate::test_utils::asm_elf_bytes(&name);
    (name, bytes)
}

fn build_airs(
    elf: &Elf,
    opts: &ProofOptions,
    page_configs: &[crate::tables::page::PageConfig],
    table_counts: &crate::TableCounts,
) -> VmAirs {
    VmAirs::new(
        elf,
        opts,
        true,
        page_configs,
        table_counts,
        None,
        true,
        None,
        None,
        None,
    )
}

/// Prove `elf` under this arm's options, then verify with the hash counters
/// reset so the reading is the VERIFIER's work alone.
fn measure_arm(elf_bytes: &[u8], label: &str, guard: &ArmGuard) {
    let opts = arm_options();
    let elf = Elf::load(elf_bytes).expect("ELF load");
    let executor = Executor::new(&elf, Vec::new()).expect("executor");
    let result = executor.run().expect("execution");
    let max_rows = MaxRowsConfig::default();
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &max_rows, &[]).unwrap();
    let table_counts = traces.table_counts();
    let airs = build_airs(&elf, &opts, &traces.page_configs, &table_counts);
    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();

    let proof = multi_prove_ram(
        airs.air_trace_pairs(&mut traces),
        &mut DefaultTranscript::<E>::new(&[]),
    )
    .expect("prove");

    let vm_proof = VmProof {
        proof,
        runtime_page_ranges,
        table_counts: table_counts.clone(),
        public_output: traces.public_output_bytes.clone(),
        num_private_input_pages,
    };

    // Rebuild the verifier's own AIRs, exactly as a real verifier would.
    let page_configs = Traces::page_configs_from_elf_and_runtime(
        &elf,
        &vm_proof.runtime_page_ranges,
        vm_proof.num_private_input_pages,
    );
    let vairs = build_airs(&elf, &opts, &page_configs, &vm_proof.table_counts);
    let air_refs = vairs.air_refs();
    let views: Vec<StarkProofView<F, E, ()>> = vm_proof
        .proof
        .proofs
        .iter()
        .map(StarkProofView::Owned)
        .collect();

    // Part count per table, straight from the AIR — the quantity degree moves.
    let parts: Vec<usize> = air_refs
        .iter()
        .zip(views.iter())
        .map(|(air, v)| {
            let n = v.trace_length();
            air.composition_poly_degree_bound(n) / n
        })
        .collect();
    let total_parts: usize = parts.iter().sum();

    let mut replay = DefaultTranscript::<E>::new(&[]);
    let expected_bus_balance = crate::compute_expected_commit_bus_balance_view(
        &air_refs,
        &views,
        &vm_proof.public_output,
        0,
        &mut replay,
    )
    .expect("bus balance");

    guard.assert_exclusive("before reset");
    #[cfg(feature = "hash-count")]
    crypto::hash_count::reset();

    let ok = Verifier::multi_verify_views(
        &air_refs,
        &views,
        &mut DefaultTranscript::<E>::new(&[]),
        &expected_bus_balance,
    );
    assert!(ok, "[{label}] verification must succeed");

    #[cfg(feature = "hash-count")]
    {
        let (leaf_hashes, leaf_bytes, parent_hashes, perms) = crypto::hash_count::read();
        // The load-bearing check: nothing else hashed during the window.
        guard.assert_exclusive("after read");
        println!(
            "DEGREECOST label={label} vm_max_degree={} blowup={} queries={} tables={} \
             parts_per_table={} total_parts={} \
             verifier_leaf_hashes={leaf_hashes} verifier_leaf_bytes={leaf_bytes} \
             verifier_parent_hashes={parent_hashes} \
             verifier_total_hashes={} verifier_permutations={perms}",
            crate::VM_MAX_DEGREE,
            opts.blowup_factor,
            opts.fri_number_of_queries,
            air_refs.len(),
            parts[0],
            total_parts,
            leaf_hashes + parent_hashes,
        );
    }
    #[cfg(not(feature = "hash-count"))]
    println!(
        "DEGREECOST label={label} vm_max_degree={} blowup={} queries={} total_parts={total_parts} \
         (build with --features hash-count for hash volumes)",
        crate::VM_MAX_DEGREE,
        opts.blowup_factor,
        opts.fri_number_of_queries,
    );
}

/// Prover-side arm: prove only, report wall time and committed volume.
///
/// One arm per process — peak RSS is a high-water mark, so two configurations
/// in one process would only ever measure the larger. Run the test binary
/// directly under `/usr/bin/time -v` and read "Maximum resident set size".
///
/// Cell convention, stated on every number: **base-field-equivalent
/// `main + 3*aux`**, an extension element being three base felts. Composition
/// parts are extension-valued too, so each part counts as 3 per LDE point —
/// which is exactly why one part costs the same committed volume as one aux
/// column, or three main columns.
fn measure_prove_arm(elf_bytes: &[u8], label: &str) {
    let opts = arm_options();
    let elf = Elf::load(elf_bytes).expect("ELF load");
    let executor = Executor::new(&elf, Vec::new()).expect("executor");
    let result = executor.run().expect("execution");
    let max_rows = MaxRowsConfig::default();
    let mut traces =
        Traces::from_elf_and_logs_minimal(&elf, &result.logs, &max_rows, &[]).unwrap();
    let table_counts = traces.table_counts();
    let airs = build_airs(&elf, &opts, &traces.page_configs, &table_counts);

    let pairs = airs.air_trace_pairs(&mut traces);

    // Committed volume, before proving (the trace is consumed by the prove).
    let mut trace_cells = 0usize;
    let mut part_cells = 0usize;
    let mut total_parts = 0usize;
    for (air, trace, _pub) in pairs.iter() {
        let (main_w, aux_w) = air.trace_layout();
        let n = trace.num_rows();
        trace_cells += (main_w + 3 * aux_w) * n;
        let parts = air.composition_poly_degree_bound(n) / n;
        total_parts += parts;
        // Each part is one extension column over the LDE domain.
        part_cells += 3 * parts * n * opts.blowup_factor as usize;
    }

    let t0 = std::time::Instant::now();
    let proof = multi_prove_ram(pairs, &mut DefaultTranscript::<E>::new(&[])).expect("prove");
    let prove_secs = t0.elapsed().as_secs_f64();
    std::hint::black_box(&proof);

    println!(
        "DEGREEPROVE label={label} vm_max_degree={} blowup={} queries={} \
         force_generic={} total_parts={total_parts} \
         trace_cells_main_plus_3aux={trace_cells} composition_part_cells_lde={part_cells} \
         prove_secs={prove_secs:.4}",
        crate::VM_MAX_DEGREE,
        opts.blowup_factor,
        opts.fri_number_of_queries,
        std::env::var("LVM_FORCE_GENERIC_PARTS").unwrap_or_else(|_| "0".into()),
    );
}

#[test]
#[ignore = "degree-lane experiment; run explicitly"]
fn degree_prove_volume() {
    let _guard = ArmGuard::enter();
    let (name, elf) = arm_elf();
    let reps: usize = std::env::var("LVM_DEGREE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    for _ in 0..reps {
        measure_prove_arm(&elf, &name);
    }
}

#[test]
#[ignore = "degree-lane experiment; run explicitly"]
fn degree_cost_verifier_hashes() {
    let guard = ArmGuard::enter();
    guard.assert_exclusive("at entry");
    let (name, elf) = arm_elf();
    measure_arm(&elf, &name, &guard);
}

/// Full production prove path (`prove_with_options`), which emits the
/// `instruments` per-stage report. That report is the right instrument for the
/// fast-path cliff: it times `decompose_and_extend_d2` / the generic fallback
/// on its own, instead of hunting for the difference inside a noisy end-to-end
/// wall clock.
///
/// One arm per process. Run under `/usr/bin/time -v` for peak RSS.
#[test]
#[ignore = "degree-lane experiment; run explicitly"]
fn degree_prove_instrumented() {
    let _guard = ArmGuard::enter();
    let (name, elf) = arm_elf();
    let opts = arm_options();
    let max_rows = MaxRowsConfig::default();

    let t0 = std::time::Instant::now();
    let proof = crate::prove_with_options(&elf, &opts, &max_rows).expect("prove");
    let prove_secs = t0.elapsed().as_secs_f64();
    std::hint::black_box(&proof);

    println!(
        "DEGREEPROVEI label={name} vm_max_degree={} blowup={} queries={} force_generic={} \
         total_secs={prove_secs:.4}",
        crate::VM_MAX_DEGREE,
        opts.blowup_factor,
        opts.fri_number_of_queries,
        std::env::var("LVM_FORCE_GENERIC_PARTS").unwrap_or_else(|_| "0".into()),
    );
}
