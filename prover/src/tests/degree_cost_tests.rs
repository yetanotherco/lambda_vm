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
fn measure_arm(elf_bytes: &[u8], label: &str) {
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

#[test]
#[ignore = "degree-lane experiment; run explicitly"]
fn degree_cost_verifier_hashes() {
    // `LVM_DEGREE_ELF` picks the workload; taller traces mean deeper Merkle
    // trees, which shifts the balance between parts (per query) and path
    // hashing (per query per level).
    let name = std::env::var("LVM_DEGREE_ELF").unwrap_or_else(|_| "sub".to_string());
    let elf = crate::test_utils::asm_elf_bytes(&name);
    measure_arm(&elf, &name);
}
