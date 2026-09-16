//! The batched FRI, pinned at the one size where it has to agree with the
//! unbatched one.

use crate::tables::MaxRowsConfig;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use executor::elf::Elf;
use stark::prover::IsStarkProver;

/// A batch of one must reproduce the proof's FRI exactly.
///
/// `Σ αᵏ·deepₖ` over a single member is `deepₖ`, so the batched path and the
/// per-table one fold the same codeword over the same domain. If their layer
/// roots differ, the difference is in the codeword or in the domain — which is
/// the whole substance of the batching, and worth catching before any group has
/// more than one member in it.
#[test]
fn a_batch_of_one_matches_the_unbatched_fri() {
    type P = stark::prover::Prover<GoldilocksField, GoldilocksExtension, ()>;

    let elf_bytes = crate::test_utils::asm_elf_bytes("fib_iterative_160k");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let max_rows = MaxRowsConfig {
        cpu: 1 << 15,
        memw: 1 << 10,
        load: 1 << 10,
        branch: 1 << 12,
        ..Default::default()
    };
    let proof_options = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
        .expect("blowup 2 is valid");

    let vm_proof = crate::prove_with_options_and_inputs(&elf_bytes, &[], &proof_options, &max_rows)
        .expect("ordinary prove");

    let committed = crate::commit_phase::run_to_end(&elf, &[], &max_rows, &proof_options)
        .expect("commit phase");
    let challenge = crate::challenge_phase::run(&committed, &elf, &elf_bytes, &proof_options)
        .expect("challenge phase");
    drop(committed);

    let page_configs = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
        &elf,
        &vm_proof.runtime_page_ranges,
        vm_proof.num_private_input_pages,
        vm_proof.proof.proofs.len(),
    )
    .expect("page configs");
    let airs = crate::VmAirs::new(
        &elf,
        &proof_options,
        false,
        &page_configs,
        &vm_proof.table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let mut resident = crate::logup_phase::resident_tables(&elf, &[], &max_rows).expect("resident");

    // BITWISE is table 0 and the largest resident one, so it exercises a real
    // domain rather than a one-row corner.
    let idx = 0usize;
    let n = challenge.roots.len();
    let mut fork = crate::logup_phase::fork_for(&challenge, idx, n);
    let deep = <P as IsStarkProver<_, _, _>>::deep_for_table(
        airs.bitwise.as_ref(),
        &(),
        &mut resident.bitwise,
        &challenge.challenges,
        &mut fork,
    )
    .expect("deep");

    let one = math::field::element::FieldElement::<GoldilocksExtension>::one();
    let roots = <P as IsStarkProver<_, _, _>>::batch_fri(
        airs.bitwise.as_ref(),
        vec![deep],
        &one,
        &mut fork,
    )
    .expect("batched fri");

    assert_eq!(
        roots, vm_proof.proof.proofs[idx].fri_layers_merkle_roots,
        "a batch of one folded to a different FRI than the proof carries"
    );
}
