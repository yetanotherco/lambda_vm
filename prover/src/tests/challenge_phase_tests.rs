//! The Challenge phase must reconstruct the ordinary prover's transcript.

use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;
use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use executor::elf::Elf;
use stark::proof::view::MultiProofView;

/// Approach 1's second pass must draw the very challenge the tables were built
/// against.
///
/// This is the whole claim of the single-challenge design: the roots the Commit
/// phase produced, absorbed in AIR order on top of the bound statement,
/// reproduce the transcript the production prover builds. Anything that shifts
/// the order, omits a preprocessed table's precomputed root, or commits a table
/// differently moves `(z, alpha)`, and a challenge that differs from the
/// prover's is a proof that does not verify.
///
/// The roots are checked in position first, so a mismatch names the table
/// rather than surfacing only as a different field element at the end. The
/// challenge itself is then compared against the *verifier's* replay, rebuilt
/// from the proof's own statement fields, so the comparison does not run
/// through `challenge_phase` twice.
#[test]
fn challenge_matches_the_ordinary_prover() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("fib_iterative_160k");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    // Small enough that the walk closes several chunks and leaves tails: the
    // order only gets exercised when a group has more than one member.
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

    assert_eq!(
        challenge.roots.len(),
        vm_proof.proof.proofs.len(),
        "the Commit phase accounted for a different number of tables than the proof has"
    );
    let mut preprocessed = 0usize;
    for (idx, (got, want)) in challenge
        .roots
        .iter()
        .zip(vm_proof.proof.proofs.iter())
        .enumerate()
    {
        assert_eq!(
            got.main, want.lde_trace_main_merkle_root,
            "table {idx}: committed under a different root than the proof carries"
        );
        if got.precomputed.is_some() {
            preprocessed += 1;
        }
    }
    assert!(
        preprocessed >= 4,
        "the fixture must cover the preprocessed tables (BITWISE, DECODE, KECCAK_RC, \
         REGISTER and the pages); saw {preprocessed}"
    );

    // The verifier's path: statement from the proof, AIRs reconstructed the way
    // verification reconstructs them.
    let page_configs = Traces::page_configs_from_elf_and_runtime(
        &elf,
        &vm_proof.runtime_page_ranges,
        vm_proof.num_private_input_pages,
        vm_proof.proof.proofs.len(),
    )
    .expect("page configs");
    let verifier_airs = crate::VmAirs::new(
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
    let mut transcript = DefaultTranscript::new(&[]);
    crate::statement::absorb_statement(
        &mut transcript,
        crate::statement::StatementKind::Monolithic,
        &elf_bytes,
        &vm_proof.public_output,
        &vm_proof.table_counts,
        vm_proof.num_private_input_pages,
        &vm_proof.runtime_page_ranges,
        proof_options.fri_final_poly_log_degree,
    );
    let (z, alpha) = crate::replay_transcript_phase_a_view(
        &verifier_airs.air_refs(),
        MultiProofView::Owned(&vm_proof.proof),
        &mut transcript,
    );

    assert_eq!(
        challenge.challenges,
        vec![z, alpha],
        "the Challenge phase sampled a different challenge from the same execution"
    );
}

/// The LogUp pass must commit the auxiliary columns the proof carries.
///
/// The pass rebuilds every table from scratch — the ones the Commit phase
/// dropped no longer exist — and builds their auxiliary columns against the
/// challenge that phase produced. Three separate things have to hold for the
/// root to land: the rebuild is byte-identical to what was committed, the
/// challenge is the prover's, and the table is in the slot the proof expects.
/// Any one of them failing changes the bus the verifier checks, so all three
/// are pinned here at once against a real proof.
#[test]
fn logup_matches_the_ordinary_prover() {
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
    let logup = crate::logup_phase::run(&elf, &[], &max_rows, &proof_options, &challenge)
        .expect("logup phase");

    assert_eq!(
        logup.tables.len(),
        vm_proof.proof.proofs.len(),
        "the LogUp pass accounted for a different number of tables than the proof has"
    );
    let mut with_aux = 0usize;
    for (idx, (got, want)) in logup
        .tables
        .iter()
        .zip(vm_proof.proof.proofs.iter())
        .enumerate()
    {
        assert_eq!(
            got.aux_root, want.lde_trace_aux_merkle_root,
            "table {idx}: auxiliary trace committed under a different root than the proof carries"
        );
        assert_eq!(
            got.main_roots.main, want.lde_trace_main_merkle_root,
            "table {idx}: the rebuild produced a different main trace than the Commit phase did"
        );
        assert_eq!(
            got.composition_poly_root, want.composition_poly_root,
            "table {idx}: composition polynomial committed under a different root"
        );
        assert_eq!(
            got.composition_poly_parts_ood_evaluation, want.composition_poly_parts_ood_evaluation,
            "table {idx}: composition parts evaluated at a different out-of-domain point"
        );
        for (label, got, want) in [
            ("z", &got.trace_ood_evaluations, &want.trace_ood_evaluations),
            (
                "g*z",
                &got.trace_ood_next_evaluations,
                &want.trace_ood_next_evaluations,
            ),
        ] {
            assert_eq!(
                (got.width, got.columns()),
                (want.width, want.columns()),
                "table {idx}: different out-of-domain trace evaluations at {label}"
            );
        }
        if got.aux_root.is_some() {
            with_aux += 1;
        }
    }
    assert!(
        with_aux > 20,
        "the fixture must cover the tables that carry a bus; saw {with_aux}"
    );
}
