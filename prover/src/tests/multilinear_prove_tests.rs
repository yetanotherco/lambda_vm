//! The VM proved and verified end to end through the multilinear path.
//!
//! Unlike `multilinear_table_tests`, which drives the argument by hand, these
//! go through [`multilinear_prove::prove_with_options`] and
//! [`multilinear_prove::verify_with_options`]: ELF in, proof out, verdict back,
//! with the verifier holding nothing but the ELF and the proof.

use crate::multilinear_prove::{self, MultilinearVmProof};
use crate::test_utils::asm_elf_bytes;
use stark::proof::options::ProofOptions;

use crate::tables::MaxRowsConfig;

fn prove(elf: &[u8]) -> MultilinearVmProof {
    multilinear_prove::prove_with_options(
        elf,
        &ProofOptions::default_test_options(),
        &MaxRowsConfig::default(),
    )
    .expect("prove")
}

fn verify(proof: &MultilinearVmProof, elf: &[u8]) -> bool {
    multilinear_prove::verify_with_options(proof, elf, &ProofOptions::default_test_options())
        .expect("verify")
}

/// The milestone this whole port is for: a program proved and verified by the
/// VM's own entry points, with WHIR underneath.
#[test]
fn a_program_proves_and_verifies() {
    let elf = asm_elf_bytes("sub");
    let proof = prove(&elf);
    assert!(verify(&proof, &elf));
}

/// The whole 64-bit instruction set, which lights up the tables a
/// two-instruction program never reaches.
#[test]
fn the_whole_instruction_set_proves_and_verifies() {
    let elf = asm_elf_bytes("all_instructions_64");
    let proof = prove(&elf);
    assert!(verify(&proof, &elf));
}

/// A proof is only a proof if it can leave the process. Round-trips through
/// rkyv, the format the univariate `VmProof` uses, and verifies the
/// **deserialized** one.
#[test]
fn a_proof_survives_serialization() {
    let elf = asm_elf_bytes("sub");
    let proof = prove(&elf);

    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("rkyv round trip");
    let back: MultilinearVmProof =
        rkyv::from_bytes::<_, rkyv::rancor::Error>(&bytes).expect("rkyv round trip");

    assert!(verify(&back, &elf));
}

/// The proof is bound to the program: the same proof against a different ELF is
/// rejected, because the digest moves every challenge.
#[test]
fn a_proof_does_not_verify_against_another_program() {
    let elf = asm_elf_bytes("sub");
    let proof = prove(&elf);
    assert!(!verify(&proof, &asm_elf_bytes("all_instructions_64")));
}

/// And to its public output: the COMMIT bus's counterparty is the statement, so
/// restating the output leaves the balance off by exactly the difference.
#[test]
fn a_tampered_public_output_is_rejected() {
    let elf = asm_elf_bytes("sub");
    let mut proof = prove(&elf);
    proof.public_output.push(0xff);
    assert!(!verify(&proof, &elf));
}

/// A restated table height is rejected too: it is absorbed into the transcript
/// before anything is drawn.
#[test]
fn a_restated_table_height_is_rejected() {
    let elf = asm_elf_bytes("sub");
    let mut proof = prove(&elf);
    proof.table_num_vars[0] += 1;
    assert!(!verify(&proof, &elf));
}

/// The preprocessed tables are bound to the program, and a forged one is
/// rejected — the property the proof would otherwise be missing entirely.
///
/// Forging goes through the AIR the *prover* builds: the honest verifier
/// rebuilds the real columns from the ELF, so a prover that committed anything
/// else opens to the wrong value and is caught. Driven at the table level,
/// where a single AIR can be swapped.
#[test]
fn a_forged_preprocessed_column_is_rejected() {
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::element::FieldElement;
    use multilinear::mle::Mle;
    use multilinear::whir_chain::{ChainConfig, GrindBits};
    use stark::multilinear_air::Uniforms;
    use stark::multilinear_table::{self, CommittedTable, CommittedTables, TableLayout};
    use stark::traits::AIR;

    use crate::tables::keccak_rc;
    use crate::test_utils::{E, F, create_keccak_rc_air};

    let config = ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::default(),
    };
    let air = create_keccak_rc_air(&ProofOptions::default_test_options());
    let width = air.trace_layout().0;

    // The honest preprocessed columns, padded out to the table's width with
    // zeroed multiplicity columns — a trace whose preprocessed half is real.
    let honest = keccak_rc::preprocessed_columns();
    let rows = honest[0].len();
    let mut columns = honest.clone();
    columns.resize(width, vec![FieldElement::<F>::zero(); rows]);

    let num_vars = rows.trailing_zeros() as usize;
    let layout = || {
        TableLayout::<F, E>::new(
            air.constraint_program(),
            air.constraints_meta(),
            air.bus_interactions(),
            width,
            num_vars,
            Uniforms::default(),
        )
        .unwrap()
    };
    let prove = |columns: Vec<Vec<FieldElement<F>>>| {
        let table =
            CommittedTable::from_layout(layout(), |col| columns[col as usize].clone()).unwrap();
        let committed = CommittedTables::commit(vec![table], &config).unwrap();
        let mut transcript = DefaultTranscript::<E>::new(b"forged");
        let proof = multilinear_table::multi_prove(&committed, &config, &mut transcript).unwrap();
        (
            proof,
            committed.layout().clone(),
            committed.domain().clone(),
        )
    };

    // The verifier's own copy: always the real columns, never the prover's.
    let expected: Vec<Mle<F>> = honest
        .iter()
        .map(|c| Mle::new(c.clone()).unwrap())
        .collect();
    let verifier_layout = layout();
    let statement = verifier_layout.statement_with_preprocessed(&expected);

    let verify = |(proof, stacked, domain): (
        multilinear_table::MultiProof<F, E>,
        multilinear::stacking::StackedLayout,
        multilinear::whir::Domain<F>,
    )| {
        let mut transcript = DefaultTranscript::<E>::new(b"forged");
        multilinear_table::multi_verify(
            &proof,
            &[statement],
            &stacked,
            &domain,
            &FieldElement::<E>::zero(),
            &config,
            &mut transcript,
        )
        .is_ok()
    };

    assert!(verify(prove(columns.clone())), "the honest table verifies");

    let mut forged = columns;
    forged[0][3] += FieldElement::<F>::one();
    assert!(
        !verify(prove(forged)),
        "a forged preprocessed column must be rejected"
    );
}

/// And the tables themselves are checked, not just the metadata around them:
/// moving one table's bus output breaks the balance the statement owes.
#[test]
fn a_tampered_table_proof_is_rejected() {
    use math::field::element::FieldElement;

    let elf = asm_elf_bytes("sub");
    let mut proof = prove(&elf);
    proof.proof.tables[0].bus_output.0 +=
        FieldElement::<math::field::extensions_goldilocks::Degree3GoldilocksExtensionField>::one();
    assert!(!verify(&proof, &elf));
}

/// A proof missing a table is rejected before any of it is checked: the table
/// count is a function of the statement, not of the proof.
#[test]
fn a_proof_missing_a_table_is_rejected() {
    let elf = asm_elf_bytes("sub");
    let mut proof = prove(&elf);
    proof.proof.tables.pop();
    proof.table_num_vars.pop();
    assert!(
        multilinear_prove::verify_with_options(&proof, &elf, &ProofOptions::default_test_options())
            .is_err()
    );
}
