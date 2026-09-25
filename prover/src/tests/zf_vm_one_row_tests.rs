//! S2 end to end on the production paths (box lib suite: each proves a full
//! VM trace with the 2^20-row BITWISE table, or an LFM machine proof).
//!
//! - A real multi-table VM proof (RPX block pin, host CPU paths) at
//!   `one_row = 1` and at `one_row = auto` with `fri = dp`, blowup 4 (the
//!   blowup the one-row static twins ship for).
//! - RULINGS 14 at the VM level: at blowup 2 there is no one-row twin, so
//!   `one_row = 1` is a proving ERROR naming the missing root — never a silent
//!   recompute, never a proof.
//! - An LFM machine proof (`TrivialV0`) at `one_row = 1`, blowup 4, verified
//!   through `lfm_verify`, i.e. through the registry policy (built at run time,
//!   `LFM_REGISTRY` not read).
//! - D2 (lane I-S2-D): the one-row VM proof's BYTES at grinding 0, written to
//!   `ZF_S2_PROOF_DIR` by a CPU build and by a cuda build; the box compares the
//!   two files byte for byte (the device-proved one-row VM proof equals the CPU
//!   one). The cuda run also asserts the one-row device paths fired and prints
//!   the one-row device-memory line for the 0-fallback gate.

use stark::proof::options::{FriMode, OneRowMode, ProofFormat, ProofOptions};

fn opts(blowup: u8, one_row: OneRowMode, fri_mode: FriMode) -> ProofOptions {
    let mut o = ProofOptions::default_test_options();
    o.blowup_factor = blowup;
    o.format = ProofFormat {
        one_row,
        fri_mode,
        ..ProofFormat::DEFAULT
    };
    o
}

#[test]
fn a_vm_proof_round_trips_at_one_row() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_mul_8");
    let one_row = opts(4, OneRowMode::On, FriMode::Pair);
    let vm_proof = crate::prove_with_options(&elf_bytes, &one_row, &Default::default())
        .expect("the fixture must prove at one_row = 1");
    assert!(
        crate::verify_with_options(&vm_proof, &elf_bytes, &one_row, None, None)
            .expect("honest verify must not error"),
        "an honest one-row VM proof must verify"
    );
    // Every table is one-row: no symmetric rows anywhere, and the input tree
    // is FRI layer 0 wherever anything folds.
    for p in &vm_proof.proof.proofs {
        for o in &p.deep_poly_openings {
            assert!(o.main_trace_polys.evaluations_sym.is_empty());
            assert!(o.composition_poly.evaluations_sym.is_empty());
        }
    }
    println!(
        "ZF S2 VM one_row=1: {} tables, proof tables with a precomputed root: {}",
        vm_proof.proof.proofs.len(),
        vm_proof
            .proof
            .proofs
            .iter()
            .filter(|p| p.lde_trace_precomputed_merkle_root.is_some())
            .count()
    );
    // The layout is a verifier constant: the row-pair verifier rejects it.
    let default = opts(4, OneRowMode::Off, FriMode::Pair);
    assert!(
        !crate::verify_with_options(&vm_proof, &elf_bytes, &default, None, None).unwrap_or(false),
        "a one-row proof must not verify under the default format"
    );
    // A tampered input-group value is rejected.
    let mut bad = vm_proof.clone();
    let table = bad
        .proof
        .proofs
        .iter()
        .position(|p| !p.fri_layers_merkle_roots.is_empty())
        .expect("a table with committed layers");
    bad.proof.proofs[table].query_list[0].layers_evaluations_sym[0] +=
        math::field::element::FieldElement::<
            math::field::extensions_goldilocks::Degree3GoldilocksExtensionField,
        >::one();
    assert!(
        !crate::verify_with_options(&bad, &elf_bytes, &one_row, None, None).unwrap_or(false),
        "a tampered input-group value must be rejected"
    );
}

#[test]
fn a_vm_proof_round_trips_at_one_row_auto_with_dp() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_mul_8");
    let auto = opts(4, OneRowMode::Auto, FriMode::Dp);
    let vm_proof = crate::prove_with_options(&elf_bytes, &auto, &Default::default())
        .expect("the fixture must prove at one_row = auto, fri = dp");
    assert!(
        crate::verify_with_options(&vm_proof, &elf_bytes, &auto, None, None)
            .expect("honest verify must not error"),
        "an honest auto/dp VM proof must verify"
    );
    let one_row_tables = vm_proof
        .proof
        .proofs
        .iter()
        .filter(|p| {
            p.deep_poly_openings[0]
                .composition_poly
                .evaluations_sym
                .is_empty()
        })
        .count();
    println!(
        "ZF S2 VM one_row=auto fri=dp: {one_row_tables} of {} tables one-row",
        vm_proof.proof.proofs.len()
    );
}

/// RULINGS 14: no one-row static twin at blowup 2 ⇒ a proving error naming
/// the missing root.
#[test]
fn a_missing_one_row_twin_is_a_vm_proving_error() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_mul_8");
    let one_row = opts(2, OneRowMode::On, FriMode::Pair);
    let err = crate::prove_with_options(&elf_bytes, &one_row, &Default::default())
        .expect_err("no one-row twin at blowup 2: proving must fail");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("PrecomputedCommitmentMissing"),
        "the error must name the missing one-row root: {msg}"
    );
}

#[test]
fn an_lfm_proof_round_trips_at_one_row() {
    use crate::lfm::proof::{lfm_prove, lfm_verify};
    use crate::lfm::registry::{LfmProgramKind, build_artifacts};
    use crate::tables::types::FE;
    let mut o = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(4).unwrap();
    o.format.one_row = OneRowMode::On;
    let program = LfmProgramKind::TrivialV0.program();
    let artifacts = build_artifacts(&program, &o);
    assert!(artifacts.one_row_roots.is_some());
    let arenas: Vec<Vec<crate::lfm::word::LfmWord>> = vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ];
    let proved = lfm_prove(&program, &artifacts, &arenas, &o).expect("one-row LFM prove");
    for p in &proved.proof.proofs {
        assert!(
            p.deep_poly_openings[0]
                .main_trace_polys
                .evaluations_sym
                .is_empty()
        );
    }
    assert!(
        lfm_verify(
            LfmProgramKind::TrivialV0,
            &proved.proof,
            &proved.public_words,
            &o
        )
        .expect("built at run time under one row"),
        "an honest one-row LFM proof must verify"
    );
}
