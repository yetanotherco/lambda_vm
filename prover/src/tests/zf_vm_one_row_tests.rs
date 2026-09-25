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
//! - Lane I-FIX-S2's regression: the Phase-A replay that recovers `z`, `α` for
//!   the expected bus balances absorbs each preprocessed table's root AT ITS
//!   LEAF LAYOUT — an LFM proof at the wrap's options under `one_row = auto`
//!   (mixed layouts, one-row preprocessed chips, published words) and a VM
//!   proof with public output at `one_row = 1`.

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

/// ★ REGRESSION (lane I-FIX-S2): one-row PREPROCESSED tables and the Phase-A
/// replay. The prover absorbs each preprocessed table's root OF ITS LEAF
/// LAYOUT before sampling the shared LogUp `z`, `α`; the verify paths recover
/// `z`, `α` with `crate::replay_transcript_phase_a_view`, which absorbed the
/// ROW-PAIR root unconditionally. For a one-row preprocessed table the replay
/// then diverges, and every expected balance that depends on `z`, `α` is
/// wrong: an honest proof is rejected. The balance depends on them only when
/// something is published — the LFM public words, the VM commit bus — which
/// is why a VM proof without public output (`test_mul_8`) verified anyway.
///
/// At the wrap's options (blowup 4, terminal 2^8, 128-bit queries) under
/// `one_row = auto`, as the block tree proves its LFM wraps: layouts MIX
/// within one proof and at least one preprocessed chip goes one-row (asserted,
/// so this keeps exercising the bug). Verified through
/// `verify_against_artifacts` — the call the tree harness makes before
/// harvesting a child (`per_table_aggregator_tests::real_child_timed`) — and
/// through `lfm_verify`.
#[test]
fn an_lfm_proof_at_the_wrap_options_round_trips_at_one_row_auto() {
    use crate::lfm::proof::{lfm_prove, lfm_verify, verify_against_artifacts};
    use crate::lfm::registry::{LfmProgramKind, build_artifacts};
    use crate::tables::types::FE;
    use stark::leaf_layout::table_leaf_layout;
    let mut o = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(4).unwrap();
    o.fri_final_poly_log_degree = 8;
    o.format.one_row = OneRowMode::Auto;
    let program = LfmProgramKind::TrivialV0.program();
    let artifacts = build_artifacts(&program, &o);
    let arenas: Vec<Vec<crate::lfm::word::LfmWord>> = vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ];
    let proved = lfm_prove(&program, &artifacts, &arenas, &o).expect("auto LFM prove");
    assert!(
        !proved.public_words.is_empty(),
        "the balance must depend on z and alpha: the program publishes"
    );

    let mut airs = crate::lfm::airs::LfmAirs::new_chunked(
        &artifacts.roots,
        &artifacts.blake3_chunk_roots,
        &o,
        artifacts.keccak_rnd_chunks,
        artifacts.hasher,
        artifacts.chip_set,
    );
    airs = airs.with_one_row_roots(artifacts.one_row_roots.as_ref().expect("built"));
    let refs = airs.air_refs();
    let (mut prep_rows, mut rows, mut pairs) = (0usize, 0usize, 0usize);
    for (air, p) in refs.iter().zip(&proved.proof.proofs) {
        let layout = table_leaf_layout(*air, p.trace_length);
        println!(
            "ZF FIX-S2 layout {:<12} 2^{:<2} {layout:?}",
            air.name(),
            p.trace_length.trailing_zeros()
        );
        if layout.is_one_row() {
            rows += 1;
            prep_rows += usize::from(air.is_preprocessed());
        } else {
            pairs += 1;
        }
    }
    println!(
        "ZF FIX-S2 LFM one_row=auto: {rows} one-row, {pairs} row-pair, {prep_rows} one-row preprocessed"
    );
    assert!(
        prep_rows >= 1 && pairs >= 1,
        "the fixture must mix layouts with a one-row preprocessed chip \
         ({prep_rows} one-row preprocessed, {pairs} row-pair)"
    );

    assert!(
        verify_against_artifacts(&artifacts, &proved.proof, &proved.public_words, &o),
        "an honest one-row-auto LFM proof must verify (the tree harness's call)"
    );
    assert!(
        lfm_verify(
            LfmProgramKind::TrivialV0,
            &proved.proof,
            &proved.public_words,
            &o
        )
        .expect("built at run time under one row"),
        "an honest one-row-auto LFM proof must verify through lfm_verify"
    );
    // Still bound to the claimed words: one moved public word rejects.
    let mut wrong = proved.public_words.clone();
    wrong[0].1[0] += FE::from(1u64);
    assert!(
        !verify_against_artifacts(&artifacts, &proved.proof, &wrong, &o),
        "a moved public word must be rejected"
    );
}

/// The VM half of the same regression: a VM proof WITH public output (the
/// commit bus's expected balance depends on the replayed `z`, `α`) at
/// `one_row = 1`, where every preprocessed VM table (BITWISE, DECODE, the
/// pages, REGISTER) is one-row.
#[test]
fn a_vm_proof_with_public_output_round_trips_at_one_row() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("test_commit_4");
    let one_row = opts(4, OneRowMode::On, FriMode::Pair);
    let vm_proof = crate::prove_with_options(&elf_bytes, &one_row, &Default::default())
        .expect("test_commit_4 must prove at one_row = 1");
    assert_eq!(vm_proof.public_output, vec![0xAA, 0xBB, 0xCC, 0xDD]);
    assert!(
        crate::verify_with_options(&vm_proof, &elf_bytes, &one_row, None, None)
            .expect("honest verify must not error"),
        "an honest one-row VM proof with public output must verify"
    );
    let mut wrong = vm_proof.clone();
    wrong.public_output[0] ^= 1;
    assert!(
        !crate::verify_with_options(&wrong, &elf_bytes, &one_row, None, None).unwrap_or(false),
        "a moved public output byte must be rejected"
    );
}
