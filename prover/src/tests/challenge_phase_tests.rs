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
    // Small enough that MEMW and MEMW_A chunks close mid-walk: their timestamp
    // checks are LT rows, and a walk that retires a chunk must still hand LT
    // the rows the chunk owes it.
    let max_rows = MaxRowsConfig {
        cpu: 1 << 15,
        memw: 1 << 10,
        memw_aligned: 1 << 10,
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
        format!("{:?}", challenge.order.counts()),
        format!("{:?}", vm_proof.table_counts),
        "the pass laid the tables out differently from the ordinary prover"
    );
    assert_eq!(
        logup.tables.len(),
        vm_proof.proof.proofs.len(),
        "the pass accounted for a different number of tables than the proof has"
    );
    for (idx, (got, want)) in logup
        .tables
        .iter()
        .zip(vm_proof.proof.proofs.iter())
        .enumerate()
    {
        assert_eq!(
            got.lde_trace_aux_merkle_root, want.lde_trace_aux_merkle_root,
            "table {idx}: auxiliary trace committed under a different root"
        );
        assert_eq!(
            got.composition_poly_root, want.composition_poly_root,
            "table {idx}: composition polynomial committed under a different root"
        );
        assert_eq!(
            got.fri_layers_merkle_roots, want.fri_layers_merkle_roots,
            "table {idx}: a different FRI commitment"
        );
        assert_eq!(
            got.fri_final_poly_coeffs, want.fri_final_poly_coeffs,
            "table {idx}: a different FRI final polynomial"
        );
    }

    // The decisive one: the pass's own proof, verified. Everything above says
    // it matches the ordinary prover piece by piece; this says the assembled
    // whole is a proof.
    let rebuilt = crate::logup_phase::assemble_vm_proof(logup, &challenge);
    assert!(
        crate::verify(&rebuilt, &elf_bytes).expect("verify"),
        "the proof the pass assembled does not verify"
    );
}

/// Many chunks of every kind, on a program that works memory. The
/// small program above has one chunk per kind, so it cannot tell per-chunk
/// accounting from whole-table accounting; ethrex could, and did not verify.
#[test]
fn a1_verifies_with_many_chunks() {
    let elf_bytes = {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = root.parent().expect("workspace root").join(format!(
            "executor/program_artifacts/rust/{}.elf",
            std::env::var("A1_DIAG_ELF").unwrap_or_else(|_| "vector".into())
        ));
        std::fs::read(&path).unwrap_or_else(|_| panic!("Failed to read ELF: {}", path.display()))
    };
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let max_rows = MaxRowsConfig {
        cpu: 1 << 11,
        memw: 1 << 9,
        memw_aligned: 1 << 9,
        dvrm: 1 << 9,
        mul: 1 << 9,
        lt: 1 << 9,
        shift: 1 << 9,
        load: 1 << 9,
        branch: 1 << 9,
        memw_register: 1 << 9,
        eq: 1 << 9,
        bytewise: 1 << 9,
        store: 1 << 9,
        cpu32: 1 << 9,
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
    let rebuilt = crate::logup_phase::assemble_vm_proof(logup, &challenge);

    eprintln!(
        "tables: pass {} vs ordinary {}; counts pass {:?} vs ordinary {:?}",
        rebuilt.proof.proofs.len(),
        vm_proof.proof.proofs.len(),
        rebuilt.table_counts,
        vm_proof.table_counts
    );
    let differing: Vec<usize> = rebuilt
        .proof
        .proofs
        .iter()
        .zip(vm_proof.proof.proofs.iter())
        .enumerate()
        .filter(|(_, (a, b))| a.lde_trace_main_merkle_root != b.lde_trace_main_merkle_root)
        .map(|(i, _)| i)
        .collect();
    eprintln!("tables whose main root differs from the ordinary prover's: {differing:?}");
    assert!(
        crate::verify(&rebuilt, &elf_bytes).expect("verify"),
        "the proof the pass assembled does not verify"
    );
}

/// Diagnostic: where A1's BITWISE multiplicities depart from the ordinary build's.
#[test]
#[allow(clippy::needless_range_loop)]
fn bitwise_multiplicities_match_the_ordinary_build() {
    let elf_bytes = {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let name = std::env::var("A1_DIAG_ELF").unwrap_or_else(|_| "vector".into());
        std::fs::read(
            root.parent()
                .unwrap()
                .join(format!("executor/program_artifacts/rust/{name}.elf")),
        )
        .expect("ELF")
    };
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let max_rows = if std::env::var("A1_DIAG_DEFAULT_ROWS").is_ok() {
        MaxRowsConfig::default()
    } else {
        MaxRowsConfig {
            cpu: 1 << 11,
            memw: 1 << 9,
            memw_aligned: 1 << 9,
            dvrm: 1 << 9,
            mul: 1 << 9,
            lt: 1 << 9,
            shift: 1 << 9,
            load: 1 << 9,
            branch: 1 << 9,
            memw_register: 1 << 9,
            eq: 1 << 9,
            bytewise: 1 << 9,
            store: 1 << 9,
            cpu32: 1 << 9,
        }
    };
    let input = std::env::var("A1_DIAG_INPUT")
        .ok()
        .map(|path| std::fs::read(path).expect("private input"))
        .unwrap_or_default();
    let ordinary = crate::commit_phase::build_resident(&elf, &input, &max_rows).expect("ordinary");
    let walked = crate::logup_phase::walk_only(&elf, &input, &max_rows).expect("walk");
    eprintln!(
        "cpu rows: {}",
        ordinary.cpus.iter().map(|t| t.num_rows()).sum::<usize>()
    );
    let (a, b) = (&walked.bitwise, &ordinary.bitwise);
    assert_eq!(a.num_rows(), b.num_rows());
    assert_eq!(a.num_cols(), b.num_cols());
    let mut shown = 0;
    let mut per_col = vec![0usize; a.num_cols()];
    for row in 0..a.num_rows() {
        for col in 0..a.num_cols() {
            if a.get_main(row, col) != b.get_main(row, col) {
                per_col[col] += 1;
                if shown < 25 {
                    eprintln!(
                        "row {row} (x={:?} y={:?}) col {col}: walk {:?} vs ordinary {:?}",
                        a.get_main(row, 0).value(),
                        a.get_main(row, 1).value(),
                        a.get_main(row, col).value(),
                        b.get_main(row, col).value()
                    );
                    shown += 1;
                }
            }
        }
    }
    eprintln!("differing cells per column: {per_col:?}");
    assert!(
        per_col.iter().all(|&n| n == 0),
        "BITWISE multiplicities differ"
    );
}
