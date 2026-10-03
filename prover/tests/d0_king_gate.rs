//! Cross-version proof oracle for the D0 commitment-hash migration.
//!
//! A prove/verify round trip inside one build cannot see a self-consistent
//! drift: a version that changes how it commits still accepts its own proofs.
//! This exchanges proof *bytes* across versions instead — generate at the ref
//! before a change, verify at the ref after — so a moved leaf layout, transcript
//! or wire format fails loudly. It is the LFM-side counterpart of
//! `scripts/cross_verify_vm.sh`, which does the same for RV64 ELF proofs in both
//! directions.
//!
//! `#[ignore]`d because it is an oracle, not a regression test: it needs two
//! builds, an out-of-tree byte store, and an operator deciding which two refs
//! are being compared.
//!
//! ```text
//! # at the OLD ref
//! KING_GATE=generate KING_GATE_DIR=/some/dir \
//!   cargo test --release -p lambda-vm-prover --test d0_king_gate -- --ignored --nocapture
//! # at the NEW ref, same directory
//! KING_GATE=verify KING_GATE_DIR=/some/dir \
//!   cargo test --release -p lambda-vm-prover --test d0_king_gate -- --ignored --nocapture
//! ```
//!
//! This is the gate for D0 steps 3-7 (`thoughts/shared/block-compression/`):
//! the Blake3 leaf/pair backends, the B1 transcript, the `lfm_prove` wiring and
//! the registry rows all have to keep Test-hasher proofs verifying, and this is
//! what says they do. Steps that deliberately move the format re-generate the
//! bytes and say so.
//!
//! Two things worth keeping true of this file. It must compile *unchanged*
//! across the refs being compared — that is itself the API-stability half of the
//! test, and editing it to make it build defeats the purpose. And it must be
//! able to fail: flipping one byte of the stored archive has to make `verify`
//! reject.

use lambda_vm_prover::lfm::programs::trivial_program;
use lambda_vm_prover::lfm::registry::{LfmProgramKind, build_artifacts};
use lambda_vm_prover::lfm::word::LfmWord;
use lambda_vm_prover::lfm::{lfm_prove, lfm_verify};
use lambda_vm_prover::tables::types::FE;
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::proof::stark::MultiProof;

type F = lambda_vm_prover::tables::types::GoldilocksField;
type E = lambda_vm_prover::tables::types::GoldilocksExtension;

fn options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("options")
}

fn arenas() -> Vec<Vec<LfmWord>> {
    vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ]
}

fn dir() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("KING_GATE_DIR").expect("KING_GATE_DIR"))
}

#[test]
#[ignore = "cross-version oracle: needs KING_GATE=generate|verify and KING_GATE_DIR"]
fn lfm_trivial_v0_cross_version() {
    let mode = std::env::var("KING_GATE").expect(
        "set KING_GATE=generate (at the old ref) or KING_GATE=verify (at the new one), \
         plus KING_GATE_DIR pointing at a directory that outlives both builds",
    );
    let opts = options();
    let proof_path = dir().join("lfm_trivial_v0.rkyv");
    let words_path = dir().join("lfm_trivial_v0.words.rkyv");

    match mode.as_str() {
        "generate" => {
            let program = trivial_program();
            let artifacts = build_artifacts(&program, &opts);
            let proved = lfm_prove(&program, &artifacts, &arenas(), &opts).expect("prove");
            assert!(
                lfm_verify(
                    LfmProgramKind::TrivialV0,
                    &proved.proof,
                    &proved.public_words,
                    &opts
                )
                .expect("registered"),
                "the freshly built proof must verify where it was built"
            );
            let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proved.proof).expect("archive");
            let words =
                rkyv::to_bytes::<rkyv::rancor::Error>(&proved.public_words).expect("archive words");
            std::fs::write(&proof_path, &bytes).expect("write proof");
            std::fs::write(&words_path, &words).expect("write words");
            eprintln!(
                "GENERATED {} ({} bytes) + {} ({} bytes)",
                proof_path.display(),
                bytes.len(),
                words_path.display(),
                words.len()
            );
        }
        "verify" => {
            let bytes = std::fs::read(&proof_path).expect("read proof");
            let words = std::fs::read(&words_path).expect("read words");
            let proof = rkyv::from_bytes::<MultiProof<F, E, ()>, rkyv::rancor::Error>(&bytes)
                .expect("the archive from the other ref must still deserialize");
            let public_words = rkyv::from_bytes::<Vec<(u32, LfmWord)>, rkyv::rancor::Error>(&words)
                .expect("words deserialize");
            assert!(
                lfm_verify(LfmProgramKind::TrivialV0, &proof, &public_words, &opts)
                    .expect("registered"),
                "proof bytes from the other ref must verify under this build"
            );
            eprintln!(
                "VERIFIED {} ({} bytes) under this build",
                proof_path.display(),
                bytes.len()
            );
        }
        other => panic!("KING_GATE must be generate|verify, got {other:?}"),
    }
}
