//! A proof whose sub-proof claims a trace length out of range — not a power
//! of two, one doubling past the largest LDE, a claim whose product with the
//! blowup wraps, the top bit — is refused with no panic, in any table:
//! - a VM block proof, at the legacy format and under one-row `auto` with
//!   `fri = dp` (where the commit bus's phase A replay derives a leaf layout
//!   from each length before the STARK verifier runs);
//! - an LFM proof, under the legacy family (`lfm_verify`) and the block
//!   family (`verify_against_artifacts_under`, the block tree top's check).

use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;
use stark::proof::options::{FriMode, OneRowMode, ProofFormat, ProofOptions};

/// The VM's options: the legacy format, or one-row `auto` with `fri = dp` at
/// blowup 4 (the blowup the one-row static roots ship for).
fn options(one_row: OneRowMode) -> ProofOptions {
    let mut o = ProofOptions::default_test_options();
    if one_row == OneRowMode::Auto {
        o.blowup_factor = 4;
        o.format = ProofFormat {
            one_row,
            fri_mode: FriMode::Dp,
            ..ProofFormat::LEGACY
        };
    }
    o
}

#[test]
fn an_out_of_range_trace_length_in_a_block_proof_is_refused() {
    let elf = asm_elf_bytes("test_mul_8");
    for one_row in [OneRowMode::Off, OneRowMode::Auto] {
        let opts = options(one_row);
        let mut proof = crate::prove_with_options(&elf, &opts, &MaxRowsConfig::default())
            .unwrap_or_else(|e| panic!("{one_row:?}: proves: {e:?}"));
        assert!(
            matches!(
                crate::verify_with_options(&proof, &elf, &opts, None, None),
                Ok(true)
            ),
            "{one_row:?}: the control: the honest proof verifies"
        );
        for table in 0..proof.proof.proofs.len() {
            let honest = proof.proof.proofs[table].trace_length;
            for claimed in [0, 24, 1 << 32, (1 << 63) + honest] {
                proof.proof.proofs[table].trace_length = claimed;
                let verdict = crate::verify_with_options(&proof, &elf, &opts, None, None);
                proof.proof.proofs[table].trace_length = honest;
                assert!(
                    !matches!(verdict, Ok(true)),
                    "{one_row:?}: table {table} at length {claimed} was accepted"
                );
            }
        }
    }
}

/// An LFM machine proof (`TrivialV0`) under either family: the honest proof
/// verifies, and each sub-proof's length patched out of range is refused.
#[test]
fn an_out_of_range_trace_length_in_an_lfm_proof_is_refused() {
    use crate::hash_pin::{BlockFamily, LegacyFamily, ProofFamily};
    use crate::lfm::proof::{LfmProof, lfm_prove_under, verify_against_artifacts_under};
    use crate::lfm::registry::{LfmProgramKind, build_artifacts, build_artifacts_under};
    use crate::tables::types::FE;

    fn under<Fam: ProofFamily>(name: &str) {
        let o = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(4).unwrap();
        let program = LfmProgramKind::TrivialV0.program();
        let hasher = build_artifacts(&program, &o).hasher;
        let artifacts = build_artifacts_under::<Fam::Stark>(&program, &o, hasher);
        let arenas: Vec<Vec<crate::lfm::word::LfmWord>> = vec![
            (0..4u64)
                .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
                .collect(),
        ];
        let mut proved: LfmProof = lfm_prove_under::<Fam>(&program, &artifacts, &arenas, &o)
            .unwrap_or_else(|e| panic!("{name}: proves: {e:?}"));
        let verify = |p: &LfmProof| {
            verify_against_artifacts_under::<Fam>(&artifacts, &p.proof, &p.public_words, &o)
        };
        assert!(verify(&proved), "{name}: the control: the honest proof");
        for table in 0..proved.proof.proofs.len() {
            let honest = proved.proof.proofs[table].trace_length;
            for claimed in [0, 24, 1 << 31, (1 << 62) + honest, 1 << 63, usize::MAX] {
                proved.proof.proofs[table].trace_length = claimed;
                let accepted = verify(&proved);
                proved.proof.proofs[table].trace_length = honest;
                assert!(
                    !accepted,
                    "{name}: table {table} at length {claimed} was accepted"
                );
            }
        }
    }
    under::<LegacyFamily>("legacy");
    under::<BlockFamily>("block");
}
