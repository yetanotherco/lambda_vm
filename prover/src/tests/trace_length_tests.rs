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

mod an_out_of_range_trace_length_in_a_block_proof_is_refused {
    use super::*;

    #[test]
    fn legacy() {
        super::assert_block_trace_lengths_are_checked(OneRowMode::Off);
    }

    #[test]
    fn one_row_dp() {
        super::assert_block_trace_lengths_are_checked(OneRowMode::Auto);
    }
}

fn assert_block_trace_lengths_are_checked(one_row: OneRowMode) {
    let elf = asm_elf_bytes("test_mul_8");
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
    if one_row == OneRowMode::Off {
        assert_phase_a_rejects_out_of_range_lengths(&elf, &opts, &proof);
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

/// ★ The commit bus's phase A replay (`replay_transcript_phase_a_view`), called
/// directly at the default format on a block proof's own AIRs: it recovers
/// `(z, α)` from the honest proof and refuses (`None`) the proof with one
/// table's length patched out of range, before it derives anything from it.
fn assert_phase_a_rejects_out_of_range_lengths(
    elf_bytes: &[u8],
    opts: &ProofOptions,
    vm: &crate::VmProof,
) {
    use crate::tables::trace_builder::Traces;
    use stark::proof::view::MultiProofView;
    let program = executor::elf::Elf::load(elf_bytes).expect("an ELF");
    let page_configs = Traces::page_configs_from_elf_and_runtime(
        &program,
        &vm.runtime_page_ranges,
        vm.num_private_input_pages,
        vm.proof.proofs.len(),
    )
    .expect("page configs");
    let committed = u32::try_from(vm.public_output.len()).expect("a short output");
    let replay = |proof: &stark::proof::stark::MultiProof<crate::F, crate::E, ()>| {
        let view = MultiProofView::Owned(proof);
        let cpu_heights =
            crate::cpu_heights_from_proofs(view, &vm.table_counts, true).expect("the CPU tables");
        let airs = crate::VmAirs::new(
            &program,
            opts,
            false,
            &page_configs,
            &vm.table_counts,
            &cpu_heights,
            None,
            true,
            None,
            crate::RegisterFiles::halted(opts, program.entry_point, committed).preprocessed(),
        );
        let refs = airs.air_refs();
        let mut transcript = crate::hash_pin::block_transcript(&[]);
        crate::replay_transcript_phase_a_view(&refs, view, &mut transcript)
    };
    assert!(
        replay(&vm.proof).is_some(),
        "the control: the honest replay"
    );
    // A table that is not a CPU table: the CPU heights size the AIRs.
    let cpu_start = crate::cpu_proof_start(true, &vm.table_counts);
    let table = if cpu_start == 0 {
        vm.table_counts.cpu
    } else {
        0
    };
    for claimed in [0, 24, 1 << 32, (1 << 63) + 8] {
        let mut proof = vm.proof.clone();
        proof.proofs[table].trace_length = claimed;
        assert!(
            replay(&proof).is_none(),
            "table {table} at length {claimed}: the replay recovered challenges"
        );
    }
}
