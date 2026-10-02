//! A packed main trace (`TraceTable::pack_main_narrow`) changes nothing a
//! verifier can see: packed after its Round-1 commit, it is widened again (on
//! the device by the kept-top recompute, on the host on every other path) to
//! the same words, so the proof is `Retain`'s byte for byte.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use super::residency_mode_tests::traces;
use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::narrow::NarrowMain;
use crate::proof::options::ProofOptions;
use crate::proof::stark::MultiProof;
use crate::prover::{IsStarkProver, Prover, ProvingError};
use crate::residency_mode::ResidencyMode;
use crate::trace::TraceTable;
use crate::traits::AIR;

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;

fn test_options() -> ProofOptions {
    ProofOptions {
        grinding_factor: 0,
        ..ProofOptions::default_test_options()
    }
}

/// The CPU/ADD/MUL instance proved the block's way: each table's main commit
/// made ahead (`precommit_main`), its trace then packed when `pack`, and the
/// rest by `multi_prove_precommitted`.
fn prove_precommitted(
    residency: ResidencyMode,
    pack: bool,
) -> Result<MultiProof<F, E, ()>, ProvingError> {
    let (mut cpu_trace, mut add_trace, mut mul_trace) = traces();
    let proof_options = test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);
    let mut precommitted = Vec::new();
    for (air, trace) in [
        (
            &cpu_air as &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
            &mut cpu_trace,
        ),
        (&add_air, &mut add_trace),
        (&mul_air, &mut mul_trace),
    ] {
        precommitted.push(Some(Prover::precommit_main(
            air,
            trace,
            #[cfg(feature = "disk-spill")]
            crate::storage_mode::StorageMode::Ram,
            residency,
        )?));
        if pack {
            assert!(trace.pack_main_narrow(), "a Goldilocks trace packs");
            assert!(trace.is_main_narrow());
        }
    }
    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];
    Prover::multi_prove_precommitted(
        pairs,
        &mut DefaultTranscript::<E>::new(&[]),
        #[cfg(feature = "disk-spill")]
        crate::storage_mode::StorageMode::Ram,
        residency,
        precommitted,
    )
}

fn bytes(proof: &MultiProof<F, E, ()>) -> Vec<u8> {
    bincode::serialize(proof).unwrap()
}

/// Packing keeps the words: the columns read from the packed trace and the
/// trace widened back on the host are the original ones, and the table keeps
/// its shape while packed.
#[test]
fn a_packed_trace_widens_to_the_same_words() {
    let (cpu, _, _) = traces();
    let mut packed: TraceTable<F, E> = cpu.clone();
    assert!(packed.pack_main_narrow());
    assert_eq!(packed.num_rows(), cpu.num_rows());
    assert_eq!(packed.main_table.width, cpu.main_table.width);
    assert_eq!(packed.columns_main(), cpu.columns_main());
    let narrow: &NarrowMain = packed.narrow_main().expect("packed");
    assert!(narrow.data().len() < cpu.num_rows() * cpu.main_table.width * 8);
    packed.widen_main_on_host();
    assert!(!packed.is_main_narrow());
    assert_eq!(packed.main_data_row_major(), cpu.main_data_row_major());
}

/// ★ Packed after the precommit, the proof is the unpacked one's, byte for
/// byte, under every residency (on a host build every path widens on the
/// host; under `cuda` with kept top levels, see the card test below).
#[test]
fn a_packed_trace_proves_the_same_bytes() {
    for residency in [
        ResidencyMode::Retain,
        ResidencyMode::RecomputeLde,
        ResidencyMode::RecomputeLdeDevice,
    ] {
        let wide = bytes(&prove_precommitted(residency, false).unwrap());
        let packed = bytes(&prove_precommitted(residency, true).unwrap());
        assert!(
            wide == packed,
            "{residency:?}: proof bytes moved with packing"
        );
    }
}

/// ★ On the card with kept top levels: every table's packed trace is widened
/// by the device (the counter moves by one per table), and the proof is
/// `Retain`'s byte for byte.
#[cfg(feature = "cuda")]
#[test_log::test]
#[ignore = "requires a GPU, LAMBDA_VM_GPU_LDE_THRESHOLD=2 and LAMBDA_VM_RECOMMIT_TOP_LEVELS=2; run alone"]
fn a_packed_trace_widens_on_the_card() {
    use std::sync::atomic::Ordering;
    let retained = bytes(&prove_precommitted(ResidencyMode::Retain, false).unwrap());
    let before = crate::prover::NARROW_DEVICE_WIDENS.load(Ordering::SeqCst);
    let packed = bytes(&prove_precommitted(ResidencyMode::RecomputeLdeDevice, true).unwrap());
    let widens = crate::prover::NARROW_DEVICE_WIDENS.load(Ordering::SeqCst) - before;
    assert_eq!(
        widens, 3,
        "every table must widen on the device (is LAMBDA_VM_RECOMMIT_TOP_LEVELS set?)"
    );
    assert!(
        retained == packed,
        "proof bytes moved between Retain and the device widen"
    );
}

/// ★ A bad device widen is refused: one packed bit flipped before the widen
/// makes the recomputed LDE disagree with the kept top levels, and the prover
/// refuses with `RecomputedCommitmentMismatch` instead of proving.
#[cfg(feature = "cuda")]
#[test_log::test]
#[ignore = "requires a GPU, LAMBDA_VM_GPU_LDE_THRESHOLD=2 and LAMBDA_VM_RECOMMIT_TOP_LEVELS=2; run alone"]
fn a_bad_device_widen_is_refused() {
    for table in 0..3 {
        crate::residency_mode::test_hooks::perturb_narrow_before_recommit(table);
        let out = prove_precommitted(ResidencyMode::RecomputeLdeDevice, true);
        assert_eq!(
            crate::residency_mode::test_hooks::PERTURB_NARROW_BEFORE_RECOMMIT
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "table {table}: the perturbation never fired"
        );
        // The kept-top check is the one that fires: its message names the
        // rebuilt subtree and the kept tree (no table).
        match out {
            Err(ProvingError::RecomputedCommitmentMismatch(msg)) => {
                assert!(
                    msg.contains("does not match the kept tree"),
                    "table {table}: {msg}"
                )
            }
            Err(e) => panic!("table {table}: wrong refusal {e:?}"),
            Ok(_) => panic!("table {table}: a bad widen was proved"),
        }
    }
}

/// ★ The device widen is the host widen, word for word, at every width.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a GPU"]
fn the_device_widen_is_the_host_widen() {
    let cols = 9;
    let rows = (1 << 13) + 5;
    let caps = [
        0u64,
        0xff,
        0x100,
        0xffff,
        0x1_0000,
        0xffff_ffff,
        0x1_0000_0000,
        u64::MAX,
        1,
    ];
    let words: Vec<u64> = (0..rows * cols)
        .map(|i| {
            let (r, c) = (i / cols, i % cols);
            if r == rows - 1 {
                caps[c]
            } else {
                (r as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) % (caps[c] / 2 + 1)
            }
        })
        .collect();
    let narrow = NarrowMain::pack(&words, cols);
    let offsets: Vec<u64> = narrow.offsets().iter().map(|&o| o as u64).collect();
    let device =
        math_cuda::narrow::widen_to_host(narrow.data(), &offsets, narrow.widths(), rows, cols)
            .expect("the device widen");
    assert!(
        device == words,
        "the device widen differs from the packed words"
    );
}
