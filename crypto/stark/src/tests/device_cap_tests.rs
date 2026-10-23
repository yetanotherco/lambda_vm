//! A thread's device cap ([`crate::prover::cap_device_bytes`]): a prove under a
//! cap below a table's device set is refused by name before its first commit,
//! and a cap every table fits under changes no byte of the proof.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::proof::options::ProofOptions;
use crate::proof::stark::MultiProof;
use crate::prover::{IsStarkProver, Prover, ProvingError, cap_device_bytes, device_cap};
use crate::residency_mode::ResidencyMode;
use crate::traits::AIR;

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;

fn options() -> ProofOptions {
    ProofOptions {
        grinding_factor: 0,
        ..ProofOptions::default_test_options()
    }
}

/// The CPU/ADD/MUL instance proved on `transcript`, under this thread's cap.
fn prove_on(transcript: &mut DefaultTranscript<E>) -> Result<MultiProof<F, E, ()>, ProvingError> {
    let (mut cpu, mut add, mut mul) = super::residency_mode_tests::traces();
    let opts = options();
    let (cpu_air, add_air, mul_air) = (
        new_cpu_air_with_lookup(&opts),
        new_add_air_with_lookup(&opts),
        new_mul_air_with_lookup(&opts),
    );
    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu, &()),
        (&add_air, &mut add, &()),
        (&mul_air, &mut mul, &()),
    ];
    Prover::multi_prove(
        pairs,
        transcript,
        #[cfg(feature = "disk-spill")]
        crate::storage_mode::StorageMode::Ram,
        ResidencyMode::Retain,
    )
}

/// ★ Under a cap below every table's device set the prove is refused, naming
/// the first table in index order with its bytes and the cap, and the
/// transcript has absorbed nothing: no root was committed. Without the
/// pre-check the gate admits an oversized table alone and the prove succeeds,
/// so this test is red with it removed.
#[test]
fn a_prove_under_a_cap_below_a_table_is_refused_before_its_first_commit() {
    let mut transcript = DefaultTranscript::<E>::new(&[]);
    let before = transcript.state();
    let refused = {
        let _cap = cap_device_bytes(1);
        prove_on(&mut transcript)
    };
    match refused {
        Err(ProvingError::OverDeviceCap { table, bytes, cap }) => {
            let cpu = new_cpu_air_with_lookup(&options());
            assert_eq!(table, cpu.name(), "the first table over the cap is named");
            assert_eq!(cap, 1);
            assert!(bytes > 1, "its device set is the bytes reported: {bytes}");
        }
        other => panic!(
            "a prove under a one-byte cap must be refused: {:?}",
            other.err()
        ),
    }
    assert_eq!(
        transcript.state(),
        before,
        "refused before any commit: the transcript absorbed nothing"
    );
    assert_eq!(device_cap(), None, "the guard put the uncapped state back");
}

/// A cap every table fits under proves the same bytes as no cap.
#[test]
fn a_cap_every_table_fits_under_changes_no_byte() {
    let open = prove_on(&mut DefaultTranscript::<E>::new(&[])).expect("proves uncapped");
    let capped = {
        let _cap = cap_device_bytes(u64::MAX / 2);
        prove_on(&mut DefaultTranscript::<E>::new(&[])).expect("proves under a wide cap")
    };
    let bytes = |p: &MultiProof<F, E, ()>| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    assert_eq!(bytes(&open), bytes(&capped));
}

/// The guard is the thread's and nests: the inner cap is undone, the outer
/// one kept, and another thread never sees either.
#[test]
fn the_cap_is_the_threads_and_restored_by_its_guard() {
    assert_eq!(device_cap(), None);
    {
        let _outer = cap_device_bytes(10 << 30);
        {
            let _inner = cap_device_bytes(1 << 30);
            assert_eq!(device_cap(), Some(1 << 30));
        }
        assert_eq!(device_cap(), Some(10 << 30));
        let elsewhere = std::thread::spawn(device_cap)
            .join()
            .expect("a plain thread");
        assert_eq!(elsewhere, None, "the cap is this thread's alone");
    }
    assert_eq!(device_cap(), None);
}

/// ★ The per-table drivers draw their streams in their spawner's class, so a
/// proof marked background stays on the background pool on every table's
/// thread. Thread-local state only: no card is touched. Red if `run_admitted`
/// stops copying the class into its workers.
#[cfg(feature = "cuda")]
#[test]
fn the_table_drivers_take_their_spawners_stream_class() {
    use math_cuda::stream_class::{StreamClass, current, enter};
    let classes = |spawner: StreamClass| -> Vec<StreamClass> {
        let _class = enter(spawner);
        let gate_budget = u64::MAX;
        let order: Vec<usize> = (0..6).collect();
        let estimates = vec![1u64; 6];
        crate::prover::run_admitted_for_tests(&order, &estimates, gate_budget, 3, |_| current())
            .into_iter()
            .map(|c| c.expect("every slot is filled"))
            .collect()
    };
    assert!(
        classes(StreamClass::Background)
            .iter()
            .all(|&c| c == StreamClass::Background),
        "a background prove's drivers are background"
    );
    assert!(
        classes(StreamClass::Foreground)
            .iter()
            .all(|&c| c == StreamClass::Foreground),
        "a foreground prove's drivers are foreground"
    );
}
