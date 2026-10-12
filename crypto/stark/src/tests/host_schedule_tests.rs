//! Host-schedule switches that move work between threads, or earlier, and must
//! leave every proof byte where it was.
//!
//! - `LAMBDA_VM_OOD_COLUMNS_ON_CALLER`: the out-of-domain tables' columns are
//!   built on the per-table driver instead of through the rayon pool.
//! - The domain and twiddle warm-up a base's head helper runs: the cache entry
//!   a prove finds warm is the one it would have built.
//!
//! The oracle for the first is the whole proof, as in `residency_mode_tests`:
//! both sides are proved in this process from the same traces with grinding
//! off, so byte equality is exactly "invisible to the proof".

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField, goldilocks::GoldilocksField,
};

use super::residency_mode_tests::traces;
use crate::domain::Domain;
use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::proof::options::ProofOptions;
use crate::proof::stark::MultiProof;
use crate::prover::{
    IsStarkProver, OOD_COLUMNS_ON_CALLER_READS, Prover, domain_and_twiddles_for_options,
    ood_columns_on_caller_setting, pin_ood_columns_on_caller, warm_domain_and_twiddles,
};
use crate::residency_mode::ResidencyMode;
use crate::table::Table;
use crate::traits::AIR;

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

/// Grinding off: under `parallel` the nonce search is rayon's `find_any`, so a
/// ground proof is not byte-reproducible whatever the schedule.
fn test_options() -> ProofOptions {
    ProofOptions {
        grinding_factor: 0,
        ..ProofOptions::default_test_options()
    }
}

fn prove_with_ood_columns_on_caller(on: bool) -> MultiProof<F, E, ()> {
    let (mut cpu_trace, mut add_trace, mut mul_trace) = traces();
    let proof_options = test_options();
    let cpu_air = new_cpu_air_with_lookup(&proof_options);
    let add_air = new_add_air_with_lookup(&proof_options);
    let mul_air = new_mul_air_with_lookup(&proof_options);
    let pairs: Vec<(
        &dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&cpu_air, &mut cpu_trace, &()),
        (&add_air, &mut add_trace, &()),
        (&mul_air, &mut mul_trace, &()),
    ];
    pin_ood_columns_on_caller(Some(on));
    let proof = Prover::multi_prove(
        pairs,
        &mut DefaultTranscript::<E>::new(&[]),
        #[cfg(feature = "disk-spill")]
        crate::storage_mode::StorageMode::Ram,
        ResidencyMode::Retain,
    );
    pin_ood_columns_on_caller(None);
    proof.unwrap()
}

/// The serial transpose is the parallel one: same columns, same order, for a
/// one-row, a two-row and a taller table, and for a table with no rows.
#[test]
fn columns_serial_is_columns() {
    for (height, width) in [(1usize, 7usize), (2, 5), (9, 3), (0, 4)] {
        let data: Vec<FE> = (0..height * width)
            .map(|i| FE::from(1000 + i as u64))
            .collect();
        let table = Table::new(data, width);
        assert_eq!(
            table.columns_serial(),
            table.columns(),
            "{height} x {width}"
        );
    }
}

/// The switch: unset, empty or `0` is the pool, `1` the calling thread.
#[test]
fn the_ood_columns_switch_is_the_pool_unless_exactly_one() {
    assert!(!ood_columns_on_caller_setting(None));
    assert!(!ood_columns_on_caller_setting(Some("")));
    assert!(!ood_columns_on_caller_setting(Some("0")));
    assert!(ood_columns_on_caller_setting(Some("1")));
    assert!(ood_columns_on_caller_setting(Some(" 1 ")));
}

/// Anything else stops the run rather than measuring the default under the
/// switch's name.
#[test]
#[should_panic(expected = "LAMBDA_VM_OOD_COLUMNS_ON_CALLER must be 0 or 1")]
fn the_ood_columns_switch_refuses_anything_else() {
    ood_columns_on_caller_setting(Some("yes"));
}

/// ★ Byte for byte, the OOD columns built on the calling thread prove what the
/// pool's prove: the round-3 absorb and the DEEP terms read the same values in
/// the same order. The counter shows the arm named is the arm that ran — a
/// comparison of two runs of one arm would pass without testing anything.
#[test]
fn ood_columns_on_the_caller_prove_the_same_bytes() {
    let before = OOD_COLUMNS_ON_CALLER_READS.load(std::sync::atomic::Ordering::SeqCst);
    let on_caller = prove_with_ood_columns_on_caller(true);
    let after_on = OOD_COLUMNS_ON_CALLER_READS.load(std::sync::atomic::Ordering::SeqCst);
    assert!(
        after_on >= before + 3,
        "the calling-thread arm must have read each of the three tables' OOD columns \
         ({before} -> {after_on})"
    );
    let in_pool = prove_with_ood_columns_on_caller(false);
    let a = bincode::serialize(&on_caller).unwrap();
    let b = bincode::serialize(&in_pool).unwrap();
    assert_eq!(
        a.len(),
        b.len(),
        "proof size moved between the OOD schedules"
    );
    assert!(a == b, "proof bytes moved between the OOD schedules");
}

/// A domain built from the options alone is the one `Domain::new` builds from
/// an AIR with those options.
#[test]
fn a_domain_from_options_is_the_domain_from_the_air() {
    let options = test_options();
    let air = new_cpu_air_with_lookup(&options);
    for n in [8usize, 64, 1024] {
        let from_air = Domain::<F>::new(&air, n);
        let from_options = Domain::<F>::from_options(&options, n);
        assert_eq!(
            from_air.lde_roots_of_unity_coset,
            from_options.lde_roots_of_unity_coset
        );
        assert_eq!(
            from_air.trace_roots_of_unity,
            from_options.trace_roots_of_unity
        );
        assert_eq!(
            from_air.trace_primitive_root,
            from_options.trace_primitive_root
        );
        assert_eq!(from_air.coset_offset, from_options.coset_offset);
        assert_eq!(from_air.blowup_factor, from_options.blowup_factor);
        assert_eq!(
            from_air.interpolation_domain_size,
            from_options.interpolation_domain_size
        );
    }
}

/// ★ A warmed entry is the entry a prove looks up: after the warm-up the
/// lookup is a cache HIT that returns the very `Arc`s the warm-up stored, so
/// the prove reads the warm-up's domain and twiddles and builds nothing.
///
/// Keyed on a coset offset no other test uses, so the process-wide cache is
/// cold for it here whatever ran before.
#[test]
fn a_warmed_domain_is_the_one_the_prove_looks_up() {
    let options = ProofOptions {
        coset_offset: 11,
        ..test_options()
    };
    let n = 256;
    super::domain_cache_stats::reset();
    warm_domain_and_twiddles::<F>(&options, n);
    assert_eq!(
        super::domain_cache_stats::get(),
        (0, 1),
        "the warm-up builds the entry: one miss"
    );
    let (d1, t1) = domain_and_twiddles_for_options::<F>(&options, n);
    let (d2, t2) = domain_and_twiddles_for_options::<F>(&options, n);
    assert_eq!(
        super::domain_cache_stats::get(),
        (2, 1),
        "the lookups after it are hits"
    );
    assert!(std::sync::Arc::ptr_eq(&d1, &d2) && std::sync::Arc::ptr_eq(&t1, &t2));
    assert_eq!(
        d1.lde_roots_of_unity_coset,
        Domain::<F>::from_options(&options, n).lde_roots_of_unity_coset
    );
}
