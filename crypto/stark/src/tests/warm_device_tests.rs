//! The prove-constants warm-up and the staged composition drain on the card.
//! Box only (`--features cuda`, ignored; process-global counters, one thread):
//!
//! ```text
//! cargo test -p stark --release --features cuda --lib tests::warm_device_tests \
//!     -- --ignored --test-threads=1
//! ```

use crate::config::Blake3StarkHash;
use crate::examples::read_only_memory_logup::LogReadOnlyRAP;
use crate::proof::options::ProofFormat;
use crate::prover::{set_warm_prove_constants, warm_prove_constants, warmed_prove_constants};
use crate::tests::zf_golden_tests::{golden_options, prove_logup, sha3_hex, verify_logup};
use crate::traits::AIR;

struct Restore;
impl Drop for Restore {
    fn drop(&mut self) {
        set_warm_prove_constants(None);
    }
}

/// The staged drain returns the slab drain's words, raw.
#[test]
#[ignore = "card: drains a resident H both ways; run on the box with --ignored"]
fn the_staged_composition_drain_matches_the_slab_drain() {
    let rows = (1usize << 21) + 5; // 48 MiB and change: two staged chunks and a short tail
    let words: Vec<u64> = (0..rows as u64 * 3)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5555)
        .collect();
    let h = math_cuda::constraint_interp::comp_h_from_host_interleaved(&words, rows).expect("card");
    let slab = math_cuda::constraint_interp::download_comp_h(&h).expect("slab drain");
    let staged = math_cuda::constraint_interp::download_comp_h_staged(&h).expect("staged drain");
    assert_eq!(slab, words);
    assert_eq!(staged, words);
}

/// A two-part LogUp table at blowup 4 over the device floor (2^17 rows: the
/// device-only envelope), proved with the warm-up off, then warmed and proved
/// with it on: the same bytes, both verify, and the warm-up ran over the
/// table's constants (boundary zerofier inverse, their card copies, the
/// split's table, the coset points).
#[test]
#[ignore = "card: proves on the GPU; run on the box with --ignored --test-threads=1"]
fn a_warmed_prove_is_byte_identical() {
    let _restore = Restore;
    let rows = 1usize << 17;
    let o = golden_options(4, 1, 7, ProofFormat::DEFAULT);
    set_warm_prove_constants(Some(false));
    let (air_off, off, pi) = prove_logup::<Blake3StarkHash>(rows, &o);

    set_warm_prove_constants(Some(true));
    let air = LogReadOnlyRAP::<
        math::field::goldilocks::GoldilocksField,
        math::field::extensions_goldilocks::Degree3GoldilocksExtensionField,
    >::new(&o);
    let before = warmed_prove_constants();
    let tables: Vec<(
        &dyn AIR<Field = _, FieldExtension = _, PublicInputs = _>,
        usize,
        &_,
    )> = vec![(&air, rows, &pi)];
    warm_prove_constants(&tables);
    let warmed = warmed_prove_constants() - before;
    let (air_on, on, _) = prove_logup::<Blake3StarkHash>(rows, &o);
    set_warm_prove_constants(None);
    println!("WARM DEVICE: {warmed} constants warmed");
    assert!(
        warmed >= 4,
        "boundary inverse, its card copy, the split's table, the coset points"
    );

    assert!(verify_logup::<Blake3StarkHash>(&air_off, &off));
    assert!(verify_logup::<Blake3StarkHash>(&air_on, &on));
    let b = |p| sha3_hex(&rkyv::to_bytes::<rkyv::rancor::Error>(p).expect("rkyv"));
    assert_eq!(b(&on), b(&off), "the warm-up moved a proof byte");
}

/// The DEEP's LDE coset warmed before a prove: one upload for a length the card
/// has not seen under that offset, none on the second call. A prove after it
/// verifies and repeats byte for byte, and a prove whose DEEP found its coset in
/// the cache (the second under another offset) has the bytes of the one whose
/// DEEP uploaded it: the warm-up only fills that cache.
#[test]
#[ignore = "card: proves on the GPU; run on the box with --ignored --test-threads=1"]
fn the_deep_coset_warm_uploads_once_and_moves_no_byte() {
    use crate::prover::{warm_deep_coset_points, warmed_deep_cosets};
    type Air = LogReadOnlyRAP<
        math::field::goldilocks::GoldilocksField,
        math::field::extensions_goldilocks::Degree3GoldilocksExtensionField,
    >;
    let rows = 1usize << 17;
    // Offsets no other test proves under, so neither coset is on the card yet.
    let mut cold_o = golden_options(4, 1, 7, ProofFormat::DEFAULT);
    cold_o.coset_offset = 23;
    let mut o = cold_o.clone();
    o.coset_offset = 29;
    let (air_cold, cold, pi) = prove_logup::<Blake3StarkHash>(rows, &cold_o);
    let (_, cold_again, _) = prove_logup::<Blake3StarkHash>(rows, &cold_o);

    let air = Air::new(&o);
    let tables: Vec<(
        &dyn AIR<Field = _, FieldExtension = _, PublicInputs = _>,
        usize,
        &_,
    )> = vec![(&air, rows, &pi)];
    let before = warmed_deep_cosets();
    assert_eq!(warm_deep_coset_points(&tables), 1, "one coset uploaded");
    assert_eq!(warm_deep_coset_points(&tables), 0, "then found");
    assert_eq!(warmed_deep_cosets() - before, 1);
    let (air_w, warm, _) = prove_logup::<Blake3StarkHash>(rows, &o);
    let (_, warm_again, _) = prove_logup::<Blake3StarkHash>(rows, &o);

    assert!(verify_logup::<Blake3StarkHash>(&air_cold, &cold));
    assert!(verify_logup::<Blake3StarkHash>(&air_w, &warm));
    let b = |p| sha3_hex(&rkyv::to_bytes::<rkyv::rancor::Error>(p).expect("rkyv"));
    assert_eq!(
        b(&cold),
        b(&cold_again),
        "a cached coset moved a proof byte"
    );
    assert_eq!(b(&warm), b(&warm_again), "a warmed prove does not repeat");
    assert_ne!(b(&warm), b(&cold), "the two offsets prove differently");
}
