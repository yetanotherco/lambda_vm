//! The sub-coset composition evaluation on the card: a two-part LogUp table at
//! blowup 4, proved with the evaluation forced off and forced on, must give the
//! same proof bytes, and the forced-on prove must have run the device
//! composition kernel and the device split + extension at stride 2 (so the test
//! cannot pass by falling back to the host). Box only (`--features cuda`,
//! ignored; the counters are process-global, hence one test thread):
//!
//! ```text
//! cargo test -p stark --release --features cuda --lib \
//!     tests::subcoset_device_tests -- --ignored --test-threads=1
//! ```

use crate::config::Blake3StarkHash;
use crate::gpu_lde::{gpu_composition_calls, gpu_device_only_calls, gpu_extend_halves_calls};
use crate::proof::options::ProofFormat;
use crate::prover::{composition_subcoset_tables, set_composition_subcoset};
use crate::tests::zf_golden_tests::{golden_options, prove_logup, sha3_hex, verify_logup};

struct Restore;
impl Drop for Restore {
    fn drop(&mut self) {
        set_composition_subcoset(None);
    }
}

fn counters() -> (u64, u64, u64, u64) {
    (
        composition_subcoset_tables(),
        gpu_composition_calls(),
        gpu_extend_halves_calls(),
        gpu_device_only_calls(),
    )
}

/// 2^13 rows (LDE 2^15: the card commits and keeps a host copy) and 2^17 rows
/// (LDE 2^19: the device-only envelope).
#[test]
#[ignore = "card: proves on the GPU; run on the box with --ignored --test-threads=1"]
fn subcoset_composition_on_the_card_is_byte_identical() {
    let _restore = Restore;
    for log_rows in [13u32, 17] {
        let rows = 1usize << log_rows;
        let o = golden_options(4, 1, 7, ProofFormat::DEFAULT);

        set_composition_subcoset(Some(false));
        let (air_off, off, _) = prove_logup::<Blake3StarkHash>(rows, &o);

        set_composition_subcoset(Some(true));
        let before = counters();
        let (air_on, on, _) = prove_logup::<Blake3StarkHash>(rows, &o);
        let after = counters();
        set_composition_subcoset(None);

        eprintln!(
            "SUBCOSET CARD rows=2^{log_rows}: subcoset tables +{} · composition kernels +{} · \
             device split+extend +{} · device-only tables +{}",
            after.0 - before.0,
            after.1 - before.1,
            after.2 - before.2,
            after.3 - before.3,
        );
        assert!(
            after.0 > before.0,
            "rows 2^{log_rows}: sub-coset path not taken"
        );
        assert!(
            after.1 > before.1,
            "rows 2^{log_rows}: the device composition kernel did not run"
        );
        assert!(
            after.2 > before.2,
            "rows 2^{log_rows}: the device split + extension did not run"
        );

        assert!(verify_logup::<Blake3StarkHash>(&air_off, &off));
        assert!(verify_logup::<Blake3StarkHash>(&air_on, &on));
        let off_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&off).expect("rkyv");
        let on_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&on).expect("rkyv");
        assert_eq!(
            sha3_hex(&on_bytes),
            sha3_hex(&off_bytes),
            "rows 2^{log_rows}: sub-coset proof bytes differ from the whole-coset proof on the card"
        );
    }
}
