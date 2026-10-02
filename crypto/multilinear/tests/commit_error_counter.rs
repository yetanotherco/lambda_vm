//! ★ A device commit that FAILS is counted apart from one the device declined,
//! and the chain still commits — on the host, to the same root.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda --test commit_error_counter -- --nocapture
//! ```
//!
//! Needs a GPU. The failure is a real one, not a stand-in: the whole VRAM
//! ledger is reserved first, so the commit's own reservation is refused and the
//! device commit returns `CUDA_ERROR_OUT_OF_MEMORY` through the same arm a
//! launch error takes. That arm used to turn the error into a silent `None`:
//! at WHIR stack 27 a tile launched past CUDA's grid limit, every commit fell
//! back to the host, and only `host_fallbacks()` moved
//! (`thoughts/zf/gap/fix/STRUCT.md` §9.2). Both device entry points are
//! driven — a polynomial uploaded for the commit (`parts`) and one already on
//! the card (`resident`) — and each should print one
//! `[whir] device commit (..) ... failed` line.
//!
//! Its own integration binary so the process-wide counters belong to it alone
//! (see `host_fallback_counter.rs`).
#![cfg(feature = "cuda")]

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu::{
    commit_calls, commit_errors, host_fallbacks, reset_call_counters, upload_columns,
};
use multilinear::mle::Mle;
use multilinear::whir_chain::{
    ChainConfig, ChainFormat, GrindBits, Stacked, commit, commit_stacked,
};
use multilinear::whir_hash::RpxWhir;

fn poly(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| FieldElement::from(i.wrapping_mul(6364136223846793005).wrapping_add(seed)))
            .collect(),
    )
    .expect("power of two")
}

/// `(device commits, host fallbacks, commit errors)`.
fn counts() -> (u64, u64, u64) {
    (commit_calls(), host_fallbacks(), commit_errors())
}

#[test]
fn a_failed_device_commit_is_counted_as_an_error_and_committed_on_the_host() {
    math_cuda::device::backend().expect("this test needs a GPU");
    let config = ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 3,
        grind: GrindBits::default(),
        format: ChainFormat::DEFAULT,
    };
    // 2^16 evaluations at blowup 4: a 2^18 codeword, above the device commit
    // threshold, so the device is asked.
    let num_vars = 16;
    let f = poly(num_vars, 5);
    let store = upload_columns(&[&f]).expect("the column upload");
    let resident = Stacked {
        parts: vec![(&f, 0)],
        resident: Some((&store, vec![(0, 0)])),
        num_vars,
    };
    reset_call_counters();

    // Every byte the ledger has left, held: each commit's reservation is refused.
    let mut held = Vec::new();
    while let Some(reservation) = math_cuda::device::reserve(1 << 20) {
        held.push(reservation);
    }
    let (parts_on_host, _) = commit::<F, RpxWhir>(&f, &config, true).expect("commit");
    assert_eq!(counts(), (0, 1, 1), "the uploaded commit's error");
    let (resident_on_host, _) =
        commit_stacked::<F, RpxWhir>(&resident, &config, true).expect("commit");
    assert_eq!(counts(), (0, 2, 2), "the resident commit's error");
    assert!(parts_on_host.codeword().device().is_none());
    assert!(resident_on_host.codeword().device().is_none());
    drop(held);

    // The same commit with the ledger free runs on the device, to the same root.
    let (on_device, _) = commit_stacked::<F, RpxWhir>(&resident, &config, true).expect("commit");
    assert_eq!(counts(), (1, 2, 2), "a device commit is not an error");
    assert!(
        on_device.codeword().device().is_some(),
        "the control commit must run on the card"
    );
    assert_eq!(parts_on_host.root(), on_device.root());
    assert_eq!(resident_on_host.root(), on_device.root());

    // Below the threshold the device is not asked: a fallback, not an error.
    let _ = commit::<F, RpxWhir>(&poly(8, 9), &config, true).expect("commit");
    assert_eq!(counts(), (1, 3, 2), "a decline is not an error");
}
