//! A store the card's ledger refuses (`gpu::upload_store_columns`, the upload
//! once's store of a retired prepared stack) is not an argue device fallback:
//! nothing moves to the host, the reader copies twice on the card instead. The
//! argue's own upload refused under the same ledger is counted, so the count
//! can move here.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda,parallel --test upload_store_refusal_device -- --test-threads=1
//! ```
//!
//! Needs a GPU. It moves the ledger's budget for the process, so it is the
//! only test in its binary.
#![cfg(feature = "cuda")]

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu;
use multilinear::mle::Mle;

#[test]
fn a_store_the_card_refuses_is_not_an_argue_fallback() {
    gpu::set_up_device();
    assert!(gpu::reserve_budget() > 0, "needs a card");
    let columns: Vec<Mle<F>> = (0..4u64)
        .map(|k| {
            Mle::new(
                (0..1u64 << 20)
                    .map(|i| FieldElement::<F>::from(i ^ k))
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    let refs: Vec<&Mle<F>> = columns.iter().collect();
    // Room for 1 MiB more than the ledger holds now: the 32 MiB store is refused.
    let budget = gpu::set_reserve_budget(gpu::ledger_reserved() + (1 << 20));
    let before = gpu::argue_device_fallbacks();
    let store = gpu::upload_store_columns(&refs);
    let after_store = gpu::argue_device_fallbacks();
    let argue = gpu::upload_columns(&refs);
    let after_argue = gpu::argue_device_fallbacks();
    gpu::set_reserve_budget(budget);
    assert!(store.is_none(), "the store is refused");
    assert_eq!(
        after_store, before,
        "a refused store is not an argue fallback"
    );
    assert!(argue.is_none(), "the argue's upload is refused too");
    assert_eq!(after_argue, before + 1, "the argue's refusal is one");
}
