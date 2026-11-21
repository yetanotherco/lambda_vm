//! Phase B's table uploads with their wide columns copied on several streams
//! at once (`gpu::with_copy_lanes`): the card holds the same words, column for
//! column in the same order, as one stream copying them one after another,
//! for wide tables alone and wide tables beside a narrow one.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda,parallel --test copy_lanes_device -- --test-threads=1
//! ```
//!
//! Needs a GPU.
#![cfg(feature = "cuda")]

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu;
use multilinear::mle::Mle;

fn column(rows: usize, salt: u64) -> Mle<F> {
    Mle::new(
        (0..rows as u64)
            .map(|i| {
                FieldElement::<F>::from_raw((i ^ salt).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 1)
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn a_group_uploaded_on_several_copy_lanes_holds_one_lanes_words() {
    gpu::set_up_device();
    assert!(gpu::reserve_budget() > 0, "needs a card");
    // Two wide tables of uneven columns (so the lanes split unevenly) and a
    // narrow one between them.
    let a: Vec<Mle<F>> = (0..7u64).map(|k| column(1 << 20, k)).collect();
    let b: Vec<Mle<F>> = (0..3u64).map(|k| column(1 << 14, 50 + k)).collect();
    let small: Vec<Vec<u64>> = (0..4u64)
        .map(|k| (0..1u64 << 16).map(|i| (i * 7 + k) & 0xff).collect())
        .collect();
    let small_refs: Vec<&[u64]> = small.iter().map(Vec::as_slice).collect();
    let packed = multilinear::narrow::NarrowColumns::pack_par(&small_refs).expect("packs");
    let tables = [
        gpu::TableColumns::Wide(&a),
        gpu::TableColumns::Narrow(&packed),
        gpu::TableColumns::Wide(&b),
    ];
    let one = gpu::with_copy_lanes(1, || gpu::upload_tables(&tables))
        .expect("one lane goes up")
        .download()
        .expect("reads back");
    for lanes in [2usize, 3, 4] {
        let many = gpu::with_copy_lanes(lanes, || gpu::upload_tables(&tables))
            .expect("the lanes go up")
            .download()
            .expect("reads back");
        assert!(many == one, "{lanes} lanes hold one lane's words");
    }
    assert_eq!(one.len(), 7 + 4 + 3, "a span a column");
    println!(
        "COPY LANES: {} columns equal on 2, 3 and 4 lanes",
        one.len()
    );
}
