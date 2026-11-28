//! A retired prepared stack's store sent narrow (`gpu::upload_store_columns_narrow`):
//! the card holds the words the wide store holds, column for column in the
//! same order, across heights, widths and run cuts; and a narrow store the
//! card's ledger refuses is not an argue device fallback.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda,parallel --test narrow_store_device -- --test-threads=1
//! ```
//!
//! Needs a GPU. Its second half moves the ledger's budget for the process, so
//! it is one test, the only one in its binary.
#![cfg(feature = "cuda")]

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu;
use multilinear::mle::Mle;

/// A column of `rows` words below `bound` (0 for u64::MAX), its last word
/// `bound − 1` so its width is the bound's.
fn column(rows: usize, bound: u64, salt: u64) -> Mle<F> {
    Mle::new(
        (0..rows as u64)
            .map(|i| {
                let w = (i ^ salt).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                let v = match bound {
                    0 => w,
                    b if i + 1 == rows as u64 => b - 1,
                    b => w % b,
                };
                FieldElement::<F>::from_raw(v)
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn a_narrow_store_holds_the_wide_stores_words_and_its_refusal_is_no_argue_fallback() {
    gpu::set_up_device();
    assert!(gpu::reserve_budget() > 0, "needs a card");
    // Three heights; the tallest's 10 columns are 40 M cells, more than one
    // run (2^25 cells), so the cut splits it; every width, and words above p.
    let mut columns = Vec::new();
    for (k, bound) in [1u64 << 8, 1 << 16, 1 << 32, 0, 2].into_iter().enumerate() {
        columns.push(column(1 << 12, bound, k as u64));
    }
    for k in 0..10u64 {
        let bound = [1u64 << 8, 1 << 32, 1 << 16, 0][k as usize % 4];
        columns.push(column(1 << 22, bound, 100 + k));
    }
    for k in 0..3u64 {
        columns.push(column(1 << 10, 1 << 8, 200 + k));
    }
    let refs: Vec<&Mle<F>> = columns.iter().collect();
    let lens: Vec<usize> = refs.iter().map(|c| c.len()).collect();
    assert!(
        multilinear::narrow::store_runs(&lens, multilinear::narrow::STORE_RUN_CELLS).len() > 3,
        "the tallest height is cut"
    );
    let wide = gpu::upload_store_columns(&refs).expect("the wide store goes up");
    let (narrow, stamp) =
        gpu::upload_store_columns_narrow(&refs).expect("the narrow store goes up");
    let (a, b) = (
        wide.download().expect("the wide store reads back"),
        narrow.download().expect("the narrow store reads back"),
    );
    assert_eq!(a.len(), columns.len(), "one span a column");
    assert!(a == b, "the narrow store holds the wide store's words");
    let words: u64 = lens.iter().map(|&n| n as u64 * 8).sum();
    assert!(
        stamp.packed_bytes > 0 && stamp.packed_bytes < words,
        "fewer bytes crossed: {} of {words}",
        stamp.packed_bytes
    );
    println!(
        "NARROW STORE: {} B packed of {words} B · pack {:.4}s · up {:.4}s",
        stamp.packed_bytes, stamp.pack_secs, stamp.upload_secs
    );
    // The same runs copied through the pinned staging: the same words.
    let packed = gpu::pack_store(&refs).expect("the columns pack");
    let (staged, _) = gpu::upload_packed_store_staged(&packed).expect("the staged store goes up");
    assert!(
        staged.download().expect("the staged store reads back") == a,
        "the staged narrow store holds the wide store's words"
    );
    drop((wide, narrow, staged));

    // Into a promise taken before: the same words, wide and narrow, and the
    // ledger keeps exactly the words (the narrow staging's part given back).
    let staging =
        multilinear::narrow::store_staging_bound(&lens, multilinear::narrow::STORE_RUN_CELLS);
    let before = gpu::ledger_reserved();
    let room = gpu::reserve_room(words).expect("the card promises the words");
    let wide_in =
        gpu::upload_store_columns_in(&refs, room).expect("the wide store goes into its promise");
    assert_eq!(
        gpu::ledger_reserved() - before,
        words,
        "wide: the promise is the words"
    );
    assert!(
        wide_in.download().expect("reads back") == a,
        "wide into a promise: the same words"
    );
    drop(wide_in);
    assert_eq!(
        gpu::ledger_reserved(),
        before,
        "the store's promise goes with it"
    );
    let room = gpu::reserve_room(words + staging).expect("the card promises words and staging");
    let (narrow_in, _) = gpu::upload_store_columns_narrow_in(&refs, room)
        .expect("the narrow store goes into its promise");
    assert_eq!(
        gpu::ledger_reserved() - before,
        words,
        "narrow: the staging's part given back once the runs are up"
    );
    assert!(
        narrow_in.download().expect("reads back") == a,
        "narrow into a promise: the same words"
    );
    drop(narrow_in);
    let short = gpu::reserve_room(words - 8).expect("a short promise");
    assert!(
        gpu::upload_store_columns_in(&refs, short).is_none(),
        "a promise short of the words takes nothing"
    );
    assert_eq!(gpu::ledger_reserved(), before, "every promise given back");

    // Room for 1 MiB more than the ledger holds now: the store is refused.
    let budget = gpu::set_reserve_budget(gpu::ledger_reserved() + (1 << 20));
    let before = gpu::argue_device_fallbacks();
    let refused = gpu::upload_store_columns_narrow(&refs);
    let after_store = gpu::argue_device_fallbacks();
    let argue = gpu::upload_columns(&refs);
    let after_argue = gpu::argue_device_fallbacks();
    gpu::set_reserve_budget(budget);
    assert!(refused.is_none(), "the narrow store is refused");
    assert_eq!(
        after_store, before,
        "a refused narrow store is not an argue fallback"
    );
    assert!(argue.is_none(), "the argue's upload is refused too");
    assert_eq!(after_argue, before + 1, "the argue's refusal is one");
}
