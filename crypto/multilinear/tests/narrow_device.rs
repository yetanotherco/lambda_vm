//! Narrow storage on the card: the column-major widen, the pack of a resident
//! run, and a store uploaded from packed tables, each against the host.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda,parallel --test narrow_device -- --test-threads=1
//! ```
//!
//! Needs a GPU.
#![cfg(feature = "cuda")]

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu::{self, TableColumns};
use multilinear::mle::Mle;
use multilinear::narrow::NarrowColumns;

/// Every width boundary: 0, 2^8 − 1, 2^8, 2^16 − 1, 2^16, 2^32 − 1, 2^32, the
/// Goldilocks p − 1, a non-canonical word above p, and u64::MAX.
const EDGES: [u64; 10] = [
    0,
    0xff,
    0x100,
    0xffff,
    0x1_0000,
    0xffff_ffff,
    0x1_0000_0000,
    0xffff_ffff_0000_0000,
    0xffff_ffff_0000_0001,
    u64::MAX,
];

/// One column per edge over `rows` rows, column `c` topping out at `EDGES[c]`
/// (in its last row), shifted by `salt` so two tables differ.
fn edge_columns(rows: usize, salt: u64) -> Vec<Vec<u64>> {
    EDGES
        .iter()
        .map(|&edge| {
            (0..rows)
                .map(|r| {
                    if r == rows - 1 {
                        edge
                    } else {
                        ((r as u64) ^ salt).wrapping_mul(0x9e37_79b9_7f4a_7c15) % (edge / 2 + 1)
                    }
                })
                .collect()
        })
        .collect()
}

fn mles(columns: &[Vec<u64>]) -> Vec<Mle<F>> {
    columns
        .iter()
        .map(|c| Mle::new(c.iter().map(|&w| FieldElement::<F>::from_raw(w)).collect()).unwrap())
        .collect()
}

fn raw(columns: &[Vec<u64>]) -> Vec<&[u64]> {
    columns.iter().map(Vec::as_slice).collect()
}

/// The card's column-major widen gives back every word of a host pack, and the
/// row-major widen is the same words transposed.
#[test]
fn the_card_widens_column_major_at_every_width_boundary() {
    let rows = (1usize << 12) + 1024;
    let columns = edge_columns(rows, 0);
    let packed = NarrowColumns::pack(&raw(&columns)).unwrap();
    assert_eq!(packed.widths(), &[1, 1, 2, 2, 4, 4, 8, 8, 8, 8]);
    let cols = packed.cols();
    let col_major = math_cuda::narrow::widen_col_major_to_host(
        packed.data(),
        packed.offsets(),
        packed.widths(),
        rows,
        cols,
    )
    .unwrap();
    assert_eq!(col_major, columns.concat());
    let row_major = math_cuda::narrow::widen_to_host(
        packed.data(),
        packed.offsets(),
        packed.widths(),
        rows,
        cols,
    )
    .unwrap();
    for (c, column) in columns.iter().enumerate() {
        let back: Vec<u64> = row_major.iter().skip(c).step_by(cols).copied().collect();
        assert_eq!(&back, column, "column {c}");
    }
}

/// The card packs each table's run of a resident store into the bytes the
/// host packs, for two tables of different heights in one store.
#[test]
fn the_card_packs_a_resident_run_as_the_host_does() {
    let tall = edge_columns(1 << 13, 1);
    let short = edge_columns(1 << 9, 2);
    let (tall_mle, short_mle) = (mles(&tall), mles(&short));
    let all: Vec<&Mle<F>> = tall_mle.iter().chain(&short_mle).collect();
    let store = gpu::upload_columns(&all).expect("the card takes the columns");
    for (first, columns) in [(0usize, &tall), (tall.len(), &short)] {
        let card = gpu::pack_resident(&store, first, columns.len()).expect("packed");
        assert_eq!(
            card,
            NarrowColumns::pack(&raw(columns)).unwrap(),
            "from {first}"
        );
    }
    // Not a run: a span across the two heights.
    assert!(gpu::pack_resident(&store, tall.len() - 1, 2).is_none());
}

/// A store uploaded from packed tables holds, column for column, the words a
/// store uploaded wide holds, packed and wide tables mixed; a wrong width map
/// widens to other words.
#[test]
fn a_store_uploaded_narrow_holds_the_wide_words() {
    let a = edge_columns(1 << 11, 3);
    let b = edge_columns(1 << 10, 4);
    let c = edge_columns(1 << 12, 5);
    let (a_mle, b_mle, c_mle) = (mles(&a), mles(&b), mles(&c));
    let (a_packed, c_packed) = (
        NarrowColumns::pack(&raw(&a)).unwrap(),
        NarrowColumns::pack(&raw(&c)).unwrap(),
    );
    let mixed = gpu::upload_tables::<F>(&[
        TableColumns::Narrow(&a_packed),
        TableColumns::Wide(&b_mle),
        TableColumns::Narrow(&c_packed),
    ])
    .expect("the card takes the tables");
    let all: Vec<&Mle<F>> = a_mle.iter().chain(&b_mle).chain(&c_mle).collect();
    let wide = gpu::upload_columns(&all).expect("the card takes the columns");
    let held = mixed.download().unwrap();
    assert_eq!(held, wide.download().unwrap());
    assert_eq!(held, [a.clone(), b, c].concat());
    // The run checks the readers make hold for the widened tables too.
    assert!(gpu::pack_resident(&mixed, 0, a.len()).is_some());

    let mut wrong = a_packed.clone();
    assert!(wrong.fault_width_map());
    let broken = gpu::upload_tables::<F>(&[TableColumns::Narrow(&wrong)]).unwrap();
    assert_ne!(broken.download().unwrap(), a);
}
