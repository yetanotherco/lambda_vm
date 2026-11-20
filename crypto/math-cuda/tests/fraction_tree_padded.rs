//! A fraction tree that does not carry its padding against one that does.
//!
//! Runs on the merge-queue GPU box via `make test-math-cuda` — the tree needs a
//! real device, like the other tests here.
//!
//! The interactions are padded up to a power of two with the fraction `0/1`,
//! and the widest precompiles are barely over the power below, so that padding
//! is half the layer. The tree built from the truncated layer has to be the
//! same tree: what is compared is the output fraction, which is a product and
//! sum chain over every cell of every level — a level that differed anywhere
//! would have to collide in the field to come out equal here.

use cudarc::driver::CudaStream;
use math_cuda::device::backend;
use math_cuda::gkr::DeviceFractionTree;
use std::sync::Arc;

/// Fractions with no structure a kernel could accidentally satisfy.
fn layer(cells: usize, seed: u64) -> Vec<u64> {
    (0..cells as u64)
        .flat_map(|i| {
            let mix =
                |k: u64| (i.wrapping_mul(6364136223846793005 + k).wrapping_add(seed) >> 9) + 1;
            [mix(1), mix(7), mix(13)]
        })
        .collect()
}

/// `real` fractions of a cube of `full`, with the rest written out as the
/// `0/1` the padding is.
fn padded(real: usize, full: usize, seed: u64) -> (Vec<u64>, Vec<u64>) {
    let mut p = layer(real, seed);
    let mut q = layer(real, seed ^ 0x5eed);
    for _ in real..full {
        p.extend_from_slice(&[0, 0, 0]);
        q.extend_from_slice(&[1, 0, 0]);
    }
    (p, q)
}

fn upload(stream: &Arc<CudaStream>, values: &[u64]) -> cudarc::driver::CudaSlice<u64> {
    math_cuda::device::htod_or_trim(stream, values).expect("upload")
}

#[test]
fn a_tree_that_drops_its_padding_is_the_same_tree() {
    let Ok(be) = backend() else {
        eprintln!("no device; skipping");
        return;
    };
    // 5 interactions over 8 rows: rounded up to 8 slots, so three eighths of
    // the cube is padding — the shape of a precompile, in miniature.
    for (interactions, rows) in [(5usize, 8usize), (3, 4), (7, 32), (1, 16)] {
        let slots = interactions.next_power_of_two();
        let full = slots * rows;
        let real = interactions * rows;
        let (p, q) = padded(real, full, 0x1234 + interactions as u64);

        let stream = be.next_stream();
        let carried = DeviceFractionTree::from_device(
            stream.clone(),
            upload(&stream, &p),
            upload(&stream, &q),
        )
        .expect("the tree that carries its padding");

        let stream = be.next_stream();
        let dropped = DeviceFractionTree::from_padded_input(
            stream.clone(),
            upload(&stream, &p[..real * 3]),
            upload(&stream, &q[..real * 3]),
            real,
            full.trailing_zeros() as usize,
        )
        .expect("the tree that drops it");

        assert_eq!(
            carried.num_layers(),
            dropped.num_layers(),
            "{interactions}x{rows}: same tree, same levels"
        );
        assert_eq!(
            carried.output().expect("output"),
            dropped.output().expect("output"),
            "{interactions}x{rows}: the output fraction differs"
        );
        assert!(carried.holds_input(), "the first one keeps its input layer");
        assert!(!dropped.holds_input(), "the second one gives it back");
    }
}
