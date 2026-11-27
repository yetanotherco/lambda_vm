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

/// Both input height parities, with the input carried whole, carried without
/// padding, or discarded. Compare all the levels read by the double ladder.
#[test]
fn the_double_ladder_tree_keeps_exactly_the_levels_it_reads() {
    use math_cuda::gkr::{self, InputPlan};

    let be = backend().expect("a CUDA device");
    for (interactions, rows) in [(1usize, 1usize), (1, 2), (1, 4), (3, 32), (5, 32)] {
        let full = interactions.next_power_of_two() * rows;
        let real = interactions * rows;
        let height = full.trailing_zeros() as usize;
        let (p, q) = padded(real, full, 0x1234 + interactions as u64);
        let dense = DeviceFractionTree::build(&p, &q).expect("dense tree");
        let stream = be.next_stream();
        gkr::with_skip_middle(true, || {
            let sparse = DeviceFractionTree::build(&p, &q).expect("compact tree");
            assert_kept_levels(&dense, &sparse, height);
            if full >= 2 {
                let sparse = DeviceFractionTree::from_truncated_input(
                    stream.clone(),
                    upload(&stream, &p[..real * 3]),
                    upload(&stream, &q[..real * 3]),
                    real,
                    height,
                )
                .expect("compact truncated tree");
                assert!(sparse.holds_input());
                assert_kept_levels(&dense, &sparse, height);

                let sparse = DeviceFractionTree::from_padded_input(
                    stream.clone(),
                    upload(&stream, &p[..real * 3]),
                    upload(&stream, &q[..real * 3]),
                    real,
                    height,
                )
                .expect("compact tree that hands the input back");
                assert!(!sparse.holds_input());
                assert_kept_levels(&dense, &sparse, height);
            }
        });

        if full >= 4 {
            // A shifted column term per side and distinct constants.
            let sides = 2 * interactions;
            let columns = upload(&stream, &layer(rows, 97)[..rows]);
            let starts: Vec<u32> = (0..=sides as u32).collect();
            let terms: Vec<u64> = (0..sides).flat_map(|i| [0, (i % rows) as u64]).collect();
            let plan = InputPlan::upload(
                &stream,
                &starts,
                &terms,
                &layer(sides, 11),
                &layer(sides, 19),
            )
            .expect("column input plan");
            let (p, q) = gkr::input_from_columns(&stream, &columns.slice(..), rows, &plan, true)
                .expect("separate input pass");
            let dense = DeviceFractionTree::from_device(stream.clone(), p, q).expect("dense tree");
            gkr::with_skip_middle(true, || {
                for padding in [false, true] {
                    let ((p, q), top) = gkr::input_fold2_from_columns(
                        &stream,
                        &columns.slice(..),
                        rows,
                        &plan,
                        padding,
                    )
                    .expect("fused input pass");
                    let sparse = if padding {
                        DeviceFractionTree::from_device_top(stream.clone(), p, q, Some(top))
                    } else {
                        DeviceFractionTree::from_truncated_input_top(
                            stream.clone(),
                            p,
                            q,
                            real,
                            height,
                            Some(top),
                        )
                    }
                    .expect("compact tree from the fused input pass");
                    assert_kept_levels(&dense, &sparse, height);
                }
            });
        }
    }
}

fn assert_kept_levels(dense: &DeviceFractionTree, sparse: &DeviceFractionTree, height: usize) {
    assert_eq!(dense.num_layers(), sparse.num_layers());
    assert_eq!(dense.output().unwrap(), sparse.output().unwrap());
    for level in 0..height {
        assert_eq!(sparse.layer_num_vars(level), level);
        if level % 2 == 0 {
            assert_eq!(
                dense.layer_to_host(level).unwrap(),
                sparse.layer_to_host(level).unwrap()
            );
        } else {
            assert!(
                sparse.layer_to_host(level).is_err(),
                "stored odd level {level}"
            );
        }
    }
}
