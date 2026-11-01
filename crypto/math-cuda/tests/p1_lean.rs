//! The Poseidon1 kernels' `_lean` twins
//! (`LAMBDA_VM_P1_LEAN=1`) write today's bytes in today's slots, on the card.
//!
//! The permutation itself is checked against the Rust host reference with the
//! other variants (`p1w16_device`, variant `l2`), and on the host against
//! `v2` and the field over edge words (`make test-p1w16-host-kat`). Here the
//! production twins run beside today's kernels, both sets loaded in one
//! process (`p1_stark::load_kernels`), over codewords of raw words with the
//! edge values planted, and over whole 4-ary trees. Needs a GPU (the box):
//!
//!   cargo test -p math-cuda --release --test p1_lean

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use math_cuda::device::backend;
use math_cuda::p1_stark::{Kernels, TAIL_MAX_PARENTS, load_kernels, tree_nodes};

const P: u64 = 0xFFFF_FFFF_0000_0001;
const BLOCK_DIM: u32 = 128;

/// The words a lean addition chain must survive: 0, 1, p − 1, p, p + 1, the
/// neighbours of 2^64 − 2^32, all ones, 2^63, 2^32 and its neighbour.
const EDGES: [u64; 13] = [
    0,
    1,
    P - 1,
    P,
    P + 1,
    0xFFFF_FFFE_FFFF_FFFF,
    0xFFFF_FFFF_0000_0002,
    u64::MAX - 1,
    u64::MAX,
    1 << 63,
    1 << 32,
    0xFFFF_FFFF,
    0x8000_0000_8000_0000,
];

/// Raw words (SplitMix64, any value in [0, 2^64)) with an edge word at every
/// fifth slot, and one stretch of all ones.
fn words(seed: u64, n: usize) -> Vec<u64> {
    let mut z = seed;
    let mut v: Vec<u64> = (0..n)
        .map(|i| {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            if i % 5 == 0 {
                EDGES[(i / 5) % EDGES.len()]
            } else {
                x ^ (x >> 31)
            }
        })
        .collect();
    for w in v.iter_mut().take(256) {
        *w = u64::MAX;
    }
    v
}

fn cfg(threads: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((threads as u32).div_ceil(BLOCK_DIM).max(1), 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn read(stream: &Arc<CudaStream>, s: &CudaSlice<u8>) -> Vec<u8> {
    stream.synchronize().unwrap();
    stream.clone_dtoh(s).unwrap()
}

/// The leaf kernels of one set over `codeword` (`num_leaves` leaves of
/// `block` values; ext3 reads three limbs a value), the bytes back.
fn coset_leaves(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    codeword: &CudaSlice<u64>,
    num_leaves: u64,
    block: u64,
) -> [Vec<u8>; 3] {
    let leaf_bits = num_leaves.trailing_zeros();
    let mut out = stream.alloc_zeros::<u8>(num_leaves as usize * 32).unwrap();
    let mut got = Vec::new();
    // SAFETY: each kernel reads at most `3 · num_leaves · block` values of the
    // codeword (allocated so) and writes `num_leaves` digests.
    unsafe {
        stream
            .launch_builder(&k.leaves_base_coset_rev)
            .arg(codeword)
            .arg(&num_leaves)
            .arg(&block)
            .arg(&leaf_bits)
            .arg(&mut out)
            .launch(cfg(num_leaves))
            .unwrap();
    }
    got.push(read(stream, &out));
    for f in [&k.leaves_base_coset, &k.leaves_ext3_coset] {
        stream.memset_zeros(&mut out).unwrap();
        unsafe {
            stream
                .launch_builder(f)
                .arg(codeword)
                .arg(&num_leaves)
                .arg(&block)
                .arg(&mut out)
                .launch(cfg(num_leaves))
                .unwrap();
        }
        got.push(read(stream, &out));
    }
    [got.remove(0), got.remove(0), got.remove(0)]
}

/// One set's column-row and row-pair leaves over `num_cols` columns of
/// `2^log_rows` rows.
fn column_leaves(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    cols: &CudaSlice<u64>,
    num_cols: u64,
    log_rows: u64,
) -> [Vec<u8>; 2] {
    let rows = 1u64 << log_rows;
    let mut out = stream.alloc_zeros::<u8>(rows as usize * 32).unwrap();
    // SAFETY: both read `num_cols · rows` values at column stride `rows` and
    // write at most `rows` digests.
    unsafe {
        stream
            .launch_builder(&k.leaves_cols_row)
            .arg(cols)
            .arg(&rows)
            .arg(&num_cols)
            .arg(&rows)
            .arg(&log_rows)
            .arg(&mut out)
            .launch(cfg(rows))
            .unwrap();
    }
    let row = read(stream, &out);
    stream.memset_zeros(&mut out).unwrap();
    unsafe {
        stream
            .launch_builder(&k.leaves_cols_pair)
            .arg(cols)
            .arg(&rows)
            .arg(&num_cols)
            .arg(&rows)
            .arg(&log_rows)
            .arg(&mut out)
            .launch(cfg(rows / 2))
            .unwrap();
    }
    [row, read(stream, &out)]
}

/// A whole 4-ary tree over `leaves` digests with one set's level and tail
/// kernels (`p1_stark::build_inner_tree_levels`'s walk), the node buffer back.
fn tree(stream: &Arc<CudaStream>, k: &Kernels, leaf_bytes: &[u8]) -> Vec<u8> {
    let leaves = leaf_bytes.len() / 32;
    let total = tree_nodes(leaves);
    let mut host = vec![0u8; total * 32];
    host[(total - leaves) * 32..].copy_from_slice(leaf_bytes);
    let mut nodes = stream.clone_htod(&host).unwrap();
    let mut child_off = (total - leaves) as u64;
    let mut n_children = leaves as u64;
    while n_children > 1 {
        let n_parents = n_children.div_ceil(4);
        // SAFETY: the node buffer holds the whole tree (`tree_nodes`).
        unsafe {
            if n_parents <= TAIL_MAX_PARENTS {
                let tail = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (TAIL_MAX_PARENTS as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                stream
                    .launch_builder(&k.merkle_tail4)
                    .arg(&mut nodes)
                    .arg(&child_off)
                    .arg(&n_children)
                    .launch(tail)
                    .unwrap();
                break;
            }
            let parent_off = child_off - n_parents;
            stream
                .launch_builder(&k.merkle_level4)
                .arg(&mut nodes)
                .arg(&child_off)
                .arg(&n_children)
                .arg(&parent_off)
                .arg(&n_parents)
                .launch(cfg(n_parents))
                .unwrap();
            child_off = parent_off;
            n_children = n_parents;
        }
    }
    read(stream, &nodes)
}

/// ★ Every twin writes today's bytes: coset leaves (coset and natural order,
/// ext3), column-row and row-pair leaves (12, 13 and 25 columns: whole
/// blocks, a partial tail, two tails), and whole trees over 2^14 leaves and
/// over a ragged 5,000. One planted change moves exactly one leaf under the
/// twin too, so the comparison can fail.
#[test]
fn the_lean_twins_write_todays_bytes() {
    let be = backend().expect("a GPU");
    let stream = be.ctx.new_stream().expect("stream");
    let today = load_kernels(false).expect("today's kernels");
    let lean = load_kernels(true).expect("the lean twins");
    let (num_leaves, block) = (1u64 << 14, 64u64);
    let mut host = words(0x1ea4, (3 * num_leaves * block) as usize);
    let codeword = stream.clone_htod(&host).unwrap();
    let a = coset_leaves(&stream, &today, &codeword, num_leaves, block);
    let b = coset_leaves(&stream, &lean, &codeword, num_leaves, block);
    for (name, (x, y)) in ["coset order", "natural order", "ext3"]
        .iter()
        .zip(a.iter().zip(&b))
    {
        assert_eq!(x, y, "the lean coset leaves ({name}) differ from today's");
    }
    let log_rows = 12;
    for num_cols in [12u64, 13, 25] {
        let cols = stream
            .clone_htod(&words(num_cols, (num_cols as usize) << log_rows))
            .unwrap();
        let x = column_leaves(&stream, &today, &cols, num_cols, log_rows);
        let y = column_leaves(&stream, &lean, &cols, num_cols, log_rows);
        assert_eq!(
            x[0], y[0],
            "the lean column-row leaves differ ({num_cols} columns)"
        );
        assert_eq!(
            x[1], y[1],
            "the lean row-pair leaves differ ({num_cols} columns)"
        );
    }
    assert_eq!(
        tree(&stream, &today, &a[0]),
        tree(&stream, &lean, &a[0]),
        "the lean tree over 2^14 leaves differs"
    );
    let ragged = &a[1][..5000 * 32];
    assert_eq!(
        tree(&stream, &today, ragged),
        tree(&stream, &lean, ragged),
        "the lean tree over 5,000 leaves differs"
    );

    // The comparison can fail: one value of leaf 7's coset moved.
    host[7 + 9 * num_leaves as usize] ^= 1 << 40;
    let moved = stream.clone_htod(&host).unwrap();
    let c = coset_leaves(&stream, &lean, &moved, num_leaves, block);
    let differ = (0..num_leaves as usize)
        .filter(|&j| c[1][32 * j..32 * j + 32] != a[1][32 * j..32 * j + 32])
        .count();
    assert_eq!(differ, 1, "one moved value moves one natural-order leaf");
}

/// The measurement kernels' ns a permutation (leaf, 4-ary node, grind) at
/// every variant over a 2^29 codeword, `l2` beside `v2`: the mechanism row of
/// the lean kernels' measurement (`BENCH` lines).
#[test]
#[ignore = "box tier: a measurement"]
fn the_variants_ns_a_permutation() {
    math_cuda::p1w16::microbench(29, 7, 1 << 28).expect("microbench");
}
