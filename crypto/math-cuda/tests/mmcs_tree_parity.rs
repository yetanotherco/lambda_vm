//! The device mixed-height MMCS must build the SAME tree as the host
//! `stark::fri::mmcs::MixedMmcs` — same root, same authentication paths, so a
//! proof committed on GPU is opened and verified by the same verifier as one
//! committed on CPU.
//!
//! ⚠ **This file has never been executed.** It was written on a machine with no
//! GPU and no nvcc, where `math-cuda` compiles against empty cubin stubs and
//! every device call falls back or fails. It compiles and it lints; nothing here
//! is evidence that the kernels are correct. Run it on a rented box — the exact
//! commands are in `RESUME-MMCS-INT.md` — before any claim that the batched GPU
//! path works.
//!
//! What each test is FOR, so a failure says something:
//!
//! - `single_matrix_mmcs_root_matches_the_per_table_tree` — the degenerate case.
//!   A one-matrix MMCS is the existing row-pair tree, so this failing means the
//!   absorb kernel's byte order or bit-reversal is wrong, independently of
//!   anything mixed-height.
//! - `mixed_height_root_matches_the_host` — the climb with injection. This is
//!   the kernel that has no CPU counterpart to have been debugged against.
//! - `absorption_order_is_bound` — the leaf concatenates matrices in INPUT
//!   order; two matrices absorbed the other way round must give a different root.
//!   Without this, an order bug is invisible whenever the widths happen to match.
//! - `paths_match_the_host_at_every_query` — the reason the device tree uses the
//!   standard heap layout at all: `merkle_gather_paths` unchanged must return
//!   the host's `MixedOpening::proof`.

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;
use stark::config::KeccakStarkHash;
use stark::fri::mmcs::{LeafSource, MixedMmcs};

type Fp = FieldElement<GoldilocksField>;
type Mmcs = MixedMmcs<GoldilocksField, KeccakStarkHash>;

/// Bit-reversed row-major matrices, the layout the MMCS commits and the layout
/// `mmcs_absorb_row_pair_row_major` reads (the kernel bit-reverses internally, so
/// the device buffer holds the matrix in NATURAL order).
struct Matrices {
    /// `(natural-order row-major data, log_height, width)`.
    mats: Vec<(Vec<Fp>, usize, usize)>,
}

impl LeafSource<GoldilocksField> for Matrices {
    fn num_matrices(&self) -> usize {
        self.mats.len()
    }
    fn log_height(&self, m: usize) -> usize {
        self.mats[m].1
    }
    fn width(&self, m: usize) -> usize {
        self.mats[m].2
    }
    fn append_row(&self, m: usize, bitrev_row: usize, out: &mut Vec<Fp>) {
        let (data, log_height, width) = &self.mats[m];
        let natural = math::fft::bit_reversing::reverse_index(bitrev_row, 1u64 << log_height);
        out.extend_from_slice(&data[natural * width..(natural + 1) * width]);
    }
}

fn matrix(log_height: usize, width: usize, seed: u64) -> (Vec<Fp>, usize, usize) {
    let num_rows = 1usize << log_height;
    let data = (0..num_rows * width)
        .map(|i| Fp::from(seed.wrapping_mul(1_000_003).wrapping_add(i as u64) | 1))
        .collect();
    (data, log_height, width)
}

fn raw(data: &[Fp]) -> Vec<u64> {
    data.iter().map(|x| *x.value()).collect()
}

/// Build the tree on device from `specs`, absorbing matrices in input order and
/// freeing each matrix's device buffer before the next — the residency policy the
/// whole design exists for.
fn device_tree(mats: &Matrices) -> ([u8; 32], Vec<u8>) {
    let be = math_cuda::device::backend().expect("a GPU box: no backend means nothing to test");
    let stream = be.next_stream();

    let h_max = (0..mats.num_matrices())
        .map(|m| mats.log_height(m))
        .max()
        .expect("non-empty");

    let mut group_digests: Vec<Option<cudarc::driver::CudaSlice<u8>>> =
        (0..=h_max).map(|_| None).collect();

    for h in 1..=h_max {
        let group: Vec<usize> = (0..mats.num_matrices())
            .filter(|&m| mats.log_height(m) == h)
            .collect();
        if group.is_empty() {
            continue;
        }
        let mut hasher = math_cuda::mmcs::MmcsGroupHasher::new(&stream, h as u64)
            .expect("group sponge allocation");
        for &m in &group {
            let (data, _, width) = &mats.mats[m];
            let dev = stream.clone_htod(&raw(data)).expect("H2D");
            hasher
                .absorb_row_major(&stream, &dev, *width as u64, 0, *width as u64)
                .expect("absorb");
            // The point of the streaming build: this matrix is done with.
            drop(dev);
        }
        group_digests[h] = Some(hasher.finalize(&stream).expect("finalize"));
    }

    let nodes = math_cuda::mmcs::build_mmcs_tree_on_device(&stream, &group_digests)
        .expect("device tree build");
    let root = math_cuda::mmcs::read_mmcs_root(&stream, &nodes).expect("root readback");
    let all = stream.memcpy_dtov(&nodes).expect("node readback");
    (root, all)
}

#[test]
fn single_matrix_mmcs_root_matches_the_per_table_tree() {
    let mats = Matrices {
        mats: vec![matrix(6, 5, 7)],
    };
    let (device_root, _) = device_tree(&mats);
    assert_eq!(
        device_root,
        Mmcs::commit(&mats).root(),
        "a one-matrix MMCS must be the existing row-pair tree, byte for byte"
    );
}

#[test]
fn mixed_height_root_matches_the_host() {
    // Two matrices at the tallest height (so the base group batches), one
    // injected mid-climb, one injected near the terminal.
    let mats = Matrices {
        mats: vec![
            matrix(7, 3, 11),
            matrix(7, 6, 23),
            matrix(5, 2, 41),
            matrix(2, 4, 59),
        ],
    };
    let (device_root, _) = device_tree(&mats);
    assert_eq!(
        device_root,
        Mmcs::commit(&mats).root(),
        "the device climb with injection must reproduce the host tree"
    );
}

#[test]
fn absorption_order_is_bound() {
    let forward = Matrices {
        mats: vec![matrix(6, 3, 11), matrix(6, 3, 23)],
    };
    let reversed = Matrices {
        mats: vec![matrix(6, 3, 23), matrix(6, 3, 11)],
    };
    let (forward_root, _) = device_tree(&forward);
    let (reversed_root, _) = device_tree(&reversed);

    assert_eq!(forward_root, Mmcs::commit(&forward).root());
    assert_eq!(reversed_root, Mmcs::commit(&reversed).root());
    assert_ne!(
        forward_root, reversed_root,
        "two same-shape matrices absorbed in the other order must commit a \
         different tree — input order is part of the commitment"
    );
}

#[test]
fn paths_match_the_host_at_every_query() {
    let mats = Matrices {
        mats: vec![matrix(6, 3, 11), matrix(6, 2, 23), matrix(4, 5, 41)],
    };
    let host = Mmcs::commit(&mats);
    let (device_root, _) = device_tree(&mats);
    assert_eq!(device_root, host.root());

    let be = math_cuda::device::backend().expect("a GPU box");
    let stream = be.next_stream();
    let (_, nodes_host) = device_tree(&mats);
    let nodes = stream.clone_htod(&nodes_host).expect("H2D nodes");

    let leaves_len = 1usize << (host.h_max() - 1);
    let positions: Vec<u32> = (0..leaves_len as u32).collect();
    let depth = host.h_max() - 1;
    let paths = math_cuda::merkle::gather_merkle_paths_dev(&nodes, leaves_len, &positions, &stream)
        .expect("path gather");

    for iota in 0..leaves_len {
        let expected = host.open_batch(iota, &mats).proof.merkle_path;
        for (level, node) in expected.iter().enumerate() {
            let start = (iota * depth + level) * 32;
            assert_eq!(
                &paths[start..start + 32],
                &node[..],
                "query {iota}, level {level}: the device path must be the host path — \
                 `merkle_gather_paths` is reused precisely because the layouts agree"
            );
        }
    }
}
