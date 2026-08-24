//! Device-vs-host streaming-MMCS leaf hashing at the wrap proof's dominant
//! shapes — the measurement behind the batched-commit-on-GPU verdict.
//!
//! Not a correctness test (`mmcs_tree_parity` is): both arms hash the same
//! buffers and the roots are asserted equal, but the OUTPUT is the timing
//! table this prints. Ignored by default — it allocates tens of GB and needs
//! a GPU; run explicitly:
//!
//! ```text
//! cargo test -p math-cuda --release --test bench_mmcs_wrap_shape -- --ignored --nocapture
//! ```
//!
//! Shapes: the measured wrap proof's two dominant matrices (its census of
//! record) — the LFM_BLAKE3 chip's main matrix (3056 base columns) and its
//! aux (631 ext3 columns), both at LDE height 2^20 (2^19 row-pair leaves).
//! Together they are ~78% of the wrap's committed cells, so the leaf-hash
//! delta here is the bulk of the commit-phase delta.

use std::time::Instant;

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math_cuda::DeviceHash;
use math_cuda::mmcs::MmcsGroupHasher;
use stark::config::Blake3StarkHash;
use stark::fri::mmcs::{BorrowedMatrix, StreamingMmcsBuilder};

type Fp = FieldElement<GoldilocksField>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

const LOG_LDE: usize = 20;
const MAIN_COLS: usize = 3056;
const AUX_COLS: usize = 631;

/// Cheap deterministic fill — content is irrelevant to hashing throughput,
/// generation must not dominate the setup.
fn fill_u64(n: usize, seed: u64) -> Vec<u64> {
    let mut v = Vec::with_capacity(n);
    let mut x = seed | 1;
    for _ in 0..n {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Keep values in the field so canonicalization is a no-op either way.
        v.push(x % 0xFFFF_FFFF_0000_0001);
    }
    v
}

fn host_root_base(data: &[Fp], cols: usize) -> [u8; 32] {
    let dims = vec![(LOG_LDE, cols)];
    let mut b = StreamingMmcsBuilder::<GoldilocksField, Blake3StarkHash>::new(&dims);
    let src = vec![BorrowedMatrix::RowMajorNatural {
        data,
        stride: cols,
        col_start: 0,
        width: cols,
        log_height: LOG_LDE,
    }];
    b.absorb(&src, 0);
    b.finish().root()
}

fn host_root_ext3(data: &[Fp3], cols: usize) -> [u8; 32] {
    let dims = vec![(LOG_LDE, cols)];
    let mut b =
        StreamingMmcsBuilder::<Degree3GoldilocksExtensionField, Blake3StarkHash>::new(&dims);
    let src = vec![BorrowedMatrix::RowMajorNatural {
        data,
        stride: cols,
        col_start: 0,
        width: cols,
        log_height: LOG_LDE,
    }];
    b.absorb(&src, 0);
    b.finish().root()
}

/// Device arm: H2D + leaf absorb + finalize + device climb + root D2H.
/// Symmetric with the host arm, whose wall also includes its digest climb.
fn device_root(raw: &[u64], lanes_stride: usize) -> ([u8; 32], std::time::Duration) {
    let be = math_cuda::device::backend().expect("GPU required for this bench");
    let stream = be.next_stream();
    let t = Instant::now();
    let mut hasher =
        MmcsGroupHasher::new(&stream, LOG_LDE as u64, DeviceHash::Blake3).expect("hasher");
    // SAFETY: every element is written by the staged copy before the kernel reads.
    let mut dev = unsafe { stream.alloc::<u64>(raw.len()) }.expect("alloc");
    math_cuda::device::htod_via(
        &stream,
        be.pinned_staging(),
        &be.ctx,
        raw,
        &mut dev.as_view_mut(),
    )
    .expect("pinned H2D");
    hasher
        .absorb_row_major(&stream, &dev, lanes_stride as u64, 0, lanes_stride as u64)
        .expect("absorb");
    drop(dev);
    let digests = hasher.finalize(&stream).expect("finalize");
    let mut groups: Vec<Option<math_cuda::CudaSlice<u8>>> = (0..=LOG_LDE).map(|_| None).collect();
    groups[LOG_LDE] = Some(digests);
    let nodes = math_cuda::mmcs::build_mmcs_tree_on_device(&stream, &groups, DeviceHash::Blake3)
        .expect("device climb");
    let root = math_cuda::mmcs::read_mmcs_root(&stream, &nodes).expect("root");
    stream.synchronize().expect("sync");
    (root, t.elapsed())
}

#[test]
#[ignore]
fn bench_mmcs_wrap_shape() {
    let rows = 1usize << LOG_LDE;

    println!("=== streaming-MMCS leaf hashing, wrap shapes, BLAKE3 ===");
    println!(
        "main: {} x {} base ({} GB); aux: {} x {} ext3 ({} GB)",
        rows,
        MAIN_COLS,
        rows * MAIN_COLS * 8 / (1 << 30),
        rows,
        AUX_COLS,
        rows * AUX_COLS * 24 / (1 << 30),
    );

    // ---- main matrix (base field) ----
    let raw = fill_u64(rows * MAIN_COLS, 7);
    // SAFETY: FieldElement<GoldilocksField> is one u64.
    let felts: &[Fp] = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const Fp, raw.len()) };

    for round in 0..2 {
        let t = Instant::now();
        let host = host_root_base(felts, MAIN_COLS);
        let host_wall = t.elapsed();
        let (dev_root, dev_wall) = device_root(&raw, MAIN_COLS);
        assert_eq!(host, dev_root, "device tree must match the host root");
        println!(
            "MAIN round {round}: host {host_wall:?}  device(H2D+leaves+finalize) {dev_wall:?}  speedup x{:.1}",
            host_wall.as_secs_f64() / dev_wall.as_secs_f64()
        );
    }
    drop(raw);

    // ---- aux matrix (ext3) ----
    let raw3 = fill_u64(rows * AUX_COLS * 3, 11);
    let felts3: &[Fp3] =
        unsafe { std::slice::from_raw_parts(raw3.as_ptr() as *const Fp3, rows * AUX_COLS) };

    for round in 0..2 {
        let t = Instant::now();
        let host = host_root_ext3(felts3, AUX_COLS);
        let host_wall = t.elapsed();
        let (dev_root, dev_wall) = device_root(&raw3, AUX_COLS * 3);
        assert_eq!(host, dev_root, "device ext3 tree must match the host root");
        println!(
            "AUX round {round}: host {host_wall:?}  device(H2D+leaves+finalize) {dev_wall:?}  speedup x{:.1}",
            host_wall.as_secs_f64() / dev_wall.as_secs_f64()
        );
    }
}
