//! Many small GKR layers' Gruen rounds in one launch (`gkr::gruen_round_batch`)
//! against each layer's own rounds, on the card.
//!
//! ```text
//! cargo test --release -p math-cuda --test gruen_batch -- --test-threads=1
//! ```
//!
//! Needs a GPU. Every round's sums are compared as field values and the
//! halves after the last round limb for limb, over layers of different heights stepped together
//! (each its own fold and `H(0)` pattern), and a batch the kernel cannot take
//! (a layer kept without its padding, a layer on another stream, a round over
//! more than `GRUEN_BATCH_QUARTER` indices) is refused before any launch.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use math_cuda::device::backend;
use math_cuda::gkr::{self, DeviceFractionTree, GruenLayer};

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

const P: u64 = 0xFFFF_FFFF_0000_0001;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn limb(&mut self) -> u64 {
        self.next() % P
    }

    fn ext(&mut self) -> [u64; 3] {
        [self.limb(), self.limb(), self.limb()]
    }
}

fn upload(stream: &Arc<CudaStream>, values: &[u64]) -> CudaSlice<u64> {
    math_cuda::device::htod_or_trim(stream, values).expect("upload")
}

/// A tree over random fractions of `2^vars` cells on `stream`, kept without
/// its padding (the last quarter `0/1`) when `kept`.
fn tree(vars: usize, kept: bool, stream: &Arc<CudaStream>, seed: u64) -> DeviceFractionTree {
    let full = 1usize << vars;
    let real = if kept { full - full / 4 } else { full };
    let mut rng = Rng(seed | 1);
    let mut p = Vec::with_capacity(full * 3);
    let mut q = Vec::with_capacity(full * 3);
    for i in 0..full {
        if i < real {
            p.extend_from_slice(&rng.ext());
            q.extend_from_slice(&rng.ext());
        } else {
            p.extend_from_slice(&[0, 0, 0]);
            q.extend_from_slice(&[1, 0, 0]);
        }
    }
    if kept {
        DeviceFractionTree::from_truncated_input(
            stream.clone(),
            upload(stream, &p[..real * 3]),
            upload(stream, &q[..real * 3]),
            real,
            vars,
        )
    } else {
        DeviceFractionTree::from_device(stream.clone(), upload(stream, &p), upload(stream, &q))
    }
    .expect("the tree")
}

/// Layer `layer` of `tree`'s Gruen rounds at a random point, `low` variables
/// left to the host.
fn open(tree: &DeviceFractionTree, layer: usize, low: usize, seed: u64) -> GruenLayer {
    let m = tree.layer_num_vars(layer) - 1;
    let mut rng = Rng(seed ^ 0x77);
    let point: Vec<u64> = (0..m).flat_map(|_| rng.ext()).collect();
    tree.layer_gruen(layer, &point, low).expect("a session")
}

/// Each case: input variables, the layer opened (counted from the input
/// layer down), and `low`.
const CASES: [(usize, usize, usize); 6] = [
    (8, 0, 2),
    (10, 0, 3),
    (10, 1, 2),
    (12, 0, 4),
    (12, 2, 1),
    (15, 0, 3),
];

/// ★ Six layers of different heights, each with its own fold, `λ` and
/// `H(0)` pattern, stepped together: every round's sums of the batch are each
/// layer's own round's values, and the halves its finish hands the host its
/// limbs;
/// layers drop out of the batch as their rounds end.
#[test]
fn a_batch_of_rounds_is_each_layers_own_rounds() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let be = backend().expect("a device");
    let stream = be.next_stream();
    let mut alone = Vec::new();
    let mut batched = Vec::new();
    let mut rngs = Vec::new();
    for (k, &(vars, down, low)) in CASES.iter().enumerate() {
        let seed = 0x5EED + k as u64;
        let a = tree(vars, false, &stream, seed);
        let b = tree(vars, false, &stream, seed);
        let layer = a.num_layers() - 1 - down;
        alone.push((open(&a, layer, low, seed), a));
        batched.push((open(&b, layer, low, seed), b));
        rngs.push(Rng(seed ^ 0xC4A11));
    }
    let lambdas: Vec<[u64; 3]> = rngs.iter_mut().map(Rng::ext).collect();
    let most = alone.iter().map(|(l, _)| l.rounds()).max().unwrap();
    let mut previous: Vec<Option<[u64; 3]>> = vec![None; CASES.len()];
    let mut batches = 0usize;
    for j in 0..most {
        let live: Vec<usize> = (0..CASES.len())
            .filter(|&k| j < alone[k].0.rounds())
            .collect();
        let want: Vec<bool> = live.iter().map(|&k| (j + k) % 3 == 1).collect();
        let today: Vec<Vec<u64>> = live
            .iter()
            .zip(&want)
            .map(|(&k, &h0)| {
                alone[k]
                    .0
                    .round(previous[k].as_ref(), &lambdas[k], h0)
                    .expect("a lone round")
            })
            .collect();
        let rounds: Vec<(Option<[u64; 3]>, [u64; 3], bool)> = live
            .iter()
            .zip(&want)
            .map(|(&k, &h0)| (previous[k], lambdas[k], h0))
            .collect();
        let mut layers: Vec<&mut GruenLayer> = batched
            .iter_mut()
            .enumerate()
            .filter(|(k, _)| live.contains(k))
            .map(|(_, (layer, _))| layer)
            .collect();
        let got = gkr::gruen_round_batch(&mut layers, &rounds).expect("a batch");
        batches += 1;
        // The same field values: a block reduces its own share of the cube,
        // so a limb may come back in its other representation (`v` or
        // `v + P`); the folds write the same limbs (the halves, below).
        let canonical = |v: &Vec<u64>| v.iter().map(|l| l % P).collect::<Vec<u64>>();
        for ((&k, x), y) in live.iter().zip(&today).zip(&got) {
            assert_eq!(canonical(x), canonical(y), "case {k}, round {j}: the sums");
        }
        for &k in &live {
            previous[k] = Some(rngs[k].ext());
        }
    }
    assert!(batches >= 10, "{batches} batches");
    for (k, ((a, _), (b, _))) in alone.iter_mut().zip(batched.iter_mut()).enumerate() {
        let last = previous[k];
        assert_eq!(
            a.finish(last.as_ref()).expect("a finish"),
            b.finish(last.as_ref()).expect("b finish"),
            "case {k}: the halves, raw"
        );
    }
}

/// A batch the kernel cannot take is refused before any launch: a layer kept
/// without its padding (its first round reads the padding), a layer on
/// another stream, and a round over more than `GRUEN_BATCH_QUARTER` indices.
#[test]
fn a_batch_the_kernel_cannot_take_is_refused() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let be = backend().expect("a device");
    let stream = be.next_stream();
    let lambda = [3, 5, 7];
    let first = (None, lambda, false);
    // Kept without its padding.
    let plain = tree(8, false, &stream, 1);
    let kept = tree(8, true, &stream, 2);
    let mut a = open(&plain, plain.num_layers() - 1, 2, 1);
    let mut b = open(&kept, kept.num_layers() - 1, 2, 2);
    assert!(!b.batchable(false, &stream), "a kept layer");
    let rounds = [first, first];
    assert!(gkr::gruen_round_batch(&mut [&mut a, &mut b], &rounds).is_err());
    // Another stream.
    let other = loop {
        let s = be.next_stream();
        if !Arc::ptr_eq(&s, &stream) {
            break s;
        }
    };
    let elsewhere = tree(8, false, &other, 3);
    let mut c = open(&elsewhere, elsewhere.num_layers() - 1, 2, 3);
    assert!(!c.batchable(false, &stream) && c.batchable(false, &other));
    let rounds = [first, first];
    assert!(gkr::gruen_round_batch(&mut [&mut a, &mut c], &rounds).is_err());
    // Too wide: a first round over 2^(m−1) indices past the bound.
    let vars = gkr::GRUEN_BATCH_QUARTER.trailing_zeros() as usize + 3;
    let wide = tree(vars, false, &stream, 4);
    let d = open(&wide, wide.num_layers() - 1, 2, 4);
    assert!(
        !d.batchable(false, &stream),
        "a round over 2^{} indices",
        vars - 2
    );
    // The refusals moved nothing: `a` still runs its first round alone.
    assert!(a.batchable(false, &stream));
    a.round(None, &lambda, false).expect("a lone round");
}
