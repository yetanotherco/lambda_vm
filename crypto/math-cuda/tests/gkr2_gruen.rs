//! A GKR step over two tree layers on the card (`math_cuda::gkr::Gruen2Layer`,
//! `sumcheck.cu`'s `gkr2_*`): every round's sums and the finish against a
//! host reference that composes the two folds, and an input layer kept
//! without its padding against the same layer stored whole.
//!
//! ```text
//! cargo test --release -p math-cuda --test gkr2_gruen -- --test-threads=1 --nocapture
//! ```
//!
//! Needs a GPU, like the other tests here; without one each test says so and
//! returns. The host KAT (`make test-gkr2-host-kat`) covers the kernels'
//! arithmetic on the host; this covers the driver's launches, the card's
//! warps, the slot and the kept layer's stored runs.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
use math::field::goldilocks::GoldilocksField as Gl;
use math_cuda::device::backend;
use math_cuda::gkr::{DeviceFractionTree, Gruen2Layer};
use math_cuda::round_slot::RoundSlot;

type E = FieldElement<Ext3>;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

const P: u64 = 0xFFFF_FFFF_0000_0001;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A raw limb, every eighth one a non-canonical representative in
    /// `[p, 2^64)` — what the card may hold.
    fn limb(&mut self) -> u64 {
        let v = self.next() % P;
        if self.next() & 7 == 0 && v < 0xFFFF_FFFF {
            v + P
        } else {
            v
        }
    }

    fn ext(&mut self) -> [u64; 3] {
        [self.limb(), self.limb(), self.limb()]
    }
}

fn value(limbs: &[u64]) -> E {
    let canonical = |l: u64| if l >= P { l - P } else { l };
    E::new([
        FieldElement::<Gl>::from_raw(canonical(limbs[0])),
        FieldElement::<Gl>::from_raw(canonical(limbs[1])),
        FieldElement::<Gl>::from_raw(canonical(limbs[2])),
    ])
}

/// A layer of `2^num_vars` cells a side whose first `real` are random
/// fractions and the rest the padding as today writes it (`p` 0, `q` 1).
fn layer(num_vars: usize, real: usize, seed: u64) -> (Vec<u64>, Vec<u64>) {
    let full = 1usize << num_vars;
    let mut rng = Rng(seed | 1);
    let mut p = Vec::with_capacity(full * 3);
    let mut q = Vec::with_capacity(full * 3);
    for _ in 0..real {
        p.extend_from_slice(&rng.ext());
        q.extend_from_slice(&rng.ext());
    }
    for _ in real..full {
        p.extend_from_slice(&[0, 0, 0]);
        q.extend_from_slice(&[1, 0, 0]);
    }
    (p, q)
}

fn upload(stream: &Arc<CudaStream>, values: &[u64]) -> CudaSlice<u64> {
    math_cuda::device::htod_or_trim(stream, values).expect("upload")
}

/// `eq(u, x)` over `{0,1}^|u|`, `u₀` in the high bit.
fn eq_evals(u: &[E]) -> Vec<E> {
    let mut table = vec![E::one()];
    for coordinate in u {
        let mut next = Vec::with_capacity(table.len() * 2);
        for v in &table {
            next.push(v * (E::one() - coordinate));
        }
        for v in &table {
            next.push(v * coordinate);
        }
        table = next;
    }
    table
}

/// The host's step over the same quarters: each round's `H(0..4)` by the
/// two folds composed, and the eight factors bound at the end.
struct Reference {
    /// `[p, q][quarter]`, the live cube of each.
    f: [[Vec<E>; 4]; 2],
    point: Vec<E>,
    lambda: E,
    low: usize,
}

impl Reference {
    fn new(p: &[u64], q: &[u64], point: &[E], lambda: E, low: usize) -> Self {
        let stride = p.len() / 12;
        let quarters = |side: &[u64]| -> [Vec<E>; 4] {
            std::array::from_fn(|j| {
                (0..stride)
                    .map(|x| value(&side[(j * stride + x) * 3..(j * stride + x) * 3 + 3]))
                    .collect()
            })
        };
        Self {
            f: [quarters(p), quarters(q)],
            point: point.to_vec(),
            lambda,
            low: low.min(point.len()),
        }
    }

    fn fold(&mut self, s: &E, width: usize) {
        for side in self.f.iter_mut() {
            for g in side.iter_mut() {
                for x in 0..width {
                    g[x] = &g[x] + s * (&g[x + width] - &g[x]);
                }
                g.truncate(width);
            }
        }
    }

    /// Round `j`'s `H(0..4)`, after folding by the last challenge.
    fn round(&mut self, j: usize, previous: Option<&E>) -> [E; 5] {
        let m = self.point.len();
        let quarter = 1usize << (m - j - 1);
        if let Some(s) = previous {
            self.fold(s, 2 * quarter);
        }
        let weight = eq_evals(&self.point[j + 1..]);
        let fold = |p: &E, q: &E, p2: &E, q2: &E| (p * q2 + p2 * q, q * q2);
        std::array::from_fn(|t| {
            let node = E::from(t as u64);
            (0..quarter).fold(E::zero(), |acc, x| {
                let v = |side: usize, j: usize| {
                    let (lo, hi) = (&self.f[side][j][x], &self.f[side][j][x + quarter]);
                    lo + &node * (hi - lo)
                };
                let (m0p, m0q) = fold(&v(0, 0), &v(1, 0), &v(0, 2), &v(1, 2));
                let (m1p, m1q) = fold(&v(0, 1), &v(1, 1), &v(0, 3), &v(1, 3));
                let (lp, lq) = fold(&m0p, &m0q, &m1p, &m1q);
                acc + &weight[x] * (lp + &self.lambda * lq)
            })
        })
    }

    /// The eight factors bound to the last challenge, in the proof's order.
    fn finish(&mut self, previous: Option<&E>) -> Vec<E> {
        let cells = 1usize << self.low;
        if let Some(s) = previous {
            self.fold(s, cells);
        }
        let mut out = Vec::with_capacity(8 * cells);
        for b2 in 0..2 {
            for (side, j) in [(0, b2), (0, b2 + 2), (1, b2), (1, b2 + 2)] {
                out.extend(self.f[side][j][..cells].iter().cloned());
            }
        }
        out
    }
}

/// A round's raw sums and the finish's raw values.
type Run = (Vec<Vec<u64>>, Vec<u64>);

fn run_layer(
    layer: &mut Gruen2Layer,
    lambda: &[u64; 3],
    half_lambda: &[u64; 3],
    challenges: &[[u64; 3]],
    h0: impl Fn(usize) -> bool,
) -> Run {
    let mut sums = Vec::new();
    for j in 0..layer.rounds() {
        let fold = (j > 0).then(|| &challenges[j - 1]);
        sums.push(
            layer
                .round(fold, lambda, half_lambda, h0(j))
                .expect("a round"),
        );
    }
    let last = (layer.rounds() > 0).then(|| &challenges[layer.rounds() - 1]);
    (sums, layer.finish(last).expect("the finish"))
}

fn raw(v: &E) -> [u64; 3] {
    let [a, b, c] = v.value();
    [a.canonical(), b.canonical(), c.canonical()]
}

/// ★ Every round's `H(1)..H(4)` (and `H(0)` where asked) and the finish's
/// eight factors equal the host's composition of the two folds — at quarters
/// of 2^2 to 2^12 cells, a step over the input layer and one over a level
/// above it, every host tail, with and without the slot.
#[test]
fn two_layer_rounds_equal_the_composed_reference() {
    let _serial = serial();
    let Ok(be) = backend() else {
        eprintln!("no device; skipping");
        return;
    };
    let mut layers = 0;
    for (n, &(m, low, h0, slot)) in [
        (2usize, 0usize, 1u8, RoundSlot::Off),
        (4, 2, 2, RoundSlot::Off),
        (6, 6, 0, RoundSlot::On),
        (10, 6, 2, RoundSlot::On),
        (10, 0, 0, RoundSlot::Check),
        (12, 6, 0, RoundSlot::On),
        (12, 6, 2, RoundSlot::Off),
    ]
    .iter()
    .enumerate()
    {
        for above in [false, true] {
            let what = format!("m {m} low {low} h0 {h0} slot {slot:?} above {above}");
            let num_vars = m + 2 + if above { 2 } else { 0 };
            let (p, q) = layer(num_vars, 1 << num_vars, 0x5EED + n as u64);
            let stream = be.next_stream();
            let tree = DeviceFractionTree::from_device(
                stream.clone(),
                upload(&stream, &p),
                upload(&stream, &q),
            )
            .expect("a tree");
            let input = tree.num_layers() - 1;
            let at = if above { input - 2 } else { input };
            let (lp, lq) = if above {
                tree.layer_to_host(at).expect("the level")
            } else {
                (p.clone(), q.clone())
            };
            let mut rng = Rng(0xA11CE ^ n as u64);
            let point: Vec<[u64; 3]> = (0..m).map(|_| rng.ext()).collect();
            let lambda = rng.ext();
            let challenges: Vec<[u64; 3]> = (0..m).map(|_| rng.ext()).collect();
            let lambda_e = value(&lambda);
            let half_lambda = raw(&(&lambda_e * E::from(2u64).inv().unwrap()));
            let point_e: Vec<E> = point.iter().map(|u| value(u)).collect();
            let flat: Vec<u64> = point.iter().flatten().copied().collect();
            let mut card = tree.layer_gruen2(at, &flat, low).expect("a two-layer step");
            card.use_slot(slot);
            let h0_of = |j: usize| h0 == 1 || (h0 == 2 && j % 2 == 1);
            let checked = math_cuda::round_slot::counts();
            let (sums, finish) = run_layer(&mut card, &lambda, &half_lambda, &challenges, h0_of);
            let after = math_cuda::round_slot::counts();
            assert_eq!(
                after.mismatches, checked.mismatches,
                "{what}: the slot's sums are the second launch's"
            );
            let mut reference = Reference::new(&lp, &lq, &point_e, lambda_e, low);
            for (j, got) in sums.iter().enumerate() {
                let previous = (j > 0).then(|| value(&challenges[j - 1]));
                let want = reference.round(j, previous.as_ref());
                let rows = if h0_of(j) { 5 } else { 4 };
                assert_eq!(got.len(), rows * 3, "{what}: round {j}'s rows");
                for t in 1..5 {
                    assert_eq!(
                        value(&got[(t - 1) * 3..t * 3]),
                        want[t],
                        "{what}: round {j}'s H({t})"
                    );
                }
                if h0_of(j) {
                    assert_eq!(value(&got[12..15]), want[0], "{what}: round {j}'s H(0)");
                }
            }
            let last = (card.rounds() > 0).then(|| value(&challenges[card.rounds() - 1]));
            let want = reference.finish(last.as_ref());
            let got: Vec<E> = finish.chunks_exact(3).map(value).collect();
            assert_eq!(got, want, "{what}: the eight factors at the end");
            layers += 1;
        }
    }
    eprintln!(
        "gkr2 gruen: {layers} two-layer steps equal the composed reference, every round and the finish"
    );
}

/// ★ An input layer kept without its padding runs the whole layer's
/// two-layer rounds, limb for limb: the padding starting in quarter 2 or in
/// quarter 3, one row in or at a precompile's shapes, every host tail, with
/// `H(0)` summed on the card or not, with the slot or not.
#[test]
fn a_kept_input_layer_runs_the_whole_layers_rounds_limb_for_limb() {
    let _serial = serial();
    let Ok(be) = backend() else {
        eprintln!("no device; skipping");
        return;
    };
    let shapes: [(usize, usize); 8] = [
        (5, 8),
        (3, 4),
        (7, 32),
        (17, 1 << 6),
        (20, 1 << 10),
        (1031, 1 << 3),
        (9, 1 << 11),
        (13, 1 << 9),
    ];
    let mut checked = 0;
    for (n, &(interactions, rows)) in shapes.iter().enumerate() {
        for (low, h0, slot) in [
            (6usize, 0u8, RoundSlot::On),
            (0, 1, RoundSlot::Off),
            (9, 2, RoundSlot::Off),
            (2, 0, RoundSlot::Check),
        ] {
            let what = format!("{interactions} × {rows} rows, low {low}, h0 {h0}, slot {slot:?}");
            let full = interactions.next_power_of_two() * rows;
            let real = interactions * rows;
            let num_vars = full.trailing_zeros() as usize;
            let (p, q) = layer(num_vars, real, 0xB0B + n as u64);
            let m = num_vars - 2;
            let mut rng = Rng(0xFACE ^ (n * 8 + low) as u64);
            let point: Vec<u64> = (0..m).flat_map(|_| rng.ext()).collect();
            let lambda = rng.ext();
            let half_lambda = raw(&(value(&lambda) * E::from(2u64).inv().unwrap()));
            let challenges: Vec<[u64; 3]> = (0..m).map(|_| rng.ext()).collect();
            let h0_of = |j: usize| h0 == 1 || (h0 == 2 && j % 2 == 1);
            let whole = {
                let stream = be.next_stream();
                let tree = DeviceFractionTree::from_device(
                    stream.clone(),
                    upload(&stream, &p),
                    upload(&stream, &q),
                )
                .expect("today's tree");
                let mut card = tree
                    .layer_gruen2(tree.num_layers() - 1, &point, low)
                    .expect("the whole layer's step");
                assert!(!card.truncated(), "{what}: the whole layer");
                card.use_slot(slot);
                run_layer(&mut card, &lambda, &half_lambda, &challenges, h0_of)
            };
            let kept = {
                let stream = be.next_stream();
                let tree = DeviceFractionTree::from_truncated_input(
                    stream.clone(),
                    upload(&stream, &p[..real * 3]),
                    upload(&stream, &q[..real * 3]),
                    real,
                    num_vars,
                )
                .expect("the kept tree");
                let mut card = tree
                    .layer_gruen2(tree.num_layers() - 1, &point, low)
                    .expect("the kept layer's step");
                assert_eq!(card.truncated(), real < full, "{what}: opened kept");
                card.use_slot(slot);
                run_layer(&mut card, &lambda, &half_lambda, &challenges, h0_of)
            };
            for (j, (x, y)) in whole.0.iter().zip(&kept.0).enumerate() {
                assert_eq!(x, y, "{what}: round {j}'s sums, raw");
            }
            assert_eq!(whole.1, kept.1, "{what}: the eight factors, raw");
            checked += 1;
        }
    }
    eprintln!("gkr2 gruen: {checked} kept input layers raw-identical to the whole layers");
}

/// ⛔ A layer the two-layer step cannot take is refused before anything
/// runs: one without two variables to descend, or a point of the wrong
/// length.
#[test]
fn a_two_layer_step_of_the_wrong_shape_is_refused() {
    let _serial = serial();
    let Ok(be) = backend() else {
        eprintln!("no device; skipping");
        return;
    };
    let invalid = cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE;
    let (p, q) = layer(6, 64, 9);
    let stream = be.next_stream();
    let tree =
        DeviceFractionTree::from_device(stream.clone(), upload(&stream, &p), upload(&stream, &q))
            .expect("a tree");
    let input = tree.num_layers() - 1;
    assert!(
        tree.layer_gruen2(input, &[7u64; 4 * 3], 2).is_ok(),
        "m = 4 opens"
    );
    for wrong in [3usize, 5] {
        assert_eq!(
            tree.layer_gruen2(input, &vec![7u64; wrong * 3], 2)
                .err()
                .map(|e| e.0),
            Some(invalid),
            "a point of {wrong} coordinates"
        );
    }
    assert_eq!(
        tree.layer_gruen2(1, &[], 0).err().map(|e| e.0),
        Some(invalid),
        "a layer of one variable"
    );
}
