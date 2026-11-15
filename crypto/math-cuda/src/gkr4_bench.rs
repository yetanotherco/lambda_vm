//! ⛔ A measurement module, on NO proving path: one GKR step over two tree
//! layers at once (`kernels/gkr4_bench.cu`), against today's pair of
//! single-layer steps ([`crate::gkr::GruenLayer`]) over the same cells.
//!
//! Today a step proves a layer against the layer below it, a sumcheck of
//! degree 3. This step proves a layer against its grandchildren: the four
//! quarters of the layer two below, combined as the two folds would combine
//! them, under one sumcheck of degree 5. Half the steps, and the layer in
//! between is never read. The question it answers is the card's: does the
//! round kernel over eight factors at degree 5 keep today's bandwidth?
//!
//! The module is loaded on first use from its own cubin, never by
//! [`crate::device::Backend::init`], so a proving process never loads it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use crate::Result;
use crate::device::backend;

const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gkr4_bench.cubin"));

/// The most blocks a round launches, as today's Gruen rounds.
const MAX_GRID: u32 = 2048;

/// Rows a round can write: `H(1), H(2), H(3), H(4)`, then `H(0)`.
pub const ROWS: usize = 5;

struct Module {
    _module: Arc<CudaModule>,
    round: CudaFunction,
}

fn module() -> Result<&'static Module> {
    static MODULE: OnceLock<Module> = OnceLock::new();
    if let Some(m) = MODULE.get() {
        return Ok(m);
    }
    let be = backend()?;
    let module = be.ctx.load_module(Ptx::from_binary(CUBIN.to_vec()))?;
    let m = Module {
        round: module.load_function("gkr4_round_gruen")?,
        _module: module,
    };
    Ok(MODULE.get_or_init(|| m))
}

static FAULT: AtomicBool = AtomicBool::new(false);

/// ⛔ A FAULT, a negative control's only: in a round without a fold, the last
/// quarter's lower cell is read one index off.
#[doc(hidden)]
pub fn force_fault(on: bool) {
    FAULT.store(on, Ordering::Relaxed);
}

/// Today's launch shape for a Gruen round: 256-thread blocks for a wide cube,
/// down to a warp for a narrow one, at most [`MAX_GRID`] blocks.
fn shape(quarter: u64) -> (u32, u32) {
    let block = quarter.next_power_of_two().clamp(32, 256) as u32;
    let grid = quarter.div_ceil(block as u64).clamp(1, MAX_GRID as u64) as u32;
    (grid, block)
}

/// One double step's card rounds over a layer of `2^num_vars` cells: its four
/// quarters (`2^m` cells each, `m = num_vars − 2`) in `p` and in `q`, folded
/// in place round by round.
pub struct DoubleStep {
    stream: Arc<CudaStream>,
    _p: Arc<CudaSlice<u64>>,
    _q: Arc<CudaSlice<u64>>,
    p_at: u64,
    q_at: u64,
    /// Cells in a quarter, `2^m`.
    stride: usize,
    /// The variables left to the host, `L`.
    low: usize,
    /// `m − L`: the rounds run here.
    rounds: usize,
    /// Every `eq` level, as today's: `e_lo`, then one `e_hi` level a round.
    eq: CudaSlice<u64>,
    partials: CudaSlice<u64>,
    sums: CudaSlice<u64>,
    done: usize,
}

impl DoubleStep {
    /// Over `p` and `q` of `2^num_vars` cells, the parent's point `m` ext3
    /// values, at most `low` variables left to the host.
    pub fn new(
        stream: &Arc<CudaStream>,
        p: &Arc<CudaSlice<u64>>,
        q: &Arc<CudaSlice<u64>>,
        num_vars: usize,
        point: &[u64],
        low: usize,
    ) -> Result<Self> {
        assert!(num_vars >= 3, "a double step needs a round");
        let m = num_vars - 2;
        assert_eq!(point.len(), m * 3, "the point spans a quarter");
        assert_eq!(p.len(), (1usize << num_vars) * 3, "p is the whole layer");
        assert_eq!(q.len(), (1usize << num_vars) * 3, "q is the whole layer");
        let low = low.min(m - 1);
        let rounds = m - low;
        let be = backend()?;
        let cells = (1usize << low) + (1usize << rounds) - 1;
        // SAFETY: the kernel writes every cell.
        let mut eq = unsafe { crate::device::alloc_or_trim::<u64>(stream, cells * 3) }?;
        let point_dev = crate::device::htod_or_trim(stream, point)?;
        let low_arg = low as u32;
        let rounds_arg = rounds as u32;
        unsafe {
            stream
                .launch_builder(&be.gkr_eq_levels_ext3)
                .arg(&point_dev)
                .arg(&low_arg)
                .arg(&rounds_arg)
                .arg(&mut eq)
                .launch(LaunchConfig {
                    grid_dim: ((cells as u64).div_ceil(256).min(4096) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        // SAFETY: a round writes the rows it reduces before they are read.
        let partials =
            unsafe { crate::device::alloc_or_trim::<u64>(stream, ROWS * MAX_GRID as usize * 3) }?;
        let sums = crate::device::alloc_zeros_or_trim::<u64>(stream, ROWS * 3)?;
        let p_at = {
            let (at, _guard) = p.device_ptr(stream);
            at
        };
        let q_at = {
            let (at, _guard) = q.device_ptr(stream);
            at
        };
        Ok(Self {
            stream: stream.clone(),
            _p: p.clone(),
            _q: q.clone(),
            p_at,
            q_at,
            stride: 1usize << m,
            low,
            rounds,
            eq,
            partials,
            sums,
            done: 0,
        })
    }

    /// The rounds this step runs on the card.
    pub fn rounds(&self) -> usize {
        self.rounds
    }

    /// The `eq` levels, read back: the weights a reference recomputes from.
    pub fn eq_levels(&self) -> Result<Vec<u64>> {
        let eq = crate::readback::clone_dtoh(&self.stream, &self.eq)?;
        self.stream.synchronize()?;
        Ok(eq)
    }

    /// The next round's launches, queued: the round and the reduction of its
    /// rows. `fold_by` is the previous round's challenge. Returns the rows
    /// [`round_collect`](Self::round_collect) reads.
    pub fn round_enqueue(
        &mut self,
        fold_by: Option<&[u64; 3]>,
        lambda: &[u64; 3],
        want_h0: bool,
    ) -> Result<u32> {
        assert!(self.done < self.rounds, "every card round has run");
        assert_eq!(
            fold_by.is_some(),
            self.done > 0,
            "a challenge after round 0"
        );
        let be = backend()?;
        let m = self.stride.trailing_zeros() as usize;
        let quarter = 1u64 << (m - self.done - 1);
        let (grid, block) = shape(quarter);
        let rows = if want_h0 { 5u32 } else { 4 };
        let s = fold_by.copied().unwrap_or([0; 3]);
        let fold = u32::from(fold_by.is_some());
        let at =
            (1usize << self.low) + (1usize << self.rounds) - (1usize << (self.rounds - self.done));
        let e_lo = self.eq.slice(0..(1usize << self.low) * 3);
        let e_hi = self
            .eq
            .slice(at * 3..(at + (1usize << (self.rounds - 1 - self.done))) * 3);
        let stride = self.stride as u64;
        let low = self.low as u32;
        let want = u32::from(want_h0);
        let fault = u32::from(FAULT.load(Ordering::Relaxed));
        unsafe {
            self.stream
                .launch_builder(&module()?.round)
                .arg(&self.p_at)
                .arg(&self.q_at)
                .arg(&stride)
                .arg(&quarter)
                .arg(&fold)
                .arg(&s[0])
                .arg(&s[1])
                .arg(&s[2])
                .arg(&lambda[0])
                .arg(&lambda[1])
                .arg(&lambda[2])
                .arg(&e_hi)
                .arg(&e_lo)
                .arg(&low)
                .arg(&want)
                .arg(&fault)
                .arg(&mut self.partials)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: block * 3 * 8,
                })?;
            self.stream
                .launch_builder(&be.sum_partials_ext3)
                .arg(&self.partials)
                .arg(&(grid as u64))
                .arg(&mut self.sums)
                .launch(LaunchConfig {
                    grid_dim: (rows, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 256 * 3 * 8,
                })?;
        }
        self.done += 1;
        Ok(rows)
    }

    /// The rows a [`round_enqueue`](Self::round_enqueue) queued, three u64
    /// each.
    pub fn round_collect(&self, rows: u32) -> Result<Vec<u64>> {
        let sums =
            crate::readback::clone_dtoh(&self.stream, &self.sums.slice(0..rows as usize * 3))?;
        self.stream.synchronize()?;
        Ok(sums)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gkr::{DeviceFractionTree, InputLayer};
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
    use math::field::goldilocks::GoldilocksField;

    type Fp = FieldElement<GoldilocksField>;
    type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

    const MODULUS: u64 = 0xFFFF_FFFF_0000_0001;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn limbs(&mut self, n: usize) -> Vec<u64> {
            (0..n * 3).map(|_| self.next() % MODULUS).collect()
        }
        fn ext(&mut self) -> [u64; 3] {
            [
                self.next() % MODULUS,
                self.next() % MODULUS,
                self.next() % MODULUS,
            ]
        }
    }

    fn fe(l: &[u64]) -> Fp3 {
        Fp3::new([Fp::from_raw(l[0]), Fp::from_raw(l[1]), Fp::from_raw(l[2])])
    }

    fn fes(l: &[u64]) -> Vec<Fp3> {
        l.chunks(3).map(fe).collect()
    }

    /// The parent from the grandchildren as the two folds make it: `(0, 2)`
    /// into the middle layer's lower half, `(1, 3)` into its upper half.
    fn composed(v: &[Fp3; 8], lambda: &Fp3) -> Fp3 {
        let fold = |p1: &Fp3, q1: &Fp3, p2: &Fp3, q2: &Fp3| (p1 * q2 + p2 * q1, q1 * q2);
        let (lo_p, lo_q) = fold(&v[0], &v[4], &v[2], &v[6]);
        let (hi_p, hi_q) = fold(&v[1], &v[5], &v[3], &v[7]);
        let (p, q) = fold(&lo_p, &lo_q, &hi_p, &hi_q);
        p + lambda * q
    }

    struct Back;
    impl Drop for Back {
        fn drop(&mut self) {
            force_fault(false);
        }
    }

    /// ★ The double step's rounds are the pair's composed values, and the
    /// comparison is load-bearing. Over random layers of 2^10 and 2^12 cells:
    /// (1) round 0's rows at `t = 0` and `t = 1` equal today's step
    /// (`gkr_round_gruen`) over the middle layer the production fold makes,
    /// at the same point and `λ` — at Boolean `t` the two relations agree
    /// cell by cell; (2) every card round's rows `H(1)..H(4), H(0)` equal a
    /// host reference that composes the two folds over the quarters, folded
    /// by the same challenges; (3) the last quarter's lower cell one off parts
    /// from the reference and from today's `H(0)`, and leaves `H(1)`, which
    /// reads no lower cell.
    #[test]
    #[ignore = "a card; flips a process-global fault: run alone, --ignored --test-threads=1"]
    fn the_double_step_is_the_pair_composed() {
        let _back = Back;
        let be = backend().expect("a card");
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let low = 2usize;
        let mut checked = 0;
        for num_vars in [10usize, 12] {
            let cells = 1usize << num_vars;
            let m = num_vars - 2;
            let (p, q) = (rng.limbs(cells), rng.limbs(cells));
            let point = rng.limbs(m);
            let lambda = rng.ext();
            let stream = be.next_stream();

            // Today's step over the middle layer, as the production fold makes it.
            let tree = DeviceFractionTree::build(&p, &q).expect("the tree");
            assert_eq!(tree.layer_num_vars(num_vars - 1), num_vars - 1);
            let (mid_p, mid_q) = tree.layer_to_host(num_vars - 1).expect("the middle layer");
            let middle = InputLayer::new(
                stream.clone(),
                crate::device::htod_or_trim(&stream, &mid_p).unwrap(),
                crate::device::htod_or_trim(&stream, &mid_q).unwrap(),
                num_vars - 1,
            );
            let mut today = middle.gruen(&point, low).expect("today's step");
            let today_rows = fes(&today.round(None, &lambda, true).expect("today's round 0"));

            let open = |stream: &Arc<CudaStream>| {
                let pd = Arc::new(crate::device::htod_or_trim(stream, &p).unwrap());
                let qd = Arc::new(crate::device::htod_or_trim(stream, &q).unwrap());
                DoubleStep::new(stream, &pd, &qd, num_vars, &point, low).expect("the double step")
            };
            let mut step = open(&stream);
            let eq = fes(&step.eq_levels().expect("eq"));
            let rounds = step.rounds();

            // The host's quarters: `f[k]` for k = 0..4 numerators, 4..8 denominators.
            let (hp, hq) = (fes(&p), fes(&q));
            let stride = 1usize << m;
            let mut f: Vec<Vec<Fp3>> = (0..8)
                .map(|k| {
                    let src = if k < 4 { &hp } else { &hq };
                    src[(k & 3) * stride..((k & 3) + 1) * stride].to_vec()
                })
                .collect();
            let lam = fe(&lambda);
            let challenges: Vec<[u64; 3]> = (0..rounds).map(|_| rng.ext()).collect();
            let mut first: Vec<Fp3> = Vec::new();
            for j in 0..rounds {
                let fold_by = (j > 0).then(|| challenges[j - 1]);
                if let Some(s) = fold_by {
                    let s = fe(&s);
                    for fk in f.iter_mut() {
                        let half = fk.len() / 2;
                        let folded: Vec<Fp3> = (0..half)
                            .map(|x| &fk[x] + &s * &(&fk[x + half] - &fk[x]))
                            .collect();
                        *fk = folded;
                    }
                }
                let quarter = 1usize << (m - j - 1);
                let at = (1usize << low) + (1usize << rounds) - (1usize << (rounds - j));
                let mask = (1usize << low) - 1;
                let mut want = vec![Fp3::zero(); ROWS];
                for x in 0..quarter {
                    let w = &eq[at + (x >> low)] * &eq[x & mask];
                    let lo: [Fp3; 8] = std::array::from_fn(|k| f[k][x]);
                    let hi: [Fp3; 8] = std::array::from_fn(|k| f[k][x + quarter]);
                    for (row, t) in [1u64, 2, 3, 4, 0].into_iter().enumerate() {
                        let t = Fp3::from(t);
                        let v: [Fp3; 8] = std::array::from_fn(|k| &lo[k] + &t * &(&hi[k] - &lo[k]));
                        want[row] = &want[row] + &w * composed(&v, &lam);
                    }
                }
                let rows = step
                    .round_enqueue(fold_by.as_ref(), &lambda, true)
                    .expect("a round");
                let got = fes(&step.round_collect(rows).expect("its rows"));
                assert_eq!(got, want, "2^{num_vars}, round {j}: the rows");
                if j == 0 {
                    first = got;
                }
                checked += 1;
            }
            // Round 0 at t = 1 and t = 0 against today's step: rows H(1) and H(0).
            assert_eq!(first[0], today_rows[0], "2^{num_vars}: H(1) is today's");
            assert_eq!(first[4], today_rows[2], "2^{num_vars}: H(0) is today's");

            // The off-by-one parts from both.
            force_fault(true);
            let mut faulted = open(&stream);
            let rows = faulted
                .round_enqueue(None, &lambda, true)
                .expect("a faulted round");
            let bad = fes(&faulted.round_collect(rows).expect("its rows"));
            force_fault(false);
            assert_ne!(
                bad, first,
                "2^{num_vars}: the off-by-one must part from the reference"
            );
            // The fault reads only a lower-half cell, so H(1) (built from the
            // upper half alone) keeps today's value and H(0) must leave it.
            assert_eq!(
                bad[0], today_rows[0],
                "2^{num_vars}: the off-by-one leaves H(1), which reads no lower cell"
            );
            assert_ne!(
                bad[4], today_rows[2],
                "2^{num_vars}: the off-by-one must part from today's H(0)"
            );
        }
        eprintln!(
            "GKR4 GATE equality PASS: {checked} card rounds at 2^10 and 2^12 equal the composed \
             pair; round 0's H(1), H(0) equal today's step over the middle layer; the off-by-one \
             parted from the reference and from today's H(0)"
        );
    }

    /// The host reference itself, no card: at Boolean `t` the composed
    /// grandchildren are today's relation over the middle layer the two halves
    /// fold into, cell by cell, and off the cube the composition is a
    /// polynomial of degree 4 in `t` (its value at 5 is the one its values at
    /// 0..4 interpolate), which is why a round sends five rows.
    #[test]
    fn the_composed_reference_is_the_pair_at_boolean_points() {
        let mut rng = Rng(0xA076_1D64_78BD_642F);
        let lambda = fe(&rng.ext());
        let fold = |p1: &Fp3, q1: &Fp3, p2: &Fp3, q2: &Fp3| (p1 * q2 + p2 * q1, q1 * q2);
        for _ in 0..16 {
            let lo: [Fp3; 8] = std::array::from_fn(|_| fe(&rng.ext()));
            let hi: [Fp3; 8] = std::array::from_fn(|_| fe(&rng.ext()));
            for (t, v) in [(0u64, &lo), (1, &hi)] {
                // The middle layer's lower and upper halves at this node.
                let (mlo_p, mlo_q) = fold(&v[0], &v[4], &v[2], &v[6]);
                let (mhi_p, mhi_q) = fold(&v[1], &v[5], &v[3], &v[7]);
                let today = mlo_p * mhi_q + mhi_p * mlo_q + lambda * mlo_q * mhi_q;
                assert_eq!(composed(v, &lambda), today, "t = {t}");
            }
            let at = |t: u64| {
                let t = Fp3::from(t);
                let v: [Fp3; 8] = std::array::from_fn(|k| lo[k] + t * (hi[k] - lo[k]));
                composed(&v, &lambda)
            };
            // Lagrange through t = 0..4, evaluated at 5: weights 1, −5, 10, −10, 5.
            let w = [1i64, -5, 10, -10, 5];
            let through: Fp3 = (0..5u64).fold(Fp3::zero(), |acc, t| {
                let c = Fp3::from(w[t as usize].unsigned_abs());
                if w[t as usize] < 0 {
                    acc - c * at(t)
                } else {
                    acc + c * at(t)
                }
            });
            assert_eq!(through, at(5), "degree 4 in t");
        }
    }

    /// Bytes a round moves by the kernels' own reads and writes: `factors`
    /// factors, each read at two cells a cube index (four with a fold, and
    /// two written back).
    fn round_bytes(factors: u64, quarter: u64, fold: bool) -> u64 {
        let cells = if fold { 4 + 2 } else { 2 };
        factors * cells * quarter * 24
    }

    struct Arm {
        total_ms: f64,
        big_ms: f64,
        big_bytes: u64,
    }

    /// Queues `rounds` rounds through `round`, an event after each, and
    /// returns per-round milliseconds with each round's quarter.
    fn timed(
        stream: &Arc<CudaStream>,
        rounds: usize,
        m: usize,
        mut round: impl FnMut(usize) -> Result<u32>,
    ) -> Vec<(u64, f64)> {
        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let ctx = stream.context();
        stream.synchronize().unwrap();
        let mut events = vec![ctx.new_event(flags).unwrap()];
        events[0].record(stream).unwrap();
        for j in 0..rounds {
            round(j).expect("a round");
            let e = ctx.new_event(flags).unwrap();
            e.record(stream).unwrap();
            events.push(e);
        }
        stream.synchronize().unwrap();
        (0..rounds)
            .map(|j| {
                let quarter = 1u64 << (m - j - 1);
                (
                    quarter,
                    f64::from(events[j].elapsed_ms(&events[j + 1]).unwrap()),
                )
            })
            .collect()
    }

    fn arm(per_round: &[(u64, f64)], factors: u64) -> Arm {
        let big = 4096u64;
        Arm {
            total_ms: per_round.iter().map(|r| r.1).sum(),
            big_ms: per_round.iter().filter(|r| r.0 >= big).map(|r| r.1).sum(),
            big_bytes: per_round
                .iter()
                .enumerate()
                .filter(|(_, r)| r.0 >= big)
                .map(|(j, r)| round_bytes(factors, r.0, j > 0))
                .sum(),
        }
    }

    fn median(mut xs: Vec<f64>) -> f64 {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = xs.len();
        if n % 2 == 1 {
            xs[n / 2]
        } else {
            (xs[n / 2 - 1] + xs[n / 2]) / 2.0
        }
    }

    /// The bench: at depths 24 and 22, the double step's card rounds over a
    /// random layer against today's two steps over the same cells — the layer
    /// itself, then the middle layer the production fold makes of it. Both
    /// arms run every card round down to today's host tail (`2^6` cells), the
    /// round kernel and its reduction each, with no read-back between rounds;
    /// CUDA events time each round. Five repetitions, arm order alternating;
    /// medians. `LAMBDA_VM_GKR4_BENCH_DEPTHS` (comma-separated) overrides the
    /// depths.
    #[test]
    #[ignore = "a card, gigabytes of layers: run alone, --ignored --test-threads=1"]
    fn the_double_step_against_todays_pair_on_the_card() {
        let be = backend().expect("a card");
        let depths: Vec<usize> = std::env::var("LAMBDA_VM_GKR4_BENCH_DEPTHS")
            .ok()
            .map(|v| {
                v.split(',')
                    .map(|d| d.trim().parse().expect("a depth"))
                    .collect()
            })
            .unwrap_or_else(|| vec![24, 22]);
        let low = 6usize;
        let reps = 5usize;
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for depth in depths {
            let cells = 1usize << depth;
            let stream = be.next_stream();
            let (p, q) = (rng.limbs(cells), rng.limbs(cells));
            let g_p = crate::device::htod_or_trim(&stream, &p).unwrap();
            let g_q = crate::device::htod_or_trim(&stream, &q).unwrap();
            drop((p, q));
            let tree = DeviceFractionTree::from_device(
                stream.clone(),
                stream.clone_dtod(&g_p).unwrap(),
                stream.clone_dtod(&g_q).unwrap(),
            )
            .expect("the tree");
            assert_eq!(tree.layer_num_vars(depth - 1), depth - 1);
            let (mid_p, mid_q) = tree.layer_to_host(depth - 1).expect("the middle layer");
            drop(tree);
            let m_p = crate::device::htod_or_trim(&stream, &mid_p).unwrap();
            let m_q = crate::device::htod_or_trim(&stream, &mid_q).unwrap();
            drop((mid_p, mid_q));
            let point_a = rng.limbs(depth - 1);
            let point_b = rng.limbs(depth - 2);
            let lambda = rng.ext();
            let challenges: Vec<[u64; 3]> = (0..depth).map(|_| rng.ext()).collect();
            let fold_by = |j: usize| (j > 0).then(|| challenges[j - 1]);

            let (mut pair, mut step_a, mut step_b, mut double) = (vec![], vec![], vec![], vec![]);
            let (mut pair_big, mut double_big) = (vec![], vec![]);
            let (mut pair_gbs, mut double_gbs) = (vec![], vec![]);
            for rep in 0..reps {
                let run_pair = || {
                    let a = InputLayer::new(
                        stream.clone(),
                        stream.clone_dtod(&g_p).unwrap(),
                        stream.clone_dtod(&g_q).unwrap(),
                        depth,
                    );
                    let mut a = a.gruen(&point_a, low).expect("step a");
                    let b = InputLayer::new(
                        stream.clone(),
                        stream.clone_dtod(&m_p).unwrap(),
                        stream.clone_dtod(&m_q).unwrap(),
                        depth - 1,
                    );
                    let mut b = b.gruen(&point_b, low).expect("step b");
                    let ra = a.rounds();
                    let ta = timed(&stream, ra, depth - 1, |j| {
                        a.round_enqueue(fold_by(j).as_ref(), &lambda, false)
                    });
                    let rb = b.rounds();
                    let tb = timed(&stream, rb, depth - 2, |j| {
                        b.round_enqueue(fold_by(j).as_ref(), &lambda, false)
                    });
                    (arm(&ta, 4), arm(&tb, 4))
                };
                let run_double = || {
                    let pd = Arc::new(stream.clone_dtod(&g_p).unwrap());
                    let qd = Arc::new(stream.clone_dtod(&g_q).unwrap());
                    let mut d = DoubleStep::new(&stream, &pd, &qd, depth, &point_b, low)
                        .expect("the double step");
                    let rd = d.rounds();
                    let td = timed(&stream, rd, depth - 2, |j| {
                        d.round_enqueue(fold_by(j).as_ref(), &lambda, false)
                    });
                    arm(&td, 8)
                };
                let ((a, b), d) = if rep % 2 == 0 {
                    let ab = run_pair();
                    (ab, run_double())
                } else {
                    let d = run_double();
                    (run_pair(), d)
                };
                let pair_ms = a.total_ms + b.total_ms;
                eprintln!(
                    "GKR4 REP depth {depth} rep {rep}: pair {pair_ms:.3} ms (step {depth} {:.3}, \
                     step {} {:.3}) · double {:.3} ms",
                    a.total_ms,
                    depth - 1,
                    b.total_ms,
                    d.total_ms
                );
                pair.push(pair_ms);
                step_a.push(a.total_ms);
                step_b.push(b.total_ms);
                double.push(d.total_ms);
                pair_big.push(a.big_ms + b.big_ms);
                double_big.push(d.big_ms);
                pair_gbs.push((a.big_bytes + b.big_bytes) as f64 / (a.big_ms + b.big_ms) / 1e6);
                double_gbs.push(d.big_bytes as f64 / d.big_ms / 1e6);
            }
            let (pm, dm) = (median(pair), median(double));
            let (pbm, dbm) = (median(pair_big), median(double_big));
            eprintln!(
                "GKR4 BENCH depth {depth}: pair {pm:.3} ms (step {depth} {:.3}, step {} {:.3}) · \
                 double {dm:.3} ms · ratio {:.3} · big rounds (quarter >= 4096): pair {pbm:.3} ms \
                 at {:.0} GB/s, double {dbm:.3} ms at {:.0} GB/s, ratio {:.3} · reps {reps}",
                median(step_a),
                depth - 1,
                median(step_b),
                dm / pm,
                median(pair_gbs),
                median(double_gbs),
                dbm / pbm
            );
        }
    }
}
