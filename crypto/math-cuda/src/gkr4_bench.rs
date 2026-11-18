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

use std::sync::atomic::{AtomicU32, Ordering};
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

/// The round kernels the bench compares, all over the same layout and rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// The first bench's value-form kernel, one thread a cube index, as it
    /// was compiled (`__launch_bounds__(256)`): the reference.
    V0,
    /// The same body held to two 256-thread blocks an SM.
    V1,
    /// One thread a cube index, the coefficient form, two blocks an SM.
    V2,
    /// A thread pair a cube index (each lane owns one half of the middle
    /// layer and trades a quadratic with its partner), two blocks an SM.
    V3,
    /// V3 at three blocks an SM. Information only.
    V3b,
    /// The pair kernel with its row sums in shared memory, two blocks an SM;
    /// `H(0)` in its own pass.
    V4,
    /// The pair kernel at 128-lane blocks, three an SM (up to 170
    /// registers); `H(0)` in its own pass.
    V5,
}

impl Variant {
    pub const ALL: [Variant; 7] = [
        Variant::V0,
        Variant::V1,
        Variant::V2,
        Variant::V3,
        Variant::V3b,
        Variant::V4,
        Variant::V5,
    ];

    /// The name the bench prints.
    pub fn label(self) -> &'static str {
        match self {
            Variant::V0 => "v0",
            Variant::V1 => "v1",
            Variant::V2 => "v2",
            Variant::V3 => "v3",
            Variant::V3b => "v3b",
            Variant::V4 => "v4",
            Variant::V5 => "v5",
        }
    }

    /// Lanes a cube index.
    fn lanes(self) -> u64 {
        match self {
            Variant::V3 | Variant::V3b | Variant::V4 | Variant::V5 => 2,
            _ => 1,
        }
    }

    /// The most lanes a block.
    fn max_block(self) -> u64 {
        match self {
            Variant::V5 => 128,
            _ => 256,
        }
    }

    /// `H(0)` comes from its own pass after the round kernel.
    fn split_h0(self) -> bool {
        matches!(self, Variant::V4 | Variant::V5)
    }

    /// The value form takes `want_h0` as an argument; the others have an
    /// instance with `H(0)` and one without.
    fn value_form(self) -> bool {
        matches!(self, Variant::V0 | Variant::V1)
    }
}

struct Module {
    _module: Arc<CudaModule>,
    v0: CudaFunction,
    v1: CudaFunction,
    v2: [CudaFunction; 2],
    v3: [CudaFunction; 2],
    v3b: [CudaFunction; 2],
    v4: CudaFunction,
    v5: CudaFunction,
    h0_pass: CudaFunction,
}

impl Module {
    fn function(&self, variant: Variant, want_h0: bool) -> &CudaFunction {
        let h0 = usize::from(want_h0);
        match variant {
            Variant::V0 => &self.v0,
            Variant::V1 => &self.v1,
            Variant::V2 => &self.v2[h0],
            Variant::V3 => &self.v3[h0],
            Variant::V3b => &self.v3b[h0],
            Variant::V4 => &self.v4,
            Variant::V5 => &self.v5,
        }
    }
}

fn module() -> Result<&'static Module> {
    static MODULE: OnceLock<Module> = OnceLock::new();
    if let Some(m) = MODULE.get() {
        return Ok(m);
    }
    let be = backend()?;
    let module = be.ctx.load_module(Ptx::from_binary(CUBIN.to_vec()))?;
    let pair = |name: &str| -> Result<[CudaFunction; 2]> {
        Ok([
            module.load_function(name)?,
            module.load_function(&format!("{name}_h0"))?,
        ])
    };
    let m = Module {
        v0: module.load_function("gkr4_round_gruen")?,
        v1: module.load_function("gkr4_round_gruen_lb2")?,
        v2: pair("gkr2_round_v2")?,
        v3: pair("gkr2_round_v3")?,
        v3b: pair("gkr2_round_v3b")?,
        v4: module.load_function("gkr2_round_v4")?,
        v5: module.load_function("gkr2_round_v5")?,
        h0_pass: module.load_function("gkr2_round_h0_pass")?,
        _module: module,
    };
    Ok(MODULE.get_or_init(|| m))
}

/// ⛔ The negative controls' faults, for the card gate only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub enum Fault {
    None = 0,
    /// In a round without a fold, the last quarter's lower cell is read one
    /// index off (every variant).
    OneCell = 1,
    /// The pair kernels trade with lane `^ 2` (the next pair) instead of the
    /// partner. The other variants ignore it.
    Exchange = 2,
}

static FAULT: AtomicU32 = AtomicU32::new(0);

/// ⛔ A FAULT, a negative control's only. Process-global: the tests that set
/// it run alone.
#[doc(hidden)]
pub fn force_fault(fault: Fault) {
    FAULT.store(fault as u32, Ordering::Relaxed);
}

/// Today's launch shape for a Gruen round over `lanes` lanes: blocks of
/// `max_block` threads for a wide cube, down to a warp for a narrow one, at
/// most [`MAX_GRID`] blocks.
fn shape(lanes: u64, max_block: u64) -> (u32, u32) {
    let block = lanes.next_power_of_two().clamp(32, max_block) as u32;
    let grid = lanes.div_ceil(block as u64).clamp(1, MAX_GRID as u64) as u32;
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

    /// The next round's launches, queued: the round by `variant` and the
    /// reduction of its rows. `fold_by` is the previous round's challenge.
    /// Returns the rows [`round_collect`](Self::round_collect) reads.
    pub fn round_enqueue(
        &mut self,
        variant: Variant,
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
        let (grid, block) = shape(quarter * variant.lanes(), variant.max_block());
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
        let fault = FAULT.load(Ordering::Relaxed);
        let function = module()?.function(variant, want_h0);
        // The value form reduces through a 256-row tree in shared memory; the
        // others through warp shuffles, one ext3 a warp and row; V4 also
        // keeps every lane's four row sums there.
        let shared_mem_bytes = match variant {
            Variant::V0 | Variant::V1 => block * 3 * 8,
            Variant::V4 => (12 * block + (block / 32) * 4 * 3) * 8,
            _ => (block / 32) * rows * 3 * 8,
        };
        // The value form knows one fault, the cell; the others both.
        let value_fault = u32::from(fault == Fault::OneCell as u32);
        unsafe {
            let mut launch = self.stream.launch_builder(function);
            launch
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
                .arg(&low);
            if variant.value_form() {
                launch.arg(&want).arg(&value_fault);
            } else {
                launch.arg(&fault);
            }
            launch.arg(&mut self.partials).launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes,
            })?;
            if want_h0 && variant.split_h0() {
                // Row 4 of the same partials, over the same grid and blocks.
                self.stream
                    .launch_builder(&module()?.h0_pass)
                    .arg(&self.p_at)
                    .arg(&self.q_at)
                    .arg(&stride)
                    .arg(&quarter)
                    .arg(&fold)
                    .arg(&lambda[0])
                    .arg(&lambda[1])
                    .arg(&lambda[2])
                    .arg(&e_hi)
                    .arg(&e_lo)
                    .arg(&low)
                    .arg(&fault)
                    .arg(&mut self.partials)
                    .launch(LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: (block / 32) * 3 * 8,
                    })?;
            }
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
            force_fault(Fault::None);
        }
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

    /// A quadratic in `t` as the coefficient-form kernels hold it: its values
    /// at 0 and 1 and its `t²` coefficient.
    #[derive(Clone)]
    struct Quad {
        v0: Fp3,
        v1: Fp3,
        c2: Fp3,
    }

    impl Quad {
        /// The product of two linear factors given at `t = 0` and `t = 1`.
        fn prod(lo_a: &Fp3, hi_a: &Fp3, lo_b: &Fp3, hi_b: &Fp3) -> Quad {
            Quad {
                v0: lo_a * lo_b,
                v1: hi_a * hi_b,
                c2: (hi_a - lo_a) * (hi_b - lo_b),
            }
        }
        fn add(&self, o: &Quad) -> Quad {
            Quad {
                v0: &self.v0 + &o.v0,
                v1: &self.v1 + &o.v1,
                c2: &self.c2 + &o.c2,
            }
        }
        fn scale(&self, s: &Fp3) -> Quad {
            Quad {
                v0: s * &self.v0,
                v1: s * &self.v1,
                c2: s * &self.c2,
            }
        }
        /// Its value at `t`, walked from `t = 1` by finite differences as the
        /// kernels walk it (`t = 0` is `v0`).
        fn at(&self, t: u64) -> Fp3 {
            if t == 0 {
                return self.v0;
            }
            let dd = &self.c2 + &self.c2;
            let mut d = &self.v1 - &self.v0 + &dd;
            let mut val = self.v1;
            for _ in 1..t {
                val = &val + &d;
                d = &d + &dd;
            }
            val
        }
    }

    /// `X_r, Y_r`: the middle layer's half `r` from quarters `r` and `r + 2`.
    fn half(lo: &[Fp3; 8], hi: &[Fp3; 8], r: usize) -> (Quad, Quad) {
        let (a, b, c, d) = (r, r + 2, 4 + r, 4 + r + 2);
        let x = Quad::prod(&lo[a], &hi[a], &lo[d], &hi[d])
            .add(&Quad::prod(&lo[b], &hi[b], &lo[c], &hi[c]));
        let y = Quad::prod(&lo[c], &hi[c], &lo[d], &hi[d]);
        (x, y)
    }

    /// ★ The coefficient forms the new kernels compute are the composition,
    /// at `t = 0..5`, for random factors: the one-thread form
    /// `(X_0 + λY_0)(t)·Y_1(t) + X_1(t)·Y_0(t)` and the pair form, where lane
    /// `r` adds `(X_r + λ_r·Y_r)(t)·W(t)` with `λ_0 = λ`, `λ_1 = 0` and `W`
    /// the partner's `Y`. And an exchange with the wrong lane parts.
    #[test]
    fn the_coefficient_forms_are_the_composition() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        for _ in 0..16 {
            let lambda = fe(&rng.ext());
            let w = fe(&rng.ext());
            let lo: [Fp3; 8] = std::array::from_fn(|_| fe(&rng.ext()));
            let hi: [Fp3; 8] = std::array::from_fn(|_| fe(&rng.ext()));
            let (x0, y0) = half(&lo, &hi, 0);
            let (x1, y1) = half(&lo, &hi, 1);
            // One thread.
            let z = x0.add(&y0.scale(&lambda)).scale(&w);
            let v = x1.scale(&w);
            // A pair: lane 0 and lane 1, each with its partner's Y.
            let lanes = [
                (x0.add(&y0.scale(&lambda)).scale(&w), y1.clone()),
                (x1.add(&y1.scale(&Fp3::zero())).scale(&w), y0.clone()),
            ];
            for t in 0..=5u64 {
                let tt = Fp3::from(t);
                let at: [Fp3; 8] = std::array::from_fn(|k| &lo[k] + &tt * (&hi[k] - &lo[k]));
                let want = &w * composed(&at, &lambda);
                let one = z.at(t) * y1.at(t) + v.at(t) * y0.at(t);
                assert_eq!(one, want, "the one-thread form at t = {t}");
                let pair =
                    lanes[0].0.at(t) * lanes[0].1.at(t) + lanes[1].0.at(t) * lanes[1].1.at(t);
                assert_eq!(pair, want, "the pair form at t = {t}");
                if t >= 2 {
                    // Lane 1 holding its own Y instead of the partner's.
                    let wrong = lanes[0].0.at(t) * lanes[0].1.at(t) + lanes[1].0.at(t) * y1.at(t);
                    assert_ne!(wrong, want, "a wrong exchange parts at t = {t}");
                }
            }
        }
    }

    /// ★ The card gate of one variant: its rounds are the pair's composed
    /// values, and the comparison is load-bearing. Over random layers of 2^10
    /// and 2^12 cells, with and without `H(0)`: (1) every card round's rows
    /// `H(1)..H(4)` (and `H(0)`) equal a host reference that composes the two
    /// folds over the quarters, folded by the same challenges; (2) the quarters
    /// the last round leaves equal the reference's; (3) round 0's rows at
    /// `t = 1` and `t = 0` equal today's step (`gkr_round_gruen`) over the
    /// middle layer the production fold makes, at the same point and `λ`;
    /// (4) the last quarter's lower cell one off parts from the reference and
    /// from today's `H(0)`, and leaves `H(1)`, which reads no lower cell;
    /// (5) for the pair kernels, trading with the wrong lane parts.
    fn equal_gate(variant: Variant) {
        let _back = Back;
        let be = backend().expect("a card");
        let label = variant.label();
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let low = 2usize;
        let mut checked = 0;
        let mut faults = 0;
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

            let upload = |stream: &Arc<CudaStream>| {
                let pd = Arc::new(crate::device::htod_or_trim(stream, &p).unwrap());
                let qd = Arc::new(crate::device::htod_or_trim(stream, &q).unwrap());
                (pd, qd)
            };
            let challenges: Vec<[u64; 3]> = (0..m).map(|_| rng.ext()).collect();
            let lam = fe(&lambda);
            let mut first: Vec<Fp3> = Vec::new();
            for want_h0 in [true, false] {
                let (pd, qd) = upload(&stream);
                let mut step = DoubleStep::new(&stream, &pd, &qd, num_vars, &point, low)
                    .expect("the double step");
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
                let points: &[u64] = if want_h0 {
                    &[1, 2, 3, 4, 0]
                } else {
                    &[1, 2, 3, 4]
                };
                let mut last_quarter = 0;
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
                    last_quarter = quarter;
                    let at = (1usize << low) + (1usize << rounds) - (1usize << (rounds - j));
                    let mask = (1usize << low) - 1;
                    let mut want = vec![Fp3::zero(); points.len()];
                    for x in 0..quarter {
                        let w = &eq[at + (x >> low)] * &eq[x & mask];
                        let lo: [Fp3; 8] = std::array::from_fn(|k| f[k][x]);
                        let hi: [Fp3; 8] = std::array::from_fn(|k| f[k][x + quarter]);
                        for (row, t) in points.iter().enumerate() {
                            let t = Fp3::from(*t);
                            let v: [Fp3; 8] =
                                std::array::from_fn(|k| &lo[k] + &t * &(&hi[k] - &lo[k]));
                            want[row] = &want[row] + &w * composed(&v, &lam);
                        }
                    }
                    let rows = step
                        .round_enqueue(variant, fold_by.as_ref(), &lambda, want_h0)
                        .expect("a round");
                    let got = fes(&step.round_collect(rows).expect("its rows"));
                    assert_eq!(
                        got, want,
                        "{label} 2^{num_vars} h0 {want_h0}, round {j}: the rows"
                    );
                    if j == 0 && want_h0 {
                        first = got;
                    }
                    checked += 1;
                }
                // The quarters the last round leaves: 2·quarter cells each.
                let raw_p = crate::readback::clone_dtoh(&stream, pd.as_ref()).unwrap();
                let raw_q = crate::readback::clone_dtoh(&stream, qd.as_ref()).unwrap();
                stream.synchronize().unwrap();
                let (dp, dq) = (fes(&raw_p), fes(&raw_q));
                for (k, fk) in f.iter().enumerate() {
                    let dev = if k < 4 { &dp } else { &dq };
                    let base = (k & 3) * stride;
                    assert_eq!(
                        &dev[base..base + 2 * last_quarter],
                        &fk[..2 * last_quarter],
                        "{label} 2^{num_vars} h0 {want_h0}: quarter {k} as the last round leaves it"
                    );
                }
            }
            // Round 0 at t = 1 and t = 0 against today's step: rows H(1) and H(0).
            assert_eq!(
                first[0], today_rows[0],
                "{label} 2^{num_vars}: H(1) is today's"
            );
            assert_eq!(
                first[4], today_rows[2],
                "{label} 2^{num_vars}: H(0) is today's"
            );

            // The faults part.
            let faulted = |fault: Fault| {
                let (pd, qd) = upload(&stream);
                let mut step = DoubleStep::new(&stream, &pd, &qd, num_vars, &point, low)
                    .expect("a faulted step");
                force_fault(fault);
                let rows = step
                    .round_enqueue(variant, None, &lambda, true)
                    .expect("a faulted round");
                force_fault(Fault::None);
                fes(&step.round_collect(rows).expect("its rows"))
            };
            let bad = faulted(Fault::OneCell);
            assert_ne!(
                bad, first,
                "{label} 2^{num_vars}: the off-by-one must part from the reference"
            );
            // The fault reads only a lower-half cell, so H(1) (built from the
            // upper half alone) keeps today's value and H(0) must leave it.
            assert_eq!(
                bad[0], today_rows[0],
                "{label} 2^{num_vars}: the off-by-one leaves H(1), which reads no lower cell"
            );
            assert_ne!(
                bad[4], today_rows[2],
                "{label} 2^{num_vars}: the off-by-one must part from today's H(0)"
            );
            faults += 1;
            if variant.lanes() == 2 {
                let bad = faulted(Fault::Exchange);
                assert_ne!(
                    bad, first,
                    "{label} 2^{num_vars}: the wrong exchange must part from the reference"
                );
                faults += 1;
            }
        }
        eprintln!(
            "GKR2 GATE {label} PASS: {checked} card rounds at 2^10 and 2^12 (with and without \
             H(0)) equal the composed pair, the folded quarters equal the reference's, round 0's \
             H(1), H(0) equal today's step over the middle layer; {faults} faults parted"
        );
    }

    #[test]
    #[ignore = "a card; flips a process-global fault: run alone, --ignored --test-threads=1"]
    fn equal_gate_v0() {
        equal_gate(Variant::V0);
    }

    #[test]
    #[ignore = "a card; flips a process-global fault: run alone, --ignored --test-threads=1"]
    fn equal_gate_v1() {
        equal_gate(Variant::V1);
    }

    #[test]
    #[ignore = "a card; flips a process-global fault: run alone, --ignored --test-threads=1"]
    fn equal_gate_v2() {
        equal_gate(Variant::V2);
    }

    #[test]
    #[ignore = "a card; flips a process-global fault: run alone, --ignored --test-threads=1"]
    fn equal_gate_v3() {
        equal_gate(Variant::V3);
    }

    #[test]
    #[ignore = "a card; flips a process-global fault: run alone, --ignored --test-threads=1"]
    fn equal_gate_v3b() {
        equal_gate(Variant::V3b);
    }

    #[test]
    #[ignore = "a card; flips a process-global fault: run alone, --ignored --test-threads=1"]
    fn equal_gate_v4() {
        equal_gate(Variant::V4);
    }

    #[test]
    #[ignore = "a card; flips a process-global fault: run alone, --ignored --test-threads=1"]
    fn equal_gate_v5() {
        equal_gate(Variant::V5);
    }

    /// Bytes a round moves by the kernels' own reads and writes: `factors`
    /// factors, each read at two cells a cube index (four with a fold, and
    /// two written back).
    fn round_bytes(factors: u64, quarter: u64, fold: bool) -> u64 {
        let cells = if fold { 4 + 2 } else { 2 };
        factors * cells * quarter * 24
    }

    /// Rounds over a cube of at least this many indices are the big,
    /// bandwidth-bound ones.
    const BIG: u64 = 4096;

    struct Arm {
        total_ms: f64,
        big_ms: f64,
        big_bytes: u64,
        small_ms: f64,
        small_n: usize,
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
        Arm {
            total_ms: per_round.iter().map(|r| r.1).sum(),
            big_ms: per_round.iter().filter(|r| r.0 >= BIG).map(|r| r.1).sum(),
            big_bytes: per_round
                .iter()
                .enumerate()
                .filter(|(_, r)| r.0 >= BIG)
                .map(|(j, r)| round_bytes(factors, r.0, j > 0))
                .sum(),
            small_ms: per_round.iter().filter(|r| r.0 < BIG).map(|r| r.1).sum(),
            small_n: per_round.iter().filter(|r| r.0 < BIG).count(),
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

    /// One arm's medians over the repetitions.
    #[derive(Default)]
    struct Tally {
        total: Vec<f64>,
        big: Vec<f64>,
        gbs: Vec<f64>,
        small_us: Vec<f64>,
        small_n: usize,
    }

    impl Tally {
        fn push(&mut self, arms: &[&Arm]) {
            let total: f64 = arms.iter().map(|a| a.total_ms).sum();
            let big: f64 = arms.iter().map(|a| a.big_ms).sum();
            let bytes: u64 = arms.iter().map(|a| a.big_bytes).sum();
            let small: f64 = arms.iter().map(|a| a.small_ms).sum();
            let n: usize = arms.iter().map(|a| a.small_n).sum();
            self.total.push(total);
            self.big.push(big);
            self.gbs.push(bytes as f64 / big / 1e6);
            self.small_us
                .push(if n > 0 { small * 1e3 / n as f64 } else { 0.0 });
            self.small_n = n;
        }
    }

    /// The bench: at depths 24, 22 and 20, the double step's card rounds over
    /// a random layer by each variant against today's two steps over the same
    /// cells — the layer itself, then the middle layer the production fold
    /// makes of it. Every arm runs every card round down to today's host tail
    /// (`2^6` cells), the round kernel and its reduction each, with no
    /// read-back between rounds; CUDA events time each round. Five
    /// repetitions, the arm order rotating by one each repetition; medians.
    /// `LAMBDA_VM_GKR4_BENCH_DEPTHS` (comma-separated) overrides the depths.
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
            .unwrap_or_else(|| vec![24, 22, 20]);
        let low = 6usize;
        let reps = 5usize;
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        // Arm 0 is today's pair; arm 1 + i is `Variant::ALL[i]`.
        let n_arms = 1 + Variant::ALL.len();
        let arm_label = |a: usize| {
            if a == 0 {
                "pair"
            } else {
                Variant::ALL[a - 1].label()
            }
        };
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
            let run_double = |variant: Variant| {
                let pd = Arc::new(stream.clone_dtod(&g_p).unwrap());
                let qd = Arc::new(stream.clone_dtod(&g_q).unwrap());
                let mut d = DoubleStep::new(&stream, &pd, &qd, depth, &point_b, low)
                    .expect("the double step");
                let rd = d.rounds();
                let td = timed(&stream, rd, depth - 2, |j| {
                    d.round_enqueue(variant, fold_by(j).as_ref(), &lambda, false)
                });
                arm(&td, 8)
            };

            let mut tallies: Vec<Tally> = (0..n_arms).map(|_| Tally::default()).collect();
            let (mut step_a, mut step_b) = (vec![], vec![]);
            for rep in 0..reps {
                let order: Vec<usize> = (0..n_arms).map(|i| (i + rep) % n_arms).collect();
                let mut shown = vec![0f64; n_arms];
                for &a in &order {
                    if a == 0 {
                        let (x, y) = run_pair();
                        tallies[0].push(&[&x, &y]);
                        step_a.push(x.total_ms);
                        step_b.push(y.total_ms);
                        shown[0] = x.total_ms + y.total_ms;
                    } else {
                        let d = run_double(Variant::ALL[a - 1]);
                        tallies[a].push(&[&d]);
                        shown[a] = d.total_ms;
                    }
                }
                let line: Vec<String> = (0..n_arms)
                    .map(|a| format!("{} {:.3}", arm_label(a), shown[a]))
                    .collect();
                let order: Vec<&str> = order.iter().map(|&a| arm_label(a)).collect();
                eprintln!(
                    "GKR2 REP depth {depth} rep {rep} order {}: {} ms",
                    order.join(","),
                    line.join(" · ")
                );
            }
            let pair = &tallies[0];
            let (pm, pbm) = (median(pair.total.clone()), median(pair.big.clone()));
            let (pg, pus) = (median(pair.gbs.clone()), median(pair.small_us.clone()));
            eprintln!(
                "GKR2 PAIR depth {depth}: pair {pm:.3} ms (step {depth} {:.3}, step {} {:.3}) · \
                 big rounds {pbm:.3} ms at {pg:.0} GB/s · small rounds {} at {pus:.1} us each · \
                 reps {reps}",
                median(step_a.clone()),
                depth - 1,
                median(step_b.clone()),
                pair.small_n
            );
            for (i, variant) in Variant::ALL.iter().enumerate() {
                let t = &tallies[1 + i];
                let (dm, dbm) = (median(t.total.clone()), median(t.big.clone()));
                let (dg, dus) = (median(t.gbs.clone()), median(t.small_us.clone()));
                eprintln!(
                    "GKR2 BENCH depth {depth} {}: pair {pm:.3} ms · double {dm:.3} ms · ratio \
                     {:.3} · big rounds (quarter >= {BIG}): pair {pbm:.3} ms at {pg:.0} GB/s, \
                     double {dbm:.3} ms at {dg:.0} GB/s, ratio {:.3} · small rounds: pair {} at \
                     {pus:.1} us, double {} at {dus:.1} us · reps {reps}",
                    variant.label(),
                    dm / pm,
                    dbm / pbm,
                    pair.small_n,
                    t.small_n
                );
            }
        }
    }
}
