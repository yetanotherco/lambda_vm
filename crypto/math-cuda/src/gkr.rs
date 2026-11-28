//! A LogUp fraction tree resident on device, and the sumchecks over its
//! layers.
//!
//! The tree is the biggest structure a table's argument holds — its input
//! layer spans `interactions × rows` fractions — and GKR walks down from its
//! output. Building it here lets the card prove the levels in place:
//! a layer's halves are the sumcheck's factors in place, and the round that
//! binds them is the same kernel the batched statements use.
//!
//! Each retained layer used by a sumcheck is visited once, top down, so the
//! rounds can fold it where it lies.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;
use crate::sumcheck::SumcheckSession;

/// One level: numerators and denominators over the same cube.
struct DeviceLayer {
    p: Arc<CudaSlice<u64>>,
    q: Arc<CudaSlice<u64>>,
    num_vars: usize,
    /// The cells `p` and `q` hold: the whole cube, `2^num_vars`, except for an
    /// input layer kept without its padding
    /// ([`DeviceFractionTree::from_truncated_input`]), whose cells from here
    /// on are the padding's `0/1`.
    real: usize,
}

impl DeviceLayer {
    fn whole(&self) -> bool {
        self.real == 1usize << self.num_vars
    }
}

fn invalid() -> cudarc::driver::DriverError {
    cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE)
}

/// The fraction tree, `layers[0]` the output fraction and layer `layers.len()`
/// the input — the order `multilinear::gkr::FractionTree` uses.
pub struct DeviceFractionTree {
    stream: Arc<CudaStream>,
    /// Indexed by distance from the output. Odd levels below the input can
    /// be absent when the caller uses the double ladder.
    layers: Vec<Option<DeviceLayer>>,
    /// The input layer, when this tree carries it. A tree built from the
    /// factors does not: it is the biggest thing here — as many fractions as
    /// interactions times rows — and it is read exactly twice, to fold the
    /// level above it and by its own sumcheck at the very end. Between those
    /// two the caller rebuilds it rather than carry it.
    input: Option<DeviceLayer>,
    input_num_vars: usize,
    /// The room the tree promised itself, when it is the tree that promised
    /// it. A tree that hands its input layer back does not: the promise has to
    /// outlive it, so the caller holds it.
    _room: Option<crate::device::DeviceReservation>,
}

/// A layer's two halves as whoever wrote them leaves them: `p` and `q` over
/// the same cells, before anything decides whether they are a whole cube.
pub type Halves = (CudaSlice<u64>, CudaSlice<u64>);

/// One layer's two halves on device — what a caller hands back when the tree
/// it built asks for its input layer again.
pub struct InputLayer {
    stream: Arc<CudaStream>,
    p: Arc<CudaSlice<u64>>,
    q: Arc<CudaSlice<u64>>,
    num_vars: usize,
}

impl InputLayer {
    pub fn new(
        stream: Arc<CudaStream>,
        p: CudaSlice<u64>,
        q: CudaSlice<u64>,
        num_vars: usize,
    ) -> Self {
        assert_eq!(p.len(), q.len(), "a layer's halves span one cube");
        assert_eq!(p.len(), (1usize << num_vars) * 3, "the cube is the layer");
        Self {
            stream,
            p: Arc::new(p),
            q: Arc::new(q),
            num_vars,
        }
    }

    /// The layer's own sumcheck, the one that ends GKR.
    pub fn sumcheck(
        &self,
        point: &[u64],
        nodes: &[u64],
        consts: &[u64],
        num_slots: usize,
        root_slot: u32,
    ) -> Result<SumcheckSession> {
        sumcheck_over_layer(
            &self.stream,
            &self.p,
            &self.q,
            self.num_vars,
            point,
            nodes,
            consts,
            num_slots,
            root_slot,
        )
    }
}

impl DeviceFractionTree {
    /// Uploads the input layer and folds every level above it.
    ///
    /// `p` and `q` are interleaved ext3 over the same cube.
    pub fn build(p: &[u64], q: &[u64]) -> Result<Self> {
        assert_eq!(p.len(), q.len(), "a layer's halves span one cube");
        assert!(p.len().is_multiple_of(3), "three u64 per ext3 element");
        let elements = p.len() / 3;
        assert!(elements.is_power_of_two(), "the cube is a power of two");

        let be = backend()?;
        let stream = be.next_stream();
        let input_p = crate::device::htod_or_trim(&stream, p)?;
        let input_q = crate::device::htod_or_trim(&stream, q)?;
        Self::from_device(stream, input_p, input_q)
    }

    /// The same for an input layer the device already holds — what the LogUp
    /// fingerprints write straight into.
    pub fn from_device(
        stream: Arc<CudaStream>,
        p: CudaSlice<u64>,
        q: CudaSlice<u64>,
    ) -> Result<Self> {
        Self::from_device_top(stream, p, q, None)
    }

    /// [`from_device`](Self::from_device) with the first fold pass already
    /// run by its writer ([`input_fold2_from_columns`]).
    pub fn from_device_top(
        stream: Arc<CudaStream>,
        p: CudaSlice<u64>,
        q: CudaSlice<u64>,
        top: Option<TopLevels>,
    ) -> Result<Self> {
        assert_eq!(p.len(), q.len(), "a layer's halves span one cube");
        assert!(p.len().is_multiple_of(3), "three u64 per ext3 element");
        let elements = p.len() / 3;
        assert!(elements.is_power_of_two(), "the cube is a power of two");

        let be = backend()?;
        let bytes = carried_bytes(elements);
        let Some(room) = be.reserve(bytes) else {
            crate::device::note_device_fallback();
            return Err(cudarc::driver::DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
            ));
        };
        crate::argue_probe::note_device(crate::argue_probe::Surface::Gkr, bytes);
        let input = DeviceLayer {
            p: Arc::new(p),
            q: Arc::new(q),
            num_vars: elements.trailing_zeros() as usize,
            real: elements,
        };
        let layers = fold_upwards(&stream, be, &input.p, &input.q, elements, elements, top)?;
        Ok(Self {
            stream,
            layers,
            input_num_vars: input.num_vars,
            input: Some(input),
            _room: Some(room),
        })
    }

    /// The same for an input layer that stops where its real fractions do.
    ///
    /// The interactions are padded up to a power of two with `0/1`, and that
    /// padding is half the cube for the widest precompiles. `p` and `q` hold
    /// `real` fractions of a cube of `1 << num_vars`; the first fold reads the
    /// rest as the `0/1` it is, and **the input layer is let go as soon as the
    /// level above it exists**. Its sumcheck comes last, by which time every
    /// level above is spent, so the caller rebuilds it then — which is why
    /// this tree asks for a fraction of what carrying it costs.
    ///
    /// **The caller holds the room**, because the promise has to cover the
    /// input layer it hands back after this tree is gone.
    /// [`padded_peak_bytes`] is what to promise.
    pub fn from_padded_input(
        stream: Arc<CudaStream>,
        p: CudaSlice<u64>,
        q: CudaSlice<u64>,
        real: usize,
        num_vars: usize,
    ) -> Result<Self> {
        assert_eq!(p.len(), q.len(), "a layer's halves span one cube");
        assert_eq!(p.len(), real * 3, "three u64 per ext3 element");
        let full = 1usize << num_vars;
        assert!(real <= full, "the real fractions fit the cube");
        assert!(
            num_vars == 0 || real > full / 2,
            "a count rounded up to a power of two leaves less than half padding"
        );

        let be = backend()?;
        let layers = {
            let p = Arc::new(p);
            let q = Arc::new(q);
            fold_upwards(&stream, be, &p, &q, real, full, None)?
            // `p` and `q` go here, which is the point of this constructor.
        };
        Ok(Self {
            stream,
            layers,
            input: None,
            input_num_vars: num_vars,
            _room: None,
        })
    }

    /// The same truncated input layer, **kept**: the first fold reads its
    /// padding as the `0/1` it is, as [`from_padded_input`](Self::from_padded_input)'s
    /// does, and the layer's own Gruen rounds read it that way too
    /// ([`layer_gruen`](Self::layer_gruen)), so it is never written. That is
    /// the cells of the padding left out of the write, the first fold and the
    /// input layer's first rounds, which are as large as the layer.
    ///
    /// Only the Gruen rounds read a layer kept this way; today's session over
    /// it ([`layer_sumcheck`](Self::layer_sumcheck)) is refused, and its caller
    /// writes the whole layer again.
    ///
    /// **The caller holds the room** — [`truncated_peak_bytes`] — because a
    /// caller that has to write the whole layer for today's session does so
    /// after this tree is gone. A shape no count rounded up to a power of two
    /// gives is refused before anything runs.
    pub fn from_truncated_input(
        stream: Arc<CudaStream>,
        p: CudaSlice<u64>,
        q: CudaSlice<u64>,
        real: usize,
        num_vars: usize,
    ) -> Result<Self> {
        Self::from_truncated_input_top(stream, p, q, real, num_vars, None)
    }

    /// [`from_truncated_input`](Self::from_truncated_input) with the first
    /// fold pass already run by its writer ([`input_fold2_from_columns`]).
    pub fn from_truncated_input_top(
        stream: Arc<CudaStream>,
        p: CudaSlice<u64>,
        q: CudaSlice<u64>,
        real: usize,
        num_vars: usize,
        top: Option<TopLevels>,
    ) -> Result<Self> {
        let full = 1usize.checked_shl(num_vars as u32).ok_or_else(invalid)?;
        let rounded_up = real <= full && real > full / 2;
        if p.len() != real * 3 || q.len() != real * 3 || !rounded_up {
            return Err(invalid());
        }
        let be = backend()?;
        crate::argue_probe::note_device(
            crate::argue_probe::Surface::Gkr,
            truncated_peak_bytes(real, full),
        );
        let input = DeviceLayer {
            p: Arc::new(p),
            q: Arc::new(q),
            num_vars,
            real,
        };
        let layers = fold_upwards(&stream, be, &input.p, &input.q, real, full, top)?;
        Ok(Self {
            stream,
            layers,
            input: Some(input),
            input_num_vars: num_vars,
            _room: None,
        })
    }

    /// Every level plus the input layer, whether or not this tree holds it.
    pub fn num_layers(&self) -> usize {
        self.layers.len() + 1
    }

    pub fn layer_num_vars(&self, layer: usize) -> usize {
        if layer < self.layers.len() {
            layer
        } else {
            self.input_num_vars
        }
    }

    fn layer(&self, layer: usize) -> Result<&DeviceLayer> {
        match self.layers.get(layer) {
            Some(level) => level.as_ref(),
            None if layer == self.layers.len() => self.input.as_ref(),
            None => None,
        }
        .ok_or_else(invalid)
    }

    /// Whether [`layer_sumcheck`](Self::layer_sumcheck) can still prove the
    /// input layer, or the caller has to hand it back first.
    pub fn holds_input(&self) -> bool {
        self.input.is_some()
    }

    /// Whether the input layer it holds was kept without its padding
    /// ([`from_truncated_input`](Self::from_truncated_input)).
    pub fn input_truncated(&self) -> bool {
        self.input.as_ref().is_some_and(|input| !input.whole())
    }

    /// The output fraction `(p, q)`, the one the bus balance is read off.
    pub fn output(&self) -> Result<([u64; 3], [u64; 3])> {
        let top = self
            .layers
            .first()
            .and_then(Option::as_ref)
            .or(self.input.as_ref())
            .expect("a tree has a top");
        let p = crate::readback::clone_dtoh(&self.stream, &top.p.slice(0..3))?;
        let q = crate::readback::clone_dtoh(&self.stream, &top.q.slice(0..3))?;
        self.stream.synchronize()?;
        Ok(([p[0], p[1], p[2]], [q[0], q[1], q[2]]))
    }

    /// A level's halves back on the host, interleaved ext3.
    ///
    /// Only the levels near the output are worth asking for: a round over a
    /// cube of a few hundred is a launch and a wait around a kernel with
    /// almost nothing to sum, and the whole level is a few kilobytes. The
    /// input layer is never one of them — it is the biggest thing here, and a
    /// tree that dropped it has nothing to hand over.
    pub fn layer_to_host(&self, layer: usize) -> Result<(Vec<u64>, Vec<u64>)> {
        let level = self
            .layers
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(invalid)?;
        let p = crate::readback::clone_dtoh(&self.stream, &level.p.slice(0..level.p.len()))?;
        let q = crate::readback::clone_dtoh(&self.stream, &level.q.slice(0..level.q.len()))?;
        self.stream.synchronize()?;
        Ok((p, q))
    }

    /// A sumcheck over layer `layer`, with `eq(point, ·)` as factor 0 and the
    /// layer's four halves — `p_lo`, `p_hi`, `q_lo`, `q_hi` — as factors 1..5.
    ///
    /// The rounds fold those halves where they lie. GKR visits each layer once,
    /// so nothing reads them afterwards, and what the fold leaves behind is
    /// exactly the four values the layer reduces to.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_sumcheck(
        &self,
        layer: usize,
        point: &[u64],
        nodes: &[u64],
        consts: &[u64],
        num_slots: usize,
        root_slot: u32,
    ) -> Result<SumcheckSession> {
        let level = self.layer(layer)?;
        // This session reads every cell of the halves: a layer kept without
        // its padding is the Gruen rounds' alone.
        if !level.whole() {
            return Err(invalid());
        }
        sumcheck_over_layer(
            &self.stream,
            &level.p,
            &level.q,
            level.num_vars,
            point,
            nodes,
            consts,
            num_slots,
            root_slot,
        )
    }
}

/// How the tree's levels are folded, read once.
pub mod two_level_folds {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU8, Ordering};

    static FORCED: AtomicU8 = AtomicU8::new(0);
    static FAULT: AtomicU8 = AtomicU8::new(0);

    /// Whether two levels of the fraction tree are folded in one pass
    /// (`fraction_fold2_ext3`): the middle level is written once and not read
    /// back by the next fold. The same values as one level a pass. On by
    /// default; `LAMBDA_VM_GKR_TWO_LEVEL_FOLDS=0` folds one level a pass.
    pub fn enabled() -> bool {
        match FORCED.load(Ordering::Relaxed) {
            1 => return true,
            2 => return false,
            _ => {}
        }
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("LAMBDA_VM_GKR_TWO_LEVEL_FOLDS")
                .map_or(true, |v| v != "0" && v != "false")
        })
    }

    /// Forces [`enabled`] for trees built after it; `None` back to the
    /// environment's word.
    #[doc(hidden)]
    pub fn force(on: Option<bool>) {
        FORCED.store(
            match on {
                None => 0,
                Some(true) => 1,
                Some(false) => 2,
            },
            Ordering::Relaxed,
        );
    }

    /// ⛔ A FAULT, a negative control's only: every two-level pass writes its
    /// upper cells off by one.
    #[doc(hidden)]
    pub fn force_fault(on: bool) {
        FAULT.store(u8::from(on), Ordering::Relaxed);
    }

    pub(super) fn fault() -> u32 {
        u32::from(FAULT.load(Ordering::Relaxed))
    }
}

std::thread_local! {
    static SKIP_MIDDLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Builds trees for a double ladder inside `f`. The scope covers both
/// reservations and launches and is restored on unwind.
pub fn with_skip_middle<R>(skip: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SKIP_MIDDLE.set(self.0);
        }
    }
    let _restore = Restore(SKIP_MIDDLE.replace(skip && two_level_folds::enabled()));
    f()
}

fn skip_middle() -> bool {
    SKIP_MIDDLE.get()
}

/// Allocates both output halves of a level. The caller must launch a kernel
/// that writes every element before handing them to a reader.
unsafe fn alloc_halves(stream: &Arc<CudaStream>, cells: usize) -> Result<Halves> {
    unsafe {
        Ok((
            crate::device::alloc_or_trim::<u64>(stream, cells * 3)?,
            crate::device::alloc_or_trim::<u64>(stream, cells * 3)?,
        ))
    }
}

impl DeviceLayer {
    fn from_halves((p, q): Halves, num_vars: usize) -> Self {
        Self {
            p: Arc::new(p),
            q: Arc::new(q),
            num_vars,
            real: 1usize << num_vars,
        }
    }
}

/// The levels above an input of `full` fractions, with padding read as
/// `0/1`. The double ladder reads even levels, and reads the input however
/// high it is. An odd input thus folds once before the two-level passes.
fn fold_upwards(
    stream: &Arc<CudaStream>,
    be: &crate::device::Backend,
    p: &Arc<CudaSlice<u64>>,
    q: &Arc<CudaSlice<u64>>,
    real: usize,
    full: usize,
    top: Option<TopLevels>,
) -> Result<Vec<Option<DeviceLayer>>> {
    let mut num_vars = full.trailing_zeros() as usize;
    let mut layers: Vec<Option<DeviceLayer>> = (0..num_vars).map(|_| None).collect();
    let mut below = (p.clone(), q.clone());
    if let Some(top) = top {
        let input_vars = num_vars;
        num_vars = num_vars.checked_sub(top.folds).ok_or_else(invalid)?;
        if let Some(mid) = top.mid {
            layers[input_vars - 1] = Some(DeviceLayer::from_halves(mid, input_vars - 1));
        }
        let out = DeviceLayer::from_halves(top.out, num_vars);
        below = (out.p.clone(), out.q.clone());
        layers[num_vars] = Some(out);
    }
    let two_levels = two_level_folds::enabled();
    let skip = skip_middle();
    let input_vars = full.trailing_zeros() as usize;
    while num_vars > 0 {
        let half = 1usize << (num_vars - 1);
        let padded = num_vars == input_vars && real != full;
        let pair = two_levels
            && num_vars >= 2
            && if skip {
                num_vars.is_multiple_of(2)
            } else {
                !padded
            };
        let output_vars = num_vars - if pair { 2 } else { 1 };
        // SAFETY: every element of each allocated level is written below.
        let (mut p_out, mut q_out) = unsafe { alloc_halves(stream, 1 << output_vars) }?;
        let half_arg = half as u64;
        if pair {
            let mut mid = if skip {
                None
            } else {
                Some(unsafe { alloc_halves(stream, half) }?)
            };
            let null = 0u64;
            let real_arg = if padded { real as u64 } else { 0 };
            let fault = two_level_folds::fault();
            unsafe {
                let mut launch = stream.launch_builder(&be.fraction_fold2_ext3);
                launch
                    .arg(below.0.as_ref())
                    .arg(below.1.as_ref())
                    .arg(&half_arg);
                match &mut mid {
                    Some((p_mid, q_mid)) => launch.arg(p_mid).arg(q_mid),
                    None => launch.arg(&null).arg(&null),
                };
                launch
                    .arg(&mut p_out)
                    .arg(&mut q_out)
                    .arg(&fault)
                    .arg(&real_arg)
                    .launch(LaunchConfig {
                        // The kernel's register count limits the block size.
                        grid_dim: ((half as u64 / 2).div_ceil(256) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            if let Some(mid) = mid {
                layers[num_vars - 1] = Some(DeviceLayer::from_halves(mid, num_vars - 1));
            }
        } else {
            unsafe {
                if padded {
                    let real_arg = real as u64;
                    stream
                        .launch_builder(&be.fraction_fold_padded_ext3)
                        .arg(below.0.as_ref())
                        .arg(below.1.as_ref())
                        .arg(&half_arg)
                        .arg(&real_arg)
                        .arg(&mut p_out)
                        .arg(&mut q_out)
                        .launch(LaunchConfig::for_num_elems(half as u32))?;
                } else {
                    stream
                        .launch_builder(&be.fraction_fold_ext3)
                        .arg(below.0.as_ref())
                        .arg(below.1.as_ref())
                        .arg(&half_arg)
                        .arg(&mut p_out)
                        .arg(&mut q_out)
                        .launch(LaunchConfig::for_num_elems(half as u32))?;
                }
            }
        }
        num_vars = output_vars;
        let out = DeviceLayer::from_halves((p_out, q_out), num_vars);
        below = (out.p.clone(), out.q.clone());
        layers[num_vars] = Some(out);
    }
    Ok(layers)
}

/// A sumcheck over one layer's halves, with `eq(point, ·)` as factor 0 and
/// `p_lo`, `p_hi`, `q_lo`, `q_hi` as factors 1..5.
///
/// The rounds fold those halves where they lie. GKR visits each layer once, so
/// nothing reads them afterwards, and what the fold leaves behind is exactly
/// the four values the layer reduces to.
#[allow(clippy::too_many_arguments)]
pub fn sumcheck_over_layer(
    stream: &Arc<CudaStream>,
    p: &Arc<CudaSlice<u64>>,
    q: &Arc<CudaSlice<u64>>,
    num_vars: usize,
    point: &[u64],
    nodes: &[u64],
    consts: &[u64],
    num_slots: usize,
    root_slot: u32,
) -> Result<SumcheckSession> {
    assert!(num_vars > 0, "the output layer has nothing to bind");
    let half_vars = num_vars - 1;
    assert_eq!(point.len(), half_vars * 3, "the point spans the halves");

    let half = 1usize << half_vars;
    let eq = Arc::new(crate::sumcheck::eq_table_ext3(stream, point, half)?);
    let addresses = {
        let (eq_at, _eq_guard) = eq.device_ptr(stream);
        let (p_at, _p_guard) = p.device_ptr(stream);
        let (q_at, _q_guard) = q.device_ptr(stream);
        let stride = (half * 3 * 8) as u64;
        vec![eq_at, p_at, p_at + stride, q_at, q_at + stride]
    };

    SumcheckSession::from_device(
        stream.clone(),
        &addresses,
        half,
        vec![eq, p.clone(), q.clone()],
        nodes,
        consts,
        num_slots,
        root_slot,
    )
}

/// Cells of each side above the input in a double ladder's tree: the
/// geometric sum of powers of four, including the root for a nonempty sum.
fn even_layers(full: usize) -> usize {
    let odd = full.trailing_zeros() % 2 == 1;
    let even = full >> u32::from(odd);
    even / 3 + if odd { even } else { 0 }
}

/// Whole input plus the retained levels. Dense trees keep their original
/// reservation, rounded up by one fraction.
pub fn carried_bytes(full: usize) -> u64 {
    let above = if skip_middle() {
        even_layers(full)
    } else {
        full
    };
    (full + above) as u64 * 48
}

/// What a tree built by [`DeviceFractionTree::from_padded_input`] holds at
/// once: the truncated input layer plus the first pass's outputs while it
/// runs, against the retained levels once the input is gone — and, later, the
/// full input layer handed back for its own sumcheck, by which time the levels
/// are spent.
pub fn padded_peak_bytes(real: usize, full: usize) -> u64 {
    let first = if skip_middle() && full.trailing_zeros().is_multiple_of(2) {
        full / 4
    } else {
        full / 2
    };
    (real + first).max(full) as u64 * 48
}

/// What a tree built by [`DeviceFractionTree::from_truncated_input`] holds at
/// once: the truncated input layer and the retained levels above it. Also
/// covers the whole input layer if a session needs it written again after
/// the tree is gone.
pub fn truncated_peak_bytes(real: usize, full: usize) -> u64 {
    let above = if skip_middle() {
        even_layers(full)
    } else {
        full
    };
    (real + above).max(full) as u64 * 48
}

/// The run of an upper half's cells a layer kept without its padding still
/// holds after a round over `quarter` indices: the fold leaves a half of
/// `2·quarter` cells, and a cell past the run held before it folds two padding
/// cells into the padding again.
pub fn stored_after_round(stored: usize, quarter: usize) -> usize {
    stored.min(2 * quarter)
}

/// Blocks a Gruen round launches at most; past it the grid-stride loop takes
/// over. A few blocks an SM on a 170-SM card, and the partials stay small.
const GRUEN_MAX_GRID: u32 = 2048;

/// One GKR layer's device rounds with Gruen's split (D-ARGUE S1-3): the round
/// sums `H(1)` and `H(2)` of `h = p_lo·q_hi + p_hi·q_lo + λ·q_lo·q_hi` under
/// `eq(u_{>j}, ·)`, and the host puts `eq(u_{≤j})` back in (see
/// `multilinear::gkr_gruen`). The fold by a round's challenge rides the next
/// round's pass, and the weight is two small tables a layer instead of an `eq`
/// table the size of the layer folded every round.
///
/// Folds the layer's halves in place, as [`sumcheck_over_layer`]'s session
/// does: GKR visits each layer once.
pub struct GruenLayer {
    stream: Arc<CudaStream>,
    /// The buffers the halves live in, kept alive; the kernels see addresses.
    p: Arc<CudaSlice<u64>>,
    q: Arc<CudaSlice<u64>>,
    p_at: u64,
    q_at: u64,
    /// Cells in each half: `2^m`.
    half: usize,
    /// The run of cells the upper halves `p_hi`, `q_hi` hold before the next
    /// fold — every cell of the cube they hold now, for a layer stored whole.
    /// Past it is the padding of a layer kept without it
    /// ([`DeviceFractionTree::from_truncated_input`]), read as `0/1`.
    stored: usize,
    /// Whether the layer was opened without its padding.
    truncated: bool,
    /// The point's limbs, `m` ext3 values.
    point: Vec<u64>,
    /// `L`: the variables left to the host, `2^L` cells.
    low: usize,
    /// `J = m − L`: the rounds this layer runs here.
    rounds: usize,
    /// Every `eq` level ([`gkr_eq_levels_ext3`'s layout](crate) — `e_lo`, then
    /// one `e_hi` level per round).
    eq: CudaSlice<u64>,
    partials: CudaSlice<u64>,
    sums: CudaSlice<u64>,
    /// Rounds run so far.
    done: usize,
    /// The rounds' sums through a slot (`LAMBDA_VM_ARGUE_ROUND_SLOT`):
    /// reduced by the round kernel's last block, read from pinned host memory.
    slot: Option<crate::round_slot::SlotRounds>,
}

impl GruenLayer {
    /// Over one layer's halves: `p` and `q` of `2^num_vars` cells each — or
    /// the first `real` of them, more than half, for an input layer kept
    /// without its padding — the point `m = num_vars − 1` ext3 values, and at
    /// most `low` variables left to the host (fewer when the layer is smaller).
    #[allow(clippy::too_many_arguments)]
    fn new(
        stream: &Arc<CudaStream>,
        p: &Arc<CudaSlice<u64>>,
        q: &Arc<CudaSlice<u64>>,
        num_vars: usize,
        real: usize,
        point: &[u64],
        low: usize,
    ) -> Result<Self> {
        assert!(num_vars > 0, "the output layer has nothing to bind");
        let m = num_vars - 1;
        assert_eq!(point.len(), m * 3, "the point spans the halves");
        let half = 1usize << m;
        if p.len() != real * 3 || q.len() != real * 3 || real <= half || real > 2 * half {
            return Err(invalid());
        }
        let low = low.min(m);
        let rounds = m - low;

        let be = backend()?;
        let cells = (1usize << low) + (1usize << rounds) - 1;
        // SAFETY: the kernel writes every cell.
        let mut eq = unsafe { crate::device::alloc_or_trim::<u64>(stream, cells * 3) }?;
        let point_dev = crate::device::htod_or_trim(
            stream,
            if point.is_empty() { &[0u64][..] } else { point },
        )?;
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
        let partials = unsafe {
            crate::device::alloc_or_trim::<u64>(stream, 3 * GRUEN_MAX_GRID as usize * 3)
        }?;
        let sums = crate::device::alloc_zeros_or_trim::<u64>(stream, 3 * 3)?;
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
            p: p.clone(),
            q: q.clone(),
            p_at,
            q_at,
            half,
            stored: real - half,
            truncated: real < 2 * half,
            point: point.to_vec(),
            low,
            rounds,
            eq,
            partials,
            sums,
            done: 0,
            slot: None,
        })
    }

    /// The rounds' sums through a slot from here on (`crate::round_slot`)
    /// when `mode` asks for one and one can be had; `false` keeps today's
    /// second launch and read-back.
    pub fn use_slot(&mut self, mode: crate::round_slot::RoundSlot) -> bool {
        self.slot = crate::round_slot::SlotRounds::open(&self.stream, mode);
        self.slot.is_some()
    }

    /// The rounds this layer runs on the card, `J`.
    pub fn rounds(&self) -> usize {
        self.rounds
    }

    /// Whether the layer was opened without its padding.
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// The variables left to the host, `L`: the halves come back over `2^L`.
    pub fn low(&self) -> usize {
        self.low
    }

    /// The next round's sums, `H(1)` and `H(2)` — and `H(0)` third with
    /// `want_h0` — three u64 each. `fold_by` is the previous round's challenge,
    /// and every round but the first has one.
    pub fn round(
        &mut self,
        fold_by: Option<&[u64; 3]>,
        lambda: &[u64; 3],
        want_h0: bool,
    ) -> Result<Vec<u64>> {
        let rows = self.round_enqueue(fold_by, lambda, want_h0)?;
        self.round_collect(rows)
    }

    /// [`round`](Self::round)'s launches, queued without waiting for them —
    /// so the batched argue's ladder can queue every tree's round before it
    /// reads any back. Returns the rows [`round_collect`](Self::round_collect)
    /// reads.
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
            "a challenge to fold by after the first round"
        );
        let be = backend()?;
        let m = self.half.trailing_zeros() as usize;
        let quarter = 1u64 << (m - self.done - 1);
        let (grid, block) = gruen_shape(quarter);
        let rows = if want_h0 { 3u32 } else { 2 };
        let s = fold_by.copied().unwrap_or([0; 3]);
        let fold = u32::from(fold_by.is_some());
        let at =
            (1usize << self.low) + (1usize << self.rounds) - (1usize << (self.rounds - self.done));
        let e_lo = self.eq.slice(0..(1usize << self.low) * 3);
        let e_hi = self
            .eq
            .slice(at * 3..(at + (1usize << (self.rounds - 1 - self.done))) * 3);
        let half = self.half as u64;
        let low = self.low as u32;
        let want = u32::from(want_h0);
        let stored = self.stored as u64;
        // The cube the upper halves hold coming into this round; once their
        // stored run covers it, the padding is spent and today's kernel reads
        // every cell.
        let cube = if fold_by.is_some() { 4 } else { 2 } * quarter;
        let kernel = match (stored < cube, self.slot.is_some()) {
            (true, false) => &be.gkr_round_gruen_real,
            (false, false) => &be.gkr_round_gruen,
            (true, true) => &be.gkr_round_gruen_real_slot,
            (false, true) => &be.gkr_round_gruen_slot,
        };
        let config = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: if self.slot.is_some() {
                crate::round_slot::shared_bytes(block)
            } else {
                block * 3 * 8
            },
        };
        // Today's second launch, unless the slot's last block does its work.
        let second = self.slot.as_ref().is_none_or(|slot| slot.checking());
        let mut slot_args = self.slot.as_mut().map(|slot| slot.launch());
        unsafe {
            let mut launch = self.stream.launch_builder(kernel);
            launch
                .arg(&self.p_at)
                .arg(&self.q_at)
                .arg(&half)
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
                .arg(&want);
            if stored < cube {
                launch.arg(&stored);
            }
            launch.arg(&mut self.partials);
            if let Some((ticket, dev, seq)) = slot_args.as_mut() {
                launch.arg(&mut **ticket).arg(&*dev).arg(&*seq);
            }
            launch.launch(config)?;
            if second {
                crate::round_slot::note_second_launch();
                let reduce_block = 256u32;
                self.stream
                    .launch_builder(&be.sum_partials_ext3)
                    .arg(&self.partials)
                    .arg(&(grid as u64))
                    .arg(&mut self.sums)
                    .launch(LaunchConfig {
                        grid_dim: (rows, 1, 1),
                        block_dim: (reduce_block, 1, 1),
                        shared_mem_bytes: reduce_block * 3 * 8,
                    })?;
            }
        }
        self.stored = stored_after_round(self.stored, quarter as usize);
        self.done += 1;
        Ok(rows)
    }

    /// The sums a [`round_enqueue`](Self::round_enqueue) queued, `rows` of
    /// them, three u64 each.
    pub fn round_collect(&self, rows: u32) -> Result<Vec<u64>> {
        let today = || -> Result<Vec<u64>> {
            let sums =
                crate::readback::clone_dtoh(&self.stream, &self.sums.slice(0..rows as usize * 3))?;
            self.stream.synchronize()?;
            Ok(sums)
        };
        match self.slot.as_ref() {
            Some(slot) => slot.collect(rows as usize, today),
            None => today(),
        }
    }

    /// After the last card round: the four halves bound to its challenge
    /// (`fold_by`, `None` for a layer that ran no card round), over `2^L`
    /// cells, laid end to end — `p_lo, p_hi, q_lo, q_hi`, three u64 a cell.
    /// Also left in place, where a [`shadow`](Self::shadow) reads them.
    pub fn finish(&mut self, fold_by: Option<&[u64; 3]>) -> Result<Vec<u64>> {
        assert_eq!(self.done, self.rounds, "the card's rounds are done");
        assert_eq!(
            fold_by.is_some(),
            self.rounds > 0,
            "a challenge to fold by after a card round"
        );
        let be = backend()?;
        let cells = 1u64 << self.low;
        // SAFETY: the kernel writes every cell.
        let mut out =
            unsafe { crate::device::alloc_or_trim::<u64>(&self.stream, 4 * cells as usize * 3) }?;
        let s = fold_by.copied().unwrap_or([0; 3]);
        let fold = u32::from(fold_by.is_some());
        let half = self.half as u64;
        let stored = self.stored as u64;
        // The cube the halves hold: twice the cells with a fold to run.
        let cube = if fold_by.is_some() { 2 * cells } else { cells };
        let kernel = if stored < cube {
            &be.gkr_gruen_finish_real
        } else {
            &be.gkr_gruen_finish
        };
        unsafe {
            let mut launch = self.stream.launch_builder(kernel);
            launch
                .arg(&self.p_at)
                .arg(&self.q_at)
                .arg(&half)
                .arg(&cells)
                .arg(&fold)
                .arg(&s[0])
                .arg(&s[1])
                .arg(&s[2]);
            if stored < cube {
                launch.arg(&stored);
            }
            launch.arg(&mut out).launch(LaunchConfig {
                grid_dim: ((4 * cells).div_ceil(256).min(4096) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }
        let values = crate::readback::clone_dtoh(&self.stream, &out)?;
        self.stream.synchronize()?;
        Ok(values)
    }

    /// Today's session over the same halves, for checking these rounds against
    /// it (`LAMBDA_VM_ARGUE_GKR_GRUEN_XCHECK`): factor 0 its own `eq(u, ·)`
    /// table, factors 1..5 the halves where they lie. Run its round after this
    /// layer's round `j` has folded the halves, with
    /// [`SumcheckSession::fold_first`] keeping its `eq` in step.
    ///
    /// Refused for a layer opened without its padding: today's session reads
    /// every cell of the halves.
    pub fn shadow(
        &self,
        nodes: &[u64],
        consts: &[u64],
        num_slots: usize,
        root_slot: u32,
    ) -> Result<SumcheckSession> {
        if self.truncated {
            return Err(invalid());
        }
        let eq = Arc::new(crate::sumcheck::eq_table_ext3(
            &self.stream,
            &self.point,
            self.half,
        )?);
        let eq_at = {
            let (at, _guard) = eq.device_ptr(&self.stream);
            at
        };
        let stride = (self.half * 3 * 8) as u64;
        let addresses = vec![
            eq_at,
            self.p_at,
            self.p_at + stride,
            self.q_at,
            self.q_at + stride,
        ];
        SumcheckSession::from_device(
            self.stream.clone(),
            &addresses,
            self.half,
            vec![eq, self.p.clone(), self.q.clone()],
            nodes,
            consts,
            num_slots,
            root_slot,
        )
    }

    /// Bytes [`shadow`](Self::shadow) allocates: the `eq` table and today's
    /// round scratch.
    pub fn shadow_bytes(&self, num_slots: usize) -> u64 {
        self.half as u64 * 24 + SumcheckSession::scratch_bytes(self.half, num_slots)
    }
}

/// A Gruen round's launch: 256-thread blocks for a wide cube, narrower ones —
/// down to a warp — for a narrow one, and at most [`GRUEN_MAX_GRID`] blocks.
fn gruen_shape(quarter: u64) -> (u32, u32) {
    let block = quarter.next_power_of_two().clamp(32, 256) as u32;
    let grid = quarter
        .div_ceil(block as u64)
        .clamp(1, GRUEN_MAX_GRID as u64) as u32;
    (grid, block)
}

impl DeviceFractionTree {
    /// Layer `layer`'s Gruen rounds ([`GruenLayer`]), at most `low` variables
    /// left to the host.
    pub fn layer_gruen(&self, layer: usize, point: &[u64], low: usize) -> Result<GruenLayer> {
        let level = self.layer(layer)?;
        GruenLayer::new(
            &self.stream,
            &level.p,
            &level.q,
            level.num_vars,
            level.real,
            point,
            low,
        )
    }
}

impl InputLayer {
    /// The input layer's Gruen rounds, the ones that end GKR.
    pub fn gruen(&self, point: &[u64], low: usize) -> Result<GruenLayer> {
        GruenLayer::new(
            &self.stream,
            &self.p,
            &self.q,
            self.num_vars,
            1usize << self.num_vars,
            point,
            low,
        )
    }
}

/// A GKR step's device rounds over TWO tree layers (`multilinear::gkr_lockstep`'s two-layer
/// ladder) with Gruen's split: the step proves layer `ℓ` against layer `ℓ + 2`, whose four
/// quarters (`p` and `q` at cell `j·2^m`, `m = ℓ`) are its factors. A round sums `H(1)..H(4)` of
/// `N + λQ` — `N = N₀D₁ + N₁D₀`, `Q = D₀D₁` over the middle layer's halves `N_r = p_r q_{r+2} +
/// p_{r+2} q_r`, `D_r = q_r q_{r+2}` — under `eq(u_{>j}, ·)`, and `H(0)` where asked; the host
/// puts `eq(u_{≤j})` back in. The kernels are `sumcheck.cu`'s `gkr2_*`: a lane pair a cube index,
/// each lane one half of the middle layer.
///
/// Folds the quarters in place, as [`GruenLayer`] folds its halves: GKR visits each layer once.
pub struct Gruen2Layer {
    stream: Arc<CudaStream>,
    /// The buffers the quarters live in, kept alive; the kernels see addresses.
    _p: Arc<CudaSlice<u64>>,
    _q: Arc<CudaSlice<u64>>,
    p_at: u64,
    q_at: u64,
    /// Cells in each quarter: `2^m`.
    stride: usize,
    /// The runs of cells quarters 2 and 3 hold before the next fold — every cell of the cube they
    /// hold now, for a layer stored whole. Past them is the padding of an input layer kept without
    /// it ([`DeviceFractionTree::from_truncated_input`]), read as `0/1`.
    stored: [usize; 2],
    truncated: bool,
    /// `L`: the variables left to the host, `2^L` cells a quarter.
    low: usize,
    /// `J = m − L`: the rounds this layer runs here.
    rounds: usize,
    /// Every `eq` level, as [`GruenLayer`] lays them out.
    eq: CudaSlice<u64>,
    partials: CudaSlice<u64>,
    sums: CudaSlice<u64>,
    done: usize,
    slot: Option<crate::round_slot::SlotRounds>,
    /// Whether the round queued last handed its sums to the slot.
    slotted: bool,
}

/// The rows a two-layer round sums: `H(1)..H(4)`, and `H(0)` fifth.
const GRUEN2_ROWS: usize = 5;

impl Gruen2Layer {
    /// Over layer `ℓ + 2`'s `p` and `q`, `2^num_vars` cells each — or the first `real` of them,
    /// more than half, for an input layer kept without its padding — the point `m = num_vars − 2`
    /// ext3 values, and at most `low` variables left to the host (fewer when the layer is
    /// smaller).
    #[allow(clippy::too_many_arguments)]
    fn new(
        stream: &Arc<CudaStream>,
        p: &Arc<CudaSlice<u64>>,
        q: &Arc<CudaSlice<u64>>,
        num_vars: usize,
        real: usize,
        point: &[u64],
        low: usize,
    ) -> Result<Self> {
        if num_vars < 2 {
            return Err(invalid());
        }
        let m = num_vars - 2;
        if point.len() != m * 3 {
            return Err(invalid());
        }
        let stride = 1usize << m;
        if p.len() != real * 3 || q.len() != real * 3 || real <= 2 * stride || real > 4 * stride {
            return Err(invalid());
        }
        let low = low.min(m);
        let rounds = m - low;

        let be = backend()?;
        let cells = (1usize << low) + (1usize << rounds) - 1;
        // SAFETY: the kernel writes every cell.
        let mut eq = unsafe { crate::device::alloc_or_trim::<u64>(stream, cells * 3) }?;
        let point_dev = crate::device::htod_or_trim(
            stream,
            if point.is_empty() { &[0u64][..] } else { point },
        )?;
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
        let partials = unsafe {
            crate::device::alloc_or_trim::<u64>(stream, GRUEN2_ROWS * GRUEN_MAX_GRID as usize * 3)
        }?;
        let sums = crate::device::alloc_zeros_or_trim::<u64>(stream, GRUEN2_ROWS * 3)?;
        let p_at = {
            let (at, _guard) = p.device_ptr(stream);
            at
        };
        let q_at = {
            let (at, _guard) = q.device_ptr(stream);
            at
        };
        let upper = real - 2 * stride;
        Ok(Self {
            stream: stream.clone(),
            _p: p.clone(),
            _q: q.clone(),
            p_at,
            q_at,
            stride,
            stored: [upper.min(stride), upper.saturating_sub(stride)],
            truncated: real < 4 * stride,
            low,
            rounds,
            eq,
            partials,
            sums,
            done: 0,
            slot: None,
            slotted: false,
        })
    }

    /// The rounds' sums through a slot from here on (`crate::round_slot`) when `mode` asks for
    /// one and one can be had; `false` keeps the second launch and read-back. A round that sums
    /// `H(0)` runs without it.
    pub fn use_slot(&mut self, mode: crate::round_slot::RoundSlot) -> bool {
        self.slot = crate::round_slot::SlotRounds::open(&self.stream, mode);
        self.slot.is_some()
    }

    /// The rounds this layer runs on the card, `J`.
    pub fn rounds(&self) -> usize {
        self.rounds
    }

    /// Whether the layer was opened without its padding.
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// The variables left to the host, `L`: the factors come back over `2^L` cells.
    pub fn low(&self) -> usize {
        self.low
    }

    /// The next round's sums, `H(1)..H(4)` — and `H(0)` fifth with `want_h0` — three u64 each.
    /// `fold_by` is the previous round's challenge, and every round but the first has one;
    /// `half_lambda` is `λ/2`, which the round takes, and `lambda` the `H(0)` pass's.
    pub fn round(
        &mut self,
        fold_by: Option<&[u64; 3]>,
        lambda: &[u64; 3],
        half_lambda: &[u64; 3],
        want_h0: bool,
    ) -> Result<Vec<u64>> {
        let rows = self.round_enqueue(fold_by, lambda, half_lambda, want_h0)?;
        self.round_collect(rows)
    }

    /// [`round`](Self::round)'s launches, queued without waiting for them. Returns the rows
    /// [`round_collect`](Self::round_collect) reads.
    pub fn round_enqueue(
        &mut self,
        fold_by: Option<&[u64; 3]>,
        lambda: &[u64; 3],
        half_lambda: &[u64; 3],
        want_h0: bool,
    ) -> Result<u32> {
        assert!(self.done < self.rounds, "every card round has run");
        assert_eq!(
            fold_by.is_some(),
            self.done > 0,
            "a challenge to fold by after the first round"
        );
        let be = backend()?;
        let m = self.stride.trailing_zeros() as usize;
        let quarter = 1u64 << (m - self.done - 1);
        let (grid, block) = gruen2_shape(quarter);
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
        let [stored2, stored3] = self.stored.map(|run| run as u64);
        // The cube the quarters hold coming into this round; once both stored runs cover it, the
        // padding is spent and the whole-layer kernel reads every cell.
        let cube = if fold_by.is_some() { 4 } else { 2 } * quarter;
        let real = stored2 < cube || stored3 < cube;
        let slotted = self.slot.is_some() && !want_h0;
        let kernel = match (real, slotted) {
            (true, false) => &be.gkr2_round_real,
            (false, false) => &be.gkr2_round,
            (true, true) => &be.gkr2_round_real_slot,
            (false, true) => &be.gkr2_round_slot,
        };
        let shared = if slotted {
            gruen2_shared(block).max(crate::round_slot::shared_bytes(block))
        } else {
            gruen2_shared(block)
        };
        let second = !slotted || self.slot.as_ref().is_some_and(|slot| slot.checking());
        let mut slot_args = if slotted {
            self.slot.as_mut().map(|slot| slot.launch())
        } else {
            None
        };
        unsafe {
            let mut launch = self.stream.launch_builder(kernel);
            launch
                .arg(&self.p_at)
                .arg(&self.q_at)
                .arg(&stride)
                .arg(&quarter)
                .arg(&fold)
                .arg(&s[0])
                .arg(&s[1])
                .arg(&s[2])
                .arg(&half_lambda[0])
                .arg(&half_lambda[1])
                .arg(&half_lambda[2])
                .arg(&e_hi)
                .arg(&e_lo)
                .arg(&low)
                .arg(&stored2)
                .arg(&stored3)
                .arg(&mut self.partials);
            if let Some((ticket, dev, seq)) = slot_args.as_mut() {
                launch.arg(&mut **ticket).arg(&*dev).arg(&*seq);
            }
            launch.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            })?;
        }
        self.stored = self
            .stored
            .map(|run| stored_after_round(run, quarter as usize));
        if want_h0 {
            // On the round's grid and block, so its partials sit where the round's would; it reads
            // the cells the round left, under the runs after its fold.
            let [stored2, stored3] = self.stored.map(|run| run as u64);
            let kernel = if stored2 < 2 * quarter || stored3 < 2 * quarter {
                &be.gkr2_round_h0_real
            } else {
                &be.gkr2_round_h0
            };
            unsafe {
                self.stream
                    .launch_builder(kernel)
                    .arg(&self.p_at)
                    .arg(&self.q_at)
                    .arg(&stride)
                    .arg(&quarter)
                    .arg(&lambda[0])
                    .arg(&lambda[1])
                    .arg(&lambda[2])
                    .arg(&e_hi)
                    .arg(&e_lo)
                    .arg(&low)
                    .arg(&stored2)
                    .arg(&stored3)
                    .arg(&mut self.partials)
                    .launch(LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: (block / 32).max(1) * 3 * 8,
                    })?;
            }
        }
        if second {
            crate::round_slot::note_second_launch();
            let reduce_block = 256u32;
            unsafe {
                self.stream
                    .launch_builder(&be.sum_partials_ext3)
                    .arg(&self.partials)
                    .arg(&(grid as u64))
                    .arg(&mut self.sums)
                    .launch(LaunchConfig {
                        grid_dim: (rows, 1, 1),
                        block_dim: (reduce_block, 1, 1),
                        shared_mem_bytes: reduce_block * 3 * 8,
                    })?;
            }
        }
        self.slotted = slotted;
        self.done += 1;
        Ok(rows)
    }

    /// The sums a [`round_enqueue`](Self::round_enqueue) queued, `rows` of them, three u64 each.
    pub fn round_collect(&self, rows: u32) -> Result<Vec<u64>> {
        let today = || -> Result<Vec<u64>> {
            let sums =
                crate::readback::clone_dtoh(&self.stream, &self.sums.slice(0..rows as usize * 3))?;
            self.stream.synchronize()?;
            Ok(sums)
        };
        match self.slot.as_ref().filter(|_| self.slotted) {
            Some(slot) => slot.collect(rows as usize, today),
            None => today(),
        }
    }

    /// After the last card round: the eight factors bound to its challenge (`fold_by`, `None` for
    /// a layer that ran no card round), over `2^L` cells each, laid end to end in the order a step
    /// sends its two halves — `h₀`'s `p_lo, p_hi, q_lo, q_hi` (quarters 0 and 2), then `h₁`'s
    /// (quarters 1 and 3) — three u64 a cell.
    pub fn finish(&mut self, fold_by: Option<&[u64; 3]>) -> Result<Vec<u64>> {
        assert_eq!(self.done, self.rounds, "the card's rounds are done");
        assert_eq!(
            fold_by.is_some(),
            self.rounds > 0,
            "a challenge to fold by after a card round"
        );
        let be = backend()?;
        let cells = 1u64 << self.low;
        // SAFETY: the kernel writes every cell.
        let mut out =
            unsafe { crate::device::alloc_or_trim::<u64>(&self.stream, 8 * cells as usize * 3) }?;
        let s = fold_by.copied().unwrap_or([0; 3]);
        let fold = u32::from(fold_by.is_some());
        let stride = self.stride as u64;
        let [stored2, stored3] = self.stored.map(|run| run as u64);
        let cube = if fold_by.is_some() { 2 * cells } else { cells };
        let real = stored2 < cube || stored3 < cube;
        let config = LaunchConfig {
            grid_dim: ((8 * cells).div_ceil(256).min(4096) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut launch = self.stream.launch_builder(if real {
                &be.gkr2_finish_real
            } else {
                &be.gkr2_finish
            });
            launch
                .arg(&self.p_at)
                .arg(&self.q_at)
                .arg(&stride)
                .arg(&cells)
                .arg(&fold)
                .arg(&s[0])
                .arg(&s[1])
                .arg(&s[2]);
            if real {
                launch.arg(&stored2).arg(&stored3);
            }
            launch.arg(&mut out).launch(config)?;
        }
        let values = crate::readback::clone_dtoh(&self.stream, &out)?;
        self.stream.synchronize()?;
        Ok(values)
    }
}

/// A two-layer round's launch: a lane pair a cube index, 256-lane blocks for a wide cube and
/// narrower ones — down to a warp — for a narrow one, at most [`GRUEN_MAX_GRID`] blocks.
fn gruen2_shape(quarter: u64) -> (u32, u32) {
    let lanes = 2 * quarter;
    let block = lanes.next_power_of_two().clamp(32, 256) as u32;
    let grid = lanes.div_ceil(block as u64).clamp(1, GRUEN_MAX_GRID as u64) as u32;
    (grid, block)
}

/// The shared memory a two-layer round takes at `block` lanes: each lane's four row sums
/// (`12·block` u64), then the reduction's `4·warps·3`.
fn gruen2_shared(block: u32) -> u32 {
    (12 * block + 12 * (block / 32).max(1)) * 8
}

impl DeviceFractionTree {
    /// The two-layer step over layer `layer`'s quarters ([`Gruen2Layer`]), at most `low`
    /// variables left to the host.
    pub fn layer_gruen2(&self, layer: usize, point: &[u64], low: usize) -> Result<Gruen2Layer> {
        let level = self.layer(layer)?;
        Gruen2Layer::new(
            &self.stream,
            &level.p,
            &level.q,
            level.num_vars,
            level.real,
            point,
            low,
        )
    }
}

impl InputLayer {
    /// The input layer's two-layer step, the one that ends GKR for an even-height tree.
    pub fn gruen2(&self, point: &[u64], low: usize) -> Result<Gruen2Layer> {
        Gruen2Layer::new(
            &self.stream,
            &self.p,
            &self.q,
            self.num_vars,
            1usize << self.num_vars,
            point,
            low,
        )
    }
}

/// An input layer's plan on the card (D-BATCH M1-2): each side's range of
/// terms, the terms — where a term's column starts in the table's run of base
/// columns and its shift — their ext3 coefficients, and each side's constant.
/// Uploaded once a table and kept for the layer's rewrite.
pub struct InputPlan {
    side_start: CudaSlice<u32>,
    terms: CudaSlice<u64>,
    coeffs: CudaSlice<u64>,
    constants: CudaSlice<u64>,
    interactions: usize,
}

impl InputPlan {
    /// `side_start` has `2·interactions + 1` entries (side `2i` the
    /// numerator of interaction `i`, `2i + 1` its denominator); `terms` two
    /// u64 a term, `coeffs` three, `constants` three a side.
    pub fn upload(
        stream: &Arc<CudaStream>,
        side_start: &[u32],
        terms: &[u64],
        coeffs: &[u64],
        constants: &[u64],
    ) -> Result<Self> {
        assert!(
            side_start.len() % 2 == 1,
            "two sides an interaction, and the end"
        );
        let interactions = side_start.len() / 2;
        let count = *side_start.last().expect("the end") as usize;
        assert_eq!(terms.len(), 2 * count, "two u64 a term");
        assert_eq!(coeffs.len(), 3 * count, "an ext3 coefficient a term");
        assert_eq!(constants.len(), 6 * interactions, "an ext3 constant a side");
        // An allocation to point at when a table's sides have no terms.
        let or_one = |v: &[u64]| if v.is_empty() { vec![0u64] } else { v.to_vec() };
        Ok(Self {
            side_start: crate::device::htod_or_trim(stream, side_start)?,
            terms: crate::device::htod_or_trim(stream, &or_one(terms))?,
            coeffs: crate::device::htod_or_trim(stream, &or_one(coeffs))?,
            constants: crate::device::htod_or_trim(stream, &or_one(constants))?,
            interactions,
        })
    }

    pub fn interactions(&self) -> usize {
        self.interactions
    }
}

/// The input layer written from the table's base columns in one launch
/// (`gkr_input_from_columns`): `columns` is the table's run, `rows` a column,
/// and the slots are the interactions, padded to a power of two with `0/1`
/// when `padding` — the layout the lifted path writes.
pub fn input_from_columns(
    stream: &Arc<CudaStream>,
    columns: &cudarc::driver::CudaView<'_, u64>,
    rows: usize,
    plan: &InputPlan,
    padding: bool,
) -> Result<Halves> {
    let be = backend()?;
    let interactions = plan.interactions;
    let slots = if padding {
        interactions.next_power_of_two()
    } else {
        interactions
    };
    let cells = slots * rows;
    // SAFETY: the kernel writes every cell of both.
    let mut p = unsafe { crate::device::alloc_or_trim::<u64>(stream, cells * 3) }?;
    let mut q = unsafe { crate::device::alloc_or_trim::<u64>(stream, cells * 3) }?;
    let rows_arg = rows as u64;
    let interactions_arg = interactions as u64;
    let slots_arg = slots as u64;
    unsafe {
        stream
            .launch_builder(&be.gkr_input_from_columns)
            .arg(columns)
            .arg(&rows_arg)
            .arg(&plan.side_start)
            .arg(&plan.terms)
            .arg(&plan.coeffs)
            .arg(&plan.constants)
            .arg(&interactions_arg)
            .arg(&slots_arg)
            .arg(&mut p)
            .arg(&mut q)
            .launch(LaunchConfig {
                grid_dim: ((cells as u64).div_ceil(256).clamp(1, 8192) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    Ok((p, q))
}

/// The two tree levels above an input layer, made by the pass that wrote it.
pub struct TopLevels {
    mid: Option<Halves>,
    out: Halves,
    folds: usize,
}

/// Whether the input layer's writer also makes the two tree levels above it,
/// read once.
pub mod fused_input {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU8, Ordering};

    static FORCED: AtomicU8 = AtomicU8::new(0);
    static FAULT: AtomicU8 = AtomicU8::new(0);

    /// Whether the input layer from the columns is written by
    /// `gkr_input_fold2_from_columns`, which also writes the two levels above
    /// it, so the tree never reads the input layer back. The same cells and
    /// levels. On by default; `LAMBDA_VM_GKR_FUSED_INPUT=0` writes the layer
    /// and folds it as before.
    pub fn enabled() -> bool {
        match FORCED.load(Ordering::Relaxed) {
            1 => return true,
            2 => return false,
            _ => {}
        }
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var("LAMBDA_VM_GKR_FUSED_INPUT").map_or(true, |v| v != "0" && v != "false")
        })
    }

    /// Forces [`enabled`]; `None` back to the environment's word.
    #[doc(hidden)]
    pub fn force(on: Option<bool>) {
        FORCED.store(
            match on {
                None => 0,
                Some(true) => 1,
                Some(false) => 2,
            },
            Ordering::Relaxed,
        );
    }

    /// ⛔ A FAULT, a negative control's only: the upper level's cells off by one.
    #[doc(hidden)]
    pub fn force_fault(on: bool) {
        FAULT.store(u8::from(on), Ordering::Relaxed);
    }

    pub(super) fn fault() -> u32 {
        u32::from(FAULT.load(Ordering::Relaxed))
    }
}

/// [`input_from_columns`] and the first tree fold pass in one launch.
/// `padding` writes its `0/1` input cells too. A compact tree writes only
/// the first even level above its input. A cube of fewer than four cells
/// is refused so the caller can write the input alone.
pub fn input_fold2_from_columns(
    stream: &Arc<CudaStream>,
    columns: &cudarc::driver::CudaView<'_, u64>,
    rows: usize,
    plan: &InputPlan,
    padding: bool,
) -> Result<(Halves, TopLevels)> {
    let be = backend()?;
    let interactions = plan.interactions;
    let slots = interactions.next_power_of_two();
    let cube = slots * rows;
    if cube < 4 || !cube.is_power_of_two() {
        return Err(invalid());
    }
    let stored_slots = if padding { slots } else { interactions };
    let stored = stored_slots * rows;
    let (half, quarter) = (cube / 2, cube / 4);
    let skip = skip_middle();
    let single = u32::from(skip && !cube.trailing_zeros().is_multiple_of(2));
    let folds = if single == 1 { 1 } else { 2 };
    // SAFETY: the kernel writes every stored input cell and every cell of
    // the allocated levels.
    let (mut p, mut q) = unsafe { alloc_halves(stream, stored) }?;
    let mut mid = if skip {
        None
    } else {
        Some(unsafe { alloc_halves(stream, half) }?)
    };
    let (mut p_out, mut q_out) = unsafe { alloc_halves(stream, cube >> folds) }?;
    let null = 0u64;
    let rows_arg = rows as u64;
    let interactions_arg = interactions as u64;
    let stored_arg = stored_slots as u64;
    let cube_arg = cube as u64;
    let fault = fused_input::fault();
    unsafe {
        let mut launch = stream.launch_builder(&be.gkr_input_fold2_from_columns);
        launch
            .arg(columns)
            .arg(&rows_arg)
            .arg(&plan.side_start)
            .arg(&plan.terms)
            .arg(&plan.coeffs)
            .arg(&plan.constants)
            .arg(&interactions_arg)
            .arg(&stored_arg)
            .arg(&cube_arg)
            .arg(&mut p)
            .arg(&mut q);
        match &mut mid {
            Some((p_mid, q_mid)) => launch.arg(p_mid).arg(q_mid),
            None => launch.arg(&null).arg(&null),
        };
        launch
            .arg(&mut p_out)
            .arg(&mut q_out)
            .arg(&fault)
            .arg(&single)
            .launch(LaunchConfig {
                grid_dim: ((quarter as u64).div_ceil(256).clamp(1, 8192) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    Ok((
        (p, q),
        TopLevels {
            mid,
            out: (p_out, q_out),
            folds,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where the padding of an upper half starts, and that it runs to the end.
    fn padding_start(pad: &[bool]) -> usize {
        let start = pad.iter().position(|&p| p).unwrap_or(pad.len());
        assert!(
            pad[start..].iter().all(|&p| p),
            "the padding is one run to the end"
        );
        start
    }

    /// ★ The run a kept layer's upper halves hold is exact: worked out cell
    /// by cell (a folded cell is padding exactly when both cells it folds
    /// are), the padding is one run to the end of the live half after every
    /// fold, starting where [`stored_after_round`] says — so a kernel that
    /// reads the run and the constants past it reads every cell the whole
    /// layer would. And a round reads its cube through the kept kernel
    /// exactly while padding is left in it.
    #[test]
    fn the_stored_run_is_where_the_padding_starts_after_every_fold() {
        for m in 1..=12usize {
            let half = 1usize << m;
            for stored in [1, 2, half / 3 + 1, half / 2, half / 2 + 1, half - 1, half] {
                if stored == 0 || stored > half {
                    continue;
                }
                let mut pad: Vec<bool> = (0..half).map(|i| i >= stored).collect();
                let mut run = stored;
                for j in 0..m {
                    let what = format!("m {m}, stored {stored}, round {j}");
                    let quarter = 1usize << (m - j - 1);
                    // The cube the halves hold coming into round `j`: round
                    // `j > 0` folds the previous round's on the way in.
                    let cube = if j > 0 { 4 * quarter } else { 2 * quarter };
                    assert_eq!(pad.len(), cube, "{what}: the live half");
                    let start = padding_start(&pad);
                    assert_eq!(run.min(cube), start, "{what}: the run held coming in");
                    assert_eq!(
                        run < cube,
                        start < cube,
                        "{what}: the kept kernel exactly while padding is left"
                    );
                    if j > 0 {
                        pad = (0..2 * quarter)
                            .map(|x| pad[x] && pad[x + 2 * quarter])
                            .collect();
                    }
                    run = stored_after_round(run, quarter);
                    assert_eq!(run, padding_start(&pad), "{what}: the run held after it");
                }
            }
        }
    }

    /// The room a kept tree promises: the kept layer and every level above
    /// it — what carrying it costs when there is no padding to leave out.
    #[test]
    fn a_kept_tree_promises_its_layer_and_the_levels_above() {
        let full = 1usize << 20;
        assert_eq!(truncated_peak_bytes(full, full), (4 * full) as u64 * 24);
        // A CPU table: 20 of 32 slots real, 2^21 rows.
        let (rows, slots, interactions) = (1usize << 21, 32usize, 20usize);
        let (real, full) = (interactions * rows, slots * rows);
        assert_eq!(
            truncated_peak_bytes(real, full),
            (2 * real + 2 * full) as u64 * 24
        );
        assert!(truncated_peak_bytes(real, full) < (4 * full) as u64 * 24);
        // It covers the kept layer plus every level above (full − 1 cells a
        // side), and the whole layer written again once the tree is gone.
        let levels = 2 * (full - 1);
        assert!(truncated_peak_bytes(real, full) >= (2 * real + levels) as u64 * 24);
        assert!(truncated_peak_bytes(real, full) >= (2 * full) as u64 * 24);
    }
}
