//! A LogUp fraction tree resident on device, and the sumchecks over its
//! layers.
//!
//! The tree is the biggest structure a table's argument holds — its input
//! layer spans `interactions × rows` fractions — and GKR walks every level of
//! it. Building it here and leaving it here means the layers are never copied:
//! a layer's halves are the sumcheck's factors in place, and the round that
//! binds them is the same kernel the batched statements use.
//!
//! Each layer is visited exactly once, top down, so the rounds fold it where
//! it lies and nothing reads it again.

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
}

/// The whole tree, `layers[0]` the output fraction and layer `layers.len()`
/// the input — the order `multilinear::gkr::FractionTree` uses.
pub struct DeviceFractionTree {
    stream: Arc<CudaStream>,
    /// Every level above the input layer.
    layers: Vec<DeviceLayer>,
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
        assert_eq!(p.len(), q.len(), "a layer's halves span one cube");
        assert!(p.len().is_multiple_of(3), "three u64 per ext3 element");
        let elements = p.len() / 3;
        assert!(elements.is_power_of_two(), "the cube is a power of two");

        let be = backend()?;
        // The levels above the input layer halve, so the whole tree is twice
        // it — and the input layer is `p` and `q` together.
        let Some(room) = be.reserve(p.len() as u64 * 8 * 4) else {
            crate::device::note_device_fallback();
            return Err(cudarc::driver::DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
            ));
        };
        crate::argue_probe::note_device(crate::argue_probe::Surface::Gkr, p.len() as u64 * 8 * 4);
        let input = DeviceLayer {
            p: Arc::new(p),
            q: Arc::new(q),
            num_vars: elements.trailing_zeros() as usize,
        };
        let layers = fold_upwards(&stream, be, &input.p, &input.q, elements, elements)?;
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
            fold_upwards(&stream, be, &p, &q, real, full)?
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

    /// Every level plus the input layer, whether or not this tree holds it.
    pub fn num_layers(&self) -> usize {
        self.layers.len() + 1
    }

    pub fn layer_num_vars(&self, layer: usize) -> usize {
        self.layers
            .get(layer)
            .map_or(self.input_num_vars, |level| level.num_vars)
    }

    /// Whether [`layer_sumcheck`](Self::layer_sumcheck) can still prove the
    /// input layer, or the caller has to hand it back first.
    pub fn holds_input(&self) -> bool {
        self.input.is_some()
    }

    /// The output fraction `(p, q)`, the one the bus balance is read off.
    pub fn output(&self) -> Result<([u64; 3], [u64; 3])> {
        let top = self
            .layers
            .first()
            .or(self.input.as_ref())
            .expect("a tree has a top");
        let p = self.stream.clone_dtoh(&top.p.slice(0..3))?;
        let q = self.stream.clone_dtoh(&top.q.slice(0..3))?;
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
        let level = self.layers.get(layer).ok_or(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
        ))?;
        let p = self.stream.clone_dtoh(&level.p.slice(0..level.p.len()))?;
        let q = self.stream.clone_dtoh(&level.q.slice(0..level.q.len()))?;
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
        let level = match self.layers.get(layer) {
            Some(level) => level,
            // The input layer, which this tree only has if it kept it.
            None => self.input.as_ref().ok_or(cudarc::driver::DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
            ))?,
        };
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

/// Every level above an input layer of `full` fractions, of which the first
/// `real` are stored — the rest are the padding's `0/1` and the first fold
/// reads them as that. Returned with the output first, the way the tree
/// indexes them.
fn fold_upwards(
    stream: &Arc<CudaStream>,
    be: &crate::device::Backend,
    p: &Arc<CudaSlice<u64>>,
    q: &Arc<CudaSlice<u64>>,
    real: usize,
    full: usize,
) -> Result<Vec<DeviceLayer>> {
    let mut layers: Vec<DeviceLayer> = Vec::new();
    let mut num_vars = full.trailing_zeros() as usize;
    while num_vars > 0 {
        let half = (1usize << num_vars) / 2;
        // SAFETY: the kernel writes every element of the level it produces.
        let mut p_out = unsafe { crate::device::alloc_or_trim::<u64>(stream, half * 3) }?;
        let mut q_out = unsafe { crate::device::alloc_or_trim::<u64>(stream, half * 3) }?;
        let (below_p, below_q) = match layers.last() {
            Some(level) => (level.p.clone(), level.q.clone()),
            None => (p.clone(), q.clone()),
        };
        let half_arg = half as u64;
        let padded = layers.is_empty() && real != full;
        unsafe {
            if padded {
                let real_arg = real as u64;
                stream
                    .launch_builder(&be.fraction_fold_padded_ext3)
                    .arg(below_p.as_ref())
                    .arg(below_q.as_ref())
                    .arg(&half_arg)
                    .arg(&real_arg)
                    .arg(&mut p_out)
                    .arg(&mut q_out)
                    .launch(LaunchConfig::for_num_elems(half as u32))?;
            } else {
                stream
                    .launch_builder(&be.fraction_fold_ext3)
                    .arg(below_p.as_ref())
                    .arg(below_q.as_ref())
                    .arg(&half_arg)
                    .arg(&mut p_out)
                    .arg(&mut q_out)
                    .launch(LaunchConfig::for_num_elems(half as u32))?;
            }
        }
        num_vars -= 1;
        layers.push(DeviceLayer {
            p: Arc::new(p_out),
            q: Arc::new(q_out),
            num_vars,
        });
    }
    layers.reverse();
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

/// What a tree built by [`DeviceFractionTree::from_padded_input`] holds at
/// once: the truncated input layer plus the level above it while the first
/// fold runs, against every level once the input is gone — and, later, the
/// full input layer handed back for its own sumcheck, by which time the levels
/// are spent.
pub fn padded_peak_bytes(real: usize, full: usize) -> u64 {
    ((2 * real + full).max(2 * full)) as u64 * 24
}
