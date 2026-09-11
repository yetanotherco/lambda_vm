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

/// The whole tree, `layers[0]` the output fraction and the last the input
/// layer — the order `multilinear::gkr::FractionTree` uses.
pub struct DeviceFractionTree {
    stream: Arc<CudaStream>,
    layers: Vec<DeviceLayer>,
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
        let input_p = stream.clone_htod(p)?;
        let input_q = stream.clone_htod(q)?;
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
        let mut layers = vec![DeviceLayer {
            p: Arc::new(p),
            q: Arc::new(q),
            num_vars: elements.trailing_zeros() as usize,
        }];

        while layers.last().expect("non-empty").num_vars > 0 {
            let below = layers.last().expect("non-empty");
            let half = (1usize << below.num_vars) / 2;
            // SAFETY: the kernel writes every element of the level it produces.
            let mut p_out = unsafe { stream.alloc::<u64>(half * 3) }?;
            let mut q_out = unsafe { stream.alloc::<u64>(half * 3) }?;
            let half_arg = half as u64;
            unsafe {
                stream
                    .launch_builder(&be.fraction_fold_ext3)
                    .arg(below.p.as_ref())
                    .arg(below.q.as_ref())
                    .arg(&half_arg)
                    .arg(&mut p_out)
                    .arg(&mut q_out)
                    .launch(LaunchConfig::for_num_elems(half as u32))?;
            }
            let num_vars = below.num_vars - 1;
            layers.push(DeviceLayer {
                p: Arc::new(p_out),
                q: Arc::new(q_out),
                num_vars,
            });
        }

        layers.reverse();
        Ok(Self { stream, layers })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn layer_num_vars(&self, layer: usize) -> usize {
        self.layers[layer].num_vars
    }

    /// The output fraction `(p, q)`, the one the bus balance is read off.
    pub fn output(&self) -> Result<([u64; 3], [u64; 3])> {
        let top = &self.layers[0];
        let p = self.stream.clone_dtoh(&top.p.slice(0..3))?;
        let q = self.stream.clone_dtoh(&top.q.slice(0..3))?;
        self.stream.synchronize()?;
        Ok(([p[0], p[1], p[2]], [q[0], q[1], q[2]]))
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
        let DeviceLayer { p, q, num_vars } = &self.layers[layer];
        assert!(*num_vars > 0, "the output layer has nothing to bind");
        let half_vars = num_vars - 1;
        assert_eq!(point.len(), half_vars * 3, "the point spans the halves");

        let half = 1usize << half_vars;
        let eq = Arc::new(self.eq_table(point, half)?);
        let addresses = {
            let (eq_at, _eq_guard) = eq.device_ptr(&self.stream);
            let (p_at, _p_guard) = p.device_ptr(&self.stream);
            let (q_at, _q_guard) = q.device_ptr(&self.stream);
            let stride = (half * 3 * 8) as u64;
            vec![eq_at, p_at, p_at + stride, q_at, q_at + stride]
        };

        SumcheckSession::from_device(
            self.stream.clone(),
            &addresses,
            half,
            vec![eq, p.clone(), q.clone()],
            nodes,
            consts,
            num_slots,
            root_slot,
        )
    }

    /// `eq(point, ·)` over `half` cells, doubled a variable at a time.
    ///
    /// Variables go in back to front, which is what leaves variable 0 in the
    /// high bit — the indexing every table here folds on.
    fn eq_table(&self, point: &[u64], half: usize) -> Result<CudaSlice<u64>> {
        let be = backend()?;
        let mut table = self.stream.alloc_zeros::<u64>(half * 3)?;
        // The seed is one, and the levels scale it into the whole table.
        let one = [1u64, 0, 0];
        {
            let mut head = table.slice_mut(0..3);
            self.stream.memcpy_htod(&one, &mut head)?;
        }
        for (level, coordinate) in point.chunks_exact(3).rev().enumerate() {
            let r = self.stream.clone_htod(coordinate)?;
            let filled = 1u64 << level;
            unsafe {
                self.stream
                    .launch_builder(&be.eq_expand_level_ext3)
                    .arg(&mut table)
                    .arg(&filled)
                    .arg(&r)
                    .launch(LaunchConfig::for_num_elems(filled as u32))?;
            }
        }
        Ok(table)
    }
}
