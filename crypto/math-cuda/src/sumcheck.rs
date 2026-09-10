//! A sumcheck's rounds on device, with the factors resident between them.
//!
//! The protocol is interactive by construction: every round's evaluations are
//! absorbed and a challenge is drawn before the next one, so the host is in the
//! loop once per round. What stays on device is the expensive part — the pass
//! over the cube and the fold that halves it — and what crosses the bus is a
//! handful of field elements per round.
//!
//! The program is `multilinear::program::Program` lowered to the flat node blob
//! `kernels/sumcheck.cu` reads; the lowering lives in
//! `crypto/multilinear/src/gpu.rs`, which is also where the op tags are
//! defined.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

/// Interpolation nodes per round the kernel has accumulator room for. Mirrors
/// `MAX_NODES` in `kernels/sumcheck.cu`.
pub const MAX_NODES: usize = 16;

const BLOCK_DIM: u32 = 256;

/// Scratch ceiling for the per-thread slot file, which is what sets the grid:
/// a wider program buys fewer threads. 512 MiB leaves the factors and the
/// resident codewords room on a 32 GiB device.
const SLOT_BUDGET_BYTES: u64 = 512 * 1024 * 1024;

/// Most threads a round launches, whatever the slot file allows.
const MAX_THREADS: u64 = 1 << 20;

/// One sumcheck's device state: the factors, the program, and the scratch the
/// rounds reuse.
pub struct SumcheckSession {
    stream: Arc<CudaStream>,
    /// Factor `k` at cube index `j`, component `c`: `[(k*stride + j)*3 + c]`.
    factors: CudaSlice<u64>,
    stride: usize,
    width: usize,
    /// Cube indices left. Halves with every fold.
    len: usize,
    nodes: CudaSlice<u64>,
    num_nodes: usize,
    consts: CudaSlice<u64>,
    root_slot: u32,
    slots: CudaSlice<u64>,
    partials: CudaSlice<u64>,
    grid: u32,
}

impl SumcheckSession {
    /// Uploads `factors` (each `2^num_vars` ext3 values, interleaved as three
    /// u64 per element) and the lowered program.
    ///
    /// `nodes` is two u64 per step, `consts` three u64 per constant, and
    /// `num_slots` the program's max live set — see the lowering.
    pub fn new(
        factors: &[&[u64]],
        nodes: &[u64],
        consts: &[u64],
        num_slots: usize,
        root_slot: u32,
    ) -> Result<Self> {
        assert!(!factors.is_empty(), "a sumcheck needs a factor");
        assert!(nodes.len().is_multiple_of(2), "two u64 per step");
        assert!(
            consts.len().is_multiple_of(3),
            "three u64 per ext3 constant"
        );
        let stride = factors[0].len() / 3;
        assert!(stride.is_power_of_two(), "the cube is a power of two");
        assert!(
            factors.iter().all(|f| f.len() == stride * 3),
            "every factor spans the same cube"
        );
        assert!(num_slots > 0, "a program writes at least one slot");

        let be = backend()?;
        let stream = be.next_stream();

        let width = factors.len();
        let mut buffer = unsafe { stream.alloc::<u64>(width * stride * 3) }?;
        for (k, factor) in factors.iter().enumerate() {
            let at = k * stride * 3;
            let mut slab = buffer.slice_mut(at..at + factor.len());
            stream.memcpy_htod(*factor, &mut slab)?;
        }

        // The slot file is per thread, so it is the grid that gives way.
        let per_thread = num_slots as u64 * 3 * 8;
        let threads = (SLOT_BUDGET_BYTES / per_thread)
            .min(MAX_THREADS)
            .max(BLOCK_DIM as u64);
        let grid = ((threads / BLOCK_DIM as u64) as u32).max(1);
        let num_threads = grid as u64 * BLOCK_DIM as u64;

        let nodes_dev = stream.clone_htod(nodes)?;
        // A program with no constants still needs an allocation to point at.
        let consts_dev = stream.clone_htod(if consts.is_empty() {
            &[0u64][..]
        } else {
            consts
        })?;
        let slots = unsafe { stream.alloc::<u64>(num_slots * 3 * num_threads as usize) }?;
        let partials = stream.alloc_zeros::<u64>(MAX_NODES * grid as usize * 3)?;

        Ok(Self {
            stream,
            factors: buffer,
            stride,
            width,
            len: stride,
            nodes: nodes_dev,
            num_nodes: nodes.len() / 2,
            consts: consts_dev,
            root_slot,
            slots,
            partials,
            grid,
        })
    }

    /// Bytes this session holds on device, for admission control.
    pub fn device_bytes(factors: usize, cube: usize, num_slots: usize) -> u64 {
        let per_thread = num_slots as u64 * 3 * 8;
        let threads = (SLOT_BUDGET_BYTES / per_thread.max(1))
            .min(MAX_THREADS)
            .max(BLOCK_DIM as u64);
        factors as u64 * cube as u64 * 24 + threads * per_thread
    }

    /// Cube indices left to bind.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// One round: the program's sum over the cube at each interpolation node.
    ///
    /// `t` holds the nodes as ext3 (three u64 each). Returns one ext3 sum per
    /// node, as `t.len()/3` triples.
    pub fn round(&mut self, t: &[u64]) -> Result<Vec<u64>> {
        assert!(t.len().is_multiple_of(3), "three u64 per ext3 node");
        let num_t = t.len() / 3;
        assert!(num_t > 0 && num_t <= MAX_NODES, "nodes per round");
        assert!(self.len >= 2, "a round needs a variable to bind");

        let be = backend()?;
        let half = (self.len / 2) as u64;
        let t_dev = self.stream.clone_htod(t)?;
        let cfg = LaunchConfig {
            grid_dim: (self.grid, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            // One ext3 accumulator per thread, reduced one node at a time.
            shared_mem_bytes: BLOCK_DIM * 3 * 8,
        };
        let stride = self.stride as u64;
        let num_nodes = self.num_nodes as u64;
        let num_t_u32 = num_t as u32;
        unsafe {
            self.stream
                .launch_builder(&be.sumcheck_round_ext3)
                .arg(&self.factors)
                .arg(&stride)
                .arg(&half)
                .arg(&self.nodes)
                .arg(&num_nodes)
                .arg(&self.consts)
                .arg(&self.root_slot)
                .arg(&t_dev)
                .arg(&num_t_u32)
                .arg(&mut self.slots)
                .arg(&mut self.partials)
                .launch(cfg)?;
        }

        // The per-block partials come back and are summed here: it is a few
        // kilobytes against the cube the kernel just walked, and the round
        // cannot proceed without the host anyway.
        let used = num_t * self.grid as usize * 3;
        let partials = self.stream.clone_dtoh(&self.partials.slice(0..used))?;
        self.stream.synchronize()?;
        Ok(sum_partials(&partials, num_t, self.grid as usize))
    }

    /// Binds the round's variable to `r` (ext3, three u64) in every factor.
    pub fn fold(&mut self, r: &[u64]) -> Result<()> {
        assert_eq!(r.len(), 3, "an ext3 challenge");
        assert!(self.len >= 2, "a fold needs a variable to bind");
        let be = backend()?;
        let half = (self.len / 2) as u64;
        let r_dev = self.stream.clone_htod(r)?;
        let total = self.width as u64 * half;
        let grid = total.div_ceil(BLOCK_DIM as u64).min(4096) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid.max(1), 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        let stride = self.stride as u64;
        let width = self.width as u64;
        unsafe {
            self.stream
                .launch_builder(&be.sumcheck_fold_ext3)
                .arg(&mut self.factors)
                .arg(&stride)
                .arg(&half)
                .arg(&width)
                .arg(&r_dev)
                .launch(cfg)?;
        }
        self.len /= 2;
        Ok(())
    }

    /// Every factor's remaining values, interleaved as three u64 per element —
    /// what the host needs to carry on where the device stopped.
    pub fn download(&self) -> Result<Vec<Vec<u64>>> {
        let mut out = Vec::with_capacity(self.width);
        for k in 0..self.width {
            let at = k * self.stride * 3;
            out.push(
                self.stream
                    .clone_dtoh(&self.factors.slice(at..at + self.len * 3))?,
            );
        }
        self.stream.synchronize()?;
        Ok(out)
    }
}

/// Sums the per-block partials of each interpolation node.
///
/// Goldilocks addition here mirrors the kernel's: the same EPSILON-corrected
/// wrap, so a host sum and a device sum of the same values agree as field
/// elements (they need not agree bit for bit, and nothing looks).
fn sum_partials(partials: &[u64], num_t: usize, blocks: usize) -> Vec<u64> {
    let mut out = vec![0u64; num_t * 3];
    for ti in 0..num_t {
        let mut acc = [0u64; 3];
        for b in 0..blocks {
            let at = (ti * blocks + b) * 3;
            for c in 0..3 {
                acc[c] = goldilocks_add(acc[c], partials[at + c]);
            }
        }
        out[ti * 3..ti * 3 + 3].copy_from_slice(&acc);
    }
    out
}

/// `a + b` in Goldilocks, on the raw non-canonical representation both sides
/// use.
fn goldilocks_add(a: u64, b: u64) -> u64 {
    const EPSILON: u64 = 0xFFFF_FFFF;
    let (sum, over) = a.overflowing_add(b);
    let (sum, over) = sum.overflowing_add(if over { EPSILON } else { 0 });
    if over { sum + EPSILON } else { sum }
}
