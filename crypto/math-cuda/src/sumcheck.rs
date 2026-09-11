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

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

/// Interpolation nodes per round the kernel has accumulator room for. Mirrors
/// `MAX_NODES` in `kernels/sumcheck.cu`.
pub const MAX_NODES: usize = 16;

const BLOCK_DIM: u32 = 256;

/// Scratch ceiling for the per-thread slot file, which is what caps the grid:
/// a wider program buys fewer threads. 512 MiB leaves the factors and the
/// resident codewords room on a 32 GiB device.
const SLOT_BUDGET_BYTES: u64 = 512 * 1024 * 1024;

/// Most threads a round launches, whatever the slot file allows.
const MAX_THREADS: u64 = 1 << 20;

/// One sumcheck's device state: the factors, the program, and the scratch the
/// rounds reuse.
///
/// The factors reach the kernels as a table of device pointers, so they need
/// not be one allocation: a batch uploads them together, while a GKR layer's
/// are halves of buffers the fraction tree already holds.
pub struct SumcheckSession {
    stream: Arc<CudaStream>,
    /// One device address per factor, and the same list on the host so a
    /// bound factor can be read back without owning its buffer.
    factor_ptrs: CudaSlice<u64>,
    addresses: Vec<u64>,
    /// The buffers the factors point into, kept alive for the session's
    /// lifetime. A session built by uploading owns one; a session over a
    /// fraction tree's layer shares the tree's.
    held: Vec<Arc<CudaSlice<u64>>>,
    /// True when `held[0]` is this session's own upload, laid out as `width`
    /// slabs of `stride` — the only shape [`download`](Self::download) knows.
    uploaded: bool,
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
    /// The session's widest launch, set by its first round.
    max_grid: u32,
    /// The interpolation nodes as the device last saw them, and the buffer
    /// they live in. They are the same every round of a sumcheck, so the send
    /// happens once and the comparison is what decides that.
    t_dev: CudaSlice<u64>,
    t_host: Vec<u64>,
    /// Three u64 of scratch for the round's challenge, written in place rather
    /// than allocated per fold.
    r_dev: CudaSlice<u64>,
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
        // The base address plus the slab offsets, which stay inside the
        // allocation by construction. The guard orders the read on `stream`
        // and is dropped here — every later use of the pointers is on the same
        // stream, which is what orders them.
        let addresses: Vec<u64> = {
            let (base, _record) = buffer.device_ptr(&stream);
            (0..width)
                .map(|k| base + (k * stride * 3 * 8) as u64)
                .collect()
        };
        let factor_ptrs = stream.clone_htod(&addresses)?;

        let grid = grid_for_work(grid_ceiling(num_slots), (stride / 2) as u64);
        let num_threads = grid as u64 * BLOCK_DIM as u64;

        let nodes_dev = stream.clone_htod(nodes)?;
        // A program with no constants still needs an allocation to point at.
        let consts_dev = stream.clone_htod(if consts.is_empty() {
            &[0u64][..]
        } else {
            consts
        })?;
        let slots = unsafe { stream.alloc::<u64>(num_slots * 3 * num_threads as usize) }?;
        // Every partial the round reads is one the round wrote.
        let partials = unsafe { stream.alloc::<u64>(MAX_NODES * grid as usize * 3) }?;
        let t_dev = stream.alloc_zeros::<u64>(MAX_NODES * 3)?;
        let r_dev = stream.alloc_zeros::<u64>(3)?;

        Ok(Self {
            stream,
            factor_ptrs,
            addresses,
            held: vec![Arc::new(buffer)],
            uploaded: true,
            stride,
            width,
            len: stride,
            nodes: nodes_dev,
            num_nodes: nodes.len() / 2,
            consts: consts_dev,
            root_slot,
            slots,
            partials,
            max_grid: grid,
            t_dev,
            t_host: Vec::new(),
            r_dev,
        })
    }

    /// A session over factors that already live on device.
    ///
    /// `addresses` is one device pointer per factor, `len` the cube they span,
    /// and `held` the allocations they point into — kept alive here, because
    /// the kernels only see addresses. The rounds fold those buffers in place.
    #[allow(clippy::too_many_arguments)]
    pub fn from_device(
        stream: Arc<CudaStream>,
        addresses: &[u64],
        len: usize,
        held: Vec<Arc<CudaSlice<u64>>>,
        nodes: &[u64],
        consts: &[u64],
        num_slots: usize,
        root_slot: u32,
    ) -> Result<Self> {
        assert!(!addresses.is_empty(), "a sumcheck needs a factor");
        assert!(len.is_power_of_two(), "the cube is a power of two");
        assert!(nodes.len().is_multiple_of(2), "two u64 per step");
        assert!(
            consts.len().is_multiple_of(3),
            "three u64 per ext3 constant"
        );
        assert!(num_slots > 0, "a program writes at least one slot");

        let be = backend()?;
        let width = addresses.len();
        let grid = grid_for_work(grid_ceiling(num_slots), (len / 2) as u64);
        let num_threads = grid as u64 * BLOCK_DIM as u64;
        let factor_ptrs = stream.clone_htod(addresses)?;
        let nodes_dev = stream.clone_htod(nodes)?;
        let consts_dev = stream.clone_htod(if consts.is_empty() {
            &[0u64][..]
        } else {
            consts
        })?;
        let slots = unsafe { stream.alloc::<u64>(num_slots * 3 * num_threads as usize) }?;
        let partials = unsafe { stream.alloc::<u64>(MAX_NODES * grid as usize * 3) }?;
        let t_dev = stream.alloc_zeros::<u64>(MAX_NODES * 3)?;
        let r_dev = stream.alloc_zeros::<u64>(3)?;
        let _ = be;

        Ok(Self {
            stream,
            factor_ptrs,
            addresses: addresses.to_vec(),
            held,
            uploaded: false,
            stride: len,
            width,
            len,
            nodes: nodes_dev,
            num_nodes: nodes.len() / 2,
            consts: consts_dev,
            root_slot,
            slots,
            partials,
            max_grid: grid,
            t_dev,
            t_host: Vec::new(),
            r_dev,
        })
    }

    /// Bytes this session holds on device, for admission control.
    pub fn device_bytes(factors: usize, cube: usize, num_slots: usize) -> u64 {
        let per_thread = num_slots as u64 * 3 * 8;
        let grid = grid_for_work(grid_ceiling(num_slots), (cube / 2) as u64);
        let threads = grid as u64 * BLOCK_DIM as u64;
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
        // The grid follows the cube down: a block past the indices left writes
        // a partial with nothing in it, and that partial is what comes back.
        let grid = grid_for_work(self.max_grid, half);
        if self.t_host != t {
            let mut head = self.t_dev.slice_mut(0..t.len());
            self.stream.memcpy_htod(t, &mut head)?;
            self.t_host.clear();
            self.t_host.extend_from_slice(t);
        }
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            // One ext3 accumulator per thread, reduced one node at a time.
            shared_mem_bytes: BLOCK_DIM * 3 * 8,
        };
        let num_nodes = self.num_nodes as u64;
        let num_t_u32 = num_t as u32;
        unsafe {
            self.stream
                .launch_builder(&be.sumcheck_round_ext3)
                .arg(&self.factor_ptrs)
                .arg(&half)
                .arg(&self.nodes)
                .arg(&num_nodes)
                .arg(&self.consts)
                .arg(&self.root_slot)
                .arg(&self.t_dev)
                .arg(&num_t_u32)
                .arg(&mut self.slots)
                .arg(&mut self.partials)
                .launch(cfg)?;
        }

        // The per-block partials come back and are summed here: it is a few
        // kilobytes against the cube the kernel just walked, and the round
        // cannot proceed without the host anyway.
        let used = num_t * grid as usize * 3;
        let partials = self.stream.clone_dtoh(&self.partials.slice(0..used))?;
        self.stream.synchronize()?;
        Ok(sum_partials(&partials, num_t, grid as usize))
    }

    /// Binds the round's variable to `r` (ext3, three u64) in every factor.
    pub fn fold(&mut self, r: &[u64]) -> Result<()> {
        assert_eq!(r.len(), 3, "an ext3 challenge");
        assert!(self.len >= 2, "a fold needs a variable to bind");
        let be = backend()?;
        let half = (self.len / 2) as u64;
        self.stream.memcpy_htod(r, &mut self.r_dev)?;
        let total = self.width as u64 * half;
        let grid = total.div_ceil(BLOCK_DIM as u64).min(4096) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid.max(1), 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        let width = self.width as u64;
        unsafe {
            self.stream
                .launch_builder(&be.sumcheck_fold_ext3)
                .arg(&mut self.factor_ptrs)
                .arg(&half)
                .arg(&width)
                .arg(&self.r_dev)
                .launch(cfg)?;
        }
        self.len /= 2;
        Ok(())
    }

    /// Whether [`download`](Self::download) can read the factors back: only a
    /// session that uploaded them knows their layout.
    pub fn can_download(&self) -> bool {
        self.uploaded
    }

    /// What each factor has been bound to, once every variable is gone.
    ///
    /// Reads the factors where they lie rather than through their buffers: a
    /// session over a fraction tree's layer does not own them, and three u64
    /// per factor is not worth a view for.
    pub fn bound_values(&self) -> Result<Vec<[u64; 3]>> {
        assert_eq!(self.len, 1, "a factor is bound once every variable is");
        self.stream.synchronize()?;
        let mut out = Vec::with_capacity(self.addresses.len());
        for address in &self.addresses {
            let mut value = [0u64; 3];
            // SAFETY: the address is a factor's base, which holds at least one
            // ext3 element, and the stream is idle (synchronized above).
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    value.as_mut_ptr() as *mut core::ffi::c_void,
                    *address,
                    24,
                )
                .result()?;
            }
            out.push(value);
        }
        Ok(out)
    }

    /// Every factor's remaining values, interleaved as three u64 per element —
    /// what the host needs to carry on where the device stopped.
    pub fn download(&self) -> Result<Vec<Vec<u64>>> {
        assert!(
            self.uploaded,
            "only a session that uploaded its factors knows their layout"
        );
        let factors = &self.held[0];
        let mut out = Vec::with_capacity(self.width);
        for k in 0..self.width {
            let at = k * self.stride * 3;
            out.push(
                self.stream
                    .clone_dtoh(&factors.slice(at..at + self.len * 3))?,
            );
        }
        self.stream.synchronize()?;
        Ok(out)
    }
}

/// The most blocks a program of `num_slots` live values may launch: the slot
/// file is per thread, so it is the grid that gives way to a wider program.
///
/// This is a ceiling, not a shape — what a launch actually takes is
/// [`grid_for_work`], because a grid past the indices it walks costs a partial
/// per idle block and a reduction with nothing in it.
fn grid_ceiling(num_slots: usize) -> u32 {
    let per_thread = num_slots as u64 * 3 * 8;
    let threads = (SLOT_BUDGET_BYTES / per_thread.max(1))
        .min(MAX_THREADS)
        .max(BLOCK_DIM as u64);
    ((threads / BLOCK_DIM as u64) as u32).max(1)
}

/// The blocks `work` indices need, never past `ceiling`.
fn grid_for_work(ceiling: u32, work: u64) -> u32 {
    work.div_ceil(BLOCK_DIM as u64).clamp(1, ceiling as u64) as u32
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

/// A multilinear's value at `point`, bound one variable at a time on device.
///
/// `table` is `2^point_len` ext3 values interleaved, `point` three u64 per
/// coordinate. Mirrors `Mle::evaluate_at`: the same folds in the same order,
/// with the table halving under them.
pub fn evaluate_mle_ext3(table: &[u64], point: &[u64]) -> Result<[u64; 3]> {
    assert!(point.len().is_multiple_of(3), "three u64 per coordinate");
    let vars = point.len() / 3;
    assert_eq!(
        table.len(),
        (1usize << vars) * 3,
        "the table spans the point"
    );
    assert!(vars > 0, "a point with no coordinates is the table itself");

    let be = backend()?;
    let stream = be.next_stream();
    let mut values = stream.clone_htod(table)?;
    let stride = 1u64 << vars;
    fold_to_one(&stream, be, &mut values, stride, point)?;
    let out = stream.clone_dtoh(&values.slice(0..3))?;
    stream.synchronize()?;
    Ok([out[0], out[1], out[2]])
}

/// The same for a base-field table at a point in the extension: the first fold
/// lifts, the rest stay up.
pub fn evaluate_mle_base(table: &[u64], point: &[u64]) -> Result<[u64; 3]> {
    assert!(point.len().is_multiple_of(3), "three u64 per coordinate");
    let vars = point.len() / 3;
    assert_eq!(table.len(), 1usize << vars, "the table spans the point");
    assert!(vars > 0, "a point with no coordinates is the table itself");

    let be = backend()?;
    let stream = be.next_stream();
    let base = stream.clone_htod(table)?;
    let half = table.len() / 2;
    let r = stream.clone_htod(&point[..3])?;
    // SAFETY: the kernel writes every element of the half it produces.
    let mut values = unsafe { stream.alloc::<u64>(half * 3) }?;
    let half_arg = half as u64;
    unsafe {
        stream
            .launch_builder(&be.mle_fold_base_ext3)
            .arg(&base)
            .arg(&half_arg)
            .arg(&r)
            .arg(&mut values)
            .launch(LaunchConfig::for_num_elems(half as u32))?;
    }
    drop(base);
    fold_to_one(&stream, be, &mut values, half as u64, &point[3..])?;
    let out = stream.clone_dtoh(&values.slice(0..3))?;
    stream.synchronize()?;
    Ok([out[0], out[1], out[2]])
}

/// Binds every coordinate of `point` in a resident ext3 table of `len`
/// elements, leaving the value in its first slot.
fn fold_to_one(
    stream: &Arc<CudaStream>,
    be: &crate::device::Backend,
    values: &mut CudaSlice<u64>,
    len: u64,
    point: &[u64],
) -> Result<()> {
    // The fold kernel takes a table of factors; here there is one, and it
    // folds in place, so its address holds for every level.
    let address = {
        let (base, _record) = values.device_ptr(stream);
        [base]
    };
    let mut table = stream.clone_htod(&address)?;

    let point_dev = stream.clone_htod(point)?;
    let width = 1u64;
    let mut half = len / 2;
    for at in (0..point.len()).step_by(3) {
        let r = point_dev.slice(at..at + 3);
        let cfg = LaunchConfig::for_num_elems((half * width) as u32);
        unsafe {
            stream
                .launch_builder(&be.sumcheck_fold_ext3)
                .arg(&mut table)
                .arg(&half)
                .arg(&width)
                .arg(&r)
                .launch(cfg)?;
        }
        half /= 2;
    }
    Ok(())
}

/// A table's factors, uploaded once and read by everything that walks them:
/// the LogUp input layer, and then the batched sumcheck.
pub struct DeviceFactors {
    stream: Arc<CudaStream>,
    buffer: Arc<CudaSlice<u64>>,
    addresses: Vec<u64>,
    /// The same list on device, where every program that walks these factors
    /// reads it — one send, not one per interaction.
    factor_ptrs: CudaSlice<u64>,
    len: usize,
}

impl DeviceFactors {
    /// Uploads `factors`, each `len` ext3 values interleaved.
    pub fn upload(factors: &[&[u64]]) -> Result<Self> {
        assert!(!factors.is_empty(), "a table has factors");
        let span = factors[0].len();
        assert!(span.is_multiple_of(3), "three u64 per ext3 element");
        assert!(
            factors.iter().all(|f| f.len() == span),
            "every factor spans the same cube"
        );
        let len = span / 3;

        let be = backend()?;
        let stream = be.next_stream();
        let mut buffer = unsafe { stream.alloc::<u64>(factors.len() * span) }?;
        for (k, factor) in factors.iter().enumerate() {
            let at = k * span;
            let mut slab = buffer.slice_mut(at..at + span);
            stream.memcpy_htod(*factor, &mut slab)?;
        }
        let addresses = {
            let (base, _record) = buffer.device_ptr(&stream);
            (0..factors.len())
                .map(|k| base + (k * span * 8) as u64)
                .collect()
        };
        let factor_ptrs = stream.clone_htod(&addresses)?;
        Ok(Self {
            stream,
            buffer: Arc::new(buffer),
            addresses,
            factor_ptrs,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn width(&self) -> usize {
        self.addresses.len()
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Writes `program`'s value at every row into a fresh buffer.
    ///
    /// This is how a bus interaction's two sides become an input-layer slab:
    /// the affine expression is the program, and the rows are the cube.
    pub fn map_program(
        &self,
        nodes: &[u64],
        consts: &[u64],
        num_slots: usize,
        root_slot: u32,
        out: &mut CudaSlice<u64>,
        offset: usize,
    ) -> Result<()> {
        assert!(nodes.len().is_multiple_of(2), "two u64 per step");
        assert!(
            consts.len().is_multiple_of(3),
            "three u64 per ext3 constant"
        );
        assert!(
            out.len() >= (offset + self.len) * 3,
            "the output holds the rows"
        );

        let be = backend()?;
        let grid = grid_for_work(grid_ceiling(num_slots), self.len as u64);
        let num_threads = grid as u64 * BLOCK_DIM as u64;
        let nodes_dev = self.stream.clone_htod(nodes)?;
        let consts_dev = self.stream.clone_htod(if consts.is_empty() {
            &[0u64][..]
        } else {
            consts
        })?;
        let mut slots = unsafe {
            self.stream
                .alloc::<u64>(num_slots * 3 * num_threads as usize)
        }?;
        let mut target = out.slice_mut(offset * 3..(offset + self.len) * 3);
        let rows = self.len as u64;
        let num_nodes = (nodes.len() / 2) as u64;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&be.program_map_ext3)
                .arg(&self.factor_ptrs)
                .arg(&rows)
                .arg(&nodes_dev)
                .arg(&num_nodes)
                .arg(&consts_dev)
                .arg(&root_slot)
                .arg(&mut slots)
                .arg(&mut target)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// A sumcheck over these factors followed by `extra`, which is uploaded
    /// here — the weight tables a batch adds on top of the trace's.
    ///
    /// The rounds fold the factors in place, so the handle is spent for
    /// everything else once this runs.
    #[allow(clippy::too_many_arguments)]
    pub fn session(
        &self,
        extra: &[&[u64]],
        nodes: &[u64],
        consts: &[u64],
        num_slots: usize,
        root_slot: u32,
    ) -> Result<SumcheckSession> {
        let span = self.len * 3;
        let mut addresses = self.addresses.clone();
        let mut held = vec![self.buffer.clone()];
        if !extra.is_empty() {
            assert!(
                extra.iter().all(|f| f.len() == span),
                "every factor spans the same cube"
            );
            let mut buffer = unsafe { self.stream.alloc::<u64>(extra.len() * span) }?;
            for (k, factor) in extra.iter().enumerate() {
                let at = k * span;
                let mut slab = buffer.slice_mut(at..at + span);
                self.stream.memcpy_htod(*factor, &mut slab)?;
            }
            {
                let (base, _record) = buffer.device_ptr(&self.stream);
                addresses.extend((0..extra.len()).map(|k| base + (k * span * 8) as u64));
            }
            held.push(Arc::new(buffer));
        }
        SumcheckSession::from_device(
            self.stream.clone(),
            &addresses,
            self.len,
            held,
            nodes,
            consts,
            num_slots,
            root_slot,
        )
    }
}

/// Fills `count` ext3 cells of `dst` from `offset` with one value.
pub fn fill_ext3(
    stream: &Arc<CudaStream>,
    dst: &mut CudaSlice<u64>,
    offset: usize,
    count: usize,
    value: &[u64],
) -> Result<()> {
    assert_eq!(value.len(), 3, "an ext3 value");
    if count == 0 {
        return Ok(());
    }
    let be = backend()?;
    let value_dev = stream.clone_htod(value)?;
    let mut target = dst.slice_mut(offset * 3..(offset + count) * 3);
    let count_arg = count as u64;
    unsafe {
        stream
            .launch_builder(&be.fill_ext3)
            .arg(&mut target)
            .arg(&count_arg)
            .arg(&value_dev)
            .launch(LaunchConfig::for_num_elems(count as u32))?;
    }
    Ok(())
}

/// `eq(point, ·)` as a table of `len` ext3 cells, doubled a variable at a time.
///
/// Variables go in back to front, which is what leaves variable 0 in the high
/// bit — the indexing every table here folds on.
pub fn eq_table_ext3(
    stream: &Arc<CudaStream>,
    point: &[u64],
    len: usize,
) -> Result<CudaSlice<u64>> {
    assert_eq!(
        1usize << (point.len() / 3),
        len,
        "the point spans the table"
    );
    let mut table = stream.alloc_zeros::<u64>(len * 3)?;
    eq_expand_into(stream, &mut table, 0, point, &[1, 0, 0])?;
    Ok(table)
}

/// The same, scaled by `seed` and written into `dst` from `offset`.
///
/// The scale rides the seed: the table is a product over the variables, so one
/// more factor at the start scales every cell. That is what lets a stacked
/// polynomial's weight be written column by column into one buffer.
pub fn eq_expand_into(
    stream: &Arc<CudaStream>,
    dst: &mut CudaSlice<u64>,
    offset: usize,
    point: &[u64],
    seed: &[u64],
) -> Result<()> {
    assert!(point.len().is_multiple_of(3), "three u64 per coordinate");
    assert_eq!(seed.len(), 3, "an ext3 seed");
    let len = 1usize << (point.len() / 3);
    assert!(dst.len() >= (offset + len) * 3, "the range holds the table");
    let be = backend()?;
    {
        let mut head = dst.slice_mut(offset * 3..offset * 3 + 3);
        stream.memcpy_htod(seed, &mut head)?;
    }
    // The whole point goes up once and each level reads its coordinate where
    // it lies: a stacked polynomial's weight is one of these per column, so a
    // send per level is tens of thousands of them for three u64 each.
    let point_dev = stream.clone_htod(point)?;
    let vars = point.len() / 3;
    for level in 0..vars {
        let at = (vars - 1 - level) * 3;
        let r = point_dev.slice(at..at + 3);
        let filled = 1u64 << level;
        let mut range = dst.slice_mut(offset * 3..(offset + len) * 3);
        unsafe {
            stream
                .launch_builder(&be.eq_expand_level_ext3)
                .arg(&mut range)
                .arg(&filled)
                .arg(&r)
                .launch(LaunchConfig::for_num_elems(filled as u32))?;
        }
    }
    Ok(())
}

/// A resident table's value at `point`, binding it in place.
///
/// The table is spent: what is left in its first cell is the value.
pub fn evaluate_resident_ext3(
    stream: &Arc<CudaStream>,
    values: &mut CudaSlice<u64>,
    len: usize,
    point: &[u64],
) -> Result<[u64; 3]> {
    assert!(point.len().is_multiple_of(3), "three u64 per coordinate");
    assert_eq!(
        1usize << (point.len() / 3),
        len,
        "the point spans the table"
    );
    let be = backend()?;
    fold_to_one(stream, be, values, len as u64, point)?;
    let out = stream.clone_dtoh(&values.slice(0..3))?;
    stream.synchronize()?;
    Ok([out[0], out[1], out[2]])
}
