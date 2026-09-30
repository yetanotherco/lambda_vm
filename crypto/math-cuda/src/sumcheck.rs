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
use crate::device::{alloc_or_trim, alloc_zeros_or_trim, backend};

/// Interpolation nodes per round the kernel has accumulator room for. Mirrors
/// `MAX_NODES` in `kernels/sumcheck.cu`.
pub const MAX_NODES: usize = 16;

const BLOCK_DIM: u32 = 256;

/// Smallest block a launch will take: one warp. Below this a block is not a
/// unit of scheduling any more.
const MIN_BLOCK: u32 = 32;

/// Blocks a launch aims for before it starts making them wider.
///
/// A round's threads are one per cube index, so a small cube — which is every
/// late round, and the late rounds are most of them — used to arrive as a
/// single block of 256 on a device with a hundred and seventy
/// multiprocessors. Spreading the same threads over narrow blocks puts them on
/// different multiprocessors, which is where the latency of the slot file gets
/// hidden.
const SPREAD_BLOCKS: u32 = 512;

/// Scratch ceiling for the per-thread slot file, which is what caps the grid:
/// a wider program buys fewer threads. 512 MiB leaves the factors and the
/// resident codewords room on a 32 GiB device.
const SLOT_BUDGET_BYTES: u64 = 512 * 1024 * 1024;

/// Most threads a round launches, whatever the slot file allows.
const MAX_THREADS: u64 = 1 << 20;

/// Blocks an elementwise kernel here may take. Enough to fill the device; past
/// it the grid-stride loop takes over.
const MAX_GRID: u32 = 4096;

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
    /// Threads `slots` was sized for, which is the first round's. Later
    /// rounds need fewer for the cube, and spend the difference on nodes.
    slot_threads: u64,
    partials: CudaSlice<u64>,
    /// Where the partials are summed, so a round's answer crosses the bus at
    /// three u64 per interpolation node instead of three per block.
    sums: CudaSlice<u64>,
    /// The threads this session's program can afford at once. Every round's
    /// launch is shaped out of it and the indices it has left.
    thread_ceiling: u64,
    /// The interpolation nodes as the device last saw them, and the buffer
    /// they live in. They are the same every round of a sumcheck, so the send
    /// happens once and the comparison is what decides that.
    t_dev: CudaSlice<u64>,
    t_host: Vec<u64>,
    /// Three u64 of scratch for the round's challenge, written in place rather
    /// than allocated per fold.
    r_dev: CudaSlice<u64>,
}

thread_local! {
    // ROUND-3 ARGUE DEVICE-BUSY SIZING (diagnostic; `LAMBDA_VM_ARGUE_BUSY_PROBE`).
    // A reusable TIMING-enabled CUDA event pair — created ONCE per thread, never
    // per round (a mid-prove cuEventCreate convoys the driver lock), and reused to
    // bracket the round kernels. Off by default: production round() only ever
    // reads the cached flag and skips all of this.
    static BUSY_EVENTS: std::cell::RefCell<
        Option<(cudarc::driver::CudaEvent, cudarc::driver::CudaEvent)>,
    > = const { std::cell::RefCell::new(None) };
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
        let mut buffer = unsafe { alloc_or_trim::<u64>(&stream, width * stride * 3) }?;
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
        let factor_ptrs = crate::device::htod_or_trim(&stream, &addresses)?;

        let ceiling = thread_ceiling(num_slots);
        let (grid, block) = launch_shape(ceiling, (stride / 2) as u64);
        let num_threads = grid as u64 * block as u64;
        let widest = widest_grid(ceiling, (stride / 2) as u64);

        let nodes_dev = crate::device::htod_or_trim(&stream, nodes)?;
        // A program with no constants still needs an allocation to point at.
        let consts_dev = crate::device::htod_or_trim(
            &stream,
            if consts.is_empty() {
                &[0u64][..]
            } else {
                consts
            },
        )?;
        let slots = unsafe { alloc_or_trim::<u64>(&stream, num_slots * 3 * num_threads as usize) }?;
        // Every partial the round reads is one the round wrote.
        let partials = unsafe { alloc_or_trim::<u64>(&stream, MAX_NODES * widest as usize * 3) }?;
        let t_dev = crate::device::alloc_zeros_or_trim::<u64>(&stream, MAX_NODES * 3)?;
        let r_dev = crate::device::alloc_zeros_or_trim::<u64>(&stream, 3)?;
        let sums = crate::device::alloc_zeros_or_trim::<u64>(&stream, MAX_NODES * 3)?;

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
            slot_threads: num_threads,
            partials,
            sums,
            thread_ceiling: ceiling,
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
        Self::from_device_spread(
            stream, addresses, len, held, nodes, consts, num_slots, root_slot, 1,
        )
    }

    /// [`from_device`](Self::from_device) with the slot file sized for
    /// `spread` threads per cube index, up to the slot budget.
    ///
    /// A round gives each index one thread and spreads the interpolation nodes
    /// over the grid's second dimension only as far as the slot file has room
    /// (`round`'s `nodes_wide`). Sized for one thread an index, a session's
    /// first round walks every node in turn on one thread; sized for `spread`
    /// — the round's degree — the nodes run side by side from the first round.
    #[allow(clippy::too_many_arguments)]
    pub fn from_device_spread(
        stream: Arc<CudaStream>,
        addresses: &[u64],
        len: usize,
        held: Vec<Arc<CudaSlice<u64>>>,
        nodes: &[u64],
        consts: &[u64],
        num_slots: usize,
        root_slot: u32,
        spread: usize,
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
        let ceiling = thread_ceiling(num_slots);
        let (grid, block) = launch_shape(
            ceiling,
            (len / 2) as u64 * spread.clamp(1, MAX_NODES) as u64,
        );
        let num_threads = grid as u64 * block as u64;
        let widest = widest_grid(ceiling, (len / 2) as u64);
        let factor_ptrs = crate::device::htod_or_trim(&stream, addresses)?;
        let nodes_dev = crate::device::htod_or_trim(&stream, nodes)?;
        let consts_dev = crate::device::htod_or_trim(
            &stream,
            if consts.is_empty() {
                &[0u64][..]
            } else {
                consts
            },
        )?;
        let slots = unsafe { alloc_or_trim::<u64>(&stream, num_slots * 3 * num_threads as usize) }?;
        let partials = unsafe { alloc_or_trim::<u64>(&stream, MAX_NODES * widest as usize * 3) }?;
        let t_dev = crate::device::alloc_zeros_or_trim::<u64>(&stream, MAX_NODES * 3)?;
        let r_dev = crate::device::alloc_zeros_or_trim::<u64>(&stream, 3)?;
        let sums = crate::device::alloc_zeros_or_trim::<u64>(&stream, MAX_NODES * 3)?;
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
            slot_threads: num_threads,
            partials,
            sums,
            thread_ceiling: ceiling,
            t_dev,
            t_host: Vec::new(),
            r_dev,
        })
    }

    /// Bytes this session holds on device, for admission control.
    pub fn device_bytes(factors: usize, cube: usize, num_slots: usize) -> u64 {
        let per_thread = num_slots as u64 * 3 * 8;
        let (grid, block) = launch_shape(thread_ceiling(num_slots), (cube / 2) as u64);
        let threads = grid as u64 * block as u64;
        factors as u64 * cube as u64 * 24 + threads * per_thread
    }

    /// Bytes a session over factors that are already on the card allocates
    /// for its rounds ([`from_device`](Self::from_device)): the slot file, the
    /// per-block partials of its widest round, and the node, challenge and sum
    /// scratch.
    pub fn scratch_bytes(cube: usize, num_slots: usize) -> u64 {
        let ceiling = thread_ceiling(num_slots);
        let (grid, block) = launch_shape(ceiling, (cube / 2) as u64);
        let threads = grid as u64 * block as u64;
        let widest = widest_grid(ceiling, (cube / 2) as u64) as u64;
        let nodes = MAX_NODES as u64;
        threads * num_slots as u64 * 24 + (nodes * widest + 2 * nodes + 1) * 24
    }

    /// A second session over this one's factors, walking another program:
    /// the same buffers, and a slot file of its own sized for one thread an
    /// index. It never folds — [`follow`](Self::follow) takes the cube size
    /// of the session it shadows after that one folds the buffers they share.
    /// For checking one program's rounds against another's over the same
    /// factors, in the same process.
    pub fn shadow(
        &self,
        nodes: &[u64],
        consts: &[u64],
        num_slots: usize,
        root_slot: u32,
    ) -> Result<Self> {
        Self::from_device(
            self.stream.clone(),
            &self.addresses,
            self.len,
            self.held.clone(),
            nodes,
            consts,
            num_slots,
            root_slot,
        )
    }

    /// Takes `leader`'s cube size: it folded the factors this session shares
    /// with it ([`shadow`](Self::shadow)).
    pub fn follow(&mut self, leader: &Self) {
        assert_eq!(
            self.addresses, leader.addresses,
            "a shadow follows the session whose factors it reads"
        );
        assert!(leader.len <= self.len, "a cube only shrinks");
        self.len = leader.len;
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
        // The launch follows the cube down, and narrows its blocks as it goes:
        // the same threads across more multiprocessors is what a late round
        // needs, and a block past the indices left writes a partial with
        // nothing in it.
        let (grid, block) = launch_shape(self.thread_ceiling, half);
        // Once the cube stops filling the slot file, the nodes do: a late round
        // is the same walk of the program at each of them, and they are
        // independent. The width comes out of the scratch the session already
        // has — the first round fills it with indices and every round after
        // frees half of it — so this costs no memory at all. The partials keep
        // their shape either way: a node has one owning block in the second
        // dimension.
        let per_node = grid as u64 * block as u64;
        let nodes_wide = (self.slot_threads / per_node.max(1)).clamp(1, num_t as u64) as u32;
        if self.t_host != t {
            let mut head = self.t_dev.slice_mut(0..t.len());
            self.stream.memcpy_htod(t, &mut head)?;
            self.t_host.clear();
            self.t_host.extend_from_slice(t);
        }
        let cfg = LaunchConfig {
            grid_dim: (grid, nodes_wide, 1),
            block_dim: (block, 1, 1),
            // One ext3 accumulator per thread, reduced one node at a time.
            shared_mem_bytes: block * 3 * 8,
        };
        let num_nodes = self.num_nodes as u64;
        let num_t_u32 = num_t as u32;
        // Round-3 argue device-busy sizing: bracket the round kernels with a
        // reusable event pair, read the elapsed after the EXISTING sync below.
        let busy_probe = crate::argue_probe::busy_probe_enabled();
        if busy_probe {
            BUSY_EVENTS.with(|cell| -> Result<()> {
                let mut ev = cell.borrow_mut();
                if ev.is_none() {
                    let f = cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
                    *ev = Some((be.ctx.new_event(Some(f))?, be.ctx.new_event(Some(f))?));
                }
                ev.as_ref().unwrap().0.record(&self.stream)?;
                Ok(())
            })?;
        }
        // S1-1: integer nodes take the componentwise multiply — only when
        // every node is `(k, 0, 0)`, where the two are the same value.
        let kernel = if int_nodes() && t.chunks_exact(3).all(|n| n[1] == 0 && n[2] == 0) {
            &be.sumcheck_round_ext3_int
        } else {
            &be.sumcheck_round_ext3
        };
        unsafe {
            self.stream
                .launch_builder(kernel)
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

        // The partials are summed here, on the device, and what crosses the bus
        // is the round's answer: three u64 per interpolation node. A wide
        // launch leaves megabytes of them, and every round of every sumcheck
        // waits for that copy before the transcript can move — which for a
        // whole proof is more time than the rounds themselves.
        let reduce_block = 256u32;
        unsafe {
            self.stream
                .launch_builder(&be.sum_partials_ext3)
                .arg(&self.partials)
                .arg(&(grid as u64))
                .arg(&mut self.sums)
                .launch(LaunchConfig {
                    grid_dim: (num_t as u32, 1, 1),
                    block_dim: (reduce_block, 1, 1),
                    shared_mem_bytes: reduce_block * 3 * 8,
                })?;
        }
        if busy_probe {
            BUSY_EVENTS.with(|cell| -> Result<()> {
                cell.borrow().as_ref().unwrap().1.record(&self.stream)?;
                Ok(())
            })?;
        }
        let sums = self.stream.clone_dtoh(&self.sums.slice(0..num_t * 3))?;
        self.stream.synchronize()?;
        if busy_probe {
            BUSY_EVENTS.with(|cell| -> Result<()> {
                let ev = cell.borrow();
                let (start, end) = ev.as_ref().unwrap();
                crate::argue_probe::add_device_busy_ns(
                    (start.elapsed_ms(end)? as f64 * 1.0e6) as u64,
                );
                Ok(())
            })?;
        }
        Ok(sums)
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

    /// Binds the round's variable to `r` in the first `count` factors only,
    /// and takes the cube down with them: for a shadow whose other factors a
    /// leader folds where they lie ([`crate::gkr::GruenLayer::shadow`]).
    pub fn fold_first(&mut self, count: usize, r: &[u64]) -> Result<()> {
        assert_eq!(r.len(), 3, "an ext3 challenge");
        assert!(self.len >= 2, "a fold needs a variable to bind");
        assert!(count <= self.width, "the factors to fold are the session's");
        let be = backend()?;
        let half = (self.len / 2) as u64;
        self.stream.memcpy_htod(r, &mut self.r_dev)?;
        let total = count as u64 * half;
        let grid = total.div_ceil(BLOCK_DIM as u64).min(4096) as u32;
        let width = count as u64;
        unsafe {
            self.stream
                .launch_builder(&be.sumcheck_fold_ext3)
                .arg(&mut self.factor_ptrs)
                .arg(&half)
                .arg(&width)
                .arg(&self.r_dev)
                .launch(LaunchConfig {
                    grid_dim: (grid.max(1), 1, 1),
                    block_dim: (BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        self.len /= 2;
        Ok(())
    }

    /// Whether [`download`](Self::download) can read the factors back: only a
    /// session that uploaded them knows their layout.
    pub fn can_download(&self) -> bool {
        self.uploaded
    }

    /// What every factor has left, read where it lies rather than through the
    /// buffer it sits in: a session over a fraction tree's layer does not own
    /// one, and the fold writes each factor back over its own base — so what
    /// is left of a factor is the first `len` elements at its address,
    /// whatever wrote them.
    pub fn values(&self) -> Result<Vec<Vec<u64>>> {
        self.stream.synchronize()?;
        let mut out = Vec::with_capacity(self.addresses.len());
        for address in &self.addresses {
            let mut values = vec![0u64; self.len * 3];
            // SAFETY: the address is a factor's base and the cube it spans is
            // `len` elements wide, the stream being idle (synchronized above).
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    values.as_mut_ptr() as *mut core::ffi::c_void,
                    *address,
                    self.len * 24,
                )
                .result()?;
            }
            out.push(values);
        }
        Ok(out)
    }

    /// [`values`](Self::values) — the same bytes — gathered into one buffer on
    /// the card and read back in one copy, where `values` makes a synchronous
    /// copy per factor: a table of fourteen hundred factors is fourteen hundred
    /// round trips at every session's end, for a few kilobytes each.
    ///
    /// Pageable, not pinned: what this saves is the count of copies, and a
    /// session's end is at most a megabyte or so.
    pub fn values_gathered(&self) -> Result<Vec<Vec<u64>>> {
        let be = backend()?;
        let span = self.len * 3;
        let total = self.width * span;
        // SAFETY: the kernel writes every word of it, one thread per word.
        let mut out = unsafe { alloc_or_trim::<u64>(&self.stream, total) }?;
        let width = self.width as u64;
        let cells = self.len as u64;
        let grid = (total as u64)
            .div_ceil(BLOCK_DIM as u64)
            .clamp(1, MAX_GRID as u64) as u32;
        unsafe {
            self.stream
                .launch_builder(&be.gather_factor_heads_ext3)
                .arg(&self.factor_ptrs)
                .arg(&width)
                .arg(&cells)
                .arg(&mut out)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let flat = self.stream.clone_dtoh(&out)?;
        self.stream.synchronize()?;
        Ok(flat.chunks_exact(span).map(<[u64]>::to_vec).collect())
    }

    /// Factor `k`'s remaining values, read where it lies: one factor, where
    /// [`values`](Self::values) reads every one — so a check of a table the
    /// card built reads that table and not the trace beside it.
    pub fn factor(&self, k: usize) -> Result<Vec<u64>> {
        self.stream.synchronize()?;
        let mut values = vec![0u64; self.len * 3];
        // SAFETY: as in `values`: the address is factor `k`'s base, it spans
        // `len` elements, and the stream is idle.
        unsafe {
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                values.as_mut_ptr() as *mut core::ffi::c_void,
                self.addresses[k],
                self.len * 24,
            )
            .result()?;
        }
        Ok(values)
    }

    /// Writes `value` over cell `index` of factor `k`: the way in for a fault
    /// that corrupts a table the card built, so the checks on it can be shown
    /// to see one.
    pub fn set_cell(&self, k: usize, index: usize, value: &[u64; 3]) -> Result<()> {
        assert!(index < self.len, "the cell is inside the cube");
        self.stream.synchronize()?;
        // SAFETY: the cell is inside factor `k`'s `len` elements, and the
        // stream is idle, so nothing reads or writes it meanwhile.
        unsafe {
            cudarc::driver::sys::cuMemcpyHtoD_v2(
                self.addresses[k] + (index * 24) as u64,
                value.as_ptr() as *const core::ffi::c_void,
                24,
            )
            .result()?;
        }
        Ok(())
    }

    /// What each factor has been bound to, once every variable is gone.
    pub fn bound_values(&self) -> Result<Vec<[u64; 3]>> {
        assert_eq!(self.len, 1, "a factor is bound once every variable is");
        Ok(self
            .values()?
            .into_iter()
            .map(|value| [value[0], value[1], value[2]])
            .collect())
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

/// Whether a round with integer nodes takes the componentwise multiply
/// (D-ARGUE S1-1): on by default, `LAMBDA_VM_ARGUE_INT_NODES=0` the opt-out.
/// The same value either way: `(k, 0, 0)·d = (k·d₀, k·d₁, k·d₂)`. Read once,
/// with a banner. Turned on with the fused zerocheck, by their A/B together
/// (FAST jobs 274 and 270).
pub fn int_nodes() -> bool {
    match INT_NODES_FORCED.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = int_nodes_from(std::env::var("LAMBDA_VM_ARGUE_INT_NODES").ok().as_deref());
        eprintln!(
            "[gpu] sumcheck rounds: {}",
            if on {
                "integer nodes (the default; LAMBDA_VM_ARGUE_INT_NODES=0 is the full multiply)"
            } else {
                "the full ext3 multiply (LAMBDA_VM_ARGUE_INT_NODES=0)"
            }
        );
        on
    })
}

/// The knob's reading of its variable: only `0` turns it off.
pub(crate) fn int_nodes_from(value: Option<&str>) -> bool {
    value != Some("0")
}

static INT_NODES_FORCED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// A test's own setting in place of the environment's (`None` = the environment).
#[doc(hidden)]
pub fn force_int_nodes(on: Option<bool>) {
    INT_NODES_FORCED.store(
        match on {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// The most threads a program of `num_slots` live values may run at once: the
/// slot file is per thread, so it is the thread count that gives way to a wider
/// program.
pub fn thread_ceiling(num_slots: usize) -> u64 {
    let per_thread = num_slots as u64 * 3 * 8;
    (SLOT_BUDGET_BYTES / per_thread.max(1))
        .min(MAX_THREADS)
        .max(MIN_BLOCK as u64)
}

/// The `(grid, block)` a launch of `work` indices takes, given the thread
/// ceiling: one thread per index, spread over [`SPREAD_BLOCKS`] blocks before
/// any block is made wider than a warp.
pub(crate) fn launch_shape(ceiling: u64, work: u64) -> (u32, u32) {
    let threads = work.clamp(1, ceiling);
    let wide = (threads / SPREAD_BLOCKS as u64).max(1);
    let block = (1u64 << (63 - wide.leading_zeros() as u64))
        .clamp(MIN_BLOCK as u64, BLOCK_DIM as u64) as u32;
    let grid = threads.div_ceil(block as u64).max(1) as u32;
    (grid, block)
}

/// The widest grid any round of a session will take, which is what the
/// per-block partials have to have room for. The cube halves every round and
/// the blocks narrow as it does, so the widest is not always the first.
fn widest_grid(ceiling: u64, first_half: u64) -> u32 {
    let mut widest = 1;
    let mut work = first_half.max(1);
    loop {
        widest = widest.max(launch_shape(ceiling, work).0);
        if work <= 1 {
            return widest;
        }
        work /= 2;
    }
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
    let mut values = crate::device::htod_or_trim(&stream, table)?;
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
    let base = crate::device::htod_or_trim(&stream, table)?;
    let half = table.len() / 2;
    let r = crate::device::htod_or_trim(&stream, &point[..3])?;
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

/// Every base-field table's value at `point`, in one pass per level.
///
/// A table is evaluated by folding it down to one element, which is a launch
/// per level and an upload of the table. Doing that per column turns a table's
/// argument into hundreds of round trips over the same trace; here the columns
/// go up together and every level is one launch for all of them.
///
/// `columns` are base-field slices of `2^point.len()/3` values each. Returns
/// one ext3 value per column, in order.
pub fn evaluate_many_base(
    columns: crate::columns::Columns<'_>,
    point: &[u64],
) -> Result<Vec<[u64; 3]>> {
    assert!(point.len().is_multiple_of(3), "three u64 per coordinate");
    let vars = point.len() / 3;
    let rows = 1usize << vars;
    assert!(vars > 0, "a point with no coordinates is the table itself");
    if columns.is_empty() {
        return Ok(Vec::new());
    }
    assert_eq!(columns.rows(), rows, "every column spans the point");
    let num_columns = columns.width();

    let be = backend()?;
    let stream = be.next_stream();
    // How many go up at a time: the base copy plus the ext3 half it folds to
    // is 20 bytes a row, and this keeps that transient bounded however wide
    // the table is.
    let per_column = rows as u64 * 20;
    let chunk = (CHUNK_BUDGET_BYTES / per_column.max(1)).clamp(1, num_columns as u64) as usize;

    let mut out = Vec::with_capacity(num_columns);
    for start in (0..num_columns).step_by(chunk) {
        let group = start..(start + chunk).min(num_columns);
        let group_len = group.len();
        let half = rows / 2;
        // Promised before it is used, like everything else that takes a slab
        // of the card: the caller's fallback is to evaluate the columns one at
        // a time, which needs almost nothing.
        let Some(_room) = crate::device::reserve(group_len as u64 * per_column) else {
            crate::device::note_device_fallback();
            return Err(cudarc::driver::DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
            ));
        };
        crate::argue_probe::note_device(
            crate::argue_probe::Surface::Sumcheck,
            group_len as u64 * per_column,
        );
        // Read where they lie when they are already there; a copy otherwise.
        let uploaded;
        let base = match &columns {
            crate::columns::Columns::Device { store, first, .. } => {
                store.view(first + group.start, group_len)
            }
            crate::columns::Columns::Host(host) => {
                let mut up = unsafe { alloc_or_trim::<u64>(&stream, group_len * rows) }?;
                for (k, column) in host[group.clone()].iter().enumerate() {
                    let at = k * rows;
                    let mut slab = up.slice_mut(at..at + rows);
                    stream.memcpy_htod(*column, &mut slab)?;
                }
                uploaded = up;
                uploaded.slice(0..group_len * rows)
            }
        };
        let r = crate::device::htod_or_trim(&stream, &point[..3])?;
        // SAFETY: the kernel writes every element of the halves it produces.
        let mut values = unsafe { alloc_or_trim::<u64>(&stream, group_len * half * 3) }?;
        let half_arg = half as u64;
        let tables = group_len as u64;
        let total = half_arg * tables;
        let grid = total.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
        unsafe {
            stream
                .launch_builder(&be.mle_fold_base_ext3_many)
                .arg(&base)
                .arg(&half_arg)
                .arg(&tables)
                .arg(&r)
                .arg(&mut values)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        // From here every table is ext3 and they all fold together: one launch
        // per level, over the list of addresses. A table of one variable has
        // no level left — the launch above bound it — and uploading its empty
        // rest of the point would ask the driver for a zero-byte allocation.
        if vars > 1 {
            let addresses: Vec<u64> = {
                let (at, _guard) = values.device_ptr(&stream);
                (0..group_len)
                    .map(|k| at + (k * half * 3 * 8) as u64)
                    .collect()
            };
            let mut factors = crate::device::htod_or_trim(&stream, &addresses)?;
            let mut span = half;
            let mut r_dev = crate::device::htod_or_trim(&stream, &point[3..6])?;
            for coordinate in point[3..].chunks_exact(3) {
                let fold_half = (span / 2) as u64;
                if fold_half == 0 {
                    break;
                }
                stream.memcpy_htod(coordinate, &mut r_dev)?;
                let width = group_len as u64;
                let total = width * fold_half;
                let grid = total.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
                unsafe {
                    stream
                        .launch_builder(&be.sumcheck_fold_ext3)
                        .arg(&mut factors)
                        .arg(&fold_half)
                        .arg(&width)
                        .arg(&r_dev)
                        .launch(LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (BLOCK_DIM, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
                span /= 2;
            }
        }

        // Three u64 per column, where each one's fold left it — not the
        // buffer, which is a level's worth of values nobody needs.
        //
        // Gathered there and brought back in one copy. A trace has thousands of
        // columns, and reading each one's head on its own is a transfer and a
        // stream synchronize apiece for twenty-four bytes.
        let heads: Vec<u32> = (0..group_len).map(|k| (k * half) as u32).collect();
        let packed = crate::fri::gather_ext3_at(&values, &heads, &stream)?;
        for head in packed.chunks_exact(3) {
            out.push([head[0], head[1], head[2]]);
        }
    }
    Ok(out)
}

/// How much a batched evaluation keeps on the device at once.
///
/// Small: what batching saves is the launches, and those dominate for a short
/// column — a tall one spends its time on the upload either way. Keeping the
/// slab small leaves the card to everything that has no host path.
const CHUNK_BUDGET_BYTES: u64 = 64 << 20;

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
    let mut table = crate::device::htod_or_trim(stream, &address)?;

    let point_dev = crate::device::htod_or_trim(stream, point)?;
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

/// A weight a sumcheck over resident factors adds on top of them.
///
/// A weight is `2^vars` cells; an `eq` weight is made of `vars` coordinates.
/// Given as its point it is built where it will be folded, and what crosses the
/// bus is the point.
#[derive(Clone, Copy, Debug)]
pub enum Extra<'a> {
    /// A table the host built: `len` ext3 values, three u64 each.
    Table(&'a [u64]),
    /// `eq(point, ·)`, the point three u64 per coordinate.
    Eq(&'a [u64]),
}

/// A table's factors, uploaded once and read by everything that walks them:
/// the LogUp input layer, and then the batched sumcheck.
pub struct DeviceFactors {
    stream: Arc<CudaStream>,
    buffer: Arc<CudaSlice<u64>>,
    addresses: Vec<u64>,
    /// The room this table's argument promised itself: the factors, the
    /// columns they are built from, and the rounds' scratch.
    _room: crate::device::DeviceReservation,
    /// The same list on device, where every program that walks these factors
    /// reads it — one send, not one per interaction.
    factor_ptrs: CudaSlice<u64>,
    len: usize,
}

impl DeviceFactors {
    /// Uploads `factors`, each `len` ext3 values interleaved.
    ///
    /// The prover builds them with [`from_columns`](Self::from_columns)
    /// instead; this is what that is checked against.
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
        let Some(room) = be.reserve(factors.len() as u64 * span as u64 * 8) else {
            crate::device::note_device_fallback();
            return Err(cudarc::driver::DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
            ));
        };
        crate::argue_probe::note_device(
            crate::argue_probe::Surface::Sumcheck,
            factors.len() as u64 * span as u64 * 8,
        );
        let stream = be.next_stream();
        let mut buffer = unsafe { alloc_or_trim::<u64>(&stream, factors.len() * span) }?;
        for (k, factor) in factors.iter().enumerate() {
            let at = k * span;
            let mut slab = buffer.slice_mut(at..at + span);
            stream.memcpy_htod(*factor, &mut slab)?;
        }
        let addresses: Vec<u64> = {
            let (base, _record) = buffer.device_ptr(&stream);
            (0..factors.len())
                .map(|k| base + (k * span * 8) as u64)
                .collect()
        };
        let factor_ptrs = crate::device::htod_or_trim(&stream, &addresses)?;
        Ok(Self {
            stream,
            buffer: Arc::new(buffer),
            addresses,
            _room: room,
            factor_ptrs,
            len,
        })
    }

    /// Builds the factors from the table's base columns rather than taking
    /// them built.
    ///
    /// A factor is a column read at a frame-step offset and lifted into the
    /// extension, so the columns are a third of what the factors are: building
    /// them here sends the trace instead of its lift, and spares the host the
    /// copy.
    ///
    /// `columns` is one base-field slice of `rows` values each; `plan` is three
    /// u64 per committed factor — the column it reads, its shift (already
    /// reduced mod `rows`), and the slot it fills; `public` is the extension
    /// tables that are not views of a column, each with the slot it goes to.
    pub fn from_columns(
        columns: crate::columns::Columns<'_>,
        plan: &[u64],
        public: &[(usize, &[u64])],
        rows: usize,
        width: usize,
    ) -> Result<Self> {
        assert!(rows.is_power_of_two(), "the cube is a power of two");
        assert!(width > 0, "a table has factors");
        assert!(
            plan.len().is_multiple_of(3),
            "three u64 per committed factor"
        );
        assert_eq!(
            plan.len() / 3 + public.len(),
            width,
            "every slot is filled once"
        );
        assert!(
            columns.is_empty() || columns.rows() == rows,
            "every column spans the cube"
        );

        let be = backend()?;
        // What stays: the factors. The base columns they are gathered from are
        // a third of that and are freed as soon as the gather has read them.
        let Some(room) = be.reserve(width as u64 * rows as u64 * 24) else {
            crate::device::note_device_fallback();
            return Err(cudarc::driver::DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
            ));
        };
        crate::argue_probe::note_device(
            crate::argue_probe::Surface::Sumcheck,
            width as u64 * rows as u64 * 24,
        );
        let stream = be.next_stream();

        let span = rows * 3;
        // SAFETY: every cell is written below — the public slots by their
        // copies, the rest by the kernel, which covers every (factor, row).
        let mut buffer = unsafe { alloc_or_trim::<u64>(&stream, width * span) }?;
        for (slot, table) in public {
            assert_eq!(table.len(), span, "a public factor spans the cube");
            let at = slot * span;
            let mut slab = buffer.slice_mut(at..at + span);
            stream.memcpy_htod(*table, &mut slab)?;
        }

        if !plan.is_empty() {
            // Read where they lie when they are already there; a copy otherwise.
            let uploaded;
            let base = match &columns {
                crate::columns::Columns::Device {
                    store,
                    first,
                    width,
                } => store.view(*first, *width),
                crate::columns::Columns::Host(host) => {
                    // SAFETY: every cell is written by the copies below.
                    let mut up = unsafe { alloc_or_trim::<u64>(&stream, host.len() * rows) }?;
                    for (k, column) in host.iter().enumerate() {
                        let at = k * rows;
                        let mut slab = up.slice_mut(at..at + rows);
                        stream.memcpy_htod(*column, &mut slab)?;
                    }
                    uploaded = up;
                    uploaded.slice(0..host.len() * rows)
                }
            };
            let plan_dev = crate::device::htod_or_trim(&stream, plan)?;
            let num_plan = (plan.len() / 3) as u64;
            let rows_arg = rows as u64;
            let total = num_plan * rows_arg;
            let grid = total.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&be.factors_from_columns_ext3)
                    .arg(&base)
                    .arg(&plan_dev)
                    .arg(&num_plan)
                    .arg(&rows_arg)
                    .arg(&mut buffer)
                    .launch(cfg)?;
            }
            // A copy made here is spent and goes at the end of this block,
            // which is stream-ordered behind the kernel that just read it; a
            // view of the epoch's columns frees nothing, because they stay.
        }

        let addresses: Vec<u64> = {
            let (base, _record) = buffer.device_ptr(&stream);
            (0..width).map(|k| base + (k * span * 8) as u64).collect()
        };
        let factor_ptrs = crate::device::htod_or_trim(&stream, &addresses)?;
        Ok(Self {
            stream,
            buffer: Arc::new(buffer),
            addresses,
            _room: room,
            factor_ptrs,
            len: rows,
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

    /// The factors' device addresses, as the kernels read them.
    pub(crate) fn factor_ptrs(&self) -> &CudaSlice<u64> {
        &self.factor_ptrs
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
        let (grid, block) = launch_shape(thread_ceiling(num_slots), self.len as u64);
        let num_threads = grid as u64 * block as u64;
        let nodes_dev = crate::device::htod_or_trim(&self.stream, nodes)?;
        let consts_dev = crate::device::htod_or_trim(
            &self.stream,
            if consts.is_empty() {
                &[0u64][..]
            } else {
                consts
            },
        )?;
        let mut slots = unsafe {
            self.stream
                .alloc::<u64>(num_slots * 3 * num_threads as usize)
        }?;
        let mut target = out.slice_mut(offset * 3..(offset + self.len) * 3);
        let rows = self.len as u64;
        let num_nodes = (nodes.len() / 2) as u64;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
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
        let extra: Vec<Extra<'_>> = extra.iter().map(|table| Extra::Table(table)).collect();
        self.session_with(&extra, nodes, consts, num_slots, root_slot)
    }

    /// The same, with each weight a table to upload or the point of an `eq`
    /// table to build here ([`Extra`]).
    #[allow(clippy::too_many_arguments)]
    pub fn session_with(
        &self,
        extra: &[Extra<'_>],
        nodes: &[u64],
        consts: &[u64],
        num_slots: usize,
        root_slot: u32,
    ) -> Result<SumcheckSession> {
        self.session_spread(extra, nodes, consts, num_slots, root_slot, 1)
    }

    /// [`session_with`](Self::session_with) with the slot file sized for
    /// `spread` threads per cube index
    /// ([`SumcheckSession::from_device_spread`]).
    #[allow(clippy::too_many_arguments)]
    pub fn session_spread(
        &self,
        extra: &[Extra<'_>],
        nodes: &[u64],
        consts: &[u64],
        num_slots: usize,
        root_slot: u32,
        spread: usize,
    ) -> Result<SumcheckSession> {
        let span = self.len * 3;
        let mut addresses = self.addresses.clone();
        let mut held = vec![self.buffer.clone()];
        if !extra.is_empty() {
            for weight in extra {
                match weight {
                    Extra::Table(table) => {
                        assert_eq!(table.len(), span, "every factor spans the same cube")
                    }
                    Extra::Eq(point) => assert!(
                        point.len().is_multiple_of(3) && 1usize << (point.len() / 3) == self.len,
                        "an eq weight's point spans the cube"
                    ),
                }
            }
            // SAFETY: every slab is written below — a table by its copy, an
            // `eq` table by its seed and levels, which cover all of it.
            let mut buffer = unsafe { self.stream.alloc::<u64>(extra.len() * span) }?;
            for (k, weight) in extra.iter().enumerate() {
                let at = k * span;
                match weight {
                    Extra::Table(table) => {
                        let mut slab = buffer.slice_mut(at..at + span);
                        self.stream.memcpy_htod(*table, &mut slab)?;
                    }
                    Extra::Eq(point) => {
                        eq_expand_into(&self.stream, &mut buffer, k * self.len, point, &[1, 0, 0])?
                    }
                }
            }
            {
                let (base, _record) = buffer.device_ptr(&self.stream);
                addresses.extend((0..extra.len()).map(|k| base + (k * span * 8) as u64));
            }
            held.push(Arc::new(buffer));
        }
        SumcheckSession::from_device_spread(
            self.stream.clone(),
            &addresses,
            self.len,
            held,
            nodes,
            consts,
            num_slots,
            root_slot,
            spread,
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
    let value_dev = crate::device::htod_or_trim(stream, value)?;
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
    let mut table = alloc_zeros_or_trim::<u64>(stream, len * 3)?;
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
    let point_dev = crate::device::htod_or_trim(stream, point)?;
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

/// Every share of a stacked polynomial's weight, built in one pass per level
/// instead of one pass per level per share.
///
/// A share is `(offset, point, scale)`: its subcube's start, the point its
/// `eq` table is over, and what that table is scaled by. They are disjoint, so
/// the only reason to do them one at a time was that the kernel took one — and
/// on a real trace that is tens of thousands of launches for a tenth of a
/// second of work.
///
/// `dst` must be zero where no share writes: the gaps between subcubes are
/// part of the weight.
pub fn eq_expand_shares_ext3(
    stream: &Arc<CudaStream>,
    dst: &mut CudaSlice<u64>,
    shares: &[(usize, Vec<u64>, [u64; 3])],
) -> Result<()> {
    if shares.is_empty() {
        return Ok(());
    }
    let be = backend()?;

    // Sorted by variable count, descending: then the shares still doubling at
    // level `l` are a prefix, and a thread finds its share by shifting.
    let mut order: Vec<usize> = (0..shares.len()).collect();
    order.sort_by_key(|i| std::cmp::Reverse(shares[*i].1.len()));

    let mut table = Vec::with_capacity(order.len() * 3);
    let mut points = Vec::new();
    let mut scales = Vec::with_capacity(order.len() * 3);
    for &i in &order {
        let (offset, point, scale) = &shares[i];
        assert!(point.len().is_multiple_of(3), "three u64 per coordinate");
        table.push(*offset as u64);
        table.push((point.len() / 3) as u64);
        table.push((points.len() / 3) as u64);
        points.extend_from_slice(point);
        scales.extend_from_slice(scale);
    }
    let max_vars = shares[order[0]].1.len() / 3;

    let table_dev = crate::device::htod_or_trim(stream, &table)?;
    let scales_dev = crate::device::htod_or_trim(stream, &scales)?;
    let points_dev = crate::device::htod_or_trim(
        stream,
        if points.is_empty() {
            &[0u64][..]
        } else {
            &points
        },
    )?;

    let count = order.len() as u64;
    unsafe {
        stream
            .launch_builder(&be.eq_seed_shares_ext3)
            .arg(&mut *dst)
            .arg(&table_dev)
            .arg(&scales_dev)
            .arg(&count)
            .launch(LaunchConfig::for_num_elems(order.len() as u32))?;
    }

    for level in 0..max_vars {
        // The prefix still doubling: `vars > level`, and the list is sorted.
        let active = order
            .iter()
            .take_while(|i| shares[**i].1.len() / 3 > level)
            .count() as u64;
        if active == 0 {
            break;
        }
        let total = active << level;
        let grid = total.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
        let level_arg = level as u64;
        unsafe {
            stream
                .launch_builder(&be.eq_expand_level_shares_ext3)
                .arg(&mut *dst)
                .arg(&table_dev)
                .arg(&points_dev)
                .arg(&active)
                .arg(&level_arg)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
    }
    Ok(())
}

/// The claim reduce's session with its tables built here, out of the epoch's
/// resident columns: per offset, in the reduce's order, the shift table and the
/// batched column.
///
/// A shift table is `eq(alpha)` read `offset` rows back, cyclically — the shift
/// kernel at a corner `y` is the equality kernel at `y − offset`, which
/// `multilinear::eq`'s `a_shift_table_is_the_eq_table_rotated` pins — so
/// `eq(alpha)` is built once and each offset's table is two copies of it. A
/// batched column is `Σ weight·column` over the offset's members
/// (`batched_column_ext3`).
///
/// `alpha` is three u64 per coordinate. `groups` is one `(offset, members,
/// weights)` per offset: each member a column of the run `first..first + width`
/// of `store`, each weight three u64. What crosses the bus is the point, the
/// members and their weights: a few kilobytes where the tables are gigabytes.
#[allow(clippy::too_many_arguments)]
pub fn reduce_session(
    store: &crate::columns::DeviceColumns,
    first: usize,
    width: usize,
    alpha: &[u64],
    groups: &[(usize, Vec<u64>, Vec<u64>)],
    nodes: &[u64],
    consts: &[u64],
    num_slots: usize,
    root_slot: u32,
) -> Result<SumcheckSession> {
    assert!(
        alpha.len().is_multiple_of(3) && !alpha.is_empty(),
        "a point of at least one coordinate, three u64 each"
    );
    let len = 1usize << (alpha.len() / 3);
    assert!(
        store.is_run(first, width) && store.span(first).1 == len,
        "the columns are a run spanning the cube"
    );
    assert!(!groups.is_empty(), "a reduction reads at an offset");
    for (_, members, weights) in groups {
        assert!(!members.is_empty(), "an offset has a column reading at it");
        assert_eq!(
            weights.len(),
            members.len() * 3,
            "one ext3 weight per member"
        );
        assert!(
            members.iter().all(|&m| (m as usize) < width),
            "a member is a column of the run"
        );
    }

    let be = backend()?;
    let stream = be.next_stream();
    let span = len * 3;
    // SAFETY: every slab is written below — a shift slab by `eq(alpha)` or by
    // its two copies, a batched slab by the kernel, which covers every row.
    let mut buffer = unsafe { alloc_or_trim::<u64>(&stream, 2 * groups.len() * span) }?;
    // `eq(alpha)` is the first shift slab when the first offset is no shift,
    // and a table of its own, freed once copied, otherwise.
    let in_place = groups[0].0.is_multiple_of(len);
    let own = if in_place {
        eq_expand_into(&stream, &mut buffer, 0, alpha, &[1, 0, 0])?;
        None
    } else {
        Some(eq_table_ext3(&stream, alpha, len)?)
    };
    let base = {
        let (base, _record) = buffer.device_ptr(&stream);
        base
    };
    let eq_at = match &own {
        Some(table) => {
            let (at, _record) = table.device_ptr(&stream);
            at
        }
        None => base,
    };
    let copy = |dst: u64, src: u64, bytes: usize| -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: both ranges lie inside allocations this call holds, they do
        // not overlap — a shift slab is never the table it is copied from —
        // and the copy is ordered on `stream` behind whatever wrote the source.
        unsafe {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.cu_stream())
                .result()?;
        }
        Ok(())
    };
    let columns = store.view(first, width);
    let rows = len as u64;
    let grid = rows.div_ceil(BLOCK_DIM as u64).clamp(1, MAX_GRID as u64) as u32;
    for (i, (offset, members, weights)) in groups.iter().enumerate() {
        if !(i == 0 && in_place) {
            // `shift[y] = eq[(y − k) mod len]`: the table's tail lands at its
            // head, and the rest after it.
            let k = offset % len;
            let shift_at = base + (2 * i * span * 8) as u64;
            copy(shift_at + (k * 24) as u64, eq_at, (len - k) * 24)?;
            copy(shift_at, eq_at + ((len - k) * 24) as u64, k * 24)?;
        }
        let members_dev = crate::device::htod_or_trim(&stream, members)?;
        let weights_dev = crate::device::htod_or_trim(&stream, weights)?;
        let count = members.len() as u64;
        let at = (2 * i + 1) * span;
        let mut out = buffer.slice_mut(at..at + span);
        unsafe {
            stream
                .launch_builder(&be.batched_column_ext3)
                .arg(&columns)
                .arg(&rows)
                .arg(&members_dev)
                .arg(&weights_dev)
                .arg(&count)
                .arg(&mut out)
                .launch(LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
    }
    // Freed on `stream`, so behind the copies that read it.
    drop(own);
    let addresses: Vec<u64> = (0..2 * groups.len())
        .map(|k| base + (k * span * 8) as u64)
        .collect();
    SumcheckSession::from_device(
        stream,
        &addresses,
        len,
        vec![Arc::new(buffer)],
        nodes,
        consts,
        num_slots,
        root_slot,
    )
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A launch never runs more threads than the cube has indices, nor more
    /// than the slot file can hold, and always fills at least one warp.
    #[test]
    fn a_launch_takes_a_thread_per_index_and_no_more() {
        for slots in [1usize, 6, 64, 522, 1036, 8192] {
            let ceiling = thread_ceiling(slots);
            for log_work in 0..24u32 {
                let work = 1u64 << log_work;
                let (grid, block) = launch_shape(ceiling, work);
                assert!((MIN_BLOCK..=BLOCK_DIM).contains(&block), "block {block}");
                let threads = grid as u64 * block as u64;
                assert!(threads >= work.min(ceiling), "{threads} threads for {work}");
                assert!(
                    threads < work.min(ceiling) + block as u64,
                    "{threads} threads for {work}: a whole block of nothing",
                );
            }
        }
    }

    /// A small cube spreads over blocks instead of arriving as one of them:
    /// that is the whole point, and it is what a late round looks like.
    #[test]
    fn a_small_cube_spreads_over_the_device() {
        let ceiling = thread_ceiling(1036);
        for work in [128u64, 1024, 8192] {
            let (grid, _) = launch_shape(ceiling, work);
            assert!(grid >= 4, "{work} indices arrived as {grid} block(s)");
        }
    }

    /// The partials buffer is sized for the widest round, which is not always
    /// the first — the blocks narrow as the cube shrinks.
    #[test]
    fn the_partials_hold_every_round() {
        for slots in [6usize, 1036] {
            let ceiling = thread_ceiling(slots);
            for log_half in 0..22u32 {
                let first = 1u64 << log_half;
                let widest = widest_grid(ceiling, first);
                let mut work = first;
                loop {
                    assert!(launch_shape(ceiling, work).0 <= widest);
                    if work <= 1 {
                        break;
                    }
                    work /= 2;
                }
            }
        }
    }
}
