//! Device build of the mixed-height MMCS — one tree over all of an epoch's
//! matrices.
//!
//! The host contract this must reproduce byte for byte lives in
//! `crypto/stark/src/fri/mmcs.rs`: leaf `k` of a height group is Keccak-256 over
//! the concatenation, in INPUT order, of each matrix's bit-reversed rows `2k` and
//! `2k+1`; the climb compresses pairs and, where a shorter group's height matches
//! the halved layer, compresses the parent again with that group's leaf digest.
//!
//! # Why this mirrors the streaming builder, not `commit`
//!
//! `MixedMmcs::commit` hashes a group's leaf in one pass, which needs every
//! matrix of that height readable at once. The tallest group is most of a real
//! epoch's tables, so that is `O(N)` LDE resident at the base layer — the memory
//! the batching exists to remove. [`MmcsGroupHasher`] is the device twin of the
//! host's `StreamingMmcsBuilder`: the per-leaf sponge lives in VRAM and matrices
//! are absorbed into it one at a time, so the caller produces one matrix's LDE on
//! device, absorbs it, and frees it.
//!
//! Sponge state is 204 bytes per leaf (25 lanes plus the rate cursor):
//! ~214 MiB at 2^20 leaves, against one full LDE per matrix in the group.
//!
//! # Node layout, and why the path gather is unchanged
//!
//! [`build_mmcs_tree_on_device`] writes the STANDARD heap array — inner nodes at
//! `[0, leaves_len-1)` with the root at 0, leaves at `[leaves_len-1, ..)` — the
//! same layout [`crate::merkle::build_merkle_tree_on_device`] produces. In that
//! layout the sibling a query needs at MMCS level `L` is exactly the node
//! [`crate::merkle::gather_merkle_paths_dev`] already walks to, so the batched
//! path gather is that kernel unchanged. Keeping one layout is what stops a
//! second index convention existing to drift from the first.

use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};
use std::sync::Arc;

use crate::Result;
use crate::device::backend;
use crate::merkle::keccak_launch_cfg;

/// One height group's per-leaf sponges, live on device between absorptions.
///
/// Construct once per height group, [`Self::absorb_row_major`] /
/// [`Self::absorb_ext3_slabs`] once per matrix at that height IN INPUT ORDER
/// (the leaf concatenation binds that order), then [`Self::finalize`].
pub struct MmcsGroupHasher {
    states: CudaSlice<u64>,
    rate_pos: CudaSlice<u32>,
    num_leaves: u64,
    /// `log2` of the group's row count — every matrix absorbed here must have it,
    /// since they share the leaves.
    log_num_rows: u64,
    absorbed: usize,
}

impl MmcsGroupHasher {
    /// Zeroed sponges for a height group of `2^log_num_rows` rows, i.e.
    /// `2^(log_num_rows - 1)` leaves.
    pub fn new(stream: &Arc<CudaStream>, log_num_rows: u64) -> Result<Self> {
        assert!(
            log_num_rows >= 1,
            "row-pair leaves need at least 2 rows (log_num_rows >= 1)"
        );
        let be = backend()?;
        let num_leaves = 1u64 << (log_num_rows - 1);
        let mut states = stream.alloc_zeros::<u64>((num_leaves * 25) as usize)?;
        let mut rate_pos = stream.alloc_zeros::<u32>(num_leaves as usize)?;

        // `alloc_zeros` already gives the state we want; the kernel runs anyway so
        // the zeroing is this module's own statement rather than an allocator
        // property a future change could quietly take away.
        let cfg = keccak_launch_cfg(num_leaves);
        unsafe {
            stream
                .launch_builder(&be.mmcs_states_init)
                .arg(&mut states)
                .arg(&mut rate_pos)
                .arg(&num_leaves)
                .launch(cfg)?;
        }

        Ok(Self {
            states,
            rate_pos,
            num_leaves,
            log_num_rows,
            absorbed: 0,
        })
    }

    /// Absorb one row-major matrix's row pair into every leaf. Columns
    /// `[col_start, col_end)` are absorbed while `row_stride` stays the full row
    /// width, so a preprocessed table's two column ranges over one buffer are two
    /// absorptions rather than two buffers.
    ///
    /// Base field: `row_stride` and the range are in columns. Ext3: an element's
    /// three components are consecutive, so both are in components — the same
    /// convention `keccak256_leaves_base_row_major_row_pair` documents.
    ///
    /// The caller may free `data` as soon as this returns on `stream`.
    #[allow(clippy::too_many_arguments)]
    pub fn absorb_row_major(
        &mut self,
        stream: &Arc<CudaStream>,
        data: &CudaSlice<u64>,
        row_stride: u64,
        col_start: u64,
        col_end: u64,
    ) -> Result<()> {
        assert!(
            col_start <= col_end && col_end <= row_stride,
            "column range [{col_start}, {col_end}) does not fit a stride of {row_stride}"
        );
        let be = backend()?;
        let num_rows = 1u64 << self.log_num_rows;
        let cfg = keccak_launch_cfg(self.num_leaves);
        unsafe {
            stream
                .launch_builder(&be.mmcs_absorb_row_pair_row_major)
                .arg(&mut self.states)
                .arg(&mut self.rate_pos)
                .arg(data)
                .arg(&row_stride)
                .arg(&col_start)
                .arg(&col_end)
                .arg(&num_rows)
                .arg(&self.log_num_rows)
                .arg(&self.num_leaves)
                .launch(cfg)?;
        }
        self.absorbed += 1;
        Ok(())
    }

    /// Absorb one COLUMN-MAJOR base-field matrix's row pair into every leaf —
    /// main's resident LDE layout (`GpuLdeBase`), element `(row, col)` at
    /// `col * col_stride + row`. Columns `[col_start, col_end)` are absorbed.
    ///
    /// Produces the identical leaf digests as [`Self::absorb_row_major`] over the
    /// same matrix (same absorbed byte stream, only the read layout differs), so
    /// a device commit fed from a resident column-major LDE is byte-identical to
    /// the host tree. This is the bridge for main's resident base-field LDEs,
    /// which are column-major; the ext3 parts already match [`Self::absorb_ext3_slabs`].
    ///
    /// The caller may free `data` as soon as this returns on `stream`.
    pub fn absorb_col_major(
        &mut self,
        stream: &Arc<CudaStream>,
        data: &CudaSlice<u64>,
        col_stride: u64,
        col_start: u64,
        col_end: u64,
    ) -> Result<()> {
        let num_rows = 1u64 << self.log_num_rows;
        assert!(
            col_start <= col_end,
            "column range [{col_start}, {col_end}) is empty-or-inverted"
        );
        assert!(
            num_rows <= col_stride,
            "col_stride {col_stride} must hold all {num_rows} rows of a column"
        );
        let be = backend()?;
        let cfg = keccak_launch_cfg(self.num_leaves);
        unsafe {
            stream
                .launch_builder(&be.mmcs_absorb_row_pair_col_major)
                .arg(&mut self.states)
                .arg(&mut self.rate_pos)
                .arg(data)
                .arg(&col_stride)
                .arg(&col_start)
                .arg(&col_end)
                .arg(&num_rows)
                .arg(&self.log_num_rows)
                .arg(&self.num_leaves)
                .launch(cfg)?;
        }
        self.absorbed += 1;
        Ok(())
    }

    /// Absorb one column-major ext3 slab matrix — the composition-poly LDE
    /// layout, component `k` of column `c` at `(c*3 + k) * col_stride`.
    pub fn absorb_ext3_slabs(
        &mut self,
        stream: &Arc<CudaStream>,
        parts: &CudaSlice<u64>,
        col_stride: u64,
        num_parts: u64,
    ) -> Result<()> {
        let be = backend()?;
        let num_rows = 1u64 << self.log_num_rows;
        let cfg = keccak_launch_cfg(self.num_leaves);
        unsafe {
            stream
                .launch_builder(&be.mmcs_absorb_row_pair_ext3_slabs)
                .arg(&mut self.states)
                .arg(&mut self.rate_pos)
                .arg(parts)
                .arg(&col_stride)
                .arg(&num_parts)
                .arg(&num_rows)
                .arg(&self.log_num_rows)
                .arg(&self.num_leaves)
                .launch(cfg)?;
        }
        self.absorbed += 1;
        Ok(())
    }

    /// Absorb one ROW-MAJOR ext3 matrix's row pair — the batched aux LDE layout
    /// (`expand_aux_lde_row_major`), element `(row, col)` as 3 consecutive u64 at
    /// `(row*stride + col)*3`, `stride` elements per row. Columns `[col_start,
    /// col_end)` are absorbed. Same absorbed byte order as the host's row-major
    /// ext3 leaf.
    pub fn absorb_ext3_row_major(
        &mut self,
        stream: &Arc<CudaStream>,
        data: &CudaSlice<u64>,
        stride: u64,
        col_start: u64,
        col_end: u64,
    ) -> Result<()> {
        let be = backend()?;
        let num_rows = 1u64 << self.log_num_rows;
        let cfg = keccak_launch_cfg(self.num_leaves);
        unsafe {
            stream
                .launch_builder(&be.mmcs_absorb_row_pair_ext3_row_major)
                .arg(&mut self.states)
                .arg(&mut self.rate_pos)
                .arg(data)
                .arg(&stride)
                .arg(&col_start)
                .arg(&col_end)
                .arg(&num_rows)
                .arg(&self.log_num_rows)
                .arg(&self.num_leaves)
                .launch(cfg)?;
        }
        self.absorbed += 1;
        Ok(())
    }

    /// Pad and squeeze every leaf. Panics if nothing was absorbed: an empty
    /// group's digests would be the hash of nothing, which is a leaf no verifier
    /// can rebuild from an opening.
    pub fn finalize(self, stream: &Arc<CudaStream>) -> Result<CudaSlice<u8>> {
        assert!(
            self.absorbed > 0,
            "a height group must absorb at least one matrix before it is finalized"
        );
        let be = backend()?;
        let mut digests = stream.alloc_zeros::<u8>((self.num_leaves * 32) as usize)?;
        let cfg = keccak_launch_cfg(self.num_leaves);
        unsafe {
            stream
                .launch_builder(&be.mmcs_states_finalize)
                .arg(&self.states)
                .arg(&self.rate_pos)
                .arg(&self.num_leaves)
                .arg(&mut digests)
                .launch(cfg)?;
        }
        Ok(digests)
    }

    pub fn num_leaves(&self) -> u64 {
        self.num_leaves
    }
}

/// Build the mixed-height tree from each height group's finalized leaf digests.
///
/// `group_digests[h]` is `Some(device digests)` when some matrix has
/// `log_height == h`, each `2^(h-1)` digests of 32 bytes; index `h_max` must be
/// present. Returns the standard heap node buffer
/// (`(2 * 2^(h_max-1) - 1) * 32` bytes) resident on device.
pub fn build_mmcs_tree_on_device(
    stream: &Arc<CudaStream>,
    group_digests: &[Option<CudaSlice<u8>>],
) -> Result<CudaSlice<u8>> {
    let h_max = group_digests.len() - 1;
    assert!(
        h_max >= 1 && group_digests[h_max].is_some(),
        "the tallest height group must be present"
    );
    let be = backend()?;
    let leaves_len = 1u64 << (h_max - 1);

    let mut nodes = stream.alloc_zeros::<u8>(((2 * leaves_len - 1) * 32) as usize)?;
    // Base layer into the leaf tail of the heap array.
    let base = group_digests[h_max]
        .as_ref()
        .expect("checked immediately above");
    let mut leaf_tail = nodes.slice_mut(((leaves_len - 1) * 32) as usize..);
    stream.memcpy_dtod(base, &mut leaf_tail)?;

    // Climb. Level `i` produces the layer whose codeword height is
    // `h_max - 1 - i`, which is where a group of that height injects — the same
    // schedule `MixedMmcs::from_group_digests` walks.
    let mut level_begin: u64 = leaves_len - 1;
    let mut i = 0usize;
    while level_begin != 0 {
        let new_begin = level_begin / 2;
        let n_pairs = level_begin - new_begin;
        let inject_h = h_max - 1 - i;
        let injected = group_digests.get(inject_h).and_then(Option::as_ref);
        let has_inject: u32 = u32::from(injected.is_some());

        // `keccak_mmcs_level` reads `inject` only when `has_inject` is set, so a
        // level with no injection still needs a pointer argument. Reuse the
        // node buffer's own base rather than allocating a dummy: it is a valid
        // device pointer that the kernel provably never dereferences.
        let cfg = keccak_launch_cfg(n_pairs);
        match injected {
            Some(digests) => unsafe {
                stream
                    .launch_builder(&be.keccak_mmcs_level)
                    .arg(&mut nodes)
                    .arg(&new_begin)
                    .arg(&n_pairs)
                    .arg(digests)
                    .arg(&has_inject)
                    .launch(cfg)?;
            },
            None => {
                let empty = stream.alloc_zeros::<u8>(32)?;
                unsafe {
                    stream
                        .launch_builder(&be.keccak_mmcs_level)
                        .arg(&mut nodes)
                        .arg(&new_begin)
                        .arg(&n_pairs)
                        .arg(&empty)
                        .arg(&has_inject)
                        .launch(cfg)?;
                }
            }
        }

        level_begin = new_begin;
        i += 1;
    }

    Ok(nodes)
}

/// The MMCS root — node 0 of the heap array.
pub fn read_mmcs_root(stream: &Arc<CudaStream>, nodes: &CudaSlice<u8>) -> Result<[u8; 32]> {
    let head = nodes.slice(0..32);
    let bytes = stream.clone_dtoh(&head)?;
    let mut root = [0u8; 32];
    root.copy_from_slice(&bytes);
    Ok(root)
}

/// One base-field matrix feeding a mixed-height MMCS commit, in absorb order.
/// `data` is the full ROW-MAJOR `stride`-wide LDE buffer (Goldilocks u64 words);
/// columns `[col_start, col_end)` are the ones committed (a preprocessed table
/// commits its non-precomputed range).
pub struct MmcsRowMajorInput<'a> {
    pub data: &'a [u64],
    pub stride: u64,
    pub col_start: u64,
    pub col_end: u64,
    pub log_height: u64,
}

/// Build the whole mixed-height MMCS tree on the GPU from row-major base-field
/// matrices and return its root — the device twin of the host
/// `StreamingMmcsBuilder`: matrices are grouped by height, each group's leaves
/// concatenate its matrices in the given order, and the climb injects the
/// shorter groups. Callers hand matrices in the SAME order the host absorbs them.
///
/// Uploads each matrix and frees it before the next (the streaming residency
/// policy). `inputs` must be non-empty and include the tallest height.
/// Like [`commit_mixed_row_major_root`] but returns the WHOLE standard heap node
/// array (host-side `Vec<u8>`, `(2L-1)*32` bytes) — feed it to
/// `MixedMmcs::from_heap_nodes` to make the GPU tree authoritative.
pub fn commit_mixed_row_major_nodes(inputs: &[MmcsRowMajorInput]) -> Result<Vec<u8>> {
    let be = backend()?;
    let stream = be.next_stream();

    let h_max = inputs.iter().map(|m| m.log_height).max().unwrap_or(0);
    let mut group_digests: Vec<Option<CudaSlice<u8>>> = (0..=h_max).map(|_| None).collect();

    let mut h = h_max;
    while h >= 1 {
        if inputs.iter().any(|m| m.log_height == h) {
            let mut hasher = MmcsGroupHasher::new(&stream, h)?;
            for m in inputs.iter().filter(|m| m.log_height == h) {
                let dev = stream.clone_htod(m.data)?;
                hasher.absorb_row_major(&stream, &dev, m.stride, m.col_start, m.col_end)?;
                drop(dev);
            }
            group_digests[h as usize] = Some(hasher.finalize(&stream)?);
        }
        h -= 1;
    }

    let nodes = build_mmcs_tree_on_device(&stream, &group_digests)?;
    stream.clone_dtoh(&nodes)
}

pub fn commit_mixed_row_major_root(inputs: &[MmcsRowMajorInput]) -> Result<[u8; 32]> {
    let nodes = commit_mixed_row_major_nodes(inputs)?;
    let mut root = [0u8; 32];
    root.copy_from_slice(&nodes[0..32]);
    Ok(root)
}

/// One ext3 matrix feeding a mixed-height MMCS commit, in absorb order, laid out
/// as COLUMN-MAJOR SLABS: component `k` of column `p` at `(p*3 + k) * col_stride`
/// (`col_stride` = rows per column), natural row order. This is the
/// composition-poly / `GpuLdeExt3` layout.
pub struct MmcsExt3SlabInput<'a> {
    pub data: &'a [u64],
    pub col_stride: u64,
    pub num_parts: u64,
    pub log_height: u64,
}

/// Build the whole mixed-height MMCS tree on the GPU from ext3 slab matrices and
/// return its root — the ext3 twin of [`commit_mixed_row_major_root`], grouping
/// by height and absorbing each group's matrices in the given order.
pub fn commit_mixed_ext3_slabs_nodes(inputs: &[MmcsExt3SlabInput]) -> Result<Vec<u8>> {
    let be = backend()?;
    let stream = be.next_stream();

    let h_max = inputs.iter().map(|m| m.log_height).max().unwrap_or(0);
    let mut group_digests: Vec<Option<CudaSlice<u8>>> = (0..=h_max).map(|_| None).collect();

    let mut h = h_max;
    while h >= 1 {
        if inputs.iter().any(|m| m.log_height == h) {
            let mut hasher = MmcsGroupHasher::new(&stream, h)?;
            for m in inputs.iter().filter(|m| m.log_height == h) {
                let dev = stream.clone_htod(m.data)?;
                hasher.absorb_ext3_slabs(&stream, &dev, m.col_stride, m.num_parts)?;
                drop(dev);
            }
            group_digests[h as usize] = Some(hasher.finalize(&stream)?);
        }
        h -= 1;
    }

    let nodes = build_mmcs_tree_on_device(&stream, &group_digests)?;
    stream.clone_dtoh(&nodes)
}

pub fn commit_mixed_ext3_slabs_root(inputs: &[MmcsExt3SlabInput]) -> Result<[u8; 32]> {
    let nodes = commit_mixed_ext3_slabs_nodes(inputs)?;
    let mut root = [0u8; 32];
    root.copy_from_slice(&nodes[0..32]);
    Ok(root)
}

/// One ROW-MAJOR ext3 matrix feeding a mixed-height MMCS commit, in absorb
/// order: element `(row, col)` as 3 consecutive u64 at `(row*stride + col)*3`.
/// Columns `[col_start, col_end)` are committed. This is the batched aux LDE
/// layout (`expand_aux_lde_row_major`).
pub struct MmcsExt3RowMajorInput<'a> {
    pub data: &'a [u64],
    pub stride: u64,
    pub col_start: u64,
    pub col_end: u64,
    pub log_height: u64,
}

/// Build the whole mixed-height MMCS tree on the GPU from ROW-MAJOR ext3
/// matrices and return its root — the aux twin of [`commit_mixed_row_major_root`].
pub fn commit_mixed_ext3_row_major_nodes(inputs: &[MmcsExt3RowMajorInput]) -> Result<Vec<u8>> {
    let be = backend()?;
    let stream = be.next_stream();

    let h_max = inputs.iter().map(|m| m.log_height).max().unwrap_or(0);
    let mut group_digests: Vec<Option<CudaSlice<u8>>> = (0..=h_max).map(|_| None).collect();

    let mut h = h_max;
    while h >= 1 {
        if inputs.iter().any(|m| m.log_height == h) {
            let mut hasher = MmcsGroupHasher::new(&stream, h)?;
            for m in inputs.iter().filter(|m| m.log_height == h) {
                let dev = stream.clone_htod(m.data)?;
                hasher.absorb_ext3_row_major(&stream, &dev, m.stride, m.col_start, m.col_end)?;
                drop(dev);
            }
            group_digests[h as usize] = Some(hasher.finalize(&stream)?);
        }
        h -= 1;
    }

    let nodes = build_mmcs_tree_on_device(&stream, &group_digests)?;
    stream.clone_dtoh(&nodes)
}

pub fn commit_mixed_ext3_row_major_root(inputs: &[MmcsExt3RowMajorInput]) -> Result<[u8; 32]> {
    let nodes = commit_mixed_ext3_row_major_nodes(inputs)?;
    let mut root = [0u8; 32];
    root.copy_from_slice(&nodes[0..32]);
    Ok(root)
}

/// Persistent, streaming device commit for one mixed-height MMCS round — the
/// same per-height [`MmcsGroupHasher`] machinery the `commit_mixed_*_nodes`
/// helpers use, but kept ALIVE across the prover's round loop so each table's
/// LDE is absorbed the moment it is produced and freed immediately, instead of
/// every LDE being retained until one all-at-once commit. VRAM holds only the
/// per-leaf sponges (~204 B/leaf) plus one table's uploaded buffer at a time,
/// never `O(N)` LDEs — so the device commit runs under `RecomputeLde`, not only
/// `Retain` (which is what made a large epoch OOM: it retained every LDE just to
/// feed the commit).
///
/// Absorb matrices IN INPUT ORDER (the leaf concatenation binds it), one call
/// per matrix, using the layout method matching the round (row-major base for
/// main, ext3 row-major for aux, ext3 slabs for parts). Each absorb uploads the
/// host buffer, hashes it, and synchronizes, so the caller may free the host LDE
/// as soon as the call returns. [`Self::finish`] then finalizes every group and
/// climbs, returning the standard heap node array (feed to
/// `MixedMmcs::from_heap_nodes`) — byte-identical to the all-at-once commit
/// because it is the same kernels in the same order.
pub struct StreamingMixedMmcs {
    stream: Arc<CudaStream>,
    /// Indexed by `log_height`: the group's live sponges, created on first
    /// absorb of a matrix at that height.
    hashers: Vec<Option<MmcsGroupHasher>>,
}

impl StreamingMixedMmcs {
    /// `h_max` is the tallest `log_height` in the round (its group is the tree's
    /// base layer). Groups are created lazily as matrices arrive.
    pub fn new(h_max: u64) -> Result<Self> {
        let stream = backend()?.next_stream();
        Ok(Self {
            stream,
            hashers: (0..=h_max as usize).map(|_| None).collect(),
        })
    }

    /// The stream every absorb + [`Self::finish`] runs on. A caller feeding the
    /// device-buffer absorb paths (`absorb_*_dev`) should produce the resident
    /// LDE on THIS stream (or synchronize its producer stream first), so the
    /// absorb kernel is correctly ordered after the buffer is filled without a
    /// device-wide sync.
    pub fn stream(&self) -> Arc<CudaStream> {
        self.stream.clone()
    }

    fn group_mut(&mut self, log_height: u64) -> Result<&mut MmcsGroupHasher> {
        let idx = log_height as usize;
        assert!(
            idx < self.hashers.len(),
            "log_height {log_height} exceeds the h_max this StreamingMixedMmcs was built for"
        );
        if self.hashers[idx].is_none() {
            self.hashers[idx] = Some(MmcsGroupHasher::new(&self.stream, log_height)?);
        }
        Ok(self.hashers[idx].as_mut().expect("just populated"))
    }

    /// Absorb one row-major base-field matrix (main round). `data` is the host
    /// LDE (Goldilocks u64 words); columns `[col_start, col_end)` of `row_stride`
    /// are committed.
    pub fn absorb_row_major(
        &mut self,
        log_height: u64,
        data: &[u64],
        row_stride: u64,
        col_start: u64,
        col_end: u64,
    ) -> Result<()> {
        let stream = self.stream.clone();
        let dev = stream.clone_htod(data)?;
        self.group_mut(log_height)?
            .absorb_row_major(&stream, &dev, row_stride, col_start, col_end)?;
        // Defensive: ensure the H2D copy and the absorb that reads it have
        // completed before the caller frees `data` (the streaming residency
        // policy this type exists for).
        stream.synchronize()?;
        Ok(())
    }

    /// Absorb one ext3 row-major matrix (aux round). `data` is the host LDE with
    /// an element's 3 components consecutive.
    pub fn absorb_ext3_row_major(
        &mut self,
        log_height: u64,
        data: &[u64],
        stride: u64,
        col_start: u64,
        col_end: u64,
    ) -> Result<()> {
        let stream = self.stream.clone();
        let dev = stream.clone_htod(data)?;
        self.group_mut(log_height)?
            .absorb_ext3_row_major(&stream, &dev, stride, col_start, col_end)?;
        stream.synchronize()?;
        Ok(())
    }

    /// Absorb one ext3 column-major slab matrix (parts round). `parts` is the
    /// slab buffer, component `k` of column `c` at `(c*3 + k) * col_stride`.
    pub fn absorb_ext3_slabs(
        &mut self,
        log_height: u64,
        parts: &[u64],
        col_stride: u64,
        num_parts: u64,
    ) -> Result<()> {
        let stream = self.stream.clone();
        let dev = stream.clone_htod(parts)?;
        self.group_mut(log_height)?
            .absorb_ext3_slabs(&stream, &dev, col_stride, num_parts)?;
        stream.synchronize()?;
        Ok(())
    }

    // ---- Device-buffer absorb paths ------------------------------------------
    //
    // These take a DEVICE buffer already resident on the GPU (e.g. main's
    // `GpuLdeBase.buf`, an aux `GpuLdeExt3`, or resident composition parts) and
    // hash it in place — no `clone_htod`, no host `Vec`. This is the bridge that
    // lets the batched prover commit its resident LDE where it already lives,
    // instead of expanding on the CPU and uploading.
    //
    // Precondition (vs the host wrappers): `data` must be ready on
    // [`Self::stream`] before the call — produce the LDE on that stream, or
    // synchronize the producer first. Unlike the host wrappers these do NOT
    // synchronize afterward: the resident buffer is meant to stay alive across
    // the prover's later phases, and ordering into [`Self::finish`] is already
    // guaranteed because both run on `self.stream`. The caller must therefore
    // keep `data` alive until at least `finish` (its natural residency anyway).

    /// Absorb one COLUMN-MAJOR base-field matrix resident on device — main's
    /// `GpuLdeBase.buf` layout, element `(row, col)` at `col*col_stride + row`.
    /// Byte-identical leaf digests to [`Self::absorb_row_major`] over the same
    /// matrix; this is the resident-LDE bridge for the main round.
    pub fn absorb_col_major_dev(
        &mut self,
        log_height: u64,
        data: &CudaSlice<u64>,
        col_stride: u64,
        col_start: u64,
        col_end: u64,
    ) -> Result<()> {
        let stream = self.stream.clone();
        self.group_mut(log_height)?
            .absorb_col_major(&stream, data, col_stride, col_start, col_end)
    }

    /// Absorb one ROW-MAJOR base-field matrix resident on device (same layout as
    /// [`Self::absorb_row_major`], but the buffer already lives on the GPU).
    pub fn absorb_row_major_dev(
        &mut self,
        log_height: u64,
        data: &CudaSlice<u64>,
        row_stride: u64,
        col_start: u64,
        col_end: u64,
    ) -> Result<()> {
        let stream = self.stream.clone();
        self.group_mut(log_height)?
            .absorb_row_major(&stream, data, row_stride, col_start, col_end)
    }

    /// Absorb one ROW-MAJOR ext3 matrix resident on device (aux round) — an
    /// element's 3 components consecutive, `stride` elements per row.
    pub fn absorb_ext3_row_major_dev(
        &mut self,
        log_height: u64,
        data: &CudaSlice<u64>,
        stride: u64,
        col_start: u64,
        col_end: u64,
    ) -> Result<()> {
        let stream = self.stream.clone();
        self.group_mut(log_height)?
            .absorb_ext3_row_major(&stream, data, stride, col_start, col_end)
    }

    /// Absorb one COLUMN-MAJOR ext3 slab matrix resident on device (parts round)
    /// — component `k` of column `c` at `(c*3 + k) * col_stride`.
    pub fn absorb_ext3_slabs_dev(
        &mut self,
        log_height: u64,
        parts: &CudaSlice<u64>,
        col_stride: u64,
        num_parts: u64,
    ) -> Result<()> {
        let stream = self.stream.clone();
        self.group_mut(log_height)?
            .absorb_ext3_slabs(&stream, parts, col_stride, num_parts)
    }

    /// Finalize every absorbed group and climb, returning the standard heap node
    /// array (`(2L-1)*32` bytes for `L = 2^(h_max-1)`), host-side.
    pub fn finish(self) -> Result<Vec<u8>> {
        let Self { stream, hashers } = self;
        let mut group_digests: Vec<Option<CudaSlice<u8>>> =
            (0..hashers.len()).map(|_| None).collect();
        for (h, hasher) in hashers.into_iter().enumerate() {
            if let Some(hasher) = hasher {
                group_digests[h] = Some(hasher.finalize(&stream)?);
            }
        }
        let nodes = build_mmcs_tree_on_device(&stream, &group_digests)?;
        stream.clone_dtoh(&nodes)
    }
}
