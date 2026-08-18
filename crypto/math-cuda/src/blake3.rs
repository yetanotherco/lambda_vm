//! GPU BLAKE3 for Merkle commits — the leaf kernels, the parent/level
//! compressors, and the parity-harness handles on the device compression
//! function, byte serialization and chain construction.
//!
//! Twin of [`crate::merkle`]'s keccak path, kernel for kernel, so the two read
//! against each other. Keccak stays the prover's default hash: nothing in the
//! production dispatch reaches this module yet.
//!
//! # What a parent is
//!
//! `hash_new_parent(left, right)` is one BLAKE3 compression over the 64 bytes of
//! the two child digests: `h = IV`, `t = 0`, `block_len = 64`, `flags =
//! CHUNK_START|CHUNK_END|ROOT`, digest = the low 8 output words little-endian.
//! That is `hash_bytes(left ‖ right)`, which is what `hash_new_parent` already
//! is for every host backend (`hash_new_parent_bytes`,
//! `crypto/crypto/src/merkle_tree/backends/field_element_vector.rs:74`), and at
//! 7 rounds it is literally `blake3::hash(left ‖ right)`.
//!
//! A parent is therefore construction-independent: its message is a single
//! block, and over a single block every candidate chaining construction agrees
//! bit-for-bit. Only multi-block messages — leaves — depend on the construction,
//! which is why the parent compressor could land before it was settled and the
//! leaf kernels could not.
//!
//! # What a leaf is
//!
//! A leaf's bytes are unchanged from the keccak path — `leaves_bit_reversed_grouped`
//! (`crypto/stark/src/commitment.rs:55`) serializes each element in canonical
//! big-endian and concatenates, and only the hash over those bytes moves. The
//! hash is `Blake3Chain` (PA-PLAN §1.7): standard BLAKE3 restricted to a single
//! chunk that never ends, host implementation at
//! `crypto/crypto/src/hash/blake3/chain.rs`.
//!
//! ⚠ That construction is a DRAFT pending ratification of forks F1-F3
//! (PA-PLAN §1.7.3), implemented here as the working default by standing
//! decision.
//!
//! # What is missing, and why
//!
//! Nothing on the kernel side: all seven leaf kernels, both tree compressors and
//! the six wrapper twins are here. What has NOT happened is production dispatch —
//! `stark::config::StarkHash` still requires `KeccakTreeBackend` under `cuda`
//! (`config.rs:116-122`), so no prover path reaches this module. Retiring that
//! bound is PA-PLAN's Stage 6, not track G.

use cudarc::driver::{CudaSlice, CudaStream, CudaViewMut, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use crate::Result;
use crate::device::{Backend, backend};
use crate::lde::pack_ext3_to_pinned_slabs;

/// Threads per block for the BLAKE3 kernels.
///
/// Wider than [`crate::merkle`]'s 128 because the register footprint is a third
/// of keccak's: 16 working-state words + 16 message words + the output, all u32,
/// against keccak's 25 u64 lanes plus a 25-lane scratch. The 128 there is a
/// Blackwell register-file limit, not a shape this path shares.
const BLAKE3_BLOCK_DIM: u32 = 256;

pub(crate) fn blake3_launch_cfg(num_threads: u64) -> LaunchConfig {
    debug_assert!(
        num_threads <= u32::MAX as u64,
        "blake3_launch_cfg: num_threads ({num_threads}) exceeds u32 grid range",
    );
    let grid = (num_threads as u32).div_ceil(BLAKE3_BLOCK_DIM);
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLAKE3_BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// BLAKE3 leaf hashing over a base-field column buffer. Twin of
/// [`crate::merkle::keccak_leaves_base`], argument for argument.
///
/// `columns` must hold `num_cols * col_stride` u64s with column `c`'s data at
/// `[c*col_stride .. c*col_stride + num_rows]`. `rows_per_leaf` selects the leaf
/// layout: `1` = one leaf per bit-reversed row (`num_rows` leaves), `2` = one
/// leaf per bit-reversed row pair (`num_rows/2` leaves, the trace-commit
/// layout). Returns `(num_rows / rows_per_leaf) * 32` hash bytes.
pub fn leaves_base(
    columns: &[u64],
    col_stride: usize,
    num_cols: usize,
    num_rows: usize,
    rows_per_leaf: usize,
) -> Result<Vec<u8>> {
    assert!(num_rows.is_power_of_two());
    assert!(rows_per_leaf == 1 || rows_per_leaf == 2);
    assert!(
        num_rows >= 2,
        "num_rows must be at least 2 for bit-reversed GPU leaf hashing"
    );
    assert!(
        col_stride >= num_rows,
        "col_stride must be >= num_rows to keep per-column reads in-bounds"
    );
    let total = num_cols
        .checked_mul(col_stride)
        .expect("num_cols * col_stride overflows usize");
    assert!(columns.len() >= total);
    let be = backend()?;
    let stream = be.next_stream();
    let cols_dev = stream.clone_htod(&columns[..total])?;
    let mut out_dev = stream.alloc_zeros::<u8>((num_rows / rows_per_leaf) * 32)?;
    let launch = if rows_per_leaf == 2 {
        launch_leaves_base_row_pair
    } else {
        launch_leaves_base
    };
    launch(
        stream.as_ref(),
        &cols_dev,
        col_stride as u64,
        num_cols as u64,
        num_rows as u64,
        &mut out_dev.as_view_mut(),
    )?;
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Ext3 variant of [`leaves_base`]: columns arrive as three base slabs per ext3
/// column, so `columns.len() >= num_cols * 3 * col_stride`. Twin of
/// [`crate::merkle::keccak_leaves_ext3`].
pub fn leaves_ext3(
    columns: &[u64],
    col_stride: usize,
    num_cols: usize,
    num_rows: usize,
    rows_per_leaf: usize,
) -> Result<Vec<u8>> {
    assert!(num_rows.is_power_of_two());
    assert!(rows_per_leaf == 1 || rows_per_leaf == 2);
    assert!(
        num_rows >= 2,
        "num_rows must be at least 2 for bit-reversed GPU leaf hashing"
    );
    assert!(
        col_stride >= num_rows,
        "col_stride must be >= num_rows to keep per-column reads in-bounds"
    );
    let total = num_cols
        .checked_mul(3)
        .and_then(|v| v.checked_mul(col_stride))
        .expect("num_cols * 3 * col_stride overflows usize");
    assert!(columns.len() >= total);
    let be = backend()?;
    let stream = be.next_stream();
    let cols_dev = stream.clone_htod(&columns[..total])?;
    let mut out_dev = stream.alloc_zeros::<u8>((num_rows / rows_per_leaf) * 32)?;
    // Row-pair ext3 leaves reuse the comp-poly kernel, exactly as the keccak
    // path does: hashing all ext3 columns of rows `2i`, `2i+1` is the same
    // traversal whether the columns are called "aux trace" or "parts".
    let launch = if rows_per_leaf == 2 {
        launch_ext3_row_pair
    } else {
        launch_leaves_ext3
    };
    launch(
        stream.as_ref(),
        &cols_dev,
        col_stride as u64,
        num_cols as u64,
        num_rows as u64,
        &mut out_dev.as_view_mut(),
    )?;
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}

pub(crate) fn launch_leaves_base(
    stream: &CudaStream,
    cols_dev: &CudaSlice<u64>,
    col_stride: u64,
    num_cols: u64,
    num_rows: u64,
    out_dev: &mut CudaViewMut<'_, u8>,
) -> Result<()> {
    // The kernel computes `__brevll(tid) >> (64 - log_num_rows)`, which is UB
    // for `log_num_rows == 0` (single-row trees are degenerate anyway).
    debug_assert!(num_rows >= 2, "blake3 leaf kernel: num_rows must be >= 2");
    let be = backend()?;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    let cfg = blake3_launch_cfg(num_rows);
    unsafe {
        stream
            .launch_builder(&be.blake3_leaves_base_batched)
            .arg(cols_dev)
            .arg(&col_stride)
            .arg(&num_cols)
            .arg(&num_rows)
            .arg(&log_num_rows)
            .arg(out_dev)
            .launch(cfg)?;
    }
    Ok(())
}

pub(crate) fn launch_leaves_base_row_pair(
    stream: &CudaStream,
    cols_dev: &CudaSlice<u64>,
    col_stride: u64,
    num_cols: u64,
    num_rows: u64,
    out_dev: &mut CudaViewMut<'_, u8>,
) -> Result<()> {
    debug_assert!(
        num_rows >= 2,
        "blake3 row-pair leaf kernel: num_rows must be >= 2"
    );
    let be = backend()?;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    // One thread per leaf (= row pair).
    let cfg = blake3_launch_cfg(num_rows >> 1);
    unsafe {
        stream
            .launch_builder(&be.blake3_leaves_base_row_pair_batched)
            .arg(cols_dev)
            .arg(&col_stride)
            .arg(&num_cols)
            .arg(&num_rows)
            .arg(&log_num_rows)
            .arg(out_dev)
            .launch(cfg)?;
    }
    Ok(())
}

pub(crate) fn launch_leaves_ext3(
    stream: &CudaStream,
    cols_dev: &CudaSlice<u64>,
    col_stride: u64,
    num_cols: u64,
    num_rows: u64,
    out_dev: &mut CudaViewMut<'_, u8>,
) -> Result<()> {
    debug_assert!(num_rows >= 2, "blake3 leaf kernel: num_rows must be >= 2");
    let be = backend()?;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    let cfg = blake3_launch_cfg(num_rows);
    unsafe {
        stream
            .launch_builder(&be.blake3_leaves_ext3_batched)
            .arg(cols_dev)
            .arg(&col_stride)
            .arg(&num_cols)
            .arg(&num_rows)
            .arg(&log_num_rows)
            .arg(out_dev)
            .launch(cfg)?;
    }
    Ok(())
}

pub(crate) fn launch_ext3_row_pair(
    stream: &CudaStream,
    cols_dev: &CudaSlice<u64>,
    col_stride: u64,
    num_cols: u64,
    num_rows: u64,
    out_dev: &mut CudaViewMut<'_, u8>,
) -> Result<()> {
    debug_assert!(
        num_rows >= 2,
        "blake3 row-pair leaf kernel: num_rows must be >= 2"
    );
    let be = backend()?;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    let cfg = blake3_launch_cfg(num_rows >> 1);
    unsafe {
        stream
            .launch_builder(&be.blake3_comp_poly_leaves_ext3)
            .arg(cols_dev)
            .arg(&col_stride)
            .arg(&num_cols)
            .arg(&num_rows)
            .arg(&log_num_rows)
            .arg(out_dev)
            .launch(cfg)?;
    }
    Ok(())
}

/// Row-major row-pair leaf hashing: leaf `i` hashes the two consecutive
/// bit-reversed rows `reverse_index(2i)`, `reverse_index(2i+1)`, each `m` lanes
/// read contiguously from the row-major `data`. Matches the CPU
/// `commit_bit_reversed(.., 2)`; twin of the keccak launcher in
/// [`crate::lde`].
///
/// Returns `(num_rows / 2) * 32` hash bytes. Public because the blake3 path has
/// no production caller yet — the parity tests are what reach this kernel, and
/// the keccak twin's private launcher is called from the LDE pipeline instead.
pub fn leaves_base_row_major_row_pair(data: &[u64], m: usize, num_rows: usize) -> Result<Vec<u8>> {
    leaves_row_major_row_pair_inner(data, m, 0, m, num_rows, false)
}

/// Column-range variant of [`leaves_base_row_major_row_pair`]: each leaf hashes
/// only columns `[col_start, col_end)` of the row pair, while `m` stays the full
/// row stride. Matches the CPU `commit_rows_bit_reversed_subset`, which is how
/// preprocessed tables commit their precomputed and multiplicity column ranges
/// to separate Merkle trees over one row-major LDE.
pub fn leaves_base_row_major_row_pair_range(
    data: &[u64],
    m: usize,
    col_start: usize,
    col_end: usize,
    num_rows: usize,
) -> Result<Vec<u8>> {
    leaves_row_major_row_pair_inner(data, m, col_start, col_end, num_rows, true)
}

fn leaves_row_major_row_pair_inner(
    data: &[u64],
    m: usize,
    col_start: usize,
    col_end: usize,
    num_rows: usize,
    ranged: bool,
) -> Result<Vec<u8>> {
    assert!(num_rows.is_power_of_two());
    assert!(num_rows >= 2, "num_rows must be at least 2");
    assert!(
        col_start < col_end && col_end <= m,
        "column range in bounds"
    );
    let total = num_rows
        .checked_mul(m)
        .expect("num_rows * m overflows usize");
    assert!(data.len() >= total);

    let be = backend()?;
    let stream = be.next_stream();
    let data_dev = stream.clone_htod(&data[..total])?;
    let mut out_dev = stream.alloc_zeros::<u8>((num_rows / 2) * 32)?;

    let m_u64 = m as u64;
    let num_rows_u64 = num_rows as u64;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    let cfg = blake3_launch_cfg((num_rows / 2) as u64);
    unsafe {
        if ranged {
            let cs = col_start as u64;
            let ce = col_end as u64;
            stream
                .launch_builder(&be.blake3_leaves_base_row_major_row_pair_range)
                .arg(&data_dev)
                .arg(&m_u64)
                .arg(&cs)
                .arg(&ce)
                .arg(&num_rows_u64)
                .arg(&log_num_rows)
                .arg(&mut out_dev.as_view_mut())
                .launch(cfg)?;
        } else {
            stream
                .launch_builder(&be.blake3_leaves_base_row_major_row_pair)
                .arg(&data_dev)
                .arg(&m_u64)
                .arg(&num_rows_u64)
                .arg(&log_num_rows)
                .arg(&mut out_dev.as_view_mut())
                .launch(cfg)?;
        }
    }
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Walk the inner Merkle tree on device under BLAKE3. `nodes_dev` already has
/// the `leaves_len` hashed leaves written into the tail; this fills in the inner
/// nodes bottom-up. Twin of [`crate::merkle::build_inner_tree_levels`], and the
/// tail cutover has the same rationale: one single-block launch takes over once a
/// level is no wider than the block, where per-level launch overhead dominates
/// the work and the tail's grid-striding adds no serialization over the launches
/// it replaces.
pub(crate) fn build_inner_tree_levels(
    stream: &CudaStream,
    be: &Backend,
    nodes_dev: &mut CudaSlice<u8>,
    leaves_len: usize,
) -> Result<()> {
    const TAIL_MAX_PAIRS: u64 = BLAKE3_BLOCK_DIM as u64;
    let mut level_begin: u64 = (leaves_len - 1) as u64;
    while level_begin != 0 {
        let new_begin = level_begin / 2;
        let n_pairs = level_begin - new_begin;
        if n_pairs <= TAIL_MAX_PAIRS {
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (BLAKE3_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&be.blake3_merkle_tail)
                    .arg(&mut *nodes_dev)
                    .arg(&level_begin)
                    .launch(cfg)?;
            }
            return Ok(());
        }
        let cfg = blake3_launch_cfg(n_pairs);
        unsafe {
            stream
                .launch_builder(&be.blake3_merkle_level)
                .arg(&mut *nodes_dev)
                .arg(&new_begin)
                .arg(&n_pairs)
                .launch(cfg)?;
        }
        level_begin = new_begin;
    }
    Ok(())
}

/// Given `hashed_leaves` of length `leaves_len * 32`, build the full BLAKE3
/// Merkle tree on device and return the `(2*leaves_len - 1) * 32`-byte node
/// buffer in the standard layout: `nodes[0..leaves_len - 1]` are inner nodes
/// (root at index 0) and `nodes[leaves_len - 1..]` are the leaves themselves.
///
/// Matches the CPU `crypto/crypto/src/merkle_tree/merkle.rs` construction, so
/// the result plugs into `MerkleTree::from_precomputed_nodes` the same way
/// [`crate::merkle::build_merkle_tree_on_device`]'s does.
///
/// `leaves_len` must be a power of two and >= 2.
pub fn build_merkle_tree_on_device(hashed_leaves: &[u8]) -> Result<Vec<u8>> {
    assert!(hashed_leaves.len().is_multiple_of(32));
    let leaves_len = hashed_leaves.len() / 32;
    assert!(leaves_len >= 2, "tree needs at least two leaves");
    assert!(
        leaves_len.is_power_of_two(),
        "leaves_len must be a power of two"
    );

    let total_nodes = 2 * leaves_len - 1;
    let be = backend()?;
    let stream = be.next_stream();

    // SAFETY: every byte is written before it is read — leaves by the H2D
    // below, inner nodes by the level walk that follows.
    let mut nodes_dev = unsafe { stream.alloc::<u8>(total_nodes * 32) }?;
    let leaves_offset_bytes = (leaves_len - 1) * 32;
    {
        let mut slice =
            nodes_dev.slice_mut(leaves_offset_bytes..leaves_offset_bytes + hashed_leaves.len());
        stream.memcpy_htod(hashed_leaves, &mut slice)?;
    }

    build_inner_tree_levels(stream.as_ref(), be, &mut nodes_dev, leaves_len)?;

    let out = stream.clone_dtoh(&nodes_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Build the composition Merkle tree under BLAKE3 straight from a
/// device-resident slab buffer (`3*m` slabs of `lde_size` u64s, component `k` of
/// part `c` at `(c*3 + k) * lde_size` — the [`crate::lde::GpuLdeExt3`] layout).
/// No host staging and no H2D: the leaf kernel reads `buf` in place on `stream`.
///
/// Twin of [`crate::merkle::build_comp_poly_tree_from_slabs_dev`].
pub fn build_comp_poly_tree_from_slabs_dev(
    stream: &Arc<CudaStream>,
    buf: &CudaSlice<u64>,
    m: usize,
    lde_size: usize,
) -> Result<crate::lde::GpuMerkleTree> {
    assert!(m > 0);
    assert!(lde_size.is_power_of_two() && lde_size >= 2);
    assert_eq!(buf.len(), 3 * m * lde_size, "slab buffer shape");
    let num_leaves = lde_size / 2;
    let tight_total_nodes = 2 * num_leaves - 1;
    let be = backend()?;

    // SAFETY: every byte is written before it is read — leaves by the kernel
    // below, inner nodes by the level walk after it.
    let mut nodes_dev = unsafe { stream.alloc::<u8>(tight_total_nodes * 32) }?;
    let leaves_offset_bytes = (num_leaves - 1) * 32;
    {
        let mut leaves_view =
            nodes_dev.slice_mut(leaves_offset_bytes..leaves_offset_bytes + num_leaves * 32);
        launch_ext3_row_pair(
            stream.as_ref(),
            buf,
            lde_size as u64,
            m as u64,
            lde_size as u64,
            &mut leaves_view,
        )?;
    }
    build_inner_tree_levels(stream.as_ref(), be, &mut nodes_dev, num_leaves)?;
    let mut root = [0u8; 32];
    stream.memcpy_dtoh(&nodes_dev.slice(0..32), &mut root)?;
    stream.synchronize()?;
    Ok(crate::lde::GpuMerkleTree {
        nodes: Arc::new(nodes_dev),
        leaves_len: num_leaves,
        root,
    })
}

/// Build the composition Merkle tree under BLAKE3 from host-side interleaved
/// ext3 parts, keeping the nodes device-resident so openings can gather paths on
/// device. `parts_interleaved` is `num_parts` slices, each `[a0,a1,a2,b0,b1,b2,…]`
/// of length `3*lde_size`. Leaves hash row pairs, so `leaves_len = lde_size / 2`.
///
/// Twin of [`crate::merkle::build_comp_poly_tree_from_evals_ext3_keep`], and it
/// stages through the same pinned de-interleave buffer for the same reason.
pub fn build_comp_poly_tree_from_evals_ext3_keep(
    parts_interleaved: &[&[u64]],
) -> Result<crate::lde::GpuMerkleTree> {
    assert!(!parts_interleaved.is_empty());
    let m = parts_interleaved.len();
    let ext3_elems = parts_interleaved[0].len() / 3;
    assert_eq!(
        parts_interleaved[0].len(),
        3 * ext3_elems,
        "ext3 buffer length must be 3 * lde_size"
    );
    for p in parts_interleaved.iter() {
        assert_eq!(p.len(), 3 * ext3_elems);
    }
    let lde_size = ext3_elems;
    assert!(lde_size.is_power_of_two() && lde_size >= 2);

    let be = backend()?;
    let stream = be.next_stream();
    let staging_slot = be.pinned_staging();

    // Stage: de-interleave each part into 3 base slabs in pinned memory.
    let mb = 3 * m;
    let mut staging = staging_slot.lock().unwrap();
    staging.ensure_capacity(mb * lde_size, &be.ctx)?;
    let pinned = unsafe { staging.as_mut_slice(mb * lde_size) };

    pack_ext3_to_pinned_slabs(parts_interleaved, pinned, lde_size);

    // H2D the de-interleaved parts, then release the staging lock: the tree
    // build reads the device `buf`, not `pinned`. Synchronize first so the async
    // H2D has consumed `pinned` before it can be freed or reused.
    let mut buf = stream.alloc_zeros::<u64>(mb * lde_size)?;
    stream.memcpy_htod(&pinned[..mb * lde_size], &mut buf)?;
    stream.synchronize()?;
    drop(staging);

    build_comp_poly_tree_from_slabs_dev(&stream, &buf, m, lde_size)
}

/// Build a FRI-layer Merkle tree on device under BLAKE3 from an interleaved ext3
/// eval vector, returning the full host node buffer so tests can compare it byte
/// for byte against the CPU. Each leaf hashes two consecutive ext3 values;
/// `num_leaves = evals.len() / 6`. Returns `(2*num_leaves - 1) * 32` bytes in
/// standard layout.
///
/// Twin of [`crate::merkle::build_fri_layer_tree_from_evals_ext3`], and like it a
/// parity harness rather than a production path: production folds and commits
/// through [`crate::fri::FriLayer::fold_and_commit_layer`].
pub fn build_fri_layer_tree_from_evals_ext3(evals: &[u64]) -> Result<Vec<u8>> {
    assert!(
        evals.len().is_multiple_of(6),
        "evals must hold whole pair-leaves"
    );
    let num_evals = evals.len() / 3;
    let num_leaves = num_evals / 2;
    assert!(num_leaves.is_power_of_two() && num_leaves >= 2);
    let tight_total_nodes = 2 * num_leaves - 1;

    let be = backend()?;
    let stream = be.next_stream();

    let evals_dev = stream.clone_htod(evals)?;
    // SAFETY: leaves are written by the kernel below, inner nodes by the level
    // walk after it, before either is read.
    let mut nodes_dev = unsafe { stream.alloc::<u8>(tight_total_nodes * 32) }?;

    let leaves_offset_bytes = (num_leaves - 1) * 32;
    {
        let mut leaves_view =
            nodes_dev.slice_mut(leaves_offset_bytes..leaves_offset_bytes + num_leaves * 32);
        let num_leaves_u64 = num_leaves as u64;
        let cfg = blake3_launch_cfg(num_leaves as u64);
        unsafe {
            stream
                .launch_builder(&be.blake3_fri_leaves_ext3)
                .arg(&evals_dev)
                .arg(&num_leaves_u64)
                .arg(&mut leaves_view)
                .launch(cfg)?;
        }
    }

    build_inner_tree_levels(stream.as_ref(), be, &mut nodes_dev, num_leaves)?;

    let out = stream.clone_dtoh(&nodes_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// One compression's inputs, in the argument order of the host reference
/// `blake3_compress_rounds(h, m, t, block_len, flags, rounds)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompressInput {
    pub h: [u32; 8],
    pub m: [u32; 16],
    pub t: u64,
    pub block_len: u32,
    pub flags: u32,
}

/// Which round count [`compress_probe`] should run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeRounds {
    /// 6 — the internal variant.
    Six,
    /// 7 — standard BLAKE3, where the `blake3` crate is an external anchor.
    Seven,
    /// Whatever the cubin's production kernels are compiled for. The only way to
    /// observe from host code which of the two `blake3_merkle_level` uses.
    CompiledIn,
}

/// Parity harness: run the device compression function over `inputs` and return
/// each full 16-word output.
///
/// Not a production path — the device compression is otherwise unreachable from
/// host code, so without this there would be nothing to check it against the
/// host reference with.
pub fn compress_probe(inputs: &[CompressInput], rounds: ProbeRounds) -> Result<Vec<[u32; 16]>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let n = inputs.len();
    let mut h = Vec::with_capacity(n * 8);
    let mut m = Vec::with_capacity(n * 16);
    let mut t = Vec::with_capacity(n);
    let mut block_len = Vec::with_capacity(n);
    let mut flags = Vec::with_capacity(n);
    for i in inputs {
        h.extend_from_slice(&i.h);
        m.extend_from_slice(&i.m);
        t.push(i.t);
        block_len.push(i.block_len);
        flags.push(i.flags);
    }

    let be = backend()?;
    let stream = be.next_stream();
    let h_dev = stream.clone_htod(&h)?;
    let m_dev = stream.clone_htod(&m)?;
    let t_dev = stream.clone_htod(&t)?;
    let bl_dev = stream.clone_htod(&block_len)?;
    let fl_dev = stream.clone_htod(&flags)?;
    let mut out_dev = stream.alloc_zeros::<u32>(n * 16)?;

    let kernel = match rounds {
        ProbeRounds::Six => &be.blake3_compress_probe_6r,
        ProbeRounds::Seven => &be.blake3_compress_probe_7r,
        ProbeRounds::CompiledIn => &be.blake3_compress_probe_default,
    };
    let n_u64 = n as u64;
    let cfg = blake3_launch_cfg(n_u64);
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(&h_dev)
            .arg(&m_dev)
            .arg(&t_dev)
            .arg(&bl_dev)
            .arg(&fl_dev)
            .arg(&n_u64)
            .arg(&mut out_dev)
            .launch(cfg)?;
    }
    let flat = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(flat
        .chunks_exact(16)
        .map(|c| {
            let mut w = [0u32; 16];
            w.copy_from_slice(c);
            w
        })
        .collect())
}

/// The round count `kernels/blake3.cu` was compiled for.
///
/// The host tree's round count and this one are separate crates' features, so
/// nothing forces them equal; a mismatch would be a GPU tree committing under a
/// different hash than the CPU one, with no symptom short of a failing verify.
/// Reading it back makes that assertable.
pub fn device_rounds() -> Result<u32> {
    let be = backend()?;
    let stream = be.next_stream();
    let mut out_dev = stream.alloc_zeros::<u32>(1)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.blake3_rounds_probe)
            .arg(&mut out_dev)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out[0])
}

/// Parity harness: the BLAKE3 message words each of `vals` serializes to — two
/// per element, the byte-reverse of its canonical value's high then low half.
///
/// This is the serialization the leaf kernels share with the CPU commit path
/// (`leaves_bit_reversed_grouped`, `crypto/stark/src/commitment.rs:55`), isolated
/// from any hashing: canonicalisation, big-endian element bytes, little-endian
/// word packing.
pub fn serialize_felts(vals: &[u64]) -> Result<Vec<u32>> {
    if vals.is_empty() {
        return Ok(Vec::new());
    }
    let be = backend()?;
    let stream = be.next_stream();
    let vals_dev = stream.clone_htod(vals)?;
    let mut out_dev = stream.alloc_zeros::<u32>(vals.len() * 2)?;
    let n_u64 = vals.len() as u64;
    let cfg = blake3_launch_cfg(n_u64);
    unsafe {
        stream
            .launch_builder(&be.blake3_serialize_felts_probe)
            .arg(&vals_dev)
            .arg(&n_u64)
            .arg(&mut out_dev)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Parity harness: `words` streamed through the device `Blake3Chain`, returning
/// the 32-byte digest.
///
/// ★ This is what lets the device be asserted against the COMMITTED KAT TABLE
/// (`crypto::hash::blake3::chain::CHAIN_KAT_6ROUND`) rather than only against
/// the host implementation — the difference risk R13 is about. The KAT digests
/// were produced by a Python oracle, so a device digest matching them is checked
/// against an artifact this tree did not compute.
///
/// Word-granular because that is all the device ever hashes: production messages
/// are whole numbers of 8-byte field elements. KAT lengths that are not
/// multiples of 4 are unreachable from device code by construction and are
/// covered by the host tests instead.
pub fn chain_probe(words: &[u32]) -> Result<[u8; 32]> {
    let be = backend()?;
    let stream = be.next_stream();
    // The empty message is a legitimate input (one compression, `block_len = 0`),
    // so an empty slice must still reach the kernel. `clone_htod` of an empty
    // slice is not portable, so allocate a one-word buffer and pass `n = 0`.
    let words_dev = if words.is_empty() {
        stream.alloc_zeros::<u32>(1)?
    } else {
        stream.clone_htod(words)?
    };
    let mut out_dev = stream.alloc_zeros::<u8>(32)?;
    let n_words = words.len() as u64;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.blake3_chain_probe)
            .arg(&words_dev)
            .arg(&n_words)
            .arg(&mut out_dev)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&out);
    Ok(digest)
}

/// Parity harness: `vals` streamed through the device block builder, returning
/// the `ceil(2*len/16)` completed 64-byte blocks as 16 words each, tail block
/// zero-padded.
///
/// Exercises the block framing on the code path a leaf kernel will use — one
/// thread streaming a whole leaf — with the compression sink replaced by a copy
/// out. Small inputs only; it is single-threaded by design.
pub fn blocks_of_felts(vals: &[u64]) -> Result<Vec<u32>> {
    if vals.is_empty() {
        return Ok(Vec::new());
    }
    let n_words = vals.len() * 2;
    let n_blocks = n_words.div_ceil(16);
    let be = backend()?;
    let stream = be.next_stream();
    let vals_dev = stream.clone_htod(vals)?;
    let mut out_dev = stream.alloc_zeros::<u32>(n_blocks * 16)?;
    let n_u64 = vals.len() as u64;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&be.blake3_blocks_of_felts_probe)
            .arg(&vals_dev)
            .arg(&n_u64)
            .arg(&mut out_dev)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}
