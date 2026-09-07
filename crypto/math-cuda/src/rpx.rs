//! GPU RPX256 (XHash12) for Merkle commits — the leaf kernels, the parent/level
//! compressors, and the permutation probe that is the host's only handle on the
//! bare device permutation.
//!
//! Twin of [`crate::blake3`], launcher for launcher, so the two read against
//! each other; production dispatch reaches this module through
//! [`crate::DeviceHash::Rpx256`] from the fused LDE+commit pipelines
//! ([`crate::lde`]), the comp-poly tree builders (`stark::gpu_lde`) and the FRI
//! layer commits ([`crate::fri`]).
//!
//! # What a parent is
//!
//! `hash_new_parent(left, right)` is ONE permutation of `[left ‖ right ‖ 0⁴]`
//! with the digest read from lanes 0..4 — `algebraic_commit::parent`, which is
//! `Rpx256::merge` in miden's terms. The children are decoded from their node
//! bytes as `commitment_to_digest` does (four big-endian u64s) and the parent
//! is encoded back as `digest_to_commitment` does, so device nodes are the
//! host's bytes.
//!
//! # What a leaf is
//!
//! The rate-8 OVERWRITE duplex `algebraic_commit::sponge_leaf` over the felt
//! sequence the host leaf hashes — the same read pattern as the BLAKE3 kernel
//! each leaf kernel twins (`leaves_bit_reversed_grouped`), which is exactly the
//! sequence `felts_from_bytes` rebuilds from the leaf bytes, so the
//! `hash_bytes == hash_data` contract holds on device by construction.
//!
//! # Coverage
//!
//! All seven leaf kernels, both tree compressors and the wrapper twins are
//! here; the `launch_*` functions are what the dispatch sites call. The
//! permutation itself is pinned without a GPU by
//! `tests/host_kat/rpx_host_kat.cpp` (`make test-rpx-host-kat`); the device
//! build is pinned against the host by `prover/tests/rpx_device_parity.rs`.

use cudarc::driver::{CudaSlice, CudaStream, CudaViewMut, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use crate::Result;
use crate::device::{Backend, backend};
use crate::lde::pack_ext3_to_pinned_slabs;

/// Felts in one permutation state.
pub const STATE_FELTS: usize = 12;

/// Threads per block for the RPX kernels.
///
/// [`crate::merkle`]'s 128 rather than BLAKE3's 256: a thread carries a
/// twelve-lane u64 state plus the inverse S-box's live temporaries, a register
/// footprint closer to keccak's 25 u64 lanes than to BLAKE3's 32 u32 words, and
/// 128 is the Blackwell register-file limit that keccak already runs at. To be
/// re-measured with `-Xptxas -v` (phase-2 gate); this is the safe default.
const RPX_BLOCK_DIM: u32 = 128;

pub(crate) fn rpx_launch_cfg(num_threads: u64) -> LaunchConfig {
    debug_assert!(
        num_threads <= u32::MAX as u64,
        "rpx_launch_cfg: num_threads ({num_threads}) exceeds u32 grid range",
    );
    let grid = (num_threads as u32).div_ceil(RPX_BLOCK_DIM);
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (RPX_BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// RPX leaf hashing over a base-field column buffer. Twin of
/// [`crate::blake3::leaves_base`], argument for argument.
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
/// [`crate::blake3::leaves_ext3`].
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
    // Row-pair ext3 leaves reuse the comp-poly kernel, as the keccak and BLAKE3
    // paths do: hashing all ext3 columns of rows `2i`, `2i+1` is the same
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
    debug_assert!(num_rows >= 2, "rpx leaf kernel: num_rows must be >= 2");
    let be = backend()?;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    let cfg = rpx_launch_cfg(num_rows);
    unsafe {
        stream
            .launch_builder(&be.rpx_leaves_base_batched)
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
        "rpx row-pair leaf kernel: num_rows must be >= 2"
    );
    let be = backend()?;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    // One thread per leaf (= row pair).
    let cfg = rpx_launch_cfg(num_rows >> 1);
    unsafe {
        stream
            .launch_builder(&be.rpx_leaves_base_row_pair_batched)
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
    debug_assert!(num_rows >= 2, "rpx leaf kernel: num_rows must be >= 2");
    let be = backend()?;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    let cfg = rpx_launch_cfg(num_rows);
    unsafe {
        stream
            .launch_builder(&be.rpx_leaves_ext3_batched)
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
        "rpx row-pair leaf kernel: num_rows must be >= 2"
    );
    let be = backend()?;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    let cfg = rpx_launch_cfg(num_rows >> 1);
    unsafe {
        stream
            .launch_builder(&be.rpx_comp_poly_leaves_ext3)
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
/// `commit_bit_reversed(.., 2)`; twin of [`crate::blake3::leaves_base_row_major_row_pair`].
///
/// Returns `(num_rows / 2) * 32` hash bytes.
pub fn leaves_base_row_major_row_pair(data: &[u64], m: usize, num_rows: usize) -> Result<Vec<u8>> {
    leaves_row_major_row_pair_inner(data, m, 0, m, num_rows, false)
}

/// Column-range variant of [`leaves_base_row_major_row_pair`]: each leaf hashes
/// only columns `[col_start, col_end)` of the row pair, while `m` stays the full
/// row stride. Matches the CPU `commit_rows_bit_reversed_subset`.
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
    if ranged {
        launch_leaves_base_row_major_row_pair_range(
            stream.as_ref(),
            be,
            &data_dev,
            m_u64,
            col_start as u64,
            col_end as u64,
            num_rows_u64,
            log_num_rows,
            &mut out_dev.as_view_mut(),
        )?;
    } else {
        launch_leaves_base_row_major_row_pair(
            stream.as_ref(),
            be,
            &data_dev,
            m_u64,
            num_rows_u64,
            log_num_rows,
            &mut out_dev.as_view_mut(),
        )?;
    }
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Row-major ROW-PAIR leaf hashing under RPX: leaf `i` hashes the two
/// consecutive bit-reversed rows `reverse_index(2i)`, `reverse_index(2i+1)`
/// (each `m` lanes, read contiguously from the row-major `buf`), producing
/// `num_rows / 2` leaves. Device-buffer twin of the BLAKE3 launcher the fused
/// LDE pipeline dispatches against; matches the CPU `commit_bit_reversed(.., 2)`.
pub(crate) fn launch_leaves_base_row_major_row_pair(
    stream: &CudaStream,
    be: &Backend,
    buf: &CudaSlice<u64>,
    m: u64,
    num_rows: u64,
    log_num_rows: u64,
    leaves_out: &mut CudaViewMut<'_, u8>,
) -> Result<()> {
    // The kernel derives rows as `__brevll(2*tid + k) >> (64 - log_num_rows)`;
    // a 64-bit shift is UB at `log_num_rows == 0`, so require `num_rows >= 2`.
    debug_assert!(
        num_rows >= 2,
        "row-major row-pair rpx requires num_rows >= 2"
    );
    let cfg = rpx_launch_cfg(num_rows >> 1);
    unsafe {
        stream
            .launch_builder(&be.rpx_leaves_base_row_major_row_pair)
            .arg(buf)
            .arg(&m)
            .arg(&num_rows)
            .arg(&log_num_rows)
            .arg(leaves_out)
            .launch(cfg)?;
    }
    Ok(())
}

/// Column-range variant of [`launch_leaves_base_row_major_row_pair`]: leaves
/// hash only columns `[col_start, col_end)` of each bit-reversed row pair
/// (`m` stays the full row stride). Matches the CPU
/// `commit_rows_bit_reversed_subset`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_leaves_base_row_major_row_pair_range(
    stream: &CudaStream,
    be: &Backend,
    buf: &CudaSlice<u64>,
    m: u64,
    col_start: u64,
    col_end: u64,
    num_rows: u64,
    log_num_rows: u64,
    leaves_out: &mut CudaViewMut<'_, u8>,
) -> Result<()> {
    debug_assert!(
        num_rows >= 2,
        "row-major row-pair rpx requires num_rows >= 2"
    );
    debug_assert!(
        col_start < col_end && col_end <= m,
        "column range in bounds"
    );
    let cfg = rpx_launch_cfg(num_rows >> 1);
    unsafe {
        stream
            .launch_builder(&be.rpx_leaves_base_row_major_row_pair_range)
            .arg(buf)
            .arg(&m)
            .arg(&col_start)
            .arg(&col_end)
            .arg(&num_rows)
            .arg(&log_num_rows)
            .arg(leaves_out)
            .launch(cfg)?;
    }
    Ok(())
}

/// Composition-part leaf hashing under RPX: leaf `i` hashes the ext3
/// components of every part at the two bit-reversed rows `2i`, `2i+1`, read
/// from per-component slabs with stride `col_stride`. Device-buffer twin of
/// the BLAKE3 launch the comp-poly tree build dispatches against.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_comp_poly_leaves_ext3(
    stream: &CudaStream,
    be: &Backend,
    buf: &CudaSlice<u64>,
    col_stride: u64,
    num_parts: u64,
    num_rows: u64,
    log_num_rows: u64,
    leaves_out: &mut CudaViewMut<'_, u8>,
) -> Result<()> {
    debug_assert!(num_rows >= 2, "comp-poly rpx leaves require num_rows >= 2");
    let cfg = rpx_launch_cfg(num_rows >> 1);
    unsafe {
        stream
            .launch_builder(&be.rpx_comp_poly_leaves_ext3)
            .arg(buf)
            .arg(&col_stride)
            .arg(&num_parts)
            .arg(&num_rows)
            .arg(&log_num_rows)
            .arg(leaves_out)
            .launch(cfg)?;
    }
    Ok(())
}

/// FRI-layer leaf hashing under RPX: leaf `i` hashes the two consecutive ext3
/// evals `2i`, `2i+1` of an interleaved eval vector (six felts — one block).
/// Device-buffer twin of the BLAKE3 launch the FRI layer commit dispatches
/// against; the host is `AlgebraicPairBackend::hash_data`.
pub(crate) fn launch_fri_leaves_ext3(
    stream: &CudaStream,
    be: &Backend,
    evals: &CudaSlice<u64>,
    num_leaves: u64,
    leaves_out: &mut CudaViewMut<'_, u8>,
) -> Result<()> {
    let cfg = rpx_launch_cfg(num_leaves);
    unsafe {
        stream
            .launch_builder(&be.rpx_fri_leaves_ext3)
            .arg(evals)
            .arg(&num_leaves)
            .arg(leaves_out)
            .launch(cfg)?;
    }
    Ok(())
}

/// Walk the inner Merkle tree on device under RPX. `nodes_dev` already has the
/// `leaves_len` hashed leaves written into the tail; this fills in the inner
/// nodes bottom-up. Twin of [`crate::blake3::build_inner_tree_levels`], with
/// the same tail cutover: one single-block launch takes over once a level is no
/// wider than the block, where per-level launch overhead dominates the work.
pub(crate) fn build_inner_tree_levels(
    stream: &CudaStream,
    be: &Backend,
    nodes_dev: &mut CudaSlice<u8>,
    leaves_len: usize,
) -> Result<()> {
    const TAIL_MAX_PAIRS: u64 = RPX_BLOCK_DIM as u64;
    let mut level_begin: u64 = (leaves_len - 1) as u64;
    while level_begin != 0 {
        let new_begin = level_begin / 2;
        let n_pairs = level_begin - new_begin;
        if n_pairs <= TAIL_MAX_PAIRS {
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (RPX_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&be.rpx_merkle_tail)
                    .arg(&mut *nodes_dev)
                    .arg(&level_begin)
                    .launch(cfg)?;
            }
            return Ok(());
        }
        let cfg = rpx_launch_cfg(n_pairs);
        unsafe {
            stream
                .launch_builder(&be.rpx_merkle_level)
                .arg(&mut *nodes_dev)
                .arg(&new_begin)
                .arg(&n_pairs)
                .launch(cfg)?;
        }
        level_begin = new_begin;
    }
    Ok(())
}

/// Given `hashed_leaves` of length `leaves_len * 32`, build the full RPX
/// Merkle tree on device and return the `(2*leaves_len - 1) * 32`-byte node
/// buffer in the standard layout: `nodes[0..leaves_len - 1]` are inner nodes
/// (root at index 0) and `nodes[leaves_len - 1..]` are the leaves themselves.
///
/// Matches the CPU `crypto/crypto/src/merkle_tree/merkle.rs` construction, so
/// the result plugs into `MerkleTree::from_precomputed_nodes` the same way
/// [`crate::blake3::build_merkle_tree_on_device`]'s does.
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

/// Build the composition Merkle tree under RPX straight from a device-resident
/// slab buffer (`3*m` slabs of `lde_size` u64s, component `k` of part `c` at
/// `(c*3 + k) * lde_size` — the [`crate::lde::GpuLdeExt3`] layout). No host
/// staging and no H2D: the leaf kernel reads `buf` in place on `stream`.
///
/// Twin of [`crate::blake3::build_comp_poly_tree_from_slabs_dev`].
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

/// Build the composition Merkle tree under RPX from host-side interleaved ext3
/// parts, keeping the nodes device-resident so openings can gather paths on
/// device. `parts_interleaved` is `num_parts` slices, each `[a0,a1,a2,b0,b1,b2,…]`
/// of length `3*lde_size`. Leaves hash row pairs, so `leaves_len = lde_size / 2`.
///
/// Twin of [`crate::blake3::build_comp_poly_tree_from_evals_ext3_keep`], and it
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

/// Build a FRI-layer Merkle tree on device under RPX from an interleaved ext3
/// eval vector, returning the full host node buffer so tests can compare it byte
/// for byte against the CPU `AlgebraicPairBackend` tree. Each leaf hashes two
/// consecutive ext3 values; `num_leaves = evals.len() / 6`. Returns
/// `(2*num_leaves - 1) * 32` bytes in standard layout.
///
/// Twin of [`crate::blake3::build_fri_layer_tree_from_evals_ext3`], and like it
/// a parity harness rather than a production path: production folds and commits
/// through [`crate::fri::FriCommitState::fold_and_commit_layer`], which
/// dispatches to the same two kernels.
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
        launch_fri_leaves_ext3(
            stream.as_ref(),
            be,
            &evals_dev,
            num_leaves as u64,
            &mut leaves_view,
        )?;
    }

    build_inner_tree_levels(stream.as_ref(), be, &mut nodes_dev, num_leaves)?;

    let out = stream.clone_dtoh(&nodes_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Parity harness: run the device permutation over `states` and return each
/// output state, canonical.
///
/// Not a production path — the bare device permutation is otherwise
/// unreachable from host code, so without this there would be nothing to check
/// it against the host `Rpx256` (or the host-KAT's oracle tables) with before a
/// whole tree is built. Inputs may be raw `[0, 2^64)` storage.
pub fn permute_probe(states: &[[u64; STATE_FELTS]]) -> Result<Vec<[u64; STATE_FELTS]>> {
    if states.is_empty() {
        return Ok(Vec::new());
    }
    let n = states.len();
    let flat: Vec<u64> = states.iter().flatten().copied().collect();
    let be = backend()?;
    let stream = be.next_stream();
    let states_dev = stream.clone_htod(&flat)?;
    let mut out_dev = stream.alloc_zeros::<u64>(n * STATE_FELTS)?;
    let n_u64 = n as u64;
    let cfg = rpx_launch_cfg(n_u64);
    unsafe {
        stream
            .launch_builder(&be.rpx_permute_probe)
            .arg(&states_dev)
            .arg(&n_u64)
            .arg(&mut out_dev)
            .launch(cfg)?;
    }
    let flat_out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(flat_out
        .chunks_exact(STATE_FELTS)
        .map(|c| {
            let mut s = [0u64; STATE_FELTS];
            s.copy_from_slice(c);
            s
        })
        .collect())
}
