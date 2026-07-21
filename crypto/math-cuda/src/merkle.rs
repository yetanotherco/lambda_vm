//! GPU Keccak-256 leaf hashing for Merkle commits.
//!
//! Matches `FieldElementVectorBackend<F, Keccak256, 32>::hash_data` in
//! `crypto/crypto/src/merkle_tree/backends/field_element_vector.rs`, combined
//! with the `reverse_index` row read pattern used in
//! `commit_bit_reversed` at `crypto/stark/src/commitment.rs`.
//!
//! Caller supplies base-field column slabs already laid out as
//! `[col * col_stride + row]` (the same layout `coset_lde_batch_base_into`
//! writes to the pinned staging buffer). The kernel bit-reverses `row_idx`,
//! reads each column's canonical u64 at that row, byte-swaps it into a
//! Keccak lane, absorbs lane-by-lane, and squeezes 32 bytes per leaf.
//!
//! For ext3 columns the layout is `[col*3*col_stride + k*col_stride + row]`,
//! three base-field components per ext3 column, indexed by `k ∈ {0,1,2}`,
//! and the kernel reads three u64s per column in component order 0,1,2
//! to match `FieldElement::<Ext3>::write_bytes_be`.

use cudarc::driver::{CudaSlice, CudaStream, CudaViewMut, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use crate::Result;
use crate::device::{Backend, backend};
use crate::lde::pack_ext3_to_pinned_slabs;

/// Run GPU Keccak-256 leaf hashing on a base-field column buffer.
///
/// `columns` must hold `num_cols * col_stride` u64s with column `c`'s data
/// at `[c*col_stride .. c*col_stride + num_rows]`. `rows_per_leaf` selects the
/// leaf layout: `1` = one leaf per bit-reversed row (`num_rows` leaves), `2` =
/// one leaf per bit-reversed row pair `2i`,`2i+1` (`num_rows/2` leaves, the
/// trace-commit layout). Returns `(num_rows / rows_per_leaf) * 32` hash bytes.
pub fn keccak_leaves_base(
    columns: &[u64],
    col_stride: usize,
    num_cols: usize,
    num_rows: usize,
    rows_per_leaf: usize,
) -> Result<Vec<u8>> {
    assert!(num_rows.is_power_of_two());
    assert!(rows_per_leaf == 1 || rows_per_leaf == 2);
    assert!(
        num_rows >= rows_per_leaf,
        "num_rows must be at least rows_per_leaf"
    );
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
        launch_keccak_base_row_pair
    } else {
        launch_keccak_base
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

/// Ext3 variant. Columns interleaved as three base slabs per ext3 column.
/// `columns.len() >= num_cols * 3 * col_stride`. `rows_per_leaf` as in
/// [`keccak_leaves_base`].
pub fn keccak_leaves_ext3(
    columns: &[u64],
    col_stride: usize,
    num_cols: usize,
    num_rows: usize,
    rows_per_leaf: usize,
) -> Result<Vec<u8>> {
    assert!(num_rows.is_power_of_two());
    assert!(rows_per_leaf == 1 || rows_per_leaf == 2);
    assert!(
        num_rows >= rows_per_leaf,
        "num_rows must be at least rows_per_leaf"
    );
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
    let launch = if rows_per_leaf == 2 {
        launch_keccak_ext3_row_pair
    } else {
        launch_keccak_ext3
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

/// Block size for Keccak kernels. Per-thread register footprint is ~60 regs
/// (25-lane state + auxiliaries). The default 256 threads/block pushes the
/// block register file past the hardware limit on sm_120 (Blackwell). 128
/// keeps us inside the budget with some head-room.
const KECCAK_BLOCK_DIM: u32 = 128;

pub(crate) fn keccak_launch_cfg(num_rows: u64) -> LaunchConfig {
    debug_assert!(
        num_rows <= u32::MAX as u64,
        "keccak_launch_cfg: num_rows ({num_rows}) exceeds u32 grid range",
    );
    let grid = (num_rows as u32).div_ceil(KECCAK_BLOCK_DIM);
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (KECCAK_BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Walk the inner Merkle tree on device. `nodes_dev` already has the
/// `leaves_len` hashed leaves written into the tail; this loops
/// `log2(leaves_len)` times invoking `keccak_merkle_level` to fill in the
/// inner nodes from the bottom up. Mirrors the CPU `build(nodes, leaves_len)`
/// scan in `crypto/crypto/src/merkle_tree/merkle.rs`.
pub(crate) fn build_inner_tree_levels(
    stream: &CudaStream,
    be: &Backend,
    nodes_dev: &mut CudaSlice<u8>,
    leaves_len: usize,
) -> Result<()> {
    let mut level_begin: u64 = (leaves_len - 1) as u64;
    while level_begin != 0 {
        let new_begin = level_begin / 2;
        let n_pairs = level_begin - new_begin;
        let cfg = keccak_launch_cfg(n_pairs);
        unsafe {
            stream
                .launch_builder(&be.keccak_merkle_level)
                .arg(&mut *nodes_dev)
                .arg(&new_begin)
                .arg(&n_pairs)
                .launch(cfg)?;
        }
        level_begin = new_begin;
    }
    Ok(())
}

pub(crate) fn launch_keccak_base(
    stream: &CudaStream,
    cols_dev: &CudaSlice<u64>,
    col_stride: u64,
    num_cols: u64,
    num_rows: u64,
    out_dev: &mut CudaViewMut<'_, u8>,
) -> Result<()> {
    // The kernel computes `__brevll(tid) >> (64 - log_num_rows)`, which is UB
    // for `log_num_rows == 0` (single-row trees are degenerate anyway).
    debug_assert!(num_rows >= 2, "keccak leaf kernel: num_rows must be >= 2");
    let be = backend()?;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    let cfg = keccak_launch_cfg(num_rows);
    unsafe {
        stream
            .launch_builder(&be.keccak256_leaves_base_batched)
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

/// Row-pair base-field leaf hashing: leaf `i` hashes bit-reversed rows `2i`,
/// `2i+1` (one Merkle path per FRI query). Writes `num_rows/2` leaves of 32
/// bytes into `out_dev`. Base-field analog of the comp-poly ext3 path; matches
/// the CPU `keccak_leaves_row_pair_bit_reversed`.
pub(crate) fn launch_keccak_base_row_pair(
    stream: &CudaStream,
    cols_dev: &CudaSlice<u64>,
    col_stride: u64,
    num_cols: u64,
    num_rows: u64,
    out_dev: &mut CudaViewMut<'_, u8>,
) -> Result<()> {
    debug_assert!(
        num_rows >= 2,
        "keccak row-pair leaf kernel: num_rows must be >= 2"
    );
    let be = backend()?;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    // One thread per leaf (= row pair).
    let cfg = keccak_launch_cfg(num_rows >> 1);
    unsafe {
        stream
            .launch_builder(&be.keccak256_leaves_base_row_pair_batched)
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

/// Row-pair ext3 leaf hashing for the aux trace: reuses the comp-poly kernel
/// (`keccak_comp_poly_leaves_ext3`), which hashes bit-reversed rows `2i`, `2i+1`
/// across all ext3 columns. Writes `num_rows/2` leaves of 32 bytes.
pub(crate) fn launch_keccak_ext3_row_pair(
    stream: &CudaStream,
    cols_dev: &CudaSlice<u64>,
    col_stride: u64,
    num_cols: u64,
    num_rows: u64,
    out_dev: &mut CudaViewMut<'_, u8>,
) -> Result<()> {
    debug_assert!(
        num_rows >= 2,
        "keccak row-pair leaf kernel: num_rows must be >= 2"
    );
    let be = backend()?;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    let cfg = keccak_launch_cfg(num_rows >> 1);
    unsafe {
        stream
            .launch_builder(&be.keccak_comp_poly_leaves_ext3)
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

/// Given `hashed_leaves` of length `leaves_len * 32`, build the full Merkle
/// tree on device and return the complete node buffer `(2*leaves_len - 1) *
/// 32` bytes in the standard layout:
///
///   `nodes[0..leaves_len - 1]` are inner nodes (root at index 0), and
///   `nodes[leaves_len - 1..]` are the leaves themselves.
///
/// Matches the CPU `crypto/crypto/src/merkle_tree/merkle.rs` construction so
/// the resulting `nodes` Vec plugs straight into `MerkleTree { root, nodes }`
/// for downstream proof generation.
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

    // Allocate the full node buffer without zero-fill. We overwrite the
    // leaf half via H2D immediately, and every inner node is written by the
    // pair-hash kernel below.
    // SAFETY: every byte is written before it is read: leaves are filled by
    // the H2D below; inner nodes are filled by the level loop that follows.
    let mut nodes_dev = unsafe { stream.alloc::<u8>(total_nodes * 32) }?;
    let leaves_offset_bytes = (leaves_len - 1) * 32;
    // SAFETY: target slice `nodes_dev[leaves_offset_bytes..]` has exactly
    // `leaves_len * 32 == hashed_leaves.len()` bytes capacity.
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

/// Gather Merkle authentication paths on device for `positions` (leaf indices)
/// against the resident tree `nodes_dev` (standard layout, `2*leaves_len-1`
/// nodes of 32 bytes). Returns `positions.len() * depth * 32` bytes, where
/// `depth = log2(leaves_len)`. Query `q`'s path is `[q*depth*32 ..
/// (q+1)*depth*32]`, each 32 byte node a sibling from leaf to root. These are
/// the same nodes the CPU `MerkleTree::get_proof_by_pos` collects. Runs on the
/// caller's `stream` (pass the table's session stream).
pub fn gather_merkle_paths_dev(
    nodes_dev: &CudaSlice<u8>,
    leaves_len: usize,
    positions: &[u32],
    stream: &Arc<CudaStream>,
) -> Result<Vec<u8>> {
    let num_queries = positions.len();
    if num_queries == 0 {
        return Ok(Vec::new());
    }
    assert!(
        leaves_len.is_power_of_two() && leaves_len >= 2,
        "leaves_len must be a power of two >= 2"
    );
    let depth = leaves_len.trailing_zeros() as usize;
    // Guard the kernel's device reads: a position past leaves_len would walk
    // off the node buffer. Positions are valid by construction; this catches a
    // caller bug before it becomes an out of bounds device read.
    assert!(
        positions.iter().all(|&p| (p as usize) < leaves_len),
        "gather_merkle_paths_dev: leaf position >= leaves_len"
    );
    let be = backend()?;

    let pos_dev = stream.clone_htod(positions)?;
    // SAFETY: every byte of `out` is written by the kernel below (one 32-byte
    // node per (query, level)) before the D2H reads it back.
    let mut out = unsafe { stream.alloc::<u8>(num_queries * depth * 32) }?;

    let grid = (num_queries as u32).div_ceil(KECCAK_BLOCK_DIM);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (KECCAK_BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    };
    let num_queries_u32 = num_queries as u32;
    let leaves_len_u64 = leaves_len as u64;
    let depth_u32 = depth as u32;
    unsafe {
        stream
            .launch_builder(&be.merkle_gather_paths)
            .arg(nodes_dev)
            .arg(&pos_dev)
            .arg(&num_queries_u32)
            .arg(&leaves_len_u64)
            .arg(&depth_u32)
            .arg(&mut out)
            .launch(cfg)?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// Build the composition Merkle tree on device. `parts_interleaved` is
/// `num_parts` slices, each an ext3 LDE column interleaved as
/// `[a0,a1,a2, b0,b1,b2, ...]` of length `3*lde_size`. Leaves hash row pairs, so
/// `num_leaves = lde_size / 2`. Returns the device node buffer, the leaf count,
/// and the stream it was built on. Used by the device keep wrapper below.
fn build_comp_poly_tree_nodes_dev(
    parts_interleaved: &[&[u64]],
) -> Result<(CudaSlice<u8>, usize, Arc<CudaStream>)> {
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
    let num_leaves = lde_size / 2;

    let be = backend()?;
    let stream = be.next_stream();
    let staging_slot = be.pinned_staging();

    // Stage: de-interleave each part into 3 base slabs in pinned memory.
    let mb = 3 * m;
    let mut staging = staging_slot.lock().unwrap();
    staging.ensure_capacity(mb * lde_size, &be.ctx)?;
    let pinned = unsafe { staging.as_mut_slice(mb * lde_size) };

    pack_ext3_to_pinned_slabs(parts_interleaved, pinned, lde_size);

    // H2D the de-interleaved parts, then release the staging lock (the kernels
    // below read the device `buf`, not `pinned`). Synchronize first so the
    // async H2D has consumed `pinned` before it is freed/reused.
    let mut buf = stream.alloc_zeros::<u64>(mb * lde_size)?;
    stream.memcpy_htod(&pinned[..mb * lde_size], &mut buf)?;
    stream.synchronize()?;
    crate::stagebytes::add_comp_merkle_h2d(mb * lde_size * 8);
    drop(staging);

    let nodes_dev = comp_poly_leaves_and_tree(&buf, m, lde_size, &stream)?;
    Ok((nodes_dev, num_leaves, stream))
}

/// Row-pair Keccak leaves + inner tree levels over a device-resident comp-poly
/// LDE buffer in slab layout `buf[(part*3 + k) * lde_size + r]` (`m` parts).
/// Shared by the host-evals path (which packs + H2Ds into `buf` first) and the
/// device-buf path (which reuses an already-resident `GpuLdeExt3` buffer, so no
/// re-upload). Returns the tight node buffer (`(2*num_leaves - 1) * 32` bytes,
/// root at 0, leaves at the tail).
fn comp_poly_leaves_and_tree(
    buf: &CudaSlice<u64>,
    m: usize,
    lde_size: usize,
    stream: &Arc<CudaStream>,
) -> Result<CudaSlice<u8>> {
    let be = backend()?;
    let num_leaves = lde_size / 2;
    let tight_total_nodes = 2 * num_leaves - 1;
    let mut nodes_dev = unsafe { stream.alloc::<u8>(tight_total_nodes * 32) }?;
    let leaves_offset_bytes = (num_leaves - 1) * 32;
    {
        let mut leaves_view =
            nodes_dev.slice_mut(leaves_offset_bytes..leaves_offset_bytes + num_leaves * 32);
        let col_stride_u64 = lde_size as u64;
        let num_parts_u64 = m as u64;
        let num_rows_u64 = lde_size as u64;
        let log_num_rows = lde_size.trailing_zeros() as u64;
        let cfg = keccak_launch_cfg(num_leaves as u64);
        unsafe {
            stream
                .launch_builder(&be.keccak_comp_poly_leaves_ext3)
                .arg(buf)
                .arg(&col_stride_u64)
                .arg(&num_parts_u64)
                .arg(&num_rows_u64)
                .arg(&log_num_rows)
                .arg(&mut leaves_view)
                .launch(cfg)?;
        }
    }
    build_inner_tree_levels(stream.as_ref(), be, &mut nodes_dev, num_leaves)?;
    Ok(nodes_dev)
}

/// Build the comp-poly Merkle tree directly from an already-resident device LDE
/// buffer (slab layout `buf[(part*3 + k) * lde_size + r]`, `m` parts), skipping
/// the host pack + H2D that [`build_comp_poly_tree_from_evals_ext3_keep`] does.
/// Used when the extended composition LDE is already on device (the `_keep`
/// extend handle). Returns the tree kept resident, root copied to host.
pub fn build_comp_poly_tree_from_dev_buf(
    buf: &CudaSlice<u64>,
    m: usize,
    lde_size: usize,
) -> Result<crate::lde::GpuMerkleTree> {
    assert!(lde_size.is_power_of_two() && lde_size >= 2);
    assert!(
        buf.len() >= 3 * m * lde_size,
        "device buf must hold at least 3*m*lde_size u64s"
    );
    let be = backend()?;
    let stream = be.next_stream();
    let num_leaves = lde_size / 2;
    let nodes_dev = comp_poly_leaves_and_tree(buf, m, lde_size, &stream)?;
    let mut root = [0u8; 32];
    stream.memcpy_dtoh(&nodes_dev.slice(0..32), &mut root)?;
    stream.synchronize()?;
    Ok(crate::lde::GpuMerkleTree {
        nodes: Arc::new(nodes_dev),
        leaves_len: num_leaves,
        root,
    })
}

/// Build the comp poly Merkle tree on device and keep the nodes resident
/// (returned as a [`crate::lde::GpuMerkleTree`] with its root), so R4
/// composition openings gather paths on device instead of copying the whole
/// tree to host. `leaves_len = lde_size / 2` (row pair leaves).
pub fn build_comp_poly_tree_from_evals_ext3_keep(
    parts_interleaved: &[&[u64]],
) -> Result<crate::lde::GpuMerkleTree> {
    let (nodes_dev, num_leaves, stream) = build_comp_poly_tree_nodes_dev(parts_interleaved)?;
    let mut root = [0u8; 32];
    stream.memcpy_dtoh(&nodes_dev.slice(0..32), &mut root)?;
    stream.synchronize()?;
    Ok(crate::lde::GpuMerkleTree {
        nodes: Arc::new(nodes_dev),
        leaves_len: num_leaves,
        root,
    })
}

/// Parity harness for [`build_comp_poly_tree_from_dev_buf`]: pack `num_parts`
/// interleaved ext3 eval columns into slab layout, upload to a device buffer,
/// and build the tree via the device-buf path. Lets the parity suite exercise
/// the resident-handle Merkle commit (Step B) directly, for any `m`, without a
/// prior LDE. `parts_interleaved[i]` is `3*lde_size` u64s.
#[doc(hidden)]
pub fn build_comp_poly_tree_from_host_parts_via_dev_buf(
    parts_interleaved: &[&[u64]],
) -> Result<crate::lde::GpuMerkleTree> {
    assert!(!parts_interleaved.is_empty());
    let m = parts_interleaved.len();
    let lde_size = parts_interleaved[0].len() / 3;
    for p in parts_interleaved.iter() {
        assert_eq!(p.len(), 3 * lde_size);
    }
    let be = backend()?;
    let stream = be.next_stream();
    let mut host = vec![0u64; 3 * m * lde_size];
    pack_ext3_to_pinned_slabs(parts_interleaved, &mut host, lde_size);
    let buf = stream.clone_htod(&host)?;
    stream.synchronize()?;
    build_comp_poly_tree_from_dev_buf(&buf, m, lde_size)
}

/// Test-only parity harness: build a FRI layer Merkle tree on device from an
/// interleaved ext3 eval vector and return the full host node buffer so tests
/// can compare it byte for byte against the CPU. Production folds and commits
/// via [`crate::fri::FriLayer::fold_and_commit_layer`]. Each leaf hashes two
/// consecutive ext3 values; `num_leaves = evals.len() / 6`. Returns the
/// `(2*num_leaves - 1) * 32`-byte node buffer in standard layout.
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
    let mut nodes_dev = unsafe { stream.alloc::<u8>(tight_total_nodes * 32) }?;

    // Leaf kernel: num_leaves threads, one leaf each.
    let leaves_offset_bytes = (num_leaves - 1) * 32;
    {
        let mut leaves_view =
            nodes_dev.slice_mut(leaves_offset_bytes..leaves_offset_bytes + num_leaves * 32);
        let num_leaves_u64 = num_leaves as u64;
        let cfg = keccak_launch_cfg(num_leaves as u64);
        unsafe {
            stream
                .launch_builder(&be.keccak_fri_leaves_ext3)
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

pub(crate) fn launch_keccak_ext3(
    stream: &CudaStream,
    cols_dev: &CudaSlice<u64>,
    col_stride: u64,
    num_cols: u64,
    num_rows: u64,
    out_dev: &mut CudaViewMut<'_, u8>,
) -> Result<()> {
    // The kernel computes `__brevll(tid) >> (64 - log_num_rows)`, which is UB
    // for `log_num_rows == 0` (single-row trees are degenerate anyway).
    debug_assert!(num_rows >= 2, "keccak leaf kernel: num_rows must be >= 2");
    let be = backend()?;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    let cfg = keccak_launch_cfg(num_rows);
    unsafe {
        stream
            .launch_builder(&be.keccak256_leaves_ext3_batched)
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
