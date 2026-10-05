//! ZisK's Poseidon1 on the STARK's device commit paths: the
//! [`crate::DeviceHash::Poseidon1`] arm of every leaf, tree, path and grind
//! dispatch (`p1/*` exploration branch; nothing reaches it unless a
//! configuration names `CommitmentHash::Poseidon1`).
//!
//! The kernels are the production section of `kernels/p1w16.cu` (`p1s_*`),
//! loaded on first use from that cubin, never by
//! [`crate::device::Backend::init`], so a process that commits under another
//! hash never loads them.
//!
//! # Nodes and trees
//!
//! A node is the host's commitment bytes (four canonical felts, each eight
//! big-endian bytes), as under [`crate::rpx`]. A tree is 4-ary in the host's
//! arity-4 layout (`crypto::merkle_tree::utils::level_offsets4`): levels
//! top-down, root at node 0, leaves last, each level `⌈below / 4⌉` nodes, a
//! short group padded with the zero digest when its parent is hashed. So a
//! device node buffer is what `MerkleTree::<P1BatchBackend>::from_precomputed_nodes`
//! takes, and a path is three siblings per level ([`gather_paths_dev`]).
//!
//! # Leaves
//!
//! ZisK's leaf hash (`poseidon1_stark::linear_hash`) over the felt sequence
//! the host leaf hashes, in the read patterns of [`crate::rpx`]'s kernels.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, CudaViewMut, LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::Ptx;

use crate::Result;
use crate::device::backend;

const P1W16_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/p1w16.cubin"));

/// Children per node.
pub const ARITY: usize = 4;

/// Threads per block for the leaf and level kernels (a 16-lane state: the
/// RPX kernels' 128).
const BLOCK_DIM: u32 = 128;

/// Threads of the one-block tail ([`TAIL_MAX_PARENTS`] parents a pass); at
/// ~140 registers a thread, 1,024 would not fit one multiprocessor.
const TAIL_BLOCK_DIM: u32 = 256;

/// The widest level the one-block tail takes over: below it a per-level
/// launch costs more than the level's work.
pub const TAIL_MAX_PARENTS: u64 = TAIL_BLOCK_DIM as u64;

/// The production kernels.
pub struct Kernels {
    _module: Arc<CudaModule>,
    pub leaves_cols_row: CudaFunction,
    pub leaves_cols_pair: CudaFunction,
    pub leaves_rm_pair: CudaFunction,
    pub leaves_rm_row: CudaFunction,
    pub fri_group_leaves: CudaFunction,
    pub merkle_level4: CudaFunction,
    pub merkle_tail4: CudaFunction,
    pub gather_paths4: CudaFunction,
    pub grind_w8: CudaFunction,
}

/// Load the kernels once.
pub fn kernels() -> Result<&'static Kernels> {
    static KERNELS: OnceLock<Kernels> = OnceLock::new();
    if let Some(k) = KERNELS.get() {
        return Ok(k);
    }
    let be = backend()?;
    let module = be.ctx.load_module(Ptx::from_binary(P1W16_CUBIN.to_vec()))?;
    let k = Kernels {
        leaves_cols_row: module.load_function("p1s_leaves_cols_row")?,
        leaves_cols_pair: module.load_function("p1s_leaves_cols_pair")?,
        leaves_rm_pair: module.load_function("p1s_leaves_rm_pair")?,
        leaves_rm_row: module.load_function("p1s_leaves_rm_row")?,
        fri_group_leaves: module.load_function("p1s_fri_group_leaves")?,
        merkle_level4: module.load_function("p1s_merkle_level4")?,
        merkle_tail4: module.load_function("p1s_merkle_tail4")?,
        gather_paths4: module.load_function("p1s_gather_paths4")?,
        grind_w8: module.load_function("p1s_grind_w8")?,
        _module: module,
    };
    Ok(KERNELS.get_or_init(|| k))
}

pub(crate) fn launch_cfg(threads: u64) -> LaunchConfig {
    debug_assert!(threads <= u32::MAX as u64, "p1 launch: {threads} threads");
    LaunchConfig {
        grid_dim: ((threads as u32).div_ceil(BLOCK_DIM).max(1), 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ===========================================================================
// The arity-4 layout
// ===========================================================================

/// Level sizes over `leaves ≥ 1` leaves, leaves first, root (1) last
/// (`crypto::merkle_tree::utils::level_sizes4`).
pub fn level_sizes(leaves: usize) -> Vec<usize> {
    let mut sizes = vec![leaves];
    while let Some(&n) = sizes.last().filter(|&&n| n > 1) {
        sizes.push(n.div_ceil(ARITY));
    }
    sizes
}

/// Stored nodes of a tree over `leaves` leaves.
pub fn tree_nodes(leaves: usize) -> usize {
    level_sizes(leaves).iter().sum()
}

/// Levels above the leaves: the number of three-sibling groups on a path.
pub fn depth(leaves: usize) -> usize {
    level_sizes(leaves).len() - 1
}

/// How many nodes the top `levels` levels hold (the root's level first): the
/// node-buffer prefix those levels occupy.
pub fn top_levels_nodes(leaves: usize, levels: usize) -> usize {
    level_sizes(leaves).iter().rev().take(levels).sum()
}

// ===========================================================================
// Leaves
// ===========================================================================

/// Leaves of a column-major matrix (`num_cols` columns of `num_rows` rows at
/// `cols`, column stride `col_stride`): one bit-reversed row per leaf
/// (`rows_per_leaf = 1`) or a row pair (`2`), into `leaves`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_leaves_cols_ptr(
    stream: &CudaStream,
    cols: sys::CUdeviceptr,
    col_stride: u64,
    num_cols: u64,
    num_rows: u64,
    rows_per_leaf: usize,
    leaves: sys::CUdeviceptr,
) -> Result<()> {
    assert!(rows_per_leaf == 1 || rows_per_leaf == 2);
    assert!(num_rows >= 2 && num_rows.is_power_of_two());
    let k = kernels()?;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    let (kernel, threads) = if rows_per_leaf == 2 {
        (&k.leaves_cols_pair, num_rows / 2)
    } else {
        (&k.leaves_cols_row, num_rows)
    };
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(&cols)
            .arg(&col_stride)
            .arg(&num_cols)
            .arg(&num_rows)
            .arg(&log_num_rows)
            .arg(&leaves)
            .launch(launch_cfg(threads))?;
    }
    Ok(())
}

/// [`launch_leaves_cols_ptr`] over a buffer and a leaf view.
pub(crate) fn launch_leaves_cols(
    stream: &CudaStream,
    cols: &CudaSlice<u64>,
    col_stride: u64,
    num_cols: u64,
    num_rows: u64,
    rows_per_leaf: usize,
    leaves: &mut CudaViewMut<'_, u8>,
) -> Result<()> {
    assert!(rows_per_leaf == 1 || rows_per_leaf == 2);
    assert!(num_rows >= 2 && num_rows.is_power_of_two());
    let k = kernels()?;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    let (kernel, threads) = if rows_per_leaf == 2 {
        (&k.leaves_cols_pair, num_rows / 2)
    } else {
        (&k.leaves_cols_row, num_rows)
    };
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(cols)
            .arg(&col_stride)
            .arg(&num_cols)
            .arg(&num_rows)
            .arg(&log_num_rows)
            .arg(leaves)
            .launch(launch_cfg(threads))?;
    }
    Ok(())
}

/// Leaves of a row-major matrix (`num_rows` rows of stride `m`), columns
/// `[col_start, col_end)`, `rows_per_leaf` bit-reversed rows a leaf.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_leaves_row_major(
    stream: &CudaStream,
    buf: &CudaSlice<u64>,
    m: u64,
    col_start: u64,
    col_end: u64,
    num_rows: u64,
    rows_per_leaf: usize,
    leaves: &mut CudaViewMut<'_, u8>,
) -> Result<()> {
    assert!(rows_per_leaf == 1 || rows_per_leaf == 2);
    assert!(num_rows >= 2 && num_rows.is_power_of_two());
    assert!(
        col_start < col_end && col_end <= m,
        "column range in bounds"
    );
    let k = kernels()?;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    let (kernel, threads) = if rows_per_leaf == 2 {
        (&k.leaves_rm_pair, num_rows / 2)
    } else {
        (&k.leaves_rm_row, num_rows)
    };
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(buf)
            .arg(&m)
            .arg(&col_start)
            .arg(&col_end)
            .arg(&num_rows)
            .arg(&log_num_rows)
            .arg(leaves)
            .launch(launch_cfg(threads))?;
    }
    Ok(())
}

/// FRI leaves of `group` consecutive ext3 values each from an interleaved eval
/// vector (`group = 2`: the pair leaf).
pub(crate) fn launch_fri_group_leaves(
    stream: &CudaStream,
    evals: &CudaSlice<u64>,
    num_leaves: u64,
    group: u64,
    leaves: &mut CudaViewMut<'_, u8>,
) -> Result<()> {
    let k = kernels()?;
    unsafe {
        stream
            .launch_builder(&k.fri_group_leaves)
            .arg(evals)
            .arg(&num_leaves)
            .arg(&group)
            .arg(leaves)
            .launch(launch_cfg(num_leaves))?;
    }
    Ok(())
}

// ===========================================================================
// Trees
// ===========================================================================

/// Fill every inner level of a tree whose `leaves_len` leaf digests sit at
/// the tail of `nodes_dev` (a [`tree_nodes`]`(leaves_len)`-node buffer):
/// one launch per level while a level has more than [`TAIL_MAX_PARENTS`]
/// parents, then one block for the rest.
pub fn build_inner_tree_levels(
    stream: &CudaStream,
    nodes_dev: &mut CudaSlice<u8>,
    leaves_len: usize,
) -> Result<()> {
    let k = kernels()?;
    let total = tree_nodes(leaves_len) as u64;
    assert!(
        nodes_dev.len() as u64 >= total * 32,
        "p1 tree: a node buffer of {} bytes holds no {leaves_len}-leaf tree",
        nodes_dev.len()
    );
    let mut child_off = total - leaves_len as u64;
    let mut n_children = leaves_len as u64;
    while n_children > 1 {
        let n_parents = n_children.div_ceil(ARITY as u64);
        if n_parents <= TAIL_MAX_PARENTS {
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (TAIL_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&k.merkle_tail4)
                    .arg(&mut *nodes_dev)
                    .arg(&child_off)
                    .arg(&n_children)
                    .launch(cfg)?;
            }
            return Ok(());
        }
        let parent_off = child_off - n_parents;
        unsafe {
            stream
                .launch_builder(&k.merkle_level4)
                .arg(&mut *nodes_dev)
                .arg(&child_off)
                .arg(&n_children)
                .arg(&parent_off)
                .arg(&n_parents)
                .launch(launch_cfg(n_parents))?;
        }
        child_off = parent_off;
        n_children = n_parents;
    }
    Ok(())
}

/// Parity harness: the full tree over `hashed_leaves` (32 bytes each), as the
/// host node buffer.
pub fn build_merkle_tree_on_device(hashed_leaves: &[u8]) -> Result<Vec<u8>> {
    assert!(hashed_leaves.len().is_multiple_of(32));
    let leaves_len = hashed_leaves.len() / 32;
    assert!(leaves_len >= 1);
    let total = tree_nodes(leaves_len);
    let be = backend()?;
    let stream = be.next_stream();
    // SAFETY: the leaves are written by the copy below, every inner node by
    // the level walk, before anything reads them.
    let mut nodes_dev = unsafe { stream.alloc::<u8>(total * 32) }?;
    {
        let off = (total - leaves_len) * 32;
        let mut slice = nodes_dev.slice_mut(off..off + hashed_leaves.len());
        stream.memcpy_htod(hashed_leaves, &mut slice)?;
    }
    build_inner_tree_levels(stream.as_ref(), &mut nodes_dev, leaves_len)?;
    let out = stream.clone_dtoh(&nodes_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// The composition tree straight from a resident slab buffer (`3·m` slabs of
/// `lde_size`, component `k` of part `c` at `(c·3 + k)·lde_size`),
/// `rows_per_leaf` rows a leaf. Twin of
/// [`crate::rpx::build_comp_poly_tree_from_slabs_dev_rpl`].
pub fn build_comp_poly_tree_from_slabs_dev_rpl(
    stream: &Arc<CudaStream>,
    buf: &CudaSlice<u64>,
    m: usize,
    lde_size: usize,
    rows_per_leaf: usize,
) -> Result<crate::lde::GpuMerkleTree> {
    assert!(rows_per_leaf == 1 || rows_per_leaf == 2);
    #[cfg(feature = "test-faults")]
    crate::faults::check_sticky(&crate::faults::FAULT_COMP_TREE_STICKY)?;
    assert!(m > 0);
    assert!(lde_size.is_power_of_two() && lde_size >= 2);
    assert_eq!(buf.len(), 3 * m * lde_size, "slab buffer shape");
    let num_leaves = lde_size / rows_per_leaf;
    let total = tree_nodes(num_leaves);
    // SAFETY: the leaf kernel writes the leaves, the level walk the rest.
    let mut nodes_dev = unsafe { stream.alloc::<u8>(total * 32) }?;
    {
        let off = (total - num_leaves) * 32;
        let mut leaves = nodes_dev.slice_mut(off..off + num_leaves * 32);
        launch_leaves_cols(
            stream.as_ref(),
            buf,
            lde_size as u64,
            3 * m as u64,
            lde_size as u64,
            rows_per_leaf,
            &mut leaves,
        )?;
    }
    build_inner_tree_levels(stream.as_ref(), &mut nodes_dev, num_leaves)?;
    let mut root = [0u8; 32];
    stream.memcpy_dtoh(&nodes_dev.slice(0..32), &mut root)?;
    stream.synchronize()?;
    Ok(crate::lde::GpuMerkleTree {
        nodes: Arc::new(nodes_dev),
        leaves_len: num_leaves,
        arity: ARITY,
        root,
    })
}

/// The composition tree from host interleaved ext3 parts, kept resident. Twin
/// of [`crate::rpx::build_comp_poly_tree_from_evals_ext3_keep_rpl`].
pub fn build_comp_poly_tree_from_evals_ext3_keep_rpl(
    parts_interleaved: &[&[u64]],
    rows_per_leaf: usize,
) -> Result<crate::lde::GpuMerkleTree> {
    #[cfg(feature = "test-faults")]
    crate::faults::check_sticky(&crate::faults::FAULT_COMP_TREE_STICKY)?;
    assert!(!parts_interleaved.is_empty());
    let m = parts_interleaved.len();
    let lde_size = parts_interleaved[0].len() / 3;
    for p in parts_interleaved {
        assert_eq!(
            p.len(),
            3 * lde_size,
            "ext3 buffer length must be 3 * lde_size"
        );
    }
    assert!(lde_size.is_power_of_two() && lde_size >= 2);
    let be = backend()?;
    let stream = be.next_stream();
    let staging_slot = be.pinned_staging();
    let mb = 3 * m;
    let mut staging = staging_slot.lock().unwrap();
    staging.ensure_capacity(mb * lde_size, &be.ctx)?;
    let pinned = unsafe { staging.as_mut_slice(mb * lde_size) };
    crate::lde::pack_ext3_to_pinned_slabs(parts_interleaved, pinned, lde_size);
    let mut buf = stream.alloc_zeros::<u64>(mb * lde_size)?;
    stream.memcpy_htod(&pinned[..mb * lde_size], &mut buf)?;
    stream.synchronize()?;
    drop(staging);
    build_comp_poly_tree_from_slabs_dev_rpl(&stream, &buf, m, lde_size, rows_per_leaf)
}

/// Parity harness: a FRI layer tree over an interleaved ext3 eval vector,
/// `group` values a leaf (2 = the pair layer), as the host node buffer.
pub fn build_fri_tree_from_evals_ext3(evals: &[u64], group: usize) -> Result<Vec<u8>> {
    assert!(evals.len().is_multiple_of(3 * group));
    let num_leaves = evals.len() / (3 * group);
    assert!(num_leaves >= 1);
    let total = tree_nodes(num_leaves);
    let be = backend()?;
    let stream = be.next_stream();
    let evals_dev = stream.clone_htod(evals)?;
    // SAFETY: as in `build_merkle_tree_on_device`.
    let mut nodes_dev = unsafe { stream.alloc::<u8>(total * 32) }?;
    {
        let off = (total - num_leaves) * 32;
        let mut leaves = nodes_dev.slice_mut(off..off + num_leaves * 32);
        launch_fri_group_leaves(
            stream.as_ref(),
            &evals_dev,
            num_leaves as u64,
            group as u64,
            &mut leaves,
        )?;
    }
    build_inner_tree_levels(stream.as_ref(), &mut nodes_dev, num_leaves)?;
    let out = stream.clone_dtoh(&nodes_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Parity harness: column-major leaves over a host matrix (`num_cols` columns
/// of `num_rows`, column-major), `rows_per_leaf` rows a leaf.
pub fn leaves_cols(
    columns: &[u64],
    num_cols: usize,
    num_rows: usize,
    rows_per_leaf: usize,
) -> Result<Vec<u8>> {
    assert_eq!(columns.len(), num_cols * num_rows);
    let be = backend()?;
    let stream = be.next_stream();
    let cols_dev = stream.clone_htod(columns)?;
    let mut out = stream.alloc_zeros::<u8>(num_rows / rows_per_leaf * 32)?;
    launch_leaves_cols(
        stream.as_ref(),
        &cols_dev,
        num_rows as u64,
        num_cols as u64,
        num_rows as u64,
        rows_per_leaf,
        &mut out.as_view_mut(),
    )?;
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

// ===========================================================================
// Paths and grinding
// ===========================================================================

/// Authentication paths for `positions` against a resident tree of
/// `leaves_len` leaves: `positions.len() · 3 · depth · 32` bytes, query `q`'s
/// path at `q · 3 · depth · 32`, the siblings of each level from the leaves
/// up in child order — the host `get_proof_by_pos` at arity 4.
pub fn gather_paths_dev(
    nodes_dev: &CudaSlice<u8>,
    leaves_len: usize,
    positions: &[u32],
    stream: &Arc<CudaStream>,
) -> Result<Vec<u8>> {
    let nq = positions.len();
    if nq == 0 {
        return Ok(Vec::new());
    }
    assert!(
        positions.iter().all(|&p| (p as usize) < leaves_len),
        "p1 gather_paths_dev: leaf position >= leaves_len"
    );
    let total = tree_nodes(leaves_len);
    assert!(
        nodes_dev.len() >= total * 32,
        "p1 gather: node buffer too short"
    );
    let depth4 = depth(leaves_len);
    let path_bytes = 3 * depth4 * 32;
    if path_bytes == 0 {
        return Ok(Vec::new());
    }
    let k = kernels()?;
    let be = backend()?;
    let pos_dev = stream.clone_htod(positions)?;
    // SAFETY: the kernel writes every (query, level, sibling) slot.
    let mut out = unsafe { stream.alloc::<u8>(nq * path_bytes) }?;
    let nq_u32 = nq as u32;
    let leaves_u64 = leaves_len as u64;
    let total_u64 = total as u64;
    let depth_u32 = depth4 as u32;
    unsafe {
        stream
            .launch_builder(&k.gather_paths4)
            .arg(nodes_dev)
            .arg(&pos_dev)
            .arg(&nq_u32)
            .arg(&leaves_u64)
            .arg(&total_u64)
            .arg(&depth_u32)
            .arg(&mut out)
            .launch(launch_cfg(nq as u64))?;
    }
    let pending =
        crate::device::async_dtoh_via(stream, be.pinned_hashes(), &be.ctx, &out, out.len())?;
    let mut host = vec![0u8; out.len()];
    pending.wait_into_bytes(&mut host)?;
    Ok(host)
}

/// The grind kernel and its block size, for [`crate::grinding`]'s range walk.
pub(crate) fn grind_kernel() -> Result<(&'static CudaFunction, u32)> {
    Ok((&kernels()?.grind_w8, BLOCK_DIM))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout helpers against the host's (`level_sizes4` /
    /// `level_offsets4`) on a few shapes, written out.
    #[test]
    fn the_arity4_layout_counts_real_nodes_only() {
        assert_eq!(level_sizes(1), vec![1]);
        assert_eq!(level_sizes(2), vec![2, 1]);
        assert_eq!(level_sizes(8), vec![8, 2, 1]);
        assert_eq!(level_sizes(32), vec![32, 8, 2, 1]);
        assert_eq!(level_sizes(64), vec![64, 16, 4, 1]);
        assert_eq!(tree_nodes(64), 85);
        assert_eq!(tree_nodes(32), 43);
        assert_eq!(depth(1 << 21), 11);
        assert_eq!(depth(1 << 22), 11);
        assert_eq!(depth(2), 1);
        // The top two levels of a 64-leaf tree: the root and its four children.
        assert_eq!(top_levels_nodes(64, 2), 5);
        assert_eq!(top_levels_nodes(64, 9), 85);
    }
}
