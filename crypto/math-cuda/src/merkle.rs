//! GPU Keccak-256 leaf hashing for Merkle commits.
//!
//! Matches `FieldElementVectorBackend<F, Keccak256, 32>::hash_data` in
//! `crypto/crypto/src/merkle_tree/backends/field_element_vector.rs`, combined
//! with the `reverse_index` row read pattern used in
//! `commit_columns_bit_reversed` at `crypto/stark/src/prover.rs:368`.
//!
//! Caller supplies base-field column slabs already laid out as
//! `[col * col_stride + row]` (the same layout `coset_lde_batch_base_into`
//! writes to the pinned staging buffer). The kernel bit-reverses `row_idx`,
//! reads each column's canonical u64 at that row, byte-swaps it into a
//! Keccak lane, absorbs lane-by-lane, and squeezes 32 bytes per leaf.
//!
//! For ext3 columns the layout is `[col*3*col_stride + k*col_stride + row]`
//! — three base slabs per ext3 column — and the kernel reads three u64s per
//! column in component order 0,1,2 to match `FieldElement::<Ext3>::write_bytes_be`.

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;

/// Run GPU Keccak-256 leaf hashing on a base-field column buffer.
///
/// `columns` must hold `num_cols * col_stride` u64s with column `c`'s data
/// at `[c*col_stride .. c*col_stride + num_rows]`. Returns `num_rows * 32`
/// hash bytes in natural (non-bit-reversed) row order.
pub fn keccak_leaves_base(
    columns: &[u64],
    col_stride: usize,
    num_cols: usize,
    num_rows: usize,
) -> Result<Vec<u8>> {
    assert!(num_rows.is_power_of_two());
    assert!(columns.len() >= num_cols * col_stride);
    let be = backend();
    let stream = be.next_stream();
    let cols_dev = stream.clone_htod(&columns[..num_cols * col_stride])?;
    let mut out_dev = stream.alloc_zeros::<u8>(num_rows * 32)?;
    launch_keccak_base(
        stream.as_ref(),
        &cols_dev,
        col_stride as u64,
        num_cols as u64,
        num_rows as u64,
        &mut out_dev,
    )?;
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Ext3 variant — columns interleaved as three base slabs per ext3 column.
/// `columns.len() >= num_cols * 3 * col_stride`.
pub fn keccak_leaves_ext3(
    columns: &[u64],
    col_stride: usize,
    num_cols: usize,
    num_rows: usize,
) -> Result<Vec<u8>> {
    assert!(num_rows.is_power_of_two());
    assert!(columns.len() >= num_cols * 3 * col_stride);
    let be = backend();
    let stream = be.next_stream();
    let cols_dev = stream.clone_htod(&columns[..num_cols * 3 * col_stride])?;
    let mut out_dev = stream.alloc_zeros::<u8>(num_rows * 32)?;
    launch_keccak_ext3(
        stream.as_ref(),
        &cols_dev,
        col_stride as u64,
        num_cols as u64,
        num_rows as u64,
        &mut out_dev,
    )?;
    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Block size for Keccak kernels. Per-thread register footprint is ~60 regs
/// (25-lane state + auxiliaries); the default 256 threads/block pushes the
/// block register file past the hardware limit on sm_120 (Blackwell). 128
/// keeps us inside the budget with some head-room.
const KECCAK_BLOCK_DIM: u32 = 128;

fn keccak_launch_cfg(num_rows: u64) -> LaunchConfig {
    let grid = ((num_rows as u32) + KECCAK_BLOCK_DIM - 1) / KECCAK_BLOCK_DIM;
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (KECCAK_BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub(crate) fn launch_keccak_base(
    stream: &CudaStream,
    cols_dev: &CudaSlice<u64>,
    col_stride: u64,
    num_cols: u64,
    num_rows: u64,
    out_dev: &mut CudaSlice<u8>,
) -> Result<()> {
    let be = backend();
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
/// `leaves_len` must be a power of two and ≥ 2.
pub fn build_merkle_tree_on_device(hashed_leaves: &[u8]) -> Result<Vec<u8>> {
    assert!(hashed_leaves.len() % 32 == 0);
    let leaves_len = hashed_leaves.len() / 32;
    assert!(leaves_len >= 2, "tree needs at least two leaves");
    assert!(
        leaves_len.is_power_of_two(),
        "leaves_len must be a power of two"
    );

    let total_nodes = 2 * leaves_len - 1;
    let be = backend();
    let stream = be.next_stream();

    // Allocate the full node buffer without zero-fill — we overwrite the
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

    // Build level by level. The CPU `build(nodes, leaves_len)` starts with
    //   level_begin_index = leaves_len - 1
    //   level_end_index   = 2 * level_begin_index
    // and each iteration computes:
    //   new_level_begin_index = level_begin_index / 2
    //   new_level_length       = level_begin_index - new_level_begin_index
    // The parents occupy [new_level_begin_index, level_begin_index); the
    // children occupy [level_begin_index, level_end_index + 1).
    let mut level_begin: u64 = (leaves_len - 1) as u64;
    while level_begin != 0 {
        let new_begin = level_begin / 2;
        let n_pairs = level_begin - new_begin;

        let cfg = keccak_launch_cfg(n_pairs);
        unsafe {
            stream
                .launch_builder(&be.keccak_merkle_level)
                .arg(&mut nodes_dev)
                .arg(&new_begin)
                .arg(&n_pairs)
                .launch(cfg)?;
        }
        level_begin = new_begin;
    }

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
    out_dev: &mut CudaSlice<u8>,
) -> Result<()> {
    let be = backend();
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
