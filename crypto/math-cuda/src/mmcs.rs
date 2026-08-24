//! Device build of the mixed-height MMCS — one tree over all of an epoch's
//! matrices.
//!
//! The host contract this must reproduce byte for byte lives in
//! `crypto/stark/src/fri/mmcs.rs`: leaf `k` of a height group is the
//! configuration's batched hash over the concatenation, in INPUT order, of each
//! matrix's bit-reversed rows `2k` and `2k+1`; the climb compresses pairs and,
//! where a shorter group's height matches the halved layer, compresses the
//! parent again with that group's leaf digest. Both kernel families implement
//! it — [`DeviceHash`] selects at every launch.
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
//! Retained state per leaf: 204 bytes under keccak (25 lanes plus the rate
//! cursor), 104 under BLAKE3 (the chain's cv + pending block) — ~214 / ~109 MiB
//! at 2^20 leaves, against one full LDE per matrix in the group.
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

use crate::DeviceHash;
use crate::Result;
use crate::blake3::blake3_launch_cfg;
use crate::device::backend;
use crate::merkle::keccak_launch_cfg;

/// 26 u32s per leaf: cv, the pending block's words, nwords, started. Mirrors
/// `BLAKE3_MMCS_STATE_WORDS` in `kernels/blake3.cu`.
const BLAKE3_MMCS_STATE_WORDS: usize = 26;

/// One height group's per-leaf sponges, live on device between absorptions.
///
/// Construct once per height group, [`Self::absorb_row_major`] /
/// [`Self::absorb_ext3_slabs`] once per matrix at that height IN INPUT ORDER
/// (the leaf concatenation binds that order), then [`Self::finalize`].
pub struct MmcsGroupHasher {
    state: HasherState,
    num_leaves: u64,
    /// `log2` of the group's row count — every matrix absorbed here must have it,
    /// since they share the leaves.
    log_num_rows: u64,
    absorbed: usize,
}

/// The per-leaf running state, shaped by the hash family: keccak keeps the 25
/// sponge lanes plus a rate cursor, BLAKE3 the chain's cv + pending block.
enum HasherState {
    Keccak {
        states: CudaSlice<u64>,
        rate_pos: CudaSlice<u32>,
    },
    Blake3 {
        states: CudaSlice<u32>,
    },
}

impl MmcsGroupHasher {
    /// Initialized per-leaf states for a height group of `2^log_num_rows`
    /// rows, i.e. `2^(log_num_rows - 1)` leaves, under the given hash family.
    pub fn new(stream: &Arc<CudaStream>, log_num_rows: u64, hash: DeviceHash) -> Result<Self> {
        assert!(
            log_num_rows >= 1,
            "row-pair leaves need at least 2 rows (log_num_rows >= 1)"
        );
        let be = backend()?;
        let num_leaves = 1u64 << (log_num_rows - 1);

        let state = match hash {
            DeviceHash::Keccak256 => {
                let mut states = stream.alloc_zeros::<u64>((num_leaves * 25) as usize)?;
                let mut rate_pos = stream.alloc_zeros::<u32>(num_leaves as usize)?;
                // `alloc_zeros` already gives the state we want; the kernel runs
                // anyway so the zeroing is this module's own statement rather
                // than an allocator property a future change could quietly take
                // away.
                let cfg = keccak_launch_cfg(num_leaves);
                unsafe {
                    stream
                        .launch_builder(&be.mmcs_states_init)
                        .arg(&mut states)
                        .arg(&mut rate_pos)
                        .arg(&num_leaves)
                        .launch(cfg)?;
                }
                HasherState::Keccak { states, rate_pos }
            }
            DeviceHash::Blake3 => {
                let mut states =
                    stream.alloc_zeros::<u32>((num_leaves as usize) * BLAKE3_MMCS_STATE_WORDS)?;
                // The chain's initial cv is the IV, not zero — the init kernel
                // is load-bearing here, not a statement.
                let cfg = blake3_launch_cfg(num_leaves);
                unsafe {
                    stream
                        .launch_builder(&be.blake3_mmcs_states_init)
                        .arg(&mut states)
                        .arg(&num_leaves)
                        .launch(cfg)?;
                }
                HasherState::Blake3 { states }
            }
        };

        Ok(Self {
            state,
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
        match &mut self.state {
            HasherState::Keccak { states, rate_pos } => {
                let cfg = keccak_launch_cfg(self.num_leaves);
                unsafe {
                    stream
                        .launch_builder(&be.mmcs_absorb_row_pair_row_major)
                        .arg(states)
                        .arg(rate_pos)
                        .arg(data)
                        .arg(&row_stride)
                        .arg(&col_start)
                        .arg(&col_end)
                        .arg(&num_rows)
                        .arg(&self.log_num_rows)
                        .arg(&self.num_leaves)
                        .launch(cfg)?;
                }
            }
            HasherState::Blake3 { states } => {
                let cfg = blake3_launch_cfg(self.num_leaves);
                unsafe {
                    stream
                        .launch_builder(&be.blake3_mmcs_absorb_row_pair_row_major)
                        .arg(states)
                        .arg(data)
                        .arg(&row_stride)
                        .arg(&col_start)
                        .arg(&col_end)
                        .arg(&num_rows)
                        .arg(&self.log_num_rows)
                        .arg(&self.num_leaves)
                        .launch(cfg)?;
                }
            }
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
        match &mut self.state {
            HasherState::Keccak { states, rate_pos } => {
                let cfg = keccak_launch_cfg(self.num_leaves);
                unsafe {
                    stream
                        .launch_builder(&be.mmcs_absorb_row_pair_ext3_slabs)
                        .arg(states)
                        .arg(rate_pos)
                        .arg(parts)
                        .arg(&col_stride)
                        .arg(&num_parts)
                        .arg(&num_rows)
                        .arg(&self.log_num_rows)
                        .arg(&self.num_leaves)
                        .launch(cfg)?;
                }
            }
            HasherState::Blake3 { states } => {
                let cfg = blake3_launch_cfg(self.num_leaves);
                unsafe {
                    stream
                        .launch_builder(&be.blake3_mmcs_absorb_row_pair_ext3_slabs)
                        .arg(states)
                        .arg(parts)
                        .arg(&col_stride)
                        .arg(&num_parts)
                        .arg(&num_rows)
                        .arg(&self.log_num_rows)
                        .arg(&self.num_leaves)
                        .launch(cfg)?;
                }
            }
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
        match &self.state {
            HasherState::Keccak { states, rate_pos } => {
                let cfg = keccak_launch_cfg(self.num_leaves);
                unsafe {
                    stream
                        .launch_builder(&be.mmcs_states_finalize)
                        .arg(states)
                        .arg(rate_pos)
                        .arg(&self.num_leaves)
                        .arg(&mut digests)
                        .launch(cfg)?;
                }
            }
            HasherState::Blake3 { states } => {
                let cfg = blake3_launch_cfg(self.num_leaves);
                unsafe {
                    stream
                        .launch_builder(&be.blake3_mmcs_states_finalize)
                        .arg(states)
                        .arg(&self.num_leaves)
                        .arg(&mut digests)
                        .launch(cfg)?;
                }
            }
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
    hash: DeviceHash,
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

        // The level kernels read `inject` only when `has_inject` is set, so a
        // level with no injection still needs a valid pointer argument that is
        // provably never dereferenced.
        let level_fn = match hash {
            DeviceHash::Keccak256 => &be.keccak_mmcs_level,
            DeviceHash::Blake3 => &be.blake3_mmcs_level,
        };
        let cfg = match hash {
            DeviceHash::Keccak256 => keccak_launch_cfg(n_pairs),
            DeviceHash::Blake3 => blake3_launch_cfg(n_pairs),
        };
        match injected {
            Some(digests) => unsafe {
                stream
                    .launch_builder(level_fn)
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
                        .launch_builder(level_fn)
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
