//! Committing one stacked polynomial of the multilinear path on device.
//!
//! Mirrors the host pipeline in `multilinear`: the Möbius transform that turns
//! hypercube evaluations into monomial coefficients, the lift's bit-reverse,
//! one NTT onto the blown-up domain, then the strided-coset leaf hash and the
//! Merkle tree. Parity against that pipeline is checked by `tests/whir_commit.rs`.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::{alloc_or_trim, backend};
use crate::merkle::{build_inner_tree_levels, keccak_launch_cfg};

/// A codeword the device holds, base-field or ext3.
///
/// The commit leaves one here and the chain folds it here: it is the biggest
/// array the proof moves, and the host only ever needs the few values a query
/// opens.
#[derive(Clone)]
pub struct DeviceCodeword {
    buffer: Arc<CudaSlice<u64>>,
    stream: Arc<CudaStream>,
    elements: usize,
    base: bool,
    /// The room the chain promised itself: this codeword, the folds that halve
    /// it, and the tree each of them is committed and opened through. Shared
    /// with the folds, which live inside it.
    _room: Arc<crate::device::DeviceReservation>,
}

impl DeviceCodeword {
    pub fn elements(&self) -> usize {
        self.elements
    }

    pub fn is_base(&self) -> bool {
        self.base
    }

    /// The first value, which is what the last fold leaves behind.
    pub fn first(&self) -> Result<[u64; 3]> {
        let limbs = if self.base { 1 } else { 3 };
        let head = self.stream.clone_dtoh(&self.buffer.slice(0..limbs))?;
        self.stream.synchronize()?;
        Ok(if self.base {
            [head[0], 0, 0]
        } else {
            [head[0], head[1], head[2]]
        })
    }

    /// The Merkle tree over this codeword's fold blocks, built here.
    ///
    /// A leaf is the `2^log_folding` coset that folds onto one position, and
    /// the layout is the host's: `2*num_leaves - 1` nodes of 32 bytes, root
    /// first.
    fn build_tree(
        &self,
        log_folding: usize,
        hash: crate::DeviceHash,
    ) -> Result<(CudaSlice<u8>, usize)> {
        let num_leaves = self.elements >> log_folding;
        assert!(num_leaves >= 2, "tree needs at least two leaves");
        let be = backend()?;
        let total_nodes = 2 * num_leaves - 1;
        // SAFETY: every byte is written before it is read — the leaves by the
        // kernel below, the inner nodes by the level loop after it.
        let mut nodes =
            unsafe { crate::device::alloc_or_trim::<u8>(&self.stream, total_nodes * 32) }?;
        {
            let leaves_offset = (num_leaves - 1) * 32;
            let mut leaves = nodes.slice_mut(leaves_offset..leaves_offset + num_leaves * 32);
            let num_leaves_u64 = num_leaves as u64;
            let block = 1u64 << log_folding;
            // ★ The hash is chosen HERE, not by the host backend that will
            // label the result. `hash` is the key the caller's `WhirHash`
            // supplied, so a tree labelled RPX was hashed by RPX's kernels or
            // was not built here at all.
            let kernel = match (hash, self.base) {
                (crate::DeviceHash::Keccak256, true) => &be.keccak256_leaves_base_coset,
                (crate::DeviceHash::Keccak256, false) => &be.keccak256_leaves_ext3_coset,
                (crate::DeviceHash::Rpx256, true) => &be.rpx_leaves_base_coset,
                (crate::DeviceHash::Rpx256, false) => &be.rpx_leaves_ext3_coset,
            };
            unsafe {
                self.stream
                    .launch_builder(kernel)
                    .arg(self.buffer.as_ref())
                    .arg(&num_leaves_u64)
                    .arg(&block)
                    .arg(&mut leaves)
                    .launch(keccak_launch_cfg(num_leaves_u64))?;
            }
        }
        build_inner_tree_levels(self.stream.as_ref(), be, &mut nodes, num_leaves, hash)?;
        Ok((nodes, num_leaves))
    }

    /// The root of that tree, which is the commitment.
    ///
    /// The tree itself is dropped: the only other thing anyone wants from it
    /// is a path per query, and by then the queries are known — see
    /// [`paths`](Self::paths).
    pub fn commit(&self, log_folding: usize, hash: crate::DeviceHash) -> Result<[u8; 32]> {
        let (nodes, _) = self.build_tree(log_folding, hash)?;
        let head = self.stream.clone_dtoh(&nodes.slice(0..32))?;
        self.stream.synchronize()?;
        let mut root = [0u8; 32];
        root.copy_from_slice(&head);
        Ok(root)
    }

    /// The whole tree in the host node layout — what a caller that walks it
    /// here needs, and what the parity test compares against.
    pub fn nodes_to_host(&self, log_folding: usize, hash: crate::DeviceHash) -> Result<Vec<u8>> {
        let (nodes, _) = self.build_tree(log_folding, hash)?;
        let out = self.stream.clone_dtoh(&nodes)?;
        self.stream.synchronize()?;
        Ok(out)
    }

    /// The authentication paths of `positions`, against the same tree.
    ///
    /// Rebuilt rather than kept or carried home. Keeping it costs half a
    /// gigabyte of device memory per commitment for the whole proof; bringing
    /// it back costs ten times the rehash, because a pageable copy of half a
    /// gigabyte is the slowest thing in the commit. What the host needs of a
    /// tree is a kilobyte per query.
    pub fn paths(
        &self,
        log_folding: usize,
        positions: &[u32],
        hash: crate::DeviceHash,
    ) -> Result<Vec<u8>> {
        let (nodes, num_leaves) = self.build_tree(log_folding, hash)?;
        crate::merkle::gather_merkle_paths_dev(&nodes, num_leaves, positions, &self.stream)
    }

    /// The fold blocks `indices` open — `block` values at stride `num_leaves`
    /// from each — gathered where they lie, one launch and one copy back.
    ///
    /// Query `q`'s block is `out[q*block*limbs ..]`, with one limb per value
    /// for a base codeword and three for an extension one.
    pub fn cosets(&self, indices: &[u64], num_leaves: usize, block: usize) -> Result<Vec<u64>> {
        assert!(!indices.is_empty(), "a round opens at least one block");
        let limbs = if self.base { 1usize } else { 3 };
        let be = backend()?;
        let index_dev = self.stream.clone_htod(indices)?;
        let total = indices.len() * block;
        // SAFETY: the kernel writes every value it is sized for.
        let mut out = unsafe { alloc_or_trim::<u64>(&self.stream, total * limbs) }?;
        let queries = indices.len() as u64;
        let num_leaves_u64 = num_leaves as u64;
        let block_u64 = block as u64;
        let limbs_u64 = limbs as u64;
        unsafe {
            self.stream
                .launch_builder(&be.gather_cosets)
                .arg(self.buffer.as_ref())
                .arg(&index_dev)
                .arg(&queries)
                .arg(&num_leaves_u64)
                .arg(&block_u64)
                .arg(&limbs_u64)
                .arg(&mut out)
                .launch(LaunchConfig::for_num_elems(total as u32))?;
        }
        let values = self.stream.clone_dtoh(&out)?;
        self.stream.synchronize()?;
        Ok(values)
    }
}

/// The codeword and the Merkle nodes of one stacked polynomial.
///
/// `evals` holds the multilinear's `2^m` hypercube values in canonical
/// Goldilocks form. The codeword is `2^(m + log_blowup)` values in domain
/// order — the same array the host prover folds — and the nodes are the tree in
/// the host layout (`2*num_leaves - 1` nodes of 32 bytes, root first), so they
/// plug straight into a `MerkleTree`.
///
/// `log_folding` is the first fold's width: a leaf is the `2^log_folding` coset
/// that folds onto one position.
/// The same, over a stacked polynomial that is never assembled on the host.
///
/// A stacked polynomial is its columns written at their offsets and zeros
/// everywhere else, so the parts go straight into the device buffer: what the
/// host would have built is a copy of them, and building it costs a pass over
/// every byte the commit is about to upload anyway.
///
/// `parts` is `(column, offset in elements)`; `log_evals` is the stacked
/// polynomial's variable count.
pub fn commit_codeword_parts(
    parts: &[(&[u64], usize)],
    log_evals: usize,
    log_blowup: usize,
    log_folding: usize,
    transient: bool,
    hash: crate::DeviceHash,
) -> Result<(DeviceCodeword, [u8; 32])> {
    commit_from(
        Source::Parts { parts, log_evals },
        log_blowup,
        log_folding,
        transient,
        hash,
    )
}

/// The same for parts the card already holds: `(column index, offset)` into the
/// epoch's columns. The scatter is then a copy at device bandwidth rather than
/// the trace crossing the bus again.
pub fn commit_codeword_resident(
    store: &crate::columns::DeviceColumns,
    parts: &[(usize, usize)],
    log_evals: usize,
    log_blowup: usize,
    log_folding: usize,
    transient: bool,
    hash: crate::DeviceHash,
) -> Result<(DeviceCodeword, [u8; 32])> {
    commit_from(
        Source::Resident {
            store,
            parts,
            log_evals,
        },
        log_blowup,
        log_folding,
        transient,
        hash,
    )
}

/// Where a commit's coefficients come from: one slab the host holds, or the
/// columns a stacked polynomial is made of.
enum Source<'a> {
    Whole(&'a [u64]),
    Parts {
        parts: &'a [(&'a [u64], usize)],
        log_evals: usize,
    },
    Resident {
        store: &'a crate::columns::DeviceColumns,
        parts: &'a [(usize, usize)],
        log_evals: usize,
    },
}

impl Source<'_> {
    fn log_evals(&self) -> u64 {
        match self {
            Self::Whole(evals) => evals.len().trailing_zeros() as u64,
            Self::Parts { log_evals, .. } | Self::Resident { log_evals, .. } => *log_evals as u64,
        }
    }

    /// Fills `coeffs`, which is `2^log_evals` elements long.
    fn write(&self, stream: &Arc<CudaStream>, coeffs: &mut CudaSlice<u64>) -> Result<()> {
        match self {
            Self::Whole(evals) => stream.memcpy_htod(*evals, coeffs),
            Self::Parts { parts, .. } => {
                // Everything the parts do not cover is the stacking's padding,
                // and that is zero by definition.
                stream.memset_zeros(coeffs)?;
                for (column, offset) in *parts {
                    let mut at = coeffs.slice_mut(*offset..*offset + column.len());
                    stream.memcpy_htod(*column, &mut at)?;
                }
                Ok(())
            }
            Self::Resident { store, parts, .. } => {
                stream.memset_zeros(coeffs)?;
                for (column, offset) in *parts {
                    store.copy_into(*column, coeffs, *offset, stream)?;
                }
                Ok(())
            }
        }
    }
}

pub fn commit_codeword(
    evals: &[u64],
    log_blowup: usize,
    log_folding: usize,
    transient: bool,
    hash: crate::DeviceHash,
) -> Result<(DeviceCodeword, [u8; 32])> {
    commit_from(
        Source::Whole(evals),
        log_blowup,
        log_folding,
        transient,
        hash,
    )
}

fn commit_from(
    source: Source<'_>,
    log_blowup: usize,
    log_folding: usize,
    transient: bool,
    hash: crate::DeviceHash,
) -> Result<(DeviceCodeword, [u8; 32])> {
    let log_evals = source.log_evals();
    let log_n = log_evals + log_blowup as u64;
    let n = 1usize << log_n;
    assert!(
        log_folding as u64 <= log_n,
        "a leaf cannot exceed the domain"
    );
    let num_leaves = n >> log_folding;
    assert!(num_leaves >= 2, "tree needs at least two leaves");
    assert!(
        n <= u32::MAX as usize,
        "codeword length {n} exceeds u32 range — kernel grid would silently truncate",
    );

    let be = backend()?;
    // The codeword itself stays until this commitment's opening. On top of it
    // there is a working set — the tree this commit builds, and later the
    // folds and the trees the chain's rounds build — which is another
    // codeword's worth and is **transient**: one commit or one opening uses it
    // at a time. A caller committing a group of polynomials promises that once
    // for all of them and passes `transient: false`; one committing alone
    // promises it here.
    let promise = if transient { 2 } else { 1 };
    let Some(room) = be.reserve(n as u64 * 8 * promise) else {
        return Err(cudarc::driver::DriverError(
            cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
        ));
    };
    let stream = be.next_stream();

    // The coefficients get a buffer of their own: the Möbius transform runs
    // over them, and the spread below reads them while it writes the codeword.
    // SAFETY: every element is written by the copy below.
    let mut coeffs = unsafe { alloc_or_trim::<u64>(&stream, 1usize << log_evals) }?;
    source.write(&stream, &mut coeffs)?;

    mobius(
        stream.as_ref(),
        be,
        &mut coeffs,
        1usize << log_evals,
        log_evals,
    )?;

    // The lift's bit-reverse and the NTT's cancel around the zero padding —
    // see `lift_spread`. What was two scattered passes over the codeword plus
    // the memset that zeroed it is one pass that writes all of it.
    // SAFETY: the spread writes every element, padding included.
    let mut x = unsafe { alloc_or_trim::<u64>(&stream, n) }?;
    let n_u64 = n as u64;
    let log_blowup_u32 = log_blowup as u32;
    unsafe {
        stream
            .launch_builder(&be.lift_spread)
            .arg(&coeffs)
            .arg(&n_u64)
            .arg(&log_blowup_u32)
            .arg(&mut x)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    // Spent: the spread has read them, and the free is stream-ordered.
    drop(coeffs);
    let twiddles = be.fwd_twiddles_for(log_n)?;
    crate::ntt::run_ntt_body(stream.as_ref(), &mut x, twiddles.as_ref(), n_u64, log_n)?;

    let codeword = DeviceCodeword {
        buffer: Arc::new(x),
        stream,
        elements: n,
        base: true,
        _room: Arc::new(room),
    };
    let root = codeword.commit(log_folding, hash)?;
    Ok((codeword, root))
}

/// Levels a tile fuses at once: 32 rows of 32 columns is a full block, and the
/// shared tile it needs is a few kilobytes.
const MOBIUS_TILE_LEVELS: u32 = 5;
/// Columns a tile spans — one warp, matching the kernel.
const MOBIUS_TILE_COLS: u32 = 32;
/// Levels the contiguous kernel takes: a block of 256 holds both sides of
/// every pair the first eight levels make.
const MOBIUS_LOW_LEVELS: u32 = 8;

/// The Mobius transform over `coeffs`, a window of levels at a time.
///
/// Each level is one pass over the whole array, so run one per launch and the
/// transform is bound by how many times it reads the array rather than by the
/// subtractions. The windows below fuse the levels that share a tile, which is
/// the shape the NTT's levels are already fused in.
fn mobius(
    stream: &CudaStream,
    be: &crate::device::Backend,
    coeffs: &mut CudaSlice<u64>,
    len: usize,
    log_evals: u64,
) -> Result<()> {
    let mut level: u64 = 0;

    // The contiguous window, when there is a whole block of elements to hold.
    if log_evals >= MOBIUS_LOW_LEVELS as u64 && len >= 256 {
        let k = MOBIUS_LOW_LEVELS;
        unsafe {
            stream
                .launch_builder(&be.mobius_low_levels)
                .arg(&mut *coeffs)
                .arg(&k)
                .launch(LaunchConfig {
                    grid_dim: ((len / 256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        level = k as u64;
    }

    while level < log_evals {
        let low = 1u64 << level;
        let k = MOBIUS_TILE_LEVELS.min((log_evals - level) as u32);
        // A tile needs a warp of consecutive low indices to stay coalesced; a
        // level below that is left to the one-level kernel.
        if low < MOBIUS_TILE_COLS as u64 {
            let half = (len / 2) as u64;
            let stride = low;
            unsafe {
                stream
                    .launch_builder(&be.mobius_level)
                    .arg(&mut *coeffs)
                    .arg(&half)
                    .arg(&stride)
                    .launch(LaunchConfig::for_num_elems(half as u32))?;
            }
            level += 1;
            continue;
        }
        let rows = 1u32 << k;
        let pitch = MOBIUS_TILE_COLS + 1;
        unsafe {
            stream
                .launch_builder(&be.mobius_tile)
                .arg(&mut *coeffs)
                .arg(&level)
                .arg(&k)
                .launch(LaunchConfig {
                    grid_dim: (
                        (low / MOBIUS_TILE_COLS as u64) as u32,
                        (len as u64 / (low << k)) as u32,
                        1,
                    ),
                    block_dim: (MOBIUS_TILE_COLS, rows, 1),
                    shared_mem_bytes: rows * pitch * 8,
                })?;
        }
        level += k as u64;
    }
    Ok(())
}

/// The same, with the codeword brought back — what a caller that folds on the
/// host needs.
pub fn commit_codeword_to_host(
    evals: &[u64],
    log_blowup: usize,
    log_folding: usize,
    hash: crate::DeviceHash,
) -> Result<(Vec<u64>, Vec<u8>)> {
    let (codeword, _root) = commit_codeword(evals, log_blowup, log_folding, true, hash)?;
    let values = codeword.stream.clone_dtoh(codeword.buffer.as_ref())?;
    codeword.stream.synchronize()?;
    let nodes = codeword.nodes_to_host(log_folding, hash)?;
    Ok((values, nodes))
}

/// Folds a codeword `levels` times in one residency, lifting the base field on
/// the first fold.
///
/// `g_invs` is each level's inverse domain generator (the domain squares as the
/// codeword halves) and `alphas` the folding challenges, three u64 per level.
/// Returns the folded codeword as interleaved ext3.
pub fn fold_codeword_base(
    codeword: &[u64],
    two_inv: u64,
    g_invs: &[u64],
    alphas: &[u64],
) -> Result<Vec<u64>> {
    let levels = g_invs.len();
    assert!(levels > 0, "a fold needs a level");
    assert_eq!(alphas.len(), levels * 3, "three u64 per challenge");
    assert!(
        codeword.len().is_power_of_two(),
        "a codeword is a power of two"
    );
    assert!(
        codeword.len() >> levels >= 1,
        "{levels} folds do not fit a codeword of {}",
        codeword.len()
    );

    let be = backend()?;
    let stream = be.next_stream();
    let mut half = codeword.len() / 2;
    let input = stream.clone_htod(codeword)?;
    let alpha = stream.clone_htod(alphas)?;

    // SAFETY: the kernel writes every element of the half it produces.
    let mut current = unsafe { alloc_or_trim::<u64>(&stream, half * 3) }?;
    let half_arg = half as u64;
    let g_inv = g_invs[0];
    unsafe {
        stream
            .launch_builder(&be.whir_fold_base_ext3)
            .arg(&input)
            .arg(&half_arg)
            .arg(&two_inv)
            .arg(&g_inv)
            .arg(&alpha.slice(0..3))
            .arg(&mut current)
            .launch(LaunchConfig::for_num_elems(half as u32))?;
    }
    drop(input);

    for (level, g_inv) in g_invs.iter().enumerate().skip(1) {
        half /= 2;
        // SAFETY: as above.
        let mut next = unsafe { alloc_or_trim::<u64>(&stream, half * 3) }?;
        let half_arg = half as u64;
        unsafe {
            stream
                .launch_builder(&be.whir_fold_ext3)
                .arg(&current)
                .arg(&half_arg)
                .arg(&two_inv)
                .arg(g_inv)
                .arg(&alpha.slice(level * 3..level * 3 + 3))
                .arg(&mut next)
                .launch(LaunchConfig::for_num_elems(half as u32))?;
        }
        current = next;
    }

    let out = stream.clone_dtoh(&current)?;
    stream.synchronize()?;
    Ok(out)
}

/// Folds a resident codeword `levels` times, leaving the result resident.
///
/// The chain folds the same array level after level and only opens a handful
/// of its values, so it never has to come back.
pub fn fold_resident(
    codeword: &DeviceCodeword,
    two_inv: u64,
    g_invs: &[u64],
    alphas: &[u64],
) -> Result<DeviceCodeword> {
    let levels = g_invs.len();
    assert!(levels > 0, "a fold needs a level");
    assert_eq!(alphas.len(), levels * 3, "three u64 per challenge");
    assert!(
        codeword.elements >> levels >= 1,
        "{levels} folds do not fit"
    );

    let be = backend()?;
    let stream = codeword.stream.clone();
    let alpha = stream.clone_htod(alphas)?;
    let mut half = codeword.elements / 2;

    // SAFETY: the kernel writes every element of the half it produces.
    let mut current = unsafe { alloc_or_trim::<u64>(&stream, half * 3) }?;
    let half_arg = half as u64;
    let kernel = if codeword.base {
        &be.whir_fold_base_ext3
    } else {
        &be.whir_fold_ext3
    };
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(codeword.buffer.as_ref())
            .arg(&half_arg)
            .arg(&two_inv)
            .arg(&g_invs[0])
            .arg(&alpha.slice(0..3))
            .arg(&mut current)
            .launch(LaunchConfig::for_num_elems(half as u32))?;
    }

    for (level, g_inv) in g_invs.iter().enumerate().skip(1) {
        half /= 2;
        // SAFETY: as above.
        let mut next = unsafe { alloc_or_trim::<u64>(&stream, half * 3) }?;
        let half_arg = half as u64;
        unsafe {
            stream
                .launch_builder(&be.whir_fold_ext3)
                .arg(&current)
                .arg(&half_arg)
                .arg(&two_inv)
                .arg(g_inv)
                .arg(&alpha.slice(level * 3..level * 3 + 3))
                .arg(&mut next)
                .launch(LaunchConfig::for_num_elems(half as u32))?;
        }
        current = next;
    }

    Ok(DeviceCodeword {
        buffer: Arc::new(current),
        stream,
        elements: half,
        base: false,
        // The fold lives inside the room the codeword it came from promised:
        // it is half of it, and that one is still alive.
        _room: codeword._room.clone(),
    })
}

/// The same for a codeword already in the extension.
pub fn fold_codeword_ext3(
    codeword: &[u64],
    two_inv: u64,
    g_invs: &[u64],
    alphas: &[u64],
) -> Result<Vec<u64>> {
    let levels = g_invs.len();
    assert!(levels > 0, "a fold needs a level");
    assert_eq!(alphas.len(), levels * 3, "three u64 per challenge");
    assert!(
        codeword.len().is_multiple_of(3),
        "three u64 per ext3 element"
    );
    let elements = codeword.len() / 3;
    assert!(elements.is_power_of_two(), "a codeword is a power of two");
    assert!(elements >> levels >= 1, "{levels} folds do not fit");

    let be = backend()?;
    let stream = be.next_stream();
    let mut half = elements / 2;
    let alpha = stream.clone_htod(alphas)?;
    let mut current = stream.clone_htod(codeword)?;

    for (level, g_inv) in g_invs.iter().enumerate() {
        // SAFETY: the kernel writes every element of the half it produces.
        let mut next = unsafe { alloc_or_trim::<u64>(&stream, half * 3) }?;
        let half_arg = half as u64;
        unsafe {
            stream
                .launch_builder(&be.whir_fold_ext3)
                .arg(&current)
                .arg(&half_arg)
                .arg(&two_inv)
                .arg(g_inv)
                .arg(&alpha.slice(level * 3..level * 3 + 3))
                .arg(&mut next)
                .launch(LaunchConfig::for_num_elems(half as u32))?;
        }
        current = next;
        half /= 2;
    }

    let out = stream.clone_dtoh(&current)?;
    stream.synchronize()?;
    Ok(out)
}

/// Merkle-commits an ext3 codeword's fold blocks on device, returning the tree
/// in the host node layout.
///
/// The codeword itself stays where the caller has it: a folded codeword is the
/// next round's input on the host side, so only the tree comes back.
pub fn commit_codeword_ext3(
    codeword: &[u64],
    log_folding: usize,
    hash: crate::DeviceHash,
) -> Result<Vec<u8>> {
    assert!(
        codeword.len().is_multiple_of(3),
        "three u64 per ext3 element"
    );
    let elements = codeword.len() / 3;
    assert!(elements.is_power_of_two(), "a codeword is a power of two");
    assert!(
        log_folding <= elements.trailing_zeros() as usize,
        "a leaf cannot exceed the codeword"
    );
    let num_leaves = elements >> log_folding;
    assert!(num_leaves >= 2, "tree needs at least two leaves");

    let be = backend()?;
    let stream = be.next_stream();
    let values = stream.clone_htod(codeword)?;

    let total_nodes = 2 * num_leaves - 1;
    // SAFETY: every byte is written before it is read — the leaves by the
    // kernel below, the inner nodes by the level loop after it.
    let mut nodes = unsafe { alloc_or_trim::<u8>(&stream, total_nodes * 32) }?;
    {
        let leaves_offset = (num_leaves - 1) * 32;
        let mut leaves = nodes.slice_mut(leaves_offset..leaves_offset + num_leaves * 32);
        let num_leaves_u64 = num_leaves as u64;
        let block = 1u64 << log_folding;
        unsafe {
            stream
                .launch_builder(match hash {
                    crate::DeviceHash::Keccak256 => &be.keccak256_leaves_ext3_coset,
                    crate::DeviceHash::Rpx256 => &be.rpx_leaves_ext3_coset,
                })
                .arg(&values)
                .arg(&num_leaves_u64)
                .arg(&block)
                .arg(&mut leaves)
                .launch(keccak_launch_cfg(num_leaves_u64))?;
        }
    }
    build_inner_tree_levels(stream.as_ref(), be, &mut nodes, num_leaves, hash)?;

    let out = stream.clone_dtoh(&nodes)?;
    stream.synchronize()?;
    Ok(out)
}
