//! Committing one stacked polynomial of the multilinear path on device.
//!
//! Mirrors the host pipeline in `multilinear`: the Möbius transform that turns
//! hypercube evaluations into monomial coefficients, the lift's bit-reverse,
//! one NTT onto the blown-up domain, then the strided-coset leaf hash and the
//! Merkle tree. Parity against that pipeline is checked by `tests/whir_commit.rs`.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::Result;
use crate::device::backend;
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
        let mut out = unsafe { self.stream.alloc::<u64>(total * limbs) }?;
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
pub fn commit_codeword(
    evals: &[u64],
    log_blowup: usize,
    log_folding: usize,
) -> Result<(DeviceCodeword, Vec<u8>)> {
    assert!(
        evals.len().is_power_of_two(),
        "evals must be a power of two"
    );
    let log_evals = evals.len().trailing_zeros() as u64;
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
    let stream = be.next_stream();

    // The tail past the coefficients is the zero padding `encode` adds, so the
    // buffer is allocated zeroed and only the coefficient half is written.
    let mut x = stream.alloc_zeros::<u64>(n)?;
    {
        let mut head = x.slice_mut(0..evals.len());
        stream.memcpy_htod(evals, &mut head)?;
    }

    let half = (evals.len() / 2) as u64;
    let half_cfg = LaunchConfig::for_num_elems(half as u32);
    for level in 0..log_evals {
        let stride = 1u64 << level;
        unsafe {
            stream
                .launch_builder(&be.mobius_level)
                .arg(&mut x)
                .arg(&half)
                .arg(&stride)
                .launch(half_cfg)?;
        }
    }

    // Two permutations, not one: the lift reverses the coefficient index over
    // `log_evals` bits and the NTT wants its input reversed over `log_n`.
    let coeffs = evals.len() as u64;
    unsafe {
        stream
            .launch_builder(&be.bit_reverse_permute)
            .arg(&mut x)
            .arg(&coeffs)
            .arg(&log_evals)
            .launch(LaunchConfig::for_num_elems(coeffs as u32))?;
    }
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.bit_reverse_permute)
            .arg(&mut x)
            .arg(&n_u64)
            .arg(&log_n)
            .launch(LaunchConfig::for_num_elems(n as u32))?;
    }
    let twiddles = be.fwd_twiddles_for(log_n)?;
    crate::ntt::run_ntt_body(stream.as_ref(), &mut x, twiddles.as_ref(), n_u64, log_n)?;

    // The leaf hashes are written straight into the node buffer's leaf half,
    // so the tree never needs a buffer of its own.
    let total_nodes = 2 * num_leaves - 1;
    // SAFETY: every byte is written before it is read — the leaves by the
    // kernel below, the inner nodes by the level loop after it.
    let mut nodes = unsafe { stream.alloc::<u8>(total_nodes * 32) }?;
    {
        let leaves_offset = (num_leaves - 1) * 32;
        let mut leaves = nodes.slice_mut(leaves_offset..leaves_offset + num_leaves * 32);
        let num_leaves_u64 = num_leaves as u64;
        let block = 1u64 << log_folding;
        unsafe {
            stream
                .launch_builder(&be.keccak256_leaves_base_coset)
                .arg(&x)
                .arg(&num_leaves_u64)
                .arg(&block)
                .arg(&mut leaves)
                .launch(keccak_launch_cfg(num_leaves_u64))?;
        }
    }
    build_inner_tree_levels(stream.as_ref(), be, &mut nodes, num_leaves)?;

    let nodes = stream.clone_dtoh(&nodes)?;
    stream.synchronize()?;
    Ok((
        DeviceCodeword {
            buffer: Arc::new(x),
            stream,
            elements: n,
            base: true,
        },
        nodes,
    ))
}

/// The same, with the codeword brought back — what a caller that folds on the
/// host needs.
pub fn commit_codeword_to_host(
    evals: &[u64],
    log_blowup: usize,
    log_folding: usize,
) -> Result<(Vec<u64>, Vec<u8>)> {
    let (codeword, nodes) = commit_codeword(evals, log_blowup, log_folding)?;
    let values = codeword.stream.clone_dtoh(codeword.buffer.as_ref())?;
    codeword.stream.synchronize()?;
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
    let mut current = unsafe { stream.alloc::<u64>(half * 3) }?;
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
        let mut next = unsafe { stream.alloc::<u64>(half * 3) }?;
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
    let mut current = unsafe { stream.alloc::<u64>(half * 3) }?;
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
        let mut next = unsafe { stream.alloc::<u64>(half * 3) }?;
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
    })
}

/// Merkle-commits a resident ext3 codeword's fold blocks, returning the tree
/// in the host node layout.
pub fn commit_resident_ext3(codeword: &DeviceCodeword, log_folding: usize) -> Result<Vec<u8>> {
    assert!(!codeword.base, "a folded codeword is in the extension");
    let num_leaves = codeword.elements >> log_folding;
    assert!(num_leaves >= 2, "tree needs at least two leaves");

    let be = backend()?;
    let stream = &codeword.stream;
    let total_nodes = 2 * num_leaves - 1;
    // SAFETY: every byte is written before it is read — the leaves by the
    // kernel below, the inner nodes by the level loop after it.
    let mut nodes = unsafe { stream.alloc::<u8>(total_nodes * 32) }?;
    {
        let leaves_offset = (num_leaves - 1) * 32;
        let mut leaves = nodes.slice_mut(leaves_offset..leaves_offset + num_leaves * 32);
        let num_leaves_u64 = num_leaves as u64;
        let block = 1u64 << log_folding;
        unsafe {
            stream
                .launch_builder(&be.keccak256_leaves_ext3_coset)
                .arg(codeword.buffer.as_ref())
                .arg(&num_leaves_u64)
                .arg(&block)
                .arg(&mut leaves)
                .launch(keccak_launch_cfg(num_leaves_u64))?;
        }
    }
    build_inner_tree_levels(stream.as_ref(), be, &mut nodes, num_leaves)?;

    let out = stream.clone_dtoh(&nodes)?;
    stream.synchronize()?;
    Ok(out)
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
        let mut next = unsafe { stream.alloc::<u64>(half * 3) }?;
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
pub fn commit_codeword_ext3(codeword: &[u64], log_folding: usize) -> Result<Vec<u8>> {
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
    let mut nodes = unsafe { stream.alloc::<u8>(total_nodes * 32) }?;
    {
        let leaves_offset = (num_leaves - 1) * 32;
        let mut leaves = nodes.slice_mut(leaves_offset..leaves_offset + num_leaves * 32);
        let num_leaves_u64 = num_leaves as u64;
        let block = 1u64 << log_folding;
        unsafe {
            stream
                .launch_builder(&be.keccak256_leaves_ext3_coset)
                .arg(&values)
                .arg(&num_leaves_u64)
                .arg(&block)
                .arg(&mut leaves)
                .launch(keccak_launch_cfg(num_leaves_u64))?;
        }
    }
    build_inner_tree_levels(stream.as_ref(), be, &mut nodes, num_leaves)?;

    let out = stream.clone_dtoh(&nodes)?;
    stream.synchronize()?;
    Ok(out)
}
