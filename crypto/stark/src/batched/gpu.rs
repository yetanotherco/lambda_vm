//! Device leaf hashing for the batched prover's streaming MMCS rounds.
//!
//! The seam: the expensive half of a streaming MMCS round is the LEAF
//! absorption — every committed column of every matrix, canonical big-endian,
//! through the configuration's hash — and that is what runs on device here,
//! via [`math_cuda::mmcs::MmcsGroupHasher`]. The cheap half, the digest climb,
//! stays on host: [`finish`](DeviceStreamingMmcs::finish) hands each height
//! group's finalized leaf digests back and the caller rebuilds the tree with
//! `MixedMmcs::from_group_digests`, so the resulting object — and with it the
//! openings, the root, and everything the transcript sees — is the host code
//! path byte for byte. The device is transcript-invisible by construction.
//!
//! Matrices absorb in INPUT order within their height group, exactly as the
//! host `StreamingMmcsBuilder` does; the caller's absorption order IS the
//! commitment, so these calls must mirror the host call sites one for one.

use std::any::TypeId;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsField;
use math_cuda::mmcs::MmcsGroupHasher;
use math_cuda::{CudaStream, DeviceHash};

use crate::config::{Commitment, CommitmentHash, DeviceTreeBackend, StarkHash};

/// Rounds committed with device leaf hashing, process-wide. Counter for the
/// GPU instrumentation harness; no effect on proving.
static GPU_MMCS_ROUNDS: AtomicU64 = AtomicU64::new(0);

pub fn gpu_mmcs_rounds() -> u64 {
    GPU_MMCS_ROUNDS.load(Ordering::Relaxed)
}

/// The `math_cuda` dispatch key for the configuration's batched backend.
pub(crate) fn device_hash_of<H: StarkHash>() -> DeviceHash {
    match <H::Batched<GoldilocksField> as DeviceTreeBackend>::COMMITMENT_HASH {
        CommitmentHash::Keccak256 => DeviceHash::Keccak256,
        CommitmentHash::Blake3 => DeviceHash::Blake3,
    }
}

/// Components (u64 lanes) per field element of `E`, for the two fields the
/// device kernels serve. `None` = no device support for this field.
pub(crate) fn lanes_per_element<E: 'static>() -> Option<usize> {
    let id = TypeId::of::<E>();
    if id == TypeId::of::<GoldilocksField>() {
        Some(1)
    } else if id == TypeId::of::<Degree3GoldilocksExtensionField>() {
        Some(3)
    } else {
        None
    }
}

/// View a field-element slice as its raw u64 lanes.
///
/// # Safety
/// `lanes` must be [`lanes_per_element::<E>()`] for this `E`: both supported
/// fields are `repr`-equivalent to `[u64; lanes]` per element.
pub(crate) unsafe fn felts_as_lanes<E: IsField>(s: &[FieldElement<E>], lanes: usize) -> &[u64] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u64, s.len() * lanes) }
}

/// One streaming MMCS round's device state: a leaf hasher per height group.
pub(crate) struct DeviceStreamingMmcs {
    stream: Arc<CudaStream>,
    /// Indexed by `log_height`.
    groups: Vec<Option<MmcsGroupHasher>>,
    h_max: usize,
}

impl DeviceStreamingMmcs {
    /// `None` when the device path is unavailable (no backend, or disabled via
    /// `LAMBDA_VM_DISABLE_GPU_MMCS`) — the caller falls back to the host
    /// builder BEFORE anything is absorbed, never mid-round.
    pub(crate) fn try_new(dims: &[(usize, usize)], hash: DeviceHash) -> Option<Self> {
        if std::env::var_os("LAMBDA_VM_DISABLE_GPU_MMCS").is_some() {
            return None;
        }
        let be = math_cuda::device::backend().ok()?;
        let stream = be.next_stream();
        let h_max = dims.iter().map(|&(h, _)| h).max()?;
        let mut groups: Vec<Option<MmcsGroupHasher>> = (0..=h_max).map(|_| None).collect();
        for &(h, _) in dims {
            if groups[h].is_none() {
                groups[h] = Some(MmcsGroupHasher::new(&stream, h as u64, hash).ok()?);
            }
        }
        GPU_MMCS_ROUNDS.fetch_add(1, Ordering::Relaxed);
        Some(Self {
            stream,
            groups,
            h_max,
        })
    }

    /// Absorb one natural-order row-major matrix: lanes `[col_start, col_end)`
    /// of bit-reversed rows `2k`, `2k+1` into leaf `k` of its height group.
    /// All positions in u64 LANES (an ext3 element is three consecutive lanes),
    /// matching the device kernel convention.
    pub(crate) fn absorb_row_major(
        &mut self,
        data: &[u64],
        stride_lanes: usize,
        col_start_lanes: usize,
        col_end_lanes: usize,
        log_height: usize,
    ) -> math_cuda::Result<()> {
        let dev = self.stream.clone_htod(data)?;
        let hasher = self.groups[log_height]
            .as_mut()
            .expect("a matrix's height group exists by construction of the dims");
        hasher.absorb_row_major(
            &self.stream,
            &dev,
            stride_lanes as u64,
            col_start_lanes as u64,
            col_end_lanes as u64,
        )?;
        // The upload dies here — the streaming residency the design exists for.
        drop(dev);
        Ok(())
    }

    /// Finalize every group and return its leaf digests, host-side, indexed by
    /// `log_height` — the exact shape `MixedMmcs::from_group_digests` takes.
    pub(crate) fn finish(self) -> math_cuda::Result<Vec<Option<Vec<Commitment>>>> {
        let mut out: Vec<Option<Vec<Commitment>>> = (0..=self.h_max).map(|_| None).collect();
        for (h, slot) in self.groups.into_iter().enumerate() {
            let Some(hasher) = slot else { continue };
            let digests_dev = hasher.finalize(&self.stream)?;
            let bytes = self.stream.clone_dtoh(&digests_dev)?;
            self.stream.synchronize()?;
            let digests: Vec<Commitment> = bytes
                .chunks_exact(32)
                .map(|c| {
                    let mut d = [0u8; 32];
                    d.copy_from_slice(c);
                    d
                })
                .collect();
            out[h] = Some(digests);
        }
        Ok(out)
    }
}
