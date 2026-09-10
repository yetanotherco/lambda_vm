//! Device dispatch for the multilinear path.
//!
//! Every entry point here returns `None` when the device declines — a field
//! the kernels do not cover, a size below the launch threshold, a kill switch,
//! or any CUDA error — and the caller runs the host path. A dispatch that
//! succeeded is counted, so a bench can tell a GPU number from a CPU one
//! wearing its label.

/// Successful device commits of a stacked polynomial.
static COMMIT_CALLS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

pub fn commit_calls() -> u64 {
    COMMIT_CALLS.load(core::sync::atomic::Ordering::Relaxed)
}

pub fn reset_call_counters() {
    COMMIT_CALLS.store(0, core::sync::atomic::Ordering::Relaxed);
}

/// A committed codeword and the Merkle tree over its fold blocks.
type CommittedCodeword<F> = (Vec<math::field::element::FieldElement<F>>, Vec<[u8; 32]>);

/// Codeword size below which the host wins: the kernels are a dozen launches
/// and a round trip, and a small NTT finishes in host cache before that.
#[cfg(feature = "cuda")]
const COMMIT_THRESHOLD: usize = 1 << 16;

/// Commits one stacked polynomial on device, returning its codeword in domain
/// order and its Merkle tree in the host node layout.
#[cfg(feature = "cuda")]
pub(crate) fn commit_codeword<F>(
    evals: &[math::field::element::FieldElement<F>],
    log_blowup: usize,
    log_folding: usize,
) -> Option<CommittedCodeword<F>>
where
    F: math::field::traits::IsField + 'static,
{
    use math::field::element::FieldElement;
    use math::field::goldilocks::GoldilocksField;

    if std::any::TypeId::of::<F>() != std::any::TypeId::of::<GoldilocksField>() {
        return None;
    }
    if !evals.len().is_power_of_two() || evals.len() < 2 {
        return None;
    }
    if evals.len() << log_blowup < COMMIT_THRESHOLD {
        return None;
    }
    // Presence-based kill switch, matching `LAMBDA_VM_NO_GPU_GRIND`: a
    // production escape hatch, and what makes the host path stay covered.
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_WHIR_COMMIT").is_some()) {
        return None;
    }

    // SAFETY: `F == GoldilocksField` is established above, and
    // `FieldElement<GoldilocksField>` is a transparent wrapper over its `u64`
    // representation — the same one the kernels read and write.
    let raw = unsafe { core::slice::from_raw_parts(evals.as_ptr() as *const u64, evals.len()) };
    let (codeword, nodes) = math_cuda::whir::commit_codeword(raw, log_blowup, log_folding).ok()?;
    if nodes.len() % 32 != 0 {
        return None;
    }

    // SAFETY: as above, plus `FieldElement` has no drop glue over a `u64`, so
    // the allocation changes type in place. Relabelling a gigabyte codeword
    // element by element would cost more than the kernels it came from.
    let mut codeword = core::mem::ManuallyDrop::new(codeword);
    let codeword = unsafe {
        Vec::from_raw_parts(
            codeword.as_mut_ptr() as *mut FieldElement<F>,
            codeword.len(),
            codeword.capacity(),
        )
    };
    let nodes = nodes
        .chunks_exact(32)
        .map(|node| {
            let mut out = [0u8; 32];
            out.copy_from_slice(node);
            out
        })
        .collect();
    COMMIT_CALLS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    Some((codeword, nodes))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn commit_codeword<F>(
    _evals: &[math::field::element::FieldElement<F>],
    _log_blowup: usize,
    _log_folding: usize,
) -> Option<CommittedCodeword<F>>
where
    F: math::field::traits::IsField + 'static,
{
    None
}
