//! Device dispatch for the multilinear path.
//!
//! Every entry point here returns `None` when the device declines — a field
//! the kernels do not cover, a size below the launch threshold, a kill switch,
//! or any CUDA error — and the caller runs the host path. A dispatch that
//! succeeded is counted, so a bench can tell a GPU number from a CPU one
//! wearing its label.

use core::sync::atomic::{AtomicU64, Ordering};

/// Successful device commits of a stacked polynomial.
static COMMIT_CALLS: AtomicU64 = AtomicU64::new(0);
/// Sumchecks whose rounds ran on device.
static SUMCHECK_CALLS: AtomicU64 = AtomicU64::new(0);
/// Rounds within them, so a declined tail shows up.
static SUMCHECK_ROUNDS: AtomicU64 = AtomicU64::new(0);

pub fn commit_calls() -> u64 {
    COMMIT_CALLS.load(Ordering::Relaxed)
}

pub fn sumcheck_calls() -> u64 {
    SUMCHECK_CALLS.load(Ordering::Relaxed)
}

pub fn sumcheck_rounds() -> u64 {
    SUMCHECK_ROUNDS.load(Ordering::Relaxed)
}

pub fn reset_call_counters() {
    COMMIT_CALLS.store(0, Ordering::Relaxed);
    SUMCHECK_CALLS.store(0, Ordering::Relaxed);
    SUMCHECK_ROUNDS.store(0, Ordering::Relaxed);
}

/// A sumcheck's round proofs, the challenges they drew, and the factors the
/// rounds left bound.
type SumcheckRounds<E> = (
    Vec<crate::sumcheck::RoundProof<E>>,
    Vec<math::field::element::FieldElement<E>>,
    Vec<crate::mle::Mle<E>>,
);

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
    COMMIT_CALLS.fetch_add(1, Ordering::Relaxed);
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

/// Op tags the sumcheck kernel reads. MUST stay in sync with
/// `crypto/math-cuda/kernels/sumcheck.cu`.
pub mod op {
    pub const FIXED: u32 = 0;
    pub const VAR: u32 = 1;
    pub const ADD: u32 = 2;
    pub const SUB: u32 = 3;
    pub const MUL: u32 = 4;
    pub const NEG: u32 = 5;
}

/// A program lowered for the device: the nodes (two u64 each, `op | a << 32`
/// then `b | res << 32`), the ext3 constants they read (three u64 each), the
/// slot file's width and the slot the root lands in.
#[derive(Clone, Debug)]
pub struct Lowered {
    pub nodes: Vec<u64>,
    pub consts: Vec<u64>,
    pub num_slots: usize,
    pub root_slot: u32,
}

/// Slots the round kernel will hold per thread before the dispatch declines.
/// The slot file is `slots * 24 * threads` bytes, so a program wider than this
/// buys too few threads to be worth the launch.
pub const MAX_SLOTS: usize = 512;

/// Cube size below which the host wins: the rounds are a launch and a round
/// trip each, and a small cube fits in cache.
#[cfg(feature = "cuda")]
const SUMCHECK_THRESHOLD: usize = 1 << 12;

/// Assigns every step a slot, reusing the slot of a value whose last read has
/// passed.
///
/// This is what makes the kernel possible at all: the precompile tables compile
/// to tens of thousands of steps, and a slot per step would be a megabyte of
/// scratch per thread.
pub fn lower<E>(program: &crate::program::Program<E>) -> Option<Lowered>
where
    E: math::field::traits::IsField + 'static,
{
    use crate::program::Op;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

    // The blob is ext3: the kernel reads three limbs per value, whether or not
    // this particular program happens to hold a constant that would say so.
    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<Ext3>() {
        return None;
    }
    let steps = program.steps();
    // Last step that reads each value; the root is read by the caller, so it is
    // never freed.
    let mut last_use: Vec<usize> = vec![0; steps.len()];
    for (i, step) in steps.iter().enumerate() {
        let mut mark = |operand: u32| last_use[operand as usize] = i;
        match *step {
            Op::Fixed(_) | Op::Var(_) => {}
            Op::Neg(a) => mark(a),
            Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => {
                mark(a);
                mark(b);
            }
        }
    }
    last_use[program.root() as usize] = usize::MAX;

    let mut slot_of: Vec<u32> = vec![u32::MAX; steps.len()];
    let mut free: Vec<u32> = Vec::new();
    let mut num_slots = 0usize;
    let mut nodes: Vec<u64> = Vec::with_capacity(steps.len() * 2);
    let mut consts: Vec<u64> = Vec::new();

    for (i, step) in steps.iter().enumerate() {
        let (op, a, b, operands) = match *step {
            Op::Fixed(ref value) => {
                let at = consts.len() / 3;
                consts.extend_from_slice(&ext3_raw(value)?);
                (op::FIXED, at as u32, 0, [None, None])
            }
            Op::Var(slot) => (op::VAR, slot, 0, [None, None]),
            Op::Neg(a) => (op::NEG, slot_of[a as usize], 0, [Some(a), None]),
            Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => {
                let tag = match *step {
                    Op::Add(..) => op::ADD,
                    Op::Sub(..) => op::SUB,
                    _ => op::MUL,
                };
                (
                    tag,
                    slot_of[a as usize],
                    slot_of[b as usize],
                    [Some(a), Some(b)],
                )
            }
        };
        // Freed before the result is allocated: the kernel loads both operands
        // before it stores, so the result may take a slot this step frees. A
        // value read twice — `x·x`, which is what a squaring compiles to —
        // frees its slot once, or two later values would be handed the same
        // one.
        if let Some(x) = operands[0].filter(|x| last_use[*x as usize] == i) {
            free.push(slot_of[x as usize]);
        }
        if let Some(y) =
            operands[1].filter(|y| last_use[*y as usize] == i && operands[0] != Some(*y))
        {
            free.push(slot_of[y as usize]);
        }
        let res = free.pop().unwrap_or_else(|| {
            let slot = num_slots as u32;
            num_slots += 1;
            slot
        });
        if num_slots > MAX_SLOTS {
            return None;
        }
        slot_of[i] = res;
        nodes.push(u64::from(op) | (u64::from(a) << 32));
        nodes.push(u64::from(b) | (u64::from(res) << 32));
    }

    Some(Lowered {
        nodes,
        consts,
        num_slots,
        root_slot: slot_of[program.root() as usize],
    })
}

/// An ext3 element's three limbs, or `None` when `E` is not that field.
pub fn ext3_raw<E>(value: &math::field::element::FieldElement<E>) -> Option<[u64; 3]>
where
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<Ext3>() {
        return None;
    }
    // SAFETY: `E == Ext3`, whose `FieldElement` is a transparent wrapper over
    // three Goldilocks limbs, each transparent over its `u64`.
    let limbs = unsafe { *(value as *const _ as *const [u64; 3]) };
    Some(limbs)
}

/// Rebuilds an ext3 element from its three limbs.
pub fn ext3_from_raw<E>(limbs: &[u64]) -> math::field::element::FieldElement<E>
where
    E: math::field::traits::IsField + 'static,
{
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;

    let value = FieldElement::<Ext3>::new([
        FieldElement::<Gl>::from_raw(limbs[0]),
        FieldElement::<Gl>::from_raw(limbs[1]),
        FieldElement::<Gl>::from_raw(limbs[2]),
    ]);
    // SAFETY: only called under a TypeId check that `E == Ext3`.
    unsafe { core::mem::transmute_copy::<FieldElement<Ext3>, FieldElement<E>>(&value) }
}

/// Runs a batched sumcheck's rounds on device.
///
/// `challenge` absorbs a round's evaluations and returns the challenge drawn
/// from them, which is the whole of the host's part: the transcript is
/// sequential by definition and stays where it is.
///
/// The factors are consumed — they are left folded on device and dropped — so
/// this is for a caller that wants the rounds and the point, not the tables.
/// The prover's `g(0) + g(1)` self-check does not run on this path: it costs
/// the extra interpolation node the protocol exists to skip.
///
/// `None` means the device declined **before the transcript moved**, and the
/// host path runs. `Some(Err)` means it failed after: the challenges are drawn,
/// the transcript cannot be rewound, and the proof fails rather than being
/// finished from a state the verifier will not reproduce.
#[cfg(feature = "cuda")]
pub(crate) fn prove_sumcheck<E>(
    polys: &[crate::mle::Mle<E>],
    program: &crate::program::Program<E>,
    degree: usize,
    mut challenge: impl FnMut(
        &[math::field::element::FieldElement<E>],
    ) -> math::field::element::FieldElement<E>,
) -> Option<Result<SumcheckRounds<E>, crate::Error>>
where
    E: math::field::traits::IsField + 'static,
{
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<Ext3>() {
        return None;
    }
    let first = polys.first()?;
    let num_vars = first.num_vars();
    if first.len() < SUMCHECK_THRESHOLD || num_vars == 0 {
        return None;
    }
    if degree == 0 || degree > math_cuda::sumcheck::MAX_NODES {
        return None;
    }
    if polys.iter().any(|p| p.len() != first.len()) {
        return None;
    }
    // A slot past the factor list would be an out-of-bounds device read, which
    // no kernel can check for itself.
    if program
        .max_slot()
        .is_some_and(|slot| slot as usize >= polys.len())
    {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_SUMCHECK").is_some()) {
        return None;
    }
    let lowered = lower(program)?;

    // SAFETY: `E == Ext3` is established above, and its `FieldElement` is a
    // transparent wrapper over three `u64` limbs — the layout the kernel reads.
    let raw: Vec<&[u64]> = polys
        .iter()
        .map(|p| unsafe {
            core::slice::from_raw_parts(p.evals().as_ptr() as *const u64, p.len() * 3)
        })
        .collect();

    let mut session = math_cuda::sumcheck::SumcheckSession::new(
        &raw,
        &lowered.nodes,
        &lowered.consts,
        lowered.num_slots,
        lowered.root_slot,
    )
    .ok()?;

    // The interpolation nodes are `1..=degree`: `g(0)` is not sent, the claim
    // carried into the round fixes it.
    let mut t = Vec::with_capacity(degree * 3);
    for node in 1..=degree {
        t.extend_from_slice(&ext3_raw(&FieldElement::<E>::from(node as u64))?);
    }

    // Diagnostic hook: recompute each round on the host from the factors the
    // device holds and panic on the first disagreement, naming the round. A
    // device round that differs otherwise surfaces as a proof that does not
    // verify, minutes and 55 tables later.
    static XCHECK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let xcheck = *XCHECK.get_or_init(|| std::env::var_os("LAMBDA_VM_GPU_XCHECK").is_some());

    // Past this point the transcript moves, so a failure is an error and not a
    // decline.
    let failed = |stage| Some(Err(crate::Error::DeviceFailed { stage }));
    let mut rounds = Vec::with_capacity(num_vars);
    let mut challenges = Vec::with_capacity(num_vars);
    for round in 0..num_vars {
        let expected = if xcheck {
            let Ok(tables) = session.download() else {
                return failed("cross-check download");
            };
            let factors: Vec<crate::mle::Mle<E>> = tables
                .iter()
                .map(|table| {
                    crate::mle::Mle::new(table.chunks_exact(3).map(ext3_from_raw::<E>).collect())
                })
                .collect::<Result<_, _>>()
                .expect("the device holds power-of-two tables");
            Some(
                crate::sumcheck::round_evaluations_for_program(&factors, program, degree)
                    .expect("the host round"),
            )
        } else {
            None
        };
        let Ok(sums) = session.round(&t) else {
            return failed("round");
        };
        let evaluations: Vec<FieldElement<E>> =
            sums.chunks_exact(3).map(ext3_from_raw::<E>).collect();
        if let Some(expected) = expected {
            assert_eq!(
                evaluations, expected,
                "the device round {round} differs from the host"
            );
        }
        let r = challenge(&evaluations);
        let Some(raw) = ext3_raw(&r) else {
            return failed("challenge");
        };
        if session.fold(&raw).is_err() {
            return failed("fold");
        }
        rounds.push(crate::sumcheck::RoundProof { evaluations });
        challenges.push(r);
        SUMCHECK_ROUNDS.fetch_add(1, Ordering::Relaxed);
    }
    let Ok(tables) = session.download() else {
        return failed("download");
    };
    let folded: Result<Vec<crate::mle::Mle<E>>, crate::Error> = tables
        .iter()
        .map(|table| crate::mle::Mle::new(table.chunks_exact(3).map(ext3_from_raw::<E>).collect()))
        .collect();
    let Ok(folded) = folded else {
        return failed("folded tables");
    };
    SUMCHECK_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(Ok((rounds, challenges, folded)))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn prove_sumcheck<E>(
    _polys: &[crate::mle::Mle<E>],
    _program: &crate::program::Program<E>,
    _degree: usize,
    _challenge: impl FnMut(
        &[math::field::element::FieldElement<E>],
    ) -> math::field::element::FieldElement<E>,
) -> Option<Result<SumcheckRounds<E>, crate::Error>>
where
    E: math::field::traits::IsField + 'static,
{
    None
}

/// Codeword size below which the host fold wins: one launch per level plus the
/// round trip, against a pass a few cores finish in microseconds.
#[cfg(feature = "cuda")]
const FOLD_THRESHOLD: usize = 1 << 16;

/// Folds a codeword `alphas.len()` times on device.
///
/// `generator` is the fold domain's, and the domain squares with every level —
/// the kernel takes each level's inverse generator, which is this one's inverse
/// squared level by level.
#[cfg(feature = "cuda")]
pub(crate) fn fold_codeword_k<F, C, N>(
    codeword: &[math::field::element::FieldElement<C>],
    generator: &math::field::element::FieldElement<F>,
    alphas: &[math::field::element::FieldElement<N>],
) -> Option<Vec<math::field::element::FieldElement<N>>>
where
    F: math::field::traits::IsField + 'static,
    C: math::field::traits::IsField + 'static,
    N: math::field::traits::IsField + 'static,
{
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;
    use std::any::TypeId;

    if TypeId::of::<F>() != TypeId::of::<Gl>() || TypeId::of::<N>() != TypeId::of::<Ext3>() {
        return None;
    }
    let base = TypeId::of::<C>() == TypeId::of::<Gl>();
    if !base && TypeId::of::<C>() != TypeId::of::<Ext3>() {
        return None;
    }
    if alphas.is_empty() || codeword.len() < FOLD_THRESHOLD {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_WHIR_FOLD").is_some()) {
        return None;
    }

    // SAFETY: `F == GoldilocksField`, a transparent wrapper over `u64`.
    let generator = unsafe { *(generator as *const _ as *const u64) };
    let generator = FieldElement::<Gl>::from_raw(generator);
    let two = FieldElement::<Gl>::from(2u64);
    let two_inv = *two.inv().ok()?.value();
    let mut g_inv = generator.inv().ok()?;
    let mut g_invs = Vec::with_capacity(alphas.len());
    for _ in 0..alphas.len() {
        g_invs.push(*g_inv.value());
        g_inv = g_inv.square();
    }

    let mut raw_alphas = Vec::with_capacity(alphas.len() * 3);
    for alpha in alphas {
        raw_alphas.extend_from_slice(&ext3_raw(alpha)?);
    }

    // SAFETY: `C` is one of the two fields checked above, and both wrap their
    // limbs transparently — one `u64` per base element, three per ext3.
    let limbs = if base { 1 } else { 3 };
    let raw = unsafe {
        core::slice::from_raw_parts(codeword.as_ptr() as *const u64, codeword.len() * limbs)
    };
    let folded = if base {
        math_cuda::whir::fold_codeword_base(raw, two_inv, &g_invs, &raw_alphas).ok()?
    } else {
        math_cuda::whir::fold_codeword_ext3(raw, two_inv, &g_invs, &raw_alphas).ok()?
    };

    // SAFETY: `N == Ext3`, three limbs per element and no drop glue, so the
    // allocation changes type in place instead of being copied.
    let mut folded = core::mem::ManuallyDrop::new(folded);
    Some(unsafe {
        Vec::from_raw_parts(
            folded.as_mut_ptr() as *mut FieldElement<N>,
            folded.len() / 3,
            folded.capacity() / 3,
        )
    })
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn fold_codeword_k<F, C, N>(
    _codeword: &[math::field::element::FieldElement<C>],
    _generator: &math::field::element::FieldElement<F>,
    _alphas: &[math::field::element::FieldElement<N>],
) -> Option<Vec<math::field::element::FieldElement<N>>>
where
    F: math::field::traits::IsField + 'static,
    C: math::field::traits::IsField + 'static,
    N: math::field::traits::IsField + 'static,
{
    None
}

/// Merkle-commits an ext3 codeword's fold blocks on device, returning the tree
/// in the host node layout.
#[cfg(feature = "cuda")]
pub(crate) fn commit_tree_ext3<F>(
    codeword: &[math::field::element::FieldElement<F>],
    log_folding: usize,
) -> Option<Vec<[u8; 32]>>
where
    F: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

    if std::any::TypeId::of::<F>() != std::any::TypeId::of::<Ext3>() {
        return None;
    }
    if codeword.len() < COMMIT_THRESHOLD || codeword.len() >> log_folding < 2 {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_WHIR_COMMIT").is_some()) {
        return None;
    }
    // SAFETY: `F == Ext3`, three transparent `u64` limbs per element.
    let raw =
        unsafe { core::slice::from_raw_parts(codeword.as_ptr() as *const u64, codeword.len() * 3) };
    let nodes = math_cuda::whir::commit_codeword_ext3(raw, log_folding).ok()?;
    if !nodes.len().is_multiple_of(32) {
        return None;
    }
    COMMIT_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(
        nodes
            .chunks_exact(32)
            .map(|node| {
                let mut out = [0u8; 32];
                out.copy_from_slice(node);
                out
            })
            .collect(),
    )
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn commit_tree_ext3<F>(
    _codeword: &[math::field::element::FieldElement<F>],
    _log_folding: usize,
) -> Option<Vec<[u8; 32]>>
where
    F: math::field::traits::IsField + 'static,
{
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::Builder;
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;

    type FE = FieldElement<Ext3>;

    /// The kernel's walk, in Rust: the same slot file, the same node encoding.
    ///
    /// This is what pins the lowering without a device — a slot freed too early
    /// or an operand read from the wrong class shows up here as a wrong value,
    /// not as a proof that does not verify an hour later.
    fn run_lowered(lowered: &Lowered, values: &[FE]) -> FE {
        let mut slots = vec![FE::zero(); lowered.num_slots];
        for node in lowered.nodes.chunks_exact(2) {
            let op = (node[0] & 0xFFFF_FFFF) as u32;
            let a = (node[0] >> 32) as u32 as usize;
            let b = (node[1] & 0xFFFF_FFFF) as u32 as usize;
            let res = (node[1] >> 32) as u32 as usize;
            slots[res] = match op {
                op::FIXED => ext3_from_raw::<Ext3>(&lowered.consts[a * 3..a * 3 + 3]),
                op::VAR => values[a],
                op::ADD => slots[a] + slots[b],
                op::SUB => slots[a] - slots[b],
                op::MUL => slots[a] * slots[b],
                op::NEG => -slots[a],
                _ => panic!("unknown op {op}"),
            };
        }
        slots[lowered.root_slot as usize]
    }

    fn values(n: usize) -> Vec<FE> {
        (0..n as u64)
            .map(|i| {
                FE::new([
                    FieldElement::<Gl>::from(i * 31 + 7),
                    FieldElement::<Gl>::from(i * 17 + 2),
                    FieldElement::<Gl>::from(i + 5),
                ])
            })
            .collect()
    }

    /// Every op, a constant, and a chain long enough that slots have to be
    /// recycled.
    fn sample_program() -> crate::program::Program<Ext3> {
        let mut b = Builder::<Ext3>::new();
        let mut acc = b.var(0);
        for slot in 1..6 {
            let v = b.var(slot);
            let doubled = b.add(v, v);
            let scaled = b.mul(doubled, acc);
            let shifted = b.sub(scaled, v);
            acc = b.neg(shifted);
        }
        let seven = b.fixed(FE::from(7u64));
        let root = b.add(acc, seven);
        b.finish(root).unwrap()
    }

    #[test]
    fn the_lowered_program_computes_what_the_program_does() {
        let program = sample_program();
        let lowered = lower(&program).expect("lowers");
        let v = values(6);
        let mut scratch = Vec::new();
        assert_eq!(run_lowered(&lowered, &v), program.eval(&v, &mut scratch));
    }

    /// A value read twice in the step that kills it — `x·x` — must not free its
    /// slot twice, or two later values are handed the same one and the second
    /// clobbers the first. Squarings are everywhere in a real constraint
    /// program, so this is the shape that matters.
    #[test]
    fn a_value_read_twice_frees_its_slot_once() {
        let mut b = Builder::<Ext3>::new();
        let mut acc = b.var(0);
        // Each square kills its operand, and the sums below keep enough values
        // live that a doubly-freed slot gets reused while it is still needed.
        let mut squares = Vec::new();
        for slot in 1..8 {
            let v = b.var(slot);
            let squared = b.mul(v, v);
            let with_acc = b.add(squared, acc);
            squares.push(with_acc);
            acc = b.mul(with_acc, with_acc);
        }
        squares.push(acc);
        let root = b.sum(&squares);
        let program = b.finish(root).unwrap();

        let lowered = lower(&program).expect("lowers");
        let v = values(8);
        let mut scratch = Vec::new();
        assert_eq!(run_lowered(&lowered, &v), program.eval(&v, &mut scratch));
    }

    /// The point of the slot file: a long chain of dead intermediates does not
    /// widen it.
    #[test]
    fn slots_are_reused_once_a_value_is_dead() {
        let lowered = lower(&sample_program()).expect("lowers");
        assert!(
            lowered.num_slots < lowered.nodes.len() / 2,
            "{} slots for {} steps is no reuse at all",
            lowered.num_slots,
            lowered.nodes.len() / 2
        );
    }

    /// A program wider than the slot file declines rather than asking a device
    /// for scratch it cannot have.
    #[test]
    fn a_program_past_the_slot_ceiling_declines() {
        let mut b = Builder::<Ext3>::new();
        // Every value stays live to the end, so the slots cannot be recycled.
        let terms: Vec<u32> = (0..=MAX_SLOTS).map(|slot| b.var(slot)).collect();
        let root = b.sum(&terms);
        let program = b.finish(root).unwrap();
        assert!(lower(&program).is_none());
    }

    #[test]
    fn a_field_the_kernel_does_not_cover_declines() {
        let mut b = Builder::<Gl>::new();
        let root = b.var(0);
        let program = b.finish(root).unwrap();
        assert!(lower(&program).is_none());
    }
}
